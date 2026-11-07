import math
import random
from bisect import bisect_right

from card_utils import Card
from fasteval import ALL_COMBOS, card_int, class_index, draw_bonus, evaluate
from game_state import GameStateView
from pf_table import PF_EQUITY
from player_action import DEALER, PlayerAction

NC = len(ALL_COMBOS)  # 1326 two-card combos
COMBO_INDEX = {c: i for i, c in enumerate(ALL_COMBOS)}
DRAW_KEY = 0x100000 | 6 << 12  # a draw ranks like a pair of eights

# Situations an opponent acts in, and prior frequencies of
# (fold, call or check, small raise, big raise) in each.
SITS = ("pf_open", "pf_vs_raise", "pf_option", "post_open", "post_vs_bet")
PRIORS = {
    "pf_open": (0.15, 0.25, 0.50, 0.10),
    "pf_vs_raise": (0.35, 0.40, 0.15, 0.10),
    "pf_option": (0.0, 0.70, 0.25, 0.05),
    "post_open": (0.0, 0.60, 0.33, 0.07),
    "post_vs_bet": (0.40, 0.42, 0.12, 0.06),
}
PRIOR_N = 8.0
# Prior chance the opponent folds to our bet, by the price they are
# offered (their call / the pot after it): see PRICE_EDGES.
PRICE_EDGES = (0.15, 0.25, 0.33, 0.40)
FOLD_PRIORS = {
    "pre": (0.10, 0.25, 0.40, 0.50, 0.60),
    "post": (0.15, 0.30, 0.42, 0.50, 0.58),
}
NB = len(PRICE_EDGES) + 1
# The opponent answers our bet by folding every hand weaker than a
# threshold T: T is the share of all hands (by strength on this board)
# they continue with. T is learned per size bucket over this grid.
T_GRID = [(k + 1) / 20 for k in range(20)]
T_PRIOR_SD = 0.25


def _t_prior(fold):
    centre = 1.0 - fold
    return [-((t - centre) ** 2) / (2 * T_PRIOR_SD ** 2) for t in T_GRID]


def _fold_chance(mass):
    # A little noise: nobody folds or calls with certainty.
    return 0.03 + 0.94 * mass


def _bucket(price):
    for b, edge in enumerate(PRICE_EDGES):
        if price < edge:
            return b
    return NB - 1


class Model:
    """
    Heads-up no-limit hold'em bot.

    Keeps a weighted range over the opponent's 1326 hole-card combos,
    narrowed by each of their actions according to how often that opponent
    (learned per team) takes it. Each decision deals the range out by Monte
    Carlo and picks the action with the highest expected value, using the
    opponent's learned fold rates by bet size.
    """

    def __init__(self, samples=900, seed=1234):
        self.samples = samples
        # Equity realisation when more betting is to come: strong hands
        # gain, weak ones lose; out of position loses a little more.
        self.real_k = 0.06
        self.oop_pen = 0.025
        self.ip_bonus = 0.01
        self.bet_sizes = [0.4, 0.7, 1.1]  # shares of the pot, plus all-in
        self.raise_sizes = [0.7, 1.1]
        self.rng = random.Random(seed)
        # Per team: {"counts": {sit: [4 floats]}, "folds": {...},
        # "inc": float, "exp": float}
        self.opps = {}
        self.hand = []
        self.player_id = None
        self._new_hand = True
        self.debug = False  # re-raise errors instead of checking/folding

    # ------------------------------------------------------------------
    # Engine hooks

    def seed(self, hand: list[Card], player_id: int):
        self.hand = hand
        self.player_id = player_id
        self._new_hand = True

    def run(self, actions: list[PlayerAction], game_state: GameStateView):
        me = self.player_id
        try:
            if self._new_hand:
                start = [game_state.stacks[p] + game_state.bets[p]
                         for p in range(game_state.num_players)]
                self._start_hand(game_state, start)
            for a in actions:
                self._process(a)
            if len(game_state.players_in_hand) == 2 and self.h_hu:
                return self._decide_hu(game_state)
            return self._decide_multi(game_state)
        except Exception:
            if self.debug:
                raise
            if game_state.call_amount == 0:
                return PlayerAction.call(me)
            return PlayerAction.fold(me)

    def end_game(self, actions: list[PlayerAction], game_state: GameStateView):
        try:
            if self._new_hand:
                # We never acted: rebuild the starting stacks first.
                n = game_state.num_players
                won = [0] * n
                for a in actions:
                    if a.kind == "win":
                        won[a.player_id] += a.amount
                self._start_hand(game_state, [10 ** 9] * n)
                for a in actions:
                    self._process(a, learn=False)
                contributed = [10 ** 9 - s for s in self.t_stacks]
                start = [game_state.stacks[p] - won[p] + contributed[p]
                         for p in range(n)]
                self._start_hand(game_state, start)
            for a in actions:
                self._process(a)
            self._learn_showdown()
        except Exception:
            if self.debug:
                raise
        self._new_hand = True

    # ------------------------------------------------------------------
    # Tracking the hand from the action stream

    def _start_hand(self, gs, start_stacks):
        n = gs.num_players
        me = self.player_id
        self._new_hand = False
        self.h_hole = [card_int(c) for c in self.hand]
        self.h_button = gs.button
        self.h_bb = gs.big_blind
        self.h_hu = n == 2
        self.h_opp = 1 - me if n == 2 else None
        teams = gs.teams
        if self.h_hu:
            name = teams[self.h_opp] if teams else None
            self.h_team = str(name) if name is not None else "opponent"
        else:
            self.h_team = None
        if n == 2:
            sb_p, bb_p = gs.button, 1 - gs.button
        else:
            sb_p, bb_p = (gs.button + 1) % n, (gs.button + 2) % n
        self.t_stacks = list(start_stacks)
        self.t_bets = [0] * n
        for p, blind in ((sb_p, gs.small_blind), (bb_p, gs.big_blind)):
            amt = min(blind, self.t_stacks[p])
            self.t_bets[p] = amt
            self.t_stacks[p] -= amt
        self.t_pot = sum(self.t_bets)
        self.t_board = []
        self.t_raised_pre = False
        self.t_acted_pre = [False] * n
        self.t_shown = []
        # Opponent range: weight per combo, our cards removed.
        w = [1.0] * NC
        for i, (a, b) in enumerate(ALL_COMBOS):
            if a in self.h_hole or b in self.h_hole:
                w[i] = 0.0
        self.h_w = w
        self.h_keys = {}
        self.h_abs = {}
        self.h_log = []  # (lo, hi, pct list) per narrowed opponent action
        self.h_bucket = 1  # size bucket of our last bet

    def _street(self):
        return {0: 0, 3: 1, 4: 2, 5: 3}.get(len(self.t_board), 1)

    def _process(self, a, learn=True):
        p = a.player_id
        if a.kind == "reveal_card":
            if p == DEALER:
                c = card_int(a.card)
                self.t_board.append(c)
                if len(self.t_board) in (3, 4, 5):
                    self.t_bets = [0] * len(self.t_bets)
                for i, (x, y) in enumerate(ALL_COMBOS):
                    if x == c or y == c:
                        self.h_w[i] = 0.0
            elif p == self.h_opp:
                self.t_shown.append(card_int(a.card))
            return
        if a.kind == "win":
            return
        street = self._street()
        to_call = max(self.t_bets) - self.t_bets[p]
        pot_before = self.t_pot
        stack_before = self.t_stacks[p]
        if street == 0:
            if to_call == 0:
                sit = "pf_option"
            elif not self.t_raised_pre and not self.t_acted_pre[p]:
                sit = "pf_open"
            else:
                sit = "pf_vs_raise"
        else:
            sit = "post_vs_bet" if to_call > 0 else "post_open"
        if a.kind == "fold":
            kind = 0
        elif a.kind == "call":
            kind = 1
            amt = min(to_call, self.t_stacks[p])
            self.t_bets[p] += amt
            self.t_stacks[p] -= amt
            self.t_pot += amt
        else:
            amt = a.amount - self.t_bets[p]
            big = amt >= 0.5 * stack_before or amt >= self.t_stacks[p]
            kind = 3 if big else 2
            self.t_bets[p] = a.amount
            self.t_stacks[p] -= amt
            self.t_pot += amt
            if street == 0:
                self.t_raised_pre = True
        if street == 0:
            self.t_acted_pre[p] = True
        if p == self.player_id and a.kind == "raise":
            # Remember the size, to learn how the opponent answers it.
            if self.h_hu:
                o = self.h_opp
                opp_add = min(a.amount - self.t_bets[o], self.t_stacks[o])
                self.h_bucket = _bucket(opp_add / max(1, self.t_pot + opp_add))
        if learn and self.h_hu and p == self.h_opp:
            if kind >= 2:
                cell = self._opp()["rsize"]["pre" if street == 0 else "post"]
                cell[0] += min(3.0, (amt - to_call) / max(1, pot_before
                                                          + to_call))
                cell[1] += 1.0
            self._learn_action(sit, kind)
            if kind != 0:
                self._narrow(sit, kind)

    # ------------------------------------------------------------------
    # Opponent model

    def _opp(self):
        o = self.opps.get(self.h_team)
        if o is None:
            o = {
                "counts": {s: [0.0, 0.0, 0.0, 0.0] for s in SITS},
                "tl": {ph: [_t_prior(f) for f in FOLD_PRIORS[ph]]
                       for ph in ("pre", "post")},
                "rsize": {"pre": [3.0, 3.0], "post": [2.4, 3.0]},
                "inc": 0.6,
                "exp": 4.0,
            }
            self.opps[self.h_team] = o
        return o

    def _freqs(self, sit):
        c = self._opp()["counts"][sit]
        n = c[0] + c[1] + c[2] + c[3]
        pr = PRIORS[sit]
        return [(c[i] + PRIOR_N * pr[i]) / (n + PRIOR_N) for i in range(4)]

    def _raise_size(self, phase):
        """The opponent's usual raise beyond a call, as a share of the pot."""
        total, count = self._opp()["rsize"][phase]
        return total / count

    def _t_posterior(self, street, bucket):
        tl = self._opp()["tl"]["pre" if street == 0 else "post"][bucket]
        top = max(tl)
        p = [math.exp(x - top) for x in tl]
        total = sum(p)
        return [x / total for x in p]

    def _learn_action(self, sit, kind):
        o = self._opp()
        o["counts"][sit][kind] += 1.0
        if sit not in ("pf_vs_raise", "post_vs_bet"):
            return
        # How much of their range each threshold would have folded.
        absp = self._abs_pct()
        w = self.h_w
        hist = [0.0] * 20
        for i in range(NC):
            if w[i] > 0:
                hist[min(19, int(absp[i] * 20))] += w[i]
        total = sum(hist) or 1.0
        masses = []
        above = 0.0
        for k in range(19, -1, -1):
            masses.append(above / total)  # weight in bins above T_GRID[k]
            above += hist[k]
        masses.reverse()
        tables = o["tl"]["pre" if sit == "pf_vs_raise" else "post"]
        for b in range(NB):
            share = 1.0 if b == self.h_bucket else 0.35
            row = tables[b]
            for k in range(20):
                pf = _fold_chance(masses[k])
                row[k] += share * math.log(pf if kind == 0 else 1.0 - pf)

    def _abs_pct(self):
        """Each live combo's place among all hands on this board."""
        n = len(self.t_board)
        cached = self.h_abs.get(n)
        if cached is None:
            dead = set(self.h_hole + self.t_board)
            uniform = [0.0 if a in dead or b in dead else 1.0
                       for a, b in ALL_COMBOS]
            cached = self._pct(self._keys(), uniform)
            self.h_abs[n] = cached
        return cached

    def _eps(self):
        o = self._opp()
        return min(1.0, max(0.03, o["inc"] / o["exp"]))

    def _keys(self):
        n = len(self.t_board)
        keys = self.h_keys.get(n)
        if keys is not None:
            return keys
        board = self.t_board
        w = self.h_w
        keys = [-1] * NC
        if n == 0:
            for i, (a, b) in enumerate(ALL_COMBOS):
                if w[i] > 0:
                    keys[i] = PF_EQUITY[class_index(a, b)]
        else:
            draws = n < 5
            for i, (a, b) in enumerate(ALL_COMBOS):
                if w[i] > 0:
                    k = evaluate([a, b] + board)
                    if draws and k < DRAW_KEY and draw_bonus((a, b), board):
                        k = DRAW_KEY
                    keys[i] = k
        self.h_keys[n] = keys
        return keys

    def _pct(self, keys, w=None):
        """Each combo's place in the range w (default: the current one),
        0 = strongest."""
        if w is None:
            w = self.h_w
        order = sorted((i for i in range(NC) if w[i] > 0),
                       key=keys.__getitem__, reverse=True)
        total = sum(w[i] for i in order)
        pct = [1.0] * NC
        if total <= 0:
            return pct
        cum = 0.0
        i = 0
        m = len(order)
        while i < m:
            k = keys[order[i]]
            j = i
            gw = 0.0
            while j < m and keys[order[j]] == k:
                gw += w[order[j]]
                j += 1
            mid = (cum + gw / 2) / total
            for idx in order[i:j]:
                pct[idx] = mid
            cum += gw
            i = j
        return pct

    def _narrow(self, sit, kind):
        f, c, rs, rb = self._freqs(sit)
        if kind == 3:
            lo, hi = 0.0, rb
        elif kind == 2:
            lo, hi = 0.0, rb + rs
        else:
            lo, hi = rb + rs, rb + rs + c
        hi = min(1.0, hi)
        if sit in ("pf_option", "post_open") and kind == 1:
            hi = 1.0  # a check: everything that did not bet
        keys = self._keys()
        pct = self._pct(keys)
        self.h_log.append((lo, hi, pct))
        # With chance eps the action says nothing (noise); otherwise the
        # hand is in the band.
        eps = self._eps()
        w = self.h_w
        margin = 0.03
        inside = [False] * NC
        total = band = 0.0
        for i in range(NC):
            if w[i] > 0:
                total += w[i]
                x = pct[i]
                if lo - margin <= x <= hi + margin:
                    inside[i] = True
                    band += w[i]
        if total <= 0:
            self._reset_range()
            return
        f_in = eps + (1.0 - eps) / max(band / total, 0.02)
        top = 0.0
        for i in range(NC):
            if w[i] > 0:
                w[i] *= f_in if inside[i] else eps
                if w[i] > top:
                    top = w[i]
        if top > 0:
            for i in range(NC):
                w[i] /= top
        else:
            self._reset_range()

    def _reset_range(self):
        dead = set(self.h_hole + self.t_board)
        self.h_w = [0.0 if a in dead or b in dead else 1.0
                    for a, b in ALL_COMBOS]

    def _learn_showdown(self):
        if not self.h_hu or len(self.t_shown) != 2 or not self.h_log:
            return
        a, b = sorted(self.t_shown)
        idx = COMBO_INDEX.get((a, b))
        if idx is None:
            return
        o = self._opp()
        for lo, hi, pct in self.h_log:
            x = pct[idx]
            width = min(1.0, hi - lo + 0.1)
            if x < lo - 0.05 or x > hi + 0.05:
                o["inc"] += 1.0
            o["exp"] += max(0.05, 1.0 - width)

    # ------------------------------------------------------------------
    # Decisions

    def _simulate(self):
        """Monte Carlo against the opponent's range: (pct, abs pct, result)
        sorted by the opponent hand's strength."""
        board = self.t_board
        w = self.h_w
        live = [i for i in range(NC) if w[i] > 0]
        if not live:
            self._reset_range()
            w = self.h_w
            live = [i for i in range(NC) if w[i] > 0]
        pct = self._pct(self._keys())
        absp = self._abs_pct()
        chosen = self.rng.choices(live, weights=[w[i] for i in live],
                                  k=self.samples)
        dead = set(self.h_hole + board)
        deck = [c for c in range(52) if c not in dead]
        need = 5 - len(board)
        mine_base = self.h_hole + board
        out = []
        sample = self.rng.sample
        for ci in chosen:
            a, b = ALL_COMBOS[ci]
            if need:
                run = [c for c in sample(deck, need + 2) if c != a and c != b]
                run = run[:need]
                full = board + run
                mine = evaluate(mine_base + run)
            else:
                full = board
                mine = evaluate(mine_base)
            theirs = evaluate([a, b] + full)
            res = 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
            out.append((pct[ci], absp[ci], res))
        out.sort()
        return out

    def _decide_hu(self, gs):
        me = self.player_id
        opp = self.h_opp
        sims = self._simulate()
        n = len(sims)
        pctl = [s[0] for s in sims]
        absl = [s[1] for s in sims]
        prefix = [0.0]
        for s in sims:
            prefix.append(prefix[-1] + s[2])
        e_all = prefix[n] / n

        def mean(i, j, least=25):
            """Our equity against the opponent's samples i..j (strongest
            first), widened to at least `least` samples."""
            if j - i < least:
                mid = (i + j) // 2
                i = max(0, min(n - least, mid - least // 2))
                j = min(n, i + least)
            return (prefix[j] - prefix[i]) / (j - i)

        street = self._street()
        oop = me != self.h_button
        pot = gs.pot
        to_call = gs.call_amount
        my_bet = gs.bets[me]
        opp_bet = gs.bets[opp]
        my_stack = gs.stacks[me]
        opp_stack = gs.stacks[opp]
        phase = "pre" if street == 0 else "post"
        # Both players' street total once all chips are in.
        cap = min(my_bet + my_stack, opp_bet + opp_stack)
        allin_pot = pot - my_bet - opp_bet + 2 * cap

        def eff(e, allin):
            if allin or street == 3:
                return e
            return (e + self.real_k * (e - 0.5)
                    + (-self.oop_pen if oop else self.ip_bonus))

        def facing_bet(base_pot, sit, added):
            """EV once we have added `added` chips and the opponent, still
            to act in `sit`, may bet or check/call behind us."""
            f, c, rs, rb = self._freqs(sit)
            b = rs + rb
            nb = bisect_right(pctl, b)
            e_b = mean(0, nb)
            e_nb = mean(nb, n)
            size = min(self._raise_size(phase) * base_pot, my_stack - added,
                       opp_stack)
            size = max(size, 1)
            allin = size >= my_stack - added
            go_on = eff(e_b, allin) * (base_pot + 2 * size) - size
            return (-added + (1 - b) * eff(e_nb, False) * base_pot
                     + b * max(0.0, go_on))

        options = []  # (ev, action)
        limp = (street == 0 and not self.t_raised_pre and to_call > 0
                and not oop)
        if to_call > 0:
            options.append((0.0, PlayerAction.fold(me)))
            call_amt = min(to_call, my_stack)
            refund = to_call - call_amt
            allin = call_amt == my_stack or opp_stack == 0
            if limp and not allin:
                ev = facing_bet(pot + call_amt, "pf_option", call_amt)
            else:
                ev = eff(e_all, allin) * (pot - refund + call_amt) - call_amt
            options.append((ev, PlayerAction.call(me)))
        elif street > 0 and oop:
            options.append((facing_bet(pot, "post_open", 0),
                            PlayerAction.call(me)))
        else:
            options.append((eff(e_all, False) * pot, PlayerAction.call(me)))

        if gs.can_raise and opp_stack > 0:
            lo, hi = gs.raise_range
            base = pot + to_call
            if street == 0:
                if not self.t_raised_pre and to_call > 0:
                    targets = [2.5 * self.h_bb, 3 * self.h_bb]
                else:
                    targets = [opp_bet + k * base for k in (0.7, 1.0)]
            elif to_call == 0:
                targets = [k * pot for k in self.bet_sizes]
            else:
                targets = [opp_bet + k * base for k in self.raise_sizes]
            targets.append(hi)
            sit = "pf_vs_raise" if street == 0 else "post_vs_bet"
            f0, c0, rs0, rb0 = self._freqs(sit)
            reraise = (rs0 + rb0) / max(1e-9, c0 + rs0 + rb0)
            seen = set()
            for t in targets:
                r = int(min(hi, max(lo, round(t))))
                if r >= 0.7 * hi:
                    r = hi
                if r in seen:
                    continue
                seen.add(r)
                opp_max = opp_bet + opp_stack
                eff_to = min(r, opp_max)
                opp_add = eff_to - opp_bet
                my_add = eff_to - my_bet
                price = opp_add / max(1, pot + my_add + opp_add)
                post = self._t_posterior(street, _bucket(price))
                allin = eff_to == my_bet + my_stack or eff_to == opp_max
                rr = 0.0 if allin else reraise
                # If they raise back, assume all-in: we fold or call.
                ev = 0.0
                for pk, t_k in zip(post, T_GRID):
                    if pk < 1e-4:
                        continue
                    kept = bisect_right(absl, t_k)
                    fold = _fold_chance(1.0 - kept / n)
                    nr = int(rr * kept)
                    e_call = mean(nr, kept)
                    e_call = eff(e_call, allin)
                    call_ev = e_call * (pot + my_add + opp_add) - my_add
                    if rr > 0:
                        e_r = mean(0, nr)
                        shove_ev = max(-my_add,
                                       e_r * allin_pot - (cap - my_bet))
                    else:
                        shove_ev = 0.0
                    ev += pk * (fold * pot + (1 - fold) * (
                        (1 - rr) * call_ev + rr * shove_ev))
                options.append((ev, PlayerAction.raise_to(me, r)))

        best = max(options, key=lambda o: o[0])
        if self.debug:
            self.trace = (round(e_all, 3), [(round(ev, 1), o.kind, o.amount)
                                            for ev, o in options])
        return best[1]

    def _decide_multi(self, gs):
        """Simple equity play for tables of three or more."""
        me = self.player_id
        board = list(self.t_board)
        opponents = max(1, len(gs.players_in_hand) - 1)
        dead = set(self.h_hole + board)
        deck = [c for c in range(52) if c not in dead]
        need = 5 - len(board)
        won = 0.0
        trials = 300
        for _ in range(trials):
            cards = self.rng.sample(deck, need + 2 * opponents)
            full = board + cards[:need]
            mine = evaluate(self.h_hole + full)
            best = max(evaluate(cards[need + 2 * i:need + 2 * i + 2] + full)
                       for i in range(opponents))
            if mine > best:
                won += 1
            elif mine == best:
                won += 0.5
        equity = won / trials
        if equity > 1.6 / (opponents + 1) and equity > 0.5 and gs.can_raise:
            lo, hi = gs.raise_range
            return PlayerAction.raise_to(me, int(min(hi, max(lo, gs.pot))))
        if gs.call_amount == 0 or equity > gs.pot_odds + 0.05:
            return PlayerAction.call(me)
        return PlayerAction.fold(me)
