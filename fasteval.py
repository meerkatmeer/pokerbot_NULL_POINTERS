"""Fast hand evaluation on integer cards.

A card is an int 0..51: rank * 4 + suit, rank 0 (deuce) .. 12 (ace),
suit 0..3 in "CDHS" order. evaluate() takes 5 to 7 such ints and returns
an int: higher wins, equal splits. Only compare hands of the same size.
"""

RANKS = "23456789TJQKA"
SUITS = "CDHS"

_POP = [bin(m).count("1") for m in range(1 << 13)]
# Ranks present in a 13-bit mask, highest first.
_RL = [tuple(r for r in range(12, -1, -1) if m >> r & 1) for m in range(1 << 13)]
# Top rank of the best straight in a 13-bit mask (wheel = 3), or 0.
_STRAIGHT = [0] * (1 << 13)
for _m in range(1 << 13):
    _ext = (_m << 1) | (_m >> 12 & 1)  # bit 0 is the ace played low
    for _top in range(13, 3, -1):
        if (_ext >> (_top - 4)) & 31 == 31:
            _STRAIGHT[_m] = _top
            break
del _m, _ext, _top


def card_int(card):
    """A card_utils.Card as an int."""
    return RANKS.index(card.rank) * 4 + SUITS.index(card.suit)


def evaluate(cards):
    sm = [0, 0, 0, 0]
    cnt = [0] * 13
    for c in cards:
        r = c >> 2
        sm[c & 3] |= 1 << r
        cnt[r] += 1
    for m in sm:
        if _POP[m] >= 5:
            t = _STRAIGHT[m]
            if t:
                return 0x800000 | t
            rl = _RL[m]
            return (0x500000 | rl[0] << 16 | rl[1] << 12 | rl[2] << 8
                    | rl[3] << 4 | rl[4])
    rm = sm[0] | sm[1] | sm[2] | sm[3]
    if _POP[rm] == len(cards):
        t = _STRAIGHT[rm]
        if t:
            return 0x400000 | t
        rl = _RL[rm]
        return rl[0] << 16 | rl[1] << 12 | rl[2] << 8 | rl[3] << 4 | rl[4]
    quad = -1
    trips = []
    pairs = []
    for r in _RL[rm]:
        k = cnt[r]
        if k == 1:
            continue
        if k == 4:
            quad = r
        elif k == 3:
            trips.append(r)
        else:
            pairs.append(r)
    if quad >= 0:
        return 0x700000 | quad << 4 | _RL[rm & ~(1 << quad)][0]
    if trips:
        t = trips[0]
        if len(trips) > 1 or pairs:
            p = trips[1] if len(trips) > 1 else -1
            if pairs and pairs[0] > p:
                p = pairs[0]
            return 0x600000 | t << 4 | p
    s = _STRAIGHT[rm]
    if s:
        return 0x400000 | s
    if trips:
        rl = _RL[rm & ~(1 << t)]
        return 0x300000 | t << 8 | rl[0] << 4 | rl[1]
    if len(pairs) >= 2:
        p1, p2 = pairs[0], pairs[1]
        rl = _RL[rm & ~(1 << p1) & ~(1 << p2)]
        return 0x200000 | p1 << 8 | p2 << 4 | rl[0]
    p = pairs[0]
    rl = _RL[rm & ~(1 << p)]
    return 0x100000 | p << 12 | rl[0] << 8 | rl[1] << 4 | rl[2]


def draw_bonus(hole, board):
    """True when hole + board (flop or turn) hold a flush draw or an
    open-ended straight draw that uses a hole card."""
    cards = list(hole) + list(board)
    for s in range(4):
        if sum(1 for c in cards if c & 3 == s) == 4 and any(
                c & 3 == s for c in hole):
            return True
    rm = 0
    for c in cards:
        rm |= 1 << (c >> 2)
    hm = 0
    for c in hole:
        hm |= 1 << (c >> 2)
    ext = (rm << 1) | (rm >> 12 & 1)
    hext = (hm << 1) | (hm >> 12 & 1)
    for lo in range(0, 11):
        w = (ext >> lo) & 15
        if w == 15 and (hext >> lo) & 15:
            return True
    return False


def class_index(a, b):
    """Index 0..168 of a two-card starting hand class."""
    ra, rb = a >> 2, b >> 2
    if ra < rb:
        ra, rb = rb, ra
    if ra == rb:
        return ra * 13 + ra
    if (a & 3) == (b & 3):
        return ra * 13 + rb  # suited: high row, low column
    return rb * 13 + ra  # offsuit: mirrored


ALL_COMBOS = [(a, b) for a in range(52) for b in range(a + 1, 52)]
