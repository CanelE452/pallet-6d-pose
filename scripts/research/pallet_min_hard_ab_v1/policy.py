"""Pure model-independent sampling rules; no filesystem/model imports."""
from collections import Counter, defaultdict
import hashlib
import math

def key(prefix, s):
    return hashlib.sha256((prefix + s).encode()).hexdigest()

def temporal_queue(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['recording']].append(r)
    result = []
    for rec in sorted(grouped):
        seq = sorted(grouped[rec], key=lambda r: (r['order'], r['frame_id']))
        nbin = min(48, len(seq))
        # numpy.array_split convention: early bins get one extra item.
        q, rem = divmod(len(seq), nbin)
        start = 0
        for b in range(nbin):
            stop = start + q + (b < rem)
            winner = min(seq[start:stop], key=lambda r: key('hard-tag-v1:', r['frame_id']))
            result.append(dict(winner, bin=b, round=b % 3 + 1))
            start = stop
    return sorted(result, key=lambda r: (r['round'], r['recording'], r['bin']))

def hard_capacity(rows):
    counts = Counter(r['recording'] for r in rows if r['tag'] in ('MODERATE', 'SEVERE'))
    return sum(min(3, n) for n in counts.values()), len(counts)

def round_decision(rows, round_no):
    hard = [r for r in rows if r['tag'] in ('MODERATE', 'SEVERE')]
    cap, nrec = hard_capacity(hard)
    if len(hard) >= 12 and nrec >= 3 and cap >= 10:
        return 'SELECT'
    if round_no < 3:
        return 'NEXT_ROUND'
    # After exhausting the queue reserve is optional, initial8/diversity remain mandatory.
    return 'SELECT' if len(hard) >= 8 and nrec >= 3 and cap >= 8 else 'INSUFFICIENT'

def select_hard(rows):
    """DP chooses initial8 nearest 4S/4M, then up to2 reserve under total cap3.

    Within each recording/severity the deterministic hash prefix is always used.
    Tie breaks: highest achievable reserve count, then lexicographic hash vector.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['tag'] in ('MODERATE', 'SEVERE'):
            groups[r['recording']][r['tag']].append(r)
    for g in groups.values():
        for seq in g.values():
            seq.sort(key=lambda r: key('min-hard-ab-v1:', r['frame_id']))
    total_s = sum(len(g['SEVERE']) for g in groups.values())
    desired_s = min(4, total_s)
    # state = (total, severe, used_recordings); value = selected rows
    states = {(0, 0, 0): []}
    rank = lambda rs: tuple(sorted(key('min-hard-ab-v1:', r['frame_id']) for r in rs))
    for rec in sorted(groups):
        nxt = {}
        g = groups[rec]
        for (n, s, recs), selected in states.items():
            for ns in range(min(3, len(g['SEVERE'])) + 1):
                for nm in range(min(3-ns, len(g['MODERATE'])) + 1):
                    if n + ns + nm > 8:
                        continue
                    k = (n + ns + nm, s + ns, recs + bool(ns + nm))
                    v = selected + g['SEVERE'][:ns] + g['MODERATE'][:nm]
                    if k not in nxt or rank(v) < rank(nxt[k]):
                        nxt[k] = v
        states = nxt
    valid = [(k, rs) for k, rs in states.items() if k[0] == 8 and k[2] >= 3]
    if not valid:
        raise ValueError('Cannot obtain initial8 under >=3 recordings / max3.')
    _, initial = min(valid, key=lambda t: (abs(t[0][1]-desired_s), -t[0][1], rank(t[1])))
    initial.sort(key=lambda r: key('min-hard-ab-v1:', r['frame_id']))
    counts = Counter(r['recording'] for r in initial)
    ids = {r['frame_id'] for r in initial}
    reserve = []
    for r in sorted((r for r in rows if r['tag'] in ('MODERATE', 'SEVERE')), key=lambda r: key('min-hard-ab-v1:', r['frame_id'])):
        if r['frame_id'] in ids or counts[r['recording']] >= 3:
            continue
        reserve.append(r); counts[r['recording']] += 1
        if len(reserve) == 2:
            break
    assert max(counts.values()) <= 3
    return initial, reserve
