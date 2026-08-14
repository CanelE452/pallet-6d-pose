# C13 hash reconciliation

Two hashes exist for the same control set and they differ.  The frame lists do
not.

```
prior (no_response_frames/nrf_membership.json)  9230daa96f515e11...
lock  (pdg_membership_lock.json)                8f305251980104cc...
```

## Verdict: CASE 1 -- serialization only, no membership drift

```
exact frame ID lists identical      True
symmetric difference                empty (both directions)
n                                   13 and 13
```

The two numbers hash different objects:

```
prior  sha256(json.dumps({R0, R1, C0}, sort_keys=True))
       a three-key dict covering all 26 development frames, not C13 alone
       recomputed -> 9230daa96f515e11...   matches the record

lock   sha256(json.dumps(sorted(C13), sort_keys=True))
       the control list on its own
       recomputed -> 8f305251980104cc...   matches the record
```

Both reproduce exactly under their own definition, so neither file is wrong and
no control frame was ever reselected.

## Canonical definition from here on

```
sha256(json.dumps(sorted(frame_id_list), sort_keys=True))
C13 = 8f305251980104cc739651dc3767cb41e9de42ea4844f50124a68ca1691f7f34
```

Set membership is hashed as a sorted list of frame IDs, one set at a time.  The
older grouped-dict form stays valid as a record of the earlier run and is not
recomputed retroactively.
