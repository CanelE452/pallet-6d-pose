# Failure taxonomy

```
                class  frames  share
────────────────────────────────────
T2_GLOBAL_NO_RESPONSE      13   100%
```

**All 13 are T2_GLOBAL_NO_RESPONSE**: three or fewer corner channels above 0.30
and a centroid at or below 0.30.  T1 (centroid-only) is **0**, T4
(target/validity defect) is **0** -- every centroid GT point is present, and
where it is inside the frame the map is not contradicting a valid target, the
model simply has no response anywhere.

Conditions as fixed in advance:

```
T1  >= 6 of 8 valid corners above 0.30, corner median GT error <= 20px, centroid <= 0.30
T2  <= 3 corners above 0.30 and centroid <= 0.30
T3  peaks exist but GT-local mass low / argmax far from GT
T4  centroid GT missing, or inside-frame GT with zero local mass and a non-zero peak
T5  none or several matched; evidence recorded rather than forced
```
