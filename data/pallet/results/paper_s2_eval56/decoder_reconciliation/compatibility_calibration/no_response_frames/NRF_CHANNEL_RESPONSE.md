# The 13 frames, channel by channel

H6, every channel against its own GT point.  `corners > 0.30` counts the valid
corner channels whose raw peak clears the same gate the centroid has to clear.

```
grp   domain  centroid pk  corners>0.30  corner pk med  corner pk max  corner GT err px  corner GT mass 5x5  centroid GT mass  class
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 R0  outside       0.0015           0/8         0.0009         0.0061               174              0.0094            0.0146     T2
 R0  outside       0.0032           0/8         0.0111         0.0214               136              0.0459            0.0098     T2
 R0    night       0.0034           0/8         0.0098         0.0107               244              0.0016            0.0143     T2
 R0    night       0.0126           0/8         0.0321         0.0421               292              0.0012            0.1062     T2
 R0    night       0.0131           0/8         0.0285         0.0339               326              0.0001            0.0252     T2
 R0  outside       0.0146           2/8         0.0793         0.6311               164              0.0705            0.0666     T2
 R0    night       0.0181           0/8         0.0759         0.1872               292              0.0005            0.0606     T2
 R0    night       0.0220           0/8         0.0290         0.0586               338              0.0001            0.0266     T2
 R0  outside       0.0268           0/8         0.0676         0.1058               172              0.0255            0.0239     T2
 R1    night       0.0952           0/8         0.0426         0.0762               337              0.0001            0.0071     T2
 R1    night       0.1584           0/8         0.0653         0.2644               122              0.0013            0.1327     T2
 R1    night       0.2521           0/8         0.0759         0.1244               174              0.0171            0.0187     T2
 R1  outside       0.2816           0/8         0.1407         0.1772                91              0.0817            0.3915     T2
```

## Matched comparison

```
         group  centroid peak  corners>0.30  corner peak med  corner GT err px  corner GT mass  centroid GT mass
────────────────────────────────────────────────────────────────────────────────────────────────────────────────
no-response 13         0.0181           0.0           0.0426             174.1          0.0016            0.0252
    control 13         0.8139           8.0           0.6583              29.7          0.3128            0.4124
```

**The corners die with the centroid.**  On 12 of the 13 frames not one corner
channel clears 0.30, and on the thirteenth only two do.  Corner peak medians run
0.0009 to 0.14 against a control median of 0.66, and the mass the corner
channels place within 5x5 of their own GT point is 0.0016 against 0.31.

The premise the Dual-Bandwidth hypothesis needed -- shared features still on the
pallet, only the centroid representation failing -- does not hold on a single
frame.

## A caveat about the controls

The controls are matched for being *centroid-alive*, not for being healthy.
Their own corner error median is 29.7px and 9 of 13 classify as
T3_LOCALIZATION_WRONG.  They are the right comparison for "does the centroid
fail selectively", and the wrong one for "what a good frame looks like".
