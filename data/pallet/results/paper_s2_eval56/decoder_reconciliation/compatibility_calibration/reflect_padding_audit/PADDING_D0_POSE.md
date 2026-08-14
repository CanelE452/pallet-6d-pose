# D0 pose on the recovered frames

```
        arm  PnP solved  reproj med  reproj max  yaw med  rot med  trans med m  >100px
──────────────────────────────────────────────────────────────────────────────────────
   original           0         nan         nan      nan      nan          nan       0
    reflect           7        66.6       111.1     12.0     17.0        0.636       1
  replicate          10        51.9       170.6     13.6     25.0        0.763       3
constant127          10        46.8       104.2      6.7     22.0        0.672       1
```

Ten of thirteen frames that could not be solved at all now produce a pose.  None
of them is good enough: the gate allows a 30px median and the best arm gives
46.8px, with individual frames beyond 100px on every arm.  Yaw is the one
encouraging number -- A3's median is 6.7 degrees, inside the 15 degree
allowance -- which fits the picture that the pallet is recognised and roughly
oriented while its corners are placed badly.
