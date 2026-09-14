# T1. Inputs and resources

| Model | Image evidence | New training in this run | Added parameters | Output / deployment assumptions |
|---|---|---|---:|---|
| R0 | RGB, fixed detector |0 |0 |9points, known-size PnP |
| P | existing P3/P4, predicted box/points |0; frozen3 historical6000-step heads |18,962 |candidate distribution→8corner movement; center fixed |
| L | same feature scales, predicted edges |0; frozen3 historical heads |19,810 |line readout/correction; original structural control |
| D |same candidate spatial evidence and context as P |3×6000 source-only updates |19,450 |direct residual L1, same final cap grid |
| PoseFix adaptation |separate RGB crop backbone+9pose maps |0 external full updates |NOT_YET_MEASURED |adapter tested; network integration not complete |

Total detector and allocated memory details are in RUNTIME_PANEL.json. Same sample counts do not imply equal runtime or FLOPs. ImageNet initialization for eventual PoseFix would be additional supervision and must be disclosed.
