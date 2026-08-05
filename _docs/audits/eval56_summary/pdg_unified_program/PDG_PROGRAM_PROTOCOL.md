# PDG-Net Unified Program — protocol and state

## Holdout state

```
E44-clean  44 frames  sha dff15140ff5d7a4c20b0cb40a73c73d2   SEALED
W45        45 frames  sha 48c77526befb954b49ac24925f4343f2   SEALED
```

**No inference has been run on either set in this session.**  They stay sealed
until every weight, gate and report template is frozen, and the old PPD arms are
evaluated in the same run.

## Membership lock (Phase A)

```
                        set   n       sha256 (16)
─────────────────────────────────────────────────
N87 (development reference)  87  2d9efb4d806ea722
   D13 (global no-response)  13  11c23fd60816127f
      C13 (matched control)  13  8f305251980104cc
         E44-clean (SEALED)  44  dff15140ff5d7a4c
               W45 (SEALED)  45  48c77526befb954b
```

Assertions, all satisfied:

```
D13 inter E44 = 0     C13 inter E44 = 0
D13 inter W45 = 0     C13 inter W45 = 0
N87 inter eval56 = 12   (E44 is eval56 minus exactly these)
```

## Phase B parity — reproduced before anything else

```
                item                 primary            secondary               status
──────────────────────────────────────────────────────────────────────────────────────
    base ep57 eval56               PnP 50/56    reproj 11.5578 px     cached, verified
      base ep57 wood               PnP 44/45     reproj 9.2839 px     cached, verified
       base ep57 N87               PnP 70/87  reproj 23.161629 px    recomputed, exact
          PPD L0 N87         polarity 0.0233      inversion 84/86     matches recorded
          PPD M1 N87         polarity 0.0116      inversion 85/86     matches recorded
PPD L0 synthetic val         polarity 0.9801     inversion 15/754  matches 0.977-0.980
   PPD O0 oracle N87  polarity 1.000 (86/86)                    -     matches recorded
```

Old PPD checkpoints, from run_state not from guesswork:

```
arm  epochs completed  epoch_020 sha256
───────────────────────────────────────
 L0                20  649e06a3e7bc1b04
 M0                20  f4f0034c070f8b9f
 M1                20  50df95f0099d6f06
```

## Measured cost of the remaining programme

```
canonical Stage-B loader        29,308 samples
batch 12                        2,442 optimizer steps per epoch
measured step (stages 4-6)      0.444 s,  peak 3.09 GB of 10.3 GB
one epoch                       18.1 min
A1-A4 joint, 3 epochs each      3.61 h        <- not yet run
OCSH pretrain, 10 epochs        additional     <- not yet run
```

## What is complete

```
Phase A  membership lock and identity          DONE
Phase B  old-result parity                     DONE, all reproduced
Phase D  training distribution audit           DONE  <- see PDG_DATA_AUDIT.md
Phase C  PPD evaluator generalisation          NOT STARTED
Phase E-I  TACA / OCSH / GCFM / VAPA / targets NOT STARTED
Phase K-L  pretraining and joint training      NOT STARTED
Phase O-Z  evaluation, gates, figures, reports NOT STARTED
```
