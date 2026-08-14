# PAPER_S2 RGB1 PnP `t_z`-only audit

## Decision

STOP before full training.  The one-sided PnP-depth objective has no active
training examples under the locked existing-data recipe, so a 1,500-step run
would not optimize the real severe-UC failure mode.

## Locked hypothesis

- Base: `weights/paper_s2_stageB/net_epoch_0057.pth`
- Single RGB, input 400, belief 50
- Existing six Stage-B training directories and 60:40 sampler only
- Belief tail only (`m6_2.10/12`, 17,673 parameters)
- No projected-span, edge, mask-extent, depth image, or new data
- Candidate loss:

  `Huber(ReLU(log(t_z_pred / t_z_gt) - log(1.15)))`

## Existing-data activation audit

The ep57 model was evaluated with learning rate zero for 50 balanced batches
(600 sampled frames).  DiffPnP had 52.83% valid coverage, or 317 valid PnP
frames.

| diagnostic | result |
|---|---:|
| mean `t_z_pred/t_z_gt` | 0.998 |
| mean minimum projected-span ratio | 0.980 |
| fraction with `t_z_pred/t_z_gt > 1.20` | 0.0% |
| raw one-sided depth loss | 0.0 |
| weighted loss / belief loss | 0.0 |

The real filterval failure tail is different: UC frames have median
`t_z_pred/t_z_gt=1.263`, versus 1.061 for non-UC frames.  This is a genuine PnP
failure signature, but it is absent from the permitted training distribution.
Lowering the deadband would supervise ordinary synthetic jitter rather than the
real severe tail and would introduce a global near-depth bias.

## Prior short-screen evidence

- Direct projected-span v1: real UC stayed 23--25 versus baseline 23.
- Hard W/depth edge v2: best legacy UC was 22, but the only resolved frame was
  still rejected as W/D ambiguous; raw UC stayed 15 and safe gross increased
  from 20 to 23.
- PnP contraction guard at 0.75: baseline accepted gross reduced 27 to 20,
  while all seven safe-good poses were retained.

Therefore the deployable improvement from this work is the fail-closed PnP
contraction guard.  The trained checkpoint remains ep57; none of the scoped
fine-tunes met the pre-registered model replacement gate.
