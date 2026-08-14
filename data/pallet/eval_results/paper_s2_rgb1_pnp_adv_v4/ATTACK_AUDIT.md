# PAPER_S2 targeted PnP-collapse attack audit

Base: `weights/paper_s2_stageB/net_epoch_0057.pth`; batches=4; batch=12; target scale=0.8; low-frequency grid=25x25.

This audit does not train or alter a checkpoint. It uses the locked six training directories and keeps the original labels unchanged.

| attack | raw min span | PnP tz>1.15 | all-8 retention | gate |
|---|---:|---:|---:|---|
| eps2_steps1 | 0.990 | 0.0% | 96.4% | STOP |
| eps2_steps2 | 0.989 | 3.4% | 100.0% | STOP |
| eps4_steps1 | 0.984 | 0.0% | 96.4% | STOP |
| eps4_steps2 | 0.978 | 3.4% | 96.4% | STOP |
| eps8_steps1 | 0.981 | 6.9% | 92.9% | STOP |
| eps8_steps2 | 0.971 | 6.9% | 96.4% | STOP |

Pre-registered feasibility gate: median raw minimum-span ratio <=0.90, PnP-valid tz ratio >1.15 in >=20%, and all-eight detection retention >=95%. A fine-tune is allowed only for an attack arm satisfying all three conditions.
