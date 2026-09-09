"""Pre-training candidate adequacy check. Oracle numbers are NOT predictions."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from .cache_io import ExportDataset,write_json,sha256
from .objective import candidate_targets


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    if Path(args.output).exists():raise FileExistsError('Do not overwrite an oracle audit')
    data=ExportDataset(args.manifest,verify_targets=True)
    if data.document.get('role') not in ('train','calibration'):
        raise ValueError('Preflight may read source train/calibration, never validation/real/FINAL')
    rows=[]
    for i,r in enumerate(data.records):
        if r.get('origin')!='source_synthetic':raise ValueError('Use source-synthetic predictions, not manufactured success cases')
        o=data.observation(i);t=data.supervision(i)
        mask=t['supervised']&t['matched'][:,None]&o.point_valid
        truth=candidate_targets(o.layouts,t['points'],mask,o.diagonal)
        if not truth['frame_valid'][0]:
            rows.append(dict(frame_id=r['frame_id'],observed=False));continue
        value=truth['mean_px'][0].masked_fill(~o.candidate_valid[0],float('inf'))
        best=int(value.argmin());base=float(value[0]);oracle=float(value[best]);d=float(o.diagonal[0])
        rows.append(dict(frame_id=r['frame_id'],observed=True,base_mean_px=base,oracle_layout_mean_px=oracle,
                         base_normalized=base/d,oracle_normalized=oracle/d,oracle_index=best,improvement_px=base-oracle))
    seen=[r for r in rows if r['observed']]
    base=float(np.mean([r['base_normalized'] for r in seen])) if seen else 0.
    oracle=float(np.mean([r['oracle_normalized'] for r in seen])) if seen else 0.
    gain=(base-oracle)/base if base>0 else 0.
    wins=sum(r['improvement_px']>=1. for r in seen)
    write_json(args.output,dict(schema='pointline_v4_oracle_audit_1',manifest_sha256=sha256(args.manifest),
        interpretation='GT best EXISTING WHOLE layout; diagnostic upper bound, not deployable accuracy. Identity makes nonworsening tautological.',
        records=len(rows),observed_frames=len(seen),baseline_normalized=base,oracle_normalized=oracle,
        relative_headroom=gain,frames_with_at_least_one_pixel_mean_gain=wins,
        exploratory_entry_gate=bool(gain>=.01 and wins>=32),threshold_basis='Unvalidated pilot budget rule: >=1% headroom AND >=32 naturally improvable frames; not a literature standard.',rows=rows))

if __name__=='__main__':main()
