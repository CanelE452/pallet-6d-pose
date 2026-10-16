"""Verify cached inputs, population identity and projected synthetic targets."""
from pathlib import Path
import hashlib
import json
import sys
import torch

KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[2]
sys.path.insert(0, str(KIT))
from plpose_local.geometry import corners, project
from cli_train import sha


def main():
    source = ROOT/'data/pallet/results/pallet_point_line_v5_local/export'
    out = ROOT/'data/pallet/results/pallet_point_line_decisive_v6/PREFLIGHT.json'
    if out.exists():
        raise SystemExit('preflight receipt already exists')
    by_split, report, hashes = {}, {}, {}
    worst = 0.
    for split in ('train','calibration','synth_val'):
        p = source/f'{split}.json'; hashes[str(p)] = sha(p)
        doc = json.loads(p.read_text()); rows = doc['records']
        ids, images = set(), set()
        errors = []
        for row in rows:
            assert row['id'] not in ids
            ids.add(row['id'])
            image = ROOT/row['original_image']
            assert sha(image) == row['source_image_sha256']
            images.add(row['source_image_sha256'])
            for group in ('observation','supervision'):
                assert sha(source/row[group]) == row[group+'_sha256']
            o = torch.load(source/row['observation'],weights_only=True,map_location='cpu')
            s = torch.load(source/row['supervision'],weights_only=True,map_location='cpu')
            assert o['features'].shape == (1,192,24,24)
            assert torch.isfinite(o['features']).all()
            assert torch.isfinite(s['points']).all()
            uv, _ = project(o['K'],s['R'],s['t'],corners(o['dims']))
            err = float((uv-s['points']).abs().max());errors.append(err)
            assert err < .1
            assert row['symmetry']['order'] in (1,2)
        by_split[split] = (ids,images)
        report[split] = dict(n=len(rows),unique_image_sha=len(images),
            projected_target_max_abs_px=max(errors),
            sessions=sorted({r['session'] for r in rows}))
        worst = max(worst,max(errors))
        print('preflight',split,len(rows),flush=True)
    overlaps = {}
    for a,b in (('train','calibration'),('train','synth_val'),('calibration','synth_val')):
        overlap = dict(id=len(by_split[a][0]&by_split[b][0]),
                       image_sha=len(by_split[a][1]&by_split[b][1]))
        assert overlap == dict(id=0,image_sha=0)
        overlaps[f'{a}/{b}'] = overlap
    receipt = dict(PASS=True,split_report=report,overlaps=overlaps,
        manifest_sha256=hashes,projected_target_max_abs_px=worst,
        scope='New export target/pose internal consistency and actual source-image byte separation. '
              'Original renderer/frame audit is inherited from OBJECT_FRAME_A.json; this does not '
              'independently prove real-image manual annotation quality.',
        upstream_frame_audit_sha256=sha(source.parent/'OBJECT_FRAME_A.json'))
    out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
