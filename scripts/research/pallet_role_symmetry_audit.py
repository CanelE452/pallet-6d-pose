"""Post-hoc correspondence audit of frozen v2 predictions; no training or GT edits.

Run with the pallet-yolo26 Python environment. The four-permutation minimum is
an ID-insensitive diagnostic, NOT a physical C4 pose metric for these assets.
"""
from pathlib import Path
import hashlib
import json
import sys
import html

import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/pallet/results/pallet_role_selection_v2/foreground_matched'
OUT = ROOT / 'data/pallet/results/pallet_role_symmetry_audit'
sys.path.insert(0, str(ROOT))
from scripts.annotate.convert_to_camera_facing_v4 import compute_perm_v4

PERMS = np.array([[0,1,2,3,4,5,6,7], [1,5,6,2,0,4,7,3],
                  [5,4,7,6,1,0,3,2], [4,0,3,7,5,1,2,6]])
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def cluster_ci(delta, mask, groups):
    labels = sorted(set(groups[mask]))
    sums = np.array([delta[mask & (groups == g)].sum() for g in labels])
    counts = np.array([(mask & (groups == g)).sum() for g in labels])
    draws = np.random.default_rng(917).integers(0, len(labels), (10000, len(labels)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    return dict(mean=float(delta[mask].mean()), low=float(np.quantile(means,.025)),
                high=float(np.quantile(means,.975)), clusters=len(labels),
                scope='post-hoc, seed-average, scene-cluster paired bootstrap; conditional on these meshes')

def main():
    OUT.mkdir(exist_ok=True)
    files = [SOURCE/n for n in ['test_records.json','predictions.npz','PER_FRAME.json','SUMMARY.json','RENDER_COMPLETE.json']]
    files += [ROOT/'scripts/research/pallet_role_selection_v2'/n for n in ['model.py','foreground_train.py','foreground_render.py']]
    files += [ROOT/'scripts/annotate/convert_to_camera_facing_v4.py', ROOT/'challenge/config/SQUARE_PALLET_SYMMETRY_CONTRACT.json']
    before = {str(p.relative_to(ROOT)):sha(p) for p in files}
    rows = read(SOURCE/'test_records.json')
    cached = read(SOURCE/'PER_FRAME.json')
    predictions = np.load(SOURCE/'predictions.npz')
    gt = np.array([r['gt'] for r in rows])
    matched = np.array([r['matched'] for r in cached['stock']])
    seen = np.array([r['seen'] for r in rows])
    groups = np.array([r['group'] for r in rows])
    assets = np.array([r['asset'] for r in rows])
    diag = np.array([np.linalg.norm(r['raw_hw']) for r in rows])
    masks = {'seen': matched & seen, 'unseen': matched & ~seen}
    masks.update({a:matched & (assets==a) for a in sorted(set(assets))})
    names = ['stock','canonical'] + [f'{arm}_s{s}' for arm in ['direct','selected'] for s in [1,2,3]]
    errors, result, arrays = {}, {}, {}
    edgeset = {tuple(sorted(e)) for e in EDGES}
    for perm in PERMS:
        assert {tuple(sorted((perm[a],perm[b]))) for a,b in EDGES} == edgeset
        assert set(perm[[0,1,4,5]]) == {0,1,4,5}
    for name in names:
        p = predictions[name][:,:8]
        ce = np.linalg.norm(p[:,PERMS]-gt[:,None],axis=-1).mean(-1)
        strict, orbit = ce[:,0], ce.min(-1)
        # Independent eight-point assignment lower bound; does not preserve topology.
        assignment = []
        for pp,gg in zip(p,gt):
            cost = np.linalg.norm(pp[:,None]-gg[None],axis=-1)
            a,b = linear_sum_assignment(cost)
            assignment.append(cost[a,b].mean())
        assignment = np.array(assignment)
        assert np.all(assignment <= orbit + 1e-8)
        assert np.all(orbit <= strict + 1e-8)
        for i in np.flatnonzero(matched):
            assert abs(strict[i]-cached[name][i]['matched_px']) < 1e-8
        errors[name] = (strict,orbit,assignment,ce.argmin(-1))
        arrays[name+'_strict'] = strict
        arrays[name+'_orbit'] = orbit
        arrays[name+'_assignment'] = assignment
        result[name] = {}
        for label, mask in masks.items():
            result[name][label] = dict(n=int(mask.sum()), strict_px=float(strict[mask].mean()),
                orbit_px=float(orbit[mask].mean()), assignment_px=float(assignment[mask].mean()),
                orbit_median_px=float(np.median(orbit[mask])), orbit_p90_px=float(np.quantile(orbit[mask],.9)),
                nonidentity_best=int(((ce.argmin(-1)!=0)&mask).sum()),
                strict_gt20_orbit_le10=int(((strict>20)&(orbit<=10)&mask).sum()),
                orbit_gt20=int(((orbit>20)&mask).sum()))
    assert np.allclose(errors['canonical'][1][matched],errors['stock'][1][matched],atol=1e-10)
    # Non-matched, degenerate boxes can make camera-facing canonicalization
    # leave the cyclic group. These retain the common miss penalty, not px scores.
    canonical_outside = int((np.abs(errors['canonical'][1]-errors['stock'][1])>1e-8).sum())
    comparisons = {}
    average = {a:np.mean([errors[f'{a}_s{s}'][1] for s in [1,2,3]],axis=0) for a in ['direct','selected']}
    for label in ['seen','unseen']:
        mask = masks[label]
        population = seen if label=='seen' else ~seen
        comparisons[label] = dict(selected_minus_stock=cluster_ci(average['selected']-errors['stock'][1],mask,groups),
            selected_minus_direct=cluster_ci(average['selected']-average['direct'],mask,groups),
            gain_vs_stock_per_seed=[float(1-errors[f'selected_s{s}'][1][mask].mean()/errors['stock'][1][mask].mean()) for s in [1,2,3]],
            new_orbit_gt20_from_stock_le10=[int(((errors['stock'][1]<=10)&(errors[f'selected_s{s}'][1]>20)&mask).sum()) for s in [1,2,3]],
            all_frame_normalized_orbit={n:float(np.where(matched,np.minimum(1,errors[n][1]/diag),1)[population].mean()) for n in names})
    geometry = {}
    for asset in sorted(set(assets)):
        sample = [r for r in rows if r['asset']==asset and r['view']==0 and r['state']=='clean']
        ratios = np.array([max(r['dims_xyz'][:2])/min(r['dims_xyz'][:2]) for r in sample])
        geometry[asset] = dict(groups=len(sample),footprint_aspect_min=float(ratios.min()),footprint_aspect_max=float(ratios.max()),within_5_percent_square=int((ratios<=1.05).sum()))
    rng = np.random.default_rng(912)
    jitter = []
    for r in rows:
        origin, uv = np.array(r['Xcf']),np.array(r['gt'])
        assert compute_perm_v4(origin,uv) == list(range(9))
        jitter.append(np.mean([compute_perm_v4(origin,uv+rng.normal(0,1,uv.shape)) != list(range(9)) for _ in range(32)]))
    jitter = np.array(jitter)
    write(OUT/'SUMMARY.json',dict(source_hashes=before, training_steps=0, evaluation_images=576,
        status='POST_HOC_CORRESPONDENCE_DIAGNOSIS_NOT_CONFIRMATORY_POSE_EVALUATION',
        metric='Mean Euclidean error over 8 corners, matched proposals only; orbit is minimum over 4 topology-preserving cyclic ID permutations, GT used for evaluation only.',
        limits=['Not physical C4 certification for the synthetic meshes.', 'Detection misses excluded from px table, retained in all-frame normalized diagnostic.',
                'Same frozen models; not new inference, not evidence of a new symmetry-aware trained loss.', 'No real-image or semantic rotation evaluation.'],
        results=result,comparisons=comparisons,geometry=geometry,
        gt_jitter_1px_32_trials={label:float(jitter[mask].mean()) for label,mask in masks.items()}))
    np.savez_compressed(OUT/'per_frame.npz',ids=np.array([r['id'] for r in rows]),matched=matched,**arrays)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    figures = []
    for label, fid in [('previous_failure','a2_test_g009_v2_crop25'),('previous_gain','a2_test_g009_v0_clean')]:
        i = next(i for i,r in enumerate(rows) if r['id']==fid)
        # The right two panels use identical predicted coordinates; only indices change.
        panels = [('YOLO',predictions['stock'][i,:8]),('Selected + residual',predictions['selected_s1'][i,:8]),
                  ('Same prediction, GT-best IDs',predictions['selected_s1'][i,PERMS[errors['selected_s1'][3][i]]])]
        canvas = Image.new('RGB',(1920,570),'#f6f7fa'); draw = ImageDraw.Draw(canvas)
        draw.text((12,8),f'{fid} | scene_2.usd | green: GT; magenta: prediction | right panel: diagnostic only',fill='black',font=small)
        for j,(title,p) in enumerate(panels):
            raw=Image.open(ROOT/rows[i]['image']).convert('RGB')
            canvas.paste(raw,(640*j,90))
            # Keep the cropped-out region white and show offscreen coordinates there.
            e=np.linalg.norm(p-gt[i],axis=-1).mean()
            draw.text((640*j+12,38),title,fill='black',font=font)
            draw.text((640*j+12,63),f'Indexed error: {e:.2f} px',fill='black',font=small)
            for pts,col,offset in [(gt[i],'#00db86',(-12,-20)),(p,'#f52380',(4,3))]:
                for k,(x,y) in enumerate(pts):
                    x+=640*j;y+=90
                    if 640*j<=x<640*(j+1) and 90<=y<570:
                        draw.ellipse((x-4,y-4,x+4,y+4),outline=col,width=2)
                        draw.text((x+offset[0],y+offset[1]),str(k),font=small,fill=col,stroke_width=1,stroke_fill='#222222')
        path=OUT/(label+'.png');canvas.save(path)
        figures.append(dict(label=label,id=fid,path=path.name,asset=rows[i]['asset'],dims_xyz=rows[i]['dims_xyz'],
                            stock_strict_px=float(errors['stock'][0][i]),selected_strict_px=float(errors['selected_s1'][0][i]),
                            selected_orbit_px=float(errors['selected_s1'][1][i])))
    write(OUT/'FIGURES.json',figures)
    montage=Image.new('RGB',(1280,1080),'white'); d=ImageDraw.Draw(montage)
    for ai in range(4):
        r=next(r for r in rows if r['asset_index']==ai and r['group']=='test_g009' and r['view']==0 and r['state']=='clean')
        x,y=(ai%2)*640,(ai//2)*540
        d.text((x+10,y+8),f'{r["asset"]} | {"seen" if r["seen"] else "unseen"}',font=font,fill='black')
        d.text((x+10,y+32),'X/Y/H (m): '+ '/'.join(f'{v:.3f}' for v in r['dims_xyz']),font=small,fill='black')
        montage.paste(Image.open(ROOT/r['image']),(x,y+60))
    montage.save(OUT/'asset_identity.jpg')
    assert before == {str(p.relative_to(ROOT)):sha(p) for p in files}
    write(OUT/'AUDIT.json',dict(PASS=True, frozen_sources_unchanged=True,indexed_metric_replay=True,
        topology_preserved=True, canonical_matched_has_identical_orbit_error=True,
        canonical_unmatched_outside_cyclic_group=canonical_outside,independent_assignment_lower_bound=True,
        exact_gt_relabel_identity=True, script_sha256=sha(Path(__file__))))
    print(json.dumps(dict(comparisons=comparisons,geometry=geometry,figures=figures),indent=2))

if __name__=='__main__':
    main()
