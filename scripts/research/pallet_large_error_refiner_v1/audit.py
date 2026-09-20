"""Read-only artifact/prediction audit; writes a separate audit result."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.research.pallet_large_error_refiner_v1 import run as R
from scripts.research.pallet_large_error_refiner_v1.evaluate import NAMES, top


def main():
    protocol = R.verify(); split = R.E.read(R.DOC/'SPLIT.json')
    result = R.E.read(R.DOC/'RESULTS.json'); assert result['complete']
    predictions = R.E.read(R.RAW/'PREDICTIONS.json'); gates = R.E.read(R.RAW/'GATES.json')
    metrics = R.E.read(R.RAW/'PER_FRAME_METRICS.json')
    checks = 0; moves = {}; inits = []; rngs = []; fits = {}
    for arm in R.ARMS:
        f = R.E.read(R.DOC/f'FIT_{arm}.json'); R.F.verify(f['checkpoint'])
        ck = torch.load(R.ROOT/f['checkpoint']['path'],map_location='cpu',weights_only=False)
        assert ck['complete'] and ck['step']==2000
        assert ck['protocol_sha']==R.E.sha(R.DOC/'PROTOCOL.json')
        assert ck['input_sha']==R.E.sha(R.DOC/'TRAIN_INPUT_LOCK.json')
        assert all(torch.isfinite(t).all() for t in ck['state'].values())
        inits.append(ck['initial_sha']); rngs.append((ck['source_rng'],ck['other_rng']))
        fits[arm]=dict(elapsed_seconds=ck['elapsed_seconds'],checkpoint=f['checkpoint'])
    assert len(set(inits))==1
    assert torch.equal(rngs[1][0],rngs[2][0]) and torch.equal(rngs[1][1],rngs[2][1])
    for r in split['train']+split['evaluation']:
        for key in ('image','annotation','cache'): R.F.verify(r[key])
    for r in R.E.read(R.GREEN)['records']:
        for key in ('image','annotation','camera'): R.F.verify(r[key])
    for dataset, rows in predictions.items():
        assert len(rows)==(72 if dataset=='DEV72' else 150)
        gm = {r['id']:r['gates'] for r in gates[dataset]}
        moves[dataset] = {}
        for name in NAMES:
            distances = []
            for r in rows:
                raw = r['ungated']['R0']; pred = r['ungated'][name]
                assert raw['selected_index']==pred['selected_index']
                assert len(raw['candidates'])==len(pred['candidates'])
                for i,(a,b) in enumerate(zip(raw['candidates'],pred['candidates'])):
                    for key in a:
                        if key != 'keypoints_xy': assert np.array_equal(a[key],b[key])
                    if i==raw['selected_index']: assert np.array_equal(a['keypoints_xy'][8],b['keypoints_xy'][8])
                    else: assert np.array_equal(a['keypoints_xy'],b['keypoints_xy'])
                expected = pred if gm[r['id']][name]['accepted'] else r['ungated']['A_N2']
                assert r['gated'][name]==expected
                if top(raw) is not None:
                    delta = np.linalg.norm(np.asarray(top(pred)['keypoints_xy'])[:8]-np.asarray(top(raw)['keypoints_xy'])[:8],axis=-1)
                    distances.extend(delta.tolist())
                    if name in ('A_N2','B_CAP32'):
                        cap = (.01 if name=='A_N2' else .04)*np.linalg.norm(r['raw_hw'])
                    else:
                        box = np.asarray(top(raw)['box_xyxy']); cap = .20*np.linalg.norm(box[2:]-box[:2])
                    assert float(delta.max()) <= cap + 1e-3
                checks += 1
            d = np.array(distances)
            moves[dataset][name] = dict(corners=len(d),mean_px=float(d.mean()),P90_px=float(np.quantile(d,.9)),
                max_px=float(d.max()),moved_over8=int((d>8+1e-3).sum()),moved_over20=int((d>20).sum()))
    # Independent pooled-count recomputation from per-corner outputs.
    for dataset,policies in metrics.items():
        base = policies['ungated']['R0']
        for policy,arms in policies.items():
            for name, rows in arms.items():
                assert len(rows)==len(base)
                hard=good=recovered=damaged=total=correct=0
                for b,r in zip(base,rows):
                    assert b['id']==r['id']
                    if not b['evaluable']: continue
                    for v,a,z in zip(b['canonical_valid'],b['canonical_errors'],r['canonical_errors']):
                        if not v: continue
                        total+=1; correct+=z<=10; hard+=a>20; good+=a<5
                        recovered+=(a>20 and z<=10); damaged+=(a<5 and z>10)
                s=result['results'][dataset][policy][name]
                assert (s['hard'],s['good'],s['recovered'],s['damaged'])==(hard,good,recovered,damaged)
                assert s['corners']==total and abs(s['PCK']['10']-correct/total)<1e-12
    audit = dict(PASS=True,preservation_and_fallback_cases=checks,normal_and_flip_preservation_checks=result['all_preservation_checks'],
        source_locks_verified=True,all_three_last2000=True,identical_initial_state=inits[0],
        paired_D_E_corruption_rng_end_states=True,all_weights_finite=True,aggregate_counts_recomputed=True,
        train_frames=split['train_frames'],manual_corners=split['manual_corners'],fits=fits,movement=moves,
        bindings=[R.E.bound(R.DOC/'RESULTS.json'),R.E.bound(R.RAW/'PREDICTIONS.json'),R.E.bound(R.RAW/'GATES.json'),
            R.E.bound(R.RAW/'PER_FRAME_METRICS.json'),R.E.bound(Path(__file__))])
    R.freeze(R.DOC/'AUDIT.json',audit)
    print('AUDIT PASS',checks,'preservation/fallback cases')
    print('movement',moves)


if __name__=='__main__': main()
