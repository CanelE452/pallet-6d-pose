"""Independent read-only checks of adapter parity and saved trajectory algebra."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.research.pallet_posefix_replay_diagnosis_v1 import run as R
import numpy as np

def main():
    R.verify_sources(); inputs=R.read(R.RAW/'INPUTS.json'); targets=R.read(R.RAW/'TARGETS.json')
    pe,pop=R.population_metadata(); gt_sources=[]
    for item,meta in pop:
        t=pe.E._legacy_forbidden_target(item); saved=targets[item.frame_id]
        np.testing.assert_array_equal(t.keypoints_xy,np.asarray(saved['gt'],float))
        np.testing.assert_array_equal(t.keypoint_supervision_mask,saved['valid'])
        gt_sources.append(R.bound(R.ROOT/item.label))
    for row in R.read(R.E.C.POSE/'AXIS_REVIEW_MANIFEST.json')['frames_list']:
        gt_sources.append(R.bound(R.ROOT/row['annotation']))
    from square_data import membership
    for r in membership()[696:]:
        group,frame=r['id'].split('__',1)
        path=R.ROOT/'challenge/data/01_real/live_capture_gt'/group/(frame+'.json')
        b=R.bound(path); assert b['sha256']==r['source_annotation_sha256'];gt_sources.append(b)
    R.freeze(R.DOC/'TARGET_SOURCE_BINDINGS.json',dict(files=gt_sources,
        note='Additional original annotation bindings; fresh DEV GT decode equals the preregistered TARGETS exactly. Square source hashes equal the frozen membership.'))
    data=R.E.old('train').FeatureDataset(R.E.LINE,R.E.LINE/'cache')
    rows_by_id={data.source['records'][int(data.indices[r])]['id']:int(r) for r in data.validation_rows}
    oldrows={int(r):i for i,r in enumerate(data.validation_rows)}
    checks=0; results={}; baseline_pose=R.read(R.ROOT/'_docs/experiments/pallet_final_paper_tables_v1/R0_POSE.json')
    for seed in (1,2,3):
        dev=R.read(R.PRAW/f'evaluation/PRIOR{seed}/RAW_POINT_REPLACEMENTS.json')['frames']
        source=np.load(R.PRAW/f'validation_PRIOR{seed}.npz')
        for split in R.SPLITS:
            rows=[r for r in inputs if r['split']==split]; z=np.load(R.RAW/f'predictions/seed{seed}_{split}.npz')
            a=np.load(R.RAW/f'scores/seed{seed}_{split}.npz');p=z['points'];raw=z['raw'];cap=z['capped']
            worst_crop=0.;worst_original=0.;parity_frames=0
            for i,r in enumerate(rows):
                valid=np.asarray(r['valid'],bool); valid[8]=False
                for c in range(2):
                    for step in range(3):
                        delta=raw[i,c,step]-p[i,c,step];norm=np.linalg.norm(delta,axis=-1)
                        scale=np.minimum(1,.01*np.hypot(*r['raw_hw'])/np.maximum(norm,1e-12))
                        expected=p[i,c,step].copy();expected[valid]+=delta[valid]*scale[valid,None]
                        np.testing.assert_allclose(cap[i,c,step],expected,rtol=0,atol=1e-10,equal_nan=True)
                        np.testing.assert_array_equal(p[i,c,step+1],raw[i,c,step] if c==0 else cap[i,c,step]);checks+=1
                np.testing.assert_array_equal(raw[i,0,0],raw[i,1,0])
                if not r['usable'] or split=='SQUARE_DEV':continue
                if split=='REAL_DEV':
                    old=np.array(dev[r['image']][0]['keypoints_xy'],float)
                else:
                    row=rows_by_id[r['id']];record=data.source['records'][int(data.indices[row])]
                    gain,offset=R.E.old('features').canvas_affine(record['prepared_shape_hw'],data.arrays['input_shape'][row])
                    old=(source['points'][oldrows[row]]-offset)/gain-r['pad']
                matrix=R.axis_aligned_crop_matrix(r['box'])
                delta=np.abs(raw[i,0,0,:8]-old[:8]);current=float(delta.max());crop=float((delta*np.diag(matrix)[:2]).max())
                worst_original=max(worst_original,current);worst_crop=max(worst_crop,crop);parity_frames+=1
            # Historical stored synthetic outputs have a float32 canvas round-trip.
            # Report actual differences; 3e-4 remains the reference and is never relaxed.
            results[f'{split}_seed{seed}']=dict(frames=parity_frames,max_crop_component=worst_crop,
                max_original_component=worst_original,within_3e4_reference=worst_crop<=3e-4,
                meaning='Stored PASS1 comparison; separate from identical-runtime10repeat test. Synthetic archive has float32 canvas serialization.',
                status='NOT_APPLICABLE_NO_HISTORICAL_SQUARE_PRIOR' if split=='SQUARE_DEV' else 'MEASURED')
    real=R.read(R.RAW/'pose/seed1_REAL_DEV_RAW.json')[0]
    old={r['id']:r for r in baseline_pose['metrics']};maxdiff=0.
    for r in real:
        m=r['metric'];b=old[m['id']];assert m['available']==b['available']
        if m['available']:
            for k in ['translation_cm','rotation_deg','yaw_deg','IoU3D','ADDsym_m','ADDsym_normalized']:
                maxdiff=max(maxdiff,abs(m[k]-b[k]));assert abs(m[k]-b[k])<=1e-9,(m['id'],k,m[k],b[k])
    R.write(R.DOC/'INDEPENDENT_AUDIT.json',dict(complete=True,training_updates=0,trajectory_cap_checks=checks,
        historical_PASS1_parity=results,current_pose_R0_exact_frames=len(real),current_pose_max_abs=maxdiff,
        source_hashes_unchanged=True,analysis_only_GT=True,code=R.bound(__file__),
        note='Reference-tolerance failures, if any, are reported rather than changing tolerance or invalidating accuracy automatically.'))
    print(R.serial(dict(cap_checks=checks,pose_max_abs=maxdiff,historical_parity=results)),flush=True)

if __name__=='__main__': main()
