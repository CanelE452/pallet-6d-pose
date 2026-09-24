"""Bounded common-support H10 audit, no annotation writes or pointwise matching."""
import numpy as np
from . import common as C

def main():
    targets=[r for r in C.read(C.E.RAW/'TARGETS.json') if r['role']=='H']
    pair=C.read(C.E.DOC/'TARGET_AND_PAIR_AUDIT.json')
    assert len(targets)==10 and {r['id'] for r in targets}=={p['H'] for p in pair['pairs']}
    candidates=C.read(C.CONTRACT/'C4_CANDIDATES.json')['candidates']
    assert len(candidates)==4
    # Validate proper 3D-generated, whole-object permutations afresh.
    from scripts.research.pallet_011067_corner_contract_v1.common import xyz
    base=xyz(*candidates[0]['dimensions_xyz'])
    for cc in candidates:
        rot=np.array(cc['rotation']); perm=cc['perm_new_to_stored']
        assert sorted(perm)==list(range(9)) and perm[8]==8 and np.isclose(np.linalg.det(rot),1)
        assert np.allclose(xyz(*cc['dimensions_xyz']), (base@rot.T)[perm])
    diag=C.read(C.E.RAW/'DIAGNOSTICS.json')
    caches={a:{r['id']:r['prediction'] for r in diag[key]['train']} for a,key in [('T1','T1_HARD_PSEUDO'),('T2','T2_HARD_MANUAL')]}
    rows=[]
    for r in targets:
        obj=C.read(C.ROOT/r['annotation']['path'])['objects'][0]
        mask=np.array(r['mask'],bool);mask[8]=False; gt=np.array(r['manual'])
        masks=mask & np.isfinite(gt).all(1) & ~(gt==-1).all(1)
        models={}
        for a,q in [('TEACHER',np.array(r['target'])),*((a,C.E.D.points(cache[r['id']])) for a,cache in caches.items())]:
            scores=[]
            if q is not None:
                for cc in candidates:
                    err=np.linalg.norm(q[cc['perm_new_to_stored']]-gt,axis=1)[masks]
                    scores.append(dict(c4=cc['c4'],mean_px=float(err.mean()) if len(err) and np.isfinite(err).all() else None))
            available=[s for s in scores if s['mean_px'] is not None]
            ordered=sorted(available,key=lambda x:(x['mean_px'],x['c4']))
            same=next((s['mean_px'] for s in scores if s['c4']=='YAW_0'),None)
            best=ordered[0] if ordered else None
            models[a]=dict(same_ID_mean_px=same,best_C4=best,second_best_margin_px=ordered[1]['mean_px']-ordered[0]['mean_px'] if len(ordered)>1 else None,
                all_whole_C4=scores,strong=bool(masks.sum()>=3 and same is not None and best and same>20 and best['mean_px']<=10))
        rows.append(dict(id=r['id'],severity=r['severity'],common_support_count=int(masks.sum()),common_corners=np.flatnonzero(masks).tolist(),
                         direct_count=sum(k.get('source') in ('manual','manual_click') for k in obj.get('keypoint_annotations',[])[:8]),
                         metadata=dict(keypoint_frame=obj.get('keypoint_frame'),
                                       camera_dynamic_0123_v4=obj.get('keypoint_frame')=='camera_dynamic_0123_v4',
                                       axis_assignment_confirmed=obj.get('camera_facing_pnp',{}).get('axis_assignment_confirmed'),
                                       axis_assignment=obj.get('camera_facing_pnp',{}).get('axis_assignment'),
                                       migration_status=obj.get('migration_status'),
                                       manual_review_required=obj.get('migration_status')=='MANUAL_REVIEW_REQUIRED'),models=models))
    strong=[r['id'] for r in rows if any(m['strong'] for m in r['models'].values())]
    insufficient=[r['id'] for r in rows if r['common_support_count']<3]
    status='ROLE_MISMATCH_REPEATED' if any(i!='plastic_day_01:011067' for i in strong) else 'ROLE_MISMATCH_SCAN_INCONCLUSIVE' if insufficient or not strong else 'ROLE_MISMATCH_ISOLATED'
    result=dict(status=status,frames=10,strong_frames=strong,strong_count=len(strong),insufficient=insufficient,
        rule='n>=3 AND same-ID mean>20px AND best whole-C4 mean<=10px; no margin threshold',rows=rows,
        no_free_matching=True,whole_C4_diagnostic_only=True,annotation_writes=0,GT_modified=False,
        caveat='90/270 role-coordinate W/D swap is not physical rectangular symmetry. Original physical yaw180 equivalence and fixed-ID evaluation unchanged.')
    C.save(C.DOC/'H10_ROLE_PREVALENCE.json',result)
    lines=['# H10 역할 규약 빈도 진단','',status,'', '고정 common/direct support만 사용. 전체 C4는 진단용이며 GT와 실제 평가를 바꾸지 않음.','',
        '|프레임|점|모델|같은 ID 평균 px|best C4 평균 px|margin px|strong|','|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        for a,m in r['models'].items():
            lines.append(f"|{r['id']}|{r['common_support_count']}|{a}|{m['same_ID_mean_px']}|{m['best_C4']['mean_px'] if m['best_C4'] else None}|{m['second_best_margin_px']}|{m['strong']}|")
    lines+=['','|프레임|난도|직접점 / 공통점|keypoint frame|axis 확정|manual review|','|---|---|---:|---|---|---|']
    for r in rows:
        m=r['metadata'];lines.append(f"|{r['id']}|{r['severity']}|{r['direct_count']} / {r['common_support_count']}|{m['keypoint_frame']}|{m['axis_assignment_confirmed']}|{m['manual_review_required']}|")
    C.save(C.DOC/'H10_ROLE_PREVALENCE_KO.md','\n'.join(lines)+'\n')
    print('H10',status,strong,flush=True)

if __name__=='__main__': main()
