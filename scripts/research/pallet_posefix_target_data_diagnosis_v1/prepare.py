import subprocess
from collections import Counter
import numpy as np
import torch
from . import common as C

def main():
    for p in (C.DOC,C.RAW,C.OUT):assert not p.exists(),p
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip();assert branch=='main'
    poolpath=C.B.E.RAW/'PSEUDOLABEL_MANIFEST.json';accepted=C.read(poolpath)
    candidate_records=C.read(C.B.E.DOC/'DATA_ROLE_MANIFEST.json')['train_candidates']
    pool=[C.read(C.B.E.RAW/'pseudo_frames'/f'{r["id"]}.json') for r in candidate_records]
    assert [r for r in pool if r['accepted']]==accepted
    assert len(pool)==264 and len(accepted)==253
    e2=C.read(C.B.E.DOC/'E2_INPUT_LOCK.json');old=C.read(C.P.DOC/'INPUT_LOCK.json');audit=C.read(C.P.DOC/'TRAIN_INPUT_AUDIT.json')
    role=C.read(C.B.E.DOC/'DATA_ROLE_MANIFEST.json');assert role['eval_image_hash_overlap']==role['eval_recording_overlap']==0
    assert {r['id'] for r in accepted}=={r['id'] for r in e2['records']}=={r['id'] for r in audit['real']}
    rolemap={r['id']:r for r in role['eval_records']}
    assert all(rolemap[r['id']]==r for r in old['eval_records'])
    # Current frozen experiment populations are subsets of the older pool, not new selections.
    assert len(old['eval_records'])==278 and len(old['populations']['PRIMARY_OCC96'])==93
    assert not {r['image']['sha256'] for r in accepted}&{r['image']['sha256'] for r in old['eval_records']}
    assert not any(any(alias in r['image']['path'] for alias in role['excluded_recording_aliases']) for r in old['eval_records'])
    protected={}
    def protect(p):
        b=C.bind(p);protected[b['path']]=b;return b
    for r in candidate_records:protect(C.B.E.RAW/'pseudo_frames'/f'{r["id"]}.json')
    for p in (poolpath,C.B.E.DOC/'E2_INPUT_LOCK.json',C.P.DOC/'TRAIN_INPUT_AUDIT.json',C.P.DOC/'INPUT_LOCK.json',C.P.DOC/'PREDICTION_LOCK.json',C.B.E.DOC/'DATA_ROLE_MANIFEST.json',C.ROOT/old['symmetry']['path'],C.ROOT/'_docs/experiments/pallet_gt_corner_definition_audit_v1/AUDIT.json'):
        protect(p)
    for b in old['protected']:C.verify(b);protected[b['path']]=b
    checkpoints={'PRIOR1':C.B.protocol()['base'],'FULL125':C.read(C.B.DOC/'FIT_FULL.json')['checkpoint'],
        'FULL150':C.read(C.P.DOC/'FIT_C.json')['checkpoint'],'FULL_PRESERVE':C.read(C.P.F.DOC/'FIT_FULL_PRESERVE.json')['checkpoint']}
    for b in checkpoints.values():C.verify(b);protected[b['path']]=b
    for r in audit['real']:
        for k in ('image','pair','old'):C.verify(r[k]);protected[r[k]['path']]=r[k]
    for k in ('real_order','source_orders'):C.verify(audit[k]);protected[audit[k]['path']]=audit[k]
    for p in (C.DOC,C.RAW,C.OUT):p.mkdir(parents=True)
    C.save(C.DOC/'INPUT_BINDINGS.json',dict(HEAD=head,branch=branch,git_status=subprocess.check_output(['git','status','--short'],text=True),protected=list(protected.values()),checkpoints=checkpoints,
        role_audit=protect(C.B.E.DOC/'DATA_ROLE_MANIFEST.json'),image_overlap=0,recording_overlap=0,teacher_same_recording_disclosed=role['teacher_training_same_recording']))
    C.save(C.DOC/'PURPOSE_AND_PLAN.md','# TARGET_DATA_AND_TRAIN_FIT\n\nNEW TRAINING=0 / OPTIMIZER STEPS=0 / CHECKPOINT UPDATE=0 / GT CHANGE=0 / PSEUDO CHANGE=0. 全TRAIN253 census → TRAIN-only deterministic probe protocol → all four frozen models → prediction lock → DEV comparison → fixed-threshold routing → one next experiment design only.\n\nControlled probe: up to128 eligible corners, spread over frames by deterministic round-robin then evenly spaced selection; all253 natural CLEAN/OCC evaluated. P1 radii10/20/30/40 and four cardinal directions; P2 radii20/30/40 same corners. No DEV-driven selection. Existing fixed crop of each natural CLEAN/OCC input preserved. This bounded inference sample is not the TRAIN census.\n\nStanding user publication override applies: MD+images push, private rows/checkpoints excluded.\n')
    order=torch.load(C.ROOT/audit['real_order']['path'],weights_only=True).numpy();assert order.shape==(300,8)
    exposures=np.bincount(order.ravel(),minlength=253);mapping={r['id']:r for r in pool};rows=[];frames=[];roundtrip=0.
    for i,r in enumerate(audit['real']):
        entry=C.pair(i);new=C.pair(i,1.5);meta=entry['metadata'];p=mapping[r['id']];assert meta==e2['records'][i] and meta==new['metadata'];assert p['GT_input'] is False
        gt=np.array(meta['target_original']);np.testing.assert_array_equal(gt,C.B.E.P.top(p['refined'])['keypoints_xy'])
        x=entry['pair']['CLEAN'];y=entry['pair']['OCC'];sem=new['semantic_mask'];assert not sem[8]
        np.testing.assert_array_equal(x['original_points'],C.B.E.P.top(meta['raw_prediction'])['keypoints_xy']);np.testing.assert_array_equal(y['original_points'],C.B.E.P.top(meta['occluded_R0_prediction'])['keypoints_xy'])
        np.testing.assert_array_equal(meta['raw_prediction'],p['raw'])
        for mode in ('CLEAN','OCC'):
            a=entry['pair'][mode];b=new['pair'][mode]
            np.testing.assert_array_equal(a['original_points'],b['original_points']);np.testing.assert_array_equal(a['original_gt'],gt)
            err=float(np.max(np.abs(C.D.transform_points(a['target'],np.linalg.inv(a['matrix']))-gt)));roundtrip=max(roundtrip,err);assert err<1e-4
        valid=sem&np.isfinite(gt).all(1)&np.isfinite(x['original_points']).all(1)&np.isfinite(y['original_points']).all(1)
        clean=np.linalg.norm(x['original_points']-gt,axis=1);occ=np.linalg.norm(y['original_points']-gt,axis=1)
        aug=any(q.get('rectangles') for q in meta['plans']);st=C.strict(p);q=C.B.E.P.top(p['raw'])
        pp=p.get('pnp') or {};before=pp.get('before');movement=np.linalg.norm(np.array(before)-gt,axis=1) if before is not None else None
        fr=dict(index=i,id=p['id'],display_index=p['display_index'],session=p['session'],strict=st,occlusion_applied=aug,
            stage1=p.get('stage1'),stage2=p.get('stage2'),raw_score=q['score'],pnp=pp,plans=meta['plans'],image=p['image'],exposures=int(exposures[i]))
        frames.append(fr)
        for j in range(8):
            if not valid[j]:continue
            rows.append(dict(index=i,frame_id=p['id'],display_index=p['display_index'],corner_id=j,session=p['session'],strict=st,
                clean_error_px=float(clean[j]),occ_error_px=float(occ[j]),delta_occ_px=float(occ[j]-clean[j]),
                clean_error_norm=float(clean[j]/x['bbox_diagonal']),occ_error_norm=float(occ[j]/y['bbox_diagonal']),
                target_supported_125=bool(y['target_valid'][j]),target_supported_150=bool(new['pair']['OCC']['target_valid'][j]),
                input_valid_clean=bool(x['valid'][j]),input_valid_occ=bool(y['valid'][j]),
                occlusion_applied_to_frame=bool(aug),covered_by_plan=any(j in plan.get('covered_corners',[]) for plan in meta['plans']),
                clean_input=x['original_points'][j],occ_input=y['original_points'][j],pseudo_target=gt[j],
                replay_before_pnp_to_final_px=float(movement[j]) if movement is not None else None,
                pnp_hidden=j in pp.get('hidden',[]),raw_score=q['score'],stage1=p.get('stage1'),stage2=p.get('stage2'),
                exposures=int(exposures[i]),clean_band=C.band(clean[j]),occ_band=C.band(occ[j])))
        if (i+1)%70==0:print('CENSUS',i+1,flush=True)
    sup=[r for r in rows if r['target_supported_125']];hard=[r for r in sup if r['occ_error_px']>20];sh=[r for r in hard if r['strict']]
    bandstats={}
    for mode in ('clean','occ'):
        bandstats[mode]=[]
        for j in range(5):
            rr=[r for r in sup if r[mode+'_band']==j];n=len(rr)
            bandstats[mode].append(dict(band=j,corners=n,frames=len({r['frame_id'] for r in rr}),fraction=n/len(sup),
                corner_ids=dict(Counter(r['corner_id'] for r in rr)),span=C.runs(r['display_index'] for r in rr),
                normalized_median=float(np.median([r[mode+'_error_norm'] for r in rr])) if n else None,
                occlusion_applied_fraction=sum(r['occlusion_applied_to_frame'] for r in rr)/n if n else None,
                strict_count=sum(r['strict'] for r in rr)))
    transitions=np.zeros((5,5),int)
    for r in sup:transitions[r['clean_band'],r['occ_band']]+=1
    framecounts=Counter(r['index'] for r in hard)
    summary=dict(unique_frames=253,semantic_corners=len(rows),supervised_corners=len(sup),clean=C.summary([r['clean_error_px'] for r in sup]),occ=C.summary([r['occ_error_px'] for r in sup]),
        bands=bandstats,strict_frames=sum(f['strict'] for f in frames),strict_corners=sum(r['strict'] for r in sup),
        hard20=len(hard),hard20_40=sum(r['occ_error_px']<=40 for r in hard),hard40=sum(r['occ_error_px']>40 for r in hard),
        strict_hard20=len(sh),strict_hard20_40=sum(r['occ_error_px']<=40 for r in sh),strict_hard40=sum(r['occ_error_px']>40 for r in sh),strict_hard_frames=len({r['index'] for r in sh}),
        frames_any_hard=len(framecounts),frames_2plus_hard=sum(n>=2 for n in framecounts.values()),frames_4plus_hard=sum(n>=4 for n in framecounts.values()),
        easy_to_hard20=sum(r['clean_error_px']<=10 and r['occ_error_px']>20 for r in sup),easy_to_hard40=sum(r['clean_error_px']<=10 and r['occ_error_px']>40 for r in sup),transition=transitions)
    C.save(C.RAW/'TRAIN_CORNER_CENSUS.json',rows);C.save(C.RAW/'TRAIN_FRAMES.json',frames);C.save(C.DOC/'TRAIN_CORNER_CENSUS_SUMMARY.json',summary)
    C.save(C.DOC/'TRAIN_EXPOSURE_AUDIT.json',dict(total=sum(r['exposures'] for r in sup),hard20=sum(r['exposures'] for r in hard),hard40=sum(r['exposures'] for r in hard if r['occ_error_px']>40),strict_hard20=sum(r['exposures'] for r in sh),
        unique_hard_corners=len(hard),unique_hard_frames=len(framecounts),hard_exposures_per_unique=C.summary([r['exposures'] for r in hard]),real_order_exact=True,source_order_verified=True))
    C.save(C.DOC/'TEMPORAL_DIVERSITY_AUDIT.json',dict(recordings=1,hard=C.runs(r['display_index'] for r in hard),strict_hard=C.runs(r['display_index'] for r in sh),independent_scene_count='UNKNOWN',frame_indices_are_display_order_not_timestamps=True))
    excluded=[]
    for p in pool:
        if p['accepted']:continue
        q=C.B.E.P.top(p['raw']);excluded.append(dict(id=p['id'],display_index=p['display_index'],reason=p['reason'],raw_score=q.get('score') if q else None,valid_kp=int(C.B.E.P.valid_points(q).sum()) if q else 0,stage1=p.get('stage1'),stage2=p.get('stage2')))
    assert len(excluded)==11;C.save(C.RAW/'EXCLUDED11_ROWS.json',excluded);C.save(C.DOC/'EXCLUDED11_AUDIT.json',dict(count=11,reasons=dict(Counter(r['reason'] for r in excluded)),rows=excluded))
    # Selection uses TRAIN pseudo targets only, before any model outputs or DEV error access.
    eligible=[r for r in sup if r['strict'] and r['clean_error_px']<=10 and r['target_supported_150'] and r['input_valid_clean'] and r['input_valid_occ']]
    # Round robin over corners ensures broad frame coverage before second corners.
    byframe={i:[r for r in eligible if r['index']==i] for i in sorted({r['index'] for r in eligible})}
    rr=[xs[k] for k in range(8) for xs in byframe.values() if k<len(xs)]
    selected=[rr[int(i)] for i in np.linspace(0,len(rr)-1,min(128,len(rr)),dtype=int)] if rr else []
    C.save(C.DOC/'CONTROLLED_PROBE_PROTOCOL.json',dict(models=C.ARMS,eligible_corners=len(eligible),selected=[dict(index=r['index'],corner_id=r['corner_id']) for r in selected],
        selection='TRAIN-only round-robin, evenly spaced up to128; no output/DEV selection',P1_radii=[10,20,30,40],P2_radii=[20,30,40],angles=[0,90,180,270],
        crop='P1/P2 both fixed CLEAN bbox/crop and CLEAN other points; only RGB differs; natural OCC retains its own fixed original crop',keep_other_points=True,keep_center8=True,GT_eval_input=False,
        strict_threshold=.025,routing=dict(hard_ratio=.5,strict_hard_corners=32,strict_hard_frames=16,r30=.5,r40=.3,natural_20_40=.5,top5_gap_pp=20,strict_conflict_ratio=.5),new_training=0,optimizer_steps=0))
    C.save(C.DOC/'PREFLIGHT_AUDIT.md',f'# Preflight PASS\n\nHEAD {head}, main. 候補264/accepted253 exact; paired metadata/clean R0/OCC R0/pseudo target exact, roundtrip maximum {roundtrip:.8f}px. real order300×8/source order verified. Image/recording overlap0 under existing recording alias manifest; teacher same-recording history retained. Existing checkpoints and prior1431+ input bindings verified. No GT/pseudo mutation.\n')
    C.save(C.DOC/'PREPARATION_CHECKS.json',dict(PASS=True,roundtrip_max=roundtrip,accepted253=True,candidates264=True,real_order_shape=list(order.shape),protected=len(protected),center8_excluded=True,raw_occ_target_exact=True))
    print('PREPARED',summary['hard20'],summary['strict_hard20'],len(selected),flush=True)

if __name__=='__main__':main()
