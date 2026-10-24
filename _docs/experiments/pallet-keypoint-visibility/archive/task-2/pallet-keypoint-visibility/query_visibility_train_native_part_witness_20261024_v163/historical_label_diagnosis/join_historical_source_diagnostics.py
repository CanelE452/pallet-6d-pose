#!/usr/bin/env python3
"""V163 historical source-only descriptive join after the sealed all390 review barrier."""
import copy, csv, hashlib, io, json, math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
V=Path(__file__).resolve().parents[1]; O=V/'historical_label_diagnosis'
FIELDS=['selected_instance_correspondence','native_part_visibility','query_relation','part_cover_readiness']
ON_NEAR={'ON_INTENDED_PART_SUPPORTED','NEAR_INTENDED_PART_SAME_SELECTED_PALLET'}
OWNER={'TARGET_VISIBLE_SUPPORTED':'T','FOREIGN_VISIBLE_SUPPORTED':'F','BACKGROUND_SUPPORTED':'BG','UNKNOWN':'U'}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {'path':str(p),'sha256':sha(p)}
def read(p): return json.loads(Path(p).read_text())
def jsonl(p): return [json.loads(x) for x in Path(p).read_text().splitlines()]
def write(p,x):
    assert not p.exists(),str(p)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def counter(xs): return dict(sorted(Counter(xs).items()))
def obs(r,view): return r['primary_observation'] if view=='primary' else r['effective_observation']
def owner_bucket(r,view):
    z=obs(r,view);q=z['query_relation'];part=z['native_part_visibility'];inst=z['selected_instance_correspondence']
    if inst=='CLEAR' and part=='VISIBLE_SUPPORTED' and q in ON_NEAR: return 'VISIBLE_PART_AND_Q_ON_OR_NEAR'
    if q=='OTHER_SELECTED_PALLET_PART': return 'OTHER_SELECTED_PALLET_PART'
    if q=='BACKGROUND_OR_GAP': return 'BACKGROUND_OR_GAP'
    if q=='FOREIGN_OBJECT': return 'FOREIGN_OBJECT'
    if q in ON_NEAR: return 'Q_ON_OR_NEAR_WITHOUT_CLEAR_VISIBLE_PART'
    return 'UNKNOWN_OR_DISAGREEMENT'
def dist_stats(rs):
    xs=sorted(r['nominal_reference_diagnostic']['q_to_nominal_reference_distance_px'] for r in rs)
    ns=sorted(r['nominal_reference_diagnostic']['distance_over_original_L'] for r in rs)
    def med(z):
        n=len(z);return None if not n else (z[n//2] if n%2 else (z[n//2-1]+z[n//2])/2)
    return dict(n=len(rs),median_nominal_distance_px=med(xs),median_nominal_distance_over_L=med(ns),not_physical_coordinate_error=True)
def summarize(rs,view):
    roles={};owners={}
    for role in ['D','E']:
        rr=[r for r in rs if r['original_role_state']==role]
        roles[role]=dict(N=len(rr),observations={f:counter(obs(r,view)[f] for r in rr) for f in FIELDS},
            joint_visibility_query=counter(obs(r,view)['native_part_visibility']+' | '+obs(r,view)['query_relation'] for r in rr),
            nominal_reference=dist_stats(rr))
    for name in ['T','F','BG','U']:
        rr=[r for r in rs if r['historical_owner_short']==name]
        owners[name]=dict(N=len(rr),observations={f:counter(obs(r,view)[f] for r in rr) for f in FIELDS},
            relation_decomposition=counter(owner_bucket(r,view) for r in rr),
            joint_visibility_query=counter(obs(r,view)['native_part_visibility']+' | '+obs(r,view)['query_relation'] for r in rr),
            nominal_reference=dist_stats(rr))
    return dict(N=len(rs),frames=len({r['frame_index'] for r in rs}),old_roles=counter(r['original_role_state'] for r in rs),
        old_owners=counter(r['historical_owner_short'] for r in rs),
        observations={f:counter(obs(r,view)[f] for r in rs) for f in FIELDS},
        old_role_x_new_observation=roles,old_owner_x_new_observation=owners,
        historical_TARGET53=owners['T'])
def main():
    O.mkdir(exist_ok=True)
    scope=V/'ROOT_SCOPE_BEFORE_NATIVE_PART_WITNESS_PIXELS.json'
    rules=V/'ROOT_OBSERVATION_JOIN_RULES_BEFORE_VOTES.json'
    barrier=V/'join_observations/ALL390_REVIEW_BARRIER_AND_SUMMARY.json'
    assert sha(scope)=='875f28ae4633d83f80b598a6bc0812ae312131ea22d740d7e15f52b88f44481a'
    assert sha(rules)=='5cf4fda89c169e9adb42ca93a5a54ed849d6143bded073628ec61f4c576a81a2'
    assert sha(barrier)=='986b53690ba6c6ccbd91a6dc8cbb04ab914558fa6a1e5f45826372c1f1f75945'
    b=read(barrier); s=read(scope)
    assert b['completed'] and b['old_label_comparison_barrier_satisfied'] and b['count']==390
    assert b['single_review_weak_count']==350 and b['cross40_count']==40 and b['all_new_native_quality_targets']==-1
    wpath=Path(b['all390']['path']);mpath=Path(s['source_mapping']['path'])
    assert sha(wpath)==b['all390']['sha256'] and sha(mpath)==s['source_mapping']['sha256']
    bindings=dict(scope=bind(scope),join_rules=bind(rules),sealed_all390_barrier=bind(barrier),witnesses=bind(wpath),historical_mapping=bind(mpath),code=bind(Path(__file__)))
    write(O/'RULES_AND_BINDINGS_BEFORE_DIAGNOSTIC_COMPUTATION.json',dict(completed=True,bindings=bindings,
        identity='Exact query_id AND global_index join; assert unchanged source/frame/split/native/name/q/L/selected instance. All390 retained in sealed witness order.',
        views={'primary390':'One assigned primary weak observation per original row; cross40 primary is not another vote.',
               'mixed390':'Single-review350 plus cross40 per-field four-way unanimity. Null disagreement remains DISAGREEMENT; no majority.',
               'cross40':'Report primary40, four-way40 and each reviewer40 separately, not as independent physical samples.'},
        categories='Old owner TARGET53 decomposition is mutually exclusive: clear visible intended part AND qON/NEAR; otherpart; background/gap; foreign; on/near withoutclear visiblepart; UNKNOWN/disagreement.',
        nominal='Euclidean q-to-provided nominal reference and division by originalL only. No threshold/inclusion/errorgold/3px certification.',
        new_targets=-1,training_admitted=False,no_pixels_masks_features_models_read=True))
    wr=jsonl(wpath);mr=jsonl(mpath)
    assert len(wr)==len(mr)==390
    by={r['query_id']:r for r in mr};gi={r['global_index']:r for r in mr}
    assert len(by)==len(gi)==390 and {r['query_id'] for r in wr}==set(by)
    identity=['global_index','query_id','frame_index','source_id','split','native_slot','native_name','query_xy','L','selected_gt_instance_index']
    rows=[];cross=[]
    for w in wr:
        m=by[w['query_id']]
        assert gi[w['global_index']] is m
        assert all(w[k]==m[k] for k in identity),(w['case_id'],'identity')
        assert w['split']=='SOURCE_DEVELOPMENT' and m['point_quality_target']==-1 and m['training_admitted'] is False
        assert w['all_new_native_visibility_target']==w['all_new_point_quality_target']==-1
        assert w['training_admitted'] is False and w['original_labels_modified'] is False
        if w['four_way_consensus'] is not None:
            assert w['review_support']=='FOUR_WAY_CROSS'; cross.append(w)
            for f in FIELDS:
                c=w['four_way_consensus'][f];vs=c['votes']
                assert set(vs)=={'A','B','C','ROOT'}
                unanimous=len(set(vs.values()))==1
                assert c['unanimous']==unanimous
                assert c['value']==(next(iter(vs.values())) if unanimous else None)
                assert w['effective_observation'][f]==(c['value'] if unanimous else None)
                assert vs[w['primary_observation']['reviewer']]==w['primary_observation'][f]
        else:
            assert w['review_support']=='SINGLE_REVIEWER_WEAK'
            assert all(w['effective_observation'][f]==w['primary_observation'][f] for f in FIELDS)
        r=copy.deepcopy(w)
        # Preserve all sealed witness fields. The original null effective value stays null in the row.
        r.update(original_role_state=m['original_role_state'],original_role_target=m['original_role_target'],
            weak_visible_owner_support=m['weak_visible_owner_support'],historical_owner_short=OWNER[m['weak_visible_owner_support']],
            original_point_quality_target=m['point_quality_target'],historical_source_mapping_row_binding=dict(path=str(mpath),file_sha256=sha(mpath),query_id=m['query_id'],global_index=m['global_index']),
            historical_reference_uncertainty=m['reference_point_uncertainty'],
            original_supervision_selected_instance_reason=m['original_supervision_selected_instance_reason'])
        d=math.hypot(w['query_xy'][0]-m['reference_xy'][0],w['query_xy'][1]-m['reference_xy'][1])
        r['nominal_reference_diagnostic']=dict(reference_xy=m['reference_xy'],q_to_nominal_reference_distance_px=d,distance_over_original_L=d/w['L'],
            actual_physical_pixel_error_certified=False,reference_is_nominal_not_pixel_gold=True)
        rows.append(r)
    assert len(cross)==40
    assert counter(r['original_role_state'] for r in rows)=={'D':341,'E':49}
    assert counter(r['historical_owner_short'] for r in rows)=={'BG':148,'F':52,'T':53,'U':137}
    assert len({r['frame_index'] for r in rows})==88
    assert {r['native_slot'] for r in rows if r['original_role_state']=='E'}=={4,5}
    assert counter(str(r['native_slot']) for r in rows if r['original_role_state']=='E')=={'4':11,'5':38}
    # Analysis copies only: null unanimity becomes explicit DISAGREEMENT in tables; preserved joined rows retain nulls.
    ar=copy.deepcopy(rows)
    for r in ar:
        r['effective_observation']={f:('DISAGREEMENT' if r['effective_observation'][f] is None else r['effective_observation'][f]) for f in FIELDS}
    single=[r for r in ar if r['four_way_consensus'] is None];cr=[r for r in ar if r['four_way_consensus'] is not None]
    cohorts={'primary390':(ar,'primary'),'mixed350_single_plus_cross40_unanimity':(ar,'mixed'),
        'single350':(single,'primary'),'cross40_primary':(cr,'primary'),'cross40_four_way_unanimity':(cr,'mixed')}
    summaries={name:summarize(rs,view) for name,(rs,view) in cohorts.items()}
    scenes=sorted({r['source_id'].split(':')[0] for r in ar})
    assert scenes==['val/000000','val/000002','val/000006']
    per_native={name:{str(n):summarize([r for r in rs if r['native_slot']==n],view) for n in range(8)} for name,(rs,view) in cohorts.items()}
    per_scene={name:{scene:summarize([r for r in rs if r['source_id'].split(':')[0]==scene],view) for scene in scenes} for name,(rs,view) in cohorts.items()}
    cross_reviewers={}
    for reviewer in ['A','B','C','ROOT']:
        rr=copy.deepcopy(cr)
        for r in rr:
            r['primary_observation'].update({f:r['four_way_consensus'][f]['votes'][reviewer] for f in FIELDS})
        cross_reviewers[reviewer]=summarize(rr,'primary')
    cross_agreement={f:dict(N=40,unanimous=sum(r['four_way_consensus'][f]['unanimous'] for r in cr),
        disagreement=sum(not r['four_way_consensus'][f]['unanimous'] for r in cr)) for f in FIELDS}
    # Independently cross-check frozen barrier arithmetic, without changing its criteria.
    for f in FIELDS:
        assert summaries['primary390']['observations'][f]==dict(sorted(b['all390_summary']['primary_observations'][f].items()))
        assert summaries['mixed350_single_plus_cross40_unanimity']['observations'][f]==dict(sorted(b['all390_summary']['effective_observations_with_disagreement_null'][f].items()))
    checks=dict(all390_identity_and_qL_native_selected_instance_exact=True,all390_frames88_train_only=True,
        old_D341_E49_preserved=True,old_owner_T53_F52_BG148_U137_preserved=True,
        original_primary_and_all_four_votes_preserved=True,all_cross40_unanimity_recomputed_no_majority=True,
        original_null_disagreements_retained_in_joined_rows=True,all_native_visibility_and_quality_targets_remain_minus1=True,
        no_training_admission_or_original_label_change=True,source_barrier_hash_verified=True,
        barrier_observation_arithmetic_independently_equal=True,all_native8_and_three_scene_denominators_including_zero=True,
        no_source_pixel_mask_feature_weight_or_model_body_read=True)
    summary=dict(schema='V163_HISTORICAL_SOURCE_LABEL_DIAGNOSIS_V1',completed=True,N=390,frames=88,
        all_original_D_E={'D':341,'E':49},all_original_owner={'T':53,'F':52,'BG':148,'U':137},
        cohorts=summaries,per_native8=per_native,per_scene3=per_scene,cross40_reviewers_separate=cross_reviewers,cross40_agreement=cross_agreement,
        old_E_native_support={'4':11,'5':38,'all_other_slots':0},
        limitations=['Original roles and center-owner observations are historical assistant weak source labels, not truth.',
                    'New intended-part and query-relation witnesses are contextualized assistant weak observations, not precision/recall, coordinate gold or native D/E targets.',
                    'Mixed390 combines350 single-review rows and40 four-way per-field unanimity rows; this is heterogeneous evidence strength.',
                    'Disagreement and UNKNOWN remain unresolved. No majority or nominal geometry resolves them.',
                    'All390 are Euro WOOD in three public TRAIN scenes; real SQUARE/non-Euro-WOOD are absent and native6 has only2 queries.',
                    'Nominal reference displacement is provided-pose/CAD geometry diagnostic, not actual3px error; no certified hard pixel uncertainty bound.'],
        all_new_native_visibility_target=-1,all_new_point_quality_target=-1,training_admitted=False,original_labels_modified=False,checks=checks,bindings=bindings)
    joined=O/'ALL390_WITH_HISTORICAL_SOURCE_DIAGNOSTICS.jsonl'
    assert not joined.exists()
    joined.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    write(O/'SUMMARY.json',summary)
    table=[]
    for cohort,(rs,view) in cohorts.items():
        for dimension,groups in [('all',{'ALL':rs}),('native8',{str(n):[r for r in rs if r['native_slot']==n] for n in range(8)}),
                                 ('scene3',{s:[r for r in rs if r['source_id'].split(':')[0]==s] for s in scenes})]:
            for value,gr in groups.items():
                for role in ['D','E']:
                    rr=[r for r in gr if r['original_role_state']==role]
                    for field in FIELDS:
                        vals=counter(obs(r,view)[field] for r in rr)
                        if not vals: table.append([cohort,dimension,value,role,field,'NO_ROWS',0,len(rr),len(gr)])
                        for observed,n in vals.items():table.append([cohort,dimension,value,role,field,observed,n,len(rr),len(gr)])
    csvp=O/'TABLES.csv';assert not csvp.exists()
    with csvp.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['cohort','dimension','dimension_value','original_role_state','observation_field','observation_value','count','original_role_denominator','group_denominator']);w.writerows(table)
    targets=O/'HISTORICAL_TARGET53_DECOMPOSITION.csv';assert not targets.exists()
    with targets.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['cohort','category','count','historical_TARGET_denominator'])
        categories=['VISIBLE_PART_AND_Q_ON_OR_NEAR','OTHER_SELECTED_PALLET_PART','BACKGROUND_OR_GAP','FOREIGN_OBJECT','Q_ON_OR_NEAR_WITHOUT_CLEAR_VISIBLE_PART','UNKNOWN_OR_DISAGREEMENT']
        for cohort,sm in summaries.items():
            z=sm['historical_TARGET53']
            for c in categories:w.writerow([cohort,c,z['relation_decomposition'].get(c,0),z['N']])
    complete=dict(completed=True,PASS=all(checks.values()),UTC=datetime.now(timezone.utc).isoformat(),checks=checks,bindings=bindings,
        outputs={k:bind(p) for k,p in [('all390_joined',joined),('summary',O/'SUMMARY.json'),('tables',csvp),('historical_TARGET53_decomposition',targets),('rules',O/'RULES_AND_BINDINGS_BEFORE_DIAGNOSTIC_COMPUTATION.json')]},
        source_only=True,all_new_targets_minus1=True,training_admitted=False,original_labels_modified=False,
        no_precision_recall_or_gold_claim=True)
    write(O/'COMPLETE.json',complete)
    print(json.dumps(dict(PASS=True,complete=bind(O/'COMPLETE.json'),summary=bind(O/'SUMMARY.json'),joined=bind(joined),
        primary_TARGET53=summaries['primary390']['historical_TARGET53']['relation_decomposition'],
        mixed_TARGET53=summaries['mixed350_single_plus_cross40_unanimity']['historical_TARGET53']['relation_decomposition']),ensure_ascii=False))
if __name__=='__main__':main()

