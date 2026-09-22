"""Post-freeze descriptive analysis; never feeds DEV reference into probes."""
from collections import Counter, defaultdict
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, wasserstein_distance
from . import common as C
from scripts.research.pallet_posefix_crop_completion_v2.evaluate import peak_detail

def save_verified(path,obj):
    # Resume aggregation only: prior completed tables must be identical, never overwritten.
    if path.exists():
        import json
        def scalar(x):return x.tolist() if isinstance(x,np.ndarray) else x.item()
        assert C.read(path)==json.loads(json.dumps(obj,default=scalar,allow_nan=False)),path
    else:C.save(path,obj)

def rate(vals):return float(np.mean(vals)) if len(vals) else None
def agg(rows):
    return dict(**C.summary([r['output_error'] for r in rows]),
        reachable10=rate([r['support_distance']<=10 for r in rows]),
        argmax10=rate([r['argmax_error']<=10 for r in rows]),top5_present10=rate([r['top5_error']<=10 for r in rows]),
        mass_median=float(np.median([r['probability_mass'] for r in rows])) if rows else None,
        entropy_median=float(np.median([r['normalized_entropy'] for r in rows])) if rows else None,
        peak_ratio_median=float(np.median([r['peak_ratio'] for r in rows if r['peak_ratio'] is not None])) if rows else None)

def morphology(frame,points,target,ids,diagonal):
    p=np.array(points,float);q=np.array(target,float);v=q-p;e=np.linalg.norm(v,axis=1);mean=v.mean(0);hard=e>20;unit=v/np.maximum(e[:,None],1e-12)
    z=(p-p.mean(0))@np.array([1.,1j]);w=(q-q.mean(0))@np.array([1.,1j]);den=float(np.vdot(z,z).real)
    sim=None if len(p)<3 or den<1e-8 else float(np.sqrt(np.mean(np.abs(w-z*(np.vdot(z,w)/den))**2)))
    hid=[int(j) for j,h in zip(ids,hard) if h];pairs=[(0,1),(4,5),(0,4),(1,5),(2,6),(3,7)]
    return dict(id=frame,n=len(e),hard20=int(hard.sum()),hard40=int((e>40).sum()),ids=list(map(int,ids)),errors=e.tolist(),vectors=v.tolist(),
        direction_concentration=float(np.linalg.norm(unit.mean(0))),hard_direction_concentration=float(np.linalg.norm(unit[hard].mean(0))) if hard.any() else None,
        mean_translation=mean.tolist(),translation_magnitude=float(np.linalg.norm(mean)),translation_removed_RMSE=float(np.sqrt(np.mean(np.sum((v-mean)**2,axis=1)))),
        similarity_residual=sim,hard_ids=hid,paired_hard=[f'{a}-{b}' for a,b in pairs if a in hid and b in hid],
        normalized_errors=(e/diagonal).tolist(),isolated_hard=bool(hard.sum()==1),
        coherent_shift_proxy=bool(hard.sum()>=2 and np.linalg.norm(unit[hard].mean(0))>=.8))

def morph_summary(rows):
    hard=[r for r in rows if r['hard20']]
    return dict(frames=len(rows),hard_frames=len(hard),hard_count_histogram=dict(Counter(r['hard20'] for r in rows)),
        hard_corner_ids=dict(Counter(j for r in hard for j in r['hard_ids'])),paired_hard=dict(Counter(p for r in hard for p in r['paired_hard'])),
        isolated_hard_fraction=rate([r['isolated_hard'] for r in hard]),coherent_proxy_fraction=rate([r['coherent_shift_proxy'] for r in hard]),
        direction_concentration_median=float(np.median([r['hard_direction_concentration'] for r in hard])) if hard else None,
        translation_removed_RMSE_median=float(np.median([r['translation_removed_RMSE'] for r in hard])) if hard else None,
        similarity_residual_median=float(np.median([r['similarity_residual'] for r in hard if r['similarity_residual'] is not None])) if hard else None)

def main():
    frozen=C.read(C.DOC/'PREDICTIONS_FROZEN.json');assert frozen['all_before_DEV_analysis'];C.verify(frozen['protocol'])
    code=C.bind(C.HERE/'analyze.py')
    save_verified(C.DOC/f'ANALYSIS_START_{code["sha256"][:12]}.json',dict(predictions=C.bind(C.DOC/'PREDICTIONS_FROZEN.json'),code=code))
    predictions={}
    for a,b in frozen['models'].items():C.verify(b);predictions[a]=C.read(C.ROOT/b['path'])
    train=C.read(C.RAW/'TRAIN_CORNER_CENSUS.json');frames=C.read(C.RAW/'TRAIN_FRAMES.json');tc={(r['index'],r['corner_id']):r for r in train};sup=[r for r in train if r['target_supported_125']]
    curves={};fits={};allfit={}
    for arm,rr in predictions.items():
        curves[arm]={};fits[arm]={};allfit[arm]={}
        for mode in ('P1','P2'):
            curves[arm][mode]={str(rad):agg([r for r in rr if r['mode']==mode and r['radius']==rad]) for rad in ([10,20,30,40] if mode=='P1' else [20,30,40])}
        for mode in ('NATURAL_CLEAN','NATURAL_OCC'):
            allfit[arm][mode]=agg([r for r in rr if r['mode']==mode])
        nat=[r for r in rr if r['mode']=='NATURAL_OCC']
        for tier in ('ALL_ACCEPTED','STRICT_HALF_THRESHOLD'):
            fits[arm][tier]={}
            for label,lo,hi in [('20_40',20,40),('gt40',40,float('inf')),('gt20',20,float('inf'))]:
                chosen=[r for r in nat if lo<tc[(r['index'],r['corner_id'])]['occ_error_px']<=hi and (tier=='ALL_ACCEPTED' or tc[(r['index'],r['corner_id'])]['strict'])]
                fits[arm][tier][label]=agg(chosen)
    save_verified(C.DOC/'CONTROLLED_PROBE_RESULTS.json',dict(curves=curves,all_train_fit=allfit,
        P1_P2_only_RGB_diff=True,P2_P3_other_points_and_bbox_can_differ=True,not_independent_generalization=True,
        outside_input_counts={a:sum(r['input_out_of_image'] for r in rr if r['mode'] in ('P1','P2')) for a,rr in predictions.items()}))
    save_verified(C.DOC/'NATURAL_HARD_TRAIN_FIT.json',fits)
    # Only now load existing evaluation references and frozen official metrics.
    lock=C.read(C.P.DOC/'INPUT_LOCK.json');metric=C.read(C.P.RAW/'FRAME_METRICS.json');primary=lock['populations']['PRIMARY_OCC96'];records={r['id']:r for r in lock['eval_records']}
    refs=C.read(C.B.E.V.RAW/'E1_FRAME_METRICS.json')['truth_for_display_only'];raw=C.read(C.B.E.V.RAW/'FROZEN_PREDICTIONS.json')['predictions']['R0']
    syms={r['object_type']:r['permutations'] for r in C.read(C.ROOT/lock['symmetry']['path'])['objects']}
    fullcache=C.read(C.P.DOC/'PREDICTION_LOCK.json')['arms']['A']['heatmaps'];dev=[];morph_dev=[];prov=Counter()
    for fid in primary:
        m=metric['R0'][fid]
        if not m['matched']:continue
        gt=np.array(refs[fid]['gt']);valid=np.array(m['canonical_valid']);jjs=np.flatnonzero(valid);assert max(jjs)<8
        pred=C.B.E.P.top(raw[fid]);box=np.array(pred['box_xyxy']);diagonal=np.linalg.norm(box[2:]-box[:2]);perm=syms[records[fid]['object_type']][m['branch']]
        points=np.empty((9,2));points[perm]=pred['keypoints_xy'];errors=np.linalg.norm(points-gt,axis=1);np.testing.assert_allclose(errors[jjs],np.array(m['canonical_errors'],float)[jjs],atol=1e-5,rtol=0)
        morph_dev.append(morphology(fid,points[jjs],gt[jjs],jjs,diagonal))
        obj=C.read(C.ROOT/records[fid]['annotation']['path'])['objects'][0];annotations=obj.get('keypoint_annotations',[])
        C.verify(fullcache[fid]);cache=np.load(C.ROOT/fullcache[fid]['path']);fperm=syms[records[fid]['object_type']][metric['A'][fid]['branch']]
        for j in jjs:
            ann=annotations[j] if j<len(annotations) else {};source=ann.get('source','unknown')
            row=dict(frame_id=fid,corner_id=int(j),error=float(errors[j]),error_norm=float(errors[j]/diagonal),source=source,
                visibility=ann.get('visibility','UNKNOWN'),physical_identity='UNREVIEWED',reference_kind='legacy evaluation reference',input_xy=points[j].tolist(),target=gt[j].tolist())
            if errors[j]>20:
                ch=fperm.index(int(j));d=peak_detail(cache,ch,gt[j]);row.update(output_error=float(metric['A'][fid]['canonical_errors'][j]),
                    output_xy=C.D.transform_points(cache['expectation'][ch][None],np.linalg.inv(cache['matrix']))[0].tolist(),
                    argmax_error=d['argmax_error'],top5_error=d['top5_nearest_error'],support_distance=d['nearest_output_error'],probability_mass=d['GT_radius10_mass'],normalized_entropy=d['normalized_entropy'],peak_ratio=d['peak1_peak2_ratio'])
                prov[source]+=1
            dev.append(row)
    assert len(dev)==659 and len(primary)==93
    dist={}
    for name,rr,key,norm in [('TRAIN_ALL',sup,'occ_error_px','occ_error_norm'),('TRAIN_STRICT',[r for r in sup if r['strict']],'occ_error_px','occ_error_norm'),('DEV_MATCHED',dev,'error','error_norm')]:
        dist[name]=dict(**C.summary([r[key] for r in rr]),normalized=C.summary([r[norm] for r in rr]),
            fraction_gt20=rate([r[key]>20 for r in rr]),per_corner={str(j):C.summary([r[key] for r in rr if r['corner_id']==j]) for j in range(8)})
    pa=np.array(dist['TRAIN_ALL']['bands'],float);pb=np.array(dist['DEV_MATCHED']['bands'],float)
    dist['comparisons']=dict(hard_rate_ratio=dist['TRAIN_ALL']['fraction_gt20']/dist['DEV_MATCHED']['fraction_gt20'],
        hard_rate_delta_pp=100*(dist['TRAIN_ALL']['fraction_gt20']-dist['DEV_MATCHED']['fraction_gt20']),
        Jensen_Shannon_distance_base2=float(jensenshannon(pa/pa.sum(),pb/pb.sum(),base=2)),
        Wasserstein_px=float(wasserstein_distance([r['occ_error_px'] for r in sup],[r['error'] for r in dev])),provenance_not_equivalent=True)
    C.save(C.RAW/'DEV_CORNER_ROWS.json',dev);C.save(C.DOC/'TRAIN_DEV_DISTRIBUTION.json',dist)
    trhard=[r for r in predictions['FULL125'] if r['mode']=='NATURAL_OCC' and tc[(r['index'],r['corner_id'])]['strict'] and tc[(r['index'],r['corner_id'])]['occ_error_px']>20]
    dvhard=[r for r in dev if r['error']>20];heat=dict(TRAIN_STRICT_HARD=agg(trhard),DEV_MATCHED_HARD=agg(dvhard),
        DEV_reachable=agg([r for r in dvhard if r['support_distance']<=10]),mapping='TRAIN fixed native pseudo channels; DEV FULL official whole-object symmetry branch, no pointwise remap',target_quality_different=True)
    heat['gap_pp']=100*(heat['TRAIN_STRICT_HARD']['top5_present10']-heat['DEV_MATCHED_HARD']['top5_present10']) if trhard and dvhard else None
    C.save(C.DOC/'TRAIN_DEV_HEATMAP_COMPARE.json',heat)
    morph_train=[]
    for f in frames:
        rr=[r for r in sup if r['index']==f['index']];morph_train.append(morphology(f['id'],[r['occ_input'] for r in rr],[r['pseudo_target'] for r in rr],[r['corner_id'] for r in rr],C.pair(f['index'])['pair']['OCC']['bbox_diagonal']))
    C.save(C.RAW/'ERROR_MORPHOLOGY_ROWS.json',dict(TRAIN=morph_train,DEV=morph_dev));C.save(C.DOC/'ERROR_MORPHOLOGY_AUDIT.json',dict(TRAIN=morph_summary(morph_train),DEV=morph_summary(morph_dev),coherent_threshold=.8,not_new_inference_rule=True))
    easy=[r for r in sup if r['occ_error_px']<=10];hard=[r for r in sup if r['occ_error_px']>20];correlations=[]
    full={(r['index'],r['corner_id']):r for r in predictions['FULL125'] if r['mode']=='NATURAL_OCC'}
    for field,where in [('remove1',('stage1','s_remove')),('flip',('stage1','s_flip')),('remove2',('stage2','s_remove'))]:
        for target in ('clean_error','occ_error','teacher_movement','FULL_error'):
            xs=[];ys=[]
            for f in frames:
                rr=[r for r in sup if r['index']==f['index']];v=(f.get(where[0]) or {}).get(where[1])
                if v is None or not rr:continue
                y=np.mean([full[(r['index'],r['corner_id'])]['output_error'] if target=='FULL_error' else r['occ_error_px' if target=='occ_error' else 'clean_error_px'] for r in rr]);xs.append(v);ys.append(y)
            rho=float(spearmanr(xs,ys).statistic) if np.ptp(xs)>0 and np.ptp(ys)>0 else None
            correlations.append(dict(score=field,target=target,n=len(xs),rho=rho))
    consistency=dict(all_frames=len(frames),strict_frames=sum(f['strict'] for f in frames),strict_fraction_easy=rate([r['strict'] for r in easy]),strict_fraction_hard=rate([r['strict'] for r in hard]),
        scores_available=True,physical_correctness_guaranteed=False,spearman=correlations,teacher_movement=C.summary([r['clean_error_px'] for r in sup]),
        replay_before_pnp_available_corners=sum(r['replay_before_pnp_to_final_px'] is not None for r in sup),accepted_selection_bias=True,
        excluded_reasons=C.read(C.DOC/'EXCLUDED11_AUDIT.json')['reasons'])
    C.save(C.DOC/'PSEUDO_CONSISTENCY_AUDIT.json',consistency)
    C.save(C.DOC/'EVAL_PROVENANCE_AUDIT.json',dict(primary_hard=len(dvhard),hard_sources=dict(prov),higher_confidence_manual_source=prov['manual_click'],unknown_provenance=prov['unknown'],independently_confirmed_physical_identity=0,
        cannot_identify_causal_bottleneck_from_unknown_alone=True,GREEN='separate manual-only population preserved; not mixed into PRIMARY',official_metrics_changed=False))
    s=C.read(C.DOC/'TRAIN_CORNER_CENSUS_SUMMARY.json');p=C.read(C.DOC/'CONTROLLED_PROBE_PROTOCOL.json');fullcurve=curves['FULL125']['P1'];natural=fits['FULL125']['STRICT_HALF_THRESHOLD']['20_40']
    flags=dict(HARD_SHORTAGE=dist['comparisons']['hard_rate_ratio']<.5 or s['strict_hard20']<32 or s['strict_hard_frames']<16,
        CONTROLLED_CAPACITY_OK=fullcurve['30']['PCK10'] is not None and fullcurve['30']['PCK10']>=.5 and fullcurve['40']['PCK10']>=.3,
        TRAIN_HARD_FIT_OK=natural['PCK10'] is not None and natural['PCK10']>=.5,
        CANDIDATE_TRANSFER_GAP=heat['gap_pp'] is not None and heat['gap_pp']>=20,
        TARGET_CONSISTENCY_CONFLICT=consistency['strict_fraction_hard'] is not None and consistency['strict_fraction_hard']<.5*consistency['strict_fraction_easy'])
    if flags['HARD_SHORTAGE'] and flags['CONTROLLED_CAPACITY_OK']:primary='HARD_SUPERVISION_SHORTAGE'
    elif flags['TARGET_CONSISTENCY_CONFLICT']:primary='TARGET_RELIABILITY_SHORTAGE'
    elif not flags['HARD_SHORTAGE'] and not flags['TRAIN_HARD_FIT_OK'] and not flags['CONTROLLED_CAPACITY_OK']:primary='TRAIN_FIT_LIMIT'
    elif flags['TRAIN_HARD_FIT_OK'] and flags['CONTROLLED_CAPACITY_OK'] and flags['CANDIDATE_TRANSFER_GAP']:primary='TRANSFER_MORPHOLOGY_GAP'
    else:primary='MIXED_UNRESOLVED'
    secondary='TRANSFER_MORPHOLOGY_GAP_SIGNAL' if flags['CANDIDATE_TRANSFER_GAP'] else 'CONTROLLED_CORRECTION_BASIN_LIMIT' if not flags['CONTROLLED_CAPACITY_OK'] else 'EVAL_REFERENCE_UNCERTAINTY'
    C.save(C.DOC/'DECISION.json',dict(primary=primary,secondary=secondary,flags=flags,reference_limit='unknown metadata is a limitation, not proof labels are wrong',
        next_one_experiment='Frozen-target hard-input dose replacement: controlled original-coordinate20/30/40px corruption on the same TRAIN, fixed mixture, one300-step FULL125 vs saved FULL125; design only, separate authorization required',
        no_new_training=True,thresholds_unchanged=True))
    print('ANALYZED',dist['comparisons'],flags,primary,flush=True)

if __name__=='__main__':main()
