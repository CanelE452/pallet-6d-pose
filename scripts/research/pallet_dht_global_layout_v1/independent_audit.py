"""Read-only replay of completed layout outputs and a labelled posthoc oracle.

Writes only this new run's audit artifacts. No neural forward, training, GT
correction, candidate injection or selection updates are performed.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import math

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
ARMS=('baseline','independent','global','global_shared_only')


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(c,m):
    if not c:raise AssertionError(m)
def write(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def compare(a,b,name,tol=1e-10):
    require(bool(np.allclose(a,b,atol=tol,rtol=0,equal_nan=True)),name)
def iou(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    union=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter
    return float(inter/union) if union>0 else 0.
def H(x):return np.sqrt(1+np.asarray(x)**2)-1


def independent_dlt_cost(points,diagonal,sigma,cfg):
    q=np.asarray(points,float);z=q-q.mean(0)
    rms=float(np.sqrt(np.sum(z*z)/8))
    if rms<=cfg['collapse_rms_raw_diagonal_fraction']*diagonal:return False,None
    spread=np.linalg.svd(z,compute_uv=False)
    if spread[1]<=spread[0]*cfg['image_spread_rank_ratio_min']:return False,None
    scale=np.sqrt(2)/rms;qn=z*scale
    # Standard 2-row DLT assembled independently of the decoder implementation.
    cube=np.array([[-1,-1,1,1],[1,-1,1,1],[1,1,1,1],[-1,1,1,1],
                   [-1,-1,-1,1],[1,-1,-1,1],[1,1,-1,1],[-1,1,-1,1]],float)
    A=np.vstack([row for x,(u,v) in zip(cube,qn) for row in
                 (np.r_[x,np.zeros(4),-u*x],np.r_[np.zeros(4),x,-v*x])])
    _,s,V=np.linalg.svd(A,full_matrices=False)
    if s[-2]<=s[0]*cfg['dlt_rank_ratio_min']:return False,None
    P=V[-1].reshape(3,4);ps=np.linalg.svd(P,compute_uv=False)
    if ps[-1]<=ps[0]*cfg['camera_rank_ratio_min']:return False,None
    out=cube@P.T;dep=out[:,2];margin=np.max(np.abs(dep))*cfg['same_depth_relative_margin']
    if not (np.all(dep>margin) or np.all(dep<-margin)):return False,None
    fit=out[:,:2]/dep[:,None]/scale+q.mean(0)
    return True,float(H(np.linalg.norm(fit-q,axis=1)/sigma).mean())


def score_terms(rec,points,cfg):
    q=np.asarray(points,float);ev=rec['evidence'];sigma=ev['sigma'];edges=ev['edges']
    endpoints=[];pairs=[]
    for role,(u,v) in enumerate(edges):
        h=np.asarray(ev['lines'][role],float);w=np.asarray(ev['line_weights'][role],float)
        if not len(w):endpoints.extend([0.,0.]);pairs.append(0.);continue
        require(np.isfinite(h).all() and np.isfinite(w).all() and np.all(w>0),'Finite positive line weights')
        compare(np.linalg.norm(h[:,:2],axis=1),np.ones(len(w)),'unit Hessian lines',1e-10)
        compare(w.sum(),1.,'normalized retained masses',1e-12)
        residual=np.array([h[:,:2]@q[k]+h[:,2] for k in (u,v)])/sigma
        costs=H(residual)
        for c in (costs[0],costs[1],costs.mean(0)):
            a=np.log(w)-c;top=a.max();val=-float(top+np.log(np.exp(a-top).sum()))
            endpoints.append(val)
        pairs.append(endpoints.pop())
    # endpoints contains exactly the 24 point-role incidences, degree3 each.
    independent=float(np.mean(endpoints));shared=float(np.mean(pairs))
    diagonal=math.hypot(rec['width'],rec['height']);base=np.asarray(rec['baseline']['points'],float)
    anchor=float(H(np.linalg.norm(q[:8]-base[:8],axis=1)/(.1*diagonal)).mean())
    valid,gcost=independent_dlt_cost(q[:8],diagonal,sigma,cfg)
    return independent,shared,anchor,valid,gcost


def audit(run_dir):
    run=Path(run_dir).resolve();p=read(run/'PROTOCOL.json');psha=sha(run/'PROTOCOL.json')
    comp=read(run/'EVALUATION_COMPLETION.json');require(comp['complete'] and comp['PASS'],'Completed evaluation required')
    require(comp['protocol_sha256']==psha,'Protocol binding')
    verified={}
    for path,h in comp['input_sha256'].items():
        require(sha(path)==h,'Evaluation input SHA '+path);verified[path]=h
    for name,h in comp['output_sha256'].items():require(sha(run/name)==h,'Evaluation output SHA '+name)
    pred=read(run/'PREDICTIONS.json');result=read(run/'RESULTS.json');frames=read(run/'FRAME_METRICS.json')
    cal=read(run/'CALIBRATION_SELECTION.json');fscore=read(run/'CALIBRATION_FRAME_SCORES.json')
    require(cal['complete'] and cal['PASS'] and cal['no_real_GT_used'],'Synthetic selection completed')
    require(cal['protocol_sha256']==psha and pred['calibration_selection_sha256']==sha(run/'CALIBRATION_SELECTION.json'),'Selection hash binding')
    require(sha(run/'CALIBRATION_FRAME_SCORES.json')==cal['frame_scores_sha256'],'Calibration observations hash')
    require(not pred['selection_uses_GT'] and pred['no_new_CNN_forward'],'GT-free saved selections')
    freeze=read(run/'SOURCE_FREEZE.json');fmapping=freeze.get('sha256',freeze.get('source_sha256',{}))
    for path,h in pred['source_sha256'].items():
        require(sha(path)==h and fmapping.get(path,fmapping.get(str(Path(path).relative_to(ROOT))))==h,'Frozen source '+path)
    source=ROOT/p['input_run'];cache=read(source/'CACHE_RECORDS.json');manifest=read(source/'MANIFEST.json')
    cache_byid={r['id']:r for r in cache['records']};source_byid={r['id']:r for r in manifest['records']}
    prefix=p['calibration']['sha256_sort_prefix'];train=[r for r in cache['records'] if r['population']=='synth_train']
    expected=sorted(train,key=lambda r:hashlib.sha256((prefix+r['id']).encode()).hexdigest())[:256]
    require([r['id'] for r in expected]==cal['calibration_ids'],'SHA-ranked calibration IDs')
    counts={}
    for r in expected:
        sr=source_byid[r['id']]['source_record'];kp=np.asarray(sr['targets'][0]['keypoints_normalized'])
        counts[r['id']]=int(((kp[:8,2]>0)&np.asarray(r['baseline']['point_valid'][:8],bool)&r['loss_matched']).sum())
    grid=[]
    for wp in p['calibration']['w_point']:
        for wg in p['calibration']['w_geometry']:
            rs=[r for r in fscore['records'] if r['w_point']==wp and r['w_geom']==wg]
            require(len(rs)==256 and {r['id'] for r in rs}==set(counts),'Calibration grid population')
            require(all(r['observed_corners']==counts[r['id']] for r in rs),'Calibration denominator')
            score=sum(r['sum_normalized_error'] for r in rs)/sum(counts.values())
            target=next(r for r in cal['grid'] if r['w_point']==wp and r['w_geom']==wg)
            compare(score,target['mean_normalized_corner_error'],'Calibration pooled objective',1e-12)
            grid.append((score,wp,wg))
    minimum=min(v[0] for v in grid);eligible=[v for v in grid if v[0]<=minimum+p['calibration']['tie_absolute_tolerance']]
    winner=sorted(eligible,key=lambda v:(-v[1],v[2]))[0]
    require(cal['selected']=={'w_point':winner[1],'w_geom':winner[2]},'Predeclared calibration tie ordering')
    records={r['id']:r for r in pred['records']};rows={r['id']:r for r in frames['records']}
    require(len(records)==831 and set(records)==set(rows),'831 evaluation identities')
    require(not(set(records)&set(counts)),'Calibration/evaluation disjointness')
    actual_pops=collections.Counter(r['population'] for r in records.values());require(actual_pops=={'synth_val':512,'real_dev':319},'Population counts')
    old_rows={r['id']:r for r in read(source/'FRAME_METRICS.json')['records']}
    real_items={r['frame_id']:r for r in read(ROOT/'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json')['items']}
    cfg=read(run/'GEOMETRY_SPEC.json')['config'];maxscore=0.;maxerror=0.;all_stats={};obs_masks={}
    for fid,rec in records.items():
        row=rows[fid];b=rec['baseline'];base=np.asarray(b['points'],float);bv=np.asarray(b['point_valid'],bool)&np.isfinite(base).all(1)
        if rec['population']=='real_dev':
            obj=read(ROOT/real_items[fid]['gt_v2_path'])['objects'][0];gt=np.asarray([a['xy'] for a in obj['keypoint_annotations']],float);gv=np.array([a['visibility']>0 for a in obj['keypoint_annotations']]);box=np.r_[gt[:8].min(0),gt[:8].max(0)];matched=b['box_xyxy'] is not None and iou(b['box_xyxy'],box)>=.5
        else:
            sr=source_byid[fid]['source_record'];t=np.asarray(sr['targets'][0]['keypoints_normalized'],float);hh,ww=sr['prepared_shape_hw'];gt=t[:,:2]*[ww,hh]-sr['reflect_pad_px'];gv=t[:,2]>0;matched=bool(cache_byid[fid]['loss_matched'])
        require(np.array_equal(gt,row['gt_points']) and np.array_equal(gv,row['gt_supervised']),'Scored GT '+fid)
        require(matched==row['baseline_match_iou50']==old_rows[fid]['baseline_match_iou50'],'Frozen association '+fid)
        require(np.array_equal(base,old_rows[fid]['arms']['baseline']['points']),'Original baseline '+fid)
        obs=gv&bv&matched;obs_masks[fid]=obs
        pool=np.asarray(rec['evidence']['candidate_xy'],float);poolvalid=np.asarray(rec['evidence']['candidate_valid'],bool)
        require(pool.shape==(8,16,2),'K16 pool');require(np.array_equal(pool[:,:8],np.broadcast_to(base[None,:8],(8,8,2))),'All8 baseline locations retained')
        for arm in ARMS:
            ar=b if arm=='baseline' else rec['arms'][arm];q=np.asarray(ar['points'],float);valid=np.asarray(ar['point_valid'],bool)&np.isfinite(q).all(1);require(np.array_equal(valid,bv) and np.array_equal(q[8],base[8]),'Centroid/mask preservation')
            err=np.linalg.norm(q-gt,axis=1);saved=row['arms'][arm]['errors_px']
            for i,ok in enumerate(obs):
                require((saved[i] is not None)==bool(ok),'Observation denominator')
                if ok:maxerror=max(maxerror,abs(float(err[i])-saved[i]))
            if arm=='baseline':continue
            inds=np.asarray(ar['selected_candidate_indices'],int);require(inds.shape==(8,) and poolvalid[np.arange(8),inds].all(),'Valid selected candidate IDs')
            require(np.array_equal(q[:8],pool[np.arange(8),inds]),'Selected full-layout candidate coordinates')
            ind,shared,anchor,gvalid,gcost=score_terms(rec,q,cfg)
            if arm=='independent':score=ind+winner[1]*anchor
            else:
                exception=bool(ar.get('baseline_exception_selected',False));require(gvalid or exception,'Global geometry guard')
                if exception:require(np.array_equal(q,base) and not gvalid,'Invalid-baseline exception')
                wgeom=winner[2] if arm=='global' else 0.
                score=shared+winner[1]*anchor+wgeom*(gcost if gvalid else 0.)
            maxscore=max(maxscore,abs(score-ar['score_total']));compare(score,ar['score_total'],'Saved score replay',1e-7)
    require(maxerror==0.,'Exact source error replay')
    for pop in ('synth_val','real_dev'):
        selected=[r for r in rows.values() if r['population']==pop]
        for arm in ARMS:
            vals=np.array([v for r in selected for v in r['arms'][arm]['errors_px'] if v is not None]);n_gt=sum(sum(r['gt_supervised']) for r in selected);n_frames=sum(any(v is not None for v in r['arms'][arm]['errors_px']) for r in selected)
            s=dict(n_observed_points=len(vals),n_gt_points=n_gt,n_frames_with_observations=n_frames,mean_px=float(vals.mean()),median_px=float(np.median(vals)),p90_px=float(np.quantile(vals,.9)))
            ref=next(r for r in result['summaries'] if r['population']==pop and r['arm']==arm)
            for key,value in s.items():compare(value,ref[key],pop+':'+arm+':'+key,1e-12)
            all_stats[pop+':'+arm]=s
    require(all_stats['real_dev:baseline']['n_observed_points']==2738 and all_stats['real_dev:baseline']['n_gt_points']==2818,'Official real denominator')
    # Bounded, predeclared after-selection diagnostic: never inject this oracle.
    from . import geometry as G
    review_path=ROOT/'data/pallet/results/pallet_dht_gt_audit_v1/ROOT_GT_VISUAL_REVIEW.json';review=read(review_path)
    diag_ids=sorted({r['id'] for r in review['records']}|{'eval_pallet07:1778652166837872128'})
    oracle_rows=[];replay_max=0.
    for fid in diag_ids:
        rec=records[fid];ev=rec['evidence'];row=rows[fid];obs=obs_masks[fid];raw_path=ev['raw_frame_path'];evidence_path=ev['candidate_evidence_path'];expected_raw=pred['bindings']['consumed_input_sha256'][str(Path(raw_path).resolve())]
        require(sha(raw_path)==expected_raw and sha(evidence_path)==ev['candidate_evidence_sha256'],'Oracle replay input SHA')
        with np.load(raw_path,allow_pickle=False) as ff,np.load(evidence_path,allow_pickle=False) as ee:prepared=G.prepare(rec,ff,ee)
        bank=G.build_layouts(prepared,winner[1]);selected=G.select_layout(prepared,bank,winner[2]);actual=np.asarray(rec['arms']['global']['points'],float)
        require(np.array_equal(selected['points_xy'],actual),'Frozen bounded search replay')
        replay_max=max(replay_max,abs(selected['selected_score']-rec['arms']['global']['score_total']))
        gt=np.asarray(row['gt_points'],float);oracle=actual.copy();indices=np.asarray(rec['arms']['global']['selected_candidate_indices'],int).copy();mask=obs[:8]
        for i in np.flatnonzero(mask):
            distances=np.linalg.norm(prepared['candidate_xy'][i]-gt[i],axis=1);indices[i]=int(np.argmin(np.where(prepared['candidate_valid'][i],distances,np.inf)));oracle[i]=prepared['candidate_xy'][i,indices[i]]
        candidate=G.score_layout(prepared,oracle,winner[1],winner[2],baseline_exception=np.array_equal(oracle,prepared['points_xy']))
        # The scorer is GT-free, but this particular input layout was built
        # using GT after selection. Preserve both pieces of provenance.
        candidate.update(uses_gt=True,GT_used_to_construct_input_layout=True,
                         score_function_reads_gt=False,used_for_selection=False)
        mean_before=float(np.linalg.norm(actual[:8]-gt[:8],axis=1)[mask].mean()) if mask.any() else None
        mean_oracle=float(np.linalg.norm(oracle[:8]-gt[:8],axis=1)[mask].mean()) if mask.any() else None
        if not mask.any():category='no_observed_corners'
        elif mean_oracle>=mean_before-1e-9:category='selected_already_percorner_nearest_or_tie'
        elif not candidate['eligible']:category='geometry_guard_excludes_nearer_layout'
        elif candidate['total_score']<selected['selected_score']-1e-9:category='beam_search_miss_of_lower_scoring_nearer_layout'
        else:category='score_or_tie_policy_prefers_farther_layout'
        oracle_rows.append(dict(id=fid,observed_corner_indices=np.flatnonzero(mask).tolist(),selected_corner_mean_error_px=mean_before,
            sameID_nearest16_corner_mean_error_px=mean_oracle,classification=category,oracle_candidate_indices=indices.tolist(),
            oracle_actual_score=candidate,selected_actual_score=selected['selected'],
            oracle_layout_in_actual_bank=any(np.array_equal(q,oracle) for q in bank['hypothesis_xy']),
            uses_GT=True,used_for_selection=False,centroid_copied=True,
            missing_or_unsupervised_corners='Keep selected global coordinate; no GT nearest substitution.',
            baseline_exception='Only if whole oracle layout is exactly original baseline, matching actual scorer policy.'))
    oracle={'schema':'pallet_global_layout_posthoc_oracle_v1','complete':True,'uses_GT':True,'used_for_selection':False,
        'scope':'Pre-existing root19 review IDs union original case; same-ID per-corner nearest in frozenK16 pool; not free-ID assignment or deployable correction.',
        'frame_ids':diag_ids,'counts':dict(collections.Counter(r['classification'] for r in oracle_rows)),'records':oracle_rows,
        'limit':'Unobserved corners retain selected coordinates. This feasible-point oracle need not be an admissible projective cuboid. A score misrank is relative to annotated same-ID error, not certification that GT is physically exact.'}
    oracle_path=run/'POSTHOC_SEARCH_SCORE_DIAGNOSIS.json';write(oracle_path,oracle)
    payload={'schema':'pallet_global_layout_independent_audit_v1','complete':True,'PASS':True,'protocol_sha256':psha,
        'scope':'Arithmetic, source, denominator and saved geometry selection audit; no accuracy/GT correctness PASS claim.',
        'n_eval_frames':831,'n_real_frames':319,'n_synthetic_validation_frames':512,'n_real_observed_points':2738,'n_real_supervised_points':2818,
        'same_ID':True,'centroid_unchanged':True,'all_masks_preserved':True,'source_error_max_abs_delta_px':maxerror,
        'all831_three_selected_score_replay_max_abs_delta':maxscore,'score_absolute_tolerance':1e-7,
        'score_tolerance_meaning':'Independent float64 DLT/SVD and sqrt-versus-hypot arithmetic; no selection threshold or metric changes.',
        'calibration_audit':{'SHA_ranked_256_ids_PASS':True,'real_overlap':False,'nine_grid_objective_sum_and_denominator_replay_PASS':True,
                             'selected':cal['selected'],'selection_coordinates_recomputed':False,'limit':'Recomputed grid sums from hash-bound per-frame scores and independently reconstructed masks, not all256x9 geometric searches.'},
        'independent_metric_recomputation':all_stats,'bounded_search_replay':{'frames':len(diag_ids),'coordinates_exact':True,'score_max_delta':replay_max},
        'posthoc_oracle':{'path':str(oracle_path),'sha256':sha(oracle_path),'uses_GT':True,'used_for_selection':False,'counts':oracle['counts']},
        'no_new_CNN_forwards':True,'no_new_training':True,'GT_modified':False,
        'evaluation_input_sha256':comp['input_sha256'],'input_sha256':{str(run/n):sha(run/n) for n in ['PROTOCOL.json','CALIBRATION_SELECTION.json','CALIBRATION_FRAME_SCORES.json','PREDICTIONS.json','FRAME_METRICS.json','RESULTS.json','EVALUATION_COMPLETION.json','GEOMETRY_SPEC.json','SOURCE_FREEZE.json']},
        'source_sha256':{str(Path(__file__).resolve()):sha(__file__)},'output_sha256':{str(oracle_path):sha(oracle_path)}}
    write(run/'INDEPENDENT_AUDIT.json',payload)
    print(json.dumps({'complete':True,'PASS':True,'metrics':all_stats,'oracle_counts':oracle['counts'],'max_score_delta':maxscore},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);audit(parser.parse_args().run_dir)
