"""One fixed corrected-head CAL128 pass; no training, detector, rays, or PnP."""
from pathlib import Path
from collections import Counter
import argparse
import base64
import gzip
import hashlib
import json
import math
import os
import sys
import time
import zlib
import numpy as np
from . import observations as O

sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[3]
DOC=REPO/'_docs/experiments/pallet_boundary_corner_refiner_20261010_v2'
REPAIR=REPO/'_docs/experiments/pallet_kp_supervision_repair_20261010_v1'
CORRECTED=REPO/'_docs/experiments/pallet_kp_corrected_supervision_20261010_v1'
FEATURES=Path('/dev/shm/pallet-observation-private-20261009/learned_cache/features.npy')
CHECKPOINT=Path('/dev/shm/pallet-kp-supervision-repair-private-20261010/learned_fits/IMAGE_ROLE.pt')
Z_ONE_SIDED_95=1.6448536269514722


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def binding(path):
    path=Path(path).resolve()
    return dict(path=str(path.relative_to(REPO)) if path.is_relative_to(REPO) else path.name,
                origin='public_repository' if path.is_relative_to(REPO) else 'external_readonly_dependency',
                sha256=sha(path),bytes=path.stat().st_size)


def write_new(path,value):
    with Path(path).open('x') as f: f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def save_new(path,rows):
    # The exclusive outer file guard prevents an accidental second calibration pass.
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as stream:
            for row in rows: stream.write((json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode())


def read_rows(path):
    return [json.loads(line) for line in gzip.open(path,'rt')]


def inputs(args):
    return dict(features=Path(args.features),checkpoint=Path(args.checkpoint),ready_targets=REPAIR/'READY_PREPARED_TARGETS.npz',
                ready_rows=REPAIR/'READY_SOURCE_TARGET_ROWS.jsonl.gz',checkpoint_metadata=CORRECTED/'CHECKPOINT_METADATA.json',
                original_model=REPO/'scripts/research/pallet_observation_refiner_20261009_v1/model.py',
                observations=Path(O.__file__),calibration=Path(__file__))


def freeze(args):
    DOC.mkdir(parents=True,exist_ok=True)
    value=dict(schema='source_only_boundary_observation_calibration_protocol_v2',created_before_head_calls=True,
        inputs={key:binding(path) for key,path in inputs(args).items()},arm='IMAGE_ROLE',indices=list(range(768,896)),
        partition='calibration',families=128,batch_size=16,head_forward_calls=8,feature_recompute=False,
        source_test_used=False,real_GT_or_pose_scores_read=False,new_RGB=0,new_training_updates=0,new_detector_calls=0,new_PnP_calls=0,new_rays=0,
        coordinate='one argmax among bins0..64, integer1px; no averaging of coordinates or separate basins',
        confidence='sigmoid(z_best_candidate-z_NONE)=p_best/(p_best+p_NONE); minimum0.5',
        correctness='certified sourcePOSITIVE and absolute bestbin-(lo+weight)<=2px (frozen8px admission /4); sourceNONE incorrect; IGNORE unknown excluded',
        coverage='Each semantic edge requires at least one certified CAL POSITIVE; absent coverage means MODEL_CALIBRATION_UNSUPPORTED, never physicalabsence, and does not prohibit independent N3 observations',
        threshold='least restrictive unique confidence>=0.5 whose empirical one-sided95percent Wilson lower precision>=0.95; include ties; no passing cutoff means newheadquery abstain',
        Wilson_z=Z_ONE_SIDED_95,precision_target=.95,
        confidence_limits='Queries share scenes and chosen threshold is scanned on CAL: Wilson is a calibration diagnostic, not an independent coverage or transfer guarantee',
        uncertainty='sigma=max(0.5,sqrt(sum conditional_candidate_softmax(k)*(k-best)^2)); queryscale=max(1,q95(abs_error/sigma) on accepted sourcePOS); quantile higher',
        consensus='Enumerate<=21query pairs/edge; orthogonal residual<=calibratedqueryradius; at least3distinctqueries, queryindexspan>=2, actual projected supportspan>=2*(initialedgelength/8); one weightedTLS recheck/refit',
        line_covariance='inverse weighted linear angle/offset information; weights1/queryradius^2, floor0.5px; scale at least1 by standardized residual RMS; explicitly a local surrogate',
        corner_reference='Only pairs whose contributing queries all have certified actual-wire sourcePOS. Intersect fitted true actual_target_uv support lines; this is actual-wire-supported virtual intersection, not independently certified physicalcorner or cuboidmask',
        corner_uncertainty='propagate both line surrogates through inverse normal matrix; cornerscale=max(1,q95(reference_error/sigma)); reject if scaled radius exceeds one nominalqueryspacing; numericdet>1e-6',
        extrapolation='max(1/6,q95(referencecorner distance outside predicted supports /actualsupportspan)); normal endpoints sampled1/8..7/8 need1/6 and are not blanket rejected',
        geometry_minimum_calibration_intersections=10,off_image_queries_and_corners='abstain',
        same_edge_correlations='query samples not3Dpoint IDs; one corner ID at mostonce; usedcorner edges separated from partialline output',
        downstream='nearestN3<=8px admission implemented downstream; no originalcoordinates automatically become learnedobservations')
    write_new(DOC/'CALIBRATION_PROTOCOL.json',value)
    print('CALIBRATION_PROTOCOL',sha(DOC/'CALIBRATION_PROTOCOL.json'),flush=True)


def wilson_lower(correct,count):
    if count==0: return 0.
    p=correct/count; zz=Z_ONE_SIDED_95**2
    return (p+zz/(2*count)-Z_ONE_SIDED_95*math.sqrt(p*(1-p)/count+zz/(4*count**2)))/(1+zz/count)


def choose_threshold(scores,correct):
    scores=np.asarray(scores,float); correct=np.asarray(correct,bool)
    order=np.argsort(-scores,kind='stable'); scores=scores[order]; correct=correct[order]
    cumulative=np.cumsum(correct); scan=[]; chosen=None
    for end in range(1,len(scores)+1):
        if end<len(scores) and scores[end-1]==scores[end]: continue
        threshold=float(scores[end-1])
        if threshold<.5: break
        n=end; good=int(cumulative[end-1]); lower=wilson_lower(good,n)
        row=dict(threshold=threshold,accepted=n,correct=good,incorrect=n-good,precision=good/n,Wilson_lower=lower)
        scan.append(row)
        if lower>=.95: chosen=row
    return chosen,scan


def q95(values):
    return float(np.quantile(values,.95,method='higher')) if values else None


def calibrate(saved,source,targets,protocol):
    support=sorted({q['edge'] for r in source for q in r['queries'] if q['proposed_target']=='POSITIVE'})
    allscores=[]; allcorrect=[]; working=[]
    for stored,row in zip(saved,source):
        raw=np.frombuffer(zlib.decompress(base64.b64decode(stored['logits_fp32_zlib_base64'])),dtype='<f4').reshape(84,66)
        assert hashlib.sha256(raw.tobytes()).hexdigest()==stored['raw_logits_sha256']
        _,best,score,sigma=O.logits_statistics(raw); query=O.query_geometry(row['frozen_selected_points']); query['raw_hw']=row['raw_hw']
        index=row['index']; valid=targets['valid'][index]; positive=valid&(targets['lo'][index]<65)
        error=np.abs(best-(targets['lo'][index]+targets['weight'][index]))
        xy=query['candidate'][np.arange(84),best]; h,w=row['raw_hw']; inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        eligible=valid&query['valid']&inside&np.isin(np.arange(84)//7,support)
        correct=positive&(error<=2)
        allscores.extend(score[eligible].tolist()); allcorrect.extend(correct[eligible].tolist())
        working.append(dict(row=row,query=query,raw=raw,best=best,score=score,sigma=sigma,positive=positive,error=error,eligible=eligible,correct=correct))
    chosen,scan=choose_threshold(allscores,allcorrect)
    accepted_errors=[]; ratios=[]; queryrows=[]; known=Counter()
    for x in working:
        accepted=x['eligible']&(x['score']>=chosen['threshold']) if chosen else np.zeros(84,bool)
        ap=accepted&x['positive']; accepted_errors.extend(x['error'][ap].tolist()); ratios.extend((x['error'][ap]/x['sigma'][ap]).tolist())
        known.update(valid=int(x['eligible'].sum()),accepted=int(accepted.sum()),correct=int((accepted&x['correct']).sum()),accepted_POSITIVE=int(ap.sum()),accepted_NONE=int((accepted&~x['positive']).sum()))
        queryrows.append(dict(index=x['row']['index'],id=x['row']['id'],known_eligible=x['eligible'].tolist(),correct_within2px=x['correct'].tolist(),
                              confidence=x['score'].tolist(),candidate=x['best'].tolist(),sigma_mode_px=x['sigma'].tolist(),accepted=accepted.tolist()))
    calibration=dict(schema='source_only_boundary_observation_calibration_v2',protocol_sha256=sha(DOC/'CALIBRATION_PROTOCOL.json'),
        supported_edges=support,unsupported_edges=sorted(set(range(12))-set(support)),
        unsupported_means='model calibration supervision coverage unavailable, not absence of physicalrealboundary',
        confidence=dict(enabled=chosen is not None,threshold=chosen['threshold'] if chosen else None,chosen=chosen,known_query_counts=dict(known),precision_target=.95),
        uncertainty=dict(query_scale=max(1.,q95(ratios) or 1.),query_error_q95_px=q95(accepted_errors),corner_scale=1.),
        geometry=dict(enabled=False,max_extrapolation_ratio=1/6,reference='actual-wire-certified query support line intersection; physicalcorner ownership not independently certified'),
        limits=protocol['confidence_limits'])
    geometryrows=[]
    for x in working:
        decoded=O.decode(x['query'],x['raw'],calibration); lookup={line['edge']:line for line in decoded['lines']}
        for candidate in decoded['diagnostics']['corner_candidates']:
            aa,bb=candidate['edges']; actual=[]; eligible=True
            for edge in [aa,bb]:
                line=lookup[edge]; truth=[x['row']['queries'][i] for i in line['queries']]
                if any(q['proposed_target']!='POSITIVE' for q in truth): eligible=False; break
                fitted=O.fit_line([q['actual_target_uv'] for q in truth],np.ones(len(truth)))
                if fitted is None: eligible=False; break
                fitted.update(edge=edge,nominal_query_spacing_px=line['nominal_query_spacing_px']); actual.append(fitted)
            if not eligible: continue
            reference=O.intersection(*actual)
            if reference is None: continue
            h,w=x['row']['raw_hw']; point=np.array(reference['xy'])
            if not(0<=point[0]<w and 0<=point[1]<h): continue
            error=float(np.linalg.norm(np.array(candidate['xy'])-point)); gap=[]
            for edge in [aa,bb]:
                line=lookup[edge]; s=float((point-np.array(line['center']))@line['tangent']); low,high=line['support_interval_px']
                gap.append(max(low-s,s-high,0.)/line['support_length_px'])
            geometryrows.append(dict(index=x['row']['index'],id=x['row']['id'],corner=candidate['id'],edges=[aa,bb],
                supporting_query_ids=[lookup[e]['queries'] for e in [aa,bb]],reference_xy=point.tolist(),predicted_xy=candidate['xy'],
                error_px=error,sigma_px=candidate['sigma_px'],error_over_sigma=error/max(candidate['sigma_px'],1e-12),
                true_corner_support_extrapolation_ratio=max(gap),reference_from_certified_actual_wire_queries=True,physical_corner_ownership_independently_proved=False))
    if len(geometryrows)>=10:
        calibration['uncertainty']['corner_scale']=max(1.,q95([r['error_over_sigma'] for r in geometryrows]))
        calibration['geometry'].update(enabled=True,max_extrapolation_ratio=max(1/6,q95([r['true_corner_support_extrapolation_ratio'] for r in geometryrows])),calibration_intersections=len(geometryrows),reference_error_q95_px=q95([r['error_px'] for r in geometryrows]))
    else: calibration['geometry'].update(enabled=False,calibration_intersections=len(geometryrows),reason='fewer than10supported calibration intersections')
    return calibration,scan,queryrows,geometryrows


def run(args):
    begin=time.monotonic(); protocol=json.loads((DOC/'CALIBRATION_PROTOCOL.json').read_text()); before={key:binding(path) for key,path in inputs(args).items()}
    assert before==protocol['inputs']; assert not(DOC/'CALIBRATION_ROWS.jsonl.gz').exists()
    write_new(DOC/'CALIBRATION_START.json',dict(protocol=binding(DOC/'CALIBRATION_PROTOCOL.json'),head_forward_calls=0,training_updates=0))
    source=read_rows(REPAIR/'READY_SOURCE_TARGET_ROWS.jsonl.gz')[768:896]; assert [r['index'] for r in source]==list(range(768,896)); assert all(r['partition']=='calibration' for r in source)
    features=np.load(args.features,mmap_mode='r'); assert features.shape==(1024,84,28,65) and features.dtype==np.float16
    targets=dict(np.load(REPAIR/'READY_PREPARED_TARGETS.npz'))
    os.environ.setdefault('PALLET_SOURCE_ROOT',args.source_root)
    import torch
    from scripts.research.pallet_observation_refiner_20261009_v1 import model as M
    torch.set_num_threads(1); meta=json.loads((CORRECTED/'CHECKPOINT_METADATA.json').read_text())
    expected=next(r for r in meta['rows'] if r['arm']=='IMAGE_ROLE'); assert sha(args.checkpoint)==expected['checkpoint']['sha256']
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False); assert checkpoint['arm']=='IMAGE_ROLE' and checkpoint['steps']==3000 and checkpoint['config']==M.CONFIG
    head=M.CorrespondenceHead().cuda().eval(); head.load_state_dict(checkpoint['model']); head.requires_grad_(False)
    saved=[]; forwards=0
    with torch.no_grad():
        for start in range(768,896,16):
            x=torch.tensor(np.asarray(features[start:start+16]),dtype=torch.float32,device='cuda'); logits=head(x,'IMAGE_ROLE').float().cpu().numpy(); forwards+=1
            for j,index in enumerate(range(start,start+16)):
                raw=logits[j].astype('<f4').tobytes(); row=source[index-768]
                saved.append(dict(index=index,id=row['id'],family=row['family'],partition='calibration',logits_shape=[84,66],
                    raw_logits_sha256=hashlib.sha256(raw).hexdigest(),logits_fp32_zlib_base64=base64.b64encode(zlib.compress(raw)).decode(),
                    source_ready_row_semantic_sha256=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest()))
    assert forwards==8 and len(saved)==128
    del head; torch.cuda.empty_cache()
    save_new(DOC/'CALIBRATION_ROWS.jsonl.gz',saved)
    print('CALIBRATION_GPU_COMPLETE',forwards,len(saved),sha(DOC/'CALIBRATION_ROWS.jsonl.gz'),flush=True)
    calibration,scan,queryrows,geometryrows=calibrate(saved,source,targets,protocol)
    after={key:binding(path) for key,path in inputs(args).items()}; assert before==after
    save_new(DOC/'CALIBRATION_THRESHOLD_SCAN.jsonl.gz',scan); save_new(DOC/'CALIBRATION_QUERY_DIAGNOSTICS.jsonl.gz',queryrows); save_new(DOC/'CALIBRATION_GEOMETRY_ROWS.jsonl.gz',geometryrows)
    calibration.update(raw_rows=binding(DOC/'CALIBRATION_ROWS.jsonl.gz'),threshold_scan=binding(DOC/'CALIBRATION_THRESHOLD_SCAN.jsonl.gz'),geometry_rows=binding(DOC/'CALIBRATION_GEOMETRY_ROWS.jsonl.gz'))
    write_new(DOC/'CALIBRATION.json',calibration)
    write_new(DOC/'CALIBRATION_EXECUTION.json',dict(complete=True,inputs_before_after_equal=True,input_bindings=before,protocol=binding(DOC/'CALIBRATION_PROTOCOL.json'),
        calibration=binding(DOC/'CALIBRATION.json'),head_forward_calls=8,head_image_exposures=128,arm='IMAGE_ROLE',partition='calibration',
        source_test_exposures=0,real_frames=0,training_updates=0,detector_calls=0,feature_recomputations=0,PnP_calls=0,rays=0,new_RGB=0,wall_seconds=time.monotonic()-begin))
    print('CALIBRATION_COMPLETE',sha(DOC/'CALIBRATION.json'),json.dumps(calibration),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['freeze','run']); parser.add_argument('--features',default=str(FEATURES)); parser.add_argument('--checkpoint',default=str(CHECKPOINT)); parser.add_argument('--source-root',default=os.environ.get('PALLET_SOURCE_ROOT','/home/minjae/Documents/github/pallet-pose'))
    args=parser.parse_args(); freeze(args) if args.mode=='freeze' else run(args)
