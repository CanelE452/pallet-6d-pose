"""Independent full238x8 saved-result audit; no scorer/runtime/model imports.

The old checker's pure physical-reference/metric functions are reused. Every
prediction, source/input/observation binding is validated before references.
No image decode, network, model, PnP, optimizer or learned update is called.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from scripts.research.pallet_guarded_bi_final_v1.independent import (
    Checks,metric,fixed_references,distribution,rotation)

ARMS=('R0','R1','N3_F','N3_BI','SUBPIX_F','SUBPIX_BI','N0','N1')
ERRORS=('translation_cm','rotation_deg','ADDsym_cm')
PARTITIONS={'CAL':46,'TEST':49,'REMAINDER':143,'PRIMARY':192}
BOOTSTRAP_SEED=20260917
BOOTSTRAP_DRAWS=10000
LEGACY_N3_SUBPIX='_docs/experiments/pallet_n3_subpix_20261008_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path,value):
    with Path(path).open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def _skip(text,start):
    i=start
    while text[i].isspace():i+=1
    if text[i] not in '[{':
        if text[i]=='\"':
            i+=1;escaped=False
            while i<len(text):
                c=text[i]
                if escaped:escaped=False
                elif c=='\\':escaped=True
                elif c=='\"':return i+1
                i+=1
            raise ValueError('Unclosed archived string')
        while i<len(text) and text[i] not in ',}]' and not text[i].isspace():i+=1
        return i
    stack=[text[i]];quoted=False;escaped=False;i+=1
    while i<len(text):
        c=text[i]
        if quoted:
            if escaped:escaped=False
            elif c=='\\':escaped=True
            elif c=='"':quoted=False
        elif c=='"':quoted=True
        elif c in '[{':stack.append(c)
        elif c in ']}':
            expected='[' if c==']' else '{'
            if not stack or stack.pop()!=expected:raise ValueError('Malformed archived JSON container')
            if not stack:return i+1
        i+=1
    raise ValueError('Unclosed archived JSON container')


def selected_fields(text,wanted):
    """Only selected predictor fields are decoded; nested score fields skipped."""
    decoder=json.JSONDecoder();i=0;out={};seen=set()
    while i<len(text) and text[i].isspace():i+=1
    if i==len(text) or text[i]!='{':raise ValueError('One JSON object per line required')
    i+=1
    while True:
        while i<len(text) and text[i].isspace():i+=1
        if text[i]=='}':
            if text[i+1:].strip():raise ValueError('Trailing archived JSON data')
            return out
        key,i=decoder.raw_decode(text,i)
        if not isinstance(key,str) or key in seen:raise ValueError('Duplicate archived JSON key')
        seen.add(key)
        while text[i].isspace():i+=1
        if text[i]!=':':raise ValueError('Missing archived JSON colon')
        start=i+1;end=_skip(text,start)
        if key in wanted:out[key]=json.loads(text[start:end])
        i=end
        while text[i].isspace():i+=1
        if text[i]==',':i+=1
        elif text[i]!='}':raise ValueError('Malformed archived JSON separator')


def transitions(rr,new,before):
    names=('failure_to_success','success_to_failure','both_success','both_failure','reference_unresolved');out={n:[] for n in names}
    for r in rr:
        a,b=[r['metrics'][x]['success_le5'] for x in (new,before)]
        key=('reference_unresolved' if a is None or b is None else 'failure_to_success' if a and not b else
             'success_to_failure' if b and not a else 'both_success' if a else 'both_failure')
        out[key].append(r['id'])
    return dict(counts={k:len(v) for k,v in out.items()},ids=out,full_denominator=len(rr))


def recount(rr,contrasts):
    """Frame-weight expansion, independent of production grouped summaries."""
    units=sorted({r['cluster_id'] for r in rr});lookup={c:i for i,c in enumerate(units)}
    draw=np.random.default_rng(BOOTSTRAP_SEED).multinomial(len(units),[1/len(units)]*len(units),size=BOOTSTRAP_DRAWS).astype(np.uint16)
    W=draw[:,[lookup[r['cluster_id']] for r in rr]].astype(float)
    def interval(values):
        mask=np.asarray([x is not None for x in values]);x=np.asarray([float(x) for x in values if x is not None]);w=W[:,mask]
        den=w.sum(1);keep=den>0;v=w[keep]@x/den[keep];nunits=len({r['cluster_id'] for r,m in zip(rr,mask) if m})
        return dict(mean_difference=math.fsum(x.tolist())/len(x) if len(x) else None,
            CI95=np.quantile(v,[.025,.975]).tolist() if nunits>1 and len(v) else None,
            common_available_n=int(mask.sum()),full_denominator=len(rr),excluded_unavailable_n=int((~mask).sum()),
            eligible_clusters=nunits,valid_resamples=int(keep.sum()),empty_resamples=int((~keep).sum()),
            improved=int((x<0).sum()),worsened=int((x>0).sum()),tied=int((x==0).sum()))
    arms={}
    for arm in ARMS:
        m=[r['metrics'][arm] for r in rr];s=[r['prediction_status'][arm] for r in rr]
        arms[arm]=dict(denominator=len(rr),successes_le5=sum(v['success_le5'] is True for v in m),
            successes_strict_lt5=sum(v['success_strict_lt5'] is True for v in m),
            no_pose_n=sum(not v['pose_available'] for v in m),unresolved_n=sum(v['success_le5'] is None for v in m),
            accepted_n=sum(v['accepted'] is True for v in s),fallback_n=sum(v['fallback'] is True for v in s),
            acceptance_known_n=sum(v['accepted'] is not None for v in s),
            exact5_n=sum(v['translation_cm']==5. or v['rotation_deg']==5. for v in m),
            errors={k:distribution(v[k] for v in m) for k in ERRORS},status_counts=dict(Counter(v['status'] for v in s)))
    comparisons={}
    for a,b in contrasts:
        binary=[None if any(r['metrics'][x]['success_le5'] is None for x in (a,b)) else
                int(r['metrics'][a]['success_le5'])-int(r['metrics'][b]['success_le5']) for r in rr]
        success=interval(binary);success['gains']=success.pop('worsened');success['losses']=success.pop('improved')
        if any(x is None for x in binary):success.update(conditional_known_label_mean_difference=success['mean_difference'],mean_difference=None,CI95=None)
        comparisons[a+'-'+b]=dict(transition=transitions(rr,a,b),success_rate=success,
            continuous={k:interval([None if any(r['metrics'][x][k] is None for x in (a,b)) else
                r['metrics'][a][k]-r['metrics'][b][k] for r in rr]) for k in ERRORS})
    return dict(rows=len(rr),arms=arms,comparisons=comparisons,bootstrap=dict(seed=BOOTSTRAP_SEED,resamples=BOOTSTRAP_DRAWS,
        units=len(units),clusters=units,cluster_sizes=dict(Counter(r['cluster_id'] for r in rr)),
        session_sizes=dict(Counter(r['session'] for r in rr)),draw_sha256=hashlib.sha256(draw.tobytes()).hexdigest()))


def point_source_audit(folder,prior,source_root,frames,checks):
    """Authenticate and compare the original predictor cache, never score fields."""
    old=read(prior/'inputs_v2/INPUTS.json')['frames'];legacy=Path(source_root)/LEGACY_N3_SUBPIX
    proto=read(legacy/'PROTOCOL.json');manifest={r['id']:r for r in proto['input_manifest']}
    wanted={f['id'] for f in frames};cached={}
    with gzip.open(legacy/'PREDICTIONS.jsonl.gz','rt') as f:
        for line in f:
            r=selected_fields(line,{'id','method','native_points','qFinal','prediction_support','raw_hw','fixed_metadata','actual_pose'})
            if r['id'] in wanted:
                key=(r['id'],r['method'])
                checks(key not in cached,'source/cache unique '+r['id']+'/'+r['method']);cached[key]=r
    checks(len(cached)==952,'source/four original predictor methods238 each')
    nonidentity=0;subpix_oldpose_n=0
    for f,o in zip(frames,old):
        key=f['id'];checks(key==o['id'],key+'/same frame order')
        checks({k:v for k,v in f.items() if k!='arms'}=={k:v for k,v in o.items() if k!='arms'},key+'/original metadata exact')
        for b in ('RAW_YOLO','N3_SUBPIX'):checks(f['arms'][b]==o['arms'][b],key+'/'+b+'/old input exact')
        m=manifest[key];checks(m['image_sha256']==f['image_sha256'] and m['K']==f['K'] and m['raw_hw']==f['raw_hw'],key+'/same image camera hash')
        checks(m['q0']==f['arms']['RAW_YOLO']['native_points'],key+'/exact original q0')
        for b,method in [('N3_ONLY','N3'),('SUBPIX_ONLY','SUBPIX')]:
            a=f['arms'][b];r=cached[(key,method)];q=np.asarray(a['native_points'],float)
            checks(q.shape==(9,2) and np.isfinite(q).all(),key+'/'+b+'/native9 finite')
            checks(a['native_points']==r['qFinal']==r['native_points'],key+'/'+b+'/source qFinal exact')
            if b=='N3_ONLY':checks(a['native_points']==m['qN'],key+'/N3/exact original qN')
            checks(a['native_points'][8]==m['q0'][8],key+'/'+b+'/center9 exact')
            checks(a['prediction_support']==r['prediction_support']==m['prediction_support'],key+'/'+b+'/original support exact')
            checks(a['keypoints_conf']==m['candidate_metadata']['keypoints_conf'],key+'/'+b+'/original confidence exact')
            checks(f['selected_index']==r['fixed_metadata']['selected_index']==m['selected_index'],key+'/'+b+'/same selected detection')
            coarse=a['coarse_pose'];checks(coarse.get('available') is True,key+'/'+b+'/own available F')
            rotation(coarse['R_cf']);rotation(coarse['R_physical'])
            checks(a.get('baseline_accepted') is None and a.get('baseline_fallback') is None,key+'/'+b+'/baseline unknown acceptance')
            if isinstance(r.get('actual_pose'),dict):
                checks.equal(r['actual_pose'],coarse,key+'/'+b+'/exact original saved F pose')
                if b=='SUBPIX_ONLY':subpix_oldpose_n+=1
            if b=='SUBPIX_ONLY':
                # Total cap about the original q0, not about another path.
                displacement=np.linalg.norm(q[:8]-np.asarray(m['q0'])[:8],axis=1)
                checks(float(displacement.max())<=.01*math.hypot(*f['raw_hw'])+1e-10,key+'/SubPix original1percent cap')
    checks(subpix_oldpose_n==19,'source/19 archived SubPix F poses retained')
    return dict(original_predictor_cache_rows=952,N3_coordinates_exact=238,SubPix_coordinates_exact=238,
                archived_SubPix_F_poses=19,original_image_hashes=238,
                unselected_GT_or_corner_score_fields_decoded=False,
                source_protocol_sha256=sha(legacy/'PROTOCOL.json'),source_journal_sha256=sha(legacy/'PREDICTIONS.jsonl.gz'))


def _file(binding,folder,checks,label):
    path=Path(binding['path']);path=path if path.is_absolute() else folder/path
    checks(sha(path)==binding['sha256'],label+'/sha')
    return path


def input_seal_audit(folder,prior,source_root,frames,checks):
    seal=read(folder/'inputs_v1/INPUT_SEAL.json');code=Path(__file__).resolve().parents[3]
    checks(seal['rows']==238 and seal['GT_inputs'] is False and seal['all_rows_and_failures_kept'] is True,'input/238 pre-GT rows')
    checks(seal['inputs_sha256']==sha(folder/'inputs_v1/INPUTS.json'),'input/plain byte hash')
    checks(seal['inputs_gzip_sha256']==sha(folder/'inputs_v1/INPUTS.jsonl.gz'),'input/gzip byte hash')
    checks(seal['contract_sha256']==sha(folder/'inputs_v1/CONTRACT.json'),'input/contract byte hash')
    checks(seal['F_parity_sha256']==sha(folder/'inputs_v1/F_PARITY.json'),'input/F parity byte hash')
    checks(seal['prior_input_seal_sha256']==sha(prior/'inputs_v2/INPUT_SEAL.json'),'input/unchanged prior seal')
    checks(seal['prior_inputs_sha256']==sha(prior/'inputs_v2/INPUTS.json'),'input/unchanged prior coordinates')
    checks(seal['ordered_ids']==[f['id'] for f in frames],'input/ordered IDs')
    checks(seal['canonical_row_sha256']==[hashlib.sha256(json.dumps(f,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest() for f in frames],'input/238 canonical row hashes')
    for name,h in seal['source_sha256'].items():
        path=code/name
        if not path.exists():path=Path(source_root)/name
        checks(sha(path)==h,'input/source '+name)
    for name,h in seal['input_sha256'].items():checks(sha(Path(source_root)/name)==h,'input/cache '+name)
    parity=read(folder/'inputs_v1/F_PARITY.json')
    checks(parity['failures']==0 and len(parity['rows'])==257,'input/257 saved F parity rows')
    checks(all(r['passed'] is True and r['max_abs_difference']==0. and r['available_and_selected_hypothesis_exact'] is True for r in parity['rows']),'input/exact numeric and discrete F parity')
    checks(dict(Counter(r['branch'] for r in parity['rows']))=={'N3_ONLY':238,'SUBPIX_ONLY':19},'input/cached F coverage')
    resources=seal['resources'];checks(resources['F_calls']==476 and resources['new_missing_F']==219 and resources['cached_F_parity_calls']==257,'input/declared F preparation count')
    checks(resources['GT_or_score_or_reference_objects_decoded']==0 and resources['new_NN_forwards']==0 and resources['new_GPU_calls']==0,'input/no labels or new NN declarations')
    return dict(source_files=len(seal['source_sha256']),input_files=len(seal['input_sha256']),saved_F_pose_parity_rows=257,missing_SubPix_F_computed=219)


def prediction_audit(folder,prior,source_root,checks):
    seal=read(folder/'PREDICTION_SEAL.json');protocol=read(folder/'PROTOCOL.json');code=Path(__file__).resolve().parents[3]
    checks(seal['rows']==238 and seal['arm_rows']==1904 and seal['arms']==list(ARMS),'prediction/full238x8')
    checks(seal['GT_inputs'] is False and seal['before_new_scoring'] is True and seal['all_rows_kept'] is True,'prediction/pre-score gate')
    for key,name in [('predictions_sha256','PREDICTIONS.jsonl.gz'),('protocol_sha256','PROTOCOL.json'),
        ('observation_seal_sha256','run/OBSERVATION_SEAL.json'),('independent_bindings_sha256','INDEPENDENT_BINDINGS.json')]:
        checks(seal[key]==sha(folder/name),'prediction/binding '+name)
    for name,h in protocol['source_sha256'].items():checks(sha(code/name)==h,'frozen source '+name)
    for name,h in protocol['immutable_inputs_sha256'].items():checks(sha(name)==h,'immutable input '+Path(name).name)
    for info in protocol['model_descriptors'].values():checks(sha(Path(source_root)/info['path'])==info['sha256'],'frozen model descriptor')
    data=read(folder/'inputs_v1/INPUTS.json');frames=data['frames'];inputs=read(folder/'inputs_v1/INPUT_SEAL.json')
    if 'inputs_sha256' in inputs:checks(inputs['inputs_sha256']==sha(folder/'inputs_v1/INPUTS.json'),'input/full byte binding')
    rows0=rows(folder/'PREDICTIONS.jsonl.gz');old=rows(prior/'run_v2/PREDICTIONS.jsonl.gz')
    ids=[f['id'] for f in frames];checks(len(ids)==len(set(ids))==238,'inventory/unique238')
    checks([r['id'] for r in rows0]==ids==seal['ordered_ids']==protocol['ordered_ids'],'inventory/ordered238')
    checks([r['partition'] for r in rows0]==['CAL']*46+['TEST']*49+['REMAINDER']*143,'inventory/fixed parts')
    bindings=read(folder/'INDEPENDENT_BINDINGS.json');checks(bindings['ordered_ids']==ids,'inventory/binding ids')
    checks([r['id'] for r in bindings['records']]==ids,'inventory/all binding rows')
    source_summary=point_source_audit(folder,prior,source_root,frames,checks)
    source_summary['input_seal_audit']=input_seal_audit(folder,prior,source_root,frames,checks)
    obsseal=read(folder/'run/OBSERVATION_SEAL.json');checks(obsseal['rows']==238 and obsseal['branches']==476,'observation/all476')
    checks(obsseal['protocol_sha256']==seal['protocol_sha256'] and obsseal['GT_inputs'] is False,'observation/protocol gate')
    checks(len(obsseal['observations_sha256'])==476,'observation/complete saved inventory')
    features=[]
    for i,(r,p,f,bound) in enumerate(zip(rows0,old,frames,bindings['records'])):
        fid=r['id'];checks(set(r['predictions'])==set(ARMS),fid+'/all8')
        checks(r['partition']==f['partition'] and r['session']==f['session'] and r['cluster_id']==f['cluster_id'],fid+'/same cohort metadata')
        for arm in ('R0','R1','N0','N1'):checks(r['predictions'][arm]==p['predictions'][arm],fid+'/'+arm+'/old entry exact')
        for branch,base,after in [('N3_ONLY','N3_F','N3_BI'),('SUBPIX_ONLY','SUBPIX_F','SUBPIX_BI')]:
            key=fid+'/'+branch;spec=bound['branches'][branch];op=_file(spec['observation'],folder,checks,key+'/observation')
            sp=_file(spec['solution'],folder,checks,key+'/pose');o,s=read(op),read(sp);bundle=o['result'];obs=bundle['observation'];sol=s['result'];a=f['arms'][branch]
            checks(sha(op)==obsseal['observations_sha256'][op.name],key+'/observation inventory SHA')
            checks(o['id']==s['id']==fid and o['branch']==s['branch']==branch,key+'/own branch identity')
            checks(o['result_sha256']==digest(bundle) and s['result_sha256']==digest(sol),key+'/payload digests')
            checks(s['observation_binding']==o['binding'],key+'/solver uses same observation')
            checks(o['binding']==digest(dict(frame=f,branch=branch,protocol=seal['protocol_sha256'])),key+'/exact own input binding')
            checks(obs['original_raw_points']==a['native_points'] and obs['original_coarse_pose']==a['coarse_pose'],key+'/own points and own F')
            geometry=obs['initial_geometry'];coarse=a['coarse_pose'];e=np.asarray(coarse['cf_extents'],float)/2;x,y,z=e
            expected=[[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],[-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]]
            checks(np.array_equal(geometry['native_xyz'],expected) and geometry['cf_extents']==coarse['cf_extents'],key+'/native geometry and order')
            checks(geometry['K']==f['K'] and geometry['R_cf']==coarse['R_cf'] and geometry['centroid']==coarse['centroid'],key+'/own coarse geometry')
            Q=np.asarray(coarse['R_cf']).T@coarse['R_physical'];checks(np.allclose(geometry['physical_registration_Q'],Q,atol=1e-9,rtol=0),key+'/physical registration')
            hidden={k for k,v in enumerate(obs['states']) if v=='S'};st=obs['state_observation'];mask=obs['mask_audit'];correction=obs['visible_correction'];meas=obs['frozen_visible_measurements']
            checks(not hidden.intersection(st['rgb_queried_ids']),key+'/S no RGB state query')
            checks(not hidden.intersection(mask['prompt_native_ids']),key+'/S no mask prompt')
            checks(not hidden.intersection(correction.get('queried_native_ids',[])),key+'/S no correction query')
            checks(not hidden.intersection(correction.get('recovery_queried_native_ids',[])),key+'/S no recovery query')
            states=obs['states'];expectedids=[k for k,v in enumerate(states) if v in ('B','I')]
            checks([m['native_corner'] for m in meas]==expectedids==sol['input_native_ids'],key+'/only exact BI enters solver')
            checks(sol['observed_states_preserved']==states,key+'/states preserved')
            checks(sol['frozen_visible_measurements']==meas,key+'/measurement covariance source support preserved')
            checks(all(m.get('image_measurement') is True and m.get('measurement_is_pose_projection') is False for m in meas),key+'/image coordinates not model projection')
            checks(sol['solver_configuration']==protocol['solver_configuration'],key+'/same frozen solver')
            mb=bundle['mask_binding'];checks(sha(op.parent.parent/mb['path'])==mb['sha256'],key+'/own mask bytes')
            operations=bundle['operations'];checks(operations['prediction_source']==branch and operations['record_id']==fid and operations['state_and_S_recomputed_for_own_input'] is True,key+'/own branch observation provenance')
            calls=bundle['actual_calls'];checks(calls['n3_forward_calls']==0 and calls['subpix_calls']==0 and calls['U_display_builder_calls']==0 and calls['OT_builder_calls']==0,key+'/unused calls zero')
            b=r['predictions'][base];checks(b['actual_pose']==coarse and b['accepted'] is None and b['fallback'] is None,key+'/own F baseline exact')
            v=r['predictions'][after];checks(v['actual_pose']==sol['actual_pose'] and v['accepted']==sol['accepted'] and v['fallback']==sol['fallback'],key+'/own BI solution exact')
            if sol['actual_pose']['available']:
                outputQ=np.asarray(sol['actual_pose']['R_cf']).T@sol['actual_pose']['R_physical'];checks(np.allclose(outputQ,Q,atol=1e-9,rtol=0),key+'/final registration preserved')
            features.append(dict(id=fid,branch=branch,states=states,BI_ids=expectedids,S_ids=sorted(hidden),accepted=sol['accepted'],fallback=sol['fallback'],mask_sha256=mb['sha256'],observation_sha256=sha(op),pose_sha256=sha(sp)))
    if checks.failures:raise ValueError('Prediction-only gate failed: '+repr(checks.failures[:3]))
    return rows0,protocol,source_summary,features


def run(*,folder,prior,source_root,output,expected_prediction_seal_sha256,expected_score_seal_sha256):
    folder,prior,source_root,output=map(Path,(folder,prior,source_root,output));started=time.monotonic();cpu=time.process_time()
    if output.exists():raise FileExistsError('Every independent attempt must use a new output folder')
    if sha(folder/'PREDICTION_SEAL.json')!=expected_prediction_seal_sha256:raise ValueError('Exact complete prediction seal required')
    if sha(folder/'score/SCORE_SEAL.json')!=expected_score_seal_sha256:raise ValueError('Exact complete score seal required')
    checks=Checks();predictions,protocol,point_sources,observations=prediction_audit(folder,prior,source_root,checks)
    ss=read(folder/'score/SCORE_SEAL.json');checks(ss['rows']==238 and ss['arm_rows']==1904 and ss['all_rows_kept'] is True,'score/all1904')
    checks(ss['prediction_seal_sha256']==expected_prediction_seal_sha256,'score/same prediction seal')
    for name,h in ss['artifacts'].items():checks(sha(folder/'score'/name)==h,'score/artifact '+name)
    checks(ss['source_sha256']['score.py']==sha(Path(__file__).with_name('score.py')),'score/frozen code hash')
    if checks.failures:raise ValueError('Prediction/score hash gate failed: '+repr(checks.failures[:3]))
    # Every method's saved pose and observation has passed before numeric references.
    refs=fixed_references(source_root,prior/'SCORING_REFERENCE_MANIFEST_V2.json',predictions,checks)
    saved=read(folder/'score/SCORED_ROWS.json');scored=saved['rows'];checks(saved['arm_rows']==1904,'score/exact1904')
    checks([r['id'] for r in scored]==[r['id'] for r in predictions],'score/ordered238')
    original=read(prior/'score_v2/SCORED_ROWS.json')['rows'];independent=[]
    for p,s,o in zip(predictions,scored,original):
        fid=p['id'];checks(set(s['metrics'])==set(ARMS),fid+'/all8 scored')
        direct={a:metric(p['predictions'][a]['actual_pose'],refs[p['partition']][fid]) for a in ARMS}
        checks.equal(direct,s['metrics'],fid+'/direct original physical/C2 metrics')
        statuses={a:{k:p['predictions'][a].get(k) for k in ('accepted','fallback','status','reason')} for a in ARMS}
        checks.equal(statuses,s['prediction_status'],fid+'/prediction statuses')
        for a in ('R0','R1','N0','N1'):checks(s['metrics'][a]==o['metrics'][a],fid+'/'+a+'/old score exact')
        # Saved floats are normative for exact tie and boundary recount bins,
        # after independent arithmetic is confirmed within1e-9 above.
        independent.append(dict(id=fid,partition=p['partition'],session=p['session'],cluster_id=p['cluster_id'],
                                metrics=s['metrics'],prediction_status=statuses))
    csvrows=list(csv.DictReader((folder/'score/ROW_METRICS.csv').open()));csvmap={(r['id'],r['arm']):r for r in csvrows}
    checks(len(csvrows)==len(csvmap)==1904,'CSV/unique complete1904')
    for r in independent:
        for a in ARMS:
            v=csvmap[(r['id'],a)];m=r['metrics'][a]
            checks((v['partition'],v['session'],v['cluster_id'])==(r['partition'],r['session'],r['cluster_id']),r['id']+'/'+a+'/CSV identity')
            for k in ERRORS:checks.equal(m[k],None if v[k]=='' else float(v[k]),r['id']+'/'+a+'/CSV '+k)
            for k in ('pose_available','success_le5','success_strict_lt5','reference_rotation_available'):
                checks(m[k] is (None if v[k]=='' else v[k]=='True'),r['id']+'/'+a+'/CSV '+k)
            for k in ('accepted','fallback'):
                checks(r['prediction_status'][a][k] is (None if v[k]=='' else v[k]=='True'),r['id']+'/'+a+'/CSV '+k)
    result=read(folder/'score/RESULTS.json');checks(result['no_method_selection'] is True and result['default_changed'] is False and result['all_rows_kept'] is True,'report/no method/default change or row loss')
    contrasts=[tuple(c) for c in protocol['contrasts']];report={};expected_changes=[]
    for pop in ('CAL','TEST','REMAINDER','PRIMARY'):
        rr=[r for r in independent if (r['partition']!='CAL' if pop=='PRIMARY' else r['partition']==pop)]
        checks(len(rr)==PARTITIONS[pop],pop+'/fixed denominator')
        r=recount(rr,contrasts);report[pop]=r
        checks.equal(r,result['populations'][pop],pop+'/independent pooled and paired recount')
        checks(set(result['populations'][pop]['comparisons'])=={a+'-'+b for a,b in contrasts},pop+'/only prespecified contrasts')
        for name,v in r['comparisons'].items():
            for cat in ('failure_to_success','success_to_failure'):
                expected_changes.extend(dict(population=pop,contrast=name,category=cat,id=fid) for fid in v['transition']['ids'][cat])
    changed=list(csv.DictReader((folder/'score/CHANGED_CASES.csv').open()));checks(changed==expected_changes,'CSV/all paired changed cases exact order')
    checks(sha(folder/'PREDICTION_SEAL.json')==expected_prediction_seal_sha256 and sha(folder/'score/SCORE_SEAL.json')==expected_score_seal_sha256,'final/seals unchanged')
    output.mkdir(parents=True)
    write(output/'OBSERVATION_RECOUNT.json',dict(rows=observations,point_source_audit=point_sources))
    write(output/'RECOUNT.json',report)
    from scripts.research.pallet_guarded_bi_final_v1 import independent as MATH_HELPER
    receipt=dict(schema='INDEPENDENT_SINGLE_COMPONENT_FULL238_EIGHT_ARM_CHECK_V1',checks=checks.count,
        failure_count=len(checks.failures),failures=checks.failures,rows=238,arm_rows=1904,new_own_observation_and_pose_rows=476,
        arms=list(ARMS),partitions=PARTITIONS,old_four_arms_and_scores_exact=True,point_source_audit=point_sources,
        prediction_seal_sha256=expected_prediction_seal_sha256,score_seal_sha256=expected_score_seal_sha256,
        protocol_sha256=sha(folder/'PROTOCOL.json'),input_seal_sha256=sha(folder/'inputs_v1/INPUT_SEAL.json'),
        source_sha256=dict(independent=sha(__file__),pure_physical_metric_helper=sha(MATH_HELPER.__file__)),
        direct_original_reference_metric_recomputed=True,original_center_C2_and_ADDsym_definition_unchanged=True,
        scorer_or_runtime_or_model_or_optimizer_imports=0,new_model_image_PnP_or_fit_calls=0,
        prediction_and_observation_gate_before_reference_numeric_reads=True,no_selection_or_threshold_tuning=True,
        old_development_exposure=True,new_independent_test=False,
        verification_limit='Saved hashes and independent arithmetic verify artifact/source consistency, not access to unrecorded files or independent physical ground truth.',
        resources=dict(wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu,processes=1))
    write(output/'RECEIPT.json',receipt)
    write(output/'SEAL.json',dict(artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    if checks.failures:raise ValueError('Independent differences found; full attempt preserved')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('folder','prior','source-root','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--prediction-seal-sha256',required=True);p.add_argument('--score-seal-sha256',required=True)
    a=p.parse_args();print(json.dumps(run(folder=a.folder,prior=a.prior,source_root=a.source_root,output=a.output,
        expected_prediction_seal_sha256=a.prediction_seal_sha256,expected_score_seal_sha256=a.score_seal_sha256),indent=2))
