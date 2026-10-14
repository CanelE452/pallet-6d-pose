"""Read-only CAL fit/input audit; no model fitting or pose/RGB execution."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'.git').exists())
sys.path.insert(0,str(ROOT))
import json,gzip,hashlib
import numpy as np
F=ROOT/'_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix'
O=F/'independent_review'
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt') as z:return [json.loads(x) for x in z]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def physics(spec):
    p=json.loads(json.dumps(spec));vk=('native_corner','xy','covariance_px2','protection_px','fit_covariance_px2','fit_reliability_weight')
    dk=('native_corner','native_id','state','shape','center_xy','ellipse_center','half_axes_px','half_axes','angle_deg','angle','bounds_xyxy','box_xyxy','halfplanes','normal','offset','outside_canvas_side','outside_margin_px','canvas_hw')
    p['visible']=[{k:v[k] for k in vk if k in v} for v in p['visible']];p['domains']=[{k:d[k] for k in dk if k in d} for d in p['domains']]
    return digest(p)
def run():
    dst=O/'CAL_INTERVAL_MODEL_INPUT_AUDIT_V4_2.json'
    if dst.exists():raise FileExistsError('Preserve prior audit')
    modeldir=F/'truncation_interval_model_CAL_v4_2';v=F/'truncation_pose_validation_v4_2'
    seal=read(modeldir/'MODEL_SEAL.json');model=read(modeldir/'MODEL.json');report=read(modeldir/'CAL_LOGO_REPORT.json');protocol=read(v/'PROTOCOL.json');inputs=read(v/'INPUT_SEAL.json')
    tests=[]
    def ck(name,passed,context=''):tests.append({'check':name,'passed':bool(passed),'context':context})
    ck('CAL_seal',seal['fit_partition']=='CAL' and not seal['TEST_files_opened'])
    ck('sealed_model_report',sha(modeldir/'MODEL.json')==seal['MODEL_sha256'] and sha(modeldir/'CAL_LOGO_REPORT.json')==seal['LOGO_sha256'])
    ck('sealed_prepose_input',inputs['before_pose_solves'] and inputs['before_CAL_pose_scoring'] and not inputs['TEST_files_opened'] and sha(v/'SOLVER_INPUTS.jsonl.gz')==inputs['inputs_sha256'] and sha(v/'PROTOCOL.json')==inputs['protocol_sha256'])
    for p,h in protocol['source_sha256'].items():
        ck('current_source',sha(ROOT/p)==h,p);ck('executed_snapshot',sha(v/'code_snapshot'/p)==h,p)
    for b in protocol['input_bindings'].values():ck('input_binding',sha(ROOT/b['path'])==b['sha256'],b['path'])
    for b in protocol['preserved_model_bindings']:ck('preserved_model',sha(ROOT/b['path'])==b['sha256'],b['path'])
    for p,h in protocol['protected_user_sha256'].items():ck('protected6',sha(ROOT/p)==h,p)
    if not all(t['passed'] for t in tests):raise ValueError('Binding check failed before CAL coordinate read')
    # CAL-only target read is explicitly for raw localization and fixed 3D native
    # compatibility, never pose-error scoring or TEST choice.
    labels=read(ROOT/seal['CAL_annotations']['path']);cache=rows(ROOT/seal['CAL_cache']['path'])
    ck('isolated_CAL46',labels['TEST_rows']==0 and labels['FINAL_rows']==0 and len(cache)==46 and set(labels['rows'])=={r['id'] for r in cache})
    points=[];compatible=[];incompatible=[];byid={r['id']:r for r in cache};gated_S=0
    for row in cache:
        ck('CAL_partition',row['partition']=='CAL',row['id'])
        rt=row['runtime'];g=rt['initial_geometry'];label=labels['rows'][row['id']];truth=label['target']['truth'];mapping=label['metadata']['native_to_reference_corner_ids']
        xyz=np.asarray(g['native_xyz']);Q=np.asarray(g['physical_registration_Q']);dims=np.asarray(truth['body_xyz'])/2
        a,b,c=dims;box=np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])
        mapped=box[np.asarray(mapping)]@(np.asarray(truth['body_R']).T@np.asarray(truth['R']))
        gaps=[]
        for phase in range(truth.get('order',1)):
            theta=phase*2*np.pi/truth.get('order',1);co,si=np.cos(theta),np.sin(theta);S=np.array([[co,0,si],[0,1,0],[-si,0,co]])
            gaps.append(float(np.linalg.norm(xyz@Q@S.T-mapped,axis=1).max()))
        comp=min(gaps)<1e-6
        (compatible if comp else incompatible).append(row['id'])
        hidden=g['initial_self_hidden'];nonS=[k for k in range(8) if not hidden[k]]
        raw=np.array([rt['original_raw_points'][k] for k in nonS]);valid=np.isfinite(raw).all(1)&~np.all(raw==-1,axis=1)
        scale=max(float(np.linalg.norm(np.ptp(raw[valid],axis=0))) if valid.any() else 0.,16.)
        ck('nonS_scene_scale',scale==rt['ot_domains']['nonself_prediction_scale_px'],row['id'])
        for k in range(8):
            if hidden[k]:gated_S+=1;continue
            if not comp or label['metadata']['human_states_native'][k] not in ('DIRECT_VISIBLE','EXTERNAL_OCCLUDED') or not label['target']['valid'][mapping[k]]:continue
            q=np.asarray(rt['original_raw_points'][k]);target=np.asarray(label['target']['gt'][mapping[k]])
            ck('eligible_raw_finite',np.isfinite(q).all() and not np.all(q==-1),row['id']+'/'+str(k))
            error=float(np.linalg.norm(q-target));points.append({'id':row['id'],'session':row['session'],'native_corner':k,'human_state':label['metadata']['human_states_native'][k],'raw_xy':q.tolist(),'target_xy':target.tolist(),'raw_error_px':error,'nonself_prediction_scale_px':scale,'normalized_error':error/scale})
    ck('population264_44',len(points)==264 and compatible==model['compatible_ids'] and incompatible==model['excluded_native_incompatible_ids'] and len(compatible)==44)
    maximum=max(points,key=lambda x:x['normalized_error']);ck('untrimmed_full_max',maximum['normalized_error']==model['radius_ratio'] and maximum['id']=='wood_184309:000372' and maximum['native_corner']==0 and maximum['raw_error_px']==model['max_error_point']['raw_error_px'])
    ck('model_digest',digest({k:v for k,v in model.items() if k!='model_sha256'})==model['model_sha256'])
    coverage={}
    for session,fold in report['fold_models'].items():
        train=[p for p in points if p['session']!=session];held=[p for p in points if p['session']==session];m=max(p['normalized_error'] for p in train)
        ck('fold_training_ids_exclude_session',fold['fit_ids']==[r['id'] for r in cache if r['session']!=session],session)
        ck('fold_max_exact',m==fold['radius_ratio'] and len(train)==fold['eligible_point_count'],session)
        ck('fold_digest',digest({k:x for k,x in fold.items() if k!='model_sha256'})==fold['model_sha256'],session)
        count=sum(p['normalized_error']<=m+1e-12 for p in held);pub=report['coverage']['folds'][session]
        ck('heldout_coverage_exact',count==pub['covered_points'] and len(held)==pub['eligible_points'],session)
        coverage[session]={'training_points':len(train),'radius_ratio':m,'heldout_points':len(held),'covered':count,'missed':len(held)-count}
    published={(p['id'],p['native_corner']):p for p in report['coverage']['full']['records']}
    for p in points:ck('actual_raw_annotation_error_matches_export',p=={k:v for k,v in published[(p['id'],p['native_corner'])].items() if k!='covered'},p['id']+'/'+str(p['native_corner']))
    tasks=rows(v/'SOLVER_INPUTS.jsonl.gz');ck('all46_inputs',len(tasks)==46 and [t['id'] for t in tasks]==[r['id'] for r in cache])
    baseline={r['id']:r for r in rows(F/'bias_selection_v4_1/CANDIDATE_PREDICTIONS.jsonl.gz') if r['bias_strength']==1.}
    # Source old baseline folder is read from the protocol, not guessed.
    if not baseline:raise ValueError('Missing frozen baseline')
    old_inputs={r['solver_input_sha256']:r for r in rows(F/'bias_selection_v4_1/CANDIDATE_INPUTS.jsonl.gz')}
    removed=[];reuses=0
    for task in tasks:
        rt=byid[task['id']]['runtime'];new=task['solver_input'];old=old_inputs[task['baseline_solver_input_sha256']]['solver_input'];fold=report['fold_models'][byid[task['id']]['session']]
        ck('baseline_input_bound',baseline[task['id']]['solver_input_sha256']==task['baseline_solver_input_sha256'],task['id'])
        ck('same_visible_config_pose_K_xyz',all(new[k]==old[k] for k in ('visible','config','coarse_pose','K','xyz','image_hw')),task['id'])
        ck('same_lambda_beta',new['config']['prior_weight']==1. and new['config']['bias_strength']==1.,task['id'])
        ck('math_hash_exact',physics(new)==task['solver_input_sha256'],task['id'])
        exact=physics(new)==physics(old);reuses+=exact
        ck('reuse_hash_boolean',exact==task['exact_baseline_cache_reusable'],task['id'])
        for k,(a,b) in enumerate(zip(old['states'],new['states'])):
            if a!='T':ck('only_conditional_T_changes',a==b,task['id']+'/'+str(k));continue
            scale=rt['ot_domains']['nonself_prediction_scale_px'];radius=fold['radius_ratio']*scale;q=np.asarray(rt['original_raw_points'][k]);h,w=rt['initial_geometry']['image_hw'];distance=float(np.linalg.norm(q-np.clip(q,[0,0],[w,h])))
            retain=bool(rt['state_observation']['records'][k].get('physical_truncation_certified')) or distance>radius
            ck('circle_closed_canvas_gate',b==('T' if retain else 'U'),task['id']+'/'+str(k))
            if b=='U':removed.append({'id':task['id'],'native_corner':k,'distance':distance,'radius':radius})
        ck('retained_O_numeric',[(d['native_corner'],d.get('shape'),d.get('center_xy'),d.get('half_axes_px'),d.get('halfplanes')) for d in new['domains'] if d['state']=='O']==[(d['native_corner'],d.get('shape'),d.get('center_xy'),d.get('half_axes_px'),d.get('halfplanes')) for d in old['domains'] if d['state']=='O'],task['id'])
        ck('all_declared_OT_present',{d['native_corner'] for d in new['domains']}=={k for k,s in enumerate(new['states']) if s in ('O','T')},task['id'])
    ck('39_reuse_7_new',reuses==39 and len(removed)==13 and len({x['id'] for x in removed})==7)
    out={'schema':'INDEPENDENT_CAL_INTERVAL_FIT_INPUT_AUDIT_V4_2','passed':all(x['passed'] for x in tests),'total_checks':len(tests),'failures':[x for x in tests if not x['passed']],'CAL_frames':46,'compatible_CAL':44,'eligible_points':264,'gated_initialS_corners':gated_S,'untrimmed_max_point':maximum,'fold_coverage':coverage,'demoted_T13':removed,'exact_baseline_reuses':reuses,'new_solver_tasks':46-reuses,'no_new_fits_pose_solves_RGB_TEST_access':True,'CAL_target_read_scope':'CAL raw localization error and fixed 3D native compatibility only; no pose accuracy scoring before prediction seal','physical_T_certificate':False,'future_coverage_guarantee':False,'post_TEST_exploratory':True,'source_sha256':sha(Path(__file__)),'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [modeldir/'MODEL.json',modeldir/'MODEL_SEAL.json',modeldir/'CAL_LOGO_REPORT.json',v/'PROTOCOL.json',v/'INPUT_SEAL.json',v/'SOLVER_INPUTS.jsonl.gz']}}
    dst.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('passed','total_checks','failures','eligible_points','exact_baseline_reuses','new_solver_tasks','fold_coverage')}))
if __name__=='__main__':run()
