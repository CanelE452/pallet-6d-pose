"""V156 saved-only visibility benchmark and conditional reference diagnostics.

Preparation reads code/JSON descriptors only. All labels, scores, annotation
bodies are opened only after separate root entry and completed prediction seal.
"""
from pathlib import Path
from types import SimpleNamespace, ModuleType
from collections import Counter
import argparse,ast,csv,hashlib,json,math,sys
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_DIAGNOSTIC_POLICY_BEFORE_CURRENT_BODIES.json'
POLICY_SHA='d8b36d6e8cfb560c59170238e085e9444384468087d201737cb983a77ed5297d'
PRIOR=P/'query_visibility_public_native_frozen_current_benchmark_20261024_v149'
PRIOR_CODE=PRIOR/'saved_evaluation_preparation/evaluate_saved_benchmark.py'
PRIOR_CODE_SHA='34ddd4eb8d7285d398c9d016f51a90c6f964073a9ac24547936a4b49976c5a8d'
OLD=P/'query_visibility_training_duration_control_20261023_v124'
ADAPTER=OLD/'current_evaluation_preparation/frozen_head_current.py'
CORE=P/'query_visibility_matched_query_scalar_20261023_v94/current_evaluation_preparation/evaluate_saved_only.py'
DECODER=CORE.parent/'infer_same_query_once.py'
REPORT=P/'query_visibility_spatial_target_query_features_20261023_v111/current_evaluation_preparation/evaluate_current_saved.py'
METRIC=P/'query_visibility_cue_occlusion_20261023_v72/contract/saved_evaluate_current_scores.py'
PRED=V/'frozen_four_head_current_benchmark_v1/PREDICTION_SEAL.json'
PRED_SHA='255bca46874f95fe7837f7140024c456541e4d53c2a470673b134f2e0c815c4f'
PROTOCOL=H/'PROTOCOL.json';GENERATED=H/'GENERATED_CHECKS.json';AUTHORITY=H/'ANNOTATION_AUTHORITY_AUDIT.json'
OUT=V/'saved_seven_current_evaluation_v1';REPO=Path('/home/minjae/Documents/github/pallet-pose')
MODELS=('v61','legacy','C16','C35','C36','C37','C38');NEW_MODELS=MODELS[3:];BASELINES=MODELS[:3]
FIELDS=('id','native_corner','query_x','query_y','roi_side','valid','raw_query_aux_hidden_logit','raw_p_visible_FP32','authoritative_decision')
META=('query_available','prior_valid','valid','context_available','local_central4_available','context_Q_inside','query_xy','side_L','original_query_xy','original_side_L','id','native_corner')
def require(x,why):
    if not x:raise ValueError(why)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):
    require('_v112'not in b['path'].lower(),'Forbidden phase path')
    require(sha(b['path'])==b['sha256'],'Changed binding: '+b['path']);return Path(b['path'])
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def functions(path,names,ns):
    nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names]
    require({n.name for n in nodes}==set(names),'Pure function coverage')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::pure','exec'),ns)
def byte_equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def pure():
    # Unchanged historical functions; NEW/OLD are only their two input aliases.
    ns=dict(np=np,math=math,Counter=Counter,c=SimpleNamespace(require=require),Path=Path,csv=csv,require=require,
      NEW='C35',OLD='C36',NEW_GEO='UNREPORTED_C35_GEOMETRY',OLD_GEO='UNREPORTED_C36_GEOMETRY',MODELS=MODELS,MINIMAL_SCORE_FIELDS=FIELDS)
    functions(CORE,{'readcsv','writecsv','key','boolean','numeric_same','score_index','response','join_all','calculate_all','historical_replay'},ns)
    functions(DECODER,{'decode_rows'},ns);functions(ADAPTER,{'enrich_minimal_baseline'},ns)
    core=SimpleNamespace(**ns)
    base=ModuleType('original_visibility_metric');base.__dict__.update(math=math,Counter=Counter,D='DIRECT_VISIBLE',E='EXTERNAL_OCCLUDED',S='SELF_OCCLUDED',
      TASKS={'D_vs_E':['DIRECT_VISIBLE','EXTERNAL_OCCLUDED'],'D_vs_S':['DIRECT_VISIBLE','SELF_OCCLUDED'],'D_vs_ES':['DIRECT_VISIBLE','EXTERNAL_OCCLUDED','SELF_OCCLUDED']},GROUPS=['WOOD_PRIMARY105','SQUARE118'])
    functions(METRIC,{'yes','ratio','call','calculate'},base.__dict__)
    ex=dict(np=np,math=math,Counter=Counter,c=SimpleNamespace(NEW='C35',OLD='C36',NEW_GEO='UNREPORTED_C35_GEOMETRY',OLD_GEO='UNREPORTED_C36_GEOMETRY',MODELS=MODELS))
    functions(REPORT,{'error_bins','matrices'},ex)
    helpers=dict(np=np,math=math,csv=csv,Path=Path,read=read,check=check,require=require,byte_equal=byte_equal,META=META,FIELDS=FIELDS)
    functions(PRIOR_CODE,{'load_packet','records','exact_csv_rows'},helpers)
    return core,base,SimpleNamespace(**ex),SimpleNamespace(**helpers)
def metric(y,pred,mask):
    cm=[[int(np.sum(mask&(y==t)&(pred==s)))for s in(0,1,-1)]for t in(0,1)]
    def ratio(n,d):return n/d if d else None
    a=np.array(cm);return dict(CM_rows_far_near_columns_far_near_U=json.dumps(cm),known_N=int(a.sum()),
      far_recall=ratio(a[0,0],a[0].sum()),near_recall=ratio(a[1,1],a[1].sum()),
      far_precision=ratio(a[0,0],a[:,0].sum()),near_precision=ratio(a[1,1],a[:,1].sum()),
      U_known=int(a[:,2].sum()),all_original_rows=int(mask.sum()),unknown_reference_truth=int(np.sum(mask&(y<0))))
def target(error):return 1 if error<=3 else 0 if error>20 else -1
def generated():
    c,b,_,_=pure();require(c.response(dict(valid=False,z=-100,geo=True))=='OTHER_UNRESOLVED','U first')
    require(c.response(dict(valid=True,z=0,geo=False))=='ABSENT_NONVISIBLE','raw0')
    rr=[dict(id='I',human_state=b.D,present_at_fixed_0_5='True',legacy_stored_state='U'),dict(id='I',human_state=b.E,present_at_fixed_0_5='False',legacy_stored_state='S')]
    x=b.calculate(rr,'legacy','D_vs_E',{},.5);require(x['N']==2 and x['U_visible']==1 and x['out_of_task_S_occluded']==1,'Legacy unresolved')
    require([target(x)for x in(3,3.1,20,20.1)]==[1,-1,-1,0],'Fixed quality bands')
    z=metric(np.array([0,1,-1]),np.array([-1,1,0]),np.ones(3,bool));require(z['known_N']==2 and z['U_known']==1 and z['unknown_reference_truth']==1,'Quality full denominator')
    write(GENERATED,dict(completed=True,PASS=True,code=bind(__file__),invented_only=True,
      raw0_Ufirst_legacyS_U_preserved=True,secondary_quality_bands_and_unknown_denominators=True,current_payload_body_or_hash_reads=0))
def prepare():
    require(sha(POLICY)==POLICY_SHA and sha(PRED)==PRED_SHA and sha(PRIOR_CODE)==PRIOR_CODE_SHA,'Fixed policy/prediction/code')
    old=read(PRIOR/'saved_evaluation_preparation/PROTOCOL.json')
    for b in old['operator_codes']:check(b)
    generated();original=read(P/'evaluation_preparation_20261022/SOURCE_BINDINGS.json')
    oldseal=read(PRIOR/'saved_current_benchmark_evaluation_v1/EVALUATION_SEAL.json')
    require(oldseal['completed']and oldseal['PASS'],'Completed historical evaluation metadata')
    historical={Path(b['path']).stem:b for b in oldseal['outputs']}
    keys=('operator_codes','C16_inference_seal','C16_scores','C16_raw','historical_C16_metrics','prior','truth','manual','historical_original')
    p=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),prediction_seal=bind(PRED),
      prior_code=bind(PRIOR_CODE),prior_protocol=bind(PRIOR/'saved_evaluation_preparation/PROTOCOL.json'),generated=bind(GENERATED),
      annotation_authority=bind(AUTHORITY),new_primary_real_quality_gold='ALL1784_UNKNOWN_UNCERTIFIED',
      human_sidecar=original['human_sidecar'],review_manifest=original['review_manifest'],
      original_annotation_repo=str(REPO),annotation_definition_scope=bind(V/'ROOT_ANNOTATION_DEFINITION_DOCUMENT_SCOPE_BEFORE_LABEL_PAYLOADS.json'),
      historical_three_tables={k:historical[k]for k in('METRICS','CONFUSION','HUMAN_SELF_BINARY_RESPONSE','MANUAL_NEAR_D','COORDINATE_ERROR_BINS','D_VS_EXTERNAL_E_2X3','ALL1784_MODEL_INVENTORY','ALL1784_NATIVE_LEDGER')},
      models=list(MODELS),output=str(OUT),**{k:old[k]for k in keys},
      conditional_quality='SECONDARY existing-fixed-reference coordinate-deviation diagnostic ONLY: DIRECT+explicit manual source+raw2D exact traceback; near<=3/far>20; middle/occluded/amodal/uncertified unknown. Stored evaluation native mapping remains unverified physical authority.',
      original_visibility='Unchanged all1784/paired1237/same-slot1248/manualnear376; all7 fixed models; original raw0/U45; S/OOF/unknown outside external binary. No quality/error primary filter.',
      current_payload_body_or_hash_reads_during_prepare=0)
    write(PROTOCOL,p)
    write(H/'ROOT_ENTRY_TEMPLATE.json',dict(approved_saved_evaluation=False,code=p['code'],policy=p['policy'],protocol=bind(PROTOCOL),prediction_seal=p['prediction_seal'],
      original_annotation_row_traceback=True,new_primary_quality_gold_unknown=True,quality_diagnostic_only=True,
      private_fit=False,threshold_or_model_choice=False,RGB_read=False,encoder_forward=False,V112=False,PnP=False,q_L_correction=False))
    print(json.dumps(dict(prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),template=bind(H/'ROOT_ENTRY_TEMPLATE.json'))))
def transitions(core,base,truth,call):
    out=[];pairs=[(a,b)for b in NEW_MODELS for a in BASELINES]+[('C35','C36'),('C37','C38'),('C35','C37'),('C36','C38')]
    for cohort in('paired_same_query_RGB','same_slot_pipeline'):
      for group in base.GROUPS:
       for cls in('ALL_DE',base.D,base.E):
        rr=[r for r in truth if r['group']==group and base.yes(r,cohort+'_D_vs_E')and r['human_state']in(base.D,base.E)and(cls=='ALL_DE'or r['human_state']==cls)]
        for old,new in pairs:
         counts=Counter()
         for r in rr:
          a,b=call(r,old),call(r,new);gt='VISIBLE'if r['human_state']==base.D else'ABSENT_NONVISIBLE'
          if b not in('VISIBLE','ABSENT_NONVISIBLE'):counts['new_unresolved']+=1
          elif a not in('VISIBLE','ABSENT_NONVISIBLE'):counts['baseline_unresolved_new_correct'if b==gt else'baseline_unresolved_new_wrong']+=1
          else:counts['both_correct'if a==gt and b==gt else'baseline_correct_new_wrong'if a==gt else'baseline_wrong_new_correct'if b==gt else'both_wrong']+=1
         keys=('both_correct','baseline_correct_new_wrong','baseline_wrong_new_correct','both_wrong','baseline_unresolved_new_correct','baseline_unresolved_new_wrong','new_unresolved')
         require(sum(counts.values())==len(rr),'Transition denominator')
         out.append(dict(cohort=cohort,group=group,task='D_vs_E',truth_class=cls,baseline=old,new_model=new,N=len(rr),**{k:counts[k]for k in keys}))
    return out
def quality_traceback(core,truth,packs,p):
    # Evaluation-only labels. No images, pose estimation or new semantic labels.
    sidecar=read(check(p['human_sidecar']));review=read(check(p['review_manifest']))
    humans={(r['population'],r['frame_id'],r['image_sha256'],int(r['corner_id'])):r for r in sidecar['rows']}
    cases={(r['population'],r['frame_id'],r['image']['sha256']):r for r in review['cases']}
    cache={};bindings=[];ledger=[];secondary=np.full(1784,-1,np.int8)
    for i,t in enumerate(truth):
        d=dict(id=t['id'],native_corner=int(t['native_corner']),group=t['group'],human_state=t['human_state'],
          mapping_status=t['mapping_status'],new_primary_quality_gold_target='UNKNOWN_UNCERTIFIED',
          native_physical_mapping_authority='UNVERIFIED_FROZEN_EVALUATION_BRANCH',
          secondary_fixed_reference_target=-1,secondary_error_px=None,raw_annotation_traceback='NO_KNOWN_REFERENCE',
          annotation_coordinate_source='',reference_corner_id=t.get('reference_corner_id',''),
          original_query_valid=core.boolean(t['query_valid']),primary_visibility_filter_applied=False)
        fid=t['id'];pop,source=('DEV319',fid)if t['group']=='WOOD_PRIMARY105'else('GREEN0918',fid.split(':',1)[1])
        ref=t.get('reference_corner_id','')
        if ref not in('',None,'None','null'):
            key=(pop,source,t['image_sha256'],int(ref));h=humans.get(key);case=cases.get(key[:3])
            if h is not None and case is not None:
                require(h['category']==t['human_state'],'Human reference state lineage changed')
                require(core.numeric_same(h['reference_xy'][0],t['human_reference_x'])and core.numeric_same(h['reference_xy'][1],t['human_reference_y']),'Reference coordinate lineage changed')
                d['annotation_coordinate_source']=h.get('label_evidence',{}).get('annotation_coordinate_source','UNAVAILABLE')
                ab=case['annotation'];ap=Path(ab['path']);ap=ap if ap.is_absolute()else REPO/ap
                require(ap.resolve().is_relative_to(REPO),'Annotation outside fixed repository')
                if str(ap)not in cache:
                    binding=dict(path=str(ap),sha256=ab['sha256']);cache[str(ap)]=read(check(binding));bindings.append(binding)
                ann=cache[str(ap)];obj=ann['objects'][0];entries=obj.get('keypoint_annotations',[])
                point=entries[int(ref)]if len(entries)>int(ref)else{}
                xy=point.get('xy');sourcefield=point.get('source','UNAVAILABLE')
                eqxy=isinstance(xy,list)and len(xy)==2 and all(math.isfinite(float(v))for v in xy)and xy==h['reference_xy']
                d['raw_annotation_traceback']='EXPLICIT_RAW_MANUAL_XY_EXACT'if sourcefield==d['annotation_coordinate_source']=='manual_click'and eqxy else'LEGACY_FALLBACK_SOURCE_OR_XY_UNVERIFIED'
                d['original_annotation_source']=sourcefield;d['original_annotation_path']=str(ap)
                if t['human_state']=='DIRECT_VISIBLE'and d['raw_annotation_traceback']=='EXPLICIT_RAW_MANUAL_XY_EXACT':
                    try:qx,qy=float(t['query_x']),float(t['query_y']);rx,ry=map(float,h['reference_xy'])
                    except (ValueError,TypeError):pass
                    else:
                        if all(math.isfinite(x)for x in(qx,qy,rx,ry))and(qx,qy)!=(-1.,-1.):
                            error=math.hypot(qx-rx,qy-ry)
                            require(core.numeric_same(error,t['initial_reference_error_px']),'Frozen reference error formula drift')
                            d['secondary_error_px']=error;secondary[i]=target(error);d['secondary_fixed_reference_target']=int(secondary[i])
        ledger.append(d)
    stats=[];av=packs['C16']['valid']
    for group in('ALL','WOOD_PRIMARY105','SQUARE118'):
      mask=np.array([group=='ALL'or t['group']==group for t in truth])
      for scope,m in(('ALL_ORIGINAL',mask),('INPUT_VALID',mask&av)):
       for model in NEW_MODELS:
        z=packs[model]['raw_three_logits'][:,2];pred=np.where(av,(z>=0).astype(np.int8),-1)
        stats.append(dict(group=group,scope=scope,model=model,measurement='SECONDARY_FIXED_REFERENCE_COORDINATE_DEVIATION_NOT_PHYSICAL_OR_PIXEL_GOLD',
          **metric(secondary,pred,m),primary_gold_known_N=0,untrained_neutral_auxiliary=(model=='C35')))
    counts=dict(Counter(r['raw_annotation_traceback']for r in ledger))
    return ledger,stats,dict(original_rows=1784,new_primary_quality_gold_known=0,new_primary_quality_gold_unknown=1784,
      manual_raw_xy_traceback_counts=counts,secondary_far=int(np.sum(secondary==0)),secondary_near=int(np.sum(secondary==1)),secondary_unknown=int(np.sum(secondary<0)),
      original_annotation_bindings=bindings,human_sidecar=p['human_sidecar'],review_manifest=p['review_manifest'],
      native_physical_authority_certified=False,reference_traceback_is_not_click_event_or_physical_native_certification=True)
def run(entry):
    e=read(entry);require(e['approved_saved_evaluation']is True,'Separate root evaluation entry required')
    require(all(e[k]is False for k in('private_fit','threshold_or_model_choice','RGB_read','encoder_forward','V112','PnP','q_L_correction')),'Scope')
    require(e['original_annotation_row_traceback']and e['new_primary_quality_gold_unknown']and e['quality_diagnostic_only'],'Quality authority boundary')
    require(sha(POLICY)==POLICY_SHA and e['code']==bind(__file__)and e['protocol']==bind(PROTOCOL),'Frozen code/policy')
    p=read(PROTOCOL);require(e['prediction_seal']==p['prediction_seal']==bind(PRED)and sha(PRED)==PRED_SHA,'Prediction seal binding')
    seal=read(PRED);require(seal['completed']and seal['PASS']and seal['selected_model']is None,'Completed inference')
    require(set(seal['models'])==set(NEW_MODELS),'All four candidates required')
    fixed=read(POLICY)['models_fixed_before_current_body_reads']
    for model,x in seal['models'].items():require(x['fixed_update']==(2000 if model in('C35','C36')else 8000)and x['state_before']==x['state_after']and fixed[model]['threshold_logit']==0 and x['source_current_allowed_false_unchanged']and not x['source_checks_all_pass'],'Fixed head')
    parity=read(check(seal['C16_parity']));require(parity['PASS'],'Original C16 parity')
    for b in p['operator_codes']+[p['prior_code'],p['prior_protocol'],p['annotation_authority'],p['annotation_definition_scope']]:check(b)
    require(read(check(p['generated']))['PASS']and not OUT.exists(),'Generated/exclusive output')
    OUT.mkdir();write(OUT/'BEFORE_LABEL_READ.json',dict(completed=True,entry=bind(entry),prediction_seal=p['prediction_seal'],label_bodies_read_so_far=False))
    core,base,extra,helpers=pure();truth=core.readcsv(check(p['truth']));manual=core.readcsv(check(p['manual']));prior=[json.loads(s)for s in check(p['prior']).read_text().splitlines()if s]
    require(len(truth)==len(prior)==1784 and len({r['id']for r in truth})==223,'Full1784')
    reference=helpers.load_packet(check(p['C16_raw']));require(int(reference['valid'].sum())==1739,'Original1739valid/45U')
    rows={};packs={};rows['C16'],packs['C16']=helpers.records(core,prior,p['C16_scores'],p['C16_raw'],reference)
    for model in NEW_MODELS:
        rows[model],packs[model]=helpers.records(core,prior,seal['models'][model]['scores_csv'],seal['models'][model]['raw_packet'],reference)
        a=packs[model];require(a['raw_three_logits'].dtype==np.float32 and a['raw_three_logits'].shape==(1784,3)and np.isfinite(a['raw_three_logits']).all(),'Raw three outputs')
        require(byte_equal(a['raw_three_logits'][:,0].copy(),a['raw_head_logit_before_U']),'Visibility column alias')
    oldrows=[dict(r,raw_physical_external_logit=r['raw_query_aux_hidden_logit'])for r in rows['C36']]
    indices={k:core.score_index(value,'raw_query_aux_hidden_logit')for k,value in rows.items()};original_call=base.call
    def call(r,model,task,probabilities,threshold,rules=None):
        if model in indices:return core.response(indices[model][core.key(r)],False)
        return original_call(r,model,task,probabilities,threshold,rules)
    base.call=call
    tables=core.calculate_all(base,truth,prior,rows['C35'],oldrows,manual)
    tables['coordinate_error_bins']=extra.error_bins(core,base,truth,manual,rows['C35'],oldrows)
    tables['D_vs_external_E_2x3']=extra.matrices(tables['metrics'])
    def model_call(r,model):return call(r,model,'D_vs_E',{},.5)
    tables['paired_recovery_and_damage']=transitions(core,base,truth,model_call)
    replay=core.historical_replay(tables['metrics'],core.readcsv(check(p['historical_original'])),{'v61','legacy'})
    replay16=core.historical_replay([dict(r,model='V124_C16_RAW')for r in tables['metrics']if r['model']=='C16'],core.readcsv(check(p['historical_C16_metrics'])),{'V124_C16_RAW'})
    require(sum(base.yes(r,'paired_same_query_RGB_D_vs_E')and r['human_state']in(base.D,base.E)for r in truth)==1237,'Original paired1237')
    require(sum(base.yes(r,'same_slot_pipeline_D_vs_E')and r['human_state']in(base.D,base.E)for r in truth)==1248,'Original same-slot1248')
    require({r['N']for r in tables['manual_near_D']}=={153,223},'Original manualnear376 unchanged')
    inventory=[];ledger=[]
    for i,r in enumerate(truth):
        k=core.key(r);d=dict(global_index=i,id=k[0],native_corner=int(k[1]),group=r['group'],human_state=r['human_state'],mapping_status=r['mapping_status'],
          query_x=float(reference['original_query_xy'][i,0]),query_y=float(reference['original_query_xy'][i,1]),roi_side=float(reference['original_side_L'][i]),query_available=bool(reference['valid'][i]),
          human_reference_x=r['human_reference_x'],human_reference_y=r['human_reference_y'],
          paired_same_query_RGB_D_vs_E=r['paired_same_query_RGB_D_vs_E'],same_slot_pipeline_D_vs_E=r['same_slot_pipeline_D_vs_E'],
          initial_reference_error_px=r['initial_reference_error_px'],original_query_valid=r['query_valid'],legacy_original_state=r['legacy_stored_state'])
        for model in MODELS:
            response='OTHER_UNRESOLVED'if model=='v61'and(r['query_valid']!='True'or r.get('present_at_fixed_0_5')not in('True','False'))else model_call(r,model)
            d[model+'_original_response']=response;d[model+'_D_E_U']='D'if response=='VISIBLE'else'E'if response=='ABSENT_NONVISIBLE'else'U'
        for model in ('C16',*NEW_MODELS):d[model+'_raw_visibility_logit']=float(packs[model]['raw_head_logit_before_U'][i])
        for model in NEW_MODELS:
            d[model+'_raw_entity_logit']=float(packs[model]['raw_three_logits'][i,1]);d[model+'_raw_quality_logit']=float(packs[model]['raw_three_logits'][i,2])
        d['v61_saved_p_present']=r.get('presence_probability','');d['legacy_saved_p_DIRECT']=r.get('legacy_p_DIRECT',r.get('legacy_direct_probability',''))
        ledger.append(d)
    for group in('ALL',*base.GROUPS):
        rr=ledger if group=='ALL'else[r for r in ledger if r['group']==group]
        for model in MODELS:
            cc=Counter(r[model+'_D_E_U']for r in rr);responses=Counter(r[model+'_original_response']for r in rr)
            inventory.append(dict(group=group,model=model,all_registered_slots=len(rr),D=cc['D'],E_binary_response=cc['E'],U=cc['U'],U_rate=cc['U']/len(rr),
              legacy_abstain=responses['ABSTAIN'],out_of_task_geometry_S=responses['OUT_OF_TASK_GEOMETRY_S'],other_unresolved=responses['OTHER_UNRESOLVED'],
              human_state_inventory=json.dumps(dict(Counter(r['human_state']for r in rr)),sort_keys=True)))
    tables['all1784_model_inventory']=inventory
    historical_replay={}
    for name,b in p['historical_three_tables'].items():
        hist=core.readcsv(check(b))
        if name=='ALL1784_NATIVE_LEDGER':
            fields=[k for k in hist[0]if not any(k.startswith(m+'_')for m in('C29','C32','C33','C34'))]
            current=[{k:r[k]for k in fields}for r in ledger];hist=[{k:r[k]for k in fields}for r in hist]
        else:
            key=next(k for k in tables if k.upper()==name);current=[r for r in tables[key]if r.get('model')in BASELINES];hist=[r for r in hist if r.get('model')in BASELINES]
        historical_replay[name]=helpers.exact_csv_rows(current,hist)
    qualityledger,qualitymetrics,qualitysummary=quality_traceback(core,truth,packs,p)
    for r,qr in zip(ledger,qualityledger):
        r['annotation_coordinate_source']=qr['annotation_coordinate_source'];r['raw_annotation_traceback']=qr['raw_annotation_traceback']
        r['new_primary_quality_gold_target']=qr['new_primary_quality_gold_target'];r['secondary_fixed_reference_quality_target']=qr['secondary_fixed_reference_target']
    tables['secondary_fixed_reference_quality_diagnostic']=qualitymetrics
    for name in('metrics','confusion','human_self_binary_response','manual_near_D','availability_by_group','coordinate_error_bins','D_vs_external_E_2x3','paired_recovery_and_damage','all1784_model_inventory','secondary_fixed_reference_quality_diagnostic'):
        core.writecsv(OUT/(name.upper()+'.csv'),tables[name])
    core.writecsv(OUT/'ALL1784_NATIVE_LEDGER.csv',ledger)
    # This ledger has optional provenance fields; write a fixed union schema.
    fields=list(dict.fromkeys(k for r in qualityledger for k in r));qualityledger=[{k:r.get(k)for k in fields}for r in qualityledger]
    core.writecsv(OUT/'ALL1784_POINT_QUALITY_AUTHORITY_AND_TRACEBACK.csv',qualityledger)
    result=dict(completed=True,PASS=True,models=list(MODELS),tables=tables,original_slots=1784,WOOD_frames=105,SQUARE_frames=118,
      original_input_valid=1739,original_input_U=45,additional_U=0,cohort_denominators=dict(paired_same_query_RGB=1237,same_slot_pipeline=1248),
      historical_v61_legacy_rows_exact=replay,historical_C16_rows_exact=replay16,historical_three_all_table_fields_exact=historical_replay,
      source_admission_unchanged=True,selected_model=None,adopted_model=None,development_exposed_not_blind=True,
      primary_visibility_never_filtered_by_quality_or_error=True,S_OOF_unknown_outside_external_binary=True,
      original_manualnear376_unchanged=True,new_primary_real_point_quality_gold_known=0,quality_authority=qualitysummary,
      quality_scope='SECONDARY_FIXED_REFERENCE_COORDINATE_DEVIATION_NOT_PHYSICAL_OR_PIXEL_GOLD',
      policy=p['policy'],prediction_seal=p['prediction_seal'],new_forward_fit_RGB_encoder_PnP_qL_or_threshold_choice=0)
    write(OUT/'CURRENT_FROZEN_BENCHMARK_RESULTS.json',result)
    outputs=[bind(x)for x in OUT.iterdir()if x.is_file()]
    write(OUT/'EVALUATION_SEAL.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),policy=p['policy'],entry=bind(entry),prediction_seal=p['prediction_seal'],
      outputs=outputs,selected_model=None,source_admission_changed=False,new_forward_fit_RGB_PnP=0))
    print(json.dumps(dict(completed=True,PASS=True,seal=bind(OUT/'EVALUATION_SEAL.json'),results=bind(OUT/'CURRENT_FROZEN_BENCHMARK_RESULTS.json'))))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare and not a.entry:prepare()
    elif a.entry and not a.prepare:run(a.entry)
    else:raise SystemExit('Use metadata-only --prepare or separate root --entry.')
