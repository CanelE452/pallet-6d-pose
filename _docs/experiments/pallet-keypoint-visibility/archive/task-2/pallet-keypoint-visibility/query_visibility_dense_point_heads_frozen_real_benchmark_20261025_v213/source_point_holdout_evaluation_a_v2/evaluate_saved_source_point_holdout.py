"""Fixed SOURCE HOLDOUT saved-point evaluation. No model, RGB, input NPZ or fit."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,math,os,stat,io,traceback
import numpy as np
H=Path(__file__).parent;V=H.parent;P=V.parent
S=V/'ROOT_PROSPECTIVE_FROZEN_SOURCE_POINT_HOLDOUT_EVALUATION_SCOPE.json'
S_SHA='f49887d6bf54b6e14ae175f7023a2f3fd55cde691c4c75a2ab7dd1e8c6adae8a'
OLD=P/'query_visibility_spatial_q8_nominal_grouped_source_training_20261025_v202/fit_preparation_b/PROTOCOL_BEFORE_SPATIAL_Q8_SOURCE_ARRAY_OR_FIT_BODIES.json'
OLD_SHA='8df60dc146b5f026563341a1d959ecff20ef501333a14694c3b0794a2c0e07d8'
SCENES=P/'query_visibility_full_source_query_causal_census_20261024_v173/census_preparation/ALL768_SCENE_PATHS_BEFORE_MASKS.jsonl'
SCENES_SHA='e12ae8fd3c908ed1dee5c99fa6915601b1b6e176ce6cefabc4292abbef1c1307'
FIT=P/'query_visibility_dense_point_weak_bridge_source_training_20261025_v212/runs/C62_C63_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIXED4000/COMPLETE.json'
OUT=V/'source_point_holdout_evaluation_v2'
ARMS=('C62_DENSE_POINT_RAW_Q8','C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8')
VARIANTS=('BEFORE','COVER','NEAR_SHAM','FAR_SHAM');STATES=('D','E','S','U','UNKNOWN')
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
GT_ALLOWED={};NPZ_ALLOWED={}
def need(v,m):
    if not bool(v):raise AssertionError(m)
def safe(path):
    p=Path(path);need('v112'not in str(p).casefold(),'Forbidden lexical path before filesystem')
    need(p.is_absolute()and P in p.parents and '..'not in p.parts,'Exact absolute project path');return p
def raw(path,kind='text'):
    p=safe(path)
    if p.name=='SCENES_GT_ONLY.json':need(kind=='GT'and str(p)in GT_ALLOWED,'GT only after actual entry whitelist')
    elif p.suffix=='.npz':need(kind=='prediction'and str(p)in NPZ_ALLOWED,'Only two frozen SYN prediction packets')
    else:need(kind=='text'and p.suffix in('.json','.jsonl','.py'),'Text-only permitted metadata')
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        last=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(last).st_mode),'Regular file only')
    with os.fdopen(last,'rb')as f:return f.read()
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(raw(p)).hexdigest())
def checked(b,kind='text'):
    z=raw(b['path'],kind);need(hashlib.sha256(z).hexdigest()==b['sha256'],'Exact saved binding');return z
def get(b):return json.loads(checked(b))
def lines(b):return [json.loads(x)for x in checked(b).splitlines()if x.strip()]
def new(p,d):
    safe(p);need(p.parent in(H,OUT),'Exclusive own outputs')
    with p.open('x',encoding='utf8')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def rowsout(p,rs):
    safe(p);need(p.parent in(H,OUT),'Exclusive own outputs')
    with p.open('x',encoding='utf8')as f:
        for r in rs:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def map_input(row,scene_map):
    """Image scene comes from original image identity; NEVER anchor/cache index."""
    k=row['native_slot'];image=row.get('source_image_original_global_index');anchor=row.get('anchor_original_global_index')
    base=dict(row,component_role='HOLDOUT',source_teacher_key=None,mapping_status='UNKNOWN_MISSING_EXACT_IMAGE_SOURCE_IDENTITY',new_training_admission=False)
    if type(image)is not int or type(anchor)is not int:return base
    assert image%8==anchor%8==k
    si=image//8;ai=anchor//8
    if si not in scene_map or ai not in scene_map:return base
    source=scene_map[si];anchor_scene=scene_map[ai]
    assert source['scene_index']==row['scene_index']and source['base_index']==row['base_index']and source['variant']==row['variant']
    assert row['anchor_scene_index']==ai
    own=source['original_input_metadata8'][k];anch=anchor_scene['original_input_metadata8'][k]
    assert own['original_global_index']==image and anch['original_global_index']==anchor
    assert row['query_xy']==anch['original_query_xy']and row['L']==anch['original_L']
    assert row['original_query_available']==anch['original_input_flags']['query_available']
    assert row['source_image_original_query_id']==own['original_occurrence_id']and row['anchor_original_query_id']==anch['original_occurrence_id']
    assert row['original_flags_ref']==[ai,k]and row['image_flags_ref']==[si,k]
    if row['family']=='SELF_SCENE15':assert si==ai and row['state_index']==0 and row['state_name']=='NOMINAL'
    else:assert row['family']=='BEFORE_ANCHOR4'and anchor_scene['variant']=='BEFORE'
    base.update(source_teacher_key=[si,k],mapping_status='EXACT_ORIGINAL_IMAGE_OWN_SCENE',actual_image_scene_index=si,actual_image_variant=source['variant'],
        actual_image_original_global_index=image,actual_image_original_occurrence_id=own['original_occurrence_id'],
        anchor_scene_index=ai,anchor_variant=anchor_scene['variant'],anchor_query_original_occurrence_id=anch['original_occurrence_id'],
        actual_image_source_binding_inert=source['source_RGB_binding'],original_input_flags=anch['original_input_flags'],
        actual_image_original_input_flags_context_only=own['original_input_flags'],declared_CF_to_canonical=own['declared_CF_to_canonical'])
    return base

def select_scene(doc,scene):
    if not isinstance(doc,dict)or not isinstance(doc.get('rows'),list):return None,'UNKNOWN_EXPLICIT_TOP_ROWS_SCHEMA'
    candidates=[r for r in doc['rows']if isinstance(r,dict)and r.get('variant')==scene['variant']and r.get('scene_index')==scene['scene_index']]
    assert len(candidates)<=1,'Ambiguous ownscene'
    if not candidates:return None,'UNKNOWN_MISSING_EXACT_OWN_SCENE'
    own=candidates[0];assert own['base_index']==scene['base_index']
    rr=own.get('records_CF');perm=own.get('permCF_to_canonical')
    if not isinstance(rr,list)or len(rr)!=8 or not all(isinstance(r,dict)and 'CF_index'in r and 'canonical_index'in r for r in rr):return None,'UNKNOWN_MISSING_EXPLICIT_RECORDS_CF'
    if not isinstance(perm,list)or len(perm)!=8:return None,'UNKNOWN_MISSING_EXPLICIT_SCENE_PERM'
    assert all(type(r['CF_index'])is int and type(r['canonical_index'])is int for r in rr)
    assert all(type(x)is int for x in perm)and sorted(perm)==list(range(8))
    records={r['CF_index']:r for r in rr};assert set(records)==set(range(8))and sorted(r['canonical_index']for r in rr)==list(range(8))
    for original in scene['native_rows']:
        k=original['native_slot'];r=records[k]
        assert r['canonical_index']==perm[k]==original['declared_CF_to_canonical']
        if 'native_canonical_index'in r:assert r['native_canonical_index']==perm[k]
    return records,None

def extract(record):
    def state(name,point=False):
        value=None if record is None else record.get(name)
        if point:value=value[0]if isinstance(value,list)and value else None
        return value if isinstance(value,str)and value in STATES[:-1]else'UNKNOWN'
    reference=None if record is None else record.get('projected_reference_xy')
    if not(isinstance(reference,list)and len(reference)==2 and all(type(v)in(int,float)and math.isfinite(v)for v in reference)):reference=None
    return dict(point0_target_alone_state=state('target_witness_states',True),point0_combined_state=state('combined_witness_states',True),
        region_state_context_only=state('state'),target_alone_region_state_context_only=state('target_alone_region_state'),saved_projected_reference_xy=reference)

def locality(q,L,available,reference,raw_hw):
    out=dict(original_input_available=available,original_input_U_preserved=not available,raw96_reference_index_xy=None,
        inside_sample_center_extent=None,inside_crop_canvas_extent=None,inside_original_frame_canvas_extent=None,
        not_coordinate_error_correction_or_pixel_visibility_proof=True)
    if reference is None:return out
    h,w=raw_hw;out['inside_original_frame_canvas_extent']=(-.5<=reference[0]<w-.5 and -.5<=reference[1]<h-.5)
    if not(isinstance(q,list)and len(q)==2 and all(type(x)in(int,float)and math.isfinite(x)for x in q)and type(L)in(int,float)and math.isfinite(L)and L>0):return out
    u=[47.5+96*(reference[i]-q[i])/L for i in range(2)]
    out.update(raw96_reference_index_xy=u,inside_sample_center_extent=all(0<=v<=95 for v in u),inside_crop_canvas_extent=all(-.5<=v<95.5 for v in u))
    return out

def prepare():
    need(bind(S)['sha256']==S_SHA and bind(OLD)['sha256']==OLD_SHA,'Prospective scope and source protocol')
    b=get(bind(OLD))['context_input_bindings'];role_rows=lines(b['SYNTH_AUG65536_roles']);roles={r['row_id']:r for r in role_rows if r['role']=='HOLDOUT'}
    need(len(roles)==sum(r['role']=='HOLDOUT'for r in role_rows),'Unique immutable HOLDOUT rows')
    all_aug=lines(b['SYNTH_AUG65536']);aug=[r for r in all_aug if r['row_id']in roles];bases={r['base_index']for r in aug}
    all_scenes=lines(b['original_scenes']);scenes=[r for r in all_scenes if r['base_index']in bases];smap={r['scene_index']:r for r in scenes}
    original_scene_rows=lines(dict(path=str(SCENES),sha256=SCENES_SHA));selected_old=[r for r in original_scene_rows if r['base_index']in bases]
    parents=[]
    for base in sorted(bases):
        subset=[r for r in selected_old if r['base_index']==base];need(len(subset)==4 and {r['variant']for r in subset}==set(VARIANTS),'All four own scenes, no label filter')
        receipts={json.dumps(r['base_checkpoint'],sort_keys=True)for r in subset};need(len(receipts)==1,'One exact original base receipt')
        receipt_binding=subset[0]['base_checkpoint'];receipt=get(receipt_binding);need(receipt['completed']and receipt['PASS']and receipt['base_index']==base,'Exact original parent')
        gt=receipt['GT_scenes'];p=safe(gt['path']);need(p.name=='SCENES_GT_ONLY.json'and len(gt['sha256'])==64,'Inert GT binding only')
        parents.append(dict(base_index=base,component_role='HOLDOUT',base_receipt=receipt_binding,GT_json_binding_NOT_OPENED=gt,source_component_ids=sorted({r['component_id']for r in roles.values()if r['base_index']==base})))
    pmap={p['base_index']:p for p in parents};native=[];selected_scenes=[]
    for s in sorted(scenes,key=lambda r:r['scene_index']):
        need(s['variant']in VARIANTS and [r['native_slot']for r in s['original_input_metadata8']]==list(range(8)),'Original all8 own scene')
        need(sorted(r['declared_CF_to_canonical']for r in s['original_input_metadata8'])==list(range(8)),'Declared permutation bijective')
        ns=[]
        for o in s['original_input_metadata8']:
            need(o['scene_index']==s['scene_index']and o['base_index']==s['base_index']and o['variant']==s['variant'],'Own native identity')
            r=dict(teacher_key=[s['scene_index'],o['native_slot']],role='HOLDOUT',historical_source_split=s['split'],scene_index=s['scene_index'],base_index=s['base_index'],variant=s['variant'],native_slot=o['native_slot'],declared_CF_to_canonical=o['declared_CF_to_canonical'],original_global_index=o['original_global_index'],original_occurrence_id=o['original_occurrence_id'],query_xy=o['original_query_xy'],L=o['original_L'],original_query_available=o['original_input_flags']['query_available'],original_input_flags=o['original_input_flags'],raw_hw=s['raw_hw'],factors=s['factors'],source_components=pmap[s['base_index']]['source_component_ids'],new_training_admission=False)
            native.append(r);ns.append(r)
        selected_scenes.append(dict(scene_index=s['scene_index'],base_index=s['base_index'],variant=s['variant'],native_rows=ns))
    eligible=[r for r in aug if r['family']=='BEFORE_ANCHOR4'or(r['family']=='SELF_SCENE15'and r['state_index']==0 and r['state_name']=='NOMINAL')]
    mapped=[]
    for r in eligible:
        role=roles[r['row_id']]
        for k in META:need(r[k]==role[k],'Original META6 '+k)
        need(r['source_image_original_global_index']==role['source_image_original_global_index'],'Actual image identity')
        m=map_input(r,smap);m.update(component_id=role['component_id'],input_cache_index_context_only=role['cache_index'],input_cache_binding_key='SYNTH_AUG65536',historical_source_split=r['split'],factors=role['factors']);mapped.append(m)
    counts=dict(parents=len(parents),scenes=len(scenes),own_native=len(native),eligible_inputs=len(mapped),all_HOLDOUT_aug_rows=len(aug))
    need(counts==dict(parents=18,scenes=72,own_native=576,eligible_inputs=1152,all_HOLDOUT_aug_rows=9216),'Actual metadata differs from prospective expected counts: '+str(counts))
    need(len({tuple(r['teacher_key'])for r in native})==576 and len({r['row_id']for r in mapped})==1152,'Unique source keys')
    rowsout(H/'ALL576_HOLDOUT_OWN_NATIVE_SELECTION.jsonl',native);rowsout(H/'ALL1152_HOLDOUT_INPUT_MAPPING.jsonl',mapped)
    new(H/'GT18_WHITELIST_NOT_OPENED.json',dict(completed=True,GT_json_whitelist=[r['GT_json_binding_NOT_OPENED']for r in parents],GT_stat_hash_body_reads=0))
    sel=dict(completed=True,PASS=True,parents=parents,scenes=selected_scenes,counts=counts,metadata_sources=dict(scope=bind(S),old_protocol=bind(OLD),roles=b['SYNTH_AUG65536_roles'],aug_rows=b['SYNTH_AUG65536'],original_scenes=b['original_scenes'],receipt_pointers=dict(path=str(SCENES),sha256=SCENES_SHA)),physically_parsed_metadata_rows=dict(role_container=len(role_rows),aug_container=len(all_aug),scene_container=len(all_scenes),original_pointer_container=len(original_scene_rows),selected_parent_receipts=len(parents)),metadata_counts=dict(own_inputU=sum(not r['original_query_available']for r in native),mapped_inputU=sum(not r['original_query_available']for r in mapped),mapping_status=dict(Counter(r['mapping_status']for r in mapped))),source_GT_arrays_weights_scores_reads=0,new_training_admission=False)
    new(H/'SELECTION_METADATA_ONLY.json',sel);print(json.dumps(dict(completed=True,PASS=True,counts=counts,metadata_counts=sel['metadata_counts'])))

def evaluation_target(r):
    reasons=[]
    if r.get('component_role')!='HOLDOUT':reasons.append('NOT_ORIGINAL_HOLDOUT')
    if r.get('mapping_status')!='EXACT_ORIGINAL_IMAGE_OWN_SCENE':reasons.append('MISSING_IMAGE_MAPPING')
    if r.get('family')not in('SELF_SCENE15','BEFORE_ANCHOR4')or r.get('state_name')!='NOMINAL'or r.get('state_index')!=0:reasons.append('NOT_NOMINAL_ELIGIBLE')
    if r.get('original_query_available')is not True:reasons.append('ORIGINAL_INPUT_U')
    q=r.get('query_xy');L=r.get('L');ref=r.get('saved_projected_reference_xy')
    def pair(x):return isinstance(x,list)and len(x)==2 and all(type(v)in(int,float)and math.isfinite(v)for v in x)
    finite_q=pair(q)and type(L)in(int,float)and math.isfinite(L)and L>0
    if not finite_q:reasons.append('MISSING_Q_OR_L')
    if r.get('point0_target_alone_state')!='D':reasons.append('TARGET_ALONE_NOT_D')
    state=r.get('point0_combined_state')
    if state not in('D','E'):reasons.append('COMBINED_NOT_DE')
    loc=r.get('reference_locality_for_actual_input',{})
    if not pair(ref):reasons.append('MISSING_REFERENCE')
    if loc.get('inside_original_frame_canvas_extent')is not True:reasons.append('REFERENCE_OUTSIDE_FRAME')
    if loc.get('inside_crop_canvas_extent')is not True:reasons.append('REFERENCE_OUTSIDE_ACTUAL_INPUT_CROP')
    if finite_q and pair(ref):
        uv=[47.5+96*(float(ref[j])-float(q[j]))/float(L)for j in range(2)]
        need(all(-.5<=v<95.5 for v in uv)==loc.get('inside_crop_canvas_extent'),'Exact fixed crop canvas')
        need(np.array_equal(uv,loc.get('raw96_reference_index_xy'),equal_nan=True),'Original q/L crop algebra')
    return (-1 if reasons else int(state=='E')),reasons

def original_metadata_value_equal(stored,original,key):
    """Lossless JSON null-to-stored NaN convention; never changes input values."""
    if key=='query_xy':
        wanted=np.asarray([np.nan,np.nan]if original is None else original,np.float64)
        return np.array_equal(stored,wanted,equal_nan=True)
    if key=='L':
        wanted=np.asarray(np.nan if original is None else original,np.float64)
        return np.array_equal(stored,wanted,equal_nan=True)
    return bool(stored==original)

def decode(z,available):return -1 if not available or not math.isfinite(float(z))else int(float(z)>=0)
def metric(rows,arm):
    cm=[[0,0,0],[0,0,0]]
    for r in rows:
        t=r['evaluation_target_D0_E1_UNKNOWNminus1']
        if t in(0,1):cm[t][r['predictions'][arm]['prediction']if r['predictions'][arm]['prediction']in(0,1)else 2]+=1
    n=sum(map(sum,cm));precision=[];recall=[]
    for k in(0,1):
        a=sum(cm[j][k]for j in(0,1));b=sum(cm[k]);precision.append(cm[k][k]/a if a else None);recall.append(cm[k][k]/b if b else None)
    return dict(all_rows=len(rows),known_truth=n,unknown_truth=len(rows)-n,confusion_D_E_by_D_E_U=cm,precision_D_E=precision,recall_D_E=recall,U=sum(r[2]for r in cm),U_rate=sum(r[2]for r in cm)/n if n else None)
def guard_fit(fit,barrier,review,e):
    need(fit['completed']and fit['PASS']and fit['source_only']and fit['from_scratch'],'Both completed source fits')
    need(fit['parameters_each']==118359 and fit['steps_each']==4000 and fit['total_updates']==8000,'Frozen fixed two4000')
    need(fit['all2_frozen_before_any_SOURCE_outputs']is True and fit['protected_real_COMMON_VAL_access']==0,'Pre-output source barrier')
    need(review['completed']and review['PASS']and review['fit_complete']==e['fit_complete'],'Independent audit exact completed fit')
    need(barrier['completed']and barrier['checkpoints']=={a:fit['arms'][a]['checkpoint']for a in ARMS},'Both original checkpoint bindings')
    for a in ARMS:
        r=fit['arms'][a];need(r['updates']==4000 and r['use_sam']is False and r['spatial_q8']is True,'Frozen RAW spatial flags')
        need(r['weak_bridge']==(a==ARMS[1])and r['weak_loss_weight']==(.25 if a==ARMS[1]else 0),'A priori fixed two arms')
        need(r['outputs']['SYNTH_AUG65536']==e['prediction_packets'][a],'Only exact saved full SYN predictions')

def run(entry_path):
    e=get(bind(entry_path));need(e.get('approved_frozen_SOURCE_HOLDOUT_point_evaluation')is True and e.get('new_training_admission')is False and e.get('automatic_retry')is False,'Separate root actual entry')
    need(e.get('root_actual_scope')is not None,'Root actual scope');get(e['root_actual_scope']);need(Path(e['root_actual_scope']['path']).parent==V,'Exact V213 scope')
    for k,n in [('code','evaluate_saved_source_point_holdout.py'),('protocol','PROTOCOL_BEFORE_HOLDOUT_GT_OR_SAVED_PREDICTIONS.json'),('selection','SELECTION_METADATA_ONLY.json'),('whitelist','GT18_WHITELIST_NOT_OPENED.json'),('native_rows','ALL576_HOLDOUT_OWN_NATIVE_SELECTION.jsonl'),('input_mapping','ALL1152_HOLDOUT_INPUT_MAPPING.jsonl')]:need(e[k]==bind(H/n),'Frozen preparation '+k)
    need(e['fit_complete']['path']==str(FIT),'Only completed V212 source fit')
    fit=get(e['fit_complete']);need(fit['source_split']==get(bind(OLD))['context_input_bindings']['split_complete'],'Original immutable source component split');need(fit['all2_freeze']==e['both_final_barrier'],'Exact both-final barrier');barrier=get(e['both_final_barrier']);review=get(e['source_fit_review']);guard_fit(fit,barrier,review,e)
    sel=get(e['selection']);whitelist=get(e['whitelist'])['GT_json_whitelist'];need(len(whitelist)==18 and e['GT_json_whitelist']==whitelist,'Exact existing GT18 whitelist')
    need(e['output_directory']==str(OUT),'Fresh exact output');need(not OUT.exists(),'No retry or overwrite');OUT.mkdir()
    GT_ALLOWED.update({b['path']:b for b in whitelist});original=lines(e['native_rows']);mapping=lines(e['input_mapping']);need(len(original)==576 and len(mapping)==1152,'Whole denominators')
    teachers=[];scene_map={s['scene_index']:s for s in sel['scenes']}
    for parent in sel['parents']:
        b=parent['GT_json_binding_NOT_OPENED'];doc=json.loads(checked(b,'GT'))
        for scene in sorted((s for s in sel['scenes']if s['base_index']==parent['base_index']),key=lambda s:s['scene_index']):
            rr,why=select_scene(doc,scene)
            for r in scene['native_rows']:
                z=dict(r,**extract(None if rr is None else rr[r['native_slot']]),GT_binding=b,parser_status=why or'EXACT_OWN_SCENE_RECORDS_CF',authority='SAVED_PROCEDURAL_SOURCE_GEOMETRIC_POINT0_NOT_REAL_GOLD')
                z['reference_locality_for_own_input']=locality(z['query_xy'],z['L'],z['original_query_available'],z['saved_projected_reference_xy'],z['raw_hw']);teachers.append(z)
    lookup={tuple(r['teacher_key']):r for r in teachers};allrows=[]
    for r in mapping:
        t=lookup.get(tuple(r['source_teacher_key']))if r['source_teacher_key']is not None else None;z=dict(r)
        z.update({k:(t[k]if t else None)for k in ('raw_hw','GT_binding','parser_status')});z.update({k:(t[k]if t else 'UNKNOWN'if 'state'in k else None)for k in ('point0_target_alone_state','point0_combined_state','region_state_context_only','target_alone_region_state_context_only','saved_projected_reference_xy')})
        z['reference_locality_for_actual_input']=locality(r['query_xy'],r['L'],r['original_query_available'],z['saved_projected_reference_xy'],t['raw_hw']if t else[0,0]);target,reasons=evaluation_target(z);z.update(evaluation_target_D0_E1_UNKNOWNminus1=target,evaluation_excluded_reasons=reasons,predictions={});allrows.append(z)
    for a in ARMS:
        b=e['prediction_packets'][a];need(Path(b['path']).parent==FIT.parent/a and Path(b['path']).name=='FINAL_SYNTH_AUG65536_OUTPUTS.npz','Own exact full prediction path')
        NPZ_ALLOWED[b['path']]=b
        with np.load(io.BytesIO(checked(b,'prediction')),allow_pickle=False)as npz:
            packet={k:npz[k]for k in META+('raw_direct_end_external_logit','direct_end_prediction_D0_E1_Uminus1')}
        need(len(packet['row_id'])==65536 and len(set(packet['row_id'].tolist()))==65536,'Full source prediction identity')
        byid={str(x):i for i,x in enumerate(packet['row_id'])}
        for r in allrows:
            need(r['row_id']in byid,'Exact saved row join, not stale cache index');i=byid[r['row_id']]
            for k in META:
                wanted=r[k]
                need(original_metadata_value_equal(packet[k][i],wanted,k),'Original META6 exact '+k)
            rawlogit=float(packet['raw_direct_end_external_logit'][i]);pred=decode(rawlogit,r['original_query_available']);need(int(packet['direct_end_prediction_D0_E1_Uminus1'][i])==pred,'Fixed zero/U scalar decoder')
            r['predictions'][a]=dict(prediction=pred,raw_direct_end_external_logit=rawlogit if math.isfinite(rawlogit)else None)
    result=dict(completed=True,PASS=True,cohort_counts=sel['counts'],own_point_states=dict(Counter(r['point0_combined_state']for r in teachers)),input_point_states=dict(Counter(r['point0_combined_state']for r in allrows)),known_evaluation_targets=dict(Counter(str(r['evaluation_target_D0_E1_UNKNOWNminus1'])for r in allrows)),excluded_reasons_overlapping=dict(Counter(x for r in allrows for x in r['evaluation_excluded_reasons'])),models={a:metric(allrows,a)for a in ARMS},strata={},limits=['SOURCE internal geometric point0, not physical real gold','Historically exposed fixed source HOLDOUT; not fresh blind','Both frozen models evaluated; C63 a priori primary; no selection/fit/threshold changes','TRAIN6096 apparent fit separate, not read or combined here','Original q/L/U and immutable HOLDOUT roles retained; GT reference never model input'])
    for name,fn in [('native',lambda r:r['native_slot']),('canonical',lambda r:r['declared_CF_to_canonical']),('material',lambda r:r['factors'].get('material','UNKNOWN')),('aspect',lambda r:r['factors'].get('aspect','UNKNOWN')),('geometry',lambda r:r['factors'].get('geometry_family','UNKNOWN')),('base_group',lambda r:r['base_index']),('variant',lambda r:r['variant']),('family',lambda r:r['family'])]:
        groups=defaultdict(list)
        for r in allrows:groups[str(fn(r))].append(r)
        result['strata'][name]={g:{a:metric(rs,a)for a in ARMS}for g,rs in sorted(groups.items())}
    rowsout(OUT/'ALL576_HOLDOUT_OWN_POINT_STATES.jsonl',teachers);rowsout(OUT/'ALL1152_HOLDOUT_SAVED_POINT_PREDICTION_EVALUATION.jsonl',allrows);new(OUT/'SUMMARY.json',result)
    for b in whitelist:need(hashlib.sha256(raw(b['path'],'GT')).hexdigest()==b['sha256'],'GT final rehash')
    for b in e['prediction_packets'].values():need(hashlib.sha256(raw(b['path'],'prediction')).hexdigest()==b['sha256'],'Prediction final rehash')
    new(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=e['code'],protocol=e['protocol'],entry=bind(entry_path),fit_complete=e['fit_complete'],source_fit_review=e['source_fit_review'],both_final_barrier=e['both_final_barrier'],GT_json_whitelist=whitelist,prediction_packets=e['prediction_packets'],outputs={n:bind(OUT/n)for n in('ALL576_HOLDOUT_OWN_POINT_STATES.jsonl','ALL1152_HOLDOUT_SAVED_POINT_PREDICTION_EVALUATION.jsonl','SUMMARY.json')},new_training_admission=False,model_forward_fit_or_weight_reads=0,input_RGB_NPZ_or_image_reads=0))
    print(json.dumps(dict(completed=True,PASS=True,cohort_counts=result['cohort_counts'],models=result['models'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-metadata',action='store_true');p.add_argument('--entry');a=p.parse_args()
    try:
        if a.prepare_metadata:prepare()
        else:need(a.entry is not None,'Actual entry required');run(a.entry)
    except Exception:
        if OUT.is_dir()and not(OUT/'FAILURE.json').exists():new(OUT/'FAILURE.json',dict(completed=False,PASS=False,error=traceback.format_exc(),automatic_retry=False))
        raise
