"""Independent saved SOURCE SAM math only. No SAM/runtime/weight/RGB imports or inference.
Conditional teacher strata are joined only after the complete frozen prediction packet passes.
"""
import argparse,csv,hashlib,io,json,math,os,stat,traceback
from pathlib import Path
from collections import Counter
import numpy as np

H=Path(__file__).absolute().parent
ROOT_SHA='8569a11e7b77ae5123a86660efb288a3853bd12a014113d9595c73408895cb0e'
INPUT_SHA={'frames':'ec53a055ab7de0ef7e5d3414b39a634dd0a0069534682820d62654dc2ff4341d','queries':'636df349195319949f81b4ad172f3b41161efe1753bcc39095f1a69e240c1284'}
TRUTH_SHA='a7a271f71c417e8a968cf1aaefc9dbd45d4a42597f267363f5dff862022fd02c'
STATES=('SAM_VISIBLE_MASK_SUPPORT','OUTSIDE_PROPOSAL','U')
NPZ_KEYS={'chosen_logit','quality_scores','selected_index','original_prompt_box','predictor_prompt_box','raw_hw','original_q8','original_L','native_slot','original_query_index','query_id','original_continuous_valid','original_nearest_valid','prediction_available','exact_logit','radius2_logit'}

def need(ok,msg):
    if not ok:raise ValueError(msg)

def ident(b):
    s=b['path'];need(not any(x in s.casefold() for x in ('v112','__pycache__','.pyc','current223','publicheld','oldval','source_holdout','private_evaluation')),'lexical before bound artifact FS')
    p=Path(s);need(p.is_absolute() and '..' not in p.parts,'absolute artifact');need(len(b['sha256'])==64 and all(x in '0123456789abcdef' for x in b['sha256']),'SHA');return s,b['sha256']

def raw(path):
    p=Path(path)
    for q in reversed(p.parents):need(stat.S_ISDIR(os.lstat(q).st_mode),'ancestor symlink')
    need(stat.S_ISREG(os.lstat(p).st_mode),'leaf symlink/nonregular')
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def checked(b,white):
    need(ident(b) in {ident(x) for x in white},'undeclared artifact');data=raw(b['path'])
    need(hashlib.sha256(data).hexdigest()==b['sha256'],'SHA mismatch')
    for k in ('bytes','size_bytes'):
        if k in b:need(len(data)==b[k],'size mismatch')
    return data

def prompt(rows):
    try:a=np.asarray([q['query_xy'] for q in rows],np.float64)
    except (ValueError,TypeError):return None,'Q8_MISSING_OR_NONNUMERIC'
    if a.shape!=(8,2) or not np.isfinite(a).all():return None,'Q8_MISSING_OR_NONFINITE'
    x0,x1=np.min(a[:,0]),np.max(a[:,0]);y0,y1=np.min(a[:,1]),np.max(a[:,1])
    if x1<=x0 or y1<=y0:return None,'Q8_DEGENERATE_BOX'
    return np.array([x0,y0,x1,y1],np.float64),None

def readout(mask,q,prompt_ok,runtime_ok):
    z={'prediction_available':False,'state':'U','native_query_valid':False,'pixel':None,'exact_logit':None,'radius2_logit':None,'exact_mask_support':None,'radius2_mask_support':None,'radius2_window_pixels':0,'radius2_clipped':False}
    xy=q['query_xy']
    if xy is None:return dict(z,U_reason='ORIGINAL_QUERY_ABSENT')
    if not all(math.isfinite(v) for v in xy):return dict(z,pixel=[-1,-1],U_reason='ORIGINAL_NONFINITE_QUERY')
    x,y=math.floor(xy[0]+.5),math.floor(xy[1]+.5);z['pixel']=[x,y]
    continuous=0<=xy[0]<640 and 0<=xy[1]<480;rounded=0<=x<640 and 0<=y<480
    need(continuous==q['original_query_continuous_valid'] and rounded==q['query_nearest_pixel_valid'],'original geometry flags')
    if not continuous:return dict(z,U_reason='ORIGINAL_CONTINUOUS_OUTSIDE_NATIVE')
    if not rounded:return dict(z,U_reason='ORIGINAL_ROUNDED_OUTSIDE_NATIVE')
    z['native_query_valid']=True
    if not prompt_ok:return dict(z,U_reason='PROMPT_UNAVAILABLE')
    if not runtime_ok:return dict(z,U_reason='RUNTIME_UNAVAILABLE')
    x0,x1=max(0,x-2),min(640,x+3);y0,y1=max(0,y-2),min(480,y+3)
    exact=mask[y,x].item();near=np.max(mask[y0:y1,x0:x1]).item();support=exact>0.;n=(y1-y0)*(x1-x0)
    return {'prediction_available':True,'state':STATES[0] if support else STATES[1],'native_query_valid':True,'pixel':[x,y],'exact_logit':exact,'radius2_logit':near,'exact_mask_support':support,'radius2_mask_support':near>0.,'radius2_window_pixels':n,'radius2_clipped':n!=25,'U_reason':None}

def array_equal(a,b,dtype):
    b=np.asarray(b,dtype=dtype);return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b,equal_nan=True)

def flat(row):
    q=row['original_input_unchanged'];xy=q['query_xy'];scores=row['quality3']
    d={'query_id':q['query_id'],'original_query_index':row['original_query_index'],'frame_index':q['frame_index'],'frame_input_id':q['frame_input_id'],'pair_id':q['pair_id'],'view':q['view'],'native_slot':q['native_slot'],'query_x':None if xy is None else xy[0],'query_y':None if xy is None else xy[1],'L':q['roi_side'],'original_continuous_valid':q['original_query_continuous_valid'],'original_nearest_valid':q['query_nearest_pixel_valid']}
    d.update({k:row[k] for k in ('prompt_available','runtime_available','prediction_available','state','U_reason','exact_logit','radius2_logit','exact_mask_support','radius2_mask_support','selected_index')})
    d.update({'quality'+str(k):None if scores is None else scores[k] for k in range(3)})
    return {k:'' if v is None else str(v) for k,v in d.items()}

def write(name,obj):
    p=H/name
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    b=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}

def run(entry_path):
    ident({'path':str(Path(entry_path).absolute()),'sha256':'0'*64});e=json.loads(raw(entry_path))
    need(e['approved_saved_SOURCE_SAM_QA'] is True and e['automatic_retry'] is False,'separate Root true entry required')
    need(ident(e['code'])[0]==str(Path(__file__).absolute()) and hashlib.sha256(raw(__file__)).hexdigest()==e['code']['sha256'],'checker pin')
    mw=e['metadata_whitelist'];nw=e['prediction_NPZ_whitelist'];scope=json.loads(checked(e['root_scope'],mw))
    need(scope['approved'] is True and e['postfreeze_conditional_join'] is True,'Root postfreeze scope')
    for k in ('source_images_or_checkpoint_or_model_imports_allowed','current_or_held_payloads_allowed','training_allowed','PnP_allowed','masknegative_may_become_native_E','model_admission_allowed'):need(scope[k] is False,'saved-only scope '+k)
    need(ident(scope['prediction_freeze'])==ident(e['sam_complete']) and ident(scope['saved_TRAIN_conditional_truth'])==ident(e['conditional_truth']),'scope freeze/late truth identity')
    need({ident(x) for x in scope['mask_outputs']}=={ident(x) for x in nw},'closed saved NPZ scope')
    def meta(b):return json.loads(checked(b,mw))
    c=meta(e['sam_complete'])
    need(c['completed'] is True and c['PASS'] is True and c['fatal'] is None,'44 prediction freeze completed before any teacher read')
    need(ident(c['code'])==ident(e['expected_adapter_code']) and ident(c['protocol'])==ident(e['expected_adapter_protocol']) and c['root_protocol']['sha256']==ROOT_SHA,'frozen adapter/protocol')
    root=meta(c['root_protocol']);need(root['training_admission'] is False and root['model_adopted'] is False,'fixed diagnostic authority')
    required=[e['root_scope'],e['sam_complete'],c['root_protocol'],e['train_complete'],e['train_metadata_QA'],e['conditional_truth'],c['all_rows'],c['query_CSV'],*c['inputs'].values()]
    need(len(mw)==len({ident(x) for x in required}) and {ident(x) for x in mw}=={ident(x) for x in required},'closed ten metadata artifacts')
    for k in ('all_rows','query_CSV'):need(ident(scope[k])==ident(c[k]),'scope saved identity '+k)
    need({k:ident(v) for k,v in scope['original_inputs'].items()}=={k:ident(v) for k,v in c['inputs'].items()},'scope original input identity')
    for k,v in {'frames':44,'queries':352,'model_state_byte_unchanged':True,'checkpoint_file_byte_unchanged':True,'torch_compile_calls':0,'original_continuous_available':297,'original_input_U':55,'threshold_strict_gt0':True,'GT_teacher_pose_depth_model_inputs':False,'new_training_updates':0,'PnP':0,'current_or_held_payloads':0,'native_labels_created':0,'training_admission':False,'model_adopted':False,'automatic_retry':False}.items():need(type(c[k]) is type(v) and c[k]==v,'frozen complete '+k)
    need(c['model_state_before_sha256']==c['model_state_after_sha256'] and isinstance(c['model_state_before_sha256'],str) and len(c['model_state_before_sha256'])==64,'saved state equality metadata')
    need(c['checkpoint']['sha256']==root['model_fixed_before_new_outputs']['checkpoint_sha256'],'fixed official checkpoint metadata only')
    tc=meta(e['train_complete']);qa=meta(e['train_metadata_QA']);need(tc['completed'] is True and tc['PASS'] is True and qa['completed'] is True and qa['PASS'] is True and ident(qa['dataset_complete'])==ident(e['train_complete']),'TRAIN metadata closure')
    for k,n in (('frames','TRAIN44_FRAMES_INPUT_ONLY.jsonl'),('queries','TRAIN352_ORIGINAL_NATIVE_QUERIES_INPUT_ONLY.jsonl')):need(c['inputs'][k]['sha256']==INPUT_SHA[k] and ident(c['inputs'][k])==ident(tc['outputs'][n]),'original input binding '+k)
    frames=[json.loads(x) for x in checked(c['inputs']['frames'],mw).splitlines()];queries=[json.loads(x) for x in checked(c['inputs']['queries'],mw).splitlines()]
    need((len(frames),len(queries))==(44,352) and all(x['split']=='TRAIN_LAB' for x in frames+queries),'TRAIN44/352 only')
    need(sum(q['original_query_continuous_valid'] for q in queries)==297 and all(q['native_semantic_state']=='UNKNOWN' and q['human_gold'] is False for q in queries),'original297/55/native unknown')
    outputs=c['frame_outputs'];need(len(outputs)==44,'all44 frame records')
    declared=[s['output'] for s in outputs if s['output'] is not None]
    need(len(nw)==len(declared)==len({ident(x) for x in nw}) and {ident(x) for x in nw}=={ident(x) for x in declared},'only saved available frame NPZs')
    rows=[json.loads(x) for x in checked(c['all_rows'],mw).splitlines()];need(len(rows)==352,'all352 outputs')
    computed={};tie_frames=0;npz_checks=[]
    for i,(f,s) in enumerate(zip(frames,outputs)):
        need(s['frame_position']==i and s['original_frame_input']==f,'frame identity/order unchanged')
        qs=[q for q in queries if q['frame_input_id']==f['frame_input_id']];need([q['native_slot'] for q in qs]==list(range(8)),'all8 original slots')
        box,why=prompt(qs);pok=box is not None;rok=s['runtime_available']
        need(s['prompt_available'] is pok and s['prompt_reason']==why and s['box_xyxy']==(None if box is None else box.tolist()),'all8 F64 minmax prompt without clipping/availability filtering')
        need(rok or not pok,'PASS run: all valid prompts processed')
        mask=None;scores=None;selected=None;arrays=None
        if rok:
            need(pok and s['runtime_reason'] is None and s['output'] is not None,'available runtime provenance')
            data=checked(s['output'],nw)
            with np.load(io.BytesIO(data),allow_pickle=False) as z:
                need(set(z.files)==NPZ_KEYS,'exact saved-only member schema');arrays={k:z[k] for k in NPZ_KEYS}
            a=arrays;mask=a['chosen_logit'];scores=a['quality_scores']
            need(mask.dtype==np.float32 and mask.shape==(480,640) and np.isfinite(mask).all(),'finite chosen full-native FP32; no half roundtrip')
            need(scores.dtype==np.float32 and scores.shape==(3,) and np.isfinite(scores).all(),'three saved quality floats')
            need(a['selected_index'].dtype==np.int64 and a['selected_index'].shape==(),'scalar selected index')
            selected=int(np.argmax(scores));need(int(a['selected_index'])==s['selected_index']==selected and s['quality3']==scores.tolist(),'argmax quality first exact tie')
            tie_frames+=int(np.count_nonzero(scores==scores[selected])>1)
            expect={'original_prompt_box':(box,np.float64),'predictor_prompt_box':(box.astype(np.float32),np.float32),'raw_hw':([480,640],np.int64),'original_q8':([[np.nan,np.nan] if q['query_xy'] is None else q['query_xy'] for q in qs],np.float64),'original_L':([np.nan if q['roi_side'] is None else q['roi_side'] for q in qs],np.float64),'native_slot':(list(range(8)),np.int8),'original_query_index':([q['frame_index']*8+q['native_slot'] for q in qs],np.int64),'original_continuous_valid':([q['original_query_continuous_valid'] for q in qs],bool),'original_nearest_valid':([q['query_nearest_pixel_valid'] for q in qs],bool)}
            for k,(value,dtype) in expect.items():need(array_equal(a[k],value,dtype),'NPZ original input/dtype '+k)
            need(a['query_id'].dtype.kind=='U' and a['query_id'].shape==(8,) and a['query_id'].tolist()==[q['query_id'] for q in qs],'original query IDs')
            npz_checks.append({'binding':s['output'],'quality_argmax_firsttie':selected,'chosen_native_FP32_readouts_exact':True})
        else:need(not pok and s['output'] is None and s['runtime_reason']=='PROMPT_UNAVAILABLE','prompt-U has no invented saved mask')
        readouts=[readout(mask,q,pok,rok) for q in qs]
        if arrays is not None:
            need(array_equal(arrays['prediction_available'],[x['prediction_available'] for x in readouts],bool),'NPZ prediction availability')
            for k in ('exact_logit','radius2_logit'):need(array_equal(arrays[k],[np.nan if x[k] is None else x[k] for x in readouts],np.float32),'FP32 full-mask scalar readout exact '+k)
        for q,z in zip(qs,readouts):computed[q['query_id']]={'original_input_unchanged':q,'original_query_index':q['frame_index']*8+q['native_slot'],'prompt_available':pok,'runtime_available':rok,'selected_index':selected,'quality3':None if scores is None else scores.tolist(),'box_xyxy':None if box is None else box.tolist(),'scene_output':s['output'],**z,'native_visibility_label':None,'semantic_SELF':None,'training_admission':False}
    expected=[computed[q['query_id']] for q in queries];need(rows==expected,'352 full JSON inputs/availability/logits/support/U/order/authority exact')
    states={key:sum(x['state']==key for x in expected) for key in STATES};need(c['states']==states,'saved exact state census')
    need(c['calls']=={'set_image':len(declared),'predict':len(declared)},'one frozen official API call per processed frame')
    parsed=list(csv.DictReader(io.StringIO(checked(c['query_CSV'],mw).decode(),newline='')))
    need(parsed==[flat(x) for x in expected],'all352 CSV cells independently exact')
    # The conditional table is intentionally loaded only after the frozen packet's full saved-output QA.
    need(e['conditional_truth']['sha256']==TRUTH_SHA,'same frozen V246 conditional table')
    truth=[json.loads(x) for x in checked(e['conditional_truth'],mw).splitlines()];need(len(truth)==352,'same352 conditional rows')
    labels=['CONDITIONAL_VISIBLE_SUPPORT','CONDITIONAL_EXTERNAL_SUPPORT','UNKNOWN'];matrices={mode:[[0]*3 for _ in range(3)] for mode in ('exact','radius2')}
    for q,p,t in zip(queries,expected,truth):
        need(t['original_input_unchanged']==q and t['native_visibility_label'] is None and t['training_admission'] is False,'late same-query truth join; no native promotion')
        for mode,key in (('exact','exact_state'),('radius2','radius2_state')):
            state=t['conditional_pixel_truth'][key];need(state in (-1,0,1),'conditional support encoding')
            rr={0:0,1:1,-1:2}[state]
            cc=2 if not p['prediction_available'] else 0 if p['exact_mask_support' if mode=='exact' else 'radius2_mask_support'] else 1
            matrices[mode][rr][cc]+=1
    need([sum(x) for x in matrices['exact']]==[25,9,318] and [sum(x) for x in matrices['radius2']]==[29,7,316],'fixed conditional strata retained')
    tables={'completed':True,'PASS':True,'rows':352,'row_labels':labels,'column_labels':list(STATES),'tables':matrices,'scope':'Conditional teacher support versus SAM visible-mask support/outside/U; not native accuracy or physical/instance correspondence certification. Outside mask is not an E/SELF label.'}
    table_binding=write('CONDITIONAL_SUPPORT_TABLES.json',tables)
    review={'completed':True,'PASS':True,'sam_complete':e['sam_complete'],'code':e['code'],'root_scope':e['root_scope'],'original_frames':44,'original_queries':352,'original_continuous_available':297,'original_input_U':55,'native_geometry_supported':sum(x['native_query_valid'] for x in expected),'prediction_states':states,'radius2_prediction_states':{key:sum((STATES[2] if not x['prediction_available'] else STATES[0] if x['radius2_mask_support'] else STATES[1])==key for x in expected) for key in STATES},'quality_exact_tie_frames':tie_frames,'NPZ_checks':npz_checks,'all_original_q_L_U_native_ID_order_preserved':True,'table':table_binding,'conditional_truth_loaded_only_after_complete_saved_prediction_QA':True,'model_state_before_after_equality_metadata_checked_only':True,'checkpoint_or_RGB_rawdepth_GT_pose_mesh_models_current_or_held_access':0,'new_native_labels':0,'training_admission':False,'model_adopted':False,'model_data_architecture_threshold_selection':0,'limits':'Quality argmax index and saved chosen-full-FP32 readouts were independently checked. Other two candidate full masks and encoder outputs are not stored/recomputed. No model inference, source teacher reconstruction, instance correspondence or native physical gold certification. Bound-artifact pyc paths rejected; normal library import internal FS is not audited or claimed zero.'}
    print(json.dumps(write('REVIEW.json',review)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);a=p.parse_args()
    try:run(a.entry)
    except BaseException:
        write('FAILED_EXECUTION.json',{'completed':False,'PASS':False,'traceback':traceback.format_exc(),'automatic_retry':False});raise
