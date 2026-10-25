"""One final matched data addition; pinned existing head, no encoder or warmstart.

Import is inert. Actual feature/label access and optimization require one exact
Root true-entry/scope. Current/v61 evaluation is a separate post-freeze action.
"""
import argparse, copy, csv, hashlib, io, json, os, stat, sys, traceback, types
from collections import Counter
from pathlib import Path
import numpy as np
import torch

sys.dont_write_bytecode=True
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
POLICY={'path':str(R/'V261_ROOT_FINAL_PUBLIC_REAL_COMPARISON/PROSPECTIVE_FINAL_ONE_COMPARISON_POLICY.json'),'sha256':'5bfeaf8d690b39c6e9c287d1b25340a1ee7b538d90d2d9447be9fa61e94da54c','size_bytes':4732}
UTILITY={'path':str(R/'V256_MATCHED_DATA_TRAINER_A/train_matched_data.py'),'sha256':'3ec2d316b6dfde1c6ab22f8fb8287fed8a873e29708fe8b44b8e32e8bd20c494','size_bytes':28632}
CROP={'path':str(R/'V233_DETERMINISTIC_CROP_FIX_A/crop_modules.py'),'sha256':'44f54dfcbf1671a401e748e66ec30463096e6c24b0779f07dda85d7c9f38ae88','size_bytes':4262}
CONTROL='CONTROL_ORIGINAL_SOURCE_FIT'
TREAT='TREAT_ORIGINAL_PLUS_PUBLIC_SURFACE_AUX'
ARMS=(CONTROL,TREAT)
INITIAL_SHA='d1b8da9721a5492dcfb166cf78943eb5db18c5b14b8e8fed328ccab31c12648c'
INPUT_KEYS=('original_feature_complete','original_features','original_samples',
            'public_feature_complete','public_features','public_samples','public_records',
            'public_frame_manifest','public_feature_verification')
PUBLIC_KEYS=('target','family','fit','available','row_id','frame_id','coarse_group_id')

def need(ok,message):
    if not bool(ok):raise ValueError(message)

def safe(value,exists=True):
    s=str(value)
    need(not any(x in s.casefold() for x in ('v112','cache','.pyc','current223','publicheld','oldval','holdout','private_evaluation')),'Forbidden path before filesystem access')
    p=Path(s);need(p.is_absolute() and '..' not in p.parts,'Exact absolute path')
    for a in [*reversed(p.parents),p]:
        try:mode=os.lstat(a).st_mode
        except FileNotFoundError:
            need(not exists,'Missing exact artifact');continue
        need(not stat.S_ISLNK(mode),'Symlink artifact/ancestor rejected')
    return p

def bkey(b):
    need(set(b)=={'path','sha256','size_bytes'},'Exact binding schema')
    safe_lexical=str(b['path'])
    need(not any(x in safe_lexical.casefold() for x in ('v112','cache','.pyc','current223','publicheld','oldval','holdout','private_evaluation')),'Forbidden binding before FS')
    need(Path(safe_lexical).is_absolute() and '..' not in Path(safe_lexical).parts,'Absolute binding')
    need(isinstance(b['sha256'],str) and len(b['sha256'])==64 and type(b['size_bytes']) is int and b['size_bytes']>=0,'Exact SHA/size')
    return b['path'],b['sha256'],b['size_bytes']

def read_bound(b,whitelist):
    need(bkey(b) in {bkey(x) for x in whitelist},'Explicit Root binding before read')
    raw=safe(b['path']).read_bytes()
    need(hashlib.sha256(raw).hexdigest()==b['sha256'] and len(raw)==b['size_bytes'],'Bound artifact SHA/size')
    return raw

def bind(p):
    raw=safe(p).read_bytes();return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}

def write_json(p,x):
    p=safe(p,False);need(not p.exists(),'Fresh immutable output')
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return bind(p)

def load_code(b,whitelist,name):
    raw=read_bound(b,whitelist);m=types.ModuleType(name);m.__file__=b['path'];sys.modules[name]=m
    exec(compile(raw,m.__file__,'exec'),m.__dict__);return m

def build_model(crop_module):
    """Existing CropFamilyHead76098 only, fresh seed handled by caller."""
    model=crop_module.CropFamilyHead()
    need(sum(p.numel() for p in model.parameters())==76098 and len(list(model.parameters()))==4,'Original four parameter tensors only')
    return model

def common_uniform_draws(samples,eligible_control,eligible_treat,steps=1000):
    """Same four U streams; pool-size remapping is the sole draw intervention."""
    n=len(samples['target'])
    need(eligible_control.shape==eligible_treat.shape==(n,) and eligible_control.dtype==eligible_treat.dtype==bool,'Explicit eligibility vectors')
    for eligible in (eligible_control,eligible_treat):
        need((~eligible | (samples['fit'] & samples['available'] & np.isin(samples['target'],[0,1]))).all(),'Only known available TRAIN rows in loss')
    chunks_c=[];chunks_t=[];streams=[];pool_c=[];pool_t=[]
    for family in (0,1):
        for target in (0,1):
            c=np.flatnonzero(eligible_control & (samples['family']==family) & (samples['target']==target))
            t=np.flatnonzero(eligible_treat & (samples['family']==family) & (samples['target']==target))
            need(len(c)>0 and len(t)>0,'No missing-class fallback')
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([17,family,target])))
            u=rng.random((steps,8));need(((u>=0)&(u<1)).all(),'Fixed U in [0,1)')
            chunks_c.append(c[np.floor(u*len(c)).astype(np.int64)])
            chunks_t.append(t[np.floor(u*len(t)).astype(np.int64)])
            streams.append(u);pool_c.append(c);pool_t.append(t)
    dc=np.concatenate(chunks_c,1);dt=np.concatenate(chunks_t,1)
    need(np.array_equal(dc[:,:16],dt[:,:16]),'Same SYN pools and all SYN draws byteexact')
    need(dc.shape==dt.shape==(steps,32),'Matched amount:1000x32, family8D8E')
    return dc,dt,np.stack(streams),pool_c,pool_t

def public_samples(raw,utility):
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:
        need(set(z.files)==set(PUBLIC_KEYS),'Public AUX NPZ exact seven nonpixel members')
        s={k:z[k].copy() for k in PUBLIC_KEYS}
    n=utility.validate_samples(s)
    need(all(s[k].shape==(n,) and s[k].dtype.kind=='U' for k in ('frame_id','coarse_group_id')),'Explicit public frame/group IDs')
    need((s['family']==1).all() and np.isin(s['target'],[0,1]).all(),'Selected public surface D/E AUX only; teacher U stays in full ledgers')
    return s

def validate_public_manifest(samples,record_bytes,frame_bytes):
    frames=json.loads(frame_bytes)['frames'];need(len(frames)==284,'Every fixed public frame retained')
    by_frame={};groups={};census=Counter()
    for frame in frames:
        key=frame['publisher_frame_path'];need(key not in by_frame and type(frame['fit']) is bool,'Unique public publisher path/role')
        by_frame[key]=frame;group=frame['coarse_group_id']
        need(group not in groups or groups[group]==frame['fit'],'Sibling/coarse group inherits one role')
        groups[group]=frame['fit'];census['TRAIN' if frame['fit'] else 'DEV']+=1
        need(set(frame['label_counts'])=={'D','E','U'} and all(type(v)is int and v>=0 for v in frame['label_counts'].values()),'Full teacher D/E/U census retained, U is not abstention')
        need(frame['selected_counts']=={c:min(64,frame['label_counts'][c]) for c in ('D','E')},'Frozen class64 caps including empty cells')
    need(census=={'TRAIN':168,'DEV':116},'Prospective frame split168/116 unchanged')
    records=[json.loads(line) for line in record_bytes.splitlines() if line]
    need(len(records)==len(samples['target']),'One original public record per selected feature row')
    counts=Counter();seen=set()
    for i,row in enumerate(records):
        frame_id=str(samples['frame_id'][i]);need(frame_id in by_frame,'Frame whitelist metadata join')
        f=by_frame[frame_id];target=int(samples['target'][i]);state='D' if target==0 else 'E'
        need(row['sample_index']==i and row['row_id']==str(samples['row_id'][i]) and row['publisher_frame_path']==frame_id and row['coarse_group_id']==str(samples['coarse_group_id'][i])==f['coarse_group_id'],'Exact public row/frame/coarse-group join')
        need(bool(samples['fit'][i])==f['fit'],'Public fixed role unmodified')
        need(row['q_xy']==row['r_xy'] and row['q_offset_index']==0 and row['L_pixels']==96,'q=r offset0/L96; no displaced-query label copying')
        q=np.asarray(row['q_xy'],np.float64);need(q.shape==(2,) and np.isfinite(q).all(),'Finite original emulated query')
        need(row['native_slot'] is None and row['SELF_state']=='UNKNOWN','Surface AUX never native/SELF gold')
        g=row['reference_grid_index'];need(type(g)is int and g>=0 and (frame_id,g)not in seen,'Unique r grid identity')
        seen.add((frame_id,g));counts[(frame_id,state)]+=1
    for frame_id,f in by_frame.items():
        for state in ('D','E'):need(counts[(frame_id,state)]==f['selected_counts'][state],'All selected rows retained including unsupported input')
    return records,frames,{'frames':284,'TRAIN_frames':168,'DEV_frames':116,'coarse_groups':len(groups),'selected_rows':len(records),'teacher_U_excluded_from_sampling_and_loss':sum(f['label_counts']['U'] for f in frames),'teacher_U_is_model_abstention':False,'all_frames_including_empty_retained':True,'full_hash_rank_selection_proof_authority':'Pinned Root public feature verification; trainer checks selected row/cap/role/q=r joins, not raw teacher archives'}

def write_ledger(path,arm,draws,samples,original_n):
    exposures=Counter()
    with safe(path,False).open('x',encoding='utf-8') as f:
        for step,ids in enumerate(draws,1):
            row_ids=samples['row_id'][ids].tolist();exposures.update(row_ids)
            f.write(json.dumps({'arm':arm,'update':step,'sample_indices':ids.tolist(),'row_ids':row_ids,'source':['ORIGINAL_SOURCE' if i<original_n else 'PUBLIC_SURFACE_AUX' for i in ids],'family':samples['family'][ids].tolist(),'target':samples['target'][ids].tolist(),'all_fit_available_known':True},sort_keys=True,allow_nan=False)+'\n')
    return bind(path),dict(sorted(exposures.items()))

def public_prediction_rows(head,features,samples,records,device,utility):
    rows=utility.prediction_rows(head,features,samples,device)
    for i,row in enumerate(rows):
        row.update({'publisher_frame_path':str(samples['frame_id'][i]),'coarse_group_id':str(samples['coarse_group_id'][i]),'reference_grid_index':records[i]['reference_grid_index'],'q_equals_r':True,'native_slot':None,'label_namespace':'PUBLIC_CONDITIONAL_SURFACE_AUX'})
    return rows

def write_csv(path,rows):
    with safe(path,False).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return bind(path)

def run(entry_path):
    entry_raw=safe(entry_path).read_bytes();entry=json.loads(entry_raw)
    need(entry.get('approved_actual_final_public_aux_fit') is True,'Separate Root true entry required before data/optimizer')
    own=bind(Path(__file__).absolute())
    need(entry['code']==own and entry['root_policy']==POLICY and entry['utility_code']==UTILITY and entry['crop_code']==CROP,'Pinned final-one producer/policy/old functions')
    code_white=entry['code_whitelist'];meta_white=entry['metadata_whitelist'];data_white=entry['data_whitelist']
    need({bkey(x)for x in code_white}=={bkey(x)for x in (own,UTILITY,CROP)} and len(code_white)==3,'Closed three CODE bindings, no encoder')
    need(meta_white==[POLICY],'Only frozen final-one scientific policy')
    need({bkey(x)for x in data_white}=={bkey(entry[k])for k in INPUT_KEYS} and len(data_white)==len(INPUT_KEYS),'Only exact original/public packet closure')
    scope=json.loads(read_bound(entry['root_scope'],[entry['root_scope']]))
    need(scope.get('approved_actual_final_public_aux_fit') is True and scope.get('entry_contract')=={k:v for k,v in entry.items()if k!='root_scope'},'Exact Root scope contract')
    need(entry.get('automatic_retry') is False and entry.get('current_or_v61_or_held_inputs') is False and entry.get('encoder_or_warmstart_checkpoint_inputs') is False,'Final one comparison; protected/eval/checkpoint inputs excluded')
    policy=json.loads(read_bound(POLICY,meta_white));need(policy['model_training']['updates_per_arm']==1000,'Frozen final1000')
    utility=load_code(UTILITY,code_white,'v261_existing_v256_utility');crop=load_code(CROP,code_white,'v261_existing_crop_module')
    original_complete=json.loads(read_bound(entry['original_feature_complete'],data_white))
    public_complete=json.loads(read_bound(entry['public_feature_complete'],data_white))
    need(original_complete.get('completed')is True and public_complete.get('completed')is True,'Two completed exact feature packets')
    need(original_complete['features']['ORIGINAL_ALL_FEATURES.npy']==entry['original_features'] and original_complete['samples_npz']==entry['original_samples'],'Original canonical packet unchanged')
    need(all(public_complete[k]==entry[ek] for k,ek in [('features','public_features'),('samples','public_samples'),('records','public_records'),('frame_manifest','public_frame_manifest')]),'Public direct feature/sample/record/frame COMPLETE bindings')
    verification=json.loads(read_bound(entry['public_feature_verification'],data_white))
    need(verification.get('completed')is True and verification.get('PASS')is True and verification.get('feature_complete')==entry['public_feature_complete'],'Root verification pins the actual public feature packet')
    old=utility.samples_from(entry['original_samples']);public=public_samples(read_bound(entry['public_samples'],data_white),utility)
    records,frames,public_census=validate_public_manifest(public,read_bound(entry['public_records'],data_white),read_bound(entry['public_frame_manifest'],data_white))
    old_n=len(old['target']);public_n=len(public['target'])
    need(old_n==7617 and public_n>0,'Original7617 and nonempty new public packet')
    old_fit=old['fit'] & old['available'] & np.isin(old['target'],[0,1]);public_fit=public['fit'] & public['available'] & np.isin(public['target'],[0,1])
    need([int(np.sum(old_fit & (old['family']==f)))for f in (0,1)]==[4752,1124] and public_fit.any(),'Original FIT4752SYN/1124YCB plus actual public TRAIN input')
    base_keys=('target','family','fit','available','row_id');combined={k:np.concatenate((old[k],public[k]))for k in base_keys};utility.validate_samples(combined)
    eligible_c=np.r_[old_fit,np.zeros(public_n,bool)];eligible_t=np.r_[old_fit,public_fit]
    dc,dt,u,pc,pt=common_uniform_draws(combined,eligible_c,eligible_t)
    old_x=utility.open_features(entry['original_features'],old_n);public_x=utility.open_features(entry['public_features'],public_n)
    # Opaque all-file identity/header checks above; only FIT rows are numerically fetched here.
    fit_np=np.concatenate((np.array(old_x[np.flatnonzero(old_fit)],copy=True),np.array(public_x[np.flatnonzero(public_fit)],copy=True)))
    need(fit_np.dtype==np.float32 and np.isfinite(fit_np).all() and fit_np.shape[1]==1186,'Finite unchanged FIT features')
    old_fit_f=old['family'][old_fit];old_cond=fit_np[:int(old_fit.sum()),-32:]
    need(np.isin(old_cond[old_fit_f==0,:8],[0,1]).all() and (old_cond[old_fit_f==0,:8].sum(1)==1).all(),'SYN native address unchanged')
    need(not old_cond[old_fit_f==1].any() and not fit_np[int(old_fit.sum()):,-32:].any(),'YCB/public AUX zero32, no fabricated native slots')
    utility.configure_runtime();device=torch.device(entry['device'])
    torch.manual_seed(17);initial=build_model(crop);initial_state=utility.cpu_copy(initial.state_dict());initial_sha=utility.state_sha(initial_state)
    need(initial_sha==INITIAL_SHA,'Exact historical seed17 fresh CropFamilyHead initial state')
    heads={}
    for arm in ARMS:
        heads[arm]=build_model(crop).to(device);heads[arm].load_state_dict(initial_state,strict=True)
        need(utility.state_sha(heads[arm].state_dict())==initial_sha,'Both arms identical initial76098')
    fit_ids=np.flatnonzero(eligible_t);fit_map=np.full(len(combined['target']),-1,np.int64);fit_map[fit_ids]=np.arange(len(fit_ids));fit_tensor=torch.as_tensor(fit_np,device=device)
    out=safe(entry['output_directory'],False);need(not out.exists(),'Fresh output; no resume/retry');out.mkdir()
    written=[];finals={};stage='prefit_admission'
    try:
        np.save(out/'COMMON_FOUR_UNIFORM_STREAMS.npy',u,allow_pickle=False);ub=bind(out/'COMMON_FOUR_UNIFORM_STREAMS.npy');written.append(ub)
        draws={CONTROL:dc,TREAT:dt};ledgers={};exposures={}
        for arm in ARMS:
            ledgers[arm],exposures[arm]=write_ledger(out/(arm+'_DRAW1000.jsonl'),arm,draws[arm],combined,old_n);written.append(ledgers[arm])
        eb=write_json(out/'ACTUAL_ROW_EXPOSURES.json',exposures);written.append(eb)
        admission=write_json(out/'FIT_ADMISSION_AND_COMMON_RANDOM_QA.json',{'completed':True,'PASS':True,'final_one_policy':POLICY,'original_rows':old_n,'public_selected_rows':public_n,'original_FIT_by_family':[4752,1124],'public_TRAIN_available_known':int(public_fit.sum()),'public_census':public_census,'pool_sizes_CONTROL':[len(p)for p in pc],'pool_sizes_TREAT':[len(p)for p in pt],'all_SYNdrawing_indices_byteexact':np.array_equal(dc[:,:16],dt[:,:16]),'same_auxiliary_U_stream_and_amount':True,'common_uniform_streams':ub,'same_initial_sha256':initial_sha,'head_parameters_each':76098,'numeric_feature_rows_before_freeze_only_FIT':True,'SOURCE_and_public_DEV_nonpixel_metadata_decoded':True,'DEV_metadata_used_for_training_or_recipe_selection':False,'DEV_head_forward_calls_before_freeze':0,'teacher_U_loss_rows':0,'teacher_U_not_model_abstention':True,'new_ENCODER_updates':0,'current_v61_or_held_payload_reads':0});written.append(admission)
        provenance={'entry':{'path':str(entry_path),'sha256':hashlib.sha256(entry_raw).hexdigest(),'size_bytes':len(entry_raw)},'policy':POLICY,'producer_code':own,'utility_code':UTILITY,'crop_code':CROP,'original_feature_complete':entry['original_feature_complete'],'public_feature_complete':entry['public_feature_complete'],'public_feature_verification':entry['public_feature_verification'],'draw_policy':'Four common U PCG64 streams floor-remapped per arm; not a claim to reproduce historical V256 integer-choice draws'}
        for arm,eligible in [(CONTROL,eligible_c),(TREAT,eligible_t)]:
            stage=arm
            finals[arm]=utility.fit_arm(arm,heads[arm],crop,combined,fit_tensor,fit_map,eligible,draws[arm],device,out,initial_state,dict(provenance,draw_ledger=ledgers[arm],common_u=ub))
            written.extend([finals[arm]['checkpoint'],finals[arm]['log']])
        need(set(finals)==set(ARMS),'Both final1000 and train logs complete')
        freeze=write_json(out/'BOTH_FINAL1000_LOGS_DRAWS_FROZEN_BEFORE_ANY_DEV_OR_CURRENT_OUTPUT.json',{'completed':True,'PASS':True,'checkpoints_and_logs':finals,'draw_ledgers':ledgers,'common_uniform_streams':ub,'row_exposures':eb,'fit_admission':admission,'initial_state_sha256':initial_sha,'updates':{a:1000 for a in ARMS},'head_parameters':{a:76098 for a in ARMS},'optimizer_states':{a:4 for a in ARMS},'DEV_or_current_or_v61_head_forwards_before_barrier':0,'current_or_v61_or_held_payloads_read_by_trainer':0,'chronology_evidence':'Sealed CODE order plus frozen saved bindings; this artifact is not an independent live trace','one_matched_comparison_not_two_recipe_attempts':True});written.append(freeze)
        stage='postfreeze_fixed_final_readout';source_rows={};pub_rows={};source_outputs={};public_outputs={}
        for arm in ARMS:
            source_rows[arm]=utility.prediction_rows(heads[arm],old_x,old,device);source_outputs[arm]=utility.write_rows(out/(arm+'_ORIGINAL_ALL_SOURCE_ROWS.csv'),source_rows[arm]);written.append(source_outputs[arm])
            pub_rows[arm]=public_prediction_rows(heads[arm],public_x,public,records,device,utility);public_outputs[arm]=write_csv(out/(arm+'_ALL_PUBLIC_SELECTED_POINT_ROWS.csv'),pub_rows[arm]);written.append(public_outputs[arm])
            need(utility.state_sha(heads[arm].state_dict())==finals[arm]['final_state_sha256'],'Frozen state unchanged after all final outputs')
        summary={'completed':True,'arms':list(ARMS),'final_one_comparison':True,'new_updates_each':1000,'head_parameters_each':76098,'same_initial_sha256':initial_sha,'classifier_raw_logit_threshold':0,'source_metrics_exposed_descriptive':{a:utility.scalar_metrics(rr)for a,rr in source_rows.items()},'public_selected_point_metrics':{a:utility.scalar_metrics(rr)for a,rr in pub_rows.items()},'public_metrics_by_frame':{a:{f['publisher_frame_path']:utility.scalar_metrics([r for r in rr if r['publisher_frame_path']==f['publisher_frame_path']])for f in frames}for a,rr in pub_rows.items()},'public_metrics_by_coarse_group':{a:{g:utility.scalar_metrics([r for r in rr if r['coarse_group_id']==g])for g in sorted({f['coarse_group_id']for f in frames})}for a,rr in pub_rows.items()},'paired_SOURCE_changes':utility.paired_summary(source_rows[CONTROL],source_rows[TREAT]),'paired_public_changes':utility.paired_summary(pub_rows[CONTROL],pub_rows[TREAT]),'source_original_native_address_metrics':{a:utility.native_metrics(rr)for a,rr in source_rows.items()},'public_census':public_census,'source_SYN_native_MAIN_and_YCB_associated_r_AUX_namespaces_preserved':True,'public_surface_is_native_gold':False,'public_U_is_model_abstention':False,'precision_caveat':'Class-stratified capped sample precision, not natural population precision; report counts/recall/coarse groups. Repeated frames/objects are correlated.','protected_public45_capture_disjointness_certified':False,'current_v61_evaluation':'Separate Root-only same-query evaluation permitted only after this final freeze; not performed by trainer','automatic_model_adoption':False,'subsequent_fit_or_threshold_retry_permitted':False,'PnP_changes':0}
        sb=write_json(out/'SUMMARY.json',summary);written.append(sb)
        stateqa=write_json(out/'FINAL_HEAD_STATES_UNCHANGED_AFTER_READOUT.json',{'completed':True,'PASS':True,'final_state_sha256':{a:finals[a]['final_state_sha256']for a in ARMS},'after_readout_state_sha256':{a:utility.state_sha(heads[a].state_dict())for a in ARMS},'freeze':freeze});written.append(stateqa)
        manifest=write_json(out/'MANIFEST.json',{'files':written,'treewalk':False})
        complete=write_json(out/'COMPLETE.json',{'completed':True,'PASS':True,**provenance,'inputs':{k:entry[k]for k in INPUT_KEYS},'checkpoints':{a:finals[a]['checkpoint']for a in ARMS},'logs':{a:finals[a]['log']for a in ARMS},'draw_ledgers':ledgers,'common_uniform_streams':ub,'row_exposures':eb,'fit_admission':admission,'freeze':freeze,'source_outputs':source_outputs,'public_outputs':public_outputs,'summary':sb,'head_state_after_readout_QA':stateqa,'manifest':manifest,'new_updates':{a:1000 for a in ARMS},'head_parameters_each':76098,'initial_state_sha256':initial_sha,'one_final_matched_comparison_consumed':True,'subsequent_automatic_retry':False,'current_v61_or_held_payload_reads':0,'model_adopted':False})
        print(json.dumps({'completed':True,'PASS':True,'COMPLETE':complete}),flush=True);return complete
    except BaseException as exc:
        write_json(out/'FAILURE_NO_AUTOMATIC_RETRY.json',{'completed':False,'stage':stage,'error':str(exc),'traceback':traceback.format_exc(),'completed_arms':finals,'written_files':written,'automatic_retry':False,'no_new_recipe_or_threshold_retry_authorized':True})
        raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--entry',required=True);run(ap.parse_args().entry)
