"""Bounded saved-fit evidence audit. No torch/model/array/image/GT decoding."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import traceback
import numpy as np

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_dense_point_weak_bridge_source_training_20261025_v212'
H=V/'independent_actual_fit_review_c'
RUN=V/'runs/C62_C63_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIXED4000'
FIT_ENTRY=V/'ROOT_ACTUAL_C62_C63_SOURCE_POINT_WEAK_BRIDGE_FIT_ENTRY.json'
PROTOCOL=H/'PROTOCOL_BEFORE_COMPLETED_SOURCE_FIT_EVIDENCE.json'
ARMS=('C62_DENSE_POINT_RAW_Q8','C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8')
DOMAINS=('SYNTH_AUG65536','PUBLIC_TRAIN1480','CF3816')
NAMES=('visible','foreign','synthetic_owner','public_owner','CF_owner')
DRAW_DOMAIN=(DOMAINS[0],)*3+DOMAINS[1:]
DRAW_LENGTHS=(32,32,48,48,32)

def need(v,m):
    if not bool(v):raise AssertionError(m)
def literal(p):
    p=Path(p);need(p.is_absolute()and'..'not in p.parts and'v112'not in str(p).lower(),'Lexical path before filesystem');return p
def rb(path,checkpoint=False):
    p=literal(path)
    need((checkpoint and p.suffix=='.pt'and p.name=='FINAL_CHECKPOINT.pt'and p.parent.parent==RUN and p.parent.name in ARMS)
         or(not checkpoint and p.suffix in('.json','.jsonl','.py','.md')and p.name!='SCENES_GT_ONLY.json'and not p.name.startswith('BASE_CONFIGS')),'Exact text or new owned opaque checkpoint only')
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for i,part in enumerate(p.parts[1:]):
            nxt=os.open(part,os.O_RDONLY|os.O_NOFOLLOW|(os.O_DIRECTORY if i<len(p.parts)-2 else 0),dir_fd=fd);os.close(fd);fd=nxt
        need(stat.S_ISREG(os.fstat(fd).st_mode),'Regular nofollow file')
        with os.fdopen(fd,'rb')as f:fd=-1;return f.read()
    finally:
        if fd>=0:os.close(fd)
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(rb(p)).hexdigest())
def checked(b):
    data=rb(b['path']);need(hashlib.sha256(data).hexdigest()==b['sha256'],'Exact metadata SHA');return data
def read(b):return json.loads(checked(b))
def lines(b):return [json.loads(x)for x in checked(b).splitlines()if x.strip()]
def save(path,value):
    p=literal(path);need(p.parent in(H,H/'actual_review_v1'),'Own outputs only')
    with p.open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def known_binding(b,path):
    need(set(b)=={'path','sha256'}and b['path']==str(path)and re.fullmatch('[0-9a-f]{64}',b['sha256']),'Exact inert artifact binding shape/path')
def finite_xy(q):return isinstance(q,list)and len(q)==2 and all(type(x)in(int,float)and math.isfinite(x)for x in q)

def point_target(r):
    q,L,ref=r['query_xy'],r['L'],r['saved_projected_reference_xy'];loc=r['reference_locality_for_actual_input']
    valid=finite_xy(q)and type(L)in(int,float)and math.isfinite(L)and L>0 and finite_xy(ref)
    if valid:
        uv=[47.5+96*(b-a)/L for a,b in zip(q,ref)]
        need(uv==loc['raw96_reference_index_xy']and all(-.5<=x<95.5 for x in uv)==loc['inside_crop_canvas_extent'],'Actual input crop locality exact')
    eligible=(r['component_role']=='TRAIN'and r['mapping_status']=='EXACT_ORIGINAL_IMAGE_OWN_SCENE'and r['state_name']=='NOMINAL'and r['state_index']==0
        and r['family']in('SELF_SCENE15','BEFORE_ANCHOR4')and r['original_query_available']is True and valid and r['point0_target_alone_state']=='D'
        and r['point0_combined_state']in('D','E')and loc['inside_original_frame_canvas_extent']is True and loc['inside_crop_canvas_extent']is True)
    return int(r['point0_combined_state']=='E')if eligible else -1

def log_arithmetic(r,bridge):
    values=r['old_five_losses'];need(len(values)==5,'Five original auxiliary terms')
    vals=values+[r[k]for k in('old_five_mean','point_loss','weak_loss','total_loss','grad_norm')]
    need(all(type(x)in(int,float)and math.isfinite(x)and x>=0 for x in vals),'Finite nonnegative losses/gradient norm')
    need(abs(sum(values)/5-r['old_five_mean'])<=max(2e-6,abs(r['old_five_mean'])*2e-6),'Logged original mean arithmetic')
    need(r['weak_loss_weight']==(.25 if bridge else 0.),'Separate weak weight')
    if not bridge:need(r['weak_loss']==0 and r['weak16_draws']is None,'C62 no weak draw/loss')
    expected=r['old_five_mean']+r['point_loss']+(.25*r['weak_loss']if bridge else 0.)
    need(abs(expected-r['total_loss'])<=max(2e-6,abs(expected)*2e-6),'Logged additive objective arithmetic')

def gate(e):
    need(e.get('approved_independent_postfit_evidence_review')is True and e.get('root_terminal_exit0')is True
        and e.get('both_final4000_complete')is True and e.get('automatic_retry')is False,'Separate root post-terminal actual audit entry')

def prepare():
    r=dict(old_five_losses=[1.]*5,old_five_mean=1.,point_loss=2.,weak_loss=4.,weak_loss_weight=.25,total_loss=4.,grad_norm=5.,weak16_draws=list(range(16)))
    log_arithmetic(r,True);bad=dict(r,total_loss=9.);rejected=False
    try:log_arithmetic(bad,True)
    except AssertionError:rejected=True
    need(rejected,'Incorrect objective rejected')
    need(np.array_equal(np.array([True,True,False])&np.array([True,False,True]),np.array([True,False,False])),'Fit eligibility does not reassign roles')
    prng=np.random.default_rng(1811);pools=[np.arange(20),np.arange(20,30)];take=np.concatenate([prng.choice(a,16,replace=True)for a in pools]);need(int((take<20).sum())==16,'Balanced point draws')
    rejected=False
    try:gate(dict(approved_independent_postfit_evidence_review=False))
    except AssertionError:rejected=True
    need(rejected,'False entry stops before bodies')
    rejected=False
    try:rb('/forbidden/V112/no.json')
    except AssertionError:rejected=True
    need(rejected,'V112 lexical guard before filesystem')
    checks=dict(completed=True,PASS=True,invented_only=True,checks=6,actual_source_run_or_checkpoint_body_reads=0)
    save(H/'INVENTED_CHECKS.json',checks)
    fitentry=read(bind(FIT_ENTRY));trainer_protocol=read(fitentry['protocol'])
    protocol=dict(completed=True,code=bind(__file__),checks=bind(H/'INVENTED_CHECKS.json'),fit_entry=bind(FIT_ENTRY),fit_scope=fitentry['root_scope'],
        trainer=fitentry['code'],trainer_protocol=fitentry['protocol'],source_code_checks=fitentry['checks'],independent_code_preflight=fitentry['independent_preflight'],
        teacher_QA=trainer_protocol['teacher_QA'],
        access='Only exact terminal COMPLETE/barrier/FIT_INPUT_VALIDATION/point7040 target JSONL/all8000 log rows/new2 checkpoint opaque SHA plus declared original text metadata. No source NPZ/PNG/GT/config or saved-prediction array payload reads. No torch/model decoder/forward.',
        checks_scope='All logged old192 rows TRAIN&nominal&available and paired hashes; point1811/weak1812 exact RNG replay from allowed teacher/weak metadata; balances16/16 and8/8; finite objective arithmetic; same init/final4000 and declared118359; barrier/outputs binding closure and unchanged text hashes.',
        limits='No old TARGET5 arrays: old5 class-membership/conditional probability RNG and neural losses not independently reconstructed. No checkpoint decoding: tensor/optimizer/state claims use exact code/preflight+metadata, plus new2 opaque SHA. Barrier chronology uses frozen code sequence and receipt, not outputNPZ timestamps.',automatic_retry=False)
    save(PROTOCOL,protocol)
    false=dict(approved_independent_postfit_evidence_review=False,root_terminal_exit0=False,both_final4000_complete=False,automatic_retry=False,
        code=bind(__file__),protocol=bind(PROTOCOL),root_scope=fitentry['root_scope'],fit_entry=bind(FIT_ENTRY),source_complete=None,
        all2freeze=None,fit_input_validation=dict(path=str(RUN/'FIT_INPUT_VALIDATION.json'),sha256=None),
        point_loss_rows=dict(path=str(RUN/'ALL7040_POINT_LOSS_TARGETS_AND_ELIGIBILITY.jsonl'),sha256=None),output_dir=str(H/'actual_review_v1'))
    save(H/'ROOT_ACTUAL_AUDIT_ENTRY_FALSE_TEMPLATE.json',false)
    save(H/'FINAL_READY.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(H/'INVENTED_CHECKS.json'),false_entry=bind(H/'ROOT_ACTUAL_AUDIT_ENTRY_FALSE_TEMPLATE.json'),actual_partial_run_or_checkpoint_reads=0))
    print(json.dumps(dict(completed=True,PASS=True,ready=bind(H/'FINAL_READY.json'))))

def audit(entry_path):
    e=read(bind(entry_path));gate(e);need(e['code']==bind(__file__)and e['protocol']==bind(PROTOCOL),'Frozen reader code/protocol')
    p=read(e['protocol']);need(e['fit_entry']==p['fit_entry']and e['root_scope']==p['fit_scope'],'Exact fit/root authority')
    fitentry=read(e['fit_entry']);scope=read(e['root_scope']);need(scope['approved_actual_SOURCE_fit']is True,'Root actual fit scope')
    trainer_protocol=read(p['trainer_protocol']);need(fitentry['code']==p['trainer']and fitentry['protocol']==p['trainer_protocol'],'Frozen fitted trainer/protocol')
    need(read(p['source_code_checks'])['PASS']and read(p['independent_code_preflight'])['PASS']and read(p['teacher_QA'])['PASS'],'Prior numerical/code checks remain closed')
    known_binding(e['source_complete'],RUN/'COMPLETE.json');complete=read(e['source_complete'])
    need(complete['completed']is True and complete['PASS']is True and complete['entry']==e['fit_entry']and complete['code']==p['trainer']and complete['protocol']==p['trainer_protocol'],'Exact completed source run')
    need(complete['parameters_each']==118359 and complete['steps_each']==4000 and complete['total_updates']==8000 and complete['from_scratch']is True and complete['source_only']is True,'Fixed from-scratch two4000 recipe')
    need(complete['protected_evaluation_access']==complete['protected_real_COMMON_VAL_access']==0 and complete['selected_model']is None and complete['adopted']is False,'No fit selection/protected access claimed')
    need(complete['main_target_replaces_old_finite_end_DIRECT']is True and complete['old_five_aux_unchanged']is True and complete['original_qL_native_U_changed']is False,'Input/target contract')
    need(set(complete['arms'])==set(ARMS)and complete['all2_frozen_before_any_SOURCE_outputs']is True,'Both arms closure')
    need(e['all2freeze']==complete['all2_freeze'],'Exact all2 freeze binding');known_binding(e['all2freeze'],RUN/'ALL2_FINAL4000_FROZEN_BEFORE_ANY_SOURCE_OUTPUTS.json');barrier=read(e['all2freeze'])
    need(barrier['completed']is True and barrier['source_predictions_emitted']is False and barrier['checkpoint_or_arm_selection']is False,'Pre-output freeze receipt')
    known_binding(e['fit_input_validation'],RUN/'FIT_INPUT_VALIDATION.json');validation=read(e['fit_input_validation']);need(validation['completed']and validation['PASS'],'Fit validation')
    need([validation[k]for k in('point_all7040','point_D','point_E','point_excluded_UNKNOWN','weak_rows','weak_D','weak_E')]==[7040,5853,243,944,44,26,18],'Actual runtime source denominators')
    for key in('all_old5_masks_unchanged','original_U_qL_native_q8_unchanged','original_roles_not_relabelled','teacher_reference_or_region_or_alpha_not_model_input','no_old_DIRECT_NPZ_read'):need(validation[key]is True,'Runtime input guard '+key)
    out=literal(e['output_dir']);need(out==H/'actual_review_v1','Exact fresh audit output');out.mkdir(exist_ok=False)
    touched=[p['trainer'],p['trainer_protocol'],p['source_code_checks'],p['independent_code_preflight'],p['teacher_QA'],e['fit_entry'],e['root_scope'],e['source_complete'],e['all2freeze'],e['fit_input_validation']]
    try:
        cb=fitentry['context_input_bindings'];metadata={};fitmask={};available={};identities={};original_roles={}
        for domain in DOMAINS:
            rr=lines(cb[domain]);roles=lines(cb[domain+'_roles']);need(len(rr)==len(roles),'Immutable metadata role row count')
            mask=[]
            for i,(r,s)in enumerate(zip(rr,roles)):
                need(s['cache_index']==i and s['row_id']==r['row_id'],'Exact original role/cache join')
                for key in('original_global_index','native_slot','query_xy','L','original_query_available'):need(s[key]==r[key],'Original role META '+key)
                nominal=domain!=DOMAINS[0]or r['family']=='BEFORE_ANCHOR4'or(r['family']=='SELF_SCENE15'and r['state_name']=='NOMINAL'and r['state_index']==0)
                mask.append(s['role']=='TRAIN'and nominal)
            metadata[domain]=rr;original_roles[domain]=roles;fitmask[domain]=np.asarray(mask,bool);available[domain]=np.asarray([r['original_query_available']for r in rr],bool);identities[domain]={r['row_id']:i for i,r in enumerate(rr)}
            need(validation['fit_rows'][domain]==int(fitmask[domain].sum()),'Runtime fit-mask denominator exact');touched.extend([cb[domain],cb[domain+'_roles']])
        pa=read(trainer_protocol['point_admission']);mapped=lines(pa['exact_teacher_mapping']);weak=read(trainer_protocol['weak_metadata']);touched.extend([trainer_protocol['point_admission'],pa['exact_teacher_mapping'],trainer_protocol['weak_metadata']])
        target=np.full(len(metadata[DOMAINS[0]]),-1,np.int8);expected_rows=[]
        for r in mapped:
            i=identities[DOMAINS[0]][r['row_id']];m=metadata[DOMAINS[0]][i]
            need(r['input_cache_index']==i and fitmask[DOMAINS[0]][i],'Exact original point TRAIN/nominal index')
            for key in('original_global_index','native_slot','query_xy','L','original_query_available','source_image_original_global_index','anchor_original_global_index'):need(m[key]==r[key],'Point image/anchor metadata exact')
            need(r['source_teacher_key']==[m['source_image_original_global_index']//8,m['native_slot']],'Own-image teacher key')
            target[i]=point_target(r);expected_rows.append((r['row_id'],int(target[i])))
        need(len(mapped)==7040 and Counter(target[target>=0].tolist())=={0:5853,1:243},'Full7040 admitted point labels')
        known_binding(e['point_loss_rows'],RUN/'ALL7040_POINT_LOSS_TARGETS_AND_ELIGIBILITY.jsonl');stored_targets=lines(e['point_loss_rows']);touched.append(e['point_loss_rows'])
        need([(r['row_id'],r['loss_target'])for r in stored_targets]==expected_rows,'Actual recorded point eligibility rows exact')
        pp=[np.flatnonzero(fitmask[DOMAINS[0]]&available[DOMAINS[0]]&(target==k))for k in(0,1)]
        wt=np.asarray([r['proposed_weak_target']for r in weak['rows']],np.int8);need(len(wt)==44 and Counter(wt.tolist())=={0:26,1:18},'Separate weak44 target provenance')
        need(all(r['component_role']=='TRAIN'and r['original_query_available']is True for r in weak['rows']),'Weak TRAIN/available')
        wp=[np.flatnonzero(wt==k)for k in(0,1)];arm_checks={};predictions=[];opaque_cp=[]
        for arm,bridge in zip(ARMS,(False,True)):
            record=complete['arms'][arm];folder=RUN/arm
            need(record['updates']==4000 and record['use_sam']is False and record['spatial_q8']is True and record['weak_bridge']is bridge and record['weak_loss_weight']==(.25 if bridge else 0.),'Arm flags/final step')
            need(record['initial_state_sha']==complete['same_initial_state_sha']==barrier['same_initial_state_sha'],'Same fresh initialization receipts')
            cp=record['checkpoint'];known_binding(cp,folder/'FINAL_CHECKPOINT.pt');need(barrier['checkpoints'][arm]==cp,'Exact checkpoint in pre-output barrier')
            need(hashlib.sha256(rb(cp['path'],True)).hexdigest()==cp['sha256'],'New owned checkpoint opaque SHA only');opaque_cp.append(cp)
            logb=record['draws_and_losses'];known_binding(logb,folder/'ALL4000_DRAWS_AND_LOSSES.jsonl');log=lines(logb);touched.append(logb);need(len(log)==4000,'All4000 logs, no partial state')
            prng=np.random.default_rng(1811);wrng=np.random.default_rng(1812);dh=hashlib.sha256();ph=hashlib.sha256();wh=hashlib.sha256()
            for update,row in enumerate(log,1):
                need(row['update']==update and row['all_drawn_rows_TRAIN']is True,'Every update sequential/TRAIN')
                draw=row['original192_draws'];need(tuple(draw)==NAMES,'Original draw order')
                for name,domain,n in zip(NAMES,DRAW_DOMAIN,DRAW_LENGTHS):
                    ids=np.asarray(draw[name],np.int64);need(ids.shape==(n,)and np.all((ids>=0)&(ids<len(metadata[domain]))),'Original pool draw lengths/range')
                    need(fitmask[domain][ids].all()and available[domain][ids].all(),'No HOLDOUT/jitter/U old loss row');dh.update(name.encode());dh.update(ids.astype('<i8').tobytes())
                point=np.asarray(row['point32_draws'],np.int64);expected=np.concatenate([prng.choice(a,16,replace=True)for a in pp])
                need(np.array_equal(point,expected)and Counter(target[point].tolist())=={0:16,1:16},'Exact point1811 replay/class balance');ph.update(point.astype('<i8').tobytes())
                if bridge:
                    wi=np.asarray(row['weak16_draws'],np.int64);expected=np.concatenate([wrng.choice(a,8,replace=True)for a in wp])
                    need(np.array_equal(wi,expected)and Counter(wt[wi].tolist())=={0:8,1:8},'Exact weak1812 replay/class balance');wh.update(wi.astype('<i8').tobytes())
                need(row['old_draw_sha']==dh.hexdigest()and row['point_draw_sha']==ph.hexdigest()and row['weak_draw_sha']==wh.hexdigest(),'Every logged prefix draw SHA')
                log_arithmetic(row,bridge)
            need(record['old_draw_sha']==dh.hexdigest()==complete['same_old192_draw_sha']==barrier['same_old192_draw_sha'],'Both final old draw hashes')
            need(record['point_draw_sha']==ph.hexdigest()==complete['same_point32_draw_sha']==barrier['same_point32_draw_sha'],'Both final point draw hashes')
            need(record['weak_draw_sha']==wh.hexdigest(),'Final weak draw hash')
            for domain in DOMAINS:
                known_binding(record['outputs'][domain],folder/('FINAL_'+domain+'_OUTPUTS.npz'));predictions.append(record['outputs'][domain])
                for role in('TRAIN','HOLDOUT'):
                    cohort=record['source_cohorts'][domain][role];known_binding(cohort['outputs'],folder/('FINAL_'+domain+'_'+role+'_OUTPUTS.npz'));predictions.append(cohort['outputs'])
                    expected_rows=sum(r['role']==role for r in original_roles[domain])
                    need(cohort['rows']==expected_rows,'Saved cohorts original role, not fit-mask complement')
            for domain in('POINT7040','WEAK44'):known_binding(record['outputs'][domain],folder/('FINAL_'+domain+'_OUTPUTS.npz'));predictions.append(record['outputs'][domain])
            need(set(record['outputs'])==set(DOMAINS)|{'POINT7040','WEAK44'},'No oldDIRECT outputs')
            arm_checks[arm]=dict(completed=True,PASS=True,updates=4000,all_old192_draws_TRAIN_nominal_available=True,point32_D16_E16_exact_RNG1811=True,
                weak16_D8_E8_exact_RNG1812=bridge,finite_logged_losses_and_gradnorm=True,objective_arithmetic_exact_with_FP32_tolerance=True,
                checkpoint_opaque_sha_verified=True,checkpoint_model_or_optimizer_not_decoded=True)
        need(len(predictions)==len({b['path']for b in predictions})==22,'22 inert source output bindings, zero payload reads')
        for b in trainer_protocol['source_bindings'].values():checked(b);touched.append(b)
        for b in touched:checked(b)
        review=dict(completed=True,PASS=True,source_complete=e['source_complete'],fit_complete=e['source_complete'],trainer=p['trainer'],protocol=p['trainer_protocol'],all2freeze=e['all2freeze'],
            audit_code=bind(__file__),audit_protocol=e['protocol'],root_actual_audit_entry=bind(entry_path),fit_input_validation=e['fit_input_validation'],point_loss_rows=e['point_loss_rows'],
            arms=arm_checks,all8000_draws_checked=True,parameters_each_declared=118359,final_steps_each=4000,paired_old_point_hashes_exact=True,
            new_owned_checkpoint_opaque_sha_bindings=opaque_cp,source_output_bindings_metadata_only=predictions,
            point7040_counts=dict(D=5853,E=243,UNKNOWN_loss=944),weak44_counts=dict(D=26,E=18),source_metadata_final_hashes_unchanged=True,
            no_source_NPZ_PNG_GT_or_saved_prediction_payload_reads=True,no_model_checkpoint_tensor_optimizer_decode_or_forward=True,no_protected_scores_or_selection=True,
            limits=p['limits'],no_independent_neural_training_reproduction=True)
        save(out/'REVIEW.json',review)
        with (out/'KO.md').open('x')as f:f.write('# V212 저장 source-fit 독립 검산\n\nPASS. 양팔 final4000·같은 초기 state receipt·118359 선언·두 체크포인트 opaque SHA와 양 final barrier를 확인했다. 실제8000 update의 old192 원 TRAIN/nominal/available, point32 D16/E16·RNG1811, C63 weak16 D8/E8·RNG1812와 모든 prefix/final draw hash를 검산했다. finite losses/gradient norm와 old-five mean + point + 0.25 weak 산술이 일치했다. 원 role의 TRAIN/HOLD cohort 분모를 유지하며 제외 jitter를 HOLD로 바꾸지 않는다.\n\n원 target5 NPZ·RGB/PNG·GT·saved prediction 배열을 읽지 않았다. 따라서 old5 조건부 pool 확률 전체·neural loss 재현·체크포인트 tensor/optimizer 내부 검산은 주장하지 않는다. source input의 실제 payload 해시는 동결 trainer runtime guard에 의존하고 C는 metadata/code binding closure만 독립 재확인했다. 양 final barrier 순서는 receipt와 동결 코드로 확인하며 배열 timestamp 검사는 하지 않았다. 실사 개선·모델 선택·fresh blind 일반화를 주장하지 않는다.\n')
        print(json.dumps(dict(completed=True,PASS=True,review=bind(out/'REVIEW.json'),fit_complete=e['source_complete'],all8000=True)))
    except BaseException:
        save(out/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry');args=parser.parse_args()
    prepare()if args.prepare else audit(args.entry)
