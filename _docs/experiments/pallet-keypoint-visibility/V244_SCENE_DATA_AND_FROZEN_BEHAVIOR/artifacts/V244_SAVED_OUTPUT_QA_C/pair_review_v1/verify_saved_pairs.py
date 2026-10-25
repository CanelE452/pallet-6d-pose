"""stdlib saved V235 pair arithmetic only. No feature/model or nested source read.

python -B verify_saved_pairs.py --saved-dir <copied v235_pair_replay_v1>
The fixed three local basenames retain the original SHA/size. Absolute pointers
inside COMPLETE remain inert. Optional --output-directory writes fresh REVIEW.
"""
import argparse,csv,hashlib,io,json,math,os,stat,struct
from pathlib import Path

COMPLETE_SHA='c68b19c4780929d5f0381fd625525711069f8d90d4b4c889df62a7f9dcbaf2f7'
PRODUCER_SHA='c913a1594deff7cdb010a115f2b1571c2a80086f80542c5822c8a16f922b2e3b'
NAMES=('ALL5406_OLD_PAIR_BEHAVIOR.csv','SUMMARY.json')
NULLS=('original_D_prediction','original_D_logit','COVER_logit0','COVER_logit1','SHAM_logit0','SHAM_logit1','COVER_selected_logit','SHAM_selected_logit','COVER_minus_SHAM','COVER_delta_from_original','SHAM_delta_from_original')

def need(ok,message):
    if not ok:raise ValueError(message)

def safe(value):
    p=Path(value);need('..' not in p.parts,'no traversal');p=p.absolute()
    need(not any(x in str(p).casefold() for x in ('v112','__pycache__','.pyc','current223','publicheld','private_evaluation','source_holdout','oldval','v177','v178','v182')),'lexical before filesystem')
    for q in reversed(p.parents):need(stat.S_ISDIR(os.lstat(q).st_mode),'symlink/non-directory ancestor')
    try:need(stat.S_ISDIR(os.lstat(p).st_mode) or stat.S_ISREG(os.lstat(p).st_mode),'symlink/nonregular leaf')
    except FileNotFoundError:pass
    return p

def read(p,b=None):
    with os.fdopen(os.open(safe(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:raw=f.read()
    binding=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
    if b:need(binding['sha256']==b['sha256'] and binding['size_bytes']==b['size_bytes'],'bound saved output changed')
    return raw,binding

def pred(z):return 'E' if z>=0 else 'D'
def boolean(s):need(s in ('True','False'),'CSV bool');return s=='True'
def finite(s):
    z=float(s);need(math.isfinite(z),'finite saved scalar');return z
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def median(xs):
    s=sorted(xs);n=len(s);return s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2

def same(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if a is None:return b is None
    if isinstance(a,float):return isinstance(b,(int,float)) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)
    return a==b

def metrics(rows):
    out={}
    for name,family in [('ALL',None),('SYN_MAIN',0),('YCB_ASSOCIATED_R_AUX',1)]:
        take=[r for r in rows if r['success'] and (family is None or r['family']==family)];n=len(take)
        cover=sum(r['COVER_prediction']=='E' for r in take);sham=sum(r['SHAM_prediction']=='D' for r in take)
        both=sum(r['COVER_prediction']=='E' and r['SHAM_prediction']=='D' for r in take);diff=[r['COVER_minus_SHAM'] for r in take]
        originallyD=[r for r in take if r['original_D_prediction']=='D']
        out[name]=dict(successful_pairs=n,COVER_E=cover,SHAM_D=sham,COVER_E_and_SHAM_D=both,
            COVER_gt_SHAM=sum(x>0 for x in diff),COVER_eq_SHAM=sum(x==0 for x in diff),COVER_lt_SHAM=sum(x<0 for x in diff),
            COVER_minus_SHAM_mean=math.fsum(diff)/n if n else None,COVER_minus_SHAM_median=median(diff) if n else None,
            COVER_E_rate=cover/n if n else None,SHAM_D_rate=sham/n if n else None,paired_crossing_rate=both/n if n else None,
            original_saved_D_correct=len(originallyD),original_D_to_COVER_E=sum(r['COVER_prediction']=='E' for r in originallyD),
            SHAM_preserves_original_correct_D=sum(r['SHAM_prediction']=='D' for r in originallyD),SHAM_preservation_denominator=len(originallyD))
    return out

def verify(saved):
    saved=safe(saved);raw,cb=read(saved/'COMPLETE.json');need(cb['sha256']==COMPLETE_SHA,'exact frozen diagnostic COMPLETE')
    c=json.loads(raw);need(c['completed'] is True and c['PASS'] is True and c['schema']=='V244_FROZEN_OLD_V235_PAIR_BEHAVIOR_RECONSTRUCTION' and c['code']['sha256']==PRODUCER_SHA,'frozen producer schema/code')
    need(c['threshold_logit']==0 and c['new_optimizer_or_training']==0 and c['model_threshold_or_data_selection'] is False and c['absent_historical_pair_features_or_logits_byte_comparison'] is False,'read-only reconstruction scope')
    need(c['base_state_sha_before']==c['base_state_sha_after'] and c['head_state_sha_before']==c['head_state_sha_after'],'upstream frozen-state receipt')
    inputs={'COMPLETE':cb};data={}
    for name in NAMES:
        b=c['outputs'][name];need(Path(b['path']).name==name,'fixed local basename; never follow absolute path')
        raw,inputs[name]=read(saved/name,b)
        data[name]=list(csv.DictReader(io.StringIO(raw.decode(),newline=''))) if name.endswith('.csv') else json.loads(raw)
    rows=data[NAMES[0]];s=data['SUMMARY.json'];need(len(rows)==5406 and len({r['parent_sample_index'] for r in rows})==5406 and len({r['row_id'] for r in rows})==5406,'all5406 unique original parents')
    family_counts={f:{'attempted':0,'success':0,'failed':0} for f in (0,1)}
    for i,r in enumerate(rows):
        r['pair_index']=int(r['pair_index']);r['parent_sample_index']=int(r['parent_sample_index']);r['family']=int(r['family']);r['success']=boolean(r['success'])
        f=r['family'];need(r['pair_index']==i and 0<=r['parent_sample_index']<7617 and f in (0,1) and r['source_role']=='FIT' and boolean(r['original_teacher_D']) and not boolean(r['native_gold']),'original FIT D order/family/authority')
        need(r['object_or_logical_aux']=='UNKNOWN_PENDING_SAVED_METADATA_JOIN','no guessed object/aux mapping')
        cnt=family_counts[f];cnt['attempted']+=1;cnt['success' if r['success'] else 'failed']+=1
        if not r['success']:
            need(int(r['COVER_target'])==int(r['SHAM_target'])==-1 and r['COVER_prediction']==r['SHAM_prediction']=='U' and all(r[k]=='' for k in NULLS),'all39 failure null/U/target-1 retained; original teacher D unchanged')
            continue
        need(int(r['COVER_target'])==1 and int(r['SHAM_target'])==0,'same old weak COVER/SHAM targets')
        for k in NULLS:
            if k!='original_D_prediction':r[k]=finite(r[k])
        for role in ('COVER','SHAM'):
            need(all(r[role+'_logit'+str(k)]==f32(r[role+'_logit'+str(k)]) for k in (0,1)),'both family logits FP32')
            z=r[role+'_logit'+str(f)];need(r[role+'_selected_logit']==z and r[role+'_prediction']==pred(z),'family-selected scalar threshold0, not argmax')
        cc=r['COVER_selected_logit'];ss=r['SHAM_selected_logit'];old=r['original_D_logit']
        need(r['original_D_prediction']==pred(old),'saved original natural D model prediction')
        need(r['COVER_minus_SHAM']==cc-ss and r['COVER_delta_from_original']==cc-old and r['SHAM_delta_from_original']==ss-old,'exact stored difference/delta arithmetic')
    need(family_counts=={0:{'attempted':4565,'success':4537,'failed':28},1:{'attempted':841,'success':830,'failed':11}},'original all5406/success5367/fail39 family census')
    m=metrics(rows)
    need(s['completed'] is True and s['PASS'] is True and s['status']=='FROZEN_SOURCE_FIT_REPLAY_NOT_TRAINING' and s['attempts']==5406 and s['successful_pairs']==5367 and s['successful_role_images']==10734 and s['failed_pairs_U']==39,'summary denominator/status')
    need(same(m,s['family_metrics']),'all family metrics/rates/direction/median/original-D crossings and preservation independently match')
    for v in m.values():need(v['COVER_gt_SHAM']+v['COVER_eq_SHAM']+v['COVER_lt_SHAM']==v['successful_pairs'],'direction partition')
    return dict(completed=True,PASS=True,pair_diagnostic_complete=cb,inputs=inputs,all5406_rows_preserved=True,family_counts=family_counts,family_metrics=m,all_failed39_null_U_targetminus1=True,all_success_both_family_FP32_and_selected_binary_zero_decode=True,all_original_D_deltas_exact=True,scalar_statistics_comparison_tolerance=1e-10,raw_arrays_features_checkpoints_DINO_RGB_GT_or_current_held_reads=0,new_training_or_model_selection=False,current_admission=False,limits=['Saved CSV/summary arithmetic only; upstream frozen encoder/head/feature reconstruction is cited, not reexecuted.','All pairs are preexposed original SOURCE FIT training pairs, not independent natural DEV/generalization.','SYN intended-native witness0 and YCB associated-r AUX remain separate; no physical native or literal-q gold upgrade.','Original natural saved logits came from the old prefix route; pair inputs use the old CF fullforward reconstruction. Delta/preservation is descriptive, not byte comparison to absent original pair features.','Failed39 retain original_teacher_D while child inputs/predictions are unavailable; they are not relabeled or filtered. Object/aux mapping stays UNKNOWN.'])

def write(p,value):
    with os.fdopen(os.open(safe(p),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return read(p)[1]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--saved-dir',required=True);p.add_argument('--output-directory');a=p.parse_args()
    result=verify(a.saved_dir)
    if a.output_directory:
        out=safe(a.output_directory);need(not out.exists(),'fresh QA output only');out.mkdir();result['verifier_code']=read(Path(__file__))[1]
        receipt=write(out/'REVIEW.json',result);print(json.dumps({'REVIEW':receipt,'family_metrics':result['family_metrics']},ensure_ascii=False))
    else:print(json.dumps(result,ensure_ascii=False))
