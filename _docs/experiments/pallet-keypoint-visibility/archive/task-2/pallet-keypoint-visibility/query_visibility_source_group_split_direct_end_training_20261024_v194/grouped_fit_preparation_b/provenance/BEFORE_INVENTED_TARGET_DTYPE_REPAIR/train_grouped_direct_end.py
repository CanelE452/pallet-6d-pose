"""Fixed from-scratch C56/C57 with one sealed source component TRAIN mask.
No checkpoint loading or real/COMMON/VAL loader. All outputs follow both final8000.
"""
import argparse,hashlib,importlib.util,json,os,stat,time,traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch.nn import functional as F
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
COMMON=P/'query_visibility_sampled_domain_direct_end_20261024_v189/paired_direct_end_fit_preparation_c/direct_source_common.py'
spec=importlib.util.spec_from_file_location('frozen_source_only_common',COMMON);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
PROTOCOL=H/'PROTOCOL_BEFORE_GROUPED_SOURCE_CACHE_ARRAY_OR_FIT_BODIES.json'
MASK=V/'source_component_split_preparation_b/trainer_component_mask_adapter.py'
SCOPE=V/'ROOT_PROSPECTIVE_GROUPED_SOURCE_HEAD_FIT_BEFORE_CACHE_OR_FIT.json'
SPLIT=V/'source_component_split_v1/COMPLETE.json'
RUN=V/'runs/C56_C57_DIRECT_END_GROUPED_SOURCE_FIXED8000'
OLD_ROWS={'SYNTH_AUG65536':65536,'PUBLIC_TRAIN1480':1480,'CF3816':3816}
TARGET5=('support_target','owner_soft','owner_target','owner_ambiguous_mass','owner_invalid_mass')
ARMS=(('C56_RAW96_DIRECT_END_GROUPED',False),('C57_RAW96_PLUS_FROZEN_SAM_DIRECT_END_GROUPED',True))
ROW_OUTPUTS=c.META+('raw_support_logits','raw_owner_logits','raw_direct_end_external_logit','support_prediction','owner_prediction','direct_end_prediction_D0_E1_Uminus1','regional_support_proxy_D0_E1_Uminus1')
ORDERS=('support_order','owner_order','direct_end_order')

def pure_training(head):
    ns=dict(np=np,torch=torch,F=F,require=c.need,head=head,hashlib=hashlib,DOMAIN_ROWS=OLD_ROWS)
    return c.functions(c.OLD_TRAINER,('object_sha','cpu_copy','validate_targets','pools_for','draw_step','five_losses','gather_batch_arrays'),ns)

def metadata_matches(rows,x,split):
    c.need(len(rows)==len(x['row_id']),'All source row metadata')
    for i,r in enumerate(rows):
        c.need(r['split']==split,'SOURCE TRAIN separation')
        for k in c.META:
            expected=c.qarray(r[k]) if k=='query_xy' else (np.nan if r[k] is None else r[k])
            if k in ('query_xy','L'):c.need(np.array_equal(np.asarray(expected),np.asarray(x[k][i]),equal_nan=True),'Original q/L exact')
            else:c.need(expected==x[k][i],'Original source identity/availability '+k)

def load_old(record,n,split,pure):
    x=c.validate_input10(c.arrays(record['inputs'],c.INPUTS),n)
    rows=c.lines(record['metadata']);metadata_matches(rows,x,split)
    y=pure['validate_targets'](c.arrays(record['targets'],TARGET5),x['original_query_available'])
    return dict(x=x,y=y,metadata=rows,record=record)

def load_direct(binding):
    done=c.read_bound(binding);c.need(done['completed'] is True and done['PASS'] is True and done['source_only'] is True and done['domain']=='DIRECT_CONTROL','Root-bound completed direct input cache')
    n=done['rows'];c.need(type(n) is int and n>0,'Dynamic completed source control count')
    x=c.validate_input10(c.arrays(done['inputs'],c.INPUTS),n)
    keys=c.META+('direct_end_external_target','authority');y=c.arrays(done['targets'],keys)
    for k in c.META:c.need(c.equal(y[k],x[k]),'Direct sidecar META6 exact '+k)
    target=y['direct_end_external_target'];c.need(target.dtype==np.int8 and target.shape==(n,) and np.isin(target,[-1,0,1]).all(),'Only direct labels and UNKNOWN')
    c.need((target[~x['original_query_available']]==-1).all(),'InputU never direct known')
    rows=c.lines(done['metadata']);c.need(len(rows)==n,'All constructed source occurrences')
    for i,r in enumerate(rows):
        c.need(r['authority']==str(y['authority'][i]),'Distinct semantic target authority')
        for k in c.META:
            if k=='query_xy':c.need(c.equal(c.qarray(r[k]),x[k][i]),'Direct original q')
            elif k=='L':c.need((r[k] is None and np.isnan(x[k][i])) or r[k]==x[k][i],'Direct original L')
            else:c.need(r[k]==x[k][i],'Direct occurrence metadata '+k)
        c.need(target[i]==-1 or r['root_admitted'] is True,'No direct target without root source admission')
    c.need([int((target==i).sum()) for i in [0,1]]==done['direct_known_counts'],'Cache direct class counts')
    return dict(x=x,y=y,metadata=rows,record=done,complete=binding)

def gather(old,direct,draw,direct_draw,use_sam,pure):
    g=pure['gather_batch_arrays'](old,draw,use_sam)
    if direct_draw is not None:
        for k in g:g[k]=np.concatenate((g[k],direct['x'][k][direct_draw]),axis=0)
    return g

def assemble(g,head,use_sam,device):
    t={k:torch.as_tensor(v,device=device) for k,v in g.items()}
    extra=dict(sam_mask96=t['sam_mask96'],sam_available=t['sam_available96']) if use_sam else {}
    return head.assemble_local_input(t['rgb96'],t['valid96'],use_sam=use_sam,**extra)

def decode(support,owner,direct,av):
    n=len(av);sp=np.full((n,2),-1,np.int8);op=np.full(n,-1,np.int8);dp=np.full(n,-1,np.int8);proxy=np.full(n,-1,np.int8)
    sp[av]=(support[av]>0).astype(np.int8);op[av]=owner[av].argmax(1).astype(np.int8)
    dav=av&np.isfinite(direct);dp[dav]=(direct[dav]>=0).astype(np.int8)
    proxy[av]=np.where(sp[av,1]==1,1,np.where(sp[av,0]==1,0,-1)).astype(np.int8)
    return sp,op,dp,proxy

def objective(a,target,pure,candidate):
    c.need(a['support_logits'].shape==(224,2) and a['owner_logits'].shape==(224,3) and a['direct_end_external_logit'].shape==(224,),'Fixed original192 plus separate direct32')
    base,components=pure['five_losses']({k:a[k][:192] for k in ['support_logits','owner_logits']})
    direct_loss=candidate.balanced_direct_end_loss(a['direct_end_external_logit'][192:],target,torch.ones(32,dtype=torch.bool,device=target.device))
    return base+direct_loss,base,components,direct_loss

def save_outputs(model,cache,path,use_sam,head):
    x=cache['x'];n=len(x['row_id']);support=np.full((n,2),np.nan,np.float32);owner=np.full((n,3),np.nan,np.float32);direct=np.full(n,np.nan,np.float32)
    av=x['original_query_available'];ii=np.flatnonzero(av)
    with torch.inference_mode():
        for start in range(0,len(ii),96):
            take=ii[start:start+96];keys=['rgb96','valid96']+(['sam_mask96','sam_available96'] if use_sam else [])
            inp=assemble({k:x[k][take] for k in keys},head,use_sam,'cuda');a=model(inp);b=model(inp)
            c.need(all(torch.equal(a[k],b[k]) for k in a),'Saved source same-batch repeat exact')
            support[take]=a['support_logits'].cpu().numpy();owner[take]=a['owner_logits'].cpu().numpy();direct[take]=a['direct_end_external_logit'].cpu().numpy()
    sp,op,dp,proxy=decode(support,owner,direct,av)
    return c.save_arrays(path,dict(raw_support_logits=support,raw_owner_logits=owner,raw_direct_end_external_logit=direct,support_prediction=sp,owner_prediction=op,direct_end_prediction_D0_E1_Uminus1=dp,regional_support_proxy_D0_E1_Uminus1=proxy,support_order=np.asarray(['visible_end_support','foreign_front_support']),owner_order=np.asarray(['TARGET','FOREIGN','BACKGROUND']),direct_end_order=np.asarray(['VISIBLE_END0','EXTERNAL_BLOCKED_END1','UNKNOWNminus1']),**{k:x[k] for k in c.META}))


def join_direct_split(metadata,plan,targets):
    """Order-independent exact source control join; no cache-index alias."""
    bykey={tuple(r['control_join_key']):r for r in plan}
    c.need(len(bykey)==len(plan)==len(metadata)==len(targets),'All direct source occurrence keys unique')
    result=[];used=set()
    for i,(m,target)in enumerate(zip(metadata,targets)):
        key=tuple(m['control_join_key']);c.need(key in bykey and key not in used,'One exact direct control key');used.add(key);r=bykey[key]
        c.need(key==(m['domain'],m['attempt_id'],m['variant'],m['native_slot']),'Cache canonical control identity')
        c.need(m['cache_index']==i and m['source_packet']==r['source_packet'] and m['authority']==r['authority'],'Direct original packet/authority/order')
        c.need(int(target)==r['stored_weak_direct_target'] and (target<0 or m['root_admitted'] is True),'Only root-admitted existing direct weak targets')
        # Preserve the immutable plan index separately. Only this adapter copy
        # addresses the cache order; the original split metadata never changes.
        result.append(dict(r,source_split_cache_index=r['cache_index'],cache_index=i))
    c.need(used==set(bykey),'No source control missing or additional')
    return result


def split_masks(split,old,direct,adapter):
    components=c.lines(split['outputs']['ALL_COMPONENTS_AND_CANONICAL_ORIGIN_NODES.jsonl'])
    roles={}
    for r in components:
        cid=hashlib.sha256('\n'.join(sorted(r['origin_nodes'])).encode()).hexdigest()
        expected='HOLDOUT'if int(hashlib.sha256(('pallet_visibility_source_groups_v194_20261024\n'+cid).encode()).hexdigest()[:16],16)%100<20 else'TRAIN'
        c.need(cid==r['component_id']and r['role']==expected and cid not in roles,'Frozen canonical component/hash/role');roles[cid]=expected
    masks={};aligned={}
    for domain,cache in list(old.items())+[('DIRECT_CONTROL',direct)]:
        rr=c.lines(split['outputs'][domain+'_SPLIT_ROWS.jsonl'])
        if domain=='DIRECT_CONTROL':rr=join_direct_split(cache['metadata'],rr,cache['y']['direct_end_external_target'])
        c.need(all(roles.get(r['component_id'])==r['role']for r in rr),'One component role in every loss pool')
        masks[domain]=adapter.fit_mask(cache['metadata'],rr);aligned[domain]=rr
    return masks,aligned,roles


def cohort_counts(cache,mask):
    result={}
    for name,take in [('TRAIN',mask),('HOLDOUT',~mask)]:
        ii=np.flatnonzero(take);record=dict(rows=len(ii),original_inputU=int((~cache['x']['original_query_available'][ii]).sum()),native={str(k):int((cache['x']['native_slot'][ii]==k).sum())for k in range(8)})
        if 'direct_end_external_target'in cache['y']:
            target=cache['y']['direct_end_external_target'][ii];record.update(direct_D=int((target==0).sum()),direct_E=int((target==1).sum()),direct_UNKNOWN=int((target<0).sum()),known_by_authority={str(a):int(((cache['y']['authority'][ii]==a)&(target>=0)).sum())for a in np.unique(cache['y']['authority'])})
        result[name]=record
    return result


def save_cohorts(full_binding,mask,folder,domain,aligned):
    full=c.arrays(full_binding,ROW_OUTPUTS+ORDERS);n=len(mask);out={}
    c.need(all(full[k].shape[0]==n for k in ROW_OUTPUTS),'Full prediction all-row shape')
    for name,take in [('TRAIN',mask),('HOLDOUT',~mask)]:
        ii=np.flatnonzero(take)
        packet={k:full[k][ii]for k in ROW_OUTPUTS};packet.update({k:full[k]for k in ORDERS})
        saved=c.save_arrays(folder/('FINAL_'+domain+'_'+name+'_OUTPUTS.npz'),packet)
        lineage=folder/(domain+'_'+name+'_ROW_COMPONENTS.jsonl')
        c.write_rows(lineage,[dict(cache_index=int(i),row_id=str(full['row_id'][i]),component_id=aligned[i]['component_id'],role=name,original_global_index=int(full['original_global_index'][i]),native_slot=int(full['native_slot'][i]),original_query_available=bool(full['original_query_available'][i]),direct_target_authority=aligned[i].get('authority'),control_join_key=aligned[i].get('control_join_key'))for i in ii])
        out[name]=dict(outputs=saved,metadata=c.bind(lineage),rows=len(ii),original_inputU=int((~full['original_query_available'][ii]).sum()))
    return out


def fresh_run(path):
    p=c.safe(path);c.need(p==RUN,'Fixed fresh grouped run only')
    cur=Path(p.anchor)
    for part in p.parts[1:-1]:
        cur=cur/part
        try:mode=cur.lstat().st_mode
        except FileNotFoundError:cur.mkdir();mode=cur.lstat().st_mode
        c.need(stat.S_ISDIR(mode),'Ordinary output ancestors only')
    p.mkdir(exist_ok=False);return p


def fit(entry_path):
    e=c.read(entry_path);c.need(e['approved_grouped_direct_end_source_fit']is True and e['source_only']is True and e['automatic_retry']is False,'Separate root actual grouped source fit entry')
    p=c.read_bound(e['protocol']);c.need(e['code']==c.bind(__file__)==p['trainer_code']and e['protocol']==c.bind(PROTOCOL)and p['mask_adapter']==c.bind(MASK),'Frozen grouped trainer/protocol/mask adapter')
    for b in p['source_bindings'].values():c.checked(b)
    scope=c.read_bound(p['prospective_scope']);c.read_bound(e['root_scope'])
    c.need(scope['training']==p['training']and scope['source_split']==e['split_complete']==p['split_complete'],'No recipe/split change')
    c.need(e['partition_QA']==p['partition_QA']==scope['source_partition_QA'],'Exact independent partition QA')
    qa=c.read_bound(e['partition_QA']);c.need(qa['completed']is True and qa['PASS']is True and qa['same_source_base_frame_query_donor_or_validinput_fingerprint_never_crosses_role']is True,'Completed disjoint-origin partition')
    c.need(e['augmentation_complete']==p['old_cache_bindings']['augmentation']and e['original_public_CF_complete']==p['old_cache_bindings']['public_CF'],'Original full old5 caches')
    c.need(e['direct_cache_complete']['path']==scope['source_direct_cache_expected']==p['direct_cache_expected'],'Only new expanded source direct cache')
    done=c.read_bound(e['direct_cache_complete']);c.need(done['completed']is True and done['PASS']is True and done['code']==p['new_cache_code']and done['protocol']==p['new_cache_protocol']and done['real_source_replaces_old_whole']is True and done['component_role_masks_applied_to_training']is False,'Expanded all-row cache closure')
    admission=c.read_bound(done['admission']);c.need(admission['completed']is True and admission['PASS']is True and admission['source_only']is True and admission['training_admission_direct_end']is True,'Root actual direct-label admission remains separate')
    c.need(torch.__version__=='2.1.1+cu118'and np.__version__=='1.26.4'and torch.cuda.is_available(),'Pinned FP32 CUDA runtime')
    c.need(scope['arms']==[dict(id=a,use_sam=s)for a,s in ARMS]and scope['architecture']['parameters']==115222 and scope['architecture']['head']==c.bind(c.HEAD)and scope['architecture']['candidate']==c.bind(c.CANDIDATE),'Exact two arms/architecture')
    head=c.module(c.HEAD,'frozen_original_source_head');candidate=c.module(c.CANDIDATE,'frozen_direct_candidate');adapter=c.module(MASK,'frozen_component_mask_adapter');pure=pure_training(head)
    out=fresh_run(e['output_directory']);start=time.monotonic();c.write(out/'STARTED.json',dict(entry=c.bind(entry_path),code=c.bind(__file__),protocol=e['protocol'],source_only=True,from_scratch=True,previous_checkpoint_loaded=False))
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        aug=c.read_bound(e['augmentation_complete']);orig=c.read_bound(e['original_public_CF_complete']);c.need(aug['completed']and aug['PASS']and aug['rows']==65536 and aug['original_aug_inputU']==1264 and aug['SAM_arrays_present']is True,'Full immutable source augmentation')
        old={'SYNTH_AUG65536':load_old(aug,65536,'TRAIN',pure)}
        for name in ['PUBLIC_TRAIN1480','CF3816']:old[name]=load_old(orig['domains'][name],OLD_ROWS[name],'SOURCE_DEVELOPMENT',pure)
        c.need([int((~old[k]['x']['original_query_available']).sum())for k in OLD_ROWS]==[1264,194,360],'Original old-pool inputU unchanged')
        direct=load_direct(e['direct_cache_complete']);c.need(len(direct['metadata'])==3144 and int((~direct['x']['original_query_available']).sum())==112,'Expanded full direct rows/U')
        split=c.read_bound(e['split_complete']);c.need(split['completed']is True and split['PASS']is True,'Frozen source split COMPLETE')
        masks,aligned,roles=split_masks(split,old,direct,adapter);masked=adapter.masked_old_pools(old,masks,pure['pools_for']);target=direct['y']['direct_end_external_target'];dp=adapter.masked_direct_pools(direct['x']['original_query_available'],target,masks['DIRECT_CONTROL'])
        c.need(masked['feasible']and dp['feasible'],'Every original fixed loss class remains trainable; no fallback')
        pools=masked['pools'];dpool=dp['pools'];counts={name:cohort_counts(cache,masks[name])for name,cache in list(old.items())+[('DIRECT_CONTROL',direct)]}
        c.need([counts['DIRECT_CONTROL']['TRAIN'][k]for k in ['direct_D','direct_E']]==[218,109]and[ counts['DIRECT_CONTROL']['HOLDOUT'][k]for k in ['direct_D','direct_E']]==[30,15],'Exact frozen direct split class denominator')
        c.write(out/'FIT_INPUT_VALIDATION.json',dict(completed=True,PASS=True,full_old_rows=OLD_ROWS,direct_rows=3144,cohorts=counts,filtered_old_pools=masked['counts'],filtered_direct_classes=dp['counts'],split_complete=e['split_complete'],partition_QA=e['partition_QA'],all_loss_pools_same_component_TRAIN_mask=True,original_U_targets_qL_native_unchanged=True,full_target_containers_loaded=True,HOLDOUT_labels_not_used_for_draw_or_loss_or_selection=True,no_protected_real_or_COMMON_or_VAL_loader=True))
        newclass=candidate.make_candidate_head_class(head);pending={};initial=None;common_old=None;common_direct=None
        for arm,use_sam in ARMS:
            folder=out/arm;folder.mkdir();torch.manual_seed(1810);torch.cuda.manual_seed_all(1810);model=newclass().float().cuda();ish=pure['object_sha'](model.state_dict());c.need(sum(v.numel()for v in model.parameters())==115222,'Unchanged 115222 parameter head')
            if initial is None:initial=ish
            else:c.need(ish==initial,'Same fresh initialization in both arms')
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);rng=np.random.default_rng(1810);drng=np.random.default_rng(1811);dh=hashlib.sha256();ddh=hashlib.sha256();logpath=folder/'ALL8000_DRAWS_AND_LOSSES.jsonl'
            with logpath.open('x')as log:
                for update in range(1,8001):
                    draw=pure['draw_step'](rng,pools);didx=np.concatenate([drng.choice(a,16,replace=True)for a in dpool]);adapter.assert_draws_train(draw,didx,masks)
                    for key in draw:dh.update(key.encode());dh.update(draw[key].astype('<i8').tobytes())
                    ddh.update(didx.astype('<i8').tobytes());inp=assemble(gather(old,direct,draw,didx,use_sam,pure),head,use_sam,'cuda');opt.zero_grad(set_to_none=True);a=model(inp)
                    dt=torch.as_tensor(target[didx],device='cuda');loss,base,components,direct_loss=objective(a,dt,pure,candidate);c.need(torch.isfinite(loss),'Finite fixed loss');loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);c.need(torch.isfinite(norm),'Finite gradient norm');opt.step();c.need(all(torch.isfinite(t).all()for t in model.parameters()),'Finite parameter state')
                    log.write(json.dumps(dict(update=update,original192_draws={k:v.tolist()for k,v in draw.items()},direct32_draws=didx.tolist(),old_five_losses=[float(z.detach())for z in components],old_five_mean=float(base.detach()),direct_loss=float(direct_loss.detach()),total_loss=float(loss.detach()),grad_norm=float(norm),old_draw_sha=dh.hexdigest(),direct_draw_sha=ddh.hexdigest(),all_drawn_rows_TRAIN=True),allow_nan=False)+'\n')
                    if update%250==0:print(json.dumps(dict(arm=arm,update=update,seconds=time.monotonic()-start)),flush=True)
            oldsha,directsha=dh.hexdigest(),ddh.hexdigest()
            if common_old is None:common_old,common_direct=oldsha,directsha
            else:c.need((oldsha,directsha)==(common_old,common_direct),'Both arms exact filtered old192/direct32 draws')
            checkpoint=folder/'FINAL_CHECKPOINT.pt';torch.save(dict(state_dict=pure['cpu_copy'](model.state_dict()),optimizer_state=pure['cpu_copy'](opt.state_dict()),update=8000,seed=1810,arm=arm,use_sam=use_sam,protocol=e['protocol'],source_split=e['split_complete']),checkpoint)
            model.eval().requires_grad_(False);pending[arm]=dict(model=model,use_sam=use_sam,updates=8000,initial_state_sha=ish,final_state_sha=pure['object_sha'](model.state_dict()),old_draw_sha=oldsha,direct_draw_sha=directsha,checkpoint=c.bind(checkpoint),draws_and_losses=c.bind(logpath));del opt
        barrier=out/'ALL2_FINAL8000_FROZEN_BEFORE_SOURCE_TRAIN_OR_HOLDOUT_OUTPUTS.json';c.write(barrier,dict(completed=True,checkpoints={a:r['checkpoint']for a,r in pending.items()},same_initial_state_sha=initial,same_filtered_old192_draw_sha=common_old,same_direct32_draw_sha=common_direct,source_TRAIN_or_HOLDOUT_prediction_emitted=False,protected_real_COMMON_VAL_access=False,checkpoint_or_arm_selection=False))
        for arm,r in pending.items():
            model=r.pop('model');r['outputs']={};r['source_cohorts']={};folder=out/arm
            for name,cache in list(old.items())+[('DIRECT_CONTROL',direct)]:
                full=save_outputs(model,cache,folder/('FINAL_'+name+'_OUTPUTS.npz'),r['use_sam'],head);r['outputs'][name]=full;r['source_cohorts'][name]=save_cohorts(full,masks[name],folder,name,aligned[name])
            c.need(pure['object_sha'](model.state_dict())==r['final_state_sha'],'Final model byte state unchanged after saved outputs');del model
        for cache in list(old.values())+[direct]:
            for k in ['inputs','targets','metadata']:c.checked(cache['record'][k])
        for b in split['outputs'].values():c.checked(b)
        for b in p['source_bindings'].values():c.checked(b)
        for b in [e['augmentation_complete'],e['original_public_CF_complete'],e['direct_cache_complete'],e['split_complete'],e['partition_QA']]:c.checked(b)
        c.write(out/'COMPLETE.json',dict(completed=True,PASS=True,source_only=True,from_scratch=True,code=c.bind(__file__),protocol=e['protocol'],entry=c.bind(entry_path),head=c.bind(c.HEAD),candidate=c.bind(c.CANDIDATE),mask_adapter=c.bind(MASK),source_split=e['split_complete'],partition_QA=e['partition_QA'],all2_freeze=c.bind(barrier),all2_frozen_before_any_TRAIN_or_HOLDOUT_outputs=True,arms=pending,same_initial_state_sha=initial,same_filtered_old192_draw_sha=common_old,same_direct32_draw_sha=common_direct,steps_each=8000,total_updates=16000,training_caches=dict(augmentation=e['augmentation_complete'],public_CF=e['original_public_CF_complete'],direct=e['direct_cache_complete']),cohort_denominators=counts,source_holdout_caveat='Group-withheld for new from-scratch heads within already-known source design.45direct holdout labels synthetic only; no real-end/freshblind/generalization claim; old C54/C55 not retrospectively validated.',protected_real_COMMON_VAL_access=0,original_native_owner_quality_or_qL_changed=False,selected_model=None,adopted=False,fully_solved=False,seconds=time.monotonic()-start))
        print('ALL2_GROUPED_DIRECT_END_SOURCE_FIT_COMPLETE',flush=True)
    except BaseException:
        c.write(out/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,PASS=False,automatic_retry=False,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--entry',type=Path,required=True);a=ap.parse_args();fit(a.entry)
