"""Fixed source-only RAW/SAM + separate direct-end logit; root entry only.

No validation/current/held/COMMON loader exists. Both final8000 states freeze
before saved training outputs. Same source draw192 plus separate direct draw32.
"""
import argparse, hashlib, os, time, traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch.nn import functional as F
import direct_source_common as c
H,V,P=c.H,c.V,c.P
PROTOCOL=H/'PROTOCOL_BEFORE_DIRECT_CONTROL_CACHE_OR_FIT.json'
OLD_ROWS={'SYNTH_AUG65536':65536,'PUBLIC_TRAIN1480':1480,'CF3816':3816}
TARGET5=('support_target','owner_soft','owner_target','owner_ambiguous_mass','owner_invalid_mass')
ARMS=(('C54_RAW96_DIRECT_END',False),('C55_RAW96_PLUS_FROZEN_SAM_DIRECT_END',True))

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

def fit(entry_path):
    e=c.read(entry_path);c.need(e['approved_paired_direct_end_source_fit'] is True and e['source_only'] is True,'Separate root actual fit entry')
    p=c.read_bound(e['protocol']);c.need(e['code']==c.bind(__file__) and p['trainer_code']==e['code'],'Frozen trainer/protocol')
    for b in p['source_bindings'].values():c.checked(b)
    c.need(e['augmentation_complete']==p['unchanged_TRAIN_caches']['augmentation_TRAIN_complete'] and e['original_public_CF_complete']==p['unchanged_TRAIN_caches']['original_public_CF_cache_complete'],'Exact unchanged inherited TRAIN cache bindings')
    c.need(p['training']['steps']==8000 and p['training']['seed']==1810 and p['training']['direct_weight']==1 and p['training']['direct_batch']==[16,16],'Prospective fixed source-only objective')
    c.need(torch.__version__=='2.1.1+cu118' and np.__version__=='1.26.4' and torch.cuda.is_available(),'Pinned CUDA runtime')
    head=c.module(c.HEAD,'v189_original_raw_head');candidate=c.module(c.CANDIDATE,'v189_direct_candidate');pure=pure_training(head)
    c.read_bound(e['root_scope']);out=c.fresh_output(e['output_directory']);start=time.monotonic()
    c.write(out/'STARTED.json',dict(entry=c.bind(entry_path),code=c.bind(__file__),protocol=e['protocol'],source_only=True,new_evaluation_access=False))
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        aug=c.read_bound(e['augmentation_complete']);orig=c.read_bound(e['original_public_CF_complete'])
        c.need(aug['completed'] and aug['PASS'] and aug['rows']==65536 and aug['original_aug_inputU']==1264 and aug['SAM_arrays_present'] is True,'All inherited augmented source TRAIN')
        old={'SYNTH_AUG65536':load_old(aug,65536,'TRAIN',pure)}
        for name in ['PUBLIC_TRAIN1480','CF3816']:old[name]=load_old(orig['domains'][name],OLD_ROWS[name],'SOURCE_DEVELOPMENT',pure)
        c.need([int((~old[k]['x']['original_query_available']).sum()) for k in OLD_ROWS]==[1264,194,360],'Original source U denominators')
        pools=pure['pools_for'](old);direct=load_direct(e['direct_cache_complete']);target=direct['y']['direct_end_external_target'];dpool=[np.flatnonzero(direct['x']['original_query_available']&(target==i)) for i in [0,1]];c.need(all(len(a)>0 for a in dpool),'Both known direct classes')
        c.write(out/'FIT_INPUT_VALIDATION.json',dict(completed=True,PASS=True,old_rows=OLD_ROWS,direct_rows=len(target),direct_classes=[len(a) for a in dpool],direct_UNKNOWN=int((target<0).sum()),authority_values=sorted(set(direct['y']['authority'].tolist())),original_public_CF_owner_only=True,old_five_mean_preserved=True,root_direct_admission=e['direct_cache_complete'],evaluation_or_COMMON_or_held_or_private_access=0))
        newclass=candidate.make_candidate_head_class(head);pending={};initial=None;common_old_draw=None;common_direct_draw=None
        for arm,use_sam in ARMS:
            directory=out/arm;directory.mkdir();torch.manual_seed(1810);torch.cuda.manual_seed_all(1810);model=newclass().float().cuda();ish=pure['object_sha'](model.state_dict());c.need(sum(v.numel() for v in model.parameters())==115222,'Same old trunk+65')
            if initial is None:initial=ish
            else:c.need(ish==initial,'Both candidate arms same initialization')
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);rng=np.random.default_rng(1810);drng=np.random.default_rng(1811);dh=hashlib.sha256();ddh=hashlib.sha256();logpath=directory/'ALL8000_DRAWS_AND_LOSSES.jsonl'
            with logpath.open('x') as log:
                for update in range(1,8001):
                    draw=pure['draw_step'](rng,pools);didx=np.concatenate([drng.choice(a,16,replace=True) for a in dpool])
                    for key in draw:dh.update(key.encode());dh.update(draw[key].astype('<i8').tobytes())
                    ddh.update(didx.astype('<i8').tobytes());inp=assemble(gather(old,direct,draw,didx,use_sam,pure),head,use_sam,'cuda');opt.zero_grad(set_to_none=True);a=model(inp)
                    dt=torch.as_tensor(target[didx],device='cuda');loss,base,components,direct_loss=objective(a,dt,pure,candidate)
                    c.need(torch.isfinite(loss),'Finite fixed additive total');loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);c.need(torch.isfinite(norm),'Finite clipped norm');opt.step();c.need(all(torch.isfinite(t).all() for t in model.parameters()),'Finite updated state')
                    log.write(__import__('json').dumps(dict(update=update,original192_draws={k:v.tolist() for k,v in draw.items()},direct32_draws=didx.tolist(),old_five_losses=[float(z.detach()) for z in components],old_five_mean=float(base.detach()),direct_loss=float(direct_loss.detach()),total_loss=float(loss.detach()),grad_norm=float(norm),old_draw_sha=dh.hexdigest(),direct_draw_sha=ddh.hexdigest()),allow_nan=False)+'\n')
                    if update%250==0:print(__import__('json').dumps(dict(arm=arm,update=update,seconds=time.monotonic()-start)),flush=True)
            oldsha,directsha=dh.hexdigest(),ddh.hexdigest()
            if common_old_draw is None:common_old_draw,common_direct_draw=oldsha,directsha
            else:c.need((oldsha,directsha)==(common_old_draw,common_direct_draw),'Both arms exact original and direct draw streams')
            checkpoint=directory/'FINAL_CHECKPOINT.pt';torch.save(dict(state_dict=pure['cpu_copy'](model.state_dict()),optimizer_state=pure['cpu_copy'](opt.state_dict()),update=8000,seed=1810,arm=arm,use_sam=use_sam,protocol=e['protocol']),checkpoint)
            model.eval().requires_grad_(False);pending[arm]=dict(model=model,use_sam=use_sam,updates=8000,initial_state_sha=ish,final_state_sha=pure['object_sha'](model.state_dict()),old_draw_sha=oldsha,direct_draw_sha=directsha,checkpoint=c.bind(checkpoint),draws_and_losses=c.bind(logpath));del opt
        barrier=out/'ALL2_FINAL8000_FROZEN_BEFORE_ANY_SAVED_OUTPUTS.json';c.write(barrier,dict(completed=True,checkpoints={a:r['checkpoint'] for a,r in pending.items()},same_initial_state_sha=initial,old_same_draw_sha=common_old_draw,direct_same_draw_sha=common_direct_draw,evaluation_inputs_targets_or_scores_read=False,saved_outputs_emitted=False))
        for arm,r in pending.items():
            model=r.pop('model');r['outputs']={}
            for name,cache in list(old.items())+[('DIRECT_CONTROL',direct)]:r['outputs'][name]=save_outputs(model,cache,out/arm/('FINAL_'+name+'_OUTPUTS.npz'),r['use_sam'],head)
            c.need(pure['object_sha'](model.state_dict())==r['final_state_sha'],'Frozen final model unchanged');del model
        for b in [e['augmentation_complete'],e['original_public_CF_complete'],e['direct_cache_complete']]:c.checked(b)
        for cache in list(old.values())+[direct]:
            for k in ['inputs','targets','metadata']:c.checked(cache['record'][k])
        for b in p['source_bindings'].values():c.checked(b)
        c.write(out/'COMPLETE.json',dict(completed=True,PASS=True,source_only=True,code=c.bind(__file__),head=c.bind(c.HEAD),candidate=c.bind(c.CANDIDATE),protocol=e['protocol'],entry=c.bind(entry_path),all2_freeze=c.bind(barrier),all2_frozen_before_outputs=True,arms=pending,same_initial_state_sha=initial,same_old192_draw_sha=common_old_draw,same_direct32_draw_sha=common_direct_draw,updates_each=8000,total_updates=16000,training_caches=dict(augmentation=e['augmentation_complete'],public_CF=e['original_public_CF_complete'],direct=e['direct_cache_complete']),evaluation_or_COMMON_or_publicheld_or_private_access=0,native_owner_quality_relabel=0,seconds=time.monotonic()-start,selected_model=None,adopted=False,fully_solved=False))
        print('ALL2_DIRECT_END_SOURCE_FIT_COMPLETE',flush=True)
    except BaseException:
        c.write(out/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',type=c.Path,required=True);a=p.parse_args();fit(a.entry)
