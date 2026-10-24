"""V176 C49 only: pinned V175 pure helpers, fresh head, separate CF314 loss.

--prepare reads metadata/code and invented arrays only. No source score/feature
or TRAIN-soft bodies are opened until the independent root fit entry.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, time, traceback
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode=True
import numpy as np
import torch
from torch.nn import functional as F

H=Path(__file__).resolve().parent
V,P=H.parent,H.parent.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_PUBLIC_YCB_FEATURE_OR_NEW_C49_SCORE_BODIES.json'
SCOPE_SHA='035e25d97efb5e196b334941d4932b7e8845dca2e007e43d93708e56e4be3697'
BASE=P/'query_visibility_direct_query_owner_control_20261024_v175/source_fit_preparation/train_direct_owner_control.py'
BASE_SHA='4ccfb47f16ced76b6b833b473e34cb10962ed6e1124eced86dc65430b59c5aa2'
HEAD=BASE.parent/'direct_owner_head.py'
HEAD_SHA='babde7945ec42c3587f482e42b597763019534adff5981f2f7fb1b5a14a2d480'
assert hashlib.sha256(BASE.read_bytes()).hexdigest()==BASE_SHA
assert hashlib.sha256(HEAD.read_bytes()).hexdigest()==HEAD_SHA
sys.path.insert(0,str(BASE.parent))
spec=importlib.util.spec_from_file_location('v176_pinned_v175_pure_helpers',BASE)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
require,sha,bind,read,lines,write,check=base.require,base.sha,base.bind,base.read,base.lines,base.write,base.check
PROTOCOL=H/'PROTOCOL_BEFORE_SOURCE_OR_CF_FEATURE_AND_TRAIN_SOFT_BODIES.json'
CHECKS=H/'GENERATED_INVENTED_CHECKS.json'
ARM='C49_LITERAL_SYNTHETIC_PLUS_PUBLIC_YCB_CF314'
RUN=V/'runs/C49_PUBLIC_YCB_CF314_FIXED8000'
META=['input_occurrence_id','pair_id','variant','original_query_id','original_global_index',
      'original_frame_index','source_id','split','native_slot','query_xy','L',
      'feature_constructed','query_available','central4_available','pair_pixel_proof_PASS',
      'selected_anchor_native_slot']


def cf_metadata(bindings):
    manifest=lines(check(bindings['manifest']));lineage=lines(check(bindings['feature_lineage']))
    labels=lines(check(bindings['owner_labels']));proofs=lines(check(bindings['pixel_proofs']))
    admission=read(check(bindings['owner_admission']))
    require(len(manifest)==len(lineage)==len(labels)==3816 and len(proofs)==159,'CF metadata counts')
    require(admission['completed'] and admission['PASS'] and admission['CF_known314_counts']==[157,157,0], 'Existing owner-only admission')
    bypair={r['pair_id']:r for r in proofs};require(len(bypair)==159 and sum(r['PASS'] for r in proofs)==157,'All159 attempts/failed2 retained')
    targets=np.asarray([r['owner_aux_target'] for r in labels],np.int64)
    for i,(m,l,r) in enumerate(zip(manifest,lineage,labels)):
        require(l['feature_index']==r['cf_global_index']==i,'CF exact feature/label index')
        for k,v in m.items():require(l[k]==v,'Feature lineage manifest '+k)
        for k in META:
            require(base.query_equal(m[k],r[k]) if k=='query_xy' else m[k]==r[k], 'CF label metadata '+k)
        proof=bypair[m['pair_id']]
        require(m['split']=='SOURCE_DEVELOPMENT' and proof['split']==m['split'] and m['pair_pixel_proof_PASS']==proof['PASS'],'CF TRAIN/proof join')
        require(m['selected_anchor_native_slot']==proof['native_slot'],'Selected input native')
        require(r['native_visibility_target']==r['point_quality_target']==-1 and not r['human_gold'] and not r['physical_native_gold'],'CF owner-only weak authority')
        eligible=bool(proof['PASS'] and m['variant'] in ['COVER','ADJACENT_SHAM'] and m['native_slot']==m['selected_anchor_native_slot'] and m['query_available'] and m['feature_constructed'] and m['central4_available'])
        require((targets[i]>=0)==eligible,'Exact314 eligibility; no replacements')
        require(r['eligible_only_for_prospective_augmented_owner_pool']==eligible and r['owner_auxiliary_admitted_under_root_design']==eligible,'Prior owner mapping')
        if eligible:
            require(targets[i]==(1 if m['variant']=='COVER' else 0),'COVER foreign/SHAM target')
            require(m['original_query_id']==proof['anchor_query_id'] and base.query_equal(m['query_xy'],proof['original_q']) and m['L']==proof['original_L'],'Anchor q/L join')
        else:require(targets[i]==-1,'Before/failed/nonanchor unknown')
    known=np.flatnonzero(targets>=0)
    require(len(known)==314 and np.bincount(targets[known],minlength=3).tolist()==[157,157,0] and int((targets<0).sum())==3502,'CF314 counts')
    require(sum(not r['query_available'] for r in manifest)==360,'Original CF360 U')
    return manifest,lineage,labels,targets,known


def cf_ce(logits, targets):
    require(logits.shape==(len(targets),3) and targets.dtype==torch.int64,'CF 3way logits/schema')
    require(bool(((targets==0)|(targets==1)).all()) and all(bool((targets==k).any()) for k in [0,1]),'CF two supported classes only')
    return (F.cross_entropy(logits[targets==0],targets[targets==0])+F.cross_entropy(logits[targets==1],targets[targets==1]))/2


def cf_original_join(manifest, original, facts):
    for r in manifest:
        i=r['original_global_index'];require(0<=i<1840,'Original MR row')
        e=original[i];f=facts[i]
        for key,dst in [('query_id','original_query_id'),('frame_index','original_frame_index'),('native_slot','native_slot'),('split','split'),('L','L'),('query_available','query_available'),('feature_constructed','feature_constructed')]:
            require(e[key]==r[dst],'CF original MR join '+key)
        require(base.query_equal(e['query_xy'],r['query_xy']) and f['source_id']==r['source_id'],'CF original q/source')
        require(f['nearest_valid']==r['nearest_valid'] and f['local_central4_available']==r['central4_available'] and f['context_available']==r['context_available'],'CF original input flags')


def loss_for(model,px,py,sx,sy,mass,ntrain,cfx,cfy,include_CF):
    prior,pl,sl=base.combined(model,px,py,sx,sy,mass,ntrain,True)
    if not include_CF:return prior,pl,sl,None
    extra=cf_ce(model(cfx),cfy)
    return prior+extra,pl,sl,extra


def invented():
    tests=[]
    def passed(name,ok):require(ok,name);tests.append(dict(name=name,PASS=True))
    torch.set_num_threads(1)
    z=torch.tensor([[.2,-.3,.1]]*6,dtype=torch.float64,requires_grad=True);y=torch.tensor([0,0,0,1,1,1])
    cf_ce(z,y).backward();desired=(z.detach().softmax(1)-F.one_hot(y,3))/6
    passed('CF full3way CE analytic gradient; no BG target invented',torch.allclose(z.grad,desired,rtol=0,atol=1e-12))
    za=torch.zeros((8,3),dtype=torch.float64,requires_grad=True);mask=torch.tensor([1,1,1,1,1,1,0,0],dtype=torch.bool);cf_ce(za[mask],y).backward()
    passed('UNKNOWN CF rows exactly zero gradient',not za.grad[~mask].any())
    try:cf_ce(z,torch.zeros(6,dtype=torch.int64))
    except AssertionError:reject=True
    else:reject=False
    passed('No missing CF class denominator or pseudo-BG',reject)
    torch.manual_seed(1750);a=base.DirectOwnerHead();torch.manual_seed(1750);b=base.DirectOwnerHead()
    passed('Fresh head equals pinned C48 default initialization/zero-last',base.object_sha(a.state_dict())==base.object_sha(b.state_dict()) and not a.head[2].weight.any())
    px=torch.linspace(-.1,.2,6*1928).reshape(6,1928);py=torch.tensor([0,1,2,2,1,0]);sx=px[:4];sy=torch.tensor([[.7,.1,.2],[.2,.6,.2],[.1,.2,.7],[.4,.4,.2]],dtype=torch.float64);mass=sy.sum(0)
    opts=[torch.optim.AdamW(q.parameters(),lr=1e-4,weight_decay=.001) for q in [a,b]];rngs=[np.random.default_rng(1750),np.random.default_rng(1750)];records=[]
    for step in range(3):
        draws=[r.choice(4,256,replace=True) for r in rngs];require(np.array_equal(*draws),'same source draw')
        before=torch.get_rng_state().clone()
        for i,q in enumerate([a,b]):
            opts[i].zero_grad(set_to_none=True);idx=draws[i]
            loss=base.combined(q,px,py,sx[idx],sy[idx],mass,4,True)[0] if i==0 else loss_for(q,px,py,sx[idx],sy[idx],mass,4,None,None,False)[0]
            loss.backward();torch.nn.utils.clip_grad_norm_(q.parameters(),5);opts[i].step()
        require(base.object_sha(a.state_dict())==base.object_sha(b.state_dict()) and base.object_sha(opts[0].state_dict())==base.object_sha(opts[1].state_dict()),'lambdaCF0 model/AdamW byte parity')
        require(torch.equal(before,torch.get_rng_state()) and base.object_sha(rngs[0].bit_generator.state)==base.object_sha(rngs[1].bit_generator.state),'lambdaCF0 torch/numpy RNG parity')
        records.append(dict(step=step+1,model_optimizer_RNG_byte_parity=True))
    passed('LambdaCF0 exact three-step C48 model/AdamW/RNG parity',len(records)==3)
    prior,pl,sl=base.combined(a,px,py,sx,sy,mass,4,True);combined,p2,s2,extra=loss_for(a,px,py,sx,sy,mass,4,px[:6],y,True)
    passed('Additive CF1 preserves both previous loss coefficients',torch.equal(pl,p2) and torch.equal(sl,s2) and torch.equal(combined,prior+extra))
    return tests,records


def prepare():
    require(sha(SCOPE)==SCOPE_SHA,'Scope');s=read(SCOPE);bp=read(check(s['base_training_protocol']))
    bm=base.metadata(bp['inputs'],bp['MR_inputfacts']);cm=cf_metadata(s['CF_inputs']);cf_original_join(cm[0],bm[0]['MR1840'],bm[3]);control=read(check(s['C48_fixed_control_complete']))
    baseline=control['arms']['C48_PUBLIC_WEAK253_PLUS_LITERAL_SYNTHETIC'];require(control['completed'] and control['PASS'] and baseline['updates']==8000,'Frozen C48 control metadata')
    tests,records=invented();write(CHECKS,dict(completed=True,PASS=True,code=bind(__file__),base_trainer=bind(BASE),base_head=bind(HEAD),checks=tests,lambdaCF0_three_steps=records,source_feature_trainsoft_or_new_score_bodies_read=0))
    p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),base_trainer=bind(BASE),head=bind(HEAD),base_protocol=s['base_training_protocol'],base_inputs=bp['inputs'],MR_inputfacts=bp['MR_inputfacts'],metadata_identity_sha=bm[-1],CF_inputs=s['CF_inputs'],CF_known_indices=cm[4].tolist(),CF_counts=[157,157,0],CF_unknown3502=True,CF_inputU360=True,checks=bind(CHECKS),baseline_C48_complete=s['C48_fixed_control_complete'],expected_initial_state_sha=baseline['initial_state_sha'],expected_source_draw_sha=baseline['draw_sha'],arm=ARM,steps=8000,seed=1750,batch=256,lr=1e-4,weight_decay=.001,clip5=True,architecture='Pinned221955 features-only head; no old checkpoint load',loss='Unchanged C48 public253mean3 + sourceTRAIN4016rawsoftclassbalanced + separatelyCF314mean2 full3wayCE; coefficients1/1/1; no mixed-pool normalization',validation_soft_never_read_or_hashed=True,runtime=bp['runtime'],output=str(RUN),selected_model=None,adopted=False,fully_solved=False)
    write(PROTOCOL,p)
    write(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_C49_source_fit=False,source_only=True,scope=bind(SCOPE),code=bind(__file__),head=bind(HEAD),protocol=bind(PROTOCOL),no_private_current_V112_PnP_or_encoder=True))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),head=bind(HEAD),checks=bind(CHECKS),CF_metadata_known314_unknown3502_verified=True,actual_feature_trainsoft_score_or_checkpoint_reads=0,actual_fit_performed=False))


def cf_packet(binding,manifest,lineage):
    with np.load(check(binding),allow_pickle=False) as z:pack={k:z[k].copy() for k in z.files}
    f=pack['features'];require(f.shape==(3816,1928) and f.dtype==np.float32 and np.isfinite(f).all(),'CF features')
    base.exact(pack['occurrence_index'],np.arange(3816),'CF occurrence_index')
    for key in ['input_occurrence_id','pair_id','variant','original_query_id','original_global_index','original_frame_index','source_id','split','native_slot','query_available','feature_constructed','nearest_valid','context_available']:
        base.exact(pack[key],np.asarray([r[key] for r in manifest]),'CF '+key)
    for key in ['query_xy','L']:
        value=np.asarray([([np.nan,np.nan] if key=='query_xy' else np.nan) if r[key] is None else r[key] for r in manifest],np.float64);base.exact(pack[key],value,'CF '+key)
    base.exact(pack['local_central4_available'],np.asarray([r['central4_available'] for r in manifest]),'CF center4')
    require(int((~pack['query_available']).sum())==360,'CF original U360')
    for i,r in enumerate(lineage):require(hashlib.sha256(np.ascontiguousarray(f[i]).tobytes()).hexdigest()==r['feature1928_sha256'],'CF per-row feature provenance')
    return pack


def fit(entry_path):
    e=read(entry_path);require(e.get('approved_C49_source_fit') is True and e.get('source_only') is True and e.get('no_private_current_V112_PnP_or_encoder') is True,'No root C49 entry')
    for k,f in [('scope',SCOPE),('code',Path(__file__)),('head',HEAD),('protocol',PROTOCOL)]:require(e[k]==bind(f),'Entry '+k)
    p=read(PROTOCOL);require(p['code']==bind(__file__) and p['base_trainer']==bind(BASE) and p['checks']==bind(CHECKS),'Frozen code')
    require(Path(sys.executable).resolve()==Path(base.PYTHON).resolve() and torch.__version__=='2.1.1+cu118' and np.__version__=='1.26.4' and torch.cuda.is_available(),'Pinned runtime')
    require(not RUN.exists(),'Preserve previous attempt; no retry');RUN.mkdir(parents=True);write(RUN/'STARTED.json',dict(completed=False,entry=bind(entry_path),code=bind(__file__),protocol=bind(PROTOCOL)))
    try:
        started=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        families,inventory,owner,facts,target,pub,pool,ident=base.metadata(p['base_inputs'],p['MR_inputfacts']);require(ident==p['metadata_identity_sha'],'Base metadata drift')
        manifest,lineage,labels,cy,ci=cf_metadata(p['CF_inputs']);cf_original_join(manifest,families['MR1840'],facts);require(ci.tolist()==p['CF_known_indices'],'Fixed CF314 indices')
        old=base.packet(p['base_inputs']['OLD6144_features'],families['OLD6144'],inventory=inventory);mr=base.packet(p['base_inputs']['MR1840_features'],families['MR1840'],facts=facts);cf=cf_packet(p['CF_inputs']['features'],manifest,lineage)
        with np.load(check(p['base_inputs']['TRAIN_owner_sidecar']),allow_pickle=False) as z:
            require(z['owner_class_order'].astype(str).tolist()==base.OWNER6 and (z['split'].astype(str)=='TRAIN').all(),'TRAIN only')
            require(not bool(z['training_admission']) and not bool(z['initializer_physical_native_identity_certified']),'Original source sidecar authority')
            gi=z['original_global_index'].astype(np.int64);require(len(gi)==4096 and len(np.unique(gi))==4096 and ((gi>=0)&(gi<6144)).all(),'TRAIN indices')
            for k,dst in [('global_index','original_global_index'),('occurrence_id','original_occurrence_id'),('scene_index','scene_index'),('base_index','base_index'),('native_slot','native_slot'),('query_xy','query_xy'),('L','L'),('split','split')]:base.exact(old[k][gi],z[dst],dst)
            for k in inventory[0]['original_input_flags']:base.exact(old[k][gi],z[k],k)
            base.exact(old['no_detection'][gi],z['selected_index_missing'],'noDetection')
            y,mass=base.validate_soft(z['scene_owner_point'],z['query_available'].astype(bool),z['query_xy_missing'].astype(bool));base.exact(gi[z['query_available']],pool,'TRAIN available pool')
        px=torch.as_tensor(mr['features'][pub],device='cuda');py=torch.as_tensor(target[pub],dtype=torch.int64,device='cuda');sx=torch.as_tensor(old['features'][pool],device='cuda');sy=torch.as_tensor(y,dtype=torch.float64,device='cuda');mc=torch.as_tensor(mass,dtype=torch.float64,device='cuda');cfx=torch.as_tensor(cf['features'][ci],device='cuda');cfy=torch.as_tensor(cy[ci],dtype=torch.int64,device='cuda')
        write(RUN/'FIT_INPUT_VALIDATION.json',dict(completed=True,PASS=True,base_TRAIN4016=True,public253_counts=[53,52,148],TRAIN_soft_class_mass=mass.tolist(),CF314_counts=[157,157,0],CF_unknown3502=True,CF_U360=True,held_soft_reads=0))
        directory=RUN/ARM;directory.mkdir();torch.manual_seed(1750);torch.cuda.manual_seed_all(1750);model=base.DirectOwnerHead().cuda().float();initial=base.object_sha(model.state_dict());require(initial==p['expected_initial_state_sha'],'Exact fresh C48 initialization')
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001);rng=np.random.default_rng(1750);dh=hashlib.sha256();logpath=directory/'ALL8000_DRAWS_AND_STEP_LOSSES.jsonl'
        with logpath.open('x') as log:
            for step in range(1,8001):
                pos=rng.choice(4016,256,replace=True);indices=pool[pos];dh.update(indices.astype('<i8').tobytes());opt.zero_grad(set_to_none=True)
                loss,pl,sl,cl=loss_for(model,px,py,sx[pos],sy[pos],mc,4016,cfx,cfy,True);require(torch.isfinite(loss),'Finite loss');loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5);require(torch.isfinite(gn),'Finite gradient');opt.step();require(all(torch.isfinite(q).all() for q in model.parameters()),'Finite updated state')
                log.write(json.dumps(dict(update=step,synthetic_global_indices=indices.tolist(),public_CE=float(pl.detach()),source_CE=float(sl.detach()),CF_mean2_CE=float(cl.detach()),total_loss=float(loss.detach()),grad_norm=float(gn),cumulative_draw_sha=dh.hexdigest()),allow_nan=False)+'\n')
                if step%1000==0:print(json.dumps(dict(arm=ARM,update=step)),flush=True)
        require(dh.hexdigest()==p['expected_source_draw_sha'],'Exact C48 original8000 draws');final=base.object_sha(model.state_dict());require(final!=initial,'State changed');ckpt=directory/'FINAL_CHECKPOINT.pt';torch.save(dict(state_dict=base.cpu_copy(model.state_dict()),optimizer_state=base.cpu_copy(opt.state_dict()),update=8000,seed=1750,arm=ARM,head=bind(HEAD),protocol=bind(PROTOCOL)),ckpt);model.eval().requires_grad_(False)
        freeze=RUN/'FROZEN_BEFORE_SAVED_OUTPUTS.json';write(freeze,dict(completed=True,checkpoint=bind(ckpt),update=8000,initial_state_sha=initial,final_state_sha=final,draw_sha=dh.hexdigest(),held_soft_reads=0,saved_outputs_emitted=False))
        outputs={};diagnostics={}
        with torch.inference_mode():
            for family,pack in [('OLD6144',old),('MR1840',mr),('CF3816',cf)]:
                raw=np.full((len(pack['features']),3),np.nan,np.float32);ids=np.flatnonzero(pack['query_available'])
                for lo in range(0,len(ids),512):
                    ii=ids[lo:lo+512];x=torch.as_tensor(pack['features'][ii],device='cuda');a=model(x).cpu().numpy();b=model(x).cpu().numpy();require(np.array_equal(a,b),'Final repeat');raw[ii]=a
                require(np.isfinite(raw[pack['query_available']]).all() and np.isnan(raw[~pack['query_available']]).all(),'U raw rule');pred=np.full(len(raw),-1,np.int8);pred[ids]=np.argmax(raw[ids],1).astype(np.int8)
                path=directory/f'FINAL_{family}_OWNER_OUTPUTS.npz';np.savez_compressed(path,raw_owner_three_logits=raw,owner_prediction=pred,owner_class_order=np.asarray(base.CLASSES),**{k:v for k,v in pack.items() if k!='features'});outputs[family]=bind(path)
                if family in ['MR1840','CF3816']:
                    truth,ix=(target,pub) if family=='MR1840' else (cy,ci);cm=[[int(((truth[ix]==c)&(pred[ix]==j)).sum()) for j in range(3)] for c in range(3)];dp=directory/f'{family}_FINAL_APPARENT_ONLY.json';write(dp,dict(completed=True,TRAIN_apparent_only=True,rows=len(ix),CM_T_F_BG=cm,no_held_or_native_gold_claim=True));diagnostics[family]=bind(dp)
        require(base.object_sha(model.state_dict())==final,'Frozen state unchanged')
        for b in p['base_inputs'].values():check(b)
        for b in p['CF_inputs'].values():check(b)
        item=dict(name=ARM,updates=8000,outputs=outputs,checkpoint=bind(ckpt),final_checkpoint=bind(ckpt),initial_state_sha=initial,final_state_sha=final,draw_sha=dh.hexdigest(),step_losses=bind(logpath),TRAIN_apparent_diagnostics=diagnostics)
        write(RUN/'COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),head=bind(HEAD),protocol=bind(PROTOCOL),entry=bind(entry_path),arms={ARM:item},fixed_final_frozen_before_outputs=True,freeze=bind(freeze),baseline_C48_complete=p['baseline_C48_complete'],same_C48_initial_and_original_draws=True,public253_and_source4016_weights_unchanged=True,CF314_separate_weight1=True,all6144_MR1840_CF3816_and_U_preserved=True,VALIDATION_soft_read_or_hash_calls=0,old_checkpoint_loads=0,new_initializer_encoder_private_current_PnP_qL_native_or_quality_changes=0,selected_model=None,adopted=False,fully_solved=False,automatic_retry=False,seconds=time.monotonic()-started))
    except BaseException:write(RUN/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);args=parser.parse_args();prepare() if args.prepare else fit(args.entry)
