"""V135 fixed source-only native pair sampling/margin control; inert until root entry.

The V133 loader/report/gate helpers supply unchanged source definitions. V135
policy alone defines this experiment's sampler/loss. All geometry inputs are zero.
"""
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,hashlib,importlib.util,json,time,traceback
import numpy as np
import torch
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_NATIVE_PAIR_POLICY_BEFORE_FIT.json'
POLICY_SHA='6f9581b737f578724523f19712fd21a27b110a1b561307a150f2ed97532bcc75'
SOURCE_HELPER=P/'query_visibility_native_geometry_control_20261024_v133/source_fit_preparation/train_matched_geometry.py'
SOURCE_HELPER_SHA='bce3255068fddc91400c8c575060989ca927fd70d68ff8be1928dee75cf159b5'
SOURCE_POLICY=P/'query_visibility_native_geometry_control_20261024_v133/ROOT_MATCHED_GEOMETRY_POLICY_BEFORE_INPUTS.json'
SOURCE_POLICY_SHA='1eb935c9412f828945a8a6e60fc52d7a9a9bcf667e291aaa93b744c89d6654e9'
PAIR_COVERAGE=P/'query_visibility_native_witness_diagnosis_20261024_v134/feature_diagnosis/NATIVE_TRAIN_PAIR_COVERAGE_AND_LOSS_HISTORY.json'
PAIR_COVERAGE_SHA='da1ae31b7db9a62406dea90bfcb6b99fcc4a387c27d16cb523778ff12ce27efc'
C24=P/'query_visibility_native_geometry_control_20261024_v133/runs/C24_C25_MATCHED_G42_FIXED8000/C24_ZERO_GEOMETRY_CONTROL'
C24_FIT=C24/'FIT_COMPLETE.json';C24_TRACE=C24/'TRAIN_DRAWS_AND_LOSSES.jsonl'
C24_FIT_SHA='927546f52e255a811a5e7c4712d0b5db6bf4814802f00b8d0a66329213d159c5'
PROTOCOL=H/'PROTOCOL_BEFORE_SOURCE_FIT.json';GENERATED=H/'GENERATED_MEANINGFUL_CHECKS.json'
RUN=V/'runs/C26_C27_NATIVE_PAIR_FIXED8000'
ARMS=('C26_NATIVE_PAIR_SAMPLER','C27_NATIVE_PAIR_SAMPLER_AND_MARGIN')
NOLD,NNATIVE,OLDUNITS=51004,6144,19432
assert hashlib.sha256(SOURCE_HELPER.read_bytes()).hexdigest()==SOURCE_HELPER_SHA
_spec=importlib.util.spec_from_file_location('v135_pinned_V133_helpers',SOURCE_HELPER)
m=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(m)
# Inert helpers: no V133 generated/prepare/run entry is invoked.
sha,bind,read,check,same,write,rows=m.sha,m.bind,m.read,m.check,m.same,m.write,m.rows
old_helpers,state_sha,clone_cpu=m.old_helpers,m.state_sha,m.clone_cpu
GeometryQueryHead,native_bce=m.GeometryQueryHead,m.native_bce
old_reports,native_report,make_gates,recall14=m.old_reports,m.native_report,m.make_gates,m.recall14
read_native_GT=m.read_native_GT

def make_pairpool(native,available):
    """Source-defined same physical native corner; never inspect model scores."""
    pool=[];rr=native['rows']
    for base in range(128):
        for slot in range(8):
            ii=(base*4+np.arange(4))*8+slot;e,d=int(ii[1]),int(ii[2])
            if native['state'][e]=='E'and native['state'][d]=='D'and available[e]and available[d]:
                assert all(native['split'][ii]=='TRAIN')and not native['boundary'][ii].any()
                assert rr[e]['canonical_index']==rr[d]['canonical_index']
                assert rr[e]['base_index']==rr[d]['base_index']==base and rr[e]['native_slot']==rr[d]['native_slot']==slot
                assert native['variant'][e]=='COVER'and native['variant'][d]=='NEAR_SHAM'
                pool.append((e,d))
    pool=np.asarray(pool,np.int64);assert pool.shape==(125,2)
    assert len(np.unique(pool))==250
    return pool

def assert_pair_coverage(pool):
    assert sha(PAIR_COVERAGE)==PAIR_COVERAGE_SHA
    c=read(PAIR_COVERAGE);assert c['PASS']and c['TRAIN_only_pair_definition']and c['no_modelerror_or_VAL_mining']
    expected=np.asarray([(r['global_indices'][1],r['global_indices'][2])for r in c['groups']if r['available_coverE_nearD']],np.int64)
    assert np.array_equal(pool,expected)

def draw_native64(pool,uniform_pools,pair_generator,uniform_generator):
    pi=torch.randint(len(pool),(16,),generator=pair_generator).numpy()
    du=uniform_pools[0][torch.randint(len(uniform_pools[0]),(16,),generator=uniform_generator).numpy()]
    eu=uniform_pools[1][torch.randint(len(uniform_pools[1]),(16,),generator=uniform_generator).numpy()]
    return np.r_[pool[pi,0],pool[pi,1],du,eu].astype(np.int64),pi

def native_pair_hinge(z,target,available):
    assert len(z)==len(target)==len(available)==64
    valid=available[:16]&available[16:32]&(target[:16]==1)&(target[16:32]==0)
    delta=z[:16]-z[16:32]
    return torch.relu(2-delta[valid]).mean()if valid.any()else z.sum()*0

def load_source_data_zero_geometry(policy):
    h=old_helpers();replay=read(check(policy['C16_replay_protocol']))
    freeze=read(check(policy['fixed_initialization']));duration=read(check(freeze['design']))
    design=read(check(duration['parent_C15_design']));d14=read(check(design['parent_C14_design']));p14=read(check(design['parent_C14_protocol']))
    c12,c11,c10,c9,d=h.old.source_modules()
    meta,base,parent=h.load_old_metadata(dict(source_bindings=replay['source_bindings']),d14,p14,c12,c10,d)
    new=h.join_flat(h.rows(check(design['new_input_occurrences'])),h.rows(check(design['new_GT_occurrences'])))
    op=h.load_feature_packet(design['old_features'],h.BASE,h.OLDUNITS);npk=h.load_feature_packet(design['new_features'],h.NEW,h.NEWUNITS)
    units=np.r_[op['occurrence_unit'],h.OLDUNITS+npk['occurrence_unit']].astype(np.int32)
    av=np.r_[op['query_available'],npk['query_available']]
    oldmeta={k:np.r_[meta[k],new[{'external_target':'labels'}.get(k,k)]]for k in('external_target','split','typed_state','domain','source','variant')}
    with np.load(check(policy['C16_source_scores']),allow_pickle=False)as f:baseline={k:f[k].copy()for k in f.files}
    for key,expected in oldmeta.items():assert np.array_equal(expected,baseline['raw_teacher_state'if key=='typed_state'else key]),key
    assert np.array_equal(units,baseline['input_unit_index'])and np.array_equal(av,baseline['query_available'])
    assert len(av)==NOLD and op['features'].shape==(15004,1928)and npk['features'].shape==(4428,1928)
    assert np.array_equal(oldmeta['external_target'],np.where(oldmeta['typed_state']=='D',0,np.where(oldmeta['typed_state']=='E',1,-1)))
    pools=c10.sampling_pools(base['old'],{k:base[k]for k in('known','split','external_target','typed_state','domain')},av[:h.old.BASE],d)
    mp=c12.material_pools(base,av[:h.old.BASE]);wp=h.old.available_pairs(meta,av[:h.BASE]);fp=h.flat_pools(new,av[h.BASE:]);assert len(wp)==62
    nf=read(check(policy['native_feature_seal']));assert nf['completed']and nf['PASS']and nf['GT_read']is False
    with np.load(check(nf['features']),allow_pickle=False)as f:npack={k:f[k].copy()for k in f.files}
    with np.load(check(nf['predictions']),allow_pickle=False)as f:nbaseline={k:f[k].copy()for k in f.files}
    assert npack['features'].shape==(NNATIVE,1928)and npack['features'].dtype==np.float32 and np.isfinite(npack['features']).all()
    for key in('query_available','feature_constructed'):assert npack[key].dtype==bool and npack[key].shape==(NNATIVE,)
    assert np.array_equal(npack['feature_constructed'],nbaseline['head_forwarded'])and np.array_equal(npack['query_available'],nbaseline['query_available'])
    # Both V135 arms have unknown/unused geometry. Do not read actual G42 packet.
    geom=dict(G42=np.zeros((NNATIVE,42),np.float32),geometry_constructed=npack['feature_constructed'].copy())
    for key in ('global_index','source_occurrence_index','scene_index','base_index','native_slot','occurrence_id','query_xy','L'):geom[key]=npack[key].copy()
    assert geom['G42'].shape==(NNATIVE,42)and geom['G42'].dtype==np.float32 and np.isfinite(geom['G42']).all()
    assert np.array_equal(geom['geometry_constructed'],npack['feature_constructed'])
    for key in('global_index','source_occurrence_index','scene_index','base_index','native_slot','occurrence_id','query_xy','L'):
        assert np.array_equal(geom[key],npack[key],equal_nan=True)if geom[key].dtype.kind in'fc'else np.array_equal(geom[key],npack[key]),key
    assert not geom['G42'][~geom['geometry_constructed']].any()
    ds=read(check(policy['native_fixed_GT_ledger']));assert ds['completed']and ds['PASS']and ds['RGB_read']is False
    gt=next(x for x in ds['outputs']if Path(x['path']).name=='ALL6144_SLOTS_TWO_CONVENTIONS_GT_ONLY.jsonl')
    nmeta=read_native_GT(check(gt));nm=nmeta['CF_PRIMARY_DECLARED']
    for i,r in enumerate(nm['rows']):
        q=np.array(r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]);assert np.array_equal(q,npack['query_xy'][i],equal_nan=True)
    train=(nm['split']=='TRAIN')&npack['query_available'];n_pools=[np.flatnonzero(train&(nm['target']==t))for t in(0,1)]
    assert list(map(len,n_pools))==[1869,131]and not(train&nm['boundary']).any()
    assert not any((nm['split'][p]!='TRAIN').any()for p in n_pools)
    return dict(h=h,c12=c12,c11=c11,c10=c10,c9=c9,d=d,design=design,d14=d14,meta=meta,base=base,parent=parent,new=new,
                op=op,npk=npk,units=units,available=av,oldmeta=oldmeta,baseline=baseline,pools=pools,mp=mp,wp=wp,fp=fp,
                npack=npack,nbaseline=nbaseline,geometry=geom,nmeta=nmeta,native_pools=n_pools,freeze=freeze)

def run(entry):
    e=read(entry);assert e['approved_source_fit']is True and all(e[k]is False for k in('current223','V112','PnP','q_correction','automatic_retry'))
    for key,path in(('code',Path(__file__)),('protocol',PROTOCOL),('policy',POLICY),('generated',GENERATED),('source_binding_policy',SOURCE_POLICY),('source_helper',SOURCE_HELPER),('pair_coverage',PAIR_COVERAGE),('historical_C24_fit',C24_FIT),('historical_C24_trace',C24_TRACE)):assert same(e[key],bind(path))
    assert sha(POLICY)==POLICY_SHA and sha(SOURCE_POLICY)==SOURCE_POLICY_SHA and sha(PAIR_COVERAGE)==PAIR_COVERAGE_SHA and sha(C24_FIT)==C24_FIT_SHA
    assert not RUN.exists(),'Preserve previous run; no automatic retry.'
    protocol=read(PROTOCOL);assert same(protocol['code'],e['code'])and read(GENERATED)['PASS']
    for key in ('policy','generated','source_binding_policy','source_helper','pair_coverage','historical_C24_fit','historical_C24_trace'):assert same(protocol[key],e[key])
    policy=read(SOURCE_POLICY)
    for key in('fixed_initialization','C16_replay_protocol','C16_source_scores','native_feature_seal','native_initializer_seal','native_fixed_GT_ledger','native_evaluation','geometry_code'):check(policy[key])
    data=load_source_data_zero_geometry(policy);h=data['h'];native=data['nmeta']['CF_PRIMARY_DECLARED']
    pairpool=make_pairpool(native,data['npack']['query_available']);assert_pair_coverage(pairpool)
    assert len(pairpool)==125
    baseline_report=old_reports(data,data['baseline']['query_logit']);baseline14=baseline_report['old14']
    baseline_native=native_report(data['nbaseline']['raw_head_logit_before_U'],data['npack']['query_available'],native)
    assert baseline_native['VALIDATION']['CM_D_E_rows_D_E_U_columns']==[[819,6,3],[46,13,1]]
    assert baseline_native['VALIDATION::NONBOUNDARY']['D_recall']==508/510
    assert all(v['PASS']for v in baseline_report['historical15_gates'].values())
    assert torch.__version__==protocol['runtime']['torch']and np.__version__==protocol['runtime']['numpy']and torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda';xo=torch.as_tensor(data['op']['features'],device=device);xf=torch.as_tensor(data['npk']['features'],device=device)
    xt=torch.cat((xo,xf));units=torch.as_tensor(data['units'],device=device,dtype=torch.long)
    oldtarget=torch.as_tensor(data['oldmeta']['external_target'],device=device,dtype=torch.float32);olda=torch.as_tensor(data['available'],device=device)
    xn=torch.as_tensor(data['npack']['features'],device=device);zero_native=torch.zeros((NNATIVE,42),device=device)
    na=torch.as_tensor(data['npack']['query_available'],device=device);nt=torch.as_tensor(native['target'],device=device,dtype=torch.float32)
    constructed=np.flatnonzero(data['npack']['feature_constructed']);zeros_old=torch.zeros((max(len(xo),len(xf)),42),device=device)
    RUN.mkdir(parents=True);write(RUN/'STARTED.json',dict(entry=bind(entry),protocol=bind(PROTOCOL),actual_updates=0,
        retained_old51004_native6144=True,native_TRAIN_available_DE=list(map(len,data['native_pools'])),
        sampler='Exact saved C24 old192; native pairedE16/pairedD16/uniformD16/uniformE16; identical botharms',native_pair_count125=True,actual_geometry_input_used=False))
    write(RUN/'BASELINE_REPORT_BEFORE_FIT.json',dict(old=baseline_report,native_CF=baseline_native))
    write(RUN/'NATIVE_TRAIN_PAIR_POOL_BEFORE_FIT.json',dict(completed=True,pair_count125=True,pairs_COVER_E_NEAR_SHAM_D=pairpool.tolist(),coverage=bind(PAIR_COVERAGE),source_defined_before_fit=True,actualq_differences_retained=True,not_isolated_sameq_contrast=True))
    started=time.monotonic();arm_results={};reference_draw_hash=None
    try:
        for arm in ARMS:
            directory=RUN/arm;directory.mkdir()
            torch.manual_seed(1330);torch.cuda.manual_seed_all(1330)
            model=GeometryQueryHead().to(device).float();payload=torch.load(check(protocol['C16_checkpoint']),map_location='cpu',weights_only=True)
            assert payload['update']==4500
            missing,unexpected=model.load_state_dict(payload['state_dict'],strict=False)
            assert missing==['geometry.weight']and not unexpected and sum(p.numel()for p in model.parameters())==224513
            assert all(torch.equal(v.detach().cpu(),payload['state_dict'][k])for k,v in model.state_dict().items()if k!='geometry.weight')
            del payload;assert not model.geometry.weight.any()
            geometry=zero_native
            assert not geometry.any()
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001)
            gens=[torch.Generator().manual_seed(s)for s in(208,214,215,216,217)];flatgen=torch.Generator().manual_seed(218);ng=torch.Generator().manual_seed(1332);pg=torch.Generator().manual_seed(1334)
            drawhash=hashlib.sha256();updates=0
            def checkpoint(step):
                return dict(state_dict=clone_cpu(model.state_dict()),optimizer_state=clone_cpu(optimizer.state_dict()),
                    update=step,C16_parent_update=4500,arm=arm,threshold_logit=0,policy=bind(POLICY),trainer=bind(__file__),
                    RNG=dict(old=[g.get_state()for g in gens],flat=flatgen.get_state(),native_uniform=ng.get_state(),native_pair=pg.get_state(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all()))
            @torch.no_grad()
            def predict():
                model.eval();rawunits=torch.cat((model(xo,zeros_old[:len(xo)]),model(xf,zeros_old[:len(xf)])))
                oldz=np.where(data['available'],rawunits[units].cpu().numpy(),0).astype(np.float32)
                raw=np.full(NNATIVE,np.nan,np.float32)
                for lo in range(0,len(constructed),512):
                    ii=constructed[lo:lo+512];ix=torch.as_tensor(ii,device=device,dtype=torch.long)
                    raw[ii]=model(xn[ix],geometry[ix]).cpu().numpy()
                return oldz,raw
            @torch.no_grad()
            def evaluate(step):
                oldz,raw=predict();_,parts=h.selector14(data['c12'],oldz,data['available'],data['meta'],data['new'],data['d'])
                nr=native_report(raw,data['npack']['query_available'],native)
                row=dict(update=step,old14_BCE_parts=parts,old14=recall14(oldz,data['available'],data['oldmeta']),
                         native_TRAIN=nr['TRAIN'],native_VAL=nr['VALIDATION'],native_VAL_nonboundary=nr['VALIDATION::NONBOUNDARY'])
                with(directory/'HISTORY.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                return oldz,raw
            old0,native0=evaluate(0)
            old_exact=np.array_equal(old0,data['baseline']['query_logit'])
            native_exact=np.array_equal(native0,data['nbaseline']['raw_head_logit_before_U'],equal_nan=True)
            parity=dict(completed=True,PASS=bool(old_exact and native_exact),all51004_old_scores_bitexact=old_exact,
                all6088_native_raw_scores_bitexact=native_exact,missing56_raw_NaN_exact=True,
                geometry_weights_zero=True,same_parameters=224513,old_eval_shapes=[15004,4428],native_eval_constructible512_batches=True,actual_updates=0)
            write(directory/'UPDATE0_C16_EXACT_PARITY.json',parity)
            assert parity['PASS'],'Preserve exact initialization parity failure; no training permitted.'
            torch.save(checkpoint(0),directory/'INITIAL_CHECKPOINT.pt')
            with (directory/'TRAIN_DRAWS_AND_LOSSES.jsonl').open('x')as trace,C24_TRACE.open()as historical:
                for step in range(1,8001):
                    ix,pairs,audit,*_=h.draw192(data['c12'],data['c10'],data['pools'],data['mp'],data['wp'],data['fp'],gens,flatgen)
                    ni,npi=draw_native64(pairpool,data['native_pools'],pg,ng)
                    oldrow=json.loads(next(historical));assert oldrow['update']==step
                    assert np.array_equal(ix,np.asarray(oldrow['old_occurrences']))and np.array_equal(pairs,np.asarray(oldrow['old_matched_cover_sham_positions'])), 'old192 differs from historical C24'
                    assert np.array_equal(ni[:16],pairpool[npi,0])and np.array_equal(ni[16:32],pairpool[npi,1])
                    assert np.array_equal(np.bincount(data['oldmeta']['external_target'][ix],minlength=2),[96,96])
                    assert data['available'][ix].all()and(data['oldmeta']['split'][ix]=='TRAIN').all()
                    assert np.array_equal(native['target'][ni],np.r_[np.ones(16,np.int8),np.zeros(32,np.int8),np.ones(16,np.int8)])
                    assert data['npack']['query_available'][ni].all()and(native['split'][ni]=='TRAIN').all()
                    drawhash.update(ix.astype('<i8').tobytes());drawhash.update(pairs.astype('<i8').tobytes());drawhash.update(ni.astype('<i8').tobytes())
                    oi=torch.as_tensor(ix,device=device,dtype=torch.long);nn=torch.as_tensor(ni,device=device,dtype=torch.long);pp=torch.as_tensor(pairs,device=device,dtype=torch.long)
                    model.train();optimizer.zero_grad(set_to_none=True)
                    zo=model(xt[units[oi]],zeros_old[:192]);zn=model(xn[nn],geometry[nn])
                    oldloss,bce,hinge,flatbce=h.loss192(zo,oldtarget[oi],olda[oi],pp);nl=native_bce(zn,nt[nn],na[nn]);nh=native_pair_hinge(zn,nt[nn],na[nn]);nw=.25 if arm==ARMS[1]else 0.;loss=oldloss+nl+nw*nh
                    assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(grad);optimizer.step();updates=step
                    assert not model.geometry.weight.any(),'Both geometry branches must remain zero.'
                    trace.write(json.dumps(dict(update=step,old_occurrences=ix.tolist(),old_matched_cover_sham_positions=pairs.tolist(),native_occurrences=ni.tolist(),native_pair_pool_positions=npi.tolist(),
                        loss=float(loss.detach()),old_BCE=float(bce.detach()),old_hinge=float(hinge.detach()),flat_BCE=float(flatbce.detach()),native_BCE=float(nl.detach()),native_pair_hinge=float(nh.detach()),native_pair_hinge_weight=nw,gradient_before_clip=float(grad)),allow_nan=False)+'\n')
                    if step%100==0:evaluate(step)
                    if step%1000==0:print(json.dumps(dict(arm=arm,updates=step,seconds=round(time.monotonic()-started,2))),flush=True)
                assert next(historical,None)is None,'Historical C24 trace has extra updates'
            draw_sha=drawhash.hexdigest()
            if reference_draw_hash is None:reference_draw_hash=draw_sha
            else:assert draw_sha==reference_draw_hash,'Both arms must use exact same old/native draw sequence.'
            torch.save(checkpoint(8000),directory/'FINAL_RESUME_CHECKPOINT.pt');oldz,raw=predict()
            report=old_reports(data,oldz);nr=native_report(raw,data['npack']['query_available'],native)
            sensitivity=native_report(raw,data['npack']['query_available'],data['nmeta']['V4_SENSITIVITY_ONLY'])
            gates=make_gates(report,nr,baseline14,baseline_native)
            rank_recalls=[v['value']for k,v in gates.items()if not k.startswith('HISTORICAL15::')]
            assert len(rank_recalls)==27 and all(v is not None for v in rank_recalls)
            worst=min(rank_recalls)
            eligible=all(v['PASS']for v in gates.values())
            np.savez_compressed(directory/'FINAL_OLD51004_SOURCE_SCORES.npz',query_logit=oldz,prediction=np.where(data['available'],(oldz>=0).astype(np.int8),-1),
                query_available=data['available'],global_occurrence_index=np.arange(NOLD),input_unit_index=data['units'],external_target=data['oldmeta']['external_target'],
                raw_teacher_state=data['oldmeta']['typed_state'],split=data['oldmeta']['split'],domain=data['oldmeta']['domain'],source=data['oldmeta']['source'],variant=data['oldmeta']['variant'])
            np.savez_compressed(directory/'FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz',raw_head_logit_before_U=raw,query_logit=np.where(data['npack']['query_available'],raw,0).astype(np.float32),
                prediction=np.where(data['npack']['query_available'],(raw>=0).astype(np.int8),-1),query_available=data['npack']['query_available'],
                head_forwarded=data['npack']['feature_constructed'],global_index=np.arange(NNATIVE),occurrence_id=data['npack']['occurrence_id'],query_xy=data['npack']['query_xy'],L=data['npack']['L'],
                split=native['split'],raw_teacher_state_CF=native['state'],external_target_CF=native['target'],convention_boundary=native['boundary'])
            write(directory/'SOURCE_RESULTS.json',dict(completed=True,arm=arm,selected_update=8000,VAL_checkpoint_choice=False,threshold_logit=0,
                old=report,native_CF=nr,native_V4_sensitivity_only=sensitivity,gates=gates,source_eligible=eligible,worst_required27_class_recall=worst,
                old_native_all8000_draw_sha256=draw_sha,same_draws_both_arms=True,retained_old51004_native6144=True,
                teacher_SU_no_external_gradient=True,native_pair_pool_count=125,native_pair_hinge_weight=(.25 if arm==ARMS[1]else 0.),native_pair_hinge_margin=2,actual_geometry_input_used=False,old192_all8000_exact_historical_C24=True,current_V112_PnP=0))
            write(directory/'MODEL_FREEZE.json',dict(completed=True,arm=arm,update=8000,selected_update=8000,checkpoint=bind(directory/'FINAL_RESUME_CHECKPOINT.pt'),
                threshold_logit=0,parameters=224513,source_eligible=eligible,policy=bind(POLICY),trainer=bind(__file__),C16_freeze=policy['fixed_initialization'],source_binding_policy=bind(SOURCE_POLICY),all_geometry_input_zero=True,
                encoder_trainable_parameters=0,current_allowed=False,V112_allowed=False,PnP=0))
            write(directory/'FIT_COMPLETE.json',dict(completed=True,optimizer_updates=8000,arm=arm,results=bind(directory/'SOURCE_RESULTS.json'),freeze=bind(directory/'MODEL_FREEZE.json'),
                initialization_parity=bind(directory/'UPDATE0_C16_EXACT_PARITY.json'),draw_sha256=draw_sha,source_eligible=eligible,worst_required27_class_recall=worst))
            arm_results[arm]=dict(source_eligible=eligible,worst_required27_class_recall=worst,fit=bind(directory/'FIT_COMPLETE.json'),freeze=bind(directory/'MODEL_FREEZE.json'))
            del optimizer,model;torch.cuda.empty_cache()
        eligible=[a for a in ARMS if arm_results[a]['source_eligible']]
        selected=max(eligible,key=lambda a:(arm_results[a]['worst_required27_class_recall'],-ARMS.index(a)))if eligible else None
        write(RUN/'CONTROL_COMPLETE.json',dict(completed=True,PASS=True,arms=arm_results,selected_arm_source_only=selected,application_solved=False,
            same_all8000_old_native_draws=True,draw_sha256=reference_draw_hash,final_update8000_both=True,VAL_checkpoint_choice=False,
            old_replay_contains_YCB_and_material_and_original_public=True,old192_all8000_exact_historical_C24=True,geometry_all_zero_botharms=True,historical_uniform_control=bind(C24_FIT),native_pair_pool_count=125,new_private_current_V112_PnP_q_correction=0,
            seconds=time.monotonic()-started,code=bind(__file__),policy=bind(POLICY),protocol=bind(PROTOCOL)))
        print(json.dumps(read(RUN/'CONTROL_COMPLETE.json')),flush=True)
    except BaseException:
        write(RUN/'FAILED_EXECUTION.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

def generated_checks():
    """Invented inputs only: loss signs, masking, architecture and sampler draws."""
    torch.manual_seed(1330);h=old_helpers();model=GeometryQueryHead().float()
    assert sum(p.numel()for p in model.parameters())==224513 and model.geometry.bias is None and not model.geometry.weight.any()
    x=torch.randn(64,1928);zero=torch.zeros(64,42)
    T,Q,C,Pv=x[:,:384],x[:,384:768],x[:,1152:1536],x[:,1536:1920]
    base=model.head(torch.cat((x,torch.abs(T-C),T*C,torch.abs(Q-Pv),Q*Pv),1)).squeeze(-1)
    assert torch.equal(base,model(x,zero))
    y=torch.tensor([1.]*16+[0.]*32+[1.]*16);a=torch.ones(64,dtype=torch.bool)
    z=torch.cat((torch.full((16,),-1.),torch.ones(16),torch.zeros(32))).requires_grad_()
    hinge=native_pair_hinge(z,y,a);assert hinge==4
    hinge.backward();assert torch.equal(z.grad[:16],torch.full((16,),-1/16))and torch.equal(z.grad[16:32],torch.full((16,),1/16))and not z.grad[32:].any()
    z2=torch.cat((torch.full((16,),3.),torch.zeros(48))).requires_grad_()
    native_pair_hinge(z2,y,a).backward();assert not z2.grad.any()
    z3=torch.zeros(64,requires_grad=True);y3=y.clone();a3=a.clone();y3[0]=-1;a3[17]=False
    native_pair_hinge(z3,y3,a3).backward();assert z3.grad[0]==z3.grad[16]==z3.grad[1]==z3.grad[17]==0
    assert z3.grad[2:16].lt(0).all()and z3.grad[18:32].gt(0).all()and not z3.grad[32:].any()
    # Native BCE excludes unknown/self/unavailable independently of hinge masking.
    z4=torch.tensor([-.4,.8,2.,-3.],requires_grad=True);y4=torch.tensor([0.,1.,-1.,1.]);a4=torch.tensor([1,1,1,0],dtype=torch.bool)
    native_bce(z4,y4,a4).backward();assert z4.grad[:2].abs().min()>0 and not z4.grad[2:].any()
    model.zero_grad();zz=model(x,zero);loss=native_bce(zz,y,a)+.25*native_pair_hinge(zz,y,a);loss.backward()
    assert model.head[0].weight.grad.abs().sum()>0 and model.head[2].weight.grad.abs().sum()>0 and not model.geometry.weight.grad.any()
    original=torch.randn(192,requires_grad=True);target=torch.tensor([1.,0.]*96);av=torch.ones(192,dtype=torch.bool);target[191]=-1;av[190]=False
    pp=torch.arange(48).reshape(24,2);oldloss,*_=h.loss192(original,target,av,pp)
    expected=torch.nn.functional.binary_cross_entropy_with_logits(original[:128],target[:128])+.25*torch.relu(2-(original[pp[:,0]]-original[pp[:,1]])).mean()+torch.nn.functional.binary_cross_entropy_with_logits(original[128:190],target[128:190])
    assert torch.equal(oldloss,expected);oldloss.backward();assert not original.grad[190:].any()
    pool=np.c_[np.arange(125),125+np.arange(125)];uniform=[np.arange(125,500),np.r_[np.arange(125),np.arange(500,510)]]
    sequences=[]
    for arm in ARMS:
        pg=torch.Generator().manual_seed(1334);ug=torch.Generator().manual_seed(1332);seq=[]
        for step in range(100):
            ni,pi=draw_native64(pool,uniform,pg,ug)
            assert ni.shape==(64,)and pi.shape==(16,)and np.array_equal(ni[:16],pool[pi,0])and np.array_equal(ni[16:32],pool[pi,1])
            assert np.isin(ni[32:48],uniform[0]).all()and np.isin(ni[48:],uniform[1]).all()
            labels=np.r_[np.ones(125),np.zeros(375),np.ones(10)][ni]
            assert np.array_equal(labels,np.r_[np.ones(16),np.zeros(32),np.ones(16)])
            seq.append((ni,pi))
        sequences.append(seq)
    assert all(np.array_equal(a,b)and np.array_equal(ap,bp)for (a,ap),(b,bp)in zip(*sequences))
    assert np.unique(np.concatenate([z[1]for z in sequences[0]])).size==125
    cm=m.external_metric(np.array([0,0,1,1,-1,-1]),np.array([0,-1,1,-1,1,0]),np.ones(6,bool),np.array(list('DDEESU')))
    assert cm['CM_D_E_rows_D_E_U_columns']==[[1,0,1],[0,1,1]]and cm['D_recall']==cm['E_recall']==.5
    write(GENERATED,dict(completed=True,PASS=True,code=bind(__file__),source_helper=bind(SOURCE_HELPER),torch=torch.__version__,invented_data_only=True,
        equal_parameters224513=True,biasfree_geometry_zero_start=True,zero_geometry_matches_original_visual_exact=True,zero_geometry_gradient_zero_both_arms=True,
        visual_weights_receive_meaningful_gradient=True,native_hinge_margin2_sign_gradient_exact=True,native_hinge_satisfied_margin_zero_gradient=True,
        native_pair_invalid_member_excludes_both_gradients=True,native_teacherSU_unavailable_zero_BCE_gradient=True,
        native_sampler_order_E16_D16_D16_E16=True,native_sampler_paired_members_exact=True,native_sampler_uniform_pools_preserved=True,
        native_pair_and_uniform_rng_identical_both_arms=True,all125_invented_pairs_seen_in100_draws=True,
        old192_loss_exact_BCE_hinge_flatBCE_U_no_gradient=True,full_known_denominator_preserves_predictionU=True,
        actual_source_feature_checkpoint_RGB_models_optimizer_reads=0))
    print(json.dumps(dict(generated=bind(GENERATED),actual_training=False)),flush=True)

def prepare():
    assert sha(POLICY)==POLICY_SHA and sha(SOURCE_HELPER)==SOURCE_HELPER_SHA and sha(SOURCE_POLICY)==SOURCE_POLICY_SHA
    assert sha(PAIR_COVERAGE)==PAIR_COVERAGE_SHA and sha(C24_FIT)==C24_FIT_SHA
    policy=read(POLICY);assert policy['arms']==list(ARMS)
    source=read(SOURCE_POLICY)
    for key in('fixed_initialization','C16_replay_protocol','C16_source_scores','native_feature_seal','native_initializer_seal','native_fixed_GT_ledger','native_evaluation','geometry_code'):check(source[key])
    proof=read(GENERATED);assert proof['PASS']and same(proof['code'],bind(__file__))
    freeze=read(check(source['fixed_initialization']));assert freeze['update']==4500;check(freeze['checkpoint'])
    duration=read(check(freeze['design']));design=read(check(duration['parent_C15_design']))
    fit=read(C24_FIT);assert fit['completed']and fit['optimizer_updates']==8000;check(fit['results']);check(fit['freeze'])
    protocol=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),generated=bind(GENERATED),
        source_helper=bind(SOURCE_HELPER),source_binding_policy=bind(SOURCE_POLICY),source_policy_scope='Bindings and unchanged data/report/42gate definitions only; V135policy defines new sampler/loss.',
        pair_coverage=bind(PAIR_COVERAGE),historical_C24_fit=bind(C24_FIT),historical_C24_trace=bind(C24_TRACE),
        C16_freeze=source['fixed_initialization'],C16_checkpoint=freeze['checkpoint'],replay_protocol=source['C16_replay_protocol'],C16_source_scores=source['C16_source_scores'],
        old_feature_pack=design['old_features'],flat_feature_pack=design['new_features'],native_feature_seal=source['native_feature_seal'],native_initializer_seal=source['native_initializer_seal'],native_GT_ledger=source['native_fixed_GT_ledger'],
        arms=list(ARMS),parameters_each=224513,geometry_input_all_zero=True,actual_G42_packet_read=False,old_occurrences=NOLD,old_feature_units=OLDUNITS,native_occurrences=NNATIVE,
        pair_pool125_defined_only_by_TRAIN_CF_E_D_availability=True,native_sampler_order='pairedE16,pairedD16,uniformD16,uniformE16',native_pair_seed=1334,native_uniform_seed=1332,
        C26_native_margin_weight=0,C27_native_margin_weight=.25,native_margin=2,old192_exact_historical_C24_each_update=True,
        fixed_updates=8000,VAL_checkpoint_choice=False,threshold_logit=0,root_actual_entry_required=True,
        runtime=dict(executable='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python',torch='2.1.1+cu118',numpy='1.26.4',CUDA=True,TF32=False,FP32=True),
        run_directory=str(RUN),actual_head_source_parameters_or_features_load_training=0)
    write(PROTOCOL,protocol)
    template=dict(approved_source_fit=False,current223=False,V112=False,PnP=False,q_correction=False,automatic_retry=False,
        **{k:protocol[k]for k in('code','policy','generated','source_helper','source_binding_policy','pair_coverage','historical_C24_fit','historical_C24_trace')},protocol=bind(PROTOCOL))
    write(H/'ROOT_SOURCE_FIT_ENTRY_TEMPLATE.json',template)
    print(json.dumps(dict(code=bind(__file__),protocol=bind(PROTOCOL),template=bind(H/'ROOT_SOURCE_FIT_ENTRY_TEMPLATE.json'),
        command=protocol['runtime']['executable']+' '+str(Path(__file__).resolve())+' --entry ROOT_ACTUAL_SOURCE_FIT_ENTRY.json',actual_training=False)),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--generated',action='store_true');ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);args=ap.parse_args()
    if args.generated and not args.prepare and args.entry is None:generated_checks()
    elif args.prepare and not args.generated and args.entry is None:prepare()
    elif args.entry and not args.generated and not args.prepare:run(args.entry)
    else:raise SystemExit('Use --generated, --prepare, or --entry ROOT_ACTUAL_SOURCE_FIT_ENTRY.json')
