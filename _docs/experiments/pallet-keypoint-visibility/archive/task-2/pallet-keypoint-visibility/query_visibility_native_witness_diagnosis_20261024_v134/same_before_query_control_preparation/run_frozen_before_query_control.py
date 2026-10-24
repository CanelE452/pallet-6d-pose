"""Image-only source intervention with actual BEFORE q8 fixed across variants.

prepare builds GT-free input metadata and safe prior-score parity extracts.
Only an explicit root entry can load the frozen encoder/three heads or pixels.
"""
from pathlib import Path
import argparse, ast, hashlib, importlib.util, json, os, sys, time, traceback
import numpy as np

sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
V132=P/'query_visibility_actual_initializer_source_20261024_v132'
V133=P/'query_visibility_native_geometry_control_20261024_v133'
PARENT_CODE=V132/'frozen_visibility_preparation/run_frozen_visibility.py'
PARENT_PROTOCOL=V132/'frozen_visibility_preparation/PROTOCOL_BEFORE_VISIBILITY_FORWARD.json'
OLD_SEAL=V132/'frozen_C16_actual_initializer_v1/PREDICTION_SEAL.json'
INIT_SEAL=V132/'actual_initializer_v1/PREDICTION_SEAL.json'
GEOMETRY_SEAL=V133/'input_geometry_v1/INPUT_GEOMETRY_SEAL.json'
TRAINER=V133/'source_fit_preparation/train_matched_geometry.py'
RUN133=V133/'runs/C24_C25_MATCHED_G42_FIXED8000'
PROTOCOL=H/'FROZEN_BEFORE_QUERY_PROTOCOL.json'
INPUTS=H/'FROZEN_BEFORE_QUERY_ROWS6144_INPUT_ONLY.json'
INPUT_SEAL=H/'FROZEN_BEFORE_QUERY_INPUT_SEAL.json'
GFILE=H/'FROZEN_BEFORE_G42_INPUT_ONLY.npz'
PRIOR=H/'PRIOR_BEFORE_SCORES_INPUT_ONLY.npz'
OUT=V/'same_before_query_control_v1'
N=6144
MODELS=('C16','C24','C25')
ARMS={'C24':'C24_ZERO_GEOMETRY_CONTROL','C25':'C25_NATIVE_GEOMETRY42'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def same(a,b):return a['path']==b['path']and a['sha256']==b['sha256']
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def module(p,n):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def pure(p,names,extra=None):
    nodes=[n for n in ast.parse(Path(p).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names]
    assert {n.name for n in nodes}==set(names)
    ns={'np':np};ns.update(extra or{});exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p)+'::pure','exec'),ns);return ns
def class_ast(p,name):
    node=next(n for n in ast.parse(Path(p).read_text()).body if isinstance(n,ast.ClassDef)and n.name==name)
    return ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))

def intervention_rows(original,target,native_grid,geometry):
    assert len(original)==N
    rows=[];g=np.zeros((N,42),np.float32)
    for i,image in enumerate(original):
        assert image['occurrence_index']==i and image['scene_index']==i//8 and image['native_slot']==i%8
        before_index=(i//32)*32+i%8;before=original[before_index]
        assert before['variant']=='BEFORE'and image['base_index']==before['base_index']
        q8=before['initial_points'];q=before['query_xy'];L=before['L'];box=before['context_box_xyxy']
        constructed=bool(before['feature_constructed'])
        if constructed:
            q8a=np.asarray(q8,np.float64)
            assert q8a.shape==(8,2)and np.isfinite(q8a).all()and np.array_equal(q8a[i%8],q)
            assert box==target['query_extent'](q8a)and target['valid_box'](box)
            assert native_grid(q8a,image['raw_hw'])[2]==L
            g[i]=geometry(q8a,i%8,box,image['raw_hw'])
        else:
            assert q8 is None and q is None and L is None and box is None
        row=dict(occurrence_index=i,occurrence_id=image['occurrence_id'],intervention_id='V134_BEFOREQ::'+image['occurrence_id'],
                 scene_index=image['scene_index'],base_index=image['base_index'],scene_id=image['scene_id'],base_id=image['base_id'],
                 split=image['split'],variant=image['variant'],native_slot=image['native_slot'],raw_hw=image['raw_hw'],
                 raw_RGB_PNG=image['raw_RGB_PNG'],decoded_RGB_sha256=image['decoded_RGB_sha256'],
                 query_xy=q,initial_points=q8,L=L,context_box_xyxy=box,
                 finite_non_sentinel_query=bool(before['finite_non_sentinel_query']),context_geometry_valid=bool(before['context_geometry_valid']),
                 native_continuous_valid=bool(before['native_continuous_valid']),native_nearest_valid=bool(before['native_nearest_valid']),
                 source_BEFORE_occurrence_index=before_index,source_BEFORE_query_id=before['occurrence_id'],
                 source_BEFORE_query_constructed=constructed,source_emulation=True,source_input_intervention=True,
                 actual_original_initializer=False,actual_query_for_this_image=image['variant']=='BEFORE',
                 query_origin='UNMODIFIED_ACTUAL_BEFORE_INITIALIZER_NATIVE_CHANNEL',GT_query_used=False)
        rows.append(row)
    constructed=np.array([r['source_BEFORE_query_constructed']for r in rows],bool)
    assert constructed.sum()==6112 and (~constructed).sum()==32 and np.isfinite(g).all()and not g[~constructed].any()
    for base in range(192):
        for k in range(8):
            ix=[base*32+vi*8+k for vi in range(4)]
            for key in('query_xy','initial_points','L','context_box_xyxy','source_BEFORE_occurrence_index'):
                assert all(rows[i][key]==rows[ix[0]][key]for i in ix),key
            assert all(np.array_equal(g[i],g[ix[0]])for i in ix)
    return rows,g,constructed

def extract_prior_safe(p,old_rows):
    """Whitelist inference members only; never hash/decode mixed packet GT."""
    before=np.array([b*32+k for b in range(192)for k in range(8)],np.int32)
    old_constructed=np.array([r['feature_constructed']for r in old_rows],bool)
    saved=dict(before_indices=before,old_actual_constructed=old_constructed)
    origins={}
    for model,path in [('C16',Path(p['C16_seal_body']['predictions']['path']))]+[(m,RUN133/ARMS[m]/'FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz')for m in('C24','C25')]:
        # C24/C25 archives also contain labels. Do not full-file hash them.
        with np.load(path,allow_pickle=False)as f:
            selected={k:np.array(f[k],copy=True)for k in('global_index','occurrence_id','query_xy','L','raw_head_logit_before_U','prediction','query_available','head_forwarded')}
        assert np.array_equal(selected['global_index'],np.arange(N))and np.array_equal(selected['head_forwarded'],old_constructed)
        assert np.array_equal(selected['occurrence_id'],np.array([r['occurrence_id']for r in old_rows]))
        assert np.array_equal(selected['query_xy'],np.array([r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]for r in old_rows]),equal_nan=True)
        for key in('raw_head_logit_before_U','prediction','query_available','head_forwarded'):
            saved[model+'_'+key+'_before']=selected[key][before]
        origins[model]=dict(path=str(path.resolve()),read_member_whitelist=list(selected),
                            member_array_sha256={k:ash(v)for k,v in selected.items()},full_packet_hashed=False,
                            GT_members_read=False,raw_score_member_only_not_label_selection=True)
    np.savez_compressed(PRIOR,**saved)
    return origins

def generated(p):
    # Batch topology matters for exact BEFORE score parity, especially last456.
    old=np.arange(6088);new=np.r_[old,np.arange(7000,7024)]
    groups=[old[i:i+512]for i in range(0,len(old),512)]
    extra=np.setdiff1d(new,old);assert len(groups)==12 and len(groups[-1])==456 and len(extra)==24
    z=np.array([-1.,-0.,0.,1.,np.nan],np.float32);a=np.array([1,1,1,1,0],bool)
    assert np.where(a,(z>=0).astype(np.int8),-1).tolist()==[0,1,1,1,-1]
    return dict(completed=True,PASS=True,code=bind(__file__),old_head512_layout_retained_last456=True,
                supplementary24_never_BEFORE=True,hidden_zero_E_unknown_U=True,
                source_RGB_GT_checkpoint_encoder_model_reads=0)

def prepare():
    old=read(OLD_SEAL);init=read(INIT_SEAL);parent=read(PARENT_PROTOCOL)
    assert old['completed']and old['PASS']and old['GT_read']is False and init['completed']and init['PASS']and init['GT_read']is False
    assert same(old['initializer_seal'],bind(INIT_SEAL))and same(old['protocol'],bind(PARENT_PROTOCOL))
    original=read(check(old['input_crop_facts']))['rows'];assert len(original)==N
    gs=read(GEOMETRY_SEAL);assert gs['completed']and gs['PASS']and gs['GT_read']is False
    geometry_code=V133.parent/'query_visibility_loco_native_context_20261023_v74/source_contract/native_context_contract_v2.py'
    geometry=pure(geometry_code,('input_context_features',))['input_context_features']
    target=pure(parent['c16']['target_crop_code']['path'],('query_extent','valid_box'))
    grid=pure(parent['native_grid_code']['path'],('native_grid',))['native_grid']
    rows,g,constructed=intervention_rows(original,target,grid,geometry)
    write(INPUTS,dict(completed=True,rows=rows,source='FROZEN_ACTUAL_BEFORE_QUERY_SOURCE_INPUT_INTERVENTION',GT_read=False))
    np.savez_compressed(GFILE,G42=g,geometry_constructed=constructed,global_index=np.arange(N,dtype=np.int32),
                        source_BEFORE_occurrence_index=np.array([r['source_BEFORE_occurrence_index']for r in rows],np.int32))
    write(INPUT_SEAL,dict(completed=True,PASS=True,input_rows=bind(INPUTS),G42=bind(GFILE),rows=N,
                         constructed=int(constructed.sum()),noquery=int((~constructed).sum()),all192_bases4variants8slots_preserved=True,
                         original_actual_BEFORE_origin_seal=bind(INIT_SEAL),source_input_intervention=True,
                         actual_query_for_changed_image=False,query_coordinates_unmodified_from_BEFORE=True,
                         GT_RGB_feature_body_checkpoint_encoder_or_model_reads=0))
    p=dict(completed=True,code=bind(__file__),parent_code=bind(PARENT_CODE),parent_protocol=bind(PARENT_PROTOCOL),
           input_seal=bind(INPUT_SEAL),input_rows=bind(INPUTS),G42=bind(GFILE),C16_seal=bind(OLD_SEAL),
           actual_BEFORE_query_origin_seal=bind(INIT_SEAL),original_input_rows=old['input_crop_facts'],
           old_visual_features=old['features'],old_geometry_seal=bind(GEOMETRY_SEAL),old_G42=gs['G42'],
           geometry_code=bind(geometry_code),C16=parent['c16'],C16_seal_body=old,trainer=bind(TRAINER),models={},output=str(OUT),
           source_input_intervention=True,actual_query_for_changed_image=False,
           input_scope='All6144, actual BEFORE q8/L/context repeated4 across unmodified source imagevariants; no GT/oracle recrop',
           feature_math='Exact V132 C11/C8/C9/V90/V107, FP32 frozen DINO batch8 in native8 whole-frame groups',
           head_math='Old6088 actualconstructed headlayout batch512,last456 retained; newlyconstructible24 supplementary, never BEFORE',
           teacher_or_reference_or_error_reads_or_hashes=False,fit=False,current223=False,V112=False,PnP=False,
           deployed_q_correction=False,new_normalizer=False,threshold_choice=False,source_application_admitted=False)
    for m in('C24','C25'):
        freeze_path=RUN133/ARMS[m]/'MODEL_FREEZE.json';freeze=read(freeze_path)
        assert freeze['completed']and freeze['selected_update']==8000 and freeze['threshold_logit']==0
        p['models'][m]=dict(freeze=bind(freeze_path),checkpoint=freeze['checkpoint'],arm=ARMS[m],update=8000)
    p['prior_safe_member_origins']=extract_prior_safe(p,original)
    p.pop('C16_seal_body')
    p['prior_safe_scores']=bind(PRIOR)
    gp=H/'GENERATED_INPUT_AND_BATCH_CHECKS.json';write(gp,generated(p));p['generated']=bind(gp)
    p['metadata_counts']=dict(rows=N,constructed=6112,noquery=32,old_constructed=6088,newly_constructible=24,
                              BEFORE_rows=1536,BEFORE_constructed=1528,noquery_BEFORE_base_indices=[96])
    write(PROTOCOL,p)
    ep=H/'ROOT_FORWARD_ENTRY_TEMPLATE.json'
    write(ep,dict(approved_frozen_before_query_control=False,fit=False,current223=False,V112=False,PnP=False,
                  GT_read=False,threshold_choice=False,normalizer_fit=False,deployed_q_correction=False,automatic_retry=False,
                  code=p['code'],protocol=bind(PROTOCOL),input_seal=bind(INPUT_SEAL),generated=p['generated']))
    print(json.dumps(dict(completed=True,code=p['code'],protocol=bind(PROTOCOL),input_seal=bind(INPUT_SEAL),template=bind(ep),
                         actual_encoder_model_forward=0,command=parent['C16']['runtime']['executable']+' '+str(Path(__file__).resolve())+' --entry ROOT_ACTUAL_ENTRY.json')
          if 'C16'in parent else dict(completed=True,code=p['code'],protocol=bind(PROTOCOL),input_seal=bind(INPUT_SEAL),template=bind(ep),
                                    actual_encoder_model_forward=0,command=parent['c16']['runtime']['executable']+' '+str(Path(__file__).resolve())+' --entry ROOT_ACTUAL_ENTRY.json')),flush=True)

def authorize(entrypath):
    e=read(entrypath);assert e['approved_frozen_before_query_control']is True
    assert all(e[k]is False for k in('fit','current223','V112','PnP','GT_read','threshold_choice','normalizer_fit','deployed_q_correction','automatic_retry'))
    assert same(e['code'],bind(__file__))and same(e['protocol'],bind(PROTOCOL))
    p=read(PROTOCOL);assert same(p['code'],e['code'])and same(e['input_seal'],p['input_seal'])and same(e['generated'],p['generated'])
    for key in('parent_code','parent_protocol','input_seal','input_rows','G42','C16_seal','actual_BEFORE_query_origin_seal',
               'original_input_rows','old_visual_features','old_geometry_seal','old_G42','geometry_code','trainer','prior_safe_scores','generated'):
        check(p[key])
    assert read(p['generated']['path'])['PASS']and same(read(p['generated']['path'])['code'],e['code'])
    seal=read(p['input_seal']['path']);assert seal['completed']and seal['PASS']and seal['source_input_intervention']and not seal['actual_query_for_changed_image']
    for key in('C11_producer','C8_preprocess','C9_target_preprocess','local_crop_code','target_crop_code','availability_code',
               'asset_runtime','asset_plan','weight','source_model_freeze','checkpoint','head_class_code'):
        check(p['C16'][key])
    for b in p['C16']['official_source_bindings']:check(b)
    for model in('C24','C25'):
        check(p['models'][model]['freeze']);check(p['models'][model]['checkpoint'])
    rows=read(p['input_rows']['path'])['rows'];assert len(rows)==N
    assert not OUT.exists(),'Preserve completed/failed attempt; no automatic retry.'
    return e,p,rows

def encode(p,rows,out):
    import torch
    m=module(p['parent_code']['path'],'v134_frozen_input_helpers');a=p['C16']
    producer=module(a['C11_producer']['path'],'v134_exact_C11');c8=module(a['C8_preprocess']['path'],'v134_exact_C8')
    c9=module(a['C9_target_preprocess']['path'],'v134_exact_C9');target=module(a['target_crop_code']['path'],'v134_exact_context')
    crop=pure(a['local_crop_code']['path'],('crop',))['crop'];avfn=pure(a['availability_code']['path'],('factual_availability',))['factual_availability']
    constructed=np.array([r['source_BEFORE_query_constructed']for r in rows],bool);indices=np.flatnonzero(constructed)
    features=np.zeros((N,1928),np.float32);factual=np.zeros(N,bool);context=factual.copy();qinside=factual.copy();central4=factual.copy()
    metadata=[dict(r)for r in rows];cache=m.PNGCache();out.joinpath('crop_chunks').mkdir();chunks=[]
    for lo in range(0,N,512):
        batch=rows[lo:lo+512];rgb=np.zeros((len(batch),96,96,3),np.uint8);valid=np.zeros((len(batch),96,96),bool)
        for k,r in enumerate(batch):
            if r['source_BEFORE_query_constructed']:
                cr,vr=crop(cache.get(r),np.asarray(r['query_xy']),r['L'],96);rgb[k]=cr.transpose(1,2,0);valid[k]=vr[0]
            metadata[lo+k].update(rgb96_sha256=ash(rgb[k]),valid96_sha256=ash(valid[k]),crop_constructed=bool(constructed[lo+k]))
        path=out/'crop_chunks'/('CROPS_%06d_%06d.npz'%(lo,lo+len(batch)))
        np.savez_compressed(path,rgb96=rgb,valid96=valid,crop_constructed=constructed[lo:lo+len(batch)],global_index=np.arange(lo,lo+len(batch),dtype=np.int32))
        b=bind(path);chunks.append(b)
        for k in range(len(batch)):metadata[lo+k].update(rgb96_chunk=b,rgb96_row=k)
    sys.path.insert(0,a['official_repo'])
    from dinov2.hub.backbones import dinov2_vits14_reg
    encoder=dinov2_vits14_reg(pretrained=False);state=torch.load(a['weight']['path'],map_location='cpu',weights_only=True)
    encoder.load_state_dict(state,strict=True);del state
    encoder=encoder.cuda().float().eval().requires_grad_(False);es=producer.state_sha(encoder)
    assert es==read(a['asset_runtime']['path'])['state_before_SHA256'];calls=0;repeat=False;started=time.monotonic()
    for lo in range(0,len(indices),8):
        ix=indices[lo:lo+8];batch=[rows[int(i)]for i in ix];n=len(batch);lr=[];lv=[];tr=[];tv=[];fractions=[];c4s=[]
        assert n==8 and len({r['scene_index']for r in batch})==1
        for i,r in zip(ix,batch):
            image=cache.get(r);cr,vr=crop(image,np.asarray(r['query_xy']),r['L'],96)
            assert ash(cr.transpose(1,2,0))==metadata[int(i)]['rgb96_sha256']and ash(vr[0])==metadata[int(i)]['valid96_sha256']
            tc,vt=target.sample_target_context(image,r['context_box_xyxy']);fa,ca,aa,c4=avfn(r['query_xy'],r['raw_hw'],vr,vt)
            assert aa==(fa and ca);factual[i]=fa;context[i]=ca;central4[i]=c4
            lr.append(cr);lv.append(vr);tr.append(tc.transpose(2,0,1));tv.append(vt[None]);c4s.append(c4)
            fractions.append([np.float32(vt.mean()),np.float32(vr.mean()),np.float32(vr[:,32:64,32:64].mean())])
        local,_=c8.preprocess_views(torch.as_tensor(np.stack(lr),device='cuda'),torch.as_tensor(np.stack(lv),device='cuda'))
        ctx=c9.preprocess_target(torch.as_tensor(np.stack(tr),device='cuda'),torch.as_tensor(np.stack(tv),device='cuda'));views=torch.cat((ctx,local),0)
        with torch.inference_mode():
            z=encoder.forward_features(views);calls+=1
            if not repeat:
                zz=encoder.forward_features(views);calls+=1;assert all(torch.equal(z[k],zz[k])for k in('x_norm_clstoken','x_norm_patchtokens'));repeat=True
            cl=z['x_norm_clstoken'];pt=z['x_norm_patchtokens']
            x,inside=producer.assemble(cl[:n],pt[:n],cl[n:2*n],pt[n:2*n],cl[2*n:],[r['query_xy']for r in batch],[r['context_box_xyxy']for r in batch],fractions,c4s)
        features[ix]=x.cpu().numpy();qinside[ix]=np.asarray(inside,bool)
        if(lo//8+1)%64==0 or lo+n==len(indices):print(json.dumps(dict(feature_rows=lo+n,total=len(indices),encoder_calls=calls,seconds=time.monotonic()-started)),flush=True)
    assert producer.state_sha(encoder)==es and all(not z.requires_grad for z in encoder.parameters())and all(not z.training for z in encoder.modules());del encoder
    assert np.isfinite(features).all()and not features[~constructed].any()and np.array_equal(features[:,1927],qinside.astype(np.float32))
    available=constructed&factual&context
    for i,r in enumerate(metadata):r.update(feature_constructed=bool(constructed[i]),query_available=bool(available[i]),factual_query_available=bool(factual[i]),context_available=bool(context[i]),Q_inside=bool(qinside[i]),RGB_central4_available=bool(central4[i]))
    return features,constructed,available,metadata,dict(encoder_calls=calls,first_repeat_exact=repeat,encoder_state_sha256=es,encoder_state_unchanged=True)

def run(entrypath):
    e,p,rows=authorize(entrypath);OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entrypath),input_seal=p['input_seal'],GT_read=False,source_input_intervention=True))
    os.environ['XFORMERS_DISABLED']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    started=time.monotonic()
    try:
        import torch,cv2
        assert torch.__version__=='2.1.1+cu118'and np.__version__=='1.26.4'and cv2.__version__=='4.9.0'
        assert Path(sys.executable).resolve()==Path(p['C16']['runtime']['executable']).resolve()and torch.cuda.is_available()
        torch.set_num_threads(1);cv2.setNumThreads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
        features,constructed,available,metadata,encoder_proof=encode(p,rows,OUT)
        with np.load(p['prior_safe_scores']['path'],allow_pickle=False)as f:prior={k:np.array(f[k],copy=True)for k in f.files}
        before=prior['before_indices'];old_indices=np.flatnonzero(prior['old_actual_constructed']);new_indices=np.flatnonzero(constructed&~prior['old_actual_constructed'])
        assert len(old_indices)==6088 and len(new_indices)==24 and not np.isin(new_indices,before).any()
        assert constructed[old_indices].all()
        with np.load(p['old_visual_features']['path'],allow_pickle=False)as f:
            assert np.array_equal(features[before],f['features'][before])and np.array_equal(constructed[before],f['feature_constructed'][before])
        original=read(p['original_input_rows']['path'])['rows']
        for i in before:
            for key in('query_xy','initial_points','L','context_box_xyxy','rgb96_sha256','valid96_sha256','query_available'):
                assert metadata[int(i)][key]==original[int(i)][key],(int(i),key)
        with np.load(p['G42']['path'],allow_pickle=False)as f:G=np.array(f['G42'],copy=True)
        with np.load(p['old_G42']['path'],allow_pickle=False)as f:assert np.array_equal(G[before],f['G42'][before])
        common=dict(global_index=np.arange(N,dtype=np.int32),source_occurrence_index=np.arange(N,dtype=np.int32),corpus=np.full(N,'NATIVE_FROZEN_BEFORE_QUERY_INTERVENTION'),
                    occurrence_id=np.array([r['occurrence_id']for r in rows]),scene_index=np.array([r['scene_index']for r in rows],np.int32),
                    base_index=np.array([r['base_index']for r in rows],np.int32),native_slot=np.array([r['native_slot']for r in rows],np.int8),
                    query_xy=np.array([r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]for r in rows],np.float64),L=np.array([r['L']if r['L']is not None else np.nan for r in rows],np.float64),
                    source_BEFORE_occurrence_index=np.array([r['source_BEFORE_occurrence_index']for r in rows],np.int32))
        feature_path=OUT/'SOURCE_FEATURES_INPUT_ONLY.npz';np.savez_compressed(feature_path,features=features,feature_constructed=constructed,query_available=available,**common)
        geometry_path=OUT/'G42_INPUT_ONLY.npz';np.savez_compressed(geometry_path,G42=G,geometry_constructed=constructed,**common)
        facts_path=OUT/'INPUT_ROWS_AND_CROP_FACTS.json';write(facts_path,dict(completed=True,rows=metadata,source_input_intervention=True,actual_query_for_changed_image=False,GT_read=False))
        statefn=pure(p['trainer']['path'],('state_sha',),{'hashlib':hashlib})['state_sha'];model_bindings={};parity={}
        c16module=module(p['C16']['head_class_code']['path'],'v134_exact_C16_head')
        for model in MODELS:
            if model=='C16':
                head=c16module.SpatialTargetQueryHead();checkpoint=p['C16']['checkpoint'];update=4500
            else:
                ns={'torch':torch};exec(compile(class_ast(p['trainer']['path'],'GeometryQueryHead'),p['trainer']['path'],'exec'),ns)
                head=ns['GeometryQueryHead']();checkpoint=p['models'][model]['checkpoint'];update=8000
            payload=torch.load(checkpoint['path'],map_location='cpu',weights_only=True);assert payload['update']==update
            if model!='C16':assert payload['arm']==p['models'][model]['arm']and payload['threshold_logit']==0
            head.load_state_dict(payload['state_dict'],strict=True);del payload
            head=head.cuda().float().eval().requires_grad_(False);hs=statefn(head);raw=np.full(N,np.nan,np.float32);repeat=False;calls=0
            for pack in(old_indices,new_indices):
                for lo in range(0,len(pack),512):
                    ix=pack[lo:lo+512];x=torch.as_tensor(features[ix],device='cuda')
                    geometry=torch.as_tensor(G[ix]if model=='C25'else np.zeros((len(ix),42),np.float32),device='cuda')
                    with torch.inference_mode():
                        z=head(x)if model=='C16'else head(x,geometry)
                        if not repeat:
                            again=head(x)if model=='C16'else head(x,geometry);assert torch.equal(z,again);repeat=True
                    raw[ix]=z.cpu().numpy();calls+=1
            assert statefn(head)==hs and all(not z.requires_grad for z in head.parameters())and all(not z.training for z in head.modules())
            assert np.isfinite(raw[constructed]).all()and np.isnan(raw[~constructed]).all()
            prediction=np.where(available,(raw>=0).astype(np.int8),-1).astype(np.int8)
            assert np.array_equal(raw[before],prior[model+'_raw_head_logit_before_U_before'],equal_nan=True)
            assert np.array_equal(prediction[before],prior[model+'_prediction_before'])and np.array_equal(available[before],prior[model+'_query_available_before'])
            path=OUT/(model+'_PREDICTIONS_INPUT_ONLY.npz')
            np.savez_compressed(path,raw_head_logit_before_U=raw,query_logit=np.where(available,raw,0).astype(np.float32),prediction=prediction,query_available=available,head_forwarded=constructed,**common)
            model_bindings[model]=dict(predictions=bind(path),checkpoint=checkpoint,update=update,head_state_sha256=hs,state_unchanged=True,repeat_exact=repeat,head_batches=calls)
            parity[model]=dict(BEFORE1536_raw_exact=True,BEFORE1536_prediction_availability_exact=True,old6088_head512_last456_preserved=True,supplementary24_not_BEFORE=True)
            check(checkpoint);del head
        paritypath=OUT/'BEFORE_PARITY.json';write(paritypath,dict(completed=True,PASS=True,BEFORE1536_feature_G42_crop_qLcontext_exact=True,models=parity,GT_read=False))
        seal=dict(completed=True,PASS=True,code=e['code'],protocol=e['protocol'],entry=bind(entrypath),input_seal=p['input_seal'],features=bind(feature_path),G42=bind(geometry_path),
                  input_crop_facts=bind(facts_path),models=model_bindings,BEFORE_parity=bind(paritypath),rows=N,constructed=int(constructed.sum()),available=int(available.sum()),
                  original_actual_BEFORE_origin_seal=p['actual_BEFORE_query_origin_seal'],source_input_intervention=True,actual_query_for_changed_image=False,
                  noquery_BEFORE_base96_all32_preserved=True,newly_constructible_changed_image_slots=24,threshold_hidden_logit=0,**encoder_proof,
                  GT_read=False,fit_current_V112_PnP_deployed_q_correction=0,seconds=time.monotonic()-started)
        write(OUT/'PREDICTION_SEAL.json',seal);print(json.dumps(dict(completed=True,seal=bind(OUT/'PREDICTION_SEAL.json'),available=int(available.sum()),seconds=time.monotonic()-started)),flush=True)
    except BaseException:
        write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare:prepare()
    elif a.entry:run(a.entry)
    else:raise SystemExit('Use --prepare or root --entry')
