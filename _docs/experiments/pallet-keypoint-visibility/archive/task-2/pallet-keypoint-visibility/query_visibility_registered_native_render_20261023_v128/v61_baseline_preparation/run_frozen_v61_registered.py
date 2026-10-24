"""Inert source-only six-full-RGB -> native P3 ROI65 -> frozen v61 baseline.

No teacher reader. Model imports/parameters/image payloads are delayed to --entry.
Normal detector outputs from the pinned backbone wrapper are ignored entirely.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse, ast, hashlib, importlib.util, json, os, time, traceback
import numpy as np

H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
TASK = P.parent
PROTOCOL = H / 'BASELINE_PROTOCOL_BEFORE_FORWARD.json'
OUT = V / 'v61_frozen_registered_baseline'
ASSETS = {
    'backbone_code': (P/'query_visibility_public_real_20261022_v63/model/deployment_assets/frozen_yolo_features.py', '296d81b2adde5ce74192f0012e554ae546c95f309a82dd808024ee34105721ab'),
    'backbone_checkpoint': (P/'query_visibility_public_real_20261022_v63/model/deployment_assets/yolo26_frozen.pt', '970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'),
    'spatial_math': (P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py', '6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'),
    'head_class': (TASK/'external_ray_proxy_head_20261022_v60/head_and_roi.py', '2c0f3bbeb29cb6f93152a62a3183d592d7b8c22bb45c20b2a84b80dc06843de0'),
    'head_checkpoint': (TASK/'target_presence_classifier_20261022_v61/BEST_PRESENCE_HEAD.pt', 'c75e04c41b8097ff63c2f134f76e12ec560603e7bcfe585446fe6042e31e4387'),
}
PROXY_AST_SHA = '3ad61bc66c48d177832830d7277d462a47542979c0bf4f215eb147aa253471ad'
RUNTIME = '/home/minjae/anaconda3/envs/pallet-yolo26/bin/python3.10'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def bind(p): return {'path':str(Path(p).resolve()), 'sha256':sha(p)}
def check(b):
    assert sha(b['path'])==b['sha256'], b['path']
    return Path(b['path'])
def read(p): return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f: json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def arrsha(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def module(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def state_sha(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        a=v.detach().cpu().contiguous().numpy()
        h.update(k.encode()+b'\0'+str(a.dtype).encode()+b'\0'+str(a.shape).encode()+b'\0'+a.tobytes())
    return h.hexdigest()
def decode(z,a): return np.where(a,np.where(np.asarray(z)>=0,0,1),-1).astype(np.int8)
def proxy_ast():
    node=next(n for n in ast.parse(ASSETS['head_class'][0].read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ProxyHead')
    assert hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()==PROXY_AST_SHA
    return ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
def validate_rows(rows):
    assert len(rows)==432 and [r['occurrence_index']for r in rows]==list(range(432))
    groups={}
    for r in rows:
        q=np.asarray(r['query_xy'],np.float64);p=np.asarray(r['initial_points'],np.float64)
        assert q.shape==(2,) and p.shape==(8,2) and np.isfinite(p).all() and np.isfinite(q).all()
        assert np.array_equal(q,p[r['carrier_slot']]) and np.isfinite(r['L']) and r['L']>0
        assert r['full_RGB_format']=='NPZ_RGB_UINT8_HWC' and r['full_RGB_key']=='RGB_uint8' and r['raw_hw']==[256,256]
        k=(r['full_RGB']['path'],r['full_RGB']['sha256'])
        groups.setdefault(k,[]).append(r['occurrence_index'])
        first=rows[groups[k][0]]
        assert r['decoded_RGB_sha256']==first['decoded_RGB_sha256'] and r['scene_id']==first['scene_id']
    assert len(groups)==6 and sorted(map(len,groups.values()))==[72]*6
    return list(groups.values())
def authorize(entry):
    e=read(entry)
    assert e['approved_source_baseline'] is True
    assert all(e[k] is False for k in ('fit','current223','V112','PnP','query_reselection','automatic_retry'))
    for k in ('code','protocol','generated','input_rows','input_crops','input_complete','render_complete'): check(e[k])
    p=read(PROTOCOL)
    assert e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL) and p['code']==e['code'] and p['generated']==e['generated']
    assert read(e['generated']['path'])['PASS'] and read(e['generated']['path'])['code']==e['code']
    receipt=read(e['input_complete']['path'])
    assert receipt['completed'] and receipt['PASS'] and read(e['render_complete']['path'])['completed']
    assert receipt['input_manifest']==e['input_rows'] and receipt['stored_crops']==e['input_crops'] and receipt['render_complete']==e['render_complete']
    for b in p['assets'].values(): check(b)
    assert not OUT.exists(), 'Preserve completed/failed attempt; never silently rerun.'
    return e,p
def run(entry):
    e,p=authorize(entry)
    import torch, cv2, PIL
    assert str(Path(sys.executable).resolve())==str(Path(RUNTIME).resolve())
    assert torch.__version__=='2.1.1+cu118' and np.__version__=='1.26.4' and cv2.__version__=='4.9.0' and PIL.__version__=='12.2.0'
    torch.set_num_threads(1);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=read(e['input_rows']['path'])['rows'];groups=validate_rows(rows)
    OUT.mkdir();write(OUT/'STARTED.json',{'entry':bind(entry),'automatic_retry':False})
    back=None;start=time.monotonic()
    try:
        os.environ['YOLO_CONFIG_DIR']=str(H/'runtime/ultralytics')
        math=module(p['assets']['spatial_math']['path'],'v128_v90_spatial_math')
        bm=module(p['assets']['backbone_code']['path'],'v128_v63_frozen_yolo')
        back=bm.FrozenYoloFeatures(p['assets']['backbone_checkpoint']['path'],device='cuda:0')
        # Explicit setup/fusion warmup, never reused as a source scene.
        with torch.inference_mode(): back.predict(np.zeros((256,256,3),np.uint8),already_padded=False,cpu_features=True)
        before=state_sha(back.yolo.model);roi=np.empty((432,65,9,9),np.float16)
        qvalid=np.zeros(432,bool);any9=qvalid.copy();central4=qvalid.copy();scene_facts=[]
        with np.load(e['input_crops']['path'],allow_pickle=False)as crops:
            assert np.array_equal(crops['occurrence_index'],np.arange(432))
            for indices in groups:
                r=rows[indices[0]];check(r['full_RGB'])
                # NPZ is lazy: access the RGB key only, never owner/depth/teacher arrays.
                with np.load(r['full_RGB']['path'],allow_pickle=False) as f: rgb=np.array(f['RGB_uint8'],copy=True)
                assert rgb.dtype==np.uint8 and rgb.shape==(256,256,3) and arrsha(rgb)==r['decoded_RGB_sha256']
                with torch.inference_mode(): z=back.predict(np.ascontiguousarray(rgb[...,::-1]),already_padded=False,cpu_features=True)
                a=math.expected_affine(r['raw_hw']);gain,offset=bm.canvas_affine(z['canvas_shape'],z['input_shape'])
                assert z['added_border']==100 and list(z['canvas_shape'])==a['canvas_shape'] and list(z['input_shape'])==a['input_shape']
                assert gain==a['gain'] and np.array_equal(offset,a['offset'])
                h,w=a['p3_actual_hw'];plane=z['p3'].numpy()
                assert plane.dtype==np.float16 and plane.shape==(64,80,80) and not plane[:,h:,:].any() and not plane[:,:,w:].any()
                q=np.array([rows[i]['query_xy']for i in indices],np.float64);L=np.array([rows[i]['L']for i in indices],np.float64)
                x,av,v=math.sample_p3(plane[:,:h,:w],q,L,r['raw_hw'],gain,offset,100)
                roi[indices]=x;qvalid[indices]=math.query_validity(q,r['raw_hw'])[0];any9[indices]=av
                for k,i in enumerate(indices):
                    c,valid=math.crop(rgb,q[k],L[k],96);expected=c.transpose(1,2,0)
                    stored_v=np.asarray(crops['valid96'][rows[i]['rgb96_row']]).reshape(96,96)
                    assert np.array_equal(crops['rgb96'][rows[i]['rgb96_row']],expected) and np.array_equal(stored_v,valid[0])
                    assert arrsha(expected)==rows[i]['rgb96_sha256'] and arrsha(stored_v)==rows[i]['valid96_sha256']
                    central4[i]=valid[0,47:49,47:49].all()
                scene_facts.append({'scene_id':r['scene_id'],'full_RGB':r['full_RGB'],'decoded_RGB_sha256':arrsha(rgb),'rows':indices,'canvas_shape':list(z['canvas_shape']),'input_shape':list(z['input_shape']),'P3_actual_shape':[64,h,w],'gain':gain,'offset':offset.tolist(),'added_border':100,'full_P3_persisted':False})
        assert np.isfinite(roi).all() and np.array_equal(roi[:,64].astype(bool).reshape(432,-1).any(1),any9)
        assert state_sha(back.yolo.model)==before and all(not v.requires_grad for v in back.yolo.model.parameters())
        back.close();back=None
        ns={'nn':torch.nn};exec(compile(proxy_ast(),str(ASSETS['head_class'][0]),'exec'),ns)
        head=ns['ProxyHead']();assert sum(x.numel()for x in head.parameters())==28033
        head.load_state_dict(torch.load(p['assets']['head_checkpoint']['path'],map_location='cpu'),strict=True)
        head.requires_grad_(False).eval().cuda();hs=state_sha(head);logits=[]
        with torch.inference_mode():
            for lo in range(0,432,128):
                x=torch.from_numpy(np.ascontiguousarray(roi[lo:lo+128])).float().cuda();z=head(x).float().cpu().numpy()
                if lo==0: assert np.array_equal(z,head(x).float().cpu().numpy())
                logits.append(z)
        z=np.concatenate(logits);assert np.isfinite(z).all() and state_sha(head)==hs
        available=qvalid&any9;pred=decode(z,available)
        np.savez_compressed(OUT/'P3_ROI432_INPUT_ONLY.npz',p3_roi=roi,occurrence_index=np.arange(432,dtype=np.int32))
        np.savez_compressed(OUT/'PREDICTIONS432.npz',raw_visible_logit=z,raw_hidden_logit=-z,prediction_D0_E1_Uminus1=pred,query_available=available,native_query_valid=qvalid,ROI_any_valid=any9,RGB_central4_available=central4,occurrence_index=np.arange(432,dtype=np.int32),query_id=np.array([r['id']for r in rows]),query_xy=np.array([r['query_xy']for r in rows],np.float64),L=np.array([r['L']for r in rows],np.float64),logical_slot=np.array([r['logical_slot']for r in rows],np.int32),scene_id=np.array([r['scene_id']for r in rows]))
        write(OUT/'INPUT_FACTS.json',{'completed':True,'scenes':scene_facts,'all432_saved_RGB96_and_valid_exact':True,'availability':'Original v61 native centre continuous AND nearest valid AND ROI-any; central4 stored separately, never used as head tensor.','native_query_valid':int(qvalid.sum()),'ROI_any_valid':int(any9.sum()),'RGB_central4_available':int(central4.sum()),'query_available':int(available.sum())})
        for b in p['assets'].values():check(b)
        write(OUT/'PREDICTION_SEAL.json',{'completed':True,'PASS':True,'code':e['code'],'protocol':e['protocol'],'entry':bind(entry),'input_rows':e['input_rows'],'input_crops':e['input_crops'],'input_complete':e['input_complete'],'render_complete':e['render_complete'],'checkpoint':p['assets']['head_checkpoint'],'backbone_checkpoint':p['assets']['backbone_checkpoint'],'predictions':bind(OUT/'PREDICTIONS432.npz'),'P3_ROIs':bind(OUT/'P3_ROI432_INPUT_ONLY.npz'),'input_facts':bind(OUT/'INPUT_FACTS.json'),'occurrences':432,'source_scene_predict_calls':6,'blank_setup_warmup_calls':1,'head_batches':4,'head_repeat_batches':1,'backbone_state_unchanged':True,'head_state_unchanged':True,'decoder':'raw_visible_logit>=0 D; <0 E; factual-unavailable U; signedzero D','teacher_read':False,'fit_current_V112_PnP_query_reselection':0,'seconds':time.monotonic()-start})
        print(json.dumps({'completed':True,'seal':bind(OUT/'PREDICTION_SEAL.json'),'seconds':time.monotonic()-start}),flush=True)
    except BaseException:
        write(OUT/'FAILED.json',{'traceback':traceback.format_exc(),'automatic_retry':False});raise
    finally:
        if back is not None:back.close()

def prepare():
    assets={k:{'path':str(v[0]),'sha256':v[1]}for k,v in ASSETS.items()}
    for k in ('backbone_code','spatial_math','head_class'):check(assets[k])
    proxy_ast();g=H/'GENERATED_PREP_CHECK.json';proof=read(g);assert proof['PASS'] and proof['code']==bind(__file__)
    p={'completed':True,'code':bind(__file__),'generated':bind(g),'assets':assets,'ProxyHead_AST_SHA':PROXY_AST_SHA,'runtime':{'python':RUNTIME,'torch':'2.1.1+cu118','numpy':'1.26.4','cv2':'4.9.0','PIL':'12.2.0','device':'cuda:0','FP32_head':True,'TF32':False,'autocast':False},'input':'Root-bound completed input rows432 and exact RGB96/valid packet; six RGB keys from render NPZ only. GT excluded.','feature_recipe':'Pinned V63 border100 REFLECT_101/native640 letterbox -> FP16 P3 -> pinned V90 CPUF32 sample_p3(gridFP64->FP32, index=network/8, alignFalse, paddingzeros, 65thnativevalidchannel) -> FP16 ROI; head CPUfloat->CUDA.', 'decoder':'visible z>=0 D, z<0 E; unavailable U; signedzero D','calls':{'scene_backbone':6,'blank_setup_warmup':1,'head_batches':4,'head_repeat':1},'availability':'Native continuous AND nearest centre valid AND ROI-any. RGBcentral4 separately reported; RGB96 is parity/evidence only and is never a v61 head input.','output':str(OUT),'teacher_body_read':False,'fit_current_V112_PnP_allowed':False,'actual_forward_by_preparation':0}
    write(PROTOCOL,p)
    t={'approved_source_baseline':False,'fit':False,'current223':False,'V112':False,'PnP':False,'query_reselection':False,'automatic_retry':False,'code':p['code'],'protocol':bind(PROTOCOL),'generated':p['generated'],'input_rows':None,'input_crops':None,'input_complete':None,'render_complete':None}
    write(H/'ROOT_BASELINE_ENTRY_TEMPLATE.json',t)
    print(json.dumps({'protocol':bind(PROTOCOL),'template':bind(H/'ROOT_BASELINE_ENTRY_TEMPLATE.json'),'command':RUNTIME+' '+str(Path(__file__).resolve())+' --entry ROOT_ACTUAL_V61_ENTRY.json','actual_forward':0}),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--entry',type=Path);o=a.parse_args()
    if o.prepare:prepare()
    elif o.entry:run(o.entry)
    else:raise SystemExit('Use --prepare (inert), or root-bound --entry.')
