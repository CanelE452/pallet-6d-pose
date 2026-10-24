"""Frozen DINO pooled source features from sealed RGB input packs only."""
from pathlib import Path
import argparse,collections,hashlib,json,os,subprocess,sys,time
import numpy as np
import torch
from pooling import image_input,token_weights,pool_local,pool_target

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;P=ROOT.parent;OUT=ROOT/'feature_pack'
DECISION_SHA='a008fc7d35e9aa1d0d4b7d631a796f238aaba7f0aa052fe7da087080b70115e9'
V76=P/'query_visibility_reference_local_support_20261023_v76'
V77=P/'query_visibility_controlled_pairs_20261023_v77'
REPO=Path('/home/minjae/Documents/github/pallet-pose/outputs/pallet_type_selftrain_v1/selftrain_recovery_v1/dino_localization_assets/dinov2')
COPY_KEYS=['context42','query8_xy','L','query_center_valid','ROI_anyvalid','ROI_center_valid','query_box_xyxy','target_crop_center_xy','target_crop_side','target_context_valid','native_index','initial8_norm','initial8_norm_valid','query_valid_cached','pure_continuous_query_valid','nearest_pixel_valid','query_missing']
IMAGE_KEYS=['rgb96','crop_valid96','target_rgb128','target_valid128']
FORWARD_FIELDS=['rgb_uint8','canvas_valid_bool']

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path']
def load(p):return json.loads(Path(p).read_text())
def rows(p):return[json.loads(x)for x in Path(p).read_text().splitlines()]
def write(p,d):
 with open(p,'x')as f:f.write(json.dumps(d,indent=2)+'\n')
def decisions():
 d=ROOT/'EXPERIMENT_DECISIONS_BEFORE_FEATURES.json';assert sha(d)==DECISION_SHA
 data=load(d)
 for k in ('DINO_checkpoint','DINO_official_code','DINO_official_factory'):check(data['bindings'][k])
 return data
def state_digest(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):
  h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def configure():
 os.environ['XFORMERS_DISABLED']='1'
 os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
 torch.set_num_threads(1);torch.manual_seed(20261023);np.random.seed(20261023)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 assert torch.cuda.is_available()
def model():
 d=decisions();configure()
 head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
 assert head==d['backbone']['code_commit']
 sys.path.insert(0,str(REPO))
 # No pretrained URL path is invoked; fail closed if a local dependency tries one.
 import urllib.request
 def no_network(*a,**kw):raise RuntimeError('Network forbidden by frozen local-only producer')
 urllib.request.urlopen=no_network;urllib.request.urlretrieve=no_network
 torch.hub.load_state_dict_from_url=no_network;torch.hub.download_url_to_file=no_network
 from dinov2.hub.backbones import dinov2_vits14
 m=dinov2_vits14(pretrained=False)
 state=torch.load(d['bindings']['DINO_checkpoint']['path'],map_location='cpu',weights_only=True)
 m.load_state_dict(state,strict=True);del state
 m.eval().requires_grad_(False);m=m.float().cuda()
 assert all(not x.training for x in m.modules())and all(not p.requires_grad for p in m.parameters())
 assert all(p.dtype==torch.float32 for p in m.parameters())
 return m
@torch.inference_mode()
def forward(m,rgb,valid):
 x=image_input(rgb.cuda());w=token_weights(valid.cuda())
 t=m.forward_features(x)['x_norm_patchtokens']
 assert t.shape==(len(rgb),256,384)and t.dtype==torch.float32 and torch.isfinite(t).all()
 return t,w
def independent_pool(t,w,kind):
 # Independent NumPy float64 summation of transient CPU tokens; never archived.
 t=t.detach().cpu().numpy().astype(np.float64).reshape(-1,16,16,384)
 w=w.detach().cpu().numpy().astype(np.float64)
 regions=[(slice(7,9),slice(7,9)),(slice(None),slice(None))]if kind=='local'else[(slice(None),slice(None))]
 out=[]
 for a,b in regions:
  ww=w[:,a,b];mass=ww.sum((1,2));v=(t[:,a,b]*ww[...,None]).sum((1,2))/np.maximum(mass,1e-12)[:,None];v[mass==0]=0;out.append(v)
 return np.stack(out,1)if kind=='local'else out[0]
def preflight():
 assert not(HERE/'GENERATED_GPU_PREFLIGHT.json').exists()
 m=model();before=state_digest(m);g=torch.Generator().manual_seed(20261023)
 rgb=torch.randint(0,256,(32,3,96,96),dtype=torch.uint8,generator=g);v=torch.rand((32,1,96,96),generator=g)>.2;v[0]=False
 t,w=forward(m,rgb,v);tt,ww=forward(m,rgb,v);assert torch.equal(t,tt)and torch.equal(w,ww)
 a,f=pool_local(t,w);b,fb=pool_target(t,w);ref=independent_pool(t,w,'local');diff=float(np.max(np.abs(a.cpu().numpy()-ref)))
 assert diff<=2e-5 and not a[0].any()and not b[0].any()
 assert np.isfinite(a.cpu().numpy().astype(np.float16)).all()
 assert state_digest(m)==before
 d=dict(PASS=True,generated_views=32,backbone_forward_calls=2,repeat_token_bit_differences=0,independent_pool_max_abs_diff=diff,pool_atol_predeclared=2e-5,
  weights=decisions()['bindings']['DINO_checkpoint'],state_digest=before,peak_GPU_bytes=torch.cuda.max_memory_allocated(),runtime=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__),source_RGB_GT_current_reads=0,training_PnP=0,code=bind(Path(__file__)),pooling=bind(HERE/'pooling.py'))
 write(HERE/'GENERATED_GPU_PREFLIGHT.json',d);print(json.dumps(bind(HERE/'GENERATED_GPU_PREFLIGHT.json')))
def banks():
 d=decisions();b=load(d['bindings']['V77_source_inputs']['path']);check(d['bindings']['V77_source_inputs'])
 orig=b['bindings']['original_input_bindings'];check(orig);ob=load(orig['path'])
 result=[];start=0
 for i,x in enumerate(ob['packs']):
  for k in ('manifest','seal','lineage'):check(x[k])
  man=load(x['manifest']['path']);n=man['frame_count']
  result.append(dict(name='ORIGINAL_'+x['source'],domain=0,start=start,end=start+n,manifest=x['manifest'],seal=x['seal'],lineage=x['lineage']));start+=n
 for k in ('after_input_manifest','after_input_seal','after_input_lineage','after_input_QA'):check(b['bindings'][k])
 man=load(b['bindings']['after_input_manifest']['path']);assert load(b['bindings']['after_input_QA']['path'])['PASS']
 result.append(dict(name='CONTROLLED_AFTER',domain=1,start=start,end=start+man['frame_count'],manifest=b['bindings']['after_input_manifest'],seal=b['bindings']['after_input_seal'],lineage=b['bindings']['after_input_lineage']))
 assert[(x['start'],x['end'])for x in result]==[(0,1003),(1003,5203),(5203,10002)]
 return result
def freeze():
 assert load(HERE/'GENERATED_POOLING_TESTS.json')['PASS']and load(HERE/'GENERATED_GPU_PREFLIGHT.json')['PASS']
 assert sha(Path(__file__))==load(HERE/'GENERATED_GPU_PREFLIGHT.json')['code']['sha256']
 bs=banks();code=[]
 for part in ('hub','models','layers'):
  code.extend(bind(p)for p in sorted((REPO/'dinov2'/part).rglob('*.py')))
 code.extend([bind(REPO/'dinov2/__init__.py'),bind(REPO/'LICENSE')])
 bindings=[bind(ROOT/'EXPERIMENT_DECISIONS_BEFORE_FEATURES.json'),bind(HERE/'pooling.py'),bind(Path(__file__)),bind(HERE/'test_generated_pooling.py'),bind(HERE/'GENERATED_POOLING_TESTS.json'),bind(HERE/'GENERATED_GPU_PREFLIGHT.json')]
 for bank in bs:
  bindings.extend(bank[k]for k in ('manifest','seal','lineage'))
  bindings.extend(c['file']for c in load(bank['manifest']['path'])['chunks'])
 write(HERE/'FEATURE_PROTOCOL.json',dict(frozen=True,decisions=bind(ROOT/'EXPERIMENT_DECISIONS_BEFORE_FEATURES.json'),code=bind(Path(__file__)),pooling=bind(HERE/'pooling.py'),bindings=bindings,official_code=code,
  weight=decisions()['bindings']['DINO_checkpoint'],banks=bs,input_NPZ_fields_whitelist=IMAGE_KEYS+COPY_KEYS,forward_fields=FORWARD_FIELDS,
  frames=10002,query_slots=80016,batch_views=32,dtype_model='float32',dtype_features='float16',storage='No patchtokens archived. Target features once/frame. Original8query order unchanged.',
  actual_fixed_repeat_regimes=['ORIGINAL_SYNTHETIC','ORIGINAL_YCB_PUBLIC_REAL','CONTROLLED_AFTER_SYNTHETIC','CONTROLLED_AFTER_YCB_PUBLIC_REAL'],repeat_for_each_kind=['local','target'],independent_pool_atol=2e-5,
  output_local_shape=['N',8,2,384],output_target_shape=['N',384],output_fraction_shape=['N',8,3],context42_bit_exact=True,
  new_training_current_GT_PnP=0,network_fallback=False))
 print(json.dumps(bind(HERE/'FEATURE_PROTOCOL.json')))
def load_input_chunk(c):
 check(c['file'])
 with np.load(c['file']['path'],allow_pickle=False)as z:
  assert all(k in z.files for k in IMAGE_KEYS+COPY_KEYS)
  return{k:z[k]for k in IMAGE_KEYS+COPY_KEYS}
def regime(bank,meta):return bank['name']if bank['domain']==0 else 'CONTROLLED_AFTER_'+meta['source']
def run():
 prot=load(HERE/'FEATURE_PROTOCOL.json');assert prot['frozen']and not OUT.exists()
 for b in prot['bindings']+prot['official_code']+[prot['weight']]:check(b)
 assert prot['decisions']['sha256']==DECISION_SHA;check(prot['code']);check(prot['pooling'])
 OUT.mkdir();(OUT/'chunks').mkdir();(OUT/'fixed_pool_proofs').mkdir()
 m=model();before=state_digest(m);assert before==load(HERE/'GENERATED_GPU_PREFLIGHT.json')['state_digest'];start=time.monotonic()
 proofs=[];done=set();calls=0;views=0;repeat_calls=0;outchunks=[];outrows=[];disk=0;counts=collections.Counter()
 for bank in prot['banks']:
  man=load(bank['manifest']['path']);metadata=rows(bank['lineage']['path']);assert len(metadata)==bank['end']-bank['start']
  for c in man['chunks']:
   a=load_input_chunk(c);F=len(a['rgb96']);local=np.empty((F,8,2,384),np.float16);target=np.empty((F,384),np.float16);lf=np.empty((F,8,2),np.float32);tf=np.empty(F,np.float32)
   for kind in ('local','target'):
    rg=a['rgb96'].reshape(F*8,3,96,96)if kind=='local'else a['target_rgb128']
    vv=a['crop_valid96'].reshape(F*8,1,96,96)if kind=='local'else a['target_valid128']
    features=[];fractions=[]
    for st in range(0,len(rg),32):
     en=min(st+32,len(rg));rgb=torch.from_numpy(rg[st:en]);valid=torch.from_numpy(vv[st:en]);t,w=forward(m,rgb,valid);calls+=1;views+=en-st
     p,f=pool_local(t,w)if kind=='local'else pool_target(t,w)
     regimes={regime(bank,metadata[c['start']+(j//8 if kind=='local'else j)])for j in range(st,en)}
     new=sorted(x for x in regimes if(x,kind)not in done)
     if new:
      tt,ww=forward(m,rgb,valid);calls+=1;repeat_calls+=1;assert torch.equal(t,tt)and torch.equal(w,ww)
      ref=independent_pool(t,w,kind);err=float(np.max(np.abs(p.cpu().numpy()-ref)));assert err<=prot['independent_pool_atol']
      global_frame=np.asarray([bank['start']+c['start']+(j//8 if kind=='local'else j)for j in range(st,en)],np.int64)
      query=np.asarray([j%8 if kind=='local'else-1 for j in range(st,en)],np.int8)
      proof=OUT/'fixed_pool_proofs'/f'{len(proofs):02d}_{kind}.npz'
      np.savez_compressed(proof,pooled_fp32=p.cpu().numpy(),fractions_fp32=f.cpu().numpy(),token_validity_weights=w.cpu().numpy(),frame_index=global_frame,query_index=query)
      proofs.append(dict(regimes=new,kind=kind,file=bind(proof),repeat_token_bit_differences=0,independent_float64_pool_max_abs_diff=err))
      done.update((r,kind)for r in new)
     features.append(p.cpu().numpy().astype(np.float16));fractions.append(f.cpu().numpy())
     del t,w,p,f
    pf=np.concatenate(features);ff=np.concatenate(fractions)
    assert np.isfinite(pf).all()and np.isfinite(ff).all()
    if kind=='local':local[:]=pf.reshape(F,8,2,384);lf[:]=ff.reshape(F,8,2)
    else:target[:]=pf;tf[:]=ff
   global_start=bank['start']+c['start'];global_end=bank['start']+c['end'];meta=metadata[c['start']:c['end']]
   original=np.asarray([global_start+j if bank['domain']==0 else r['original_row_index']for j,r in enumerate(meta)],np.int64)
   fractions=np.concatenate((lf,np.broadcast_to(tf[:,None,None],(F,8,1))),2).astype(np.float16)
   payload={k:a[k]for k in COPY_KEYS};payload.update(local_features=local,target_features=target,validity_fractions=fractions,
    frame_index=np.arange(global_start,global_end,dtype=np.int64),original_row_index=original,domain=np.full(F,bank['domain'],np.int8),
    id=np.asarray([r['id']for r in meta]),source=np.asarray([r.get('source','SYNTHETIC')for r in meta]),split=np.asarray(['val'if r['split']in('val','validation','VALIDATION')else'train'for r in meta]),
    localany=a['crop_valid96'].reshape(F,8,-1).any(2),targetany=a['target_valid128'].reshape(F,-1).any(1))
   p=OUT/'chunks'/f'features_{global_start:06d}_{global_end:06d}.npz';np.savez_compressed(p,**payload);disk+=p.stat().st_size
   outchunks.append(dict(start=global_start,end=global_end,file=bind(p),input_chunk=c['file'],input_bank=bank['name'],input_bank_start=bank['start'],input_start=c['start'],input_end=c['end'],schema={k:dict(shape=list(v.shape),dtype=str(v.dtype))for k,v in payload.items()}))
   for j,r in enumerate(meta):
    outrows.append(dict(frame_index=global_start+j,id=str(payload['id'][j]),source=str(payload['source'][j]),split=str(payload['split'][j]),domain=bank['domain'],original_row_index=int(original[j]),input_bank=bank['name'],input_row_index=c['start']+j))
    counts[(str(payload['source'][j]),str(payload['split'][j]),bank['domain'])]+=1
   print(json.dumps(dict(frames=global_end,total=10002,source_bank=bank['name'],disk_bytes=disk,seconds=time.monotonic()-start)),flush=True)
 assert len(outrows)==10002 and len(done)==8 and sum(x['end']-x['start']for x in outchunks)==10002
 after=state_digest(m);assert after==before and all(not x.training for x in m.modules())and all(not p.requires_grad for p in m.parameters());check(prot['weight'])
 lp=OUT/'INDEX_LINEAGE_INPUT_ONLY.jsonl'
 with open(lp,'x')as f:f.write(''.join(json.dumps(r,separators=(',',':'))+'\n'for r in outrows))
 write(OUT/'FEATURE_INFERENCE_RECEIPT.json',dict(PASS=True,completed=True,unique_source_frames=10002,unique_views_encoded=views,backbone_forward_calls=calls,repeat_forward_calls=repeat_calls,
  state_digest_before=before,state_digest_after=after,weights_unchanged=True,weights=prot['weight'],all_model_parameters_float32=True,no_autocast=True,TF32=False,
  peak_GPU_bytes=torch.cuda.max_memory_allocated(),runtime=dict(python=sys.version,torch=str(torch.__version__),numpy=np.__version__),fixed_batch_repeat_and_pool_proofs=proofs,
  no_patch_tokens_stored=True,RGB_input_only_from_sealed_packs=True,GT_reference_mask_label_current_reads=0,new_training_PnP=0,seconds=time.monotonic()-start))
 write(OUT/'PACK_MANIFEST.json',dict(completed=True,frame_count=10002,query_count=80016,chunks=outchunks,lineage=bind(lp),protocol=bind(HERE/'FEATURE_PROTOCOL.json'),inference_receipt=bind(OUT/'FEATURE_INFERENCE_RECEIPT.json'),
  source_split_domain_counts=[dict(source=s,split=p,domain=d,frames=n)for(s,p,d),n in sorted(counts.items())],NPZ_disk_bytes=disk,no_full_patch_tokens_stored=True,GT_current_training_PnP=0))
 write(OUT/'PACK_SEAL.json',dict(completed=True,PASS=True,manifest=bind(OUT/'PACK_MANIFEST.json'),lineage=bind(lp),inference_receipt=bind(OUT/'FEATURE_INFERENCE_RECEIPT.json'),chunks=[c['file']for c in outchunks],code=bind(Path(__file__)),pooling=bind(HERE/'pooling.py'),protocol=bind(HERE/'FEATURE_PROTOCOL.json'),GT_current_fit_PnP=0))
 print(json.dumps(bind(OUT/'PACK_SEAL.json')),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','freeze','run']);args=p.parse_args();dict(preflight=preflight,freeze=freeze,run=run)[args.stage]()
