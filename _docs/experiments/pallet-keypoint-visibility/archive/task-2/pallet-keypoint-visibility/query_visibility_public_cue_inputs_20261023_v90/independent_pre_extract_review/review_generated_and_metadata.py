"""Independent V90 code/inputmetadata/generated-only pre-extract audit."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,datetime,ast,importlib.util,sys,math
import numpy as np
import torch
from torch.nn import functional as F
O=Path(__file__).resolve().parent;R=O.parent;P=R.parent

def bind(p):
 p=Path(p);v=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(v).hexdigest(),'bytes':len(v)}
def read(p):return json.loads(Path(p).read_text())
def jl(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s.strip()]
def save(p,d):
 with Path(p).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def pure(p,names):
 ns={'np':np,'torch':torch,'F':F};nodes=[n for n in ast.parse(Path(p).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names];assert len(nodes)==len(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns);return ns

def main():
 torch.set_num_threads(1)
 scope=read(O/'SCOPE_BEFORE_GENERATED_REVIEW.json')
 for q in scope['bindings'].values():assert bind(q['path'])==q
 protocol=read(R/'INPUT_PRODUCER_PROTOCOL_BEFORE_PIXELS.json');assert bind(R/'INPUT_PRODUCER_PROTOCOL_BEFORE_PIXELS.json')['sha256']=='1a46e45652712705b452cfea9c527ed0725d06cae8d095cd67757082dd214804'
 assert protocol['code']==bind(R/'producer/prepare_public_spatial_inputs.py')and protocol['math_code']==bind(R/'producer/spatial_input_math.py')
 # Deliberately skip checkpoint even bytehash; only code/inputmetadata bindings.
 verified=[]
 for z in protocol['bindings']:
  assert Path(z['path']).suffix in {'.json','.jsonl','.py'};assert bind(z['path'])==z;verified.append(z)
 assert bind(protocol['assets']['backbone_code']['path'])['sha256']==protocol['assets']['backbone_code']['sha256']
 proof=read(protocol['generated']['path']);assert proof['PASS']is True and bind(protocol['generated']['path'])==protocol['generated']
 template=read(R/'SOURCE_INPUT_ENTRYLOCK_TEMPLATE.json');assert template['approved_by_root']is False and template['current_allowed']is False and template['training_allowed']is False and template['PnP_allowed']is False
 assert template['protocol']==bind(R/'INPUT_PRODUCER_PROTOCOL_BEFORE_PIXELS.json')
 frames=jl(P/'query_visibility_public_cue_grid_20261023_v89/inputs_metadata/FRAMES_INPUTS_ONLY.jsonl');tiles=jl(P/'query_visibility_public_cue_grid_20261023_v89/inputs_metadata/INPUTS_GRID_ONLY.jsonl')
 assert len(frames)==100 and len(tiles)==12900
 assert len({f['image_sha256']for f in frames})==len({f['image']for f in frames})==100
 assert Counter(f['split']for f in frames)=={'TRAIN_LAB':44,'VAL_HALL':56}
 assert Counter(t['roi_side']for t in tiles)=={128:9900,256:3000}
 assert all(sorted(f)==protocol['frame_schema']for f in frames)and all(sorted(t)==protocol['tile_metadata_schema']for t in tiles)
 byframe=defaultdict(list);fmap={f['frame_input_id']:f for f in frames}
 for i,t in enumerate(tiles):
  f=fmap[t['frame_input_id']];assert all(t[k]==v for k,v in f.items());byframe[t['frame_input_id']].append((i,t))
  H,W=t['raw_hw'];x,y=t['query_xy'];L=t['roi_side'];assert t['window_xyxy_half_open']==[x-L/2,y-L/2,x+L/2,y+L/2]
 assert all(len(v)==129 for v in byframe.values())and[i for f in frames for i,t in byframe[f['frame_input_id']]]==list(range(12900))
 # Import only pure math module: no files/weights/source operations at import.
 spec=importlib.util.spec_from_file_location('independent_v90_math',R/'producer/spatial_input_math.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 oldpath=P.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py';old=pure(oldpath,['native_grid','patches'])
 cropold=P/'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py'
 new_ast=next(n for n in ast.parse((R/'producer/spatial_input_math.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='crop')
 old_ast=next(n for n in ast.parse(cropold.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='crop')
 assert ast.dump(new_ast,include_attributes=False)==ast.dump(old_ast,include_attributes=False)
 a=m.expected_affine([480,640]);assert a['canvas_shape']==[680,840]and a['input_shape']==[544,640]and a['p3_actual_hw']==[68,80]
 assert a['gain']==640/840 and np.array_equal(a['offset'],[0,13])
 # Every distinct sourceinput geometry; independent per-axis literal native9.
 one=byframe[frames[0]['frame_input_id']];qs=np.array([t['query_xy']for _,t in one],np.float64);ss=np.array([t['roi_side']for _,t in one],np.float64)
 grid,valid=m.native_grid_explicit(qs,ss,[480,640]);ind=np.empty_like(grid);iv=np.empty_like(valid)
 for n,(q,L)in enumerate(zip(qs,ss)):
  for iy in range(9):
   for ix in range(9):
    x=q[0]+((ix+.5)/9-.5)*L;y=q[1]+((iy+.5)/9-.5)*L;ind[n,iy,ix]=[x,y];iv[n,iy,ix]=0<=x<640 and 0<=y<480
 assert np.array_equal(grid,ind)and np.array_equal(valid,iv)
 network=(ind+100)*a['gain']+a['offset'];back=(network-a['offset'])/a['gain']-100;roundtrip=float(np.max(abs(back-ind)));assert roundtrip<1e-10
 # GeneratedFP16random plane only, actual128/256L controlled by oldspan fixture.
 rng=np.random.default_rng(99120261023);plane=rng.normal(size=(64,68,80)).astype(np.float16);pad=np.pad(plane,((0,0),(0,12),(0,0)))
 tested=[]
 for L in (128.,256.):
  q8=np.array([[64,64],[64+4*L,64],[100,100],[200,200],[300,300],[150,250],[400,150],[200,100]],np.float64)
  og,ov,ol=old['native_grid'](q8,[480,640]);assert ol==L
  expected,ev,el=old['patches']({'p3':torch.from_numpy(pad)[None],'added_border':100},q8,[480,640],a['gain'],a['offset'])
  actual,av,v9=m.sample_p3(plane,q8,np.repeat(L,8),[480,640],a['gain'],a['offset'],100)
  assert np.array_equal(actual,expected.numpy())and np.array_equal(av,ev)and np.array_equal(v9,ov)
  tested.append({'L':L,'old_V60_bitexact':True})
 # All129fixedinput geometries, masks/zeros/centres; no learned forward.
 x,anyv,v9=m.sample_p3(plane,qs,ss,[480,640],a['gain'],a['offset'],100)
 assert np.array_equal(v9,valid)and np.array_equal(x[:,64].astype(bool),valid)and np.array_equal(anyv,valid.reshape(129,-1).any(1))
 assert np.array_equal(grid[:,4,4],qs)and not x[:,:64][~np.broadcast_to(valid[:,None],x[:,:64].shape)].any()
 # independent scalar RGB interpolation/rounding: no sourceRGB decoded.
 rgb=rng.integers(0,256,(480,640,3),np.uint8);pixel_checks=0
 for q in ([0.,0.],[639.,479.],[320.,240.],[123.3,97.7]):
  for L in (128.,256.):
   pixels,pv=m.crop(rgb,q,L,96)
   for iy,ix in [(0,0),(47,47),(48,48),(95,95),(13,75)]:
    xx=q[0]+((ix+.5)/96-.5)*L;yy=q[1]+((iy+.5)/96-.5)*L;vx=-.5<=xx<639.5 and -.5<=yy<479.5
    bx,by=math.floor(xx),math.floor(yy);dx,dy=xx-bx,yy-by;acc=np.zeros(3,np.float64)
    for ox,oy,w in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
     if 0<=bx+ox<640 and 0<=by+oy<480:acc+=rgb[by+oy,bx+ox].astype(np.float64)*w
    ep=np.floor(np.clip(acc,0,255)+.5).astype(np.uint8)if vx else np.zeros(3,np.uint8)
    assert np.array_equal(pixels[:,iy,ix],ep)and bool(pv[0,iy,ix])==vx;pixel_checks+=1
 # Static dataflow/execution boundary review certificates, not hidden-process proof.
 src=(R/'producer/prepare_public_spatial_inputs.py').read_text();tree=ast.parse(src);run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run');text=ast.get_source_segment(src,run)
 assert text.index('authorize(entry_path)')<text.index('input_metadata()')<text.index('FrozenYoloFeatures(')<text.index('backbone.predict(')
 assert 'out.mkdir(exist_ok=False)'in text and 'source_predicts==100 and repeat_predicts==1'in text
 assert 'state_after==state_before'in text and "np.savez_compressed(pp,p3=p3"in text
 assert 'GT_ONLY'not in text and 'cue_coverage_gt_only'not in text and 'teacher'not in text
 generated={'completed':True,'PASS':True,'scope':bind(O/'SCOPE_BEFORE_GENERATED_REVIEW.json'),'code':bind(__file__),'runtime':{'sys_executable':sys.executable,'torch':str(torch.__version__),'numpy':np.__version__},'same_oldV60_explicitL128_256_generated_ROI_bitexact':tested,'all129_distinct_q_L_native9_10449cells_scalar_exact':True,'all129_native_masks_zero_features_and65channel_exact':True,'analytic_affine_roundtrip_max_px':roundtrip,'scalar_generated_RGB96_pixels':pixel_checks,'RGB_crop_AST_identical_V74':True,'source_pixel_P3_weightparameter_reads':0,'model_constructor_forward_fit_current_PnP':0}
 save(O/'INDEPENDENT_GENERATED_AND_METADATA_FIXTURES.json',generated)
 report={'schema':'V90_INDEPENDENT_PRE_EXTRACT_CODE_METADATA_ANALYTIC_REVIEW','completed':True,'PASS':True,'blockers':[],'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':bind(O/'SCOPE_BEFORE_GENERATED_REVIEW.json'),'protocol':bind(R/'INPUT_PRODUCER_PROTOCOL_BEFORE_PIXELS.json'),'entry_template':bind(R/'SOURCE_INPUT_ENTRYLOCK_TEMPLATE.json'),'producer_code':bind(R/'producer/prepare_public_spatial_inputs.py'),'math_code':bind(R/'producer/spatial_input_math.py'),'producer_generated':bind(R/'producer/GENERATED_SPATIAL_INPUT_TESTS.json'),'independent_generated':bind(O/'INDEPENDENT_GENERATED_AND_METADATA_FIXTURES.json'),'independent_code':bind(__file__),'metadata_code_bindings_verified':verified,'code_and_metadata_findings_KO':['기존 V89 100frame/12900tile의 정확한 schema/order/query/L/split/imageSHA가 유지되며 GT cue/ref/mask/depth/pose/native8/context42 입력이 없다.','fullP3 actual64×68×80 F16을 저장하고 ROI계산때만 RAM에서80×80 zero-pad하므로 기존V60 norm80 경로를 보존한다. nativevalid support가 actualplane내인지 assert한다.','reflect100은 frozen wrapper에서1회, gain640/840와 integeroffset[0,13]을 actualcapturedcanvas/inputshape와 대조한다.','ROI는 explicitL128/256 nativefloat64→gridF32, CPUF32bilinear/zeros/alignfalse→nativevalid곱/65channel→F16이다. RGB96은 V74AST그대로이며 q가47.5index사이에놓이는 phase를 보존한다.','승인entry/SHA/PASSguard가 sourcemetadata/modelconstruct/pixels보다 앞선다. output.mkdir(exist_ok=False)와 exclusiveJSONwrite가 기존/실패출력을 덮어쓰는 재시작을 막는다. code에는 실패시 삭제/재시도 loop가 없다.','blank warmup1회 이후stateSHA고정, unique source100+firstrepeat1회=sourcepredict101회가 명시된다. blank포함 직접predict총102회이며 internallazywarmupforward는별도로그된다. sourcefirst P3/affine repeatbitexact 및 endstate/checkpointSHA불변을 실제run에서검사한다.','disk500000000byte/GPU10GB guards가 extraction중·finalmanifest작성후 적용된다. 마지막seal 자체bytes는 guard이후추가되므로 최종실제diskcount를 outputaudit에서별도확인한다. 이는 현재예상크기의blocker는아니나 budgetPASS는actualoutput후확인해야한다.'],'limits_KO':['코드/메타데이터/생성수학만검토했다. 실제sourceRGB/realP3/checkpointparameter/modelconstructor/forward를읽거나실행하지않았다.','실제생성물의전체SHA/수치및메모리·디스크budgetPASS는 추출이끝난뒤 별도saved-onlyaudit에서확인해야한다.','scope는 source input 준비만이며 semanticgold/native8/학습/currentadmission이아니다.'],'expected_actual_execution':{'unique_RGBframes':100,'unique_source_predicts':100,'first_source_repeat_predicts':1,'blank_warmup_predicts':1,'explicit_total_predict_calls':102,'internal_lazy_warmup_separately_logged':True,'old8query_ROIs_reused':False,'source_fullP3_files':100,'tile_inputs':12900},'no_actual_source_pixels_model_weightsparams_forward_fit_current_PnP':0,'training_admission':False}
 save(O/'REVIEW.json',report)
 save(O/'SEAL.json',{'completed':True,'PASS':True,'review':bind(O/'REVIEW.json'),'independent_generated':bind(O/'INDEPENDENT_GENERATED_AND_METADATA_FIXTURES.json'),'scope':bind(O/'SCOPE_BEFORE_GENERATED_REVIEW.json'),'protocol':bind(R/'INPUT_PRODUCER_PROTOCOL_BEFORE_PIXELS.json'),'producer':bind(R/'producer/prepare_public_spatial_inputs.py'),'math':bind(R/'producer/spatial_input_math.py'),'source_actual_forward_fit_current_PnP':0})
 print(json.dumps({'review':bind(O/'REVIEW.json'),'seal':bind(O/'SEAL.json'),'generated':bind(O/'INDEPENDENT_GENERATED_AND_METADATA_FIXTURES.json')},indent=2))
if __name__=='__main__':main()
