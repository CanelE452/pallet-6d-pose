"""One frozen detector/N3 ROI; no pose targets in generation, separate scoring."""
import pathlib,sys,json,gzip,ast,time,os,math,subprocess,argparse
from collections import Counter
import numpy as np
import cv2
import torch
R=pathlib.Path(__file__).resolve().parent;T=R.parent
SRC=pathlib.Path('${SOURCE_REPO}');REPO=pathlib.Path('${MODEL_REPO}');DATA=pathlib.Path('${DATA_ROOT}');B=DATA/'pallet-boundary-action-repair-20261018-v1'
sys.path[:0]=[str(T/'mean_gap_audit_20261020_v1'),str(T/'minimal_improvement_20261020_v5'),str(SRC)]
from analyze import read,rows,index,write,csvwrite,sha,metric
from kernel_probe import stats
SAN=DATA/'pallet-statewise-end-to-end-recovery-v1/synthetic_v1/UPSTREAM_INPUTS_NO_GT.jsonl.gz'
BOX=DATA/'pallet-risk-information-support-20261018-v1/v_box_inputs_v1/V_BOX_INPUTS.jsonl.gz'
def save(p,z):
 with gzip.open(p,'xt')as h:
  for a in z:h.write(json.dumps(a,allow_nan=False)+'\n')
def bound():
 p=read(R/'PROTOCOL_SCHEMA_CORRECTION.json')
 for k,v in p['bindings'].items():assert sha(v['path'])==v['sha256'],k
 return p
def firewall():
 def hook(e,a):
  if e=='open'and isinstance(a[0],(str,bytes))and pathlib.Path(os.fsdecode(a[0])).name in ['SYNTH_GT_LABELS.jsonl.gz','FIXED_REFERENCES.json','DERIVED_C2_ACTION_TARGETS.jsonl.gz','C2_TARGETS.jsonl.gz']:raise PermissionError('No answers during generation')
 sys.addaudithook(hook)
def gpu():
 p=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True);other=[int(x)for x in p.splitlines()if x.strip().isdigit()and int(x)!=os.getpid()];assert not other,('FOREIGN_GPU_COMPUTE_PRESERVE_STOP',other)
 return subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader'],text=True).strip()
def functions():
 from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses
 ns=dict(np=np,cv2=cv2,Counter=Counter,METHODS=('SUBPIX',),select_pnp_hypotheses=select_pnp_hypotheses,_configuration=lambda:dict(SUBPIX=dict(minimum_image_extent=15)))
 for path,names in [('scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py',['cuboid','solve']),('scripts/research/pallet_dim_conditioned_p_v1/pose.py',['rotations','infer']),('scripts/research/pallet_training_free_compare_20261007_v1/methods.py',['_inputs','correct','cap_points'])]:
  nodes=[n for n in ast.parse((SRC/path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names];assert len(nodes)==len(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),ns)
 return ns
def init():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1);assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.25);before=gpu()
 sys.path[:0]=[str(SRC/'scripts/research/pallet_dim_conditioned_p_v1'),str(SRC/'scripts/research/pallet_final_ml_contribution_test_v1')]
 import dcp_env as E
 E.RAW=REPO/'data/pallet/results/pallet_dim_conditioned_p_v1';E.DOC=REPO/'_docs/experiments/pallet_dim_conditioned_p_v1'
 from inference import load_head,predict_captured
 head,_=load_head('N3_DIM_SYM',1);features=E.old('features');extractor=features.FrozenYoloFeatures(weights=REPO/E.R0.relative_to(SRC),device='cuda')
 cfg=read(E.DOC/'CALIBRATION_AND_SELECTION.json');return extractor,head,predict_captured,features,read(E.DOC/'DIM_NORMALIZATION_LOCK.json'),cfg['temperatures']['N3_DIM_SYM_seed1']['temperature'],cfg['rule'],before
def infer_one(u,image,origin,tools,fn):
 ex,head,predict,features,norm,temp,rule,_=tools;t=time.perf_counter();cap=ex.predict(image);torch.cuda.synchronize();rawsec=time.perf_counter()-t;xyz=np.asarray(u['known_physical_dimensions_xyz_m']);t=time.perf_counter();z,_=predict(head,'N3_DIM_SYM',cap,xyz[[0,2,1]],2,temp,rule,u['raw_hw'],norm);torch.cuda.synchronize();headsec=time.perf_counter()-t;idx=z['selected_index']
 if idx is None:return dict(pose=dict(available=False),q0=None,qn=None,q=None,raw_seconds=rawsec,head_seconds=headsec,detections=0)
 q0=np.asarray(cap['candidates'][idx]['keypoints_xy'])+origin;qn=np.asarray(z['candidates'][idx]['keypoints_xy'])+origin;support=np.isfinite(q0).all(-1)&~np.all(q0==-1,axis=1);gray=cv2.cvtColor(cv2.imread(u['image']),cv2.COLOR_BGR2GRAY);native,diag=fn['correct'](gray,qn,support,'SUBPIX');h,w=u['raw_hw'];q=fn['cap_points'](q0,native,w,h,support);pose=fn['infer'](q,np.asarray(u['K']),xyz,source=True);gain,offset=features.canvas_affine(cap['canvas_shape'],cap['input_shape']);ch,cw=cap['canvas_shape'];ih,iw=cap['input_shape'];rh,rw=round(ch*gain),round(cw*gain);C=np.eye(3);C[:2,2]=-origin;A=np.diag([gain,gain,1.]);A[:2,2]=gain*(100-origin)+offset
 # Label-coordinate affine and raster half-pixel affine are deliberately distinct.
 raster=np.diag([rw/cw,rh/ch,1.]);raster[:2,2]=np.array([rw/cw,rh/ch])*(100-origin+.5)-.5+offset
 points=np.array([[0,0,1],[w-1,h-1,1],[w/2,h/2,1]],float);rt=points@A.T@np.linalg.inv(A).T;err=float(np.max(abs(rt-points)));assert err<1e-9
 K=np.asarray(u['K']);test=np.array([[.3,.2,3.],[1.,-.4,5.]])@K.T;expected=(test[:,:2]/test[:,2:])-origin;v=np.array([[.3,.2,3.],[1.,-.4,5.]])@(C@K).T;ke=float(np.max(abs(v[:,:2]/v[:,2:]-expected)));assert ke<1e-9
 return dict(pose=pose,q0=q0.tolist(),qn=qn.tolist(),q=q.tolist(),detector_box_original=(np.asarray(cap['candidates'][idx]['box_xyxy'])+np.tile(origin,2)).tolist(),selected_score=cap['candidates'][idx]['score'],detections=len(cap['candidates']),raw_seconds=rawsec,head_seconds=headsec,subpix=diag,transform=dict(origin_xy=origin.tolist(),canvas_hw=list(cap['canvas_shape']),input_hw=list(cap['input_shape']),resized_hw=[rh,rw],label_A=A.tolist(),raster_half_pixel_A=raster.tolist(),nominal_gain=gain,label_integer_pad_xy=offset.tolist(),decoder_point_half_pad_xy=[(iw-rw)/2,(ih-rh)/2],original_K=K.tolist(),crop_K=(C@K).tolist(),affine_roundtrip_max_px=err,K_projection_roundtrip_max_px=ke,returned_keypoints_already_inverse_letterboxed=True,clipped_decoder_values_not_invertible=True,manual_halfpixel_correction=False))
def pilot():
 bound();firewall();out=R/'PILOT_RUNTIME_V3';out.mkdir(exist_ok=False);start=time.monotonic();tools=init();fn=functions();train=[u for u in rows(SAN)if u['split']=='train'];fs=[train[i]for i in np.linspace(0,len(train)-1,8,dtype=int)];up=index(rows(B/'d1_upstream_v1/SYNTHETIC1003_UPSTREAM.jsonl.gz'));rr=[];details=[]
 for u in fs:
  assert sha(u['image'])==u['image_sha256'];image=cv2.imread(u['image']);p=infer_one(u,image,np.zeros(2),tools,fn);old=up[u['id']];r=dict(id=u['id'],raw_max_px=float(np.max(abs(np.asarray(p['q0'])-u['raw_YOLO_points']))),N3_max_px=float(np.max(abs(np.asarray(p['qn'])-u['N3_points']))),SubPix_max_px=float(np.max(abs(np.asarray(p['q'])-old['initial_points']))));r['pose_T_delta_cm']=float(100*np.linalg.norm(np.asarray(p['pose']['centroid'])-old['initial_pose']['centroid']));rel=np.asarray(p['pose']['R_physical']).T@old['initial_pose']['R_physical'];r['pose_R_delta_deg']=float(np.degrees(np.arccos(np.clip((np.trace(rel)-1)/2,-1,1))));r['pass']=r['raw_max_px']<.001 and r['N3_max_px']<.002 and r['SubPix_max_px']<.002 and r['pose_T_delta_cm']<.01 and r['pose_R_delta_deg']<.01;rr.append(r);details.append(dict(id=u['id'],**p));print(json.dumps(r),flush=True)
 save(out/'REPLAY.jsonl.gz',details);csvwrite(out/'PARITY8.csv',rr);write(out/'SEAL.json',dict(status='PASS'if all(r['pass']for r in rr)else'BLOCKED_FULL_REPLAY_PARITY',frames=8,seconds=time.monotonic()-start,full_detector_N3_forward_pairs=8,new_PnP_pipelines=8,new_training=0,new_SAM=0,GPU_start=tools[-1],max_allocated=torch.cuda.max_memory_allocated(),max_reserved=torch.cuda.max_memory_reserved()));print(read(out/'SEAL.json'))
def generate():
 bound();assert read(R/'PILOT_RUNTIME_V3/SEAL.json')['status']=='PASS';firewall();out=R/'SYNTHETIC_VAL';out.mkdir(exist_ok=False);start=time.monotonic();tools=init();fn=functions();fs=[u for u in rows(SAN)if u['split']=='validation'];box=index(rows(BOX));up=index(rows(B/'d1_upstream_v1/SYNTHETIC1003_UPSTREAM.jsonl.gz'));ps=[];tt=[]
 for i,u in enumerate(fs):
  if i%25==0:gpu()
  tick=time.monotonic();assert sha(u['image'])==u['image_sha256'];image=cv2.imread(u['image']);h,w=u['raw_hw'];b=np.asarray(box[u['id']]['detector_box_xyxy_raw'],float);center=(b[:2]+b[2:])/2;radius=(b[2:]-b[:2])*.75;x0,y0=np.maximum(np.floor(center-radius),0).astype(int);x1,y1=np.minimum(np.ceil(center+radius),[w,h]).astype(int);assert x1>x0 and y1>y0;p=infer_one(u,image[y0:y1,x0:x1],np.array([x0,y0]),tools,fn);ps.append(dict(id=u['id'],image_sha256=u['image_sha256'],split=u['split'],recording='SYNTHETIC_NO_SESSION_FEATURE',old_detector_box=b.tolist(),crop_xyxy=[int(x0),int(y0),int(x1),int(y1)],N0=up[u['id']]['initial_pose'],ROI=p));tt.append(dict(id=u['id'],seconds=time.monotonic()-tick,detector_seconds=p['raw_seconds'],head_seconds=p['head_seconds'],crop_width=int(x1-x0),crop_height=int(y1-y0),old_net_box_width=float((b[2]-b[0])*640/max(h+200,w+200)),crop_net_box_width=float((b[2]-b[0])*640/max(y1-y0+200,x1-x0+200)),proposal_available=p['pose']['available']))
  if(i+1)%25==0:print(json.dumps(dict(stage='FROZEN_ROI',done=i+1,total=len(fs),seconds=time.monotonic()-start)),flush=True)
  assert time.monotonic()-start<600,'600second minimal comparison cap'
 save(out/'PREDICTIONS.jsonl.gz',ps);csvwrite(out/'TIMING.csv',tt);write(out/'SEAL.json',dict(status='SEALED_BEFORE_SCORING',rows=len(ps),prediction_sha256=sha(out/'PREDICTIONS.jsonl.gz'),seconds=time.monotonic()-start,new_detector_N3_forward_pairs=len(ps),new_PnP_pipelines=sum(p['ROI']['q']is not None for p in ps),new_training=0,new_SAM=0,new_H1=0,GPU_start=tools[-1],max_allocated=torch.cuda.max_memory_allocated(),max_reserved=torch.cuda.max_memory_reserved(),timing_scope='source RGB hash/decode; crop; frozen YOLO+N3; original full-gray SubPix; same prediction-only W/D selector SQPnP/LM; serialization. excludes cached baseline and SAM/fullOCPR'));print(read(out/'SEAL.json'))
def gate(a,b):
 why=[]
 if a['successes']<b['successes']:why.append('joint_success_decreased')
 for k in ['T_mean','R_mean','T_P90','T_P95','R_P90','R_P95','T_max','R_max','no_pose','T_gt50','T_gt100','R_gt45']:
  if a[k]>b[k]+1e-7:why.append(k+'_increased')
 if not(a['T_mean']<b['T_mean']-1e-7 or a['R_mean']<b['R_mean']-1e-7):why.append('no_mean_improvement')
 return why
def evaluate():
 bound();out=R/'SYNTHETIC_VAL';assert sha(out/'PREDICTIONS.jsonl.gz')==read(out/'SEAL.json')['prediction_sha256'];ps=rows(out/'PREDICTIONS.jsonl.gz');gt=index(rows(DATA/'pallet-statewise-end-to-end-recovery-v1/synthetic_v1/SYNTH_GT_LABELS.jsonl.gz'));lab=index(rows(T/'minimal_improvement_20261020_v5/H1_DEPLOYMENT_C2_TARGET_V1/DERIVED_C2_ACTION_TARGETS.jsonl.gz'));groups={a:[]for a in ['N0','ROI']};paired=[];coverage=[]
 for p in ps:
  g=gt[p['id']];ref=dict(R=g['R_fixed'],t=g['t'],order=2);r=dict(id=p['id']);ms={}
  for arm in groups:
   pose=p['N0']if arm=='N0'else p['ROI']['pose'];m=metric(pose,ref);groups[arm].append(m);ms[arm]=m;r.update({arm+'_'+k:v for k,v in m.items()})
  paired.append(r);good=any(a['success_5cm5deg']for a in lab[p['id']]['action_metrics']);coverage.append(dict(id=p['id'],original_bank_good=good,original_plus_ROI_good=good or ms['ROI']['success'],new_good=not good and ms['ROI']['success']))
 st={a:stats(z)for a,z in groups.items()};reasons=gate(st['ROI'],st['N0']);result=dict(methods=st,gate_passed=not reasons,gate_reasons=reasons,recover=sum(not a['success']and b['success']for a,b in zip(groups['N0'],groups['ROI'])),damage=sum(a['success']and not b['success']for a,b in zip(groups['N0'],groups['ROI'])),coverage_diagnostic={k:sum(r[k]for r in coverage)for k in ['original_bank_good','original_plus_ROI_good','new_good']},oracle_is_not_selector=True,denominator=235,exposed_development=True,physical_owner_labels=False,new_independent_data=0,policy='always ROI; no GT-based union choice or fallback; unavailable ROI counts failure',expansion_allowed=not reasons);csvwrite(out/'PAIRED_ROWS.csv',paired);csvwrite(out/'BANK_COVERAGE.csv',coverage);write(out/'EVALUATION.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','generate','evaluate']);a=p.parse_args();{'pilot':pilot,'generate':generate,'evaluate':evaluate}[a.stage]()
