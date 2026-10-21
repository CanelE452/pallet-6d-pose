"""Sealed face ablation predictions and separate post-seal scoring."""
import pathlib,importlib.util,argparse,time,json,csv
import numpy as np
import torch
R=pathlib.Path(__file__).resolve().parent;T=R.parent
from encoder import F,G,OLD,bound as encoder_bound
F.R=R
def folder(pop,seed,bg):return R/((f'BG_{bg}_'if bg else'')+f'{pop}_s{seed}')
def bound():
 encoder_bound();p=G.read(R/'CONTROLLER_PROTOCOL.json')
 for k,v in p['bindings'].items():assert G.sha(v['path'])==v['sha256'],k
def predict(pop,seed,bg=None):
 G.setup();bound();G.firewall();syn=pop=='SYNTHETIC_VAL';assert syn or bg is None
 if not syn:assert G.read(folder('SYNTHETIC_VAL',seed,None)/'EVALUATION.json')['main_gate_passed']and G.read(R/'CROSS_BACKGROUND_DECISION.json')['gate_passed']
 out=folder(pop,seed,bg);out.mkdir(exist_ok=False);start=time.monotonic();models={};arms=['GEOMETRY','FUSION']+([]if bg else['APPEARANCE','HOG'])
 for arm in arms:
  cp=(OLD/f'FUSION_s{seed}/best.pt')if arm=='HOG'else R/(arm+(f'_BG_{bg}'if bg else'')+f'_s{seed}')/'best.pt';c=torch.load(cp,weights_only=False,map_location='cpu');m=F.model(arm,seed);m.load_state_dict(c['model_state_dict']);m.eval().requires_grad_(False);models[arm]=(m,c)
 for arm,path in [('BASELINE',G.V5/f's{seed}/best.pt'),('UNION',F.V24/f's{seed}/best.pt')]:
  c=torch.load(path,weights_only=False,map_location='cpu');m=G.M.make_model('MLP');m.load_state_dict(c['model_state_dict']);m.eval().requires_grad_(False);models[arm]=(m,c)
 fs=G.rows((F.U if syn else F.V24/'CAL46_v3')/'HEAD_FEATURES.jsonl.gz');fs=[f for f in fs if f['split']=='validation']if syn else fs;bgmap={z['id']:z['background_asset']for z in csv.DictReader((R/'BACKGROUND_GROUPS.csv').open())};fs=[f for f in fs if bgmap[f['id']]!=bg]if bg else fs;old=G.index(G.rows(G.B/('d1_synthetic_candidates_v1'if syn else'd1_real_candidates_v1')/'HEAD_FEATURES.jsonl.gz'));extra=G.index(G.rows(R/('SYNTHETIC'if syn else'CAL46')/'FEATURES.jsonl.gz'));oldHOG=G.index(G.rows(OLD/('SYNTHETIC'if syn else'CAL46')/'FEATURES.jsonl.gz'));ps=[];timings=[]
 with torch.inference_mode():
  for f in fs:
   tick=time.monotonic();p=dict(id=f['id'],partition=f['partition'],recording=f['cluster_id'],methods={},bank_ids=[],background_evaluation_subset=bgmap.get(f['id'])if bg else None);m,c=models['BASELINE'];x,es=G.A.inputs(old[f['id']],c['normalizer'],'H1');k=int(m(torch.from_numpy(x)).flatten().argmax());inc=dict(candidate_id=es[k]['candidate_id'],pose=es[k]['actual_pose']);p['methods']['BASELINE']=inc;hasRGB=any(any(e['validity'][204:396])for e in extra[f['id']]['actions'])
   for arm in ['UNION']+arms:
    m,c=models[arm]
    if arm=='UNION':x,es=G.A.inputs(f,c['normalizer'],'H1')
    else:x,es=F.input(f,c['normalizer'],oldHOG[f['id']]if arm=='HOG'else extra[f['id']],c['extra_normalizer'],arm)
    k=int(m(torch.from_numpy(x)).flatten().argmax());p['methods'][arm]=dict(candidate_id=es[k]['candidate_id'],pose=es[k]['actual_pose'])if hasRGB or arm=='UNION'else inc;p['bank_ids']=[e['candidate_id']for e in es]
   p['raw_RGB_evidence_present']=hasRGB;ps.append(p);timings.append(dict(id=f['id'],seconds=time.monotonic()-tick,candidate_n=len(es),raw_RGB_evidence_present=hasRGB))
 G.save(out/'PREDICTIONS.jsonl.gz',ps);G.csvwrite(out/'TIMING.csv',timings);G.write(out/'SEAL.json',dict(status='SEALED_BEFORE_EVALUATION',prediction_sha256=G.sha(out/'PREDICTIONS.jsonl.gz'),rows=len(ps),seconds=time.monotonic()-start,reference_inputs=0,background_used_only_for_diagnostic_subset=bool(bg),new_PnP=0,new_RGB_extraction=0));print(dict(stage='PREDICTED_FACE',pop=pop,bg=bg,n=len(ps)))
def evaluate(pop,seed,bg=None):
 bound();out=folder(pop,seed,bg);assert G.sha(out/'PREDICTIONS.jsonl.gz')==G.read(out/'SEAL.json')['prediction_sha256'];ps=G.rows(out/'PREDICTIONS.jsonl.gz');syn=pop=='SYNTHETIC_VAL';labels=G.index(G.rows(F.U/'C2_TARGETS.jsonl.gz'))if syn else None;refs=G.read(G.B.parent/'pallet-statewise-correspondence-diagnosis-20261016-v1/score_v2/FIXED_REFERENCES.json')['references']if not syn else None;groups={k:[]for k in ps[0]['methods']};paired=[]
 for p in ps:
  by={a['candidate_id']:a for a in labels[p['id']]['action_metrics']}if syn else None;r=dict(id=p['id'],recording=p['recording'],candidate_n=len(p['bank_ids']),raw_RGB_evidence_present=p['raw_RGB_evidence_present'])
  for arm,pred in p['methods'].items():
   if syn:y=by[pred['candidate_id']];m=dict(pose_available=True,translation_cm=y['translation_cm'],rotation_deg=y['rotation_deg'],success=y['success_5cm5deg'])
   else:m=G.metric(pred['pose'],refs[p['partition']][p['id']])
   groups[arm].append(m);r[arm+'_id']=pred['candidate_id'];r.update({arm+'_'+k:v for k,v in m.items()})
  paired.append(r)
 st={k:G.stats(v)for k,v in groups.items()};comparators=['GEOMETRY']if bg else['BASELINE','UNION','GEOMETRY'];reasons={k:G.gate(st['FUSION'],st[k])for k in comparators};rd={k:dict(recover=sum(not a['success']and b['success']for a,b in zip(groups[k],groups['FUSION'])),damage=sum(a['success']and not b['success']for a,b in zip(groups[k],groups['FUSION'])))for k in comparators};result=dict(population=pop,seed=seed,trained_background=bg,evaluation_background='OTHER_BACKGROUND_ONLY'if bg else'FULL_SPLIT',methods=st,main_gate_passed=not any(reasons.values()),gate_reasons=reasons,recover_damage=rd,missing_evidence_fallback_frames=sum(not p['raw_RGB_evidence_present']for p in ps),exposed_development=True,independent_new_real=0);G.csvwrite(out/'PAIRED_ROWS.csv',paired);G.write(out/'EVALUATION.json',result);print(json.dumps(result,indent=2))
def parity():
 bound();G.firewall();fs=G.rows(F.U/'HEAD_FEATURES.jsonl.gz');old=G.index(G.rows(OLD/'SYNTHETIC/FEATURES.jsonl.gz'));new=G.index(G.rows(R/'SYNTHETIC/FEATURES.jsonl.gz'));on=G.read(OLD/'SYNTHETIC/NORMALIZER.json');nn=G.read(R/'SYNTHETIC/NORMALIZER.json');norm=G.read(F.U/'NORMALIZER.json');count=0;availabilitychanges=[]
 for f in fs:
  ox,oe=F.input(f,norm,old[f['id']],on,'GEOMETRY');nx,ne=F.input(f,norm,new[f['id']],nn,'GEOMETRY');assert np.array_equal(ox,nx)and[e['candidate_id']for e in oe]==[e['candidate_id']for e in ne];count+=len(ne)
  op=any(any(a['validity'][204:])for a in old[f['id']]['actions']);npresent=any(any(a['validity'][204:])for a in new[f['id']]['actions'])
  if op!=npresent:availabilitychanges.append(f['id'])
 G.write(R/'CONTROL_PARITY_ALL1003.json',dict(status='PASS',frames=len(fs),actions=count,all_normalized_GEOMETRY_inputs_bitwise_identical=True,availability_policy_change_frames=availabilitychanges,new_GEOMETRY_fit=0,rawRGBcolumns_zeroed_in_control=True));print(dict(stage='ALL_CONTROL_PARITY_PASS',frames=len(fs),actions=count,availabilitychanges=len(availabilitychanges)))
F.bound=bound
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['parity','train','predict','evaluate']);p.add_argument('--population',default='SYNTHETIC_VAL');p.add_argument('--arm',choices=['APPEARANCE','FUSION','GEOMETRY']);p.add_argument('--seed',type=int,default=20261018);p.add_argument('--background');a=p.parse_args();parity()if a.stage=='parity'else F.train(a.arm,a.seed,a.background)if a.stage=='train'else(predict if a.stage=='predict'else evaluate)(a.population,a.seed,a.background)
