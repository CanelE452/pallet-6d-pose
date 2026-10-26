"""Freeze four learned finals, final SYN-only rule readout, then config freeze."""
import io,json,sys,types
from pathlib import Path
import numpy as np
import torch
H=Path(__file__).absolute().parent

def load(path,name):
 m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
 exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m


def summarize_location(output,targets,rule,spatial):
 q=torch.from_numpy(targets['q0']);r=torch.from_numpy(targets['reference_xy']);side=torch.from_numpy(targets['crop_side']);eligible=targets['LOC_valid']
 accepted=spatial.acceptance(output,rule)[0] if rule is not None else torch.zeros(len(q),dtype=torch.bool)
 final=spatial.image_points(output,q,side,accepted)
 before=np.linalg.norm(q.numpy()-r.numpy(),axis=1);after=np.linalg.norm(final.numpy()-r.numpy(),axis=1)
 by={}
 for native in [None,*range(8)]:
  mask=eligible if native is None else eligible&(targets['native_slot']==native)
  v=before[mask];a=after[mask]
  by['ALL' if native is None else str(native)]={'eligible':int(mask.sum()),'accepted':int((accepted.numpy()&mask).sum()),
   'q0_median_px':float(np.median(v)) if len(v) else None,'q0_p90_px':float(np.quantile(v,.9)) if len(v) else None,
   'final_median_px':float(np.median(a)) if len(a) else None,'q0_within3':int((v<=3).sum()),'final_within3':int((a<=3).sum()),
   'recovery_gt3_to_le3':int(((v>3)&(a<=3)).sum()),'damage_le3_to_gt3':int(((v<=3)&(a>3)).sum())}
 return by


def run():
 p=load(H/'probe_runtime.py','v263_final_io')
 names={'LOC_ONLY_SIMPLE2D':'LOC_simple2D_v1','LOC_ONLY_INITIAL_GEOMETRY':'LOC_initial_geometry_v1','VIS_ONLY_SELECTED':'VIS_selected_v1','JOINT_SELECTED':'JOINT_selected_v1'}
 bindings={k:p.binding(H/v/'COMPLETE.json') for k,v in names.items()};arms={k:json.loads(p.raw(v)) for k,v in bindings.items()}
 assert all(v['result']['optimizer_updates']==1000 and v['new_DEV_head_forwards']==0 for v in arms.values())
 assert len({v['result']['initial_state_sha256'] for v in arms.values()})==1 and len({v['result']['draws_sha256'] for v in arms.values()})==1
 cpbar=H/'ALL4_CP_LOGS_FROZEN_BEFORE_FINAL_SYN_DEV.json';assert not cpbar.exists()
 cpbar.write_text(json.dumps({'completed':True,'PASS':True,'arms':bindings,'checkpoints':{k:v['outputs']['FINAL1000.pt'] for k,v in arms.items()},'optimizer_updates_each':1000,'shared_initial_and_draws':True,'current_or_held_reads':0},indent=2,sort_keys=True)+'\n')
 s=p.module(arms['JOINT_SELECTED']['head_code'],'v263_final_spatial');t=p.module(arms['JOINT_SELECTED']['training_code'],'v263_final_train')
 select=load(H/'select_acceptance.py','v263_final_rule_selector');crop_eval=load(H/'select_syn_crop.py','v263_fixed_eval_helper')
 rules=json.loads((H/'ACCEPTANCE_RULES_BEFORE_FIT.json').read_text())['rules']
 selection=json.loads((H/'SYN_DEV_CROP_AND_RULE_SELECTION.json').read_text())
 joint=crop_eval.evaluate_arm(p,bindings['JOINT_SELECTED'],s,t,select,rules,H/'SYN_DEV_JOINT_SELECTED_OUTPUTS.npz')
 vis=crop_eval.evaluate_arm(p,bindings['VIS_ONLY_SELECTED'],s,t,select,rules,H/'SYN_DEV_VIS_SELECTED_OUTPUTS.npz')
 # Untrained VIS-only location readout is never used or reported as a result.
 vis={'arm_complete':vis['arm_complete'],'saved_DEV_outputs':vis['saved_DEV_outputs'],'LOC_trained':False,'selected_rule':'ALL_ABSTAIN'}
 chosen_loc='LOC_ONLY_INITIAL_GEOMETRY' if selection['selected_crop']=='initial_geometry' else 'LOC_ONLY_SIMPLE2D'
 selected_rules={chosen_loc:selection['selected_LOC_rule'],'VIS_ONLY_SELECTED':'ALL_ABSTAIN','JOINT_SELECTED':joint['selected_rule']}
 reports={};
 for name,folder in names.items():
  if name.startswith('LOC_'):
   crop='simple2D' if name=='LOC_ONLY_SIMPLE2D' else 'initial_geometry';output_file=H/('SYN_DEV_'+crop+'_OUTPUTS.npz');rule_name=selection['evaluations'][crop]['selected_rule']
  else:output_file=H/('SYN_DEV_'+('JOINT_SELECTED' if name=='JOINT_SELECTED' else 'VIS_SELECTED')+'_OUTPUTS.npz');rule_name=selected_rules[name]
  with np.load(output_file,allow_pickle=False) as z:out={k:torch.from_numpy(np.array(z[k],copy=True)) for k in z.files if k!='source_index'};indices=np.array(z['source_index'],copy=True)
  with np.load(io.BytesIO(p.raw(arms[name]['outputs']['TARGETS_NONPIXEL.npz'])),allow_pickle=False) as z:targets={k:np.array(z[k],copy=True)[indices] for k in z.files}
  result={'DEV_rows':len(indices),'LOC_trained':name!='VIS_ONLY_SELECTED','VIS_trained':not name.startswith('LOC_')}
  if result['LOC_trained']:result['LOC']=summarize_location(out,targets,None if rule_name=='ALL_ABSTAIN' else rules[rule_name],s)
  if result['VIS_trained']:
   pred=s.semantic_visibility(out).numpy();gt=targets['visibility_target'];cm=[[int(((gt==a)&(pred==b)).sum()) for b in [0,1,-1]] for a in [0,1]]
   result['semantic_VIS']={'CM_rows_D_E_cols_D_E_U':cm,'known':int((gt>=0).sum()),'masked_SU':int((gt<0).sum())}
  reports[name]=result
 summary={'completed':True,'PASS':True,'freeze_CP_logs':p.binding(cpbar),'arms':bindings,'selected_crop':selection['selected_crop'],
  'current_primary_configurations':['ORIGINAL',chosen_loc,'VIS_ONLY_SELECTED','JOINT_SELECTED'],
  'selected_rules':selected_rules,'JOINT_rule_selection':joint,'VIS_output':vis,'SYN_DEV_reports':reports,
  'current_or_held_reads':0,'source_geometry_cue_is_not_photometric_or_real_native_gold':True}
 summary_path=H/'FOUR_FIT_COMPLETE.json';assert not summary_path.exists();summary_path.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 freeze={'completed':True,'PASS':True,'summary':p.binding(summary_path),'arms':bindings,'checkpoints':{k:v['outputs']['FINAL1000.pt'] for k,v in arms.items()},
  'selected_crop':selection['selected_crop'],'current_primary_configurations':summary['current_primary_configurations'],'selected_rules':selected_rules,
  'rule_definition':p.binding(H/'ACCEPTANCE_RULES_BEFORE_FIT.json'),'head_code':arms['JOINT_SELECTED']['head_code'],'training_code':arms['JOINT_SELECTED']['training_code'],
  'all_learned_settings':4,'optimizer_updates_each':1000,'current_inputs_or_GT_used_for_selection':0,'no_further_recipe_or_threshold_change_after_current':True}
 path=H/'ALL4_FINAL_FREEZE.json';assert not path.exists();path.write_text(json.dumps(freeze,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'completed':True,'PASS':True,'freeze':p.binding(path),'summary':p.binding(summary_path),'selected_rules':selected_rules,'SYN_DEV_reports':reports}),flush=True)

if __name__=='__main__':run()
