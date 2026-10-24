#!/usr/bin/env python3
"""Actual two-stage source fixture: saved2.1 features -> frozen2.5 CPU head."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np,torch
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
if __name__=='__main__':
 destination=OUT/'HELPER_AUDIT_TORCH25_SUPPLEMENT.json';assert not destination.exists();assert torch.__version__.startswith('2.5.1'),torch.__version__
 bound_report=OUT/'HELPER_AUDIT_REPORT.json';bound_before=sha(bound_report);base=json.loads(bound_report.read_text());assert base['PASS']
 actual_path=OUT/'HELPER_PRODUCED_FEATURES.npz';expected_path=OUT/'EXPECTED_FEATURES.npz';produced=load(actual_path);expected=load(expected_path);fixtures=json.loads((OUT/'FIXTURE_INPUTS.json').read_text())['inputs']
 helper_path=ROOT/'inference/infer_current_frozen.py';helper=module(helper_path,'torch25_public_fixture_helper');headpath=ROOT/'runs/MIXED_GLOBAL_UNWEIGHTED/code/query_global_amodal_head.py';head=module(headpath,'torch25_public_fixture_head');checkpoint=ROOT/'runs/MIXED_GLOBAL_UNWEIGHTED/BEST_GLOBAL_AMODAL_HEAD.pt'
 if not checkpoint.exists():
  files=list(checkpoint.parent.glob('BEST*.pt'));assert len(files)==1;checkpoint=files[0]
 torch.set_num_threads(1);model,payload=head.load_model(checkpoint,'cpu');model.eval().requires_grad_(False);before={k:v.detach().clone()for k,v in model.state_dict().items()};result=[]
 keys=['rgb_native','box_grid','sam','global_rgb','global_sam','global_prompt','global_index64','global_index8','global_query_valid']
 for r in fixtures:
  i=r['fixture_index'];pre=f'f{i:03d}_';f={k:produced[pre+k]for k in keys};e={k:expected[pre+k]for k in keys}
  for k in ['global_rgb','global_sam','global_prompt']:e[k]=e[k][None]
  assert all(np.array_equal(f[k],e[k])for k in keys)
  x=expected[pre+'x8'];context=produced[pre+'context'];a=helper.forward_frame(model,x,context,f,'cpu');b=helper.forward_frame(model,x,expected[pre+'context'],e,'cpu');previous=produced[pre+'head_logits'];aa=helper.probabilities(a);bb=helper.probabilities(b);pp=helper.probabilities(previous)
  summary=dict(fixture_index=i,id=r['id'],source=r['source'],split=r['split'],same_Torch25_actual_vs_expected_max_logit_difference=float(np.max(np.abs(a-b))),same_Torch25_logical_state_mismatches=int((aa['logical']!=bb['logical']).sum()),same_Torch25_jointMAP_mismatches=int((aa['joint'].argmax(1)!=bb['joint'].argmax(1)).sum()),same_Torch25_U_mismatches=int(((aa['joint'].max(1)<.5)!=(bb['joint'].max(1)<.5)).sum()),Torch21_vs_Torch25_max_logit_difference=float(np.max(np.abs(a-previous))),Torch21_vs_Torch25_logical_state_mismatches=int((aa['logical']!=pp['logical']).sum()),Torch21_vs_Torch25_jointMAP_mismatches=int((aa['joint'].argmax(1)!=pp['joint'].argmax(1)).sum()),Torch21_vs_Torch25_U_mismatches=int(((aa['joint'].max(1)<.5)!=(pp['joint'].max(1)<.5)).sum()))
  result.append(summary);print('TORCH25_PUBLIC_FIXTURE',i+1,12,summary['same_Torch25_actual_vs_expected_max_logit_difference'],flush=True)
 assert all(torch.equal(before[k],v)for k,v in model.state_dict().items());assert sha(bound_report)==bound_before
 passed=all(r['same_Torch25_actual_vs_expected_max_logit_difference']==0 and r['same_Torch25_logical_state_mismatches']==0 and r['same_Torch25_jointMAP_mismatches']==0 and r['same_Torch25_U_mismatches']==0 for r in result)
 report=dict(schema='PUBLIC_SOURCE_V65_TORCH25_HEAD_RUNTIME_SUPPLEMENT_V1',PASS=passed,source_only=True,private_RGB_GT_scores_or_inputs_read=False,GPU_used=False,optimizer_updates=0,PnP=False,threshold_changes=False,engineering_tolerance_used_for_PASS=False,canonical_same_runtime_requires_bit_exact=True,raw_layout_prior_max_logit_difference=9.5367431640625e-7,raw_layout_diagnostic_scope='Observed before source/helper contiguous normalization; preserved in MEMORY_LAYOUT_DIAGNOSTIC_V1. No1e-5 tolerance was preregistered and none was needed for this canonical PASS.',runtime=dict(python=sys.executable,torch=torch.__version__,numpy=np.__version__,device='cpu'),preparation_runtime=base['runtime'],helper_file_sha256=sha(helper_path),checkpoint_sha256=sha(checkpoint),original_bound_helper_report=dict(path=str(bound_report),sha256=bound_before,unchanged=True),helper_produced_features_sha256=sha(actual_path),expected_features_sha256=sha(expected_path),frozen_model_state_unchanged=True,cross_runtime_differences_are_diagnostic=True,code_sha256=sha(__file__),fixtures=result)
 destination.write_text(json.dumps(report,indent=2)+'\n');print('TORCH25 SUPPLEMENT PASS',passed,'sha256',sha(destination),flush=True)
