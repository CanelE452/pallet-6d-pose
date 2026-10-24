#!/usr/bin/env python3
"""Compare pure current-inference helpers against public-only source fixtures."""
from pathlib import Path
import ast,hashlib,importlib.util,json,sys
import numpy as np,torch
from PIL import Image
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def load(path):
 with np.load(path,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def function_sha(path,names):
 tree=ast.parse(Path(path).read_text());return {node.name:hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))and node.name in names}
if __name__=='__main__':
 report=OUT/'HELPER_AUDIT_REPORT.json';assert not report.exists()
 helperpath=ROOT/'inference/infer_current_frozen.py';recipepath=ROOT/'global_recipe.py';names=['context64','initial_query_box','query_validity','input_features','forward_frame','probabilities'];before_funcs=function_sha(helperpath,names)
 helper=module(helperpath,'public_fixture_inference_helper');recipe=module(recipepath,'public_fixture_global_recipe');fixtures=json.loads((OUT/'FIXTURE_INPUTS.json').read_text())['inputs'];logits=load(OUT/'CPU_LOGIT_FIXTURES.npz');expected=load(OUT/'EXPECTED_FEATURES.npz')
 modelpath=ROOT/'runs/MIXED_GLOBAL_UNWEIGHTED/code/query_global_amodal_head.py';checkpoint=ROOT/'runs/MIXED_GLOBAL_UNWEIGHTED/BEST_GLOBAL_AMODAL_HEAD.pt'
 if not checkpoint.exists():
  candidates=list((ROOT/'runs/MIXED_GLOBAL_UNWEIGHTED').glob('BEST*.pt'));assert len(candidates)==1;checkpoint=candidates[0]
 head=module(modelpath,'public_fixture_frozen_head');torch.set_num_threads(1);model,payload=head.load_model(checkpoint,'cpu');model.eval().requires_grad_(False);state={k:v.detach().clone()for k,v in model.state_dict().items()}
 results=[];produced={}
 for r in fixtures:
  i=int(r['fixture_index']);prefix=f'f{i:03d}_';rgb=np.asarray(Image.open(r['image']).convert('RGB'));assert sha(r['image'])==r['image_sha256'];f=helper.input_features(rgb,r['query'],r['side'],r['box'],logits[f'logit_{i:03d}'],recipe)
  truth={k:expected[prefix+k]for k in ['rgb_native','box_grid','sam','global_rgb','global_sam','global_prompt','global_index64','global_index8','global_query_valid','global_valid']}
  truth['native_valid']=expected[prefix+'dense_valid'];truth['global_rgb']=truth['global_rgb'][None];truth['global_sam']=truth['global_sam'][None];truth['global_prompt']=truth['global_prompt'][None];truth['global_valid']=truth['global_valid'][None]
  for k,v in f.items():produced[prefix+k]=v
  comparisons={k:dict(shape=list(f[k].shape),dtype=str(f[k].dtype),value_mismatches=int(np.count_nonzero(f[k]!=v)),max_abs=float(np.max(np.abs(f[k].astype(float)-v.astype(float)))))for k,v in truth.items()}
  x8=expected[prefix+'x8'];ctx=helper.context64(x8[None])[0];comparisons['context64']=dict(shape=list(ctx.shape),dtype=str(ctx.dtype),value_mismatches=int(np.count_nonzero(ctx!=expected[prefix+'context'])),max_abs=float(np.max(np.abs(ctx-expected[prefix+'context']))))
  qvalid,pixel=helper.query_validity(r['query'],r['raw_hw'],np.ones(8,bool));comparisons['query_validity']=dict(shape=list(qvalid.shape),dtype=str(qvalid.dtype),value_mismatches=int(np.count_nonzero(qvalid!=expected[prefix+'query_validity'])),max_abs=float(np.max(qvalid!=expected[prefix+'query_validity'])))
  initial=helper.initial_query_box(r['query'],r['raw_hw']);box_matches_source=None
  if r['source']=='SYNTH':box_matches_source=bool(np.array_equal(initial,np.asarray(r['box'])))
  # Same CPU logit, same fixed frozen head, two independently built recipes.
  actual_logits=helper.forward_frame(model,x8,ctx,f,'cpu');expected_logits=helper.forward_frame(model,x8,expected[prefix+'context'],truth,'cpu');model_difference=float(np.max(np.abs(actual_logits-expected_logits)))
  canonical_actual={k:np.ascontiguousarray(v)for k,v in f.items()};canonical_expected={k:np.ascontiguousarray(v)for k,v in truth.items()}
  ca=helper.forward_frame(model,x8,np.ascontiguousarray(ctx),canonical_actual,'cpu');ce=helper.forward_frame(model,x8,np.ascontiguousarray(expected[prefix+'context']),canonical_expected,'cpu');canonical_difference=float(np.max(np.abs(ca-ce)))
  raw0=helper.probabilities(actual_logits);raw1=helper.probabilities(expected_logits);logical_flips=int((raw0['logical']!=raw1['logical']).sum());MAP_flips=int((raw0['joint'].argmax(1)!=raw1['joint'].argmax(1)).sum());Uflips=int(((raw0['joint'].max(1)<.5)!=(raw1['joint'].max(1)<.5)).sum())
  layouts={k:dict(helper_strides=list(f[k].strides),NPZ_expected_strides=list(truth[k].strides),helper_C_contiguous=bool(f[k].flags.c_contiguous),NPZ_expected_C_contiguous=bool(truth[k].flags.c_contiguous))for k in truth if f[k].strides!=truth[k].strides}
  produced[prefix+'context']=ctx;produced[prefix+'head_logits']=actual_logits
  results.append(dict(fixture_index=i,id=r['id'],source=r['source'],split=r['split'],comparisons=comparisons,SYNTH_initial_box_matches_source_prompt=box_matches_source,initial_box=initial.tolist()if initial is not None else None,source_prompt_box_is_N0_based=r['source']=='SYNTH',same_values_original_memory_layout_CPU_model_max_logit_difference=model_difference,same_logit_same_contiguous_layout_CPU_model_max_logit_difference=canonical_difference,original_layout_logical_state_mismatches=logical_flips,original_layout_jointMAP_mismatches=MAP_flips,original_layout_U_mismatches=Uflips,memory_layout_differences=layouts))
  print('PURE_HELPER_AUDIT',i+1,len(fixtures),'mismatches',sum(v['value_mismatches']for v in comparisons.values()),'head_diff',model_difference,flush=True)
 after_funcs=function_sha(helperpath,names);assert before_funcs==after_funcs,'pure helper changed during fixture audit'
 assert all(torch.equal(state[k],v)for k,v in model.state_dict().items()),'frozen model changed'
 passtest=all(not any(v['value_mismatches']for v in r['comparisons'].values())and r['same_logit_same_contiguous_layout_CPU_model_max_logit_difference']==0 and r['original_layout_logical_state_mismatches']==0 and r['original_layout_jointMAP_mismatches']==0 and r['original_layout_U_mismatches']==0 and r['SYNTH_initial_box_matches_source_prompt']is not False for r in results)
 np.savez_compressed(OUT/'HELPER_PRODUCED_FEATURES.npz',**produced)
 output=dict(schema='PUBLIC_SOURCE_FROZEN_V65_INFERENCE_PURE_HELPER_AUDIT_V2',PASS=passtest,source_only=True,private_RGB_GT_scores_read=False,GPU_used=False,optimizer_updates=0,PnP=False,threshold_changes=False,root_files_modified=False,helper_path=str(helperpath),helper_file_sha256_at_review=sha(helperpath),pure_function_AST_sha256=before_funcs,recipe_sha256=sha(recipepath),checkpoint_sha256=sha(checkpoint),frozen_model_state_unchanged=True,source_fixture_manifest_sha256=sha(OUT/'FIXTURE_INPUTS.json'),expected_features_sha256=sha(OUT/'EXPECTED_FEATURES.npz'),CPU_logit_sha256=sha(OUT/'CPU_LOGIT_FIXTURES.npz'),code_sha256=sha(__file__),SAM_drift_scope='This PASS compares the SAME full CPU logit through two recipes. CPU-versus-saved-GPU numerical differences remain separately in EXPECTED_FEATURES_REPORT.json; this is not a claim of bit-exact CPU/GPU SAM.',CPU_layout_scope='Original feature values match bit-exactly but strided helper arrays versus NPZ-contiguous arrays can select different CPU convolution layouts. Record original-layout numerical differences; canonicalize BOTH audit-only arrays to C-contiguous to verify identical head outputs. No inference/model/threshold change was made.',YCB_source_box_scope='YCB source jittered annotation box was supplied explicitly. Source box is not N0-query derived; deployment initialbox formula itself is tested independently and SYNTH source box equals it.',execution_section_scope='Pure helper ASTs verified stable. Main private execution section was still being finalized and must be reviewed/bound separately before private file opens.',fixtures=results)
 output['runtime']=dict(python=sys.executable,torch=torch.__version__,numpy=np.__version__,device='cpu');output['helper_produced_features_sha256']=sha(OUT/'HELPER_PRODUCED_FEATURES.npz')
 report.write_text(json.dumps(output,indent=2)+'\n');print('PUBLIC PREPROCESSING FIXTURE AUDIT PASS',passtest,flush=True)
