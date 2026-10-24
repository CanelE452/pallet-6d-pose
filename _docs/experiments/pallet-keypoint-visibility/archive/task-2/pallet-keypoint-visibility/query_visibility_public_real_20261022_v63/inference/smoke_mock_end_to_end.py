"""Exercise the gated runner against generated mock files only, then delete them.

The approved gate created here points exclusively to temporary random inputs,
mock RGB, a random mock checkpoint, and source code. It cannot open real data.
"""
import csv,gzip,hashlib,importlib.util,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('mock_gated_runner',HERE/'infer_current_frozen.py')
runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)

def bind(path):return dict(path=str(Path(path).resolve()),sha256=runner.sha(path))


torch.set_num_threads(1);torch.manual_seed(20261022)
rng=np.random.default_rng(20261022)
head=runner.import_bound(ROOT/'model/query_head.py',runner.sha(ROOT/'model/query_head.py'),'e2e_mock_head')
model=head.QueryHead(head.HeadConfig(target_context=True,rgb_branch=True,rgb_sampling_version='native9')).eval().requires_grad_(False)
with tempfile.TemporaryDirectory(prefix='pallet_v63_mock_only_')as tmp:
    tmp=Path(tmp);checkpoint=tmp/'MOCK_RANDOM_HEAD.pt';head.save_checkpoint(checkpoint,model,threshold=.5)
    freeze=tmp/'MOCK_FREEZE.json';runner.write(freeze,dict(checkpoint_sha256=runner.sha(checkpoint),head_config=model.config.__dict__,threshold=.5,eval_data_used=False,PnP_used=False,scope='GENERATED_MOCK_ONLY_NOT_A_SOURCE_SELECTION'))
    frames=[];metadata=[]
    for i in range(2):
        image=tmp/f'MOCK_RGB_{i}.png';Image.fromarray(rng.integers(0,256,(90,110,3),dtype=np.uint8)).save(image)
        points=[[20.+k*3,30.+k]for k in range(8)]+[[45.,45.]]
        if i==1:points[0]=[109.8,30.]
        frames.append(dict(id=f'MOCK_NOT_CURRENT_EVAL:{i:06d}',image=str(image),image_sha256=runner.sha(image),raw_hw=[90,110],initial_points=points,initial_pose=dict(available=True,cf_extents=[1.,.2,.6],R_cf=np.eye(3).tolist(),centroid=[0.,0.,2.])))
        metadata.append(dict(id=frames[-1]['id'],side=32.))
    roi=tmp/'MOCK_ROI.npz';x=rng.normal(size=(2,8,65,9,9)).astype(np.float16);x[:,:,64]=1;np.savez_compressed(roi,x=x,pixel_valid=np.ones((2,8),bool))
    prepared=tmp/'MOCK_INPUTS.json.gz'
    with gzip.open(prepared,'wt')as f:json.dump(frames,f)
    meta=tmp/'MOCK_METADATA.json';runner.write(meta,metadata)
    seal=tmp/'MOCK_ONLY_SELECTION.json';runner.write(seal,dict(scope='GENERATED_MOCK_ONLY',current_evaluation=False))
    gate=tmp/'MOCK_ONLY_GATE.json';out=tmp/'MOCK_OUTPUT'
    candidate=dict(name='MOCK_RANDOM_RGB_CONTEXT',checkpoint=bind(checkpoint),freeze=bind(freeze),model_code=bind(ROOT/'model/query_head.py'),rgb_recipe=bind(ROOT/'model/build_rgb_source_cache.py'),geometry_helper=bind(ROOT/'contract/geometry_self_rule.py'),inference_code_sha256=runner.sha(HERE/'infer_current_frozen.py'),output_directory=str(out),compute_sam_baseline=False)
    runner.write(gate,dict(scope='GENERATED_MOCK_ONLY_NOT_REAL_APPROVAL',approved=True,source_selection_complete=True,allow_current_evaluation_inference=True,current_eval_used_for_selection=False,source_selection_seal=bind(seal),inputs=dict(roi_cache=bind(roi),prepared_inputs=bind(prepared),feature_metadata=bind(meta)),approved_candidates=[candidate]))
    runner.run(SimpleNamespace(gate=str(gate),candidate=candidate['name'],out=None,device='cpu',sam_device=None,batch_size=8))
    with(out/'REAL_QUERY_SCORES.csv').open(newline='')as f:rows=list(csv.DictReader(f))
    actual=json.loads((out/'INFERENCE_SEAL.json').read_text())
    assert len(rows)==16 and all(r['id'].startswith('MOCK_NOT_CURRENT_EVAL:')for r in rows)
    assert sum(r['valid']=='False'for r in rows)==1
    assert all(0<=float(r['p_visible'])<=1 for r in rows)
    assert actual['head_state']['frozen_state_unchanged']and not actual['SAM_used']
    assert actual['semantic_annotation_reads']==actual['real_CAL_model_reads']==actual['new_PnP_calls']==0
    assert actual['real_RGB_images_opened']==2
    # A completed or partial run directory prevents repeat evaluation.
    try:runner.run(SimpleNamespace(gate=str(gate),candidate=candidate['name'],out=None,device='cpu',sam_device=None,batch_size=8));raise AssertionError('Duplicate run accepted')
    except FileExistsError:pass
    proof=dict(schema='FROZEN_INFERENCE_GENERATED_MOCK_END_TO_END_V1',scope='Only temporary generated random images, ROI arrays, mock initial poses, random checkpoint; all temporary data deleted',mock_frames=2,mock_slots=16,invalid_query_retained=1,gate_hash_checks_passed=True,shared_RGB_center_verified=True,head_state_unchanged=True,repeat_batch_identical=True,raw_probability_separate_from_geometry=True,duplicate_run_refused=True,SAM_model_or_checkpoint_loaded=False,actual_current_evaluation_RGB_cache_metadata_or_scores_read=0,new_PnP_calls=0,new_training=0,runner_sha256=runner.sha(HERE/'infer_current_frozen.py'))
runner.write(HERE/'MOCK_END_TO_END_SMOKE.json',proof);print(json.dumps(proof))
