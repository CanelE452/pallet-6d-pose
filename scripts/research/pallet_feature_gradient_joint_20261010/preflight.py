"""Gate A: authenticate inference inputs without reading GT or DEV scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

sys.dont_write_bytecode = True
WORKTREE = Path(__file__).resolve().parents[3]
DOC = WORKTREE / '_docs/experiments/pallet_feature_gradient_joint_20261010'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def finite(value):
    if isinstance(value, dict):
        return {str(k): finite(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if hasattr(value, 'tolist'):
        return finite(value.tolist())
    return value


def digest(value):
    return hashlib.sha256(json.dumps(finite(value), sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(finite(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def metadata(candidate):
    return {k: finite(v) for k,v in candidate.items() if k != 'keypoints_xy'}


def run(source, private, output):
    start = time.monotonic()
    assert not output.exists(), 'Do not overwrite sealed input lock'
    fusion = DOC / 'FUSION_METHOD_LOCK.json'
    assert fusion.exists(), 'Numerical fusion settings must be sealed first'
    previous_dir = WORKTREE / '_docs/experiments/pallet_n3_subpix_final_20261010'
    previous = read(previous_dir / 'INPUT_AND_METHOD_LOCK.json')
    previous_audit = read(previous_dir / 'INPUT_AUDIT.json')
    assert previous_audit['status'] == 'PASS'
    old_inputs = {r['id']:r for r in previous_audit['inputs']}
    dim_doc = source / '_docs/experiments/pallet_dim_conditioned_p_v1'
    trained = read(dim_doc/'TRAINING_COMPLETE.json')
    inferred = read(dim_doc/'DEV_INFERENCE_COMPLETE.json')
    calibrated = read(dim_doc/'CALIBRATION_AND_SELECTION.json')
    assert trained['complete'] and inferred['complete'] and calibrated['complete']
    assert calibrated['real_access'] is False and calibrated['new_lambda_cap_sweep'] is False
    assert calibrated['rule'] == {'lam':1.,'max_move_image_diagonal_fraction':.01}
    models = []
    predictions = {}
    bindings = []
    for name in ('TRAINING_COMPLETE.json','DEV_INFERENCE_COMPLETE.json','CALIBRATION_AND_SELECTION.json','DIM_NORMALIZATION_LOCK.json'):
        p=dim_doc/name
        bindings.append(dict(path=str(p.relative_to(source)),sha256=sha(p),bytes=p.stat().st_size))
    for seed in (1,2,3):
        name=f'N3_DIM_SYM_seed{seed}'
        checkpoint=next(r for r in trained['checkpoints'] if f'{name}/' in r['path'])
        pred=inferred['files'][name]
        assert sha(source/checkpoint['path']) == checkpoint['sha256']
        assert sha(source/pred['path']) == pred['sha256']
        cal=calibrated['temperatures'][name]
        assert cal['checkpoint'] == checkpoint
        ck=torch.load(source/checkpoint['path'],map_location='cpu',weights_only=False)
        assert ck['complete'] and ck['step']==6000 and ck['baseline_checkpoint_sha256']==trained['R0_hash']
        packet=read(source/pred['path'])
        assert packet['complete'] and packet['GT_input'] is False and packet['checkpoint']==checkpoint
        predictions[seed]={r['id']:r for r in packet['records']}
        assert len(predictions[seed])==319
        models.append(dict(seed=seed,checkpoint=checkpoint,prediction=pred,temperature=cal['temperature'],step=ck['step'],config=ck['config']))
    protected_code = ['scripts/research/pallet_dim_conditioned_p_v1/'+name for name in ('refiner.py','inference.py','dcp_env.py')]
    protected_code += ['scripts/research/pallet_final_ml_contribution_test_v1/'+name for name in ('generic_point_refiner.py','point_inference.py','common.py')]
    protected_code += ['scripts/research/pallet_line_pose_v1/features.py']
    for relative in protected_code:
        assert sha(source/relative)==sha(WORKTREE/relative), relative
        bindings.append(dict(path=relative,sha256=sha(source/relative),bytes=(source/relative).stat().st_size))
    registry=read(source/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json')['objects']
    registry={r['object_type']:r for r in registry}
    groups=read(source/'_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']
    groups={r['object_type']:r for r in groups}
    ids=previous['population']['frame_ids']
    assert len(ids)==len(set(ids))==319
    assert set(ids)==set(old_inputs)
    manifests=[]
    private_map=[]
    for b in previous['input_manifest']:
        fid=b['id'];old=old_inputs[fid]
        cache_path=source/old['detector_feature_cache']['path']
        image_path=source/old['image']['path']
        assert sha(cache_path)==old['detector_feature_cache']['sha256']==b['cache_sha256'],fid
        assert sha(image_path)==old['image']['sha256']==b['image_sha256'],fid
        assert sha(source/old['annotation']['path'])==old['annotation']['sha256'],fid
        cache=torch.load(cache_path,map_location='cpu',weights_only=False)
        assert cache['GT_input'] is False and cache['image_sha256']==b['image_sha256']
        assert cache['id']==fid and cache['session']==b['session']
        captured=cache['captured'];selected=captured['selected_index']
        assert selected==b['selected_index'] and selected is not None,fid
        q0=np.asarray(captured['candidates'][selected]['keypoints_xy'],dtype=float)
        support=np.isfinite(q0).all(-1)&~np.all(q0==-1,axis=-1)
        assert np.array_equal(q0,b['q0'],equal_nan=True)
        assert np.array_equal(support,b['prediction_support'])
        assert list(cache['raw_hw'])==b['raw_hw']
        assert np.array_equal(cache['dimensions'],b['dimensions_context_WD_H_m'])
        assert np.array_equal(np.asarray(cache['dimensions'])[[0,2,1]],b['dimensions_pnp_WH_D_m'])
        dims=registry[cache['object_type']]['physical_dimensions_m']
        assert np.array_equal([dims['x'],dims['z'],dims['y']],cache['dimensions'])
        assert groups[cache['object_type']]['group_order']==cache['order']==b['canonical_symmetry_order']
        assert [metadata(r) for r in captured['candidates']]==b['candidates_metadata']
        for seed in (1,2,3):
            record=predictions[seed][fid]
            assert record['selected_index']==selected and record['key']==cache['key']
            assert [metadata(r) for r in record['candidates']]==b['candidates_metadata']
            for index,(x,y) in enumerate(zip(captured['candidates'],record['candidates'])):
                if index!=selected:
                    assert np.array_equal(x['keypoints_xy'],y['keypoints_xy'],equal_nan=True)
            qn=np.asarray(record['candidates'][selected]['keypoints_xy'],dtype=float)
            assert np.array_equal(qn[8],q0[8],equal_nan=True)
            assert np.array_equal(qn[~support],q0[~support],equal_nan=True)
        public=dict(b)
        public['object_type']=cache['object_type']
        public['annotation_sha256']=old['annotation']['sha256']
        manifests.append(public)
        private_map.append(dict(id=fid,image=old['image']['path'],cache=old['detector_feature_cache']['path']))
    assert len({r['session'] for r in manifests})==13
    assert {g:sum(r['grade']==g for r in manifests) for g in ('clean','moderate','severe')}==dict(clean=153,moderate=92,severe=74)
    for seed in (1,2,3):
        assert set(predictions[seed])==set(ids)
    write(private/'INPUT_PATHS.json',dict(source_root=str(source),frames=private_map))
    lock=dict(schema='feature_gradient_joint_input_lock_v1',status='PASS',main_sha=subprocess.check_output(['git','rev-parse','origin/main'],cwd=WORKTREE,text=True).strip(),
        fusion_method_lock_sha256=sha(fusion),population=previous['population'],models=models,input_manifest=manifests,
        input_manifest_sha256=digest(manifests),source_bindings=bindings,
        baseline_input_lock=dict(path='_docs/experiments/pallet_n3_subpix_final_20261010/INPUT_AND_METHOD_LOCK.json',sha256=sha(previous_dir/'INPUT_AND_METHOD_LOCK.json')),
        baseline_coordinate_rows=dict(path='_docs/experiments/pallet_n3_subpix_final_20261010/PREDICTIONS.jsonl.gz',sha256=sha(previous_dir/'PREDICTIONS.jsonl.gz')),
        checks=dict(checkpoint_prediction_hashes=True,all_319_raw_image_cache_annotation_hashes=True,canonical_dimensions_K_support_metadata=True,
            previous_input_contract_identical=True,center_unselected_preserved=True,GT_or_DEV_error_reads=False,pose_reference_reads=False),
        coordinate_metadata_note='K is reused from authenticated immutable baseline input lock; annotation bytes hashed but GT fields not parsed.',
        execution=dict(elapsed_seconds=time.monotonic()-start,forward_calls=0,F_calls=0,optimizer_updates=0))
    write(output,lock)
    print('GATE_A_PASS',len(manifests),'INPUT_LOCK_SHA',sha(output),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--private-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,default=DOC/'INPUT_LOCK.json')
    a=p.parse_args();run(a.source_root.resolve(),a.private_dir,a.output)


if __name__=='__main__':main()
