"""Repair the G38 conditioning source BEFORE any DIM main optimization.

Original exporter dimensions are camera-facing. Use audited fixed object XYZ,
not a current visible-face swap. Preserve all pre-correction evidence.
"""
from collections import Counter
import shutil
import numpy as np
import dcp_env as E
from refiner import dimension_features

def archive(path):
    root=E.DOC if path.is_relative_to(E.DOC) else E.RAW
    dst=root/'history/camera_facing_dimension_correction'/path.relative_to(root)
    if not dst.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
    return dst

def main():
    marker=E.DOC/'CANONICAL_DIMENSION_CORRECTION.json'
    if marker.exists() and E.read(marker).get('complete'):print('CORRECTION_ALREADY_COMPLETE');return
    for a in ['N0_BASE_REPLAY','N1_SYM_ONLY']:
        for s in [1,2,3]:assert E.read(E.DOC/f'fits/{a}_seed{s}.json')['complete']
    assert not any((E.RAW/f'runs/{a}_seed{s}').exists() for a in E.ARMS[2:] for s in [1,2,3]),'DIM main initialization already occurred: stop for a different correction plan'
    assert not (E.DOC/'TRAIN_PROTOCOL_LOCK.json').exists(),'Expected deliberate gate preserving old protocol in history'
    oldpaths={}
    for name in ['DIM_NORMALIZATION_LOCK.json','DIMENSION_INPUT_PROVENANCE.json','SOURCE_BINDINGS.json','METADATA_DIAGNOSTIC_LOCK.json','SMOKE_COMPLETE.json','ACTUAL_ADAPTER_TESTS.json']:
        oldpaths[name]=archive(E.DOC/name)
    old_npz=archive(E.RAW/'DIMENSION_SIDECAR.npz');old_json=archive(E.RAW/'DIMENSION_SIDECAR.json')
    oldside=np.load(old_npz);side={k:oldside[k].copy() for k in oldside.files};records=E.read(old_json)['records'];source=E.read(E.LINE/'SOURCE_MANIFEST.json')['records']
    geometric=E.ROOT/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz';geo=np.load(geometric);index={str(fid):i for i,fid in enumerate(geo['stems'])};fixed_dims=geo['dims'];geometry_hash=E.sha(geometric)
    builder=E.ROOT/'challenge/yolo_pose_one_model/spatial_concat_scratch/build_probe_metadata.py';module=E.C.module('dcp_fixed_dimension_provenance',builder)
    changes=Counter();axis=Counter()
    for i,(r,src) in enumerate(zip(records,source)):
        assert r['frame_id']==src['id'];xyz=fixed_dims[index[src['id']]]
        if src['source']=='G38':
            obj=E.read(E.ROOT/src['renderer_annotation_locator_provenance_only'])['objects'][0]
            camera,fixed,permutation,case=module.fixed_dimensions(obj,src['id'])
            assert np.array_equal(np.array(fixed),xyz),(src['id'],fixed,xyz)
            assert obj['source_asset']==r['source_asset'];axis[case]+=1
            dim=xyz[[0,2,1]]
            if not np.array_equal(dim,side['dimensions'][i]):changes[src['source']]+=1
            side['dimensions'][i]=dim;r['canonical_WDH']=dim.tolist()
            r['camera_facing_dimension_source_provenance_only']=r['dimension_source']
            r['dimension_source']='GEOMETRY_SIDETABLE.dims from audited fixed_renderer_dimensions_m_xyz_model_input; full G38 reconstruction cross-check'
            r['dimension_source_file']=str(geometric.relative_to(E.ROOT));r['dimension_source_sha256']=geometry_hash
        else:assert np.array_equal(xyz[[0,2,1]],side['dimensions'][i]),src['id']
        if i%10000==0:print('FIXED_DIM_AUDIT',i,60000,flush=True)
    # Geometry metadata changes only; no target/order/permutation/cache change.
    for k in side:
        if k!='dimensions':assert np.array_equal(side[k],oldside[k])
    E.write(E.RAW/'DIMENSION_SIDECAR.json',dict(records=records,GT_pose_used=False,canonical_order='fixed renderer XYZ -> WDH = X,Z,Y',
      correction='G38 camera-facing dimensions removed. Fixed object axes are recovered offline using existing renderer index metadata, never using inference-time pose or evaluation-selected branch.'))
    np.savez(E.RAW/'DIMENSION_SIDECAR.npz',**side)
    tr=np.array([r['partition']=='train' for r in records]);f=dimension_features(side['dimensions'][tr]);std=f.std(0);norm=E.read(oldpaths['DIM_NORMALIZATION_LOCK.json'])
    norm.update(mean=f.mean(0).tolist(),scale=np.where(std<1e-8,1,std).tolist(),std=std.tolist(),canonical_dimension_correction='CANONICAL_DIMENSION_CORRECTION.json')
    E.write(E.DOC/'DIM_NORMALIZATION_LOCK.json',norm)
    provenance=E.read(oldpaths['DIMENSION_INPUT_PROVENANCE.json']);provenance.update(sidecar=E.bound(E.RAW/'DIMENSION_SIDECAR.json'),normalization=E.bound(E.DOC/'DIM_NORMALIZATION_LOCK.json'),
      axis_audit='Supersedes initial sorted-extent check: G38 dimensions_m was camera-facing, explicitly forbidden as model input in existing source builder. Fixed stored renderer XYZ is now used and cross-checked for all G38 rows.',
      fixed_dimension_source=E.bound(geometric),fixed_dimension_builder=E.bound(builder),G38_rows_reconstructed=40000,G38_input_rows_changed=dict(changes),
      no_GT_pose_or_clicked_keypoints_input=True,no_framewise_WD_swap=True,offline_dimension_reconstruction_uses_renderer_index_metadata=True,
      inference_receives_fixed_XYZ_only=True,original_PROBE_METADATA_60K_jsonl_present=False,
      PROBE_audit=E.bound(E.ROOT/'challenge/yolo_pose_one_model/spatial_concat_scratch/PROBE_METADATA_60K_AUDIT.json'))
    E.write(E.DOC/'DIMENSION_INPUT_PROVENANCE.json',provenance)
    binding=E.read(oldpaths['SOURCE_BINDINGS.json']);binding['files'] += [E.bound(geometric),E.bound(builder),E.bound(E.ROOT/'challenge/yolo_pose_one_model/spatial_concat_scratch/PROBE_METADATA_60K_AUDIT.json')]
    E.write(E.DOC/'SOURCE_BINDINGS.json',binding)
    diagnostic=E.read(oldpaths['METADATA_DIAGNOSTIC_LOCK.json']);diagnostic['train_mean_dimensions_WDH']=side['dimensions'][tr].mean(0).tolist();diagnostic['canonical_dimension_correction']='CANONICAL_DIMENSION_CORRECTION.json'
    E.write(E.DOC/'METADATA_DIAGNOSTIC_LOCK.json',diagnostic)
    # Preserve invalid metadata smoke exactly. N0 smoke is metadata-free and kept.
    for path in [E.RAW/'smoke/N4_META_SYM_seed1',E.DOC/'smoke/N4_META_SYM_seed1.json',E.DOC/'SMOKE_COMPLETE.json',E.DOC/'ACTUAL_ADAPTER_TESTS.json']:
        root=E.DOC if path.is_relative_to(E.DOC) else E.RAW;dst=root/'history/invalid_metadata_smoke'/path.relative_to(root)
        if path.exists():
            assert not dst.exists();dst.parent.mkdir(parents=True,exist_ok=True);path.rename(dst)
    protocol=E.read(E.DOC/'history/TRAIN_PROTOCOL_LOCK_BEFORE_CANONICAL_CORRECTION.json');protocol.update(canonical_dimension_correction='G38 fixed renderer XYZ from validated immutable geometry side table, not camera-facing dimensions_m',
      valid_smoke_updates=200,archived_invalid_smoke_updates=100,total_smoke_updates=300)
    E.write(E.DOC/'TRAIN_PROTOCOL_LOCK.json',protocol)
    E.write(marker,dict(complete=True,time=E.now(),cause='Referenced export.metadata reads G38 camera-facing dimensions; its use conflicts with higher-level fixed-physical-dimension requirement.',
      corrected_input='fixed_renderer_dimensions_m_xyz_model_input via immutable GEOMETRY_SIDETABLE.dims, checked against original fixed_dimensions reconstruction for all 40,000 G38 rows; legacy input unchanged',
      changed_input_rows=dict(changes),axis_cases=dict(axis),metadata_free_main_fits_retained=6,metadata_conditioned_main_updates_before_correction=0,
      metadata_free_retention_proof='N0/N1 forward receives no dimension context; targets/groups/features/order/base initialization unchanged. Their historical protocol bindings are archived, not rewritten.',
      data_arrays_other_than_dimensions_exact=True,original_sources_modified=False,original_exporter_modified=False,new_architecture_or_budget_search=False,
      archived_invalid_N4_smoke_updates=100,archived_CPU_adapter_updates=9,rerun_required=['N4 seed1 100-step smoke','actual-cache adapter test'],
      previous_protocol=E.bound(E.DOC/'history/TRAIN_PROTOCOL_LOCK_BEFORE_CANONICAL_CORRECTION.json'),old_sidecar=E.bound(old_json),new_sidecar=E.bound(E.RAW/'DIMENSION_SIDECAR.json')))
    print('CANONICAL_DIMENSION_CORRECTION_COMPLETE',changes,flush=True)
if __name__=='__main__':main()
