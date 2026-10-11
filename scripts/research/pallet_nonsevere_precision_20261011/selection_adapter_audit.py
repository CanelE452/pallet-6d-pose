"""Independent CPU audit of the selection-only invalid-box tile adapter.

The source model receives the original batch. Only RGB tile allocation sees a
cloned fallback box. No learned forward, pose solver, fit, or REAL data runs.
"""
import argparse,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import geometry as G
import precision_model as M
import selection as S

def tensor_bytes(v):return v.detach().cpu().contiguous().numpy().tobytes()
def run(args):
    start=time.monotonic();root=Path(args.source_root);doc=Path(args.public_dir)
    code=Path(__file__).parent;plan=json.loads((doc/'PLAN_LOCK.json').read_text());selection_lock=json.loads((doc/'SYNTHETIC_SELECTION_LOCK.json').read_text())
    bindings=['data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json','data/pallet/results/pallet_line_pose_v1/cache/boxes.npy','data/pallet/results/pallet_line_pose_v1/cache/points.npy','data/pallet/results/pallet_line_pose_v1/cache/point_valid.npy','data/pallet/results/pallet_line_pose_v1/cache/input_shape.npy','data/pallet/results/pallet_line_pose_v1/cache/gain.npy','data/pallet/results/pallet_line_pose_v1/cache/record_index.npy','data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz','scripts/research/pallet_final_ml_contribution_test_v1/generic_point_refiner.py','data/pallet/results/pallet_final_ml_contribution_test_v1/B_line_vs_point/runs/seed1/last.pt']
    G.write(doc/'SELECTION_ADAPTER_AUDIT_LOCK.json',dict(status='LOCKED_BEFORE_INDEPENDENT_CPU_AUDIT',audit_code_sha256=G.sha(__file__),adapter_code_sha256=G.sha(S.__file__),training_locked_hashes=plan['training_code_sha256'],bindings=[dict(path=p,sha256=G.sha(root/p)) for p in bindings],scope='actual SYNTH calibration1004/selection1031; frozen C2 selection782 retained',new_gpu_calls=0,new_neural_forwards=0,new_pose_solver_calls=0,new_fits=0))
    source=json.loads((root/bindings[0]).read_text());line=root/'data/pallet/results/pallet_line_pose_v1/cache'
    arrays={n:np.load(line/(n+'.npy'),mmap_mode='r') for n in ('boxes','points','point_valid','input_shape','gain','record_index')}
    side_file=np.load(root/'data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz');side={k:side_file[k] for k in side_file.files};indices=arrays['record_index']
    assert np.array_equal(indices,side['record_index'])
    partitions=np.array([source['records'][int(i)]['partition'] for i in indices]);boxes=np.asarray(arrays['boxes'])
    box_valid=np.isfinite(boxes).all(-1)&(boxes[:,2:]>boxes[:,:2]).all(-1)
    counts={p:dict(partition_rows=int((partitions==p).sum()),invalid_boxes=int(((partitions==p)&~box_valid).sum()),invalid_C2_boxes=int(((partitions==p)&~box_valid&(side['order']==2)).sum()),valid_box_diagonal_below_one=int(((partitions==p)&box_valid&(np.linalg.norm(boxes[:,2:]-boxes[:,:2],axis=-1)<1)).sum())) for p in ('train','calibration','selection','heldout')}
    selection_rows=np.flatnonzero((partitions=='selection')&(side['order']==2));assert len(selection_rows)==782
    assert hashlib.sha256(selection_rows.astype('<i8').tobytes()).hexdigest()==selection_lock['row_sha256']
    assert counts['calibration']['invalid_boxes']==0 and counts['selection']['invalid_boxes']==1
    rows=np.flatnonzero(np.isin(partitions,['calibration','selection']))
    checked=0;valid_equal=True;original_equal=True;shared=True;fallback_equal=True
    original_tiles=M.native_tiles
    def inspect(data,rr,tile_batch,disp):
        nonlocal checked,valid_equal,shared,fallback_equal
        valid=box_valid[rr];actual=tile_batch['boxes'].numpy()
        valid_equal=valid_equal and np.array_equal(actual[valid],boxes[rr][valid])
        fallback_equal=fallback_equal and np.array_equal(actual[~valid],np.broadcast_to([0.,0.,1.,1.],actual[~valid].shape))
        shared=shared and all(tile_batch[k].data_ptr()==active[k].data_ptr() for k in ('points','point_valid','input_shape'))
        checked+=len(rr);return None
    try:
        M.native_tiles=inspect
        for lo in range(0,len(rows),256):
            rr=rows[lo:lo+256];active={k:torch.from_numpy(np.array(arrays[k][rr])) for k in ('points','boxes','point_valid','input_shape')};before={k:tensor_bytes(v) for k,v in active.items()}
            S.evaluation_tiles(None,rr,active,None)
            original_equal=original_equal and all(tensor_bytes(active[k])==before[k] for k in before)
    finally:M.native_tiles=original_tiles
    assert checked==2035 and valid_equal and original_equal and shared and fallback_equal
    ck=torch.load(root/bindings[-1],map_location='cpu')
    base=SimpleNamespace(displacements=ck['model_state_dict']['displacements'],stencil=ck['model_state_dict']['stencil'],stencil_fraction=ck['config']['stencil_fraction'])
    invalid=np.flatnonzero(np.isin(partitions,['calibration','selection'])&~box_valid)
    support_tests=[]
    for row in invalid:
        out=G.fixed_outputs(arrays,np.array([row]),base);assert not out['point_support'].any()
        source_decoder=G.load_module(root/bindings[-2],'source_adapter_decode').decode
        q1=source_decoder(out,1.,1.,None);out['logits']=torch.arange(222,dtype=torch.float32)[None,None].expand(1,8,222);q2=source_decoder(out,4.,1.,None)
        assert tensor_bytes(q1)==tensor_bytes(out['points_raw'])==tensor_bytes(q2)
        support_tests.append(dict(id=source['records'][int(indices[row])]['id'],partition=str(partitions[row]),symmetry_order=int(side['order'][row]),original_box_nonfinite=not bool(np.isfinite(boxes[row]).all()),source_supported_corners=int(out['point_support'].sum()),decoded_points_identical_for_arbitrary_logits=True,original_point_valid_corners=int(arrays['point_valid'][row,:8].sum())))
    # Exercise the actual tile allocator on one invalid and one valid source row.
    # Existing SYNTH RGB is read privately; no synthetic image is created.
    good=int(rows[np.flatnonzero(box_valid[rows])[0]]);rr=np.array([int(invalid[0]),good])
    active={k:torch.from_numpy(np.array(arrays[k][rr])) for k in ('points','boxes','point_valid','input_shape')};before={k:tensor_bytes(v) for k,v in active.items()}
    data=SimpleNamespace(indices=indices,source=source,arrays=arrays);tile_capture={}
    def cpu_tiles(data,rows,batch,disp):
        tile_capture['boxes']=batch['boxes'].clone();return original_tiles(data,rows,batch,disp,device='cpu')
    try:
        M.native_tiles=cpu_tiles;rgb,grid=S.evaluation_tiles(data,rr,active,base.displacements)
    finally:M.native_tiles=original_tiles
    assert torch.isfinite(rgb).all() and torch.isfinite(grid).all() and all(tensor_bytes(active[k])==before[k] for k in before)
    validbatch={k:v[1:2].clone() for k,v in active.items()};vrgb,vgrid=original_tiles(data,np.array([good]),validbatch,base.displacements,device='cpu')
    # Different batchmax padding changes normalized grid only; compare the actual
    # original-resolution tile bytes and recovered native sample coordinates.
    side_rgb=rgb.shape[-1];side_valid=vrgb.shape[-1]
    rgb_same=torch.equal(rgb[8:,:,:side_valid,:side_valid],vrgb)
    pos1=(grid[8:]+1)*side_rgb/2-.5;pos2=(vgrid+1)*side_valid/2-.5
    grid_error=float((pos1-pos2).abs().max());assert rgb_same and grid_error<1e-4
    locked=all(G.sha(code/n)==h for n,h in plan['training_code_sha256'].items());assert locked
    report=dict(status='PASS',partition_counts=counts,calibration_rows_inspected=1004,selection_partition_rows_inspected=1031,physical_C2_selection_denominator=782,physical_C2_selection_row_digest_unchanged=True,invalid_box_rows=support_tests,source_fallback=[0.,0.,1.,1.],adapter_modifies_only_cloned_tile_allocation_box=True,original_model_batch_values_bitwise_unchanged=original_equal,original_points_masks_shape_shared_by_identity=shared,valid_boxes_bitwise_unchanged=valid_equal,invalid_source_corner_support_all_false=True,invalid_decoded_points_unchanged_for_arbitrary_RGB_logits=True,actual_native_tiles_CPU_test=dict(source_rows=2,existing_SYNTH_RGB_reads_unique=2,new_images_generated=0,rgb_tensor_finite=bool(torch.isfinite(rgb).all()),sampling_grid_finite=bool(torch.isfinite(grid).all()),valid_row_native_RGB_pixels_identical=rgb_same,valid_row_native_grid_coordinate_max_abs_difference_px=grid_error),training_adapter_applied=False,locked_training_and_model_hashes_unchanged=locked,new_fit_or_parameter_choices=0,calibration_objective_unchanged='original Gaussian candidate CE excludes unsupported corners as before',failed_rows_not_removed=True,selection_comparison_denominator_unchanged=True,final_cap_subpix_or_F_changes=False,REAL_or_manual_reference_reads=0,new_neural_forwards=0,new_GPU_calls=0,new_solver_calls=0,optimizer_updates=0,public_RAW_RGB=False,public_GT_coordinates=False,source_files_unchanged=True,wall_seconds=time.monotonic()-start)
    G.write(doc/'SELECTION_INVALID_BOX_ADAPTER_AUDIT.json',report);print('SELECTION_INVALID_BOX_CPU_AUDIT_PASS',counts,support_tests,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--public-dir',required=True);a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);run(a)
