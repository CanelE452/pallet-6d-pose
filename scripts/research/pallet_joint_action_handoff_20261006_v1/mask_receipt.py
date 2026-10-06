"""Reconstruct sampling masks from locked coordinates; no CNN/final F/fitting.

Executed GPU mask tensors were not saved. This records CPU FP32 coordinate
reconstruction, including separately counted real-bank initialization PnP.
"""
from pathlib import Path
import argparse, collections, hashlib, json, time
import cv2, numpy as np, torch
from .a_common import ROOT, DOC, POSE, read, sha, write, hash_value
from .a_data import Data, network_bank
from .a_experiment import BankStore, bank_job
from .geometry import permute_bank
from .a_evaluate import finite_json


def masks(points, boxes, point_valid, input_shape, candidates, action_valid, stencil, fraction):
    """Exact published scorer coordinate-mask algebra; accepts no GT/evidence."""
    p, box, hw, q = [torch.as_tensor(v, dtype=torch.float32) for v in (points, boxes, input_shape, candidates)]
    valid = torch.as_tensor(point_valid).bool() & torch.isfinite(p).all(-1) & ~(p == -1).all(-1)
    box_valid = torch.isfinite(box).all(-1) & (box[2:] > box[:2]).all()
    safe = torch.where(valid[:, None], p, torch.zeros_like(p))
    bb = torch.where(box_valid, box, box.new_tensor([0, 0, 1, 1]))
    diag = (bb[2:] - bb[:2]).norm().clamp_min(1)
    safe_q = torch.where(valid[None, :, None], q, safe[None])
    positions = safe_q[1:, :8].transpose(0, 1)[:, :, None] + diag * fraction * stencil[None, None]
    inside = valid[:8, None, None] & (positions[..., 0] >= 0) & (positions[..., 1] >= 0) & (positions[..., 0] < hw[1]) & (positions[..., 1] < hw[0])
    act = torch.as_tensor(action_valid).bool()
    support = valid[:8] & box_valid & (inside.any(-1) & act[None, 1:]).any(-1)
    return inside.numpy(), support.numpy(), valid.numpy()


def resume_guard(source_root, cache_dir):
    """Verify current inputs without rerunning 73GB feature content audit."""
    dependency = read(DOC/'INPUT_DEPENDENCIES.json')
    assert dependency['status']=='DONE' and Path(dependency['source_root'])==source_root
    for b in dependency['inputs']:
        p=Path(b['path']);assert p.stat().st_size==b['bytes'] and sha(p)==b['sha256'],str(p)
    source = read(DOC/'SOURCE_CACHE_HASHES.json')
    assert source['status']=='DONE' and Path(source['source_root'])==source_root
    stat_bindings=[]
    for b in source['arrays']:
        p=source_root/b['path'];s=p.stat();assert s.st_size==b['bytes'] and s.st_mtime_ns==b['mtime_ns'],str(p)
        stat_bindings.append({k:b[k] for k in ['name','path','sha256','bytes','mtime_ns']})
    protocol=read(DOC/'A_protocol.json')
    for b in protocol['input_bindings']:
        p=source_root/b['path'];assert p.stat().st_size==b['bytes'] and sha(p)==b['sha256'],str(p)
    code=[Path(__file__),*(Path(__file__).parent/f for f in ['a_common.py','a_data.py','a_experiment.py','geometry.py','scorer.py']),
          ROOT/'scripts/research/pallet_final_ml_contribution_test_v1/generic_point_refiner.py',
          ROOT/'scripts/research/pallet_dim_conditioned_p_v1/pose.py',ROOT/'challenge/evaluation_v2/pnp_selector.py',
          ROOT/'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py']
    checkpoints=[*(source_root/f'data/pallet/results/pallet_dim_conditioned_p_v1/runs/N3_DIM_SYM_seed{s}/last.pt' for s in [1,2,3]),
                 *(cache_dir/f'fits/{a}_seed{s}/last.pt' for s in [1,2,3] for a in ['GEO','PERM'])]
    return dict(protocol_sha256=sha(DOC/'A_protocol.json'),code_bindings={str(p):sha(p) for p in code},
        source_bank_sha256=sha(cache_dir/'source_banks.npy'),
        source_bank_auxiliary={str(cache_dir/f):sha(cache_dir/f) for f in ['source_counts.npy','source_hypothesis.npy','BANK_NAMES.json','BANK_BINDING.json','GENERATION_CODE_BINDINGS.json']},
        all9_checkpoint_bindings={str(p):sha(p) for p in checkpoints},
        input_dependency_inputs_sha256=hash_value(dependency['inputs']),source_array_bindings_sha256=hash_value(stat_bindings),
        source_feature_validation='reuse completed full-byte SHA receipt plus current unchanged size/mtime; no second73GB byte pass',
        input_dependencies_receipt=str(DOC/'INPUT_DEPENDENCIES.json'),source_content_receipt=str(DOC/'SOURCE_CACHE_HASHES.json'))


def verify_external(receipt):
    for b in receipt['external_mask_files']:
        p=Path(b['path']);assert p.stat().st_size==b['bytes'] and sha(p)==b['sha256'],str(p)


def sampler_cells(data, frames, dest):
    """Record legacy feature-cell padding masks without sampling features."""
    saved, metadata = [], {}
    for split, fs in frames:
        for level, stride in [('p3',8),('p4',16)]:
            shapes=[tuple(data.arrays[level].shape[-2:]) if split=='SYNTH_HELDOUT' else tuple(f['captured']['captured'][level].shape[-2:]) for f in fs]
            bits=max(h*w for h,w in shapes); packed=np.zeros((len(fs),(bits+7)//8),dtype='uint8')
            for i,(f,(h,w)) in enumerate(zip(fs,shapes)):
                hw=data.arrays['input_shape'][f['cache_row']] if split=='SYNTH_HELDOUT' else f['captured']['captured']['input_shape']
                ys=(np.arange(h,dtype='float32')+.5)*stride;xs=(np.arange(w,dtype='float32')+.5)*stride
                support=(ys[:,None]<hw[0])&(xs[None,:]<hw[1]);v=np.packbits(support.reshape(-1),bitorder='little');packed[i,:len(v)]=v
                metadata.setdefault((split,f['id']),{})[level]=dict(shape_hw=[h,w],stride=stride,inside_cells=int(support.sum()),outside_cells=int((~support).sum()),input_shape_hw=np.asarray(hw).tolist())
            path=dest/f'{split}_{level}_feature_cell_support_packed.npy';np.save(path,packed)
            saved.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,shape=list(packed.shape),layout='little bit order,row-major featureH*W prefix; zero unused padding'))
    return saved,metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1); cv2.setNumThreads(1)
    begin, cpu = time.monotonic(), time.process_time()
    receipt_path=DOC/'results/A_SAMPLING_MASK_RECEIPT.json'
    guard=resume_guard(args.source_root,args.cache_dir)
    if receipt_path.exists():
        old=read(receipt_path);assert old['status']=='DONE'
        assert old.get('resume_guard')==guard,'Mask receipt inputs/code changed; do not reuse stale masks'
        verify_external(old)
        print(json.dumps(dict(status='REUSED_VERIFIED_DONE',new_CNN_forward_calls=0,new_final_F_scoring_calls=0,
                              new_initialization_PnP_calls=0,new_optimizer_updates=0,seconds_wall=time.monotonic()-begin)))
        return
    protocol = read(DOC / 'A_protocol.json')
    data = Data(args.source_root)
    store = BankStore(data, args.cache_dir, protocol)
    checkpoint = data.dim / 'runs/N3_DIM_SYM_seed1/last.pt'
    stencil = torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state_dict']['stencil']
    assert stencil.shape == (32, 2) and stencil.dtype == torch.float32
    checked = []
    for path in [*(data.dim / f'runs/N3_DIM_SYM_seed{s}/last.pt' for s in [1, 2, 3]),
                 *(args.cache_dir / f'fits/{a}_seed{s}/last.pt' for s in [1, 2, 3] for a in ['GEO', 'PERM'])]:
        st = torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']['stencil']
        assert torch.equal(st, stencil)
        checked.append(dict(path=str(path), sha256=sha(path), stencil_equal=True))
    frames = [('SYNTH_HELDOUT', [data.source_frame(r) for r in data.eval_rows]), ('REAL_DEV', data.real_frames())]
    raw_counts = collections.Counter()
    original_cv = {}
    for name in ['solvePnP', 'solvePnPGeneric', 'solvePnPRefineLM']:
        if hasattr(cv2, name):
            original_cv[name] = getattr(cv2, name)
            def counted(*a, _name=name, **kw):
                raw_counts[_name] += 1
                return original_cv[_name](*a, **kw)
            setattr(cv2, name, counted)
    real_banks, generation_seconds = {}, 0.
    start = time.monotonic()
    for f in frames[1][1]:
        if f['q'] is None:
            b = dict(points=np.full((1, 9, 2), np.nan), hypotheses=['NoOp'], details=[], reason='NO_DETECTION')
        else:
            _, b = bank_job((0, f['q'], f['K'], f['xyz'], f['raw_hw'], f['point_valid']))
        real_banks[f['id']] = b
    generation_seconds = time.monotonic() - start
    for name, fn in original_cv.items():setattr(cv2, name, fn)
    dest = args.cache_dir / 'sampling_masks'; dest.mkdir(exist_ok=True)
    saved, frame_rows = [], []
    for split, fs in frames:
        packed = {arm:np.zeros((len(fs), 8, 200, 4), dtype='uint8') for arm in ['GEO', 'PERM']}
        support_rows = np.zeros((len(fs), 8), bool); counts = np.zeros(len(fs), dtype='uint16')
        for index, f in enumerate(fs):
            bank = store.get(f['cache_row']) if split == 'SYNTH_HELDOUT' else real_banks[f['id']]
            n = len(bank['points']); counts[index] = n
            if split == 'SYNTH_HELDOUT':
                row=f['cache_row']; points=data.arrays['points'][row]; box=data.arrays['boxes'][row]; hw=data.arrays['input_shape'][row]
            else:
                c=f['captured']['captured']; i=c['selected_index']
                if i is None:
                    frame_rows.append(dict(split=split,id=f['id'],actions=1,status='NO_HEAD_NO_DETECTION',source_bank_reused=False));continue
                points=(f['q']*f['scale']+f['offset']).astype('float32'); points[~f['point_valid']]=f['q'][~f['point_valid']]
                box=(np.array(c['candidates'][i]['box_xyxy']).reshape(2,2)*f['scale']+f['offset']).reshape(4).astype('float32');hw=c['input_shape']
            arm_results={}
            for arm in ['GEO','PERM']:
                b=bank if arm=='GEO' else permute_bank(bank,f['id'])
                q=network_bank(b,f,points);m=max(2,n);pad=np.repeat(q[:1],m,axis=0);pad[:n]=q
                inside,support,valid=masks(points,box,f['point_valid'],hw,pad,np.arange(m)<n,stencil,protocol['config']['stencil_fraction'])
                if n>1:packed[arm][index,:,:n-1]=np.packbits(inside,axis=-1,bitorder='little')
                arm_results[arm]=(inside,support)
                assert np.array_equal(valid,f['point_valid'])
            g,gs=arm_results['GEO'];pm,ps=arm_results['PERM']
            assert np.array_equal(gs,ps)
            if n>1:
                permutations=permute_bank(bank,f['id'])['permutation']
                for corner in range(8):assert np.array_equal(pm[corner],g[corner,permutations[corner,1:]-1])
            support_rows[index]=gs
            frame_rows.append(dict(split=split,id=f['id'],actions=n,point_valid=valid[:8].tolist(),common_point_support=gs.tolist(),
                outside_stencil_samples_GEO=int((~g).sum()),inside_stencil_samples_GEO=int(g.sum()),total_stencil_samples=int(g.size),
                unsupported_corners=int((~gs).sum()),GEO_PERM_mask_multiset_equal=True,
                source_bank_reused=split=='SYNTH_HELDOUT',status='HEAD_MASK_RECONSTRUCTED' if n>1 else 'NO_HEAD_EXACT_NOOP'))
        for arm in ['GEO','PERM']:
            path=dest/f'{split}_{arm}_inside_packed.npy';np.save(path,packed[arm]);saved.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,shape=list(packed[arm].shape)))
        for name,value in [('point_support',support_rows),('action_counts',counts)]:
            path=dest/f'{split}_{name}.npy';np.save(path,value);saved.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,shape=list(value.shape)))
    bank_path=dest/'REAL_DEV_generated_banks.npz'
    banks=list(real_banks.values());maxn=max(len(b['points']) for b in banks);q=np.stack([np.concatenate([b['points'],np.repeat(b['points'][:1],maxn-len(b['points']),axis=0)]) for b in banks])
    np.savez_compressed(bank_path,ids=np.array(list(real_banks)),points=q,counts=np.array([len(b['points']) for b in banks]))
    saved.append(dict(path=str(bank_path),sha256=sha(bank_path),bytes=bank_path.stat().st_size))
    cell_files,cell_metadata=sampler_cells(data,frames,dest);saved+=cell_files
    for r in frame_rows:r['feature_cell_padding_masks']=cell_metadata[(r['split'],r['id'])]
    result=dict(schema='newly_defined_handoff_A_static_sampling_mask_receipt_v1',status='DONE',
        reconstruction_scope='CPU FP32 coordinate algebra; original executed GPU tensors were not preserved; no feature values or CNN scores reconstructed',
        inputs='saved source bank + raw captured prediction/box/valid mask/input_hw/affine + frozen identical32-point stencil; no GT visibility or reference branch used',
        padding='inside input pixel bounds, multiplied before patch-body evidence; stored feature-cell masks use centre=(index+.5)*stride < inputHW; bilinear grid_sample padding_mode=zeros/align_corners=False; no new padding policy',
        stored_mask_layout='uint8 [frame,corner8,nonNoOpAction200,packed32Stencil4]; little bit order; use action_counts-1 valid slots; NoOp has pooled evidence, not its own stencil action',
        same_GEO_PERM_point_mask_multiset=True, all9_checkpoint_stencils_equal=True,checked_checkpoints=checked,
        source_initialization_PnP_calls=0,real_initialization=dict(frames=319,cv2_calls=dict(raw_counts),seconds_wall=generation_seconds,
            reason='real full banks were only in original evaluator memory; rebuild prediction-only initialization once and save for future reuse'),
        new_CNN_forward_calls=0,new_final_F_scoring_calls=0,new_optimizer_updates=0,
        frame_rows=frame_rows,external_mask_files=saved,protocol_sha256=sha(DOC/'A_protocol.json'),bank_binding=store.binding,
        source_bank_sha256=sha(args.cache_dir/'source_banks.npy'),code_sha256=sha(Path(__file__)),
        resume_guard=guard,seconds_wall=time.monotonic()-begin,seconds_process_cpu=time.process_time()-cpu)
    write(receipt_path,finite_json(result))
    print(json.dumps(dict(status='DONE',frames=len(frame_rows),real_initialization_cv2_calls=dict(raw_counts),CNN_forward=0,final_F=0,optimizer=0,seconds_wall=result['seconds_wall'])))

if __name__=='__main__':main()
