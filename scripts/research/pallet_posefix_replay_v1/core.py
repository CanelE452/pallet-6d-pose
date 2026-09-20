"""New-root-only replay control. Historical models/data are never overwritten."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts.research.pallet_posefix_large_error_v1 import core as C

ROOT, E, F = C.ROOT, C.E, C.F
HERE = Path(__file__).resolve().parent
DOC = ROOT/'_docs/experiments/pallet_posefix_replay_v1'
RAW = ROOT/'data/pallet/results/pallet_posefix_replay_v1'
OUT = ROOT/'outputs/pallet_posefix_replay_v1'


def write(path, obj):
    path=Path(path).resolve()
    assert any(path.is_relative_to(p) for p in (DOC,RAW,OUT)),path
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.pending')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def freeze(path,obj):
    if path.exists(): assert E.read(path)==obj,('Immutable artifact differs',str(path))
    else: write(path,obj)


def save(path,obj):
    assert Path(path).resolve().is_relative_to(RAW)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.pending.pt'); torch.save(obj,temp); temp.replace(path)


def setup():
    import cv2
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False


def array_sha(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256(); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()


def source_cache_signature(data,rows):
    keys=('points','boxes','point_valid','input_shape','gt_points','gt_valid','matched','record_index','gain')
    return {k:array_sha(data.arrays[k][rows]) for k in keys if k in data.arrays}


def make_real_order(real):
    from scripts.research.pallet_posefix_large_error_v1.train import corrupted
    rng=np.random.default_rng(6401); ids=[]; points=[]; valid=[]
    for _ in range(300):
        ii=rng.integers(0,len(real),8)
        rows=[corrupted(real[int(i)],rng) for i in ii]
        ids.append(ii);points.append(np.stack([r['points'] for r in rows]));valid.append(np.stack([r['valid'] for r in rows]))
    return dict(real_rows=np.asarray(ids),real_points=np.asarray(points),real_valid=np.asarray(valid))


def prepare():
    from scripts.research.pallet_posefix_large_error_v1 import train as O
    from scripts.research.pallet_posefix_replay_v1.source import SourceData
    F.checked_lock(); old=E.read(C.DOC/'PROTOCOL.json')
    C.verify_bindings(old['sources'])
    for b in old['code'].values():F.verify(b)
    oldsupport=E.bound(C.DOC/'TRAIN_SUPPORT.json'); real=O.items();F.verify(oldsupport)
    source=SourceData(); data=source.data
    rng=np.random.default_rng(7101)
    pool=np.sort(rng.choice(source.train_rows,2048,replace=False))
    source_order=rng.choice(pool,(300,8))
    held=np.sort(np.random.default_rng(7102).choice(source.held_rows,256,replace=False))
    assert not set(pool)&set(held)
    orders=dict(**make_real_order(real),source_rows=source_order,source_pool=pool,held_rows=held)
    path=RAW/'ORDERS.npz';RAW.mkdir(parents=True,exist_ok=True)
    if path.exists():
        z=np.load(path)
        assert set(z.files)==set(orders)
        for k,v in orders.items():assert np.array_equal(z[k],v),k
    else: np.savez(path,**orders)
    selected=np.unique(np.r_[source_order.ravel(),held])
    records=[]
    for row in selected:
        r=data.source['records'][int(data.indices[int(row)])]
        image=E.bound(Path(r['image']))
        assert image['sha256']==r['image_sha256'],r['id']
        records.append(dict(row=int(row),id=r['id'],partition=r['partition'],image=image))
    inputlock=dict(orders=E.bound(path),real_ids=[r['id'] for r in real],real_support=oldsupport,
        real_order_and_corruption='Exact original default_rng(6401) interleaved index/noise draws; synthetic RNG completely separate',
        real_array_hashes={k:array_sha(v) for k,v in orders.items() if k.startswith('real_')},
        source_train_rows=len(source.train_rows),source_pool=2048,source_train_exposures=2400,
        source_unique_train_rows=len(np.unique(source_order)),source_eval_rows=256,
        source_eval_partition='heldout only, matched and GT-supported; historical R0 development exposure not excluded',
        source_images=records,source_cache_values=source_cache_signature(data,selected),
        cache_bindings=[E.bound(data.directory/p) for p in ('CACHE_MANIFEST.json','CACHE_COMPLETE.json')]+[E.bound(data.run_dir/'SOURCE_MANIFEST.json')],
        train_eval_row_overlap=False,eval_images_in_training=False)
    freeze(DOC/'INPUT_LOCK.json',inputlock)
    bindings=C.source_bindings()
    for p in [C.DOC/'PROTOCOL.json',C.DOC/'FIT.json',C.RAW/'last300.pt',C.RAW/'TRAIN_STEPS.json',
              C.DOC/'EXISTING_RESULTS.json',C.DOC/'FINETUNED_RESULTS.json',
              C.RAW/'EXISTING_PER_FRAME_METRICS.json',C.RAW/'FINETUNED_PER_FRAME_METRICS.json',
              C.HERE/'core.py',C.HERE/'train.py',C.HERE/'evaluate.py',C.HERE/'visualize.py',
              HERE.parent/'pallet_line_pose_v1/train.py']:
        bindings[str(p.relative_to(ROOT))]=E.bound(p)
    protocol=dict(experiment='pallet_posefix_replay_v1',seed=1,steps=300,
        purpose='Single preservation-control screen; conditional selective self-training only after all preregistered gates',
        initialization='Original synthetic-trained PoseFix PRIOR1 last6000, never the real-only last300',
        architecture='Unchanged official-network-validated PoseFixPallet9 RGB ResNet152, all parameters trainable',
        batch=dict(real=8,synthetic=8,micro=2,real_exposures=2400,synthetic_exposures=2400),
        optimizer=dict(name='TFAdam',lr=1e-4,gradient_clip=None),BN='frozen running stats, trainable affine; same as real-only control',
        objective='mean_real(heatmap_CE+coordinate_L1) + 1.0 * mean_source(heatmap_CE+coordinate_L1) + ONE graph_L2; real term and L2 weight unchanged from real-only',
        comparison='Same real indices, perturbations, loss, seed, initialization, updates. Adds source data term and source exposures: NOT compute matched. Expected approximately 2x forward/backward compute.',
        corruption='Real: bit-exact original 50% raw R0 /50% manual GT+noise. Source: same input corruption rule on source GT valid corners using independent RNG7103. Normal and perturbed source replay are one intervention; their separate effects not isolated.',
        source_noise_rng=7103,source_probe_noise_rng=7104,
        supervision='Real only 38 verified manual corners in9images; synthetic actualGT all supported corners; centroid excluded; no unknown/PnP labels, pseudo labels, teacher anchor or new loss',
        inputs='RGB plus R0 initial points and predicted box; no depth/CAD/PnP/dimensions at inference',
        evaluation='Fixed DEV72 + GREEN150; identical frozenR0 detections; approved whole-object symmetry and fixedGTdenominator. Raw main; .01image-diagonal cap safety diagnostic only, no selecting best cap.',
        source_probe='256 disjoint replay-heldout source rows, original R0 input and one independent forced-corruption draw per row. Fixed index, supported GT, original prepared-image pixels; before/original real-only/replay.',
        promotion_gate=dict(trainability='same noisy_manual mean<=10px, recovery>=.5 and mean<=.5input_mean',
            source_clean_PCK10_drop_max_pp=1.0,source_clean_P90_ratio_max=1.10,
            DEV_raw_min_recovered_hard_corners=5,raw_good5_to_bad10_rate_max=.01,
            raw_PCK10_vs_N2_drop_max_pp=.5,raw_P90_vs_N2_ratio_max=1.05,
            populations=['DEV72','GREEN150_MANUAL'],
            note='Every gate required. GREEN matched hard only4 -> no standalone recovery-rate gate; safety and full metrics required. Thresholds are screening tolerances, not formal noninferiority evidence.'),
        conditional_stage2='Only if all gates pass: design/lock corner-selective pseudo supervision on disjoint unlabeled TRAIN data. Otherwise DO_NOT_START_SELFTTRAIN and report failed gates; no rescue sweeps.',
        checkpoint='last300 only; resumable fixed-order checkpoint each100 steps, no eval-based selection',
        evaluation_reused=True,independent_confirmation=False,auto_promote=False,
        gpu_temperature_limit_C=80,system_changes=False,original_release_untouched=True,
        bindings=bindings,input_lock=E.bound(DOC/'INPUT_LOCK.json'),
        code={p.name:E.bound(p) for p in [HERE/n for n in ('core.py','train.py','source.py','evaluate.py','decision.py','test_replay.py','test_source.py')]})
    freeze(DOC/'PROTOCOL.json',protocol)
    print('PREPARED',DOC,flush=True)


def verify():
    p=E.read(DOC/'PROTOCOL.json'); C.verify_bindings(p['bindings']);F.verify(p['input_lock'])
    for b in p['code'].values():F.verify(b)
    inputs=E.read(DOC/'INPUT_LOCK.json');F.verify(inputs['orders']);F.verify(inputs['real_support'])
    for b in inputs['cache_bindings']:F.verify(b)
    F.checked_lock();return p


def load_model():
    verify();fit=E.read(DOC/'FIT.json');assert fit['complete'] and fit['step']==300
    F.verify(fit['checkpoint']);ck=torch.load(ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)
    assert ck['protocol_sha256']==E.sha(DOC/'PROTOCOL.json') and ck['step']==300
    model=C.PoseFixPallet9();model.load_state_dict(ck['model_state_dict'])
    return model.cuda().eval().requires_grad_(False)
