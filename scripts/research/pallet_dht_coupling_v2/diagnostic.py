"""Predeclared GPU diagnostic on the already frozen v1 synthetic batch.

This does not import the old CPU runner or bypass/change its RAM guard. It has
its own immutable protocol, reads unchanged v1 final raw weights, and performs
one disposable train-mode forward per seed with no optimizer or checkpoint save.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from scripts.research.pallet_dht_joint_v1 import train as V1
from scripts.research.pallet_dht_joint_v1.line_targets import auxiliary_line_loss
from .gradient_surgery import group_statistics, mask_digest

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'data/pallet/results/pallet_dht_joint_v1'
OLD_DIAG=OLD/'provenance/gradient_diagnostic_v1'
sha,read,json_write=V1.sha,V1.read,V1.json_write


def source_hashes():
    import ultralytics.utils.loss,ultralytics.nn.modules.head
    return {str(p):sha(p) for p in [Path(__file__).resolve(),Path(__file__).with_name('gradient_surgery.py'),
        *[V1.HERE/n for n in ['train.py','integration.py','hough_block.py','line_targets.py']],
        Path(ultralytics.utils.loss.__file__),Path(ultralytics.nn.modules.head.__file__)]}


def location(run_dir):return Path(run_dir).resolve()/'provenance/gradient_diagnostic_gpu_v2'


def prepare(run_dir):
    root=Path(run_dir).resolve()
    if not (root/'PURPOSE.md').exists():raise RuntimeError('Missing result PURPOSE.md')
    out=location(root);out.mkdir(parents=True,exist_ok=True)
    protocol_path=out/'PROTOCOL.json'
    if protocol_path.exists():raise RuntimeError('Diagnostic already frozen')
    old=read(OLD_DIAG/'PROTOCOL.json')
    sample=OLD_DIAG/'SAMPLE_MANIFEST.json'
    assert sha(sample)==old['sample_manifest_sha256']
    (out/'SAMPLE_MANIFEST.json').write_bytes(sample.read_bytes())
    runs=[]
    for seed in [1,2,3]:
        cell=OLD/'runs'/f'hough_joint_seed{seed}'
        done=read(cell/'COMPLETION.json');path=cell/'weights/final.pt'
        assert done['complete'] and done['PASS'] and done['optimizer_steps']==6998
        assert sha(path)==done['checkpoint_sha256']
        runs.append(dict(seed=seed,checkpoint=str(path),checkpoint_sha256=sha(path),
            completion=str(cell/'COMPLETION.json'),completion_sha256=sha(cell/'COMPLETION.json')))
    protocol=dict(schema='pallet_dht_coupling_gpu_gradient_protocol_v2',complete=True,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        device='cuda',precision='float32',batch=16,torch_threads=1,
        samples=16,source_counts={'G38':6,'P0':5,'TEX':5},sample_manifest_sha256=sha(sample),
        previous_cpu_protocol_sha256=sha(OLD_DIAG/'PROTOCOL.json'),
        previous_cpu_deferral_sha256=sha(OLD_DIAG/'DEFERRED_RESOURCE.json'),
        unchanged_v1_training_protocol_sha256=sha(OLD/'TRAIN_PROTOCOL.json'),
        source_manifest_sha256=sha(V1.SOURCE_MANIFEST),source_sha256=source_hashes(),
        runs=runs,dataset_training_args=old['dataset_training_args'],augmentation_seed=90216,
        augmentation_epoch=0,minimum_free_gpu_gib=7.0,
        losses=['all stock E2E components','isolated point-location+RLE','weighted line(.1)'],
        weight_state={'one2many':.1,'one2one':.9,'line':.1},
        surgery_definition='Descriptive gradients only, no projection or update is applied',
        limits=['One fixed source-stratified synthetic batch, not population gradient conflict prevalence.',
            'Raw final training weights and train-mode BN differ from real-evaluated EMA weights.',
            'One disposable forward per seed; no optimizer, checkpoint write, real-image input or selection.',
            'Separate GPU protocol, not a modification or override of the old CPU14GiB guard.',
            'CUDA sparse accumulation has known numerical nondeterminism; norm/cosine are observations.'])
    json_write(protocol_path,protocol)
    print(json.dumps({'protocol':str(protocol_path),'sha256':sha(protocol_path),'GPU_used':False}))


def make_batch(protocol,sample,device):
    rows=sample['records']
    if len(rows)!=16:raise RuntimeError('Frozen diagnostic sample count differs')
    for row in rows:
        if row['source_kind']!='synthetic' or row['source_split']!='val':raise RuntimeError('Not synthetic val')
        for kind in ['image','label']:
            if sha(row[kind])!=row[kind+'_sha256']:raise RuntimeError('Sample source drift')
    hyp=SimpleNamespace(**protocol['dataset_training_args'])
    dataset=V1.ManifestDataset(rows,seed=protocol['augmentation_seed'],
        data=dict(nc=1,names={0:'pallet'},channels=3,kpt_shape=[9,3],flip_idx=[1,0,3,2,5,4,7,6,8]),
        imgsz=640,batch_size=16,augment=True,hyp=hyp,rect=False,cache=False,single_cls=True,
        stride=32,pad=0.,task='pose',fraction=1.)
    batch=dataset.collate_fn([dataset[(0,i)] for i in range(16)])
    labels=hashlib.sha256()
    for key in ['keypoints','bboxes','cls','batch_idx']:
        labels.update(key.encode());labels.update(batch[key].contiguous().numpy().tobytes())
    fingerprint=dict(image_sha256=hashlib.sha256(batch['img'].contiguous().numpy().tobytes()).hexdigest(),
        labels_sha256=labels.hexdigest(),source_indices=list(batch['audit_source_index']),
        augmentation_seeds=list(batch['audit_augmentation_seed']),shape=list(batch['img'].shape))
    expected=read(OLD_DIAG/'PREPARATION_AUDIT.json')['deterministic_actual_training_transform']
    if fingerprint!=expected:raise RuntimeError('Actual batch differs from original frozen preparation')
    batch=DetectionTrainer.preprocess_batch(SimpleNamespace(device=device,args=hyp,stride=32),batch)
    return batch,fingerprint


def run(run_dir,seed):
    out=location(run_dir);p=read(out/'PROTOCOL.json')
    if source_hashes()!=p['source_sha256']:raise RuntimeError('Frozen diagnostic source changed')
    if sha(out/'SAMPLE_MANIFEST.json')!=p['sample_manifest_sha256']:raise RuntimeError('Sample changed')
    if sha(OLD/'TRAIN_PROTOCOL.json')!=p['unchanged_v1_training_protocol_sha256']:raise RuntimeError('Old training protocol changed')
    result_path=out/f'SEED{seed}.json'
    if result_path.exists():raise RuntimeError('Refusing diagnostic overwrite')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0')
    free,_=torch.cuda.mem_get_info(device)
    if free/1024**3<p['minimum_free_gpu_gib']:raise RuntimeError('Insufficient declared GPU memory')
    entry=next(x for x in p['runs'] if x['seed']==seed)
    if sha(entry['checkpoint'])!=entry['checkpoint_sha256']:raise RuntimeError('Frozen checkpoint drift')
    cp=torch.load(entry['checkpoint'],map_location='cpu')
    assert cp['complete'] and cp['epoch']==1 and cp['optimizer_steps']==cp['updates']==6998
    model=cp['model'].float().train().to(device);del cp
    model.criterion=model.init_criterion();model.criterion.update()
    assert abs(model.criterion.stock.o2m-.1)<1e-12 and model.criterion.weight==.1
    batch,fingerprint=make_batch(p,read(out/'SAMPLE_MANIFEST.json'),device)
    named=[(n,v) for n,v in model.named_parameters() if v.requires_grad and (not n.startswith('model.23.') or '.hough.' in n)]
    names,params=zip(*named)
    before={n:v.detach().cpu().clone() for n,v in model.named_parameters()}
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    pred=model(batch['img'])
    base,items=model.criterion.stock(pred,batch)
    raw=pred[1] if isinstance(pred,(tuple,list)) else pred
    line,_=auxiliary_line_loss(raw['line_logits'],batch['keypoints'],batch['batch_idx'],batch['img'].shape[-2:],raw['line_lattice'])
    weighted_line=line*.1*batch['img'].shape[0]
    stock_grad=torch.autograd.grad(base.sum(),params,retain_graph=True,allow_unused=True)
    pose_grad=torch.autograd.grad(base[1]+base[5],params,retain_graph=True,allow_unused=True)
    line_grad=torch.autograd.grad(weighted_line,params,allow_unused=True)
    stock_mask=[a is not None and b is not None for a,b in zip(stock_grad,line_grad)]
    pose_mask=[a is not None and b is not None for a,b in zip(pose_grad,line_grad)]
    stats=group_statistics(names,stock_grad,line_grad,stock_mask)
    pose_stats=group_statistics(names,pose_grad,line_grad,pose_mask)
    shared_names,digest=mask_digest(names,stock_mask)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    assert all(torch.equal(v.detach().cpu(),before[n]) for n,v in model.named_parameters())
    assert sha(entry['checkpoint'])==entry['checkpoint_sha256']
    result=dict(schema='pallet_dht_coupling_gpu_gradient_result_v2',complete=True,PASS=True,
        PASS_semantics='Measurement integrity only; not absence of conflict or evidence of accuracy benefit',seed=seed,
        protocol_sha256=sha(out/'PROTOCOL.json'),checkpoint_sha256=entry['checkpoint_sha256'],
        batch_fingerprint=fingerprint,neural_forward_calls=1,optimizer_steps=0,parameters_unchanged=True,
        full_stock_vs_weighted_line=stats,pose_location_RLE_vs_weighted_line=pose_stats,
        shared_parameter_names=shared_names,shared_mask_sha256=digest,
        loss_components_batch_scaled=base.detach().cpu().tolist(),weighted_line_loss=float(weighted_line.detach()),
        diagnostic_seconds=elapsed,peak_allocated_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
        free_gpu_before_gib=free/1024**3,device=torch.cuda.get_device_name(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,limits=p['limits'])
    json_write(result_path,result)
    paths=[out/f'SEED{s}.json' for s in [1,2,3]]
    if all(path.exists() for path in paths):
        rows=[read(path) for path in paths]
        if any(row['protocol_sha256']!=sha(out/'PROTOCOL.json') for row in rows):raise RuntimeError('Mixed protocol result')
        json_write(Path(run_dir)/'GRADIENT_DIAGNOSIS.json',dict(schema='pallet_dht_coupling_gradient_diagnosis_v2',
            complete=True,PASS=True,PASS_semantics='Three declared gradient measurements completed; no causal performance conclusion',
            protocol_sha256=sha(out/'PROTOCOL.json'),source_results={str(path):sha(path) for path in paths},
            runs=rows,limits=p['limits']))
    print(json.dumps({'path':str(result_path),'stock':stats['shared_all'],'pose':pose_stats['shared_all']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','run'],required=True)
    parser.add_argument('--seed',type=int,choices=[1,2,3])
    args=parser.parse_args()
    if args.phase=='prepare':prepare(args.run_dir)
    elif args.seed is None:parser.error('run requires --seed')
    else:run(args.run_dir,args.seed)
