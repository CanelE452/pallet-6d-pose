"""Separate v2 criterion/surgery with the exact frozen v1 dataset and epoch trainer."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import torch
from ultralytics.utils.torch_utils import unwrap_model
from scripts.research.pallet_dht_joint_v1 import train as V1
from .integration import ARMS, DIAGNOSTIC_STEPS, R0_PATH, R0_SHA256, build_model
from .gradient_surgery import project_from_sum, group_statistics, mask_digest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE_MANIFEST, SOURCE_SHA256=V1.SOURCE_MANIFEST,V1.SOURCE_SHA256
sha, read, canonical, json_write, torch_write=V1.sha,V1.read,V1.canonical,V1.json_write,V1.torch_write
ManifestDataset, EpochSampler, seed_for=V1.ManifestDataset,V1.EpochSampler,V1.seed_for


def training_sources():
    import ultralytics.utils.loss, ultralytics.nn.modules.head, ultralytics.engine.trainer
    return [*[HERE/name for name in ['integration.py','train.py','gradient_surgery.py','incidence.py']],
        *[V1.HERE/name for name in ['integration.py','train.py','hough_block.py','line_targets.py']],
        ROOT/'scripts/research/deep_hough_side_v1/dht.py',
        *[Path(m.__file__) for m in [ultralytics.utils.loss,ultralytics.nn.modules.head,ultralytics.engine.trainer]]]


class CouplingTrainer(V1.JointTrainer):
    def __init__(self, **kwargs):
        self.surgery_rows=[];self.diagnostic_rows=[];self.shared_masks={};self.resource_rows=[]
        super().__init__(**kwargs)

    def get_model(self,cfg=None,weights=None,verbose=True):
        return build_model(self.cell_config['arm'],R0_PATH,self.cell_config['line_weight'],
            self.cell_config['incidence_weight'],verbose=verbose)

    def get_validator(self):
        result=super().get_validator()
        self.loss_names+=('incidence_loss',)
        return result

    def validate(self):
        # Stock validation uses EMA and does not update its criterion schedule.
        # Report auxiliary validation loss with THIS training epoch's weight;
        # predictions and stock logged one2one components remain unchanged.
        model=self.ema.ema if self.ema else unwrap_model(self.model)
        if model.criterion is None:model.criterion=model.init_criterion()
        criterion=model.criterion
        criterion.capture_gradients=False
        criterion.stock.updates=self.epoch
        criterion.stock.o2m=criterion.stock.decay(self.epoch)
        criterion.stock.o2o=1.-criterion.stock.o2m
        return super().validate()

    def preprocess_batch(self,batch):
        if self.model.training and self.cell_config['stage']=='smoke':
            if self.device.type=='cuda':torch.cuda.synchronize()
            self.batch_started=time.perf_counter()
        value=super().preprocess_batch(batch)
        if self.model.training:
            model=unwrap_model(self.model)
            if model.criterion is None:model.criterion=model.init_criterion()
            criterion=model.criterion
            criterion.next_step=self.optimizer_steps+1
            criterion.capture_gradients=(self.cell_config['arm'] in ('pcgrad','balanced_pcgrad')
                or criterion.next_step in DIAGNOSTIC_STEPS)
        return value

    def optimizer_step(self):
        if self.amp or self.accumulate!=1:
            raise RuntimeError('PCGrad integration requires FP32 and one batch per optimizer step')
        gradients=[p.grad for p in self.model.parameters() if p.grad is not None]
        global_norm=torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gradients)))
        if not bool(torch.isfinite(global_norm)):
            raise FloatingPointError('Nonfinite full-model gradient before clipping')
        criterion=unwrap_model(self.model).criterion
        pending=criterion.pending
        if pending is not None:
            total=[p.grad for p in criterion.parameters]
            stat_total=total
            if self.cell_config['arm']=='incidence' and pending.get('main_gradient') is not None:
                stat_total=[(a+b if a is not None and b is not None else a if b is None else b)
                    for a,b in zip(pending['main_gradient'],pending['auxiliary'])]
            adjusted,stats,mask,main=project_from_sum(stat_total,pending['auxiliary'],pending['main_present'],
                enabled=self.cell_config['arm'] in ('pcgrad','balanced_pcgrad'))
            if self.cell_config['arm']=='incidence':adjusted=total
            selected,digest=mask_digest(criterion.names,mask)
            self.shared_masks[digest]=selected
            row=dict(optimizer_step=self.optimizer_steps+1,epoch=self.epoch+1,
                shared_mask_sha256=digest,combined_gradient_norm_before_surgery=float(global_norm),
                **pending['scalars'],**stats)
            self.surgery_rows.append(row)
            if pending['pose'] is not None:
                reference=pending['main_gradient']
                equivalence=None
                if self.cell_config['arm']!='incidence':
                    errors=[(a-b).detach() for a,b,keep in zip(main,reference,mask) if keep]
                    ref=[b.detach() for b,keep in zip(reference,mask) if keep]
                    err2=float(torch.stack([e.float().square().sum() for e in errors]).sum()) if errors else 0.
                    ref2=float(torch.stack([e.float().square().sum() for e in ref]).sum()) if ref else 0.
                    count=sum(e.numel() for e in errors)
                    equivalence=dict(max_abs_delta=max((float(e.abs().max()) for e in errors),default=0.),
                        rms=(err2/max(count,1))**.5,relative_l2=(err2/ref2)**.5 if ref2 else None,
                        values=count,numerical_identity_required=False,
                        note='Independent same-graph stock backward versus total-minus-aux; CUDA sparse reduction roundoff may differ.')
                main=reference
                posemask=[a is not None and b is not None for a,b in zip(pending['pose'],pending['auxiliary'])]
                self.diagnostic_rows.append(dict(**row,
                    main_gradient_reconstruction=equivalence,
                    full_stock_vs_weighted_line=group_statistics(criterion.names,main,pending['auxiliary'],mask),
                    pose_location_RLE_vs_weighted_line=group_statistics(criterion.names,pending['pose'],pending['auxiliary'],posemask)))
            for p,g in zip(criterion.parameters,adjusted):
                p.grad=g
            criterion.pending=None
        super().optimizer_step()
        if self.cell_config['stage']=='smoke':
            if self.device.type=='cuda':torch.cuda.synchronize()
            self.resource_rows.append(dict(optimizer_step=self.optimizer_steps,
                diagnostic_step=self.optimizer_steps in DIAGNOSTIC_STEPS,
                batch_to_updated_EMA_ms=1000*(time.perf_counter()-self.batch_started),
                allocated_gpu_bytes=torch.cuda.memory_allocated() if self.device.type=='cuda' else 0))

    def resume_training(self,ckpt):
        super().resume_training(ckpt)
        if not self.resume_path:return
        self.surgery_rows=ckpt.get('surgery_rows',[])
        self.diagnostic_rows=ckpt.get('task_gradient_diagnostics',[])
        self.shared_masks=ckpt.get('shared_masks',{})
        self.resource_rows=ckpt.get('resource_rows',[])

    def save_model(self):
        super().save_model()
        checkpoint=torch.load(self.last,map_location='cpu')
        checkpoint.update(surgery_rows=self.surgery_rows,task_gradient_diagnostics=self.diagnostic_rows,
            shared_masks=self.shared_masks,resource_rows=self.resource_rows)
        torch_write(self.last,checkpoint)
        if self.best_fitness==self.fitness:torch_write(self.best,checkpoint)
        if checkpoint['complete']:torch_write(self.wdir/'final.pt',checkpoint)
        if self.epoch==0:torch_write(self.wdir/'epoch_1.pt',checkpoint)
        if self.cell_config['stage']=='smoke':
            json_write(self.cell_dir/'RESOURCE_BENCHMARK.json',dict(complete=checkpoint['complete'],
                batches=self.resource_rows,peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated() if self.device.type=='cuda' else 0,
                precision='FP32',batch=self.expected_batch,
                includes='preprocessing/forward/loss/task gradients/backward/surgery/clipping/optimizer/EMA with GPU synchronization',
                excludes='disk decode/dataloader wait, validation, checkpoint serialization'))
        json_write(self.cell_dir/'GRADIENT_AUDIT.json',dict(schema='pallet_dht_coupling_gradient_audit_v2',
            complete=checkpoint['complete'],arm=self.cell_config['arm'],
            surgery_enabled=self.cell_config['arm'] in ('pcgrad','balanced_pcgrad'),
            first16_preclip_probes_after_optional_surgery=self.gradient_probe,actual_task_diagnostics=self.diagnostic_rows,
            surgery_steps=self.surgery_rows,shared_masks=self.shared_masks,
            interpretation='Actual training graph; no extra forward/BN/RNG. TaskA all stock components, taskB weighted line. Pose+RLE separately measured at fixed steps. Shared structural intersection only; summed symmetric projection before original global clipping.'))
        return True


def make_config(args,records):
    protocol=args.run_dir/'TRAIN_PROTOCOL.json'
    if not args.smoke_limit and not protocol.exists():raise RuntimeError('Main training requires frozen TRAIN_PROTOCOL.json')
    if sha(R0_PATH)!=R0_SHA256 or sha(SOURCE_MANIFEST)!=SOURCE_SHA256:raise RuntimeError('Frozen R0/source SHA mismatch')
    required_sources = training_sources()
    if not all(p.exists() for p in required_sources):raise RuntimeError('All bound integration sources must exist before training')
    config=dict(schema='pallet_dht_coupling_cell_v2',stage='smoke' if args.smoke_limit else 'main',arm=args.arm,seed=args.seed,
        epochs=args.epochs,batch=args.batch,lr=args.lr,optimizer=args.optimizer,line_weight=args.line_weight,incidence_weight=args.incidence_weight,
        workers=args.workers,amp=args.amp,amp_init_scale=args.amp_init_scale,smoke_limit=args.smoke_limit,train_frames=len(records['train']),val_frames=len(records['val']),
        training_recipe=dict(imgsz=640,rect=False,mosaic=0.,mixup=0.,cutmix=0.,copy_paste=0.,fliplr=0.,flipud=0.,
            scale=.25,translate=.1,degrees=0.,shear=0.,perspective=0.,hsv_h=.015,hsv_s=.5,hsv_v=.35,
            nbs=args.batch,lr_final_fraction=args.lrf,warmup_epochs=args.warmup_epochs,warmup_bias_lr=args.warmup_bias_lr,weight_decay=.0005,
            momentum=.937,cos_lr=True,freeze=None,close_mosaic=0,multi_scale=0.0),
        source_code_sha256={str(p):sha(p) for p in required_sources})
    if protocol.exists():
        registered=read(protocol).get('training',{})
        if not args.smoke_limit:
            for k in ['epochs','batch','lr','optimizer','line_weight','incidence_weight','workers']:
                if k in registered and registered[k]!=config[k]:raise RuntimeError('Protocol/CLI mismatch: '+k)
            for k,actual in [('lrf',args.lrf),('warmup_epochs',args.warmup_epochs),('warmup_bias_lr',args.warmup_bias_lr),('nbs',args.batch),('amp',args.amp),('amp_init_scale',args.amp_init_scale)]:
                if k in registered and registered[k]!=actual:raise RuntimeError('Protocol/CLI mismatch: '+k)
            if 'seeds' in registered and args.seed not in registered['seeds']:raise RuntimeError('Unregistered seed')
    binding=dict(arm=args.arm,seed=args.seed,baseline_sha256=R0_SHA256,source_manifest_sha256=SOURCE_SHA256,
        protocol_sha256=sha(protocol) if protocol.exists() else None,
        code_sha256=config['source_code_sha256'])
    binding['config_sha256']=hashlib.sha256(canonical(config).encode()).hexdigest()
    config['bindings']=binding
    return config


def run(args):
    args.run_dir=args.run_dir.resolve()
    if not (args.run_dir/'PURPOSE.md').exists():raise RuntimeError('Result root must contain PURPOSE.md')
    if args.amp: raise ValueError('This experiment requires FP32 training')
    if args.epochs<1 or args.batch<1 or args.workers<0 or args.lr<=0:raise ValueError('Positive budget required')
    if args.smoke_limit not in (0,32,128):raise ValueError('Smoke subset must be32 or128')
    source=read(SOURCE_MANIFEST)
    records={'train':[r for r in source['records'] if r['partition']=='train'],
             'val':[r for r in source['records'] if r['source_split']=='val']}
    if len(records['train'])!=55980 or len(records['val'])!=4020:raise RuntimeError('Canonical source denominator changed')
    if args.smoke_limit:
        records['train']=records['train'][:args.smoke_limit];records['val']=records['val'][:min(32,args.smoke_limit)]
    config=make_config(args,records)
    if not args.smoke_limit and (args.epochs != 2 or args.batch != 16 or args.lr != .0001 or len(records['train']) != 55980):
        raise RuntimeError('Main budget differs from the matched v1 experiment')
    cell=args.run_dir/('smoke/runs' if args.smoke_limit else 'runs')/f'{args.arm}_seed{args.seed}'
    cell.mkdir(parents=True,exist_ok=True)
    cp=cell/'CELL_CONFIG.json'
    if cp.exists():
        if read(cp)!=config:raise RuntimeError('Cell configuration/source changed')
        if (cell/'COMPLETION.json').exists():
            done=read(cell/'COMPLETION.json')
            if done.get('complete') and done.get('PASS'):
                if sha(cell/'weights/final.pt')!=done['checkpoint_sha256']:raise RuntimeError('Completed checkpoint SHA mismatch')
                print(canonical(done));return
        if not args.resume:raise RuntimeError('Cell already exists and is unfinished; use --resume')
    else:
        if args.resume:raise RuntimeError('Cannot resume an uncreated cell')
        json_write(cp,config)
    overrides=dict(model=str(R0_PATH),data='frozen-source-manifest.yaml',task='pose',epochs=args.epochs,batch=args.batch,
        imgsz=640,seed=args.seed,device=args.device,workers=args.workers,project=str(cell.parent),name=cell.name,
        exist_ok=True,pretrained=True,resume=False,optimizer=args.optimizer,lr0=args.lr,lrf=args.lrf,
        weight_decay=.0005,momentum=.937,nbs=args.batch,warmup_epochs=args.warmup_epochs,
        warmup_momentum=.8,warmup_bias_lr=args.warmup_bias_lr,cos_lr=True,
        mosaic=0.,mixup=0.,cutmix=0.,copy_paste=0.,close_mosaic=0,fliplr=0.,flipud=0.,
        scale=.25,translate=.1,degrees=0.,shear=0.,perspective=0.,hsv_h=.015,hsv_s=.5,hsv_v=.35,
        multi_scale=0.,rect=False,single_cls=True,freeze=None,patience=0,amp=args.amp,plots=False,
        deterministic=True,cache=False,val=True,save=True,save_period=-1,compile=False,
        pose=12.,kobj=1.,box=7.5,cls=.5,dfl=1.5,rle=1.)
    try:
        trainer=CouplingTrainer(cell_config=config,records=records,cell_dir=cell,
            resume_path=cell/'weights/last.pt' if args.resume else None,overrides=overrides)
        trainer.train()
        final=cell/'weights/final.pt'
        ckpt=torch.load(final,map_location='cpu')
        if not ckpt['complete'] or ckpt['optimizer_steps']!=ckpt['expected_optimizer_steps']:raise RuntimeError('Final checkpoint incomplete')
        if read(cp)!=config:raise RuntimeError('Cell config changed during training')
        for p,h in config['source_code_sha256'].items():
            if sha(p)!=h:raise RuntimeError('Training source changed during run: '+p)
        if config['bindings']['protocol_sha256'] and sha(args.run_dir/'TRAIN_PROTOCOL.json')!=config['bindings']['protocol_sha256']:
            raise RuntimeError('Training protocol changed during run')
        completion=dict(schema='pallet_dht_coupling_training_completion_v2',complete=True,PASS=True,stage=config['stage'],
            arm=args.arm,seed=args.seed,epochs_completed=ckpt['epoch']+1,expected_epochs=args.epochs,
            optimizer_steps=ckpt['optimizer_steps'],expected_optimizer_steps=ckpt['expected_optimizer_steps'],
            train_frames=len(records['train']),val_frames=len(records['val']),checkpoint=str(final),checkpoint_sha256=sha(final),
            checkpoint_uses_final_epoch=True,EMA_available=True,config_sha256=config['bindings']['config_sha256'],
            cell_config_file_sha256=sha(cp),bindings=config['bindings'],
            history_sha256=sha(cell/'history.json'),batch_trace_sha256=sha(cell/'BATCH_TRACE.jsonl'),
            final_parameter_BN_changes=ckpt['history'][-1]['parameter_BN_changes'],
            resume_semantics='exact completed-epoch restart; interrupted partial epoch is replayed',new_real_inference=False)
        json_write(cell/'COMPLETION.json',completion);print(canonical(completion))
    except BaseException as exc:
        probes=getattr(locals().get('trainer'),'gradient_probe',[])
        probes=json.loads(json.dumps(probes),parse_constant=lambda value:value)
        json_write(cell/'FAILURE.json',dict(complete=False,PASS=False,type=type(exc).__name__,message=str(exc),
            successful_optimizer_steps=getattr(locals().get('trainer'),'optimizer_steps',0),gradient_probe=probes,config=config['bindings']))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--seed',type=int,required=True);parser.add_argument('--epochs',type=int,required=True)
    parser.add_argument('--batch',type=int,required=True);parser.add_argument('--lr',type=float,required=True)
    parser.add_argument('--optimizer',choices=['SGD','AdamW'],default='AdamW');parser.add_argument('--line-weight',type=float,default=.1)
    parser.add_argument('--incidence-weight',type=float,default=.1)
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--warmup-epochs',type=float,default=.1)
    parser.add_argument('--lrf',type=float,default=.1);parser.add_argument('--warmup-bias-lr',type=float,default=0.)
    parser.add_argument('--device',default='0');parser.add_argument('--amp',action='store_true');parser.add_argument('--amp-init-scale',type=float,default=16.);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke-limit',type=int,default=0)
    run(parser.parse_args())
