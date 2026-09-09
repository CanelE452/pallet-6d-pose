"""Matched synthetic-only full-network YOLO26/DHT continuation training.

Run from a results directory containing PURPOSE.md. Source PNGs/labels/caches
are never rewritten. Per-sample augmentation is determined by seed, epoch and
source index, independently of arm or worker scheduling. Resume restores the
last completed epoch (partial epochs are replayed), including full-precision
training/EMA/optimizer state. No real images are used here.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from ultralytics import __version__ as ULTRALYTICS_VERSION
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.utils.torch_utils import ModelEMA, unwrap_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
from integration import ARMS, R0_PATH, R0_SHA256, build_model, snapshot, changes_since
SOURCE_MANIFEST = ROOT / 'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json'
SOURCE_SHA256 = 'feaa24075d31c4227e3397b7450a19a0f1d3a73dc0203d12a8ca5b927eb59789'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
def json_write(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def torch_write(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)
def read(path): return json.loads(Path(path).read_text())
def seed_for(seed, epoch, index):
    return int.from_bytes(hashlib.sha256(f'pallet_dht_joint_v1:{seed}:{epoch}:{index}'.encode()).digest()[:4],'little')


class EpochSampler(Sampler):
    def __init__(self, size, seed, shuffle): self.size,self.seed,self.shuffle,self.epoch=size,seed,shuffle,0
    def __len__(self): return self.size
    def set_epoch(self, epoch): self.epoch=int(epoch)
    def __iter__(self):
        generator=torch.Generator().manual_seed(seed_for(self.seed,self.epoch,0))
        order=torch.randperm(self.size,generator=generator).tolist() if self.shuffle else range(self.size)
        return iter((self.epoch,int(i)) for i in order)


class ManifestDataset(YOLODataset):
    """Use unchanged YOLO transforms with source-file labels, avoiding *.cache writes."""
    def __init__(self, records, seed, **kwargs):
        self.source_records=records;self.audit_seed=seed
        super().__init__(img_path='frozen-source-manifest', **kwargs)
        if self.im_files != [r['image'] for r in records]:
            raise RuntimeError('YOLO loader changed source order or filtered source frames')

    def get_img_files(self, _): return [r['image'] for r in self.source_records]

    def get_labels(self):
        labels=[]
        for r in self.source_records:
            if sha(r['label']) != r['label_sha256']:
                raise RuntimeError('Source label SHA mismatch: '+r['label'])
            rows=[list(map(float,line.split())) for line in Path(r['label']).read_text().splitlines() if line.strip()]
            array=np.asarray(rows,dtype=np.float32).reshape(-1,32)
            expected=np.asarray([[t['class_id'],*t['box_xywh_normalized'],
                *np.asarray(t['keypoints_normalized']).reshape(-1)] for t in r['targets']],dtype=np.float32).reshape(-1,32)
            if not np.array_equal(array,expected): raise RuntimeError('Manifest/file label drift: '+r['label'])
            labels.append(dict(im_file=r['image'],shape=tuple(r['prepared_shape_hw']),cls=array[:,:1],
                bboxes=array[:,1:5],segments=[],keypoints=array[:,5:].reshape(-1,9,3),normalized=True,bbox_format='xywh'))
        self.label_files=[r['label'] for r in self.source_records]
        return labels

    def __getitem__(self, item):
        epoch,index=item if isinstance(item,tuple) else (0,int(item))
        source_index=int(self.source_records[index]['index'])
        seed=seed_for(self.audit_seed,epoch,source_index)
        py_state,np_state,torch_state=random.getstate(),np.random.get_state(),torch.random.get_rng_state()
        random.seed(seed);np.random.seed(seed)
        torch.random.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        # Albumentations may own an RNG independent of Python/NumPy.
        for transform in self.transforms.transforms:
            inner=getattr(transform,'transform',None)
            if hasattr(inner,'set_random_seed'): inner.set_random_seed(seed)
        try: value=super().__getitem__(index)
        finally:
            random.setstate(py_state);np.random.set_state(np_state);torch.random.set_rng_state(torch_state)
        value['audit_source_index']=source_index;value['audit_augmentation_seed']=seed
        return value


class JointTrainer(PoseTrainer):
    def __init__(self, *, cell_config, records, cell_dir, resume_path=None, overrides=None):
        self.cell_config=cell_config;self.records=records;self.cell_dir=Path(cell_dir)
        self.resume_path=Path(resume_path) if resume_path else None
        self.optimizer_steps=0;self.batch_count=0;self.trace_rows=0;self.hist=[];self.gradient_probe=[]
        self.expected_batch=int(cell_config['batch']);self.initial_state=None;self._building_pipeline=0
        super().__init__(overrides=overrides)
        self.add_callback('on_train_epoch_start',self._epoch_start)
        self.add_callback('on_pretrain_routine_end',self._setup_audit)

    def get_dataset(self):
        return dict(train='manifest-train',val='manifest-val',nc=1,names={0:'pallet'},channels=3,
                    kpt_shape=[9,3],flip_idx=[1,0,3,2,5,4,7,6,8])

    def get_model(self,cfg=None,weights=None,verbose=True):
        return build_model(self.cell_config['arm'],R0_PATH,self.cell_config['line_weight'],verbose=verbose)

    def setup_model(self):
        if self.resume_path:
            ckpt=torch.load(self.resume_path,map_location='cpu')
            if ckpt['joint_provenance']!=self.cell_config['bindings']:
                raise RuntimeError('Resume checkpoint provenance mismatch')
            if ckpt.get('complete'): raise RuntimeError('Refusing to resume an already completed training cell')
            self.model=copy.deepcopy(ckpt['model']).float();self.model.criterion=None
            self.resume=True
            return ckpt
        self.model=self.get_model(verbose=True)
        return None

    def build_dataset(self,img_path,mode='train',batch=None):
        return ManifestDataset(self.records[mode],seed=self.args.seed,data=self.data,
            imgsz=640,batch_size=batch,augment=mode=='train',hyp=copy.deepcopy(self.args),rect=False,
            cache=False,single_cls=True,stride=32,pad=0.0 if mode=='train' else .5,prefix=f'{mode}: ',
            task='pose',fraction=1.0)

    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        if rank not in (-1,0): raise RuntimeError('This matched experiment supports one device only')
        dataset=self.build_dataset(dataset_path,mode,batch_size)
        sampler=EpochSampler(len(dataset),self.args.seed,mode=='train')
        generator=torch.Generator().manual_seed(seed_for(self.args.seed,0,1))
        return DataLoader(dataset,batch_size=min(batch_size,len(dataset)),sampler=sampler,
            num_workers=self.args.workers,pin_memory=self.device.type=='cuda',
            collate_fn=dataset.collate_fn,generator=generator,persistent_workers=False)

    def get_validator(self):
        result=super().get_validator()
        if self.cell_config['arm']!='point_only': self.loss_names+=('line_loss',)
        return result

    def _build_train_pipeline(self):
        if self.batch_size!=self.expected_batch:
            raise RuntimeError('OOM or automatic batch change violates the frozen matched budget; stop and register a common budget')
        self._building_pipeline+=1
        if self._building_pipeline>1: raise RuntimeError('Automatic training pipeline restart is disallowed')
        return super()._build_train_pipeline()

    def _setup_audit(self,_):
        model=unwrap_model(self.model)
        if not self.resume_path:
            self.scaler=torch.cuda.amp.GradScaler(enabled=self.amp,init_scale=self.cell_config['amp_init_scale'])
        if any(not p.requires_grad for p in model.parameters()):
            raise RuntimeError('Unexpected frozen parameter in full-network training')
        initial_path=self.cell_dir/'INITIAL_STATE.pt'
        if self.resume_path:
            self.initial_state=torch.load(initial_path,map_location='cpu')
        else:
            self.initial_state=snapshot(model);torch_write(initial_path,self.initial_state)
        self.initial_sha=sha(initial_path)
        if self.resume_path and self.initial_sha!=self.resume_initial_sha:raise RuntimeError('Initial state artifact changed before resume')
        self.expected_steps=math.ceil(len(self.records['train'])/self.expected_batch)*self.epochs
        json_write(self.cell_dir/'SETUP_AUDIT.json',dict(complete=True,PASS=True,
            train_frames=len(self.train_loader.dataset),val_frames=len(self.test_loader.dataset),
            expected_epochs=self.epochs,expected_optimizer_steps=self.expected_steps,
            nbs=self.args.nbs,accumulation=self.accumulate,all_parameters_trainable=True,
            parameters=sum(p.numel() for p in model.parameters()),initial_state_sha256=self.initial_sha,
            dataset_filter_applied=False,labels_cache_written=False,
            augmentation_runtime=[dict(name=type(t).__name__,enabled=getattr(t,'transform',True) is not None)
                for t in self.train_loader.dataset.transforms.transforms],bindings=self.cell_config['bindings']))

    def _epoch_start(self,_):
        if self.batch_size!=self.expected_batch: raise RuntimeError('Changed batch size')
        self.train_loader.sampler.set_epoch(self.epoch)
        self.test_loader.sampler.set_epoch(0)
        self.train_loader.generator.manual_seed(seed_for(self.args.seed,self.epoch,1))
        self.test_loader.generator.manual_seed(seed_for(self.args.seed,0,2))
        self.epoch_batch=0

    def preprocess_batch(self,batch):
        if self.model.training:
            if tuple(batch['img'].shape[-2:])!=(640,640): raise RuntimeError('Train input is not640square')
            labels=hashlib.sha256()
            for key in ['keypoints','bboxes','cls','batch_idx']:
                labels.update(key.encode());labels.update(batch[key].contiguous().numpy().tobytes())
            row=dict(epoch=self.epoch+1,batch=self.epoch_batch,
                source_indices=list(batch['audit_source_index']),augmentation_seeds=list(batch['audit_augmentation_seed']),
                image_sha256=hashlib.sha256(batch['img'].contiguous().numpy().tobytes()).hexdigest(),labels_sha256=labels.hexdigest())
            with (self.cell_dir/'BATCH_TRACE.jsonl').open('a') as stream: stream.write(canonical(row)+'\n')
            self.epoch_batch+=1;self.batch_count+=1;self.trace_rows+=1
        return super().preprocess_batch(batch)

    def optimizer_step(self):
        if not bool(torch.isfinite(self.loss).all()): raise RuntimeError('Nonfinite loss; automatic recovery disabled')
        old_scale=self.scaler.get_scale()
        if self.optimizer_steps<16:
            groups={'early_backbone':[],'hough_reduce':[],'hough_line_head':[],'hough_output_projection':[]}
            for name,p in self.model.named_parameters():
                if p.grad is None:continue
                group=('early_backbone' if name.startswith('model.0.') else 'hough_reduce' if '.hough.reduce.' in name
                    else 'hough_line_head' if '.hough.line_head.' in name else 'hough_output_projection' if '.hough.outputs.' in name else None)
                if group:groups[group].append(p.grad.detach().float().abs().sum())
            self.gradient_probe.append(dict(epoch=self.epoch+1,optimizer_step=self.optimizer_steps+1,
                gradient_l1_unscaled={k:float(torch.stack(v).sum()/old_scale) if v else None for k,v in groups.items()}))
        super().optimizer_step()
        if self.scaler.get_scale()<old_scale:
            raise RuntimeError('AMP overflow skipped an optimizer update; frozen successful-step budget not met')
        self.optimizer_steps+=1

    def _handle_nan_recovery(self,epoch):
        if self.loss is not None and not bool(torch.isfinite(self.loss).all()):
            raise RuntimeError('Nonfinite loss; automatic rollback disabled')
        return False

    def resume_training(self,ckpt):
        if not self.resume_path:return
        self.start_epoch=int(ckpt['epoch'])+1
        if not 0<self.start_epoch<self.epochs: raise RuntimeError('No unfinished epochs remain')
        self.optimizer.load_state_dict(ckpt['optimizer']);self.scaler.load_state_dict(ckpt['scaler'])
        self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict());self.ema.updates=ckpt['updates']
        self.best_fitness=ckpt.get('best_fitness')
        self.optimizer_steps=int(ckpt['optimizer_steps']);self.batch_count=int(ckpt['batch_count'])
        self.trace_rows=int(ckpt['trace_rows']);self.hist=ckpt['history'];self.gradient_probe=ckpt.get('gradient_probe',[])
        self.resume_initial_sha=ckpt['initial_state_sha256']
        self.model.criterion=self.model.init_criterion()
        self.model.criterion.updates=self.start_epoch-1;self.model.criterion.update()
        trace=self.cell_dir/'BATCH_TRACE.jsonl';lines=trace.read_text().splitlines(True) if trace.exists() else []
        if len(lines)<self.trace_rows:raise RuntimeError('Missing committed batch trace')
        if hashlib.sha256(''.join(lines[:self.trace_rows]).encode()).hexdigest()!=ckpt['batch_trace_sha256']:
            raise RuntimeError('Committed batch trace changed before resume')
        if len(lines)>self.trace_rows:
            (self.cell_dir/f'INTERRUPTED_TRACE_{time.time_ns()}.jsonl').write_text(''.join(lines[self.trace_rows:]))
            trace.write_text(''.join(lines[:self.trace_rows]))
        json_write(self.cell_dir/'history.json',self.hist)
        if self.csv.exists():
            csv_lines=self.csv.read_text().splitlines(True)
            self.csv.write_text(''.join(csv_lines[:self.start_epoch+1]))
        rng=ckpt['rng_state'];random.setstate(rng['python']);np.random.set_state(rng['numpy'])
        torch.random.set_rng_state(rng['torch'])
        if self.device.type=='cuda':torch.cuda.set_rng_state_all(rng['cuda'])

    def save_model(self):
        model=unwrap_model(self.model)
        delta=changes_since(model,self.initial_state)
        completed=self.epoch+1==self.epochs and self.optimizer_steps==self.expected_steps
        if self.epoch+1==self.epochs and not completed:raise RuntimeError('Final optimizer-step count mismatch')
        row=dict(epoch=self.epoch+1,optimizer_steps=self.optimizer_steps,batches=self.batch_count,
            loss_items=[float(x) for x in self.tloss.detach().cpu()],loss_names=list(self.loss_names),
            metrics={k:float(v) for k,v in (self.metrics or {}).items()},parameter_BN_changes=delta)
        self.hist.append(row);json_write(self.cell_dir/'history.json',self.hist)
        json_write(self.cell_dir/'GRADIENT_AUDIT.json',dict(arm=self.cell_config['arm'],
            auxiliary_weight=unwrap_model(self.model).line_weight if self.cell_config['arm']=='hough_joint' else 0.,
            probes=self.gradient_probe,interpretation='First16 actual backward passes before optimizer zero_grad; unscaled gradient L1.'))
        criterion=model.criterion;model.criterion=None
        try:train_model=copy.deepcopy(model).cpu().float()
        finally:model.criterion=criterion
        ema_source=self.ema.ema;ema_criterion=getattr(ema_source,'criterion',None);ema_source.criterion=None
        try:ema=copy.deepcopy(ema_source).cpu().float()
        finally:ema_source.criterion=ema_criterion
        checkpoint=dict(epoch=self.epoch,complete=completed,stage=self.cell_config['stage'],
            expected_epochs=self.epochs,expected_optimizer_steps=self.expected_steps,optimizer_steps=self.optimizer_steps,
            batch_count=self.batch_count,trace_rows=self.trace_rows,batch_trace_sha256=sha(self.cell_dir/'BATCH_TRACE.jsonl'),
            history=self.hist,gradient_probe=self.gradient_probe,
            model=train_model,ema=ema,updates=self.ema.updates,
            optimizer=copy.deepcopy(self.optimizer.state_dict()),scaler=self.scaler.state_dict(),
            scheduler=self.scheduler.state_dict(),best_fitness=self.best_fitness,
            train_args=vars(self.args),train_metrics={**(self.metrics or {}),'fitness':self.fitness},
            train_results=self.read_results_csv(),version=ULTRALYTICS_VERSION,
            joint_provenance=self.cell_config['bindings'],initial_state_sha256=self.initial_sha,
            rng_state=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.random.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else None))
        torch_write(self.last,checkpoint)
        if self.best_fitness==self.fitness:torch_write(self.best,checkpoint)
        if completed:torch_write(self.wdir/'final.pt',checkpoint)
        return True

    def final_eval(self):
        # Final epoch validation already ran. Keep final/last resumable FP32;
        # stock final_eval strips optimizer and substitutes best-epoch weights.
        if self.epoch+1!=self.epochs:raise RuntimeError('Training stopped before registered final epoch')


def make_config(args,records):
    protocol=args.run_dir/'TRAIN_PROTOCOL.json'
    if not args.smoke_limit and not protocol.exists():raise RuntimeError('Main training requires frozen TRAIN_PROTOCOL.json')
    if sha(R0_PATH)!=R0_SHA256 or sha(SOURCE_MANIFEST)!=SOURCE_SHA256:raise RuntimeError('Frozen R0/source SHA mismatch')
    required_sources=[HERE/name for name in ['integration.py','train.py','hough_block.py','line_targets.py']]
    if not all(p.exists() for p in required_sources):raise RuntimeError('All four integration sources must exist before training')
    config=dict(schema='pallet_dht_joint_cell_v1',stage='smoke' if args.smoke_limit else 'main',arm=args.arm,seed=args.seed,
        epochs=args.epochs,batch=args.batch,lr=args.lr,optimizer=args.optimizer,line_weight=args.line_weight,
        workers=args.workers,amp=args.amp,amp_init_scale=args.amp_init_scale,smoke_limit=args.smoke_limit,train_frames=len(records['train']),val_frames=len(records['val']),
        training_recipe=dict(imgsz=640,rect=False,mosaic=0.,mixup=0.,cutmix=0.,copy_paste=0.,fliplr=0.,flipud=0.,
            scale=.25,translate=.1,degrees=0.,shear=0.,perspective=0.,hsv_h=.015,hsv_s=.5,hsv_v=.35,
            nbs=args.batch,lr_final_fraction=args.lrf,warmup_epochs=args.warmup_epochs,warmup_bias_lr=args.warmup_bias_lr,weight_decay=.0005,
            momentum=.937,cos_lr=True,freeze=None,close_mosaic=0,multi_scale=0.0),
        source_code_sha256={str(p):sha(p) for p in required_sources})
    if protocol.exists():
        registered=read(protocol).get('training',{})
        if not args.smoke_limit:
            for k in ['epochs','batch','lr','optimizer','line_weight','workers']:
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
    if args.epochs<1 or args.batch<1 or args.workers<0 or args.lr<=0:raise ValueError('Positive budget required')
    if args.smoke_limit not in (0,32,128):raise ValueError('Smoke subset must be32 or128')
    source=read(SOURCE_MANIFEST)
    records={'train':[r for r in source['records'] if r['partition']=='train'],
             'val':[r for r in source['records'] if r['source_split']=='val']}
    if len(records['train'])!=55980 or len(records['val'])!=4020:raise RuntimeError('Canonical source denominator changed')
    if args.smoke_limit:
        records['train']=records['train'][:args.smoke_limit];records['val']=records['val'][:min(32,args.smoke_limit)]
    config=make_config(args,records)
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
        trainer=JointTrainer(cell_config=config,records=records,cell_dir=cell,
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
        completion=dict(schema='pallet_dht_joint_training_completion_v1',complete=True,PASS=True,stage=config['stage'],
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
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--warmup-epochs',type=float,default=.1)
    parser.add_argument('--lrf',type=float,default=.1);parser.add_argument('--warmup-bias-lr',type=float,default=0.)
    parser.add_argument('--device',default='0');parser.add_argument('--amp',action='store_true');parser.add_argument('--amp-init-scale',type=float,default=16.);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke-limit',type=int,default=0)
    run(parser.parse_args())
