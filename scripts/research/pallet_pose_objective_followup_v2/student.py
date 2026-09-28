"""One immutable paired recipe at a time, reusing the Plastic/Wood main protocol."""
import argparse
from collections import Counter
import copy
import csv
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from scripts.research.pallet_type_selftrain_v1.recovery_pose_trainer import PoseOnlyTrainer, pose_parameter
from scripts.research.pallet_oracle_mechanism_followup_v1.cycle_affine import protocol, make_dataset, digest, seed, rng_digest, sample_fingerprint, gpu
from . import common as C
from .occlusion import SharedOcclusion, reference_labels

class StudentTrainer(PoseOnlyTrainer):
    def build_dataset(self,img_path,mode='train',batch=None):
        dataset=super().build_dataset(img_path,mode=mode,batch=batch)
        if mode=='train' and self.recipe['occlusion']:
            dataset.transforms=SharedOcclusion(dataset.transforms,reference_labels(self.parent_protocol,C.ROOT))
        return dataset

    def preprocess_batch(self,batch):
        info=batch.pop('occlusion_info',None)
        batch=super().preprocess_batch(batch)
        names=[Path(p).name for p in batch['im_file']]; mask=batch['keypoints'][...,2]
        is_source=torch.tensor([n.startswith('syn__') for n in names],device=mask.device)[batch['batch_idx'].long()]
        record=dict(batch=len(self.trace),epoch=int(self.epoch),names=names,images=digest(batch['img']),
            boxes=digest(batch['bboxes']),support=digest(mask),coordinates=digest(batch['keypoints'][...,:2]),
            batch_idx=digest(batch['batch_idx']),roles={},occlusion=list(info) if info is not None else [])
        for role,select in [('SOURCE',is_source),('REAL',~is_source)]:
            values=mask[select]
            record['roles'][role]=dict(images=sum(n.startswith('syn__')==(role=='SOURCE') for n in names),
                instances=int(select.sum()),supervised=int((values==2).sum()),ignore=int((values==1).sum()),invisible=int((values==0).sum()))
        self.trace.append(record)
        return batch

def preflight(material='PLASTIC'):
    dest=C.DOC/f'OCCLUSION_PREFLIGHT_{material}.json'
    if dest.exists(): assert C.read(dest)['passed']; print('PREFLIGHT_ALREADY_COMPLETE'); return
    torch.set_num_threads(4); cv2.setNumThreads(1)
    pp,p=protocol(material); canonical=reference_labels(p,C.ROOT)
    datasets={a:make_dataset(material,a) for a in ('RAW','REF')}
    base={a:copy.deepcopy(ds.transforms) for a,ds in datasets.items()}
    for ds in datasets.values(): ds.transforms=SharedOcclusion(ds.transforms,canonical)
    rows=[]
    for role in ('SOURCE','REAL'):
        ii=[i for i,path in enumerate(datasets['RAW'].im_files) if Path(path).name.startswith('syn__')==(role=='SOURCE')]
        indices=[ii[int((j+.5)*len(ii)/64)] for j in range(64)]
        for index in indices:
            sample_seed=280901+index; samples={}
            for arm,ds in datasets.items():
                seed(sample_seed); old=base[arm](copy.deepcopy(ds.get_image_and_label(index))); old_rng=rng_digest()
                seed(sample_seed); new=ds[index]; new_rng=rng_digest()
                assert old_rng==new_rng
                assert torch.equal(old['keypoints'],new['keypoints']) and torch.equal(old['bboxes'],new['bboxes'])
                if role=='SOURCE': assert torch.equal(old['img'],new['img'])
                samples[arm]=dict(old=sample_fingerprint(old),new=sample_fingerprint(new),plan=new['occlusion_info'])
            a,b=samples['RAW'],samples['REF']
            assert a['new']['image']==b['new']['image'] and a['new']['boxes']==b['new']['boxes']
            for key in ('seed','rectangle','size','fill_seed','applied','covered','remaining'):
                assert a['plan'].get(key)==b['plan'].get(key)
            rows.append(dict(role=role,index=index,arms=samples,support_same=a['new']['support']==b['new']['support']))
    C.save(C.RAW/f'OCCLUSION_PREFLIGHT_{material}_PRIVATE.json',rows,True)
    real=[r for r in rows if r['role']=='REAL']; counts=Counter(r['arms']['REF']['plan']['reason'] for r in real)
    covered=sum(r['arms']['REF']['plan'].get('actual_covered',0) for r in real)
    result=dict(passed=True,protocol=C.bind(pp),source_bit_exact=64,RGB_box_plan_pair_exact=128,
        targets_bit_exact_to_same_seed_original=128,baseline_RNG_stream_exact=128,
        real_applied=sum(r['arms']['REF']['plan']['applied'] for r in real),real_sampled=64,
        REF_covered_supervised=covered,reasons=dict(counts),
        augmented_support_difference=sum(not r['support_same'] for r in rows),
        support_note='Stored common support equal. Existing geometric clipping can produce coordinate-dependent out-of-frame support; preserved rather than repaired.',
        files=[C.bind(C.RAW/f'OCCLUSION_PREFLIGHT_{material}_PRIVATE.json')],fits=0,updates=0)
    assert result['real_applied']>0 and covered>0
    C.save(dest,result,True); print('PREFLIGHT',result,flush=True)

def train(cycle,material,target,training_seed=42):
    doc=C.DOC/'cycles'/cycle; raw=C.RAW/'cycles'/cycle
    recipe=C.read(doc/'PROTOCOL.json'); assert recipe['locked_before_fit']
    assert material in recipe['materials'] and target in ('RAW','REF') and training_seed in recipe['seeds']
    if recipe['occlusion']: assert C.read(C.DOC/f'OCCLUSION_PREFLIGHT_{material}.json')['passed']
    arm=f'{material}_{target}_S{training_seed}'; fit_path=raw/f'FIT_{arm}.json'
    if fit_path.exists():
        fit=C.read(fit_path); assert fit['complete']; C.verify(fit['checkpoint']); print('FIT_ALREADY_COMPLETE',arm); return
    pp,p=protocol(material)
    for b in p['inputs']+p['sources']+[p['initialization']]: C.verify(b)
    assert (C.DOC/'METRIC_AND_SELECTION_LOCK.json').exists()
    ledger=C.read(C.DOC/'RESOURCE_LEDGER.json')
    assert ledger['totals']['fits']<12 and ledger['totals']['optimizer_updates']+320<=7680
    run_dir=raw/'runs'/arm; assert not run_dir.exists(), 'Incomplete run is preserved; do not silently restart'
    from scripts.research.pallet_type_selftrain_v1 import train as T
    T.C.N.setup(); torch.set_num_interop_threads(1); assert torch.cuda.is_available(); gpu()
    args=dict(p['args'],lr0=1e-5,seed=training_seed,model=str(C.ROOT/p['initialization']['path']),
        data=str(C.ROOT/p['datasets'][target]['data']['path']),project=str(raw/'runs'),name=arm,exist_ok=False)
    trainer=StudentTrainer(overrides=args); trainer.recipe=recipe; trainer.parent_protocol=p; trainer.trace=[]
    base=torch.load(C.ROOT/p['initialization']['path'],map_location='cpu',weights_only=False)['model'].float().state_dict()
    steps=[]; history=[]; start=time.monotonic()
    C.save(raw/f'START_{arm}.json',dict(utc=C.now(),protocol=C.bind(doc/'PROTOCOL.json'),parent=C.bind(pp),arm=arm),True)
    def step_done(opt,args,kwargs):
        steps.append(1); assert len(steps)<=320
    def on_start(t):
        state=t.model.state_dict(); assert set(state)==set(base)
        assert all(torch.equal(base[k],state[k].detach().cpu()) for k in base)
        t.optimizer.register_step_post_hook(step_done)
        print('EXACT_R0',cycle,arm,len(t.recovery_trainable),flush=True)
    def on_epoch(t):
        fixed=t.check_frozen(); temperature=gpu()
        history.append(dict(epoch=t.epoch+1,steps=len(steps),protected_tensors=fixed,temperature_C=temperature))
        C.save(raw/f'RUN_STATE_{arm}.json',dict(status='RUNNING',arm=arm,steps=len(steps),history=history,utc=C.now()))
        print('EPOCH',cycle,arm,t.epoch+1,len(steps),flush=True)
    trainer.add_callback('on_train_start',on_start); trainer.add_callback('on_train_epoch_end',on_epoch)
    try:
        trainer.train()
        checkpoint=run_dir/'weights/last.pt'
        final=torch.load(checkpoint,map_location='cpu',weights_only=False)['model'].float().state_dict()
        protected=[k for k in base if not pose_parameter(k) or k.endswith(('.running_mean','.running_var','.num_batches_tracked'))]
        assert all(torch.equal(base[k],final[k]) for k in protected)
        changed=[k for k in base if not torch.equal(base[k],final[k])]; assert changed and all(pose_parameter(k) for k in changed)
        assert len(steps)==len(trainer.trace)==320
        assert len(list(csv.DictReader((run_dir/'results.csv').open())))==5
        C.save(raw/f'TRACE_{arm}.json',trainer.trace,True)
        fit=dict(complete=True,arm=arm,cycle=cycle,material=material,target=target,seed=training_seed,
            optimizer_steps=320,checkpoint=C.bind(checkpoint),protocol=C.bind(doc/'PROTOCOL.json'),parent_protocol=C.bind(pp),
            initialization=p['initialization'],exact_R0_initialization=True,protected_state_exact=True,
            protected_tensors=len(protected),changed_tensors=changed,
            trace=C.bind(raw/f'TRACE_{arm}.json'),results_csv=C.bind(run_dir/'results.csv'),history=history,
            seconds=time.monotonic()-start,manual_added=0,new_loss=recipe.get('loss','UNCHANGED'),
            checkpoint_selection='last only',implementation=[C.bind(Path(__file__)),C.bind(Path(__file__).with_name('occlusion.py'))])
        C.save(fit_path,fit,True)
        C.resource(f'FIT_{cycle}_{arm}',fit['seconds'],True,1,320,dict(checkpoint=fit['checkpoint']))
        print('FIT_COMPLETE',cycle,arm,fit['seconds'],flush=True)
    except BaseException as exc:
        C.save(raw/f'FAILURE_{arm}.json',dict(error=repr(exc),executed_optimizer_steps=len(steps),seconds=time.monotonic()-start),True)
        C.resource(f'FAILED_{cycle}_{arm}',time.monotonic()-start,True,int(bool(steps)),len(steps),repr(exc))
        raise

def parity(cycle,material,training_seed):
    raw=C.RAW/'cycles'/cycle; rows={}
    for target in ('RAW','REF'):
        arm=f'{material}_{target}_S{training_seed}'; fit=C.read(raw/f'FIT_{arm}.json'); C.verify(fit['checkpoint'])
        rows[target]=C.read(raw/f'TRACE_{arm}.json')
    assert len(rows['RAW'])==len(rows['REF'])==320
    differences=Counter(); roles=Counter(); occlusion=Counter()
    for a,b in zip(rows['RAW'],rows['REF']):
        for k in ('names','images','boxes','batch_idx'): assert a[k]==b[k], (k,a['batch'])
        for k in ('support','coordinates'): differences[k]+=a[k]!=b[k]
        for ai,bi in zip(a['occlusion'],b['occlusion']):
            for k in ('seed','rectangle','fill_seed','applied'): assert ai.get(k)==bi.get(k)
            occlusion[ai['role']+'_images']+=1
            if ai['role']=='REAL':
                occlusion['applied']+=ai['applied']; occlusion['REF_masked_supervised']+=bi.get('actual_covered',0)
                occlusion['RAW_masked_supervised']+=ai.get('actual_covered',0)
        for key,value in b['roles'].items():
            for k,n in value.items(): roles[key+'_'+k]+=n
    result=dict(passed=True,batches=320,paired_RGB_order_boxes=True,paired_occlusion_plan=True,
        differences=dict(differences),exposures=dict(roles),occlusion=dict(occlusion),
        support_difference_interpretation='Original geometric out-of-frame handling depends on raw/ref coordinate values; no posthoc common-mask repair.')
    C.save(C.DOC/'cycles'/cycle/f'PARITY_{material}_S{training_seed}.json',result,True)
    print('PARITY',result,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['preflight','train','parity'])
    parser.add_argument('--cycle',default='A_INPUT_OCCLUSION'); parser.add_argument('--material',default='PLASTIC')
    parser.add_argument('--target',choices=['RAW','REF'],default='RAW'); parser.add_argument('--seed',type=int,default=42)
    a=parser.parse_args()
    if a.phase=='preflight': preflight(a.material)
    elif a.phase=='train': train(a.cycle,a.material,a.target,a.seed)
    else: parity(a.cycle,a.material,a.seed)
