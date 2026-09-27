"""Lock a single TRAIN-motivated budget intervention; validate loss without eval labels."""
import csv
import hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
from . import common as C

def main():
    torch.set_num_threads(4)
    from scripts.self_training_yolo.v3 import verify_true_ignore_contract as V
    row=C.read(C.RAW/'TRAIN_TARGETS_PRIVATE.json')[0]
    label=C.ROOT/row['padded_label_bindings']['RAW']['path'];a=np.array(label.read_text().split(),float)
    image=cv2.imread(str(C.ROOT/row['image']['path']));image=cv2.copyMakeBorder(image,100,100,100,100,cv2.BORDER_REFLECT_101)
    im=cv2.resize(image,(V.IMGSZ,V.IMGSZ));im=torch.from_numpy(im[:,:,::-1].copy()).permute(2,0,1)[None].float()/255
    V._SAMPLE=(im,torch.tensor(a[5:].reshape(9,3)[:,:2],dtype=torch.float32),torch.tensor(a[1:5][None],dtype=torch.float32))
    V.OUT=C.DOC/'TRUE_IGNORE_TRAIN_FIXTURE_TEST.json';assert V.main()==0
    protocol=C.read(C.P.REC/'pose_only/PROTOCOL.json');parity={}
    for t in ('RAW','REF'):
        pp=[Path(x) for x in (C.ROOT/protocol['datasets'][t]['train_list']['path']).read_text().splitlines()]
        support=[];box=[];coords=[];rgb=[]
        for image in pp:
            a=np.array((image.parent.parent/'labels'/f'{image.stem}.txt').read_text().split(),float)
            rgb.append((image.name,C.sha(image)));box.append(a[:5].tolist());support.append(a[5:].reshape(9,3)[:,2].tolist());coords.append(a[5:].reshape(9,3)[:,:2].tolist())
        digest=lambda x:hashlib.sha256(repr(x).encode()).hexdigest()
        parity[t]=dict(rgb_order=digest(rgb),boxes=digest(box),support=digest(support),coordinates=digest(coords),slots=len(pp))
    assert all(parity['RAW'][k]==parity['REF'][k] for k in ('rgb_order','boxes','support','slots'))
    assert parity['RAW']['coordinates']!=parity['REF']['coordinates']
    C.save(C.DOC/'NEW_PAIR_PREFLIGHT.json',dict(parity=parity,fixture='TRAIN image only; no evaluation coordinates opened in gradient test',
        initial=C.checkpoint('R0'),steps_each=640,seed=42,legacy_prefix=320,augmented_tensor_exhaustive_parity_claim=False),True)
    t=C.read(C.DOC/'TRAIN_TARGET_TRANSFER.json');v=t['all']['corners'];curves=t['curves']
    decision=dict(selected='A_budget',utc=C.now(),new_fits_planned=2,updates_each=640,max_total_updates=1280,
        purpose='Test whether additional fixed-schedule optimization improves partial target following and transfers; not tune66point threshold.',
        evidence=dict(ref_target_mean_raw=v['residuals']['RAW_LR5']['ref']['mean_px'],ref_target_mean_ref=v['residuals']['REF_LR5']['ref']['mean_px'],direction=v['direction'],last_two_pose_losses={a:[r['train/pose_loss'] for r in curves[a][-2:]] for a in curves}),
        counterarguments=['Native unaugmented residual is not augmented training objective.','Pseudo labels are not true coordinates.','Loss is mixed real/source; declining last interval does not prove convergence shortage.','Frozen representation, augmentation mismatch, contradictory targets and transfer limitations remain plausible.'],
        choices={'A':'Selected: partial aligned following + remaining TRAIN residual + late objective decline justify a bounded hypothesis test, not a causal diagnosis.',
            'B':'Not selected: no demonstrated calibration of corrected-coordinate reliability. Preserved confidence is not such evidence. No stability filter/Listen2Student implementation or calibration run.',
            'C':'No fit remains legitimate if prefix or parity cannot be reproduced. No fallback to B.'},
        condition='First320updates must reproduce original LR/loss/EMA checkpoint; mismatch stops experiment and leaves uncertainty, not another intervention.',
        development_stop_after=True,independent_test=False)
    C.save(C.DOC/'DECISION_BEFORE_FIT.json',decision,True)
    from ultralytics.utils.torch_utils import one_cycle
    lf=one_cycle(1,.1,5);lr=[1e-5*lf(min(i,4)) for i in range(10)]
    C.save(C.DOC/'INTERVENTION_LOCK.json',dict(utc=decision['utc'],selected='A_budget',args=protocol['args'],
        learning_rates_by_epoch=lr,epochs_effective=10,legacy_args_epochs=5,
        reason_args_epochs='Ultralytics E2ELoss also schedules one2many/one2one weights using args.epochs; keep5 to preserve its original trajectory, then its final0.1/0.9 weights.',
        lr_rule='Original cosine5 for epochs1..5; hold ACTUAL epoch5 rate1.8594235253127371e-6 for6..10.',
        controls='Same R0, data paths/order/support/boxes, batch16/nbs16, AdamW/seed42/augmentation,pose-only+flow,all buffers fixed; only raw/ref coordinates differ.',
        last_only=True,early_stop_on_score=False,optimizer_resume=False,prefix_rebuilt_from_R0=True,
        old_optimizer=C.read(C.DOC/'OLD_OPTIMIZER_STATE_AUDIT.json'),
        preflight=C.bind(C.DOC/'NEW_PAIR_PREFLIGHT.json'),source_files=[C.bind(p) for p in Path(__file__).parent.glob('*.py')]),True)
    print('INTERVENTION_LOCKED_A_640x2',flush=True)

if __name__=='__main__':main()
