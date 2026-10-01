"""New experiment only; historical data/checkpoints are read-only dependencies."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
NAME = HERE.name
DOC = ROOT / '_docs/experiments' / NAME
RAW = ROOT / 'data/pallet/results' / NAME
SOURCE = ROOT / 'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json'
OLD_PROTOCOL = ROOT / 'data/pallet/results/pallet_line_pose_v1/TRAIN_PROTOCOL.json'
DEV = ROOT / 'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json'
WEIGHTS = RAW / 'baseline_final.pt'
BASELINE_PROTOCOL = DOC / 'BASELINE_PROTOCOL.json'
BASELINE_COMPLETE = DOC / 'BASELINE_TRAINING_COMPLETE.json'
# No checkpoint, receipt or dataset is read at import time. A trained baseline
# is mandatory when an execution phase calls baseline_contract().
CONFIG = dict(c3=128,c4=256,stride3=8,stride4=16,hidden=16,encoded=24,
              stencil_fraction=0.1310373991727829)
SEEDS = (1,2,3)

def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def bound(path):
    p=Path(path).resolve()
    return dict(path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),sha256=sha(p))
def write(path,value,freeze=True):
    p=Path(path)
    assert any(p.resolve().is_relative_to(r.resolve()) for r in (DOC,RAW)),p
    p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if freeze and p.exists():
        assert p.read_text()==text,('Frozen output changed',p)
        return
    temp=p.with_name(p.name+'.pending')
    temp.write_text(text);temp.replace(p)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m
def gpu():
    status=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).strip()
    unrelated=[];display=[]
    for line in procs.splitlines():
        cells=[x.strip() for x in line.split(',')]
        if cells[0]==str(os.getpid()):continue
        if cells[1]=='/usr/share/rustdesk/rustdesk':display.append(line)
        else:unrelated.append(line)
    if unrelated:raise RuntimeError('Other GPU compute present; no process killed: '+str(unrelated))
    if float(status.split(',')[2])>=80:raise RuntimeError('GPU temperature guard >=80 C')
    return dict(time=now(),status=status,processes=procs,existing_display_processes=display)

def verify_lock():
    p=read(DOC/'PROTOCOL.json')
    for entry in p['bindings']:
        path=ROOT/entry['path']
        assert sha(path)==entry['sha256'],path
    return p

def baseline_contract():
    """Verify the separately trained final baseline; never use a fallback."""
    for path in (BASELINE_PROTOCOL,BASELINE_COMPLETE,WEIGHTS):
        if not path.is_file():
            raise RuntimeError('ResNet18 final epoch60 baseline is not ready; complete the separately sealed baseline pipeline first: '+str(path))
    receipt=read(BASELINE_COMPLETE);protocol=read(BASELINE_PROTOCOL)
    assert receipt['complete'] is True and receipt['final_checkpoint_only'] is True
    assert receipt['epochs']==60 and receipt['seed']==42 and receipt['real_training']==0
    assert receipt['train_images']==55980 and receipt['calibration_images']==1004
    assert receipt['updates']==math.ceil(55980/16)*60 and receipt['source_exposures']==55980*60
    assert protocol['schema']=='resnet18_full_image_pallet9_baseline_protocol_v1'
    assert protocol['epochs']==60 and protocol['effective_batch']==16 and protocol['seed']==42
    assert protocol['input']['height']==384 and protocol['input']['width']==512
    assert protocol['targets']['channels']==9 and protocol['targets']['stride']==4
    assert protocol['decoding']['peak_threshold']==.1
    assert protocol['new_real_supervision']==0 and not protocol['YOLO_dependency'] and not protocol['DOPE_dependency']
    for entry in [receipt['protocol'],receipt['final_checkpoint'],*protocol['bindings']]:
        path=ROOT/entry['path']
        assert sha(path)==entry['sha256'],path
        if 'bytes' in entry:assert path.stat().st_size==entry['bytes'],path
    assert (ROOT/receipt['protocol']['path']).resolve()==BASELINE_PROTOCOL.resolve()
    assert (ROOT/receipt['final_checkpoint']['path']).resolve()==WEIGHTS.resolve()
    assert receipt['source']==protocol['source']
    assert (ROOT/receipt['source']['path']).resolve()==SOURCE.resolve()
    assert receipt['source']['sha256']==sha(SOURCE)
    return dict(protocol=bound(BASELINE_PROTOCOL),completion=bound(BASELINE_COMPLETE),
                checkpoint=bound(WEIGHTS))

def learning_rate(step,protocol):
    o=protocol['optimizer'];warm=o['warmup_steps']
    if step<=warm:return o['lr']*step/warm
    t=(step-warm)/(protocol['steps']-warm);last=o['cosine_final_lr_fraction']
    return o['lr']*(last+(1-last)*.5*(1+math.cos(math.pi*t)))

def lock_protocol():
    if (DOC/'PROTOCOL.json').exists():return verify_lock()
    old=read(OLD_PROTOCOL)
    baseline=baseline_contract()
    src=read(SOURCE);assert len(src['records'])==60000
    paths=[SOURCE,OLD_PROTOCOL,DEV,WEIGHTS,BASELINE_PROTOCOL,BASELINE_COMPLETE,
           ROOT/'scripts/research/pallet_final_ml_contribution_test_v1/generic_point_refiner.py',
           ROOT/'scripts/research/pallet_sensors_refinement_closeout_v1/direct_residual_control.py',
           *[HERE/name for name in ('base_common.py','baseline_train.py','model.py','test_model.py','input_data.py','test_input.py',
                                   'common.py','resnet_adapter.py','refiner.py','test_refiner.py','data.py','train.py','selection.py','run.py')]]
    p=dict(schema='resnet18_local_refiner_protocol_v1',created_at=now(),
      purpose='Apply the existing P local-distribution refinement principle to independently source-trained SimpleBaseline ResNet18; compare paired ResNet18/D/P, including unfavorable outcomes. YOLO and DOPE are separate backbone controls.',
      authorized_scope='User 2026-10-01: run experiments to complete IEEE Sensors manuscript; later instruction authorizes new head training beyond the earlier diagnosis-only document.',
      status='LOCKED_AFTER_BASELINE_TRAINING_BEFORE_HEAD_CACHE_AND_TRAINING',
      seeds=list(SEEDS),steps=6000,batch=16,fits={'P':3,'D':3},optimizer=old['optimizer'],
      model_config=CONFIG,baseline=baseline['checkpoint'],baseline_trainable=False,
      baseline_protocol=baseline['protocol'],baseline_training_complete=baseline['completion'],
      baseline_architecture='RGB full-image SimpleBaseline ResNet18, three 256-channel deconvolutions, nine stride4 heatmaps; independent of YOLO and DOPE',
      input=dict(preprocessing='Aspect-fit to 384x512, center letterbox value128 BGR, RGB ImageNet FP32 normalization; exact actual x/y inverse',
        height=384,width=512,decoder_peak_threshold=.1,
        source='Existing prepared images already include reflect100, no second reflection',
        real='Original BGR image reflected100 then same recipe',
        decoder='Nine semantic heatmap-channel argmax peaks with interior quarter-pixel sign adjustment, threshold0.1 inclusive',
        features='Actual ResNet18 layer2/layer3 outputs, 128 channels stride8 /256 channels stride16',
        sampling='Existing P cell-center grid convention retained; not an assertion of exact backbone receptive-field centers',
        precision='ResNet18 FP32, features FP16 roundtrip as in original P; heads FP32, no AMP or TF32',
        box='Hull of at least three valid predicted corners, frozen for all methods',
        missing='Preserve missing mask and center8; no target imputation'),
      training=dict(source_manifest=bound(SOURCE),partitions=old['partitions'],
        usable='TRAIN only, predicted hull IoU>=0.5 with sole source GT, at least one jointly valid corner; actual sampled support counted separately at each loss',
        order='Per-seed deterministic shuffled usable rows, 6000x16; D and P same exact row sequence',
        source_supervision='Same source RGB and labels as historical YOLO, new baseline-specific usable mask disclosed',
        real_training=0,augmentation='No extra augmentation; identical prepared source images',
        feature_storage='No full spatial feature cache: only prediction metadata cached. Online frozen ResNet18 prefix through layer3, no layer4/deconvolution/heatmap-head recomputation during head training',
        checkpoint='Final step6000 only; resume exact checkpoint state allowed after infrastructure interruption; no outcome-driven restart'),
      calibration=old['calibration'],selection=old['selection'],
      selection_detail='P calibration: supported-frame point-target CE. D has no temperature. Same source-only lambda/cap grid and ties as YOLO. Decode displacement to original pixels before optional image-diagonal cap (nonuniform scale).',
      evaluation=dict(population=bound(DEV),n=319,role='reused DEV, not independent TEST',
        primary='Per-estimator paired before/after 2D errors; all-GT PCK and failures plus canonical complete-point conditional medians',
        downstream='Existing canonical geometry-derived reference, K/dimensions/PnP unchanged; coverage and failures retained',
        bootstrap_draws=10000,bootstrap_seed=20260914,unit='session, same resampled frames and seed mean',
        real_based_selection=False,negative_AP='Not measured for new ResNet18 extension; no claim'),
      runtime=dict(device='same local desktop GPU',batch=1,warmup=20,images='first two by manifest order per session, 26 total',
        repeats=5,arms=['RESNET18','RESNET18+D1','RESNET18+P1'],scope='Decoded BGR through canonical pose where available; report keypoints-only separately; exclude file decoding/loading',
        timing_not_inferred_from_parameter_count=True),
      smoke='Before main training, two optimizer updates per arm on a fixed first-four-per-source-shape group; discarded fresh smoke heads, separate seed20261001; contracts/finite gradients only, no performance selection',
      stopping='Exactly six fixed-budget fits. Numerical/contract bugs require documented correction. No architecture/seed/real-based threshold search.',
      disk_policy='Keep original artifacts; new compact caches only. Do not delete unrelated files.',
      historical_goal='Stable simultaneous real translation/rotation improvement remains unproven; this experiment cannot retroactively pass its fixed gates.',
      bindings=[bound(q) for q in paths])
    write(DOC/'PROTOCOL.json',p)
    return p

if __name__=='__main__':print(json.dumps(lock_protocol(),ensure_ascii=False,indent=2))
