"""Independent ResNet-18 estimator: source-only initialization and training."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
DOC=ROOT/'_docs/experiments'/HERE.name
RAW=ROOT/'data/pallet/results'/HERE.name
SOURCE=ROOT/'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json'
PRETRAINED=RAW/'pretrained/resnet18-f37072fd.pth'
PRETRAINED_SHA='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'

def read(p):return json.loads(Path(p).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def bound(p):
    p=Path(p).resolve();return dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
def write(p,value,freeze=True):
    p=Path(p);assert p.resolve().is_relative_to(DOC) or p.resolve().is_relative_to(RAW)
    p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if p.exists() and freeze:
        assert p.read_text()==text,('Frozen artifact differs',p);return
    q=p.with_name(p.name+'.pending');q.write_text(text);q.replace(p)
def gpu():
    v=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).strip()
    for line in processes.splitlines():
        cells=[s.strip() for s in line.split(',')]
        assert cells[0]==str(os.getpid()) or cells[1]=='/usr/share/rustdesk/rustdesk',('Other GPU job',line)
    assert float(v.split(',')[2])<80,('GPU temperature',v)
    return dict(time=now(),status=v,processes=processes)
def verify():
    p=read(DOC/'BASELINE_PROTOCOL.json')
    for b in p['bindings']:assert sha(ROOT/b['path'])==b['sha256'],b['path']
    return p
def lock():
    p=DOC/'BASELINE_PROTOCOL.json'
    if p.exists():return verify()
    assert sha(PRETRAINED)==PRETRAINED_SHA
    smoke=read(DOC/'BASELINE_GPU_SMOKE.json');assert smoke['PASS']
    for entry in smoke['code']:assert sha(ROOT/entry['path'])==entry['sha256'],entry['path']
    code=['base_common.py','model.py','test_model.py','input_data.py','test_input.py','baseline_train.py']
    value=dict(schema='resnet18_full_image_pallet9_baseline_protocol_v1',time=now(),
        purpose='Third independent RGB keypoint estimator for a controlled before/after P refiner study; not official human-pose benchmark reproduction or a renamed DOPE/PoseFix branch',
        model='SimpleBaseline-derived ResNet18 plus three256-channel stride2 transposed convolutions and a9-channel heatmap head',
        official_sources=['https://github.com/microsoft/human-pose-estimation.pytorch','https://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html'],
        pretrained=bound(PRETRAINED),new_real_supervision=0,YOLO_dependency=False,DOPE_dependency=False,
        source=bound(SOURCE),partitions=dict(train=55980,calibration=1004,selection=1031,heldout=1985),
        input=dict(height=384,width=512,source_reflect_padding='Existing100 only',real_reflect_padding=100,
           geometry='Aspect fit and centered128BGR letterbox; exact actualx/y resize inverse; no GT crop',
           normalization='ImageNet RGB constants, FP32'),
        targets=dict(channels=9,stride=4,sigma=2.,gaussian='Continuous output-grid center, finite3sigma support',
            mask='visibility>0 and finite center inside output; original out-of-view/source missing counts retained',
            loss='0.5 pixel-mean MSE per channel, supervised channels weighted, fixed9-channel denominator then batch mean'),
        decoding=dict(method='Argmax with interior quarter-pixel signed neighbor correction',peak_threshold=.1,
            missing='No imputation',box='At least3valid predicted corners, hull extents>1 supplied-image pixel'),
        seed=42,epochs=60,effective_batch=16,physical_batch=smoke['physical_batch'],
        optimizer=dict(name='Adam',lr=.001,weight_decay=.0001,betas=[.9,.999],drop_epochs=[40,50],drop_factor=.1),
        augmentation=dict(brightness=[.8,1.2],contrast=[.8,1.2],saturation=[.8,1.2],order='fixed brightness/contrast/saturation, per-image RNG on GPU',
             geometric_augmentation=False,already_domain_randomized_source=True),
        precision=dict(model='FP32',AMP=False,TF32=False,cudnn_benchmark=False),
        validation='Per-epoch calibration1004 loss only; final60 checkpoint fixed, no best checkpoint and no real metric selection',
        checkpoint='Durable exact model/optimizer/RNG/epoch/batch position every500updates and epoch end; infrastructure resume records any replay',
        learning_budget_note='Fixed before training; not shortened to manufacture a weak baseline for refinement. Original DOPE/YOLO initialization and total training budgets differ and are disclosed.',
        subsequent_refiners='Freeze baseline. Same P/D principles, seeds1/2/3,6000updates×batch16; separately locked after baseline completion.',
        status='BASELINE_TRAINING_READY_NOT_A_THIRD_ESTIMATOR_RESULT_YET',
        bindings=[bound(q) for q in [SOURCE,PRETRAINED,DOC/'BASELINE_GPU_SMOKE.json',*[HERE/n for n in code]]])
    write(p,value);return value
