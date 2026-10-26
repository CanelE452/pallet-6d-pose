from pathlib import Path
import os,sys,json,hashlib,csv,datetime,fcntl,subprocess
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
OUT=Path(__file__).resolve().parents[1]
REPO=Path('/home/minjae/Documents/github/pallet-pose')
FLIP=OUT.parent/'dope-loo-flip-per-type-20261026-v1'
NOFLIP=OUT.parent/'dope-loo-no-flip-per-type-20261026-v1'
R0=REPO/'weights/paper_s2_stageB/net_epoch_0057_noseg.pth'
SEED=20261026
KINDS=('PLASTIC','WOOD','GREEN')
TYPE_TO_KIND={'plastic_standard_110x130x11':'PLASTIC','wood_small_80x59x14':'WOOD','plastic_standard_110x110x15':'GREEN','rectangular_plastic':'PLASTIC','small_wood':'WOOD','square_green_plastic':'GREEN','plastic':'PLASTIC','wood':'WOOD','green':'GREEN','PLASTIC':'PLASTIC','WOOD':'WOOD','GREEN':'GREEN'}
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def log(event,**values):
    p=OUT/'logs/RUN_LOG.jsonl'
    with p.open('a') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX)
        f.write(json.dumps({'utc':now(),'event':event,**values},ensure_ascii=False,allow_nan=False)+'\n');f.flush()
        fcntl.flock(f.fileno(),fcntl.LOCK_UN)
def start(task,inputs=()):
    log('step_start',task=task,command=[sys.executable,*sys.argv],seed=SEED,code_sha256={str(p):sha(p) for p in sorted((OUT/'code').glob('*.py'))},inputs={str(p):sha(p) for p in inputs})
def csvwrite(path,rows,fields=None):
    rows=list(rows);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:fields=list(rows[0]) if rows else []
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def mdtable(headers,rows):
    def cell(v):
        if v is None:return 'NA'
        return str(v).replace('|','\\|').replace('\n',' ')
    return '| '+' | '.join(map(cell,headers))+' |\n|'+ '|'.join('---' for _ in headers)+'|\n'+''.join('| '+' | '.join(cell(v) for v in row)+' |\n' for row in rows)
def kind_of(record):
    if record.get('kind') in KINDS:return record['kind']
    value=record.get('object_type')
    if value in TYPE_TO_KIND:return TYPE_TO_KIND[value]
    raise ValueError(('Unknown object type',value,record.get('id')))
def roles(kind):
    return {'R0':'R0_before_selftraining','control_R1':'SYNTHETIC_CONTROL_R1','control_R2':'SYNTHETIC_CONTROL_R2','flip_R1':kind+'_LOO_flip_R1','flip_R2':kind+'_LOO_flip_R2','loo_R1':kind+'_LOO_only_R1','loo_R2':kind+'_LOO_only_R2'}
def core_manifest():return read(FLIP/'MANIFEST.json')['records']
def load_module(path,name):
    import importlib.util
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def forward_log(task,model,count,**extra):
    assert task in ('T1b','T3','T4')
    log('neural_forward',task=task,model=model,image_count=count,**extra)

