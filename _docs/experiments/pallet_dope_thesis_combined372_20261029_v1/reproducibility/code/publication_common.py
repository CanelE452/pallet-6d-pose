"""Exact ROOT-supplied inputs for final372 delivery. Import performs no IO.

No discovery of raw inputs. PRIVATE error caches may be numerical-QA inputs,
but are never publication payloads. This module never imports NN/PnP code.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import re

OUT=Path(__file__).resolve().parents[1]
INPUTS=OUT/'preflight/PUBLICATION_INPUTS.json'
NUMERICAL_INPUTS=OUT/'preflight/NUMERICAL_INPUTS.json'
MANUSCRIPT_INPUTS=OUT/'preflight/MANUSCRIPT_INPUTS.json'
FORBIDDEN_KEYS={'xy','points','points_original','reference_xy','reference_valid','all_annotated_reference',
    'projected_cuboid','manual_kps','keypoint_annotations','predictions','pseudo_targets','target_coordinates'}
PRIVATE_NAMES={'SCORES.json','PREDICTIONS.json','REFERENCE_POSES_PRIVATE.json','PREDICTED_POSES_PRIVATE.json','ALLOWLIST.json'}
EXPECTED={'PLASTIC':(128,985),'WOOD':(125,963),'GREEN':(119,943)}
QUALITY=('PCK5','PCK10','PCK20','coverage','conditional_precision10')
POLICY_PATH=OUT.parent/'dope-r2-ratio-per-type-20261027-v1/CONTINUATION_POLICY.json'
POLICY_SHA='7740a3f2cc778b6a9f9a48c595d260dab38ff66d1bc4296316419ec189aac120'


def require(ok,msg):
    if not ok:raise RuntimeError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    p=Path(path);require(p.resolve().is_relative_to(OUT),'Write outside DELIVERY')
    require(not p.exists(),'Preserve prior artifact: '+str(p));p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def csvwrite(path,rows):
    p=Path(path);require(p.resolve().is_relative_to(OUT) and not p.exists(),'Unsafe/existing output CSV')
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows)


def safe_relative(value):
    p=Path(value);require(value and not p.is_absolute() and '..' not in p.parts and p.as_posix()==value,'Unsafe payload relative path')
    require(all(part not in ('.git','__pycache__') for part in p.parts) and p.name!='AGENTS.md','Blocked payload name')
    return p


def no_coordinates(value):
    if isinstance(value,dict):
        require(not(FORBIDDEN_KEYS&set(value)),'Coordinate/raw prediction keys prohibited')
        for item in value.values():no_coordinates(item)
    elif isinstance(value,list):
        for item in value:no_coordinates(item)


class Inputs:
    def __init__(self,manifest=INPUTS):
        self.manifest_path=Path(manifest).resolve();require(self.manifest_path in (INPUTS,NUMERICAL_INPUTS,MANUSCRIPT_INPUTS),'Unknown input-contract path')
        require(self.manifest_path.is_file(),'ROOT must supply an exact input manifest')
        self.manifest_sha=sha(self.manifest_path);self.spec=json.loads(self.manifest_path.read_text());self.bindings={str(self.manifest_path):self.manifest_sha}
        require(self.spec['schema']=='primary372_delivery_publication_inputs_v1','Unknown publication contract')
        require(self.spec['primary_frames']==372 and self.spec['primary_reference_points']==2891,'Primary372 contract differs')
        self.files=self.spec['source_files'];require(isinstance(self.files,dict) and self.files,'Empty source manifest')
        for p,item in self.files.items():
            require(Path(p).is_absolute() and str(Path(p).resolve())==p and not Path(p).is_symlink(),'Exact real source path required')
            require(re.fullmatch('[0-9a-f]{64}',item['sha256']) is not None,'Invalid source SHA')
            require(Path(p).suffix not in ('.pth','.pt','.ckpt','.npz','.npy','.safetensors','.pkl','.jpg','.jpeg','.png.raw'),'Model/raw source blocked')
            if item.get('destination') is not None:
                dest=safe_relative(item['destination'])
                require(Path(p).name not in PRIVATE_NAMES and dest.name not in PRIVATE_NAMES,'Private cache cannot be published')
                require(not item.get('private_input',False),'Private QA input cannot enter payload')

    def path(self,path):
        p=Path(path).resolve();require(str(p) in self.files,'Unlisted source read blocked: '+str(p))
        digest=sha(p);require(digest==self.files[str(p)]['sha256'],'Source SHA changed: '+str(p))
        self.bindings[str(p)]=digest;return p

    def read(self,path):
        p=self.path(path);value=json.loads(p.read_text());no_coordinates(value)
        require(sha(p)==self.bindings[str(p)],'Input changed while reading');return value

    def csv(self,path):
        p=self.path(path)
        with p.open(newline='') as stream:
            reader=csv.DictReader(stream);require(not(FORBIDDEN_KEYS&set(reader.fieldnames or ())), 'Coordinate CSV blocked');rows=list(reader)
        require(sha(p)==self.bindings[str(p)],'CSV changed while reading');return rows

    def role(self,role):
        paths=[Path(p) for p,item in self.files.items() if item['role']==role]
        require(paths,'Missing exact source role: '+role);return paths

    def one(self,role):
        paths=self.role(role);require(len(paths)==1,'Exactly one source required: '+role);return paths[0]

    def gates(self,phase='numerical'):
        require(self.spec['required_gates'],'ROOT must enumerate completion/numerical/figure/manuscript gates')
        for path,rule in self.spec['required_gates'].items():
            if phase not in rule.get('phases',('numerical','package','publish')):continue
            obj=self.read(path)
            for key,want in rule['equals'].items():require(obj.get(key)==want,'Required gate failed: '+path+'/'+key)
        path=self.one('latest_user_policy');require(path==POLICY_PATH and sha(path)==POLICY_SHA,'Authoritative continuation policy differs')
        policy=self.read(path)
        require(policy['passed'] is True and policy['allow_all_kinds_despite_inconclusive_wood'] is True
            and set(policy['required_kinds'])==set(EXPECTED) and set(policy['required_conditions'])=={'A','B','C','D','E'}
            and policy['technical_gates_waived'] is False,'Latest explicit continuation policy missing/technical gates waived')

    def unchanged(self):
        require(sha(self.manifest_path)==self.manifest_sha,'ROOT input manifest changed')
        for p,digest in self.bindings.items():require(sha(p)==digest,'Bound source changed: '+p)


def number(value):
    if value in ('',None):return None
    value=float(value);require(math.isfinite(value),'Nonfinite numerical value');return value


def same(actual,expected,context):
    if isinstance(expected,str):expected=number(expected)
    require((actual is None and expected is None) or (actual is not None and expected is not None
        and math.isclose(actual,expected,abs_tol=1e-10,rel_tol=1e-12)),'Numerical mismatch: '+context)


def inventory(root):
    root=Path(root);result={}
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(),'Payload symlink prohibited')
        if p.is_file():result[p.relative_to(root).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
    return result
