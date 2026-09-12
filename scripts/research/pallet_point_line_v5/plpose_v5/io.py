"""Strict prepared-instance cache I/O. Only our NEW schema is accepted.

An adapter must create these records after inspecting the source repository.
No assumptions about source label field names, strides or object axes.
"""
from __future__ import annotations
from pathlib import Path
import json,hashlib,os,tempfile
from dataclasses import asdict
import torch
from .contracts import Observation,Supervision,SymmetrySpec,validate_rotation
from .geometry import cuboid,project

OBS_KEYS=set(Observation.__dataclass_fields__)
SCHEMA='plpose_v5_instance_manifest_1'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding='utf-8') as f:return json.load(f)


def write_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    fd,tmp=tempfile.mkstemp(prefix=p.name+'.',suffix='.pending',dir=p.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:f.write(text);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    except BaseException:
        try:os.unlink(tmp)
        except FileNotFoundError:pass
        raise


def tensor_hash(state):
    h=hashlib.sha256()
    for key,x in sorted(state.items()):
        h.update(key.encode());h.update(str(tuple(x.shape)).encode());h.update(str(x.dtype).encode())
        h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def source_hashes():
    root=Path(__file__).resolve().parent
    return {p.name:sha256(p) for p in sorted(root.glob('*.py'))}

class InstanceDataset:
    def __init__(self,manifest):
        self.path=Path(manifest).resolve();self.meta=read_json(self.path)
        if self.meta.get('schema')!=SCHEMA:raise ValueError('Unknown manifest schema; export via audited adapter')
        self.records=self.meta['records'];self.target_reads=0;self.observation_reads=0
        ids=[r['id'] for r in self.records]
        if not ids or len(ids)!=len(set(ids)):raise ValueError('Nonempty unique IDs required')
        if self.meta.get('coordinate_contract')!='object_x_width_y_up_z_depth_v1':raise ValueError('Object-frame contract unresolved')
        if self.meta.get('split') not in ('train','calibration','synth_val','real_dev','generated'):
            raise ValueError('This kit cannot open a FINAL/test/unknown population')

    def __len__(self):return len(self.records)

    def resolve(self,name):
        p=Path(name)
        return p if p.is_absolute() else (self.path.parent/p).resolve()

    def observation(self,i):
        r=self.records[i];p=self.resolve(r['observation'])
        if sha256(p)!=r['observation_sha256']:raise ValueError('Observation hash mismatch')
        raw=torch.load(p,map_location='cpu',weights_only=True)
        if set(raw)!=OBS_KEYS:raise ValueError(f'Observation contains forbidden/unknown/missing keys: {set(raw)^OBS_KEYS}')
        if not all(torch.is_tensor(v) for v in raw.values()):raise TypeError('Tensor-only observation')
        self.observation_reads+=1
        return Observation(**raw).validate()

    def target(self,i):
        r=self.records[i];p=self.resolve(r['supervision'])
        if sha256(p)!=r['supervision_sha256']:raise ValueError('Supervision hash mismatch')
        raw=torch.load(p,map_location='cpu',weights_only=True)
        if set(raw)!=set(('points','valid','R','t')):raise ValueError('Unexpected target keys')
        spec=SymmetrySpec(**r['symmetry'])
        self.target_reads+=1
        return Supervision(raw['points'],raw['valid'],raw['R'],raw['t'],[spec])

    def batch(self,indices,device='cpu',with_target=True):
        indices=list(map(int,indices));observations=[self.observation(i) for i in indices]
        obs=Observation(**{k:torch.cat([getattr(o,k) for o in observations],0).to(device) for k in OBS_KEYS})
        if not with_target:return obs
        targets=[self.target(i) for i in indices]
        target=Supervision(*(torch.cat([getattr(g,k) for g in targets],0).to(device) for k in ('points','valid','R','t')),
                            [g.specs[0] for g in targets])
        return obs,target


def audit_manifests(manifests,output,max_projection_error_px=.1):
    """Actually rehash every observation and target and check GT frame consistency.
    Source image bytes are rehashed only if original_image is supplied. Absence
    is reported, not promoted to byte-level split independence.
    """
    seen_ids={};seen_images={};results=[];all_source_images_verified=True
    for path in manifests:
        ds=InstanceDataset(path);row={'path':str(ds.path),'split':ds.meta['split'],'records':len(ds),'symmetry_counts':{},'max_projection_error_px':0.,'all_source_images_rehashed':True}
        for i,r in enumerate(ds.records):
            if r['id'] in seen_ids:raise ValueError(f'ID crosses splits: {r["id"]}')
            seen_ids[r['id']]=str(ds.path)
            source_sha=r.get('source_image_sha256')
            if not source_sha or len(source_sha)!=64:raise ValueError('Source image SHA required even when bytes unavailable')
            if source_sha in seen_images and seen_images[source_sha]!=str(ds.path):raise ValueError('Image SHA crosses splits')
            seen_images[source_sha]=str(ds.path)
            if r.get('original_image'):
                if sha256(ds.resolve(r['original_image']))!=source_sha:raise ValueError('Source image content mismatch')
            else:row['all_source_images_rehashed']=False;all_source_images_verified=False
            obs=ds.observation(i);gt=ds.target(i)
            spec=gt.specs[0];spec.matrices(obs.dims[0]);validate_rotation(gt.R)
            if not (obs.features.shape[0]==1 and gt.points.shape==(1,9,2) and gt.valid.shape==(1,9)):
                raise ValueError('One instance per record')
            if not torch.isfinite(gt.points[gt.valid]).all():raise ValueError('Nonfinite supported GT')
            uv,z=project(cuboid(obs.dims),gt.R,gt.t,obs.K)
            delta=torch.linalg.vector_norm(uv-gt.points,dim=-1)[gt.valid]
            maximum=float(delta.max()) if delta.numel() else 0.
            row['max_projection_error_px']=max(row['max_projection_error_px'],maximum)
            if ds.meta['split']!='real_dev' and maximum>max_projection_error_px:
                raise ValueError(f'Object-frame/K/dimension/2D mismatch: {r["id"]}, {maximum:.4f}px')
            row['symmetry_counts'][str(spec.order)]=row['symmetry_counts'].get(str(spec.order),0)+1
        results.append(row)
    report={'schema':'plpose_v5_preflight_1','arithmetic_and_record_checks_pass':True,
            'records':results,'source_image_bytes_all_rehashed':all_source_images_verified,
            'full_source_data_parity_certified':False,
            'status':'PARTIAL_INPUT_VERIFICATION' if not all_source_images_verified else 'EXPORTED_INPUT_CHECKS_PASS',
            'limits':['Source image/feature alignment needs independent raw-image fixtures, not affine roundtrip only.',
                      'Does not certify physical symmetry, deployable dimension source, detector parity or unseen-session generalization.']}
    write_json(output,report);return report
