"""NEW export contract, not inferred names of existing repository NPZ/JSON keys."""
from __future__ import annotations
from dataclasses import fields
from functools import lru_cache
from pathlib import Path
import hashlib,json
import torch
from .model import Observation

SCHEMA='pointline_v4_export_1'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def tensor_state_sha(state):
    h=hashlib.sha256()
    for key,tensor in sorted(state.items()):
        h.update(key.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.pending')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(path)


def collate_observations(items):
    data={}
    for field in fields(Observation):
        vals=[getattr(o,field.name) for o in items]
        if isinstance(vals[0],(tuple,list)):
            data[field.name]=tuple(torch.cat([v[i] for v in vals],0) for i in range(len(vals[0])))
        else:data[field.name]=torch.cat(vals,0)
    return Observation(**data)


class ExportDataset:
    """Each observation file contains B=1 tensors. GT is a SEPARATE file.

    File digests are checked before use. Never silently resize candidates or
    feature planes; the exporter must pad with masks and bind the transform.
    """
    def __init__(self,manifest,*,verify_targets=False):
        self.path=Path(manifest).resolve();self.root=self.path.parent
        self.document=json.loads(self.path.read_text(encoding='utf8'))
        if self.document.get('schema')!=SCHEMA:raise ValueError('Wrong export schema')
        self.records=self.document['records'];self.channels=tuple(self.document['channels'])
        if not self.records or len({r['frame_id'] for r in self.records})!=len(self.records):
            raise ValueError('Empty export or duplicate frame_id')
        for record in self.records:
            if record.get('state')!='native':raise ValueError('Main loader accepts native predictions only; stress states must stay diagnostic')
            self._verify(record,'observation')
            if verify_targets:self._verify(record,'supervision')
    def _path(self,record,key):
        path=(self.root/record[key]).resolve()
        if not path.is_relative_to(self.root):raise ValueError('Manifest paths must stay inside export root')
        return path
    def _verify(self,record,key):
        path=self._path(record,key)
        if sha256(path)!=record[key+'_sha256']:raise ValueError(f'Changed {key}: {record["frame_id"]}')
    def __len__(self):return len(self.records)
    @lru_cache(maxsize=32)
    def observation(self,index):
        # Deliberately never opens supervision or image annotations.
        obj=torch.load(self._path(self.records[index],'observation'),map_location='cpu',weights_only=True)
        o=Observation.from_mapping(obj);o.validate(self.channels)
        if o.baseline.shape[0]!=1:raise ValueError('Each record must hold B=1')
        return o
    @lru_cache(maxsize=32)
    def supervision(self,index):
        obj=torch.load(self._path(self.records[index],'supervision'),map_location='cpu',weights_only=True)
        if set(obj)!= {'points','supervised','matched'}:raise ValueError('Wrong separate supervision schema')
        if obj['points'].shape!=(1,9,2) or obj['supervised'].shape!=(1,9) or obj['matched'].shape!=(1,):
            raise ValueError('Invalid supervision shapes')
        if obj['supervised'].dtype!=torch.bool or obj['matched'].dtype!=torch.bool:raise TypeError('Boolean GT/match masks required')
        return obj
    def batch(self,indices,*,targets=False):
        obs=collate_observations([self.observation(int(i)) for i in indices])
        if not targets:return obs
        t=[self.supervision(int(i)) for i in indices]
        return obs,{k:torch.cat([x[k] for x in t],0) for k in t[0]}
