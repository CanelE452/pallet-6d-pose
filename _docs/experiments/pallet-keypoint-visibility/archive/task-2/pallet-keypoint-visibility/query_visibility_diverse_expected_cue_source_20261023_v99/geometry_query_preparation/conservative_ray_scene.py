"""Conservative per-owner ray/AABB broadphase over immutable V97 F64 kernel.
No renderer, RGB, ML, labels, or model input selection imports.
"""
from pathlib import Path
from collections import OrderedDict
import hashlib,importlib.util
import numpy as np
TEACHER=Path(__file__).resolve().parents[2]/'query_visibility_expected_cue_triplets_20261023_v97/cue_contract_preparation/endpoint_cue_teacher.py'
TEACHER_SHA='9e608adf76a8c00bca23b46799decdb5dfa0e67b09efba323c0b7509c0d895d3'
EPS=64*np.finfo(np.float64).eps
def load_teacher():
    assert hashlib.sha256(TEACHER.read_bytes()).hexdigest()==TEACHER_SHA
    spec=importlib.util.spec_from_file_location('v99_immutable_v97_teacher',TEACHER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class ConservativeRayScene:
    def __init__(self,triangles,owner,orientation=None,double_sided=None,foreign_owner_min=308,cache_results=True):
        self.T=load_teacher();self.xyz=np.asarray(triangles,np.float64);self.owner=np.asarray(owner,np.int32)
        self.orientation=np.ones(len(self.xyz))if orientation is None else np.asarray(orientation,np.float64)
        self.two=np.zeros(len(self.xyz),bool)if double_sided is None else np.asarray(double_sided,bool)
        self.foreign_owner_min=foreign_owner_min;self.ids=np.unique(self.owner);self.indices=[np.flatnonzero(self.owner==i)for i in self.ids]
        lo=np.array([self.xyz[a].min((0,1))for a in self.indices]);hi=np.array([self.xyz[a].max((0,1))for a in self.indices])
        pad=EPS*np.maximum(1,np.maximum(np.abs(lo),np.abs(hi)))
        self.lo=lo-pad;self.hi=hi+pad;self.scenes=OrderedDict();self.results={}if cache_results else None
        self.ray_count=0;self.triangle_tests=0;self.full_triangle_tests=0
    def candidate_indices(self,eye,direction):
        # Infinite forward ray: do not truncate at target and change a far-hitU.
        with np.errstate(divide='ignore',invalid='ignore'):
            a=(self.lo-eye)/direction;b=(self.hi-eye)/direction
        zero=direction==0
        outside=np.any(zero[None]&((eye<self.lo)|(eye>self.hi)),axis=1)
        enter=np.where(zero,-np.inf,np.minimum(a,b)).max(1);leave=np.where(zero,np.inf,np.maximum(a,b)).min(1)
        slack=EPS*np.maximum(1,np.maximum(np.where(np.isfinite(enter),np.abs(enter),1),np.where(np.isfinite(leave),np.abs(leave),1)))
        hit=(~outside)&(leave>=-slack)&(enter<=leave+slack)
        if not hit.any():return np.empty(0,np.int64)
        return np.sort(np.concatenate([self.indices[i]for i in np.flatnonzero(hit)]))
    def cast(self,eye,points,expected_owner):
        eye=np.asarray(eye,np.float64);points=np.asarray(points,np.float64).reshape(-1,3)
        key=(eye.tobytes(),points.tobytes(),int(expected_owner))
        if self.results is not None and key in self.results:return self.results[key]
        output=[]
        for p in points:
            direction=p-eye;length=float(np.linalg.norm(direction))
            if not np.isfinite(length)or length<=0:output.append(dict(state='U',owners=[],margin=None));continue
            index=self.candidate_indices(eye,direction/length);subkey=index.tobytes()
            if subkey not in self.scenes:
                self.scenes[subkey]=self.T.RayScene(self.xyz[index],self.owner[index],self.orientation[index],self.two[index],self.foreign_owner_min)
                if len(self.scenes)>64:self.scenes.popitem(last=False)
            else:self.scenes.move_to_end(subkey)
            output.extend(self.scenes[subkey].cast(eye,p[None],expected_owner));self.ray_count+=1;self.triangle_tests+=len(index);self.full_triangle_tests+=len(self.xyz)
        if self.results is not None:self.results[key]=output
        return output
