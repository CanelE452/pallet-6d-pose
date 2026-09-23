import subprocess
import inspect
from pathlib import Path
import numpy as np
from . import common as E

def main():
    assert all(not p.exists() for p in [E.DOC,E.RAW,E.OUT])
    for p in [E.DOC,E.RAW,E.OUT]:p.mkdir(parents=True)
    E.C.immutable()
    files=[E.C.H.C.N.E.R0,Path(E.D.Pose.__file__),Path(E.D.Selector.__file__),E.ROOT/'scripts/self_training_yolo/v3/true_ignore_pose_loss.py',E.C.H.P.DOC/'SPLIT.json',E.C.DOC/'AUGMENTATION_PLAN.jsonl',E.C.DOC/'TARGETS.json']
    for folder in [E.V.DOC,E.V.RAW,E.ROOT/'scripts/research/pallet_clean19_pose_sensitive_v1']:
        files.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in str(p))
    files.extend(E.C.DOC.glob('*.json'))
    for mat in E.MATS:
        fit=E.read(E.C.DOC/f'FIT_{mat}_S1.json');files.extend([E.ROOT/fit['checkpoint']['path'],E.C.RAW/f'TRACE_{mat}_S1.json'])
    for b in E.read(E.V.DOC/'INPUT_BINDINGS.json')['files']:E.verify(b)
    binding=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),files=[E.bind(p) for p in sorted(set(files))])
    E.save(E.DOC/'INPUT_BINDINGS.json',binding)
    E.save(E.RAW/'WORKTREE_BEFORE.json',dict(status=subprocess.check_output(['git','status','--short','--branch'],text=True)))
    E.save(E.DOC/'ACCEPTANCE_RULE.json',dict(gradient_atol=E.ATOL,gradient_rtol=E.RTOL,loss_atol=E.ATOL,loss_rtol=E.RTOL,predeclared_before_runs=True,seed=42,train_only_first_synthetic_microbatch=True))
    pp=E.plans();stats={};tests={};corners=[]
    for mat in E.MATS:
        old=np.load(E.V.RAW/f'H_MODEL_{mat}.npz');W=[];enabled=[];single=True;maximum_mean_error=0.;max_weight=0.;validweights=[];audit=[]
        records=[r for r in pp if r['material']==mat]
        assert len(records)==5120
        for i,r in enumerate(records):
            E.verify(r['cache'])
            with np.load(E.ROOT/r['cache']['path']) as z:
                assert len(z['keypoints'])==len(z['cls'])==len(z['bboxes'])==1
                mask=np.repeat(z['keypoints'][0,:8,2]==2,2)
                assert E.C.digest(z['keypoints'])==r['target_sha256']
            w=np.diag(old['H'][i]).copy();en=bool(old['enabled'][i])
            assert np.isfinite(w).all() and (w>=0).all()
            assert (w[~mask]==0).all()
            if r['real']:assert (w==0).all() and not en
            if en:
                err=abs(float(w[mask].mean())-1);maximum_mean_error=max(maximum_mean_error,err)
                assert err<=1e-6
                validweights.extend(w[mask].tolist());max_weight=max(max_weight,float(w.max()))
            W.append(w);enabled.append(en)
            audit.append(dict(occ=i,epoch=r['epoch'],slot=r['slot'],real=r['real'],cache_sha256=r['cache']['sha256'],target_sha256=r['target_sha256']))
        path=E.RAW/f'WEIGHTS_{mat}.npz';np.savez(path,w=np.array(W),enabled=np.array(enabled))
        stats[mat]=dict(enabled=sum(enabled),disabled_synthetic=sum(not en and not r['real'] for en,r in zip(enabled,records)),real_zero=sum(r['real'] for r in records),
            min=float(min(validweights)),max=max_weight,median=float(np.median(validweights)),mean_valid_error_max=maximum_mean_error,weights=E.bind(path),single_object_all=True)
        for j in range(16):
            vv=[w[j] for w,en in zip(W,enabled) if en and w[j]>0]
            corners.append(dict(material=mat,corner=j//2,coordinate='xy'[j%2],median=float(np.median(vv)) if vv else None,n=len(vv)))
        E.save(E.RAW/f'OCCURRENCE_BINDINGS_{mat}.json',audit)
    E.save(E.DOC/'POSE_SENSITIVITY_WEIGHTS.json',dict(material=stats,corner_xy= corners,source='diag of frozen v1 trace-normalized model H; no offdiagonal',real_all_zero=True,ignored_all_zero=True,all_10240_occurrences_single_object=True,geometry_binding=E.bind(E.V.DOC/'SYNTH_GEOMETRY_BINDING.json'),projection=E.bind(E.V.DOC/'SYNTH_PROJECTION_PARITY.json')))
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.utils.torch_utils import init_seeds
    E.save(E.DOC/'TRAINER_SOURCE_CONTRACT.json',dict(init_seeds=inspect.getsource(init_seeds),trainer_init=inspect.getsource(BaseTrainer.__init__),setup_train=inspect.getsource(BaseTrainer._setup_train)))
    E.save(E.DOC/'PREFLIGHT_AUDIT.md','# Diagonal sensitivity preflight\n\nHEAD: `'+binding['head']+'`; branch '+binding['branch']+'\n\nPrevious v1 immutable; production D9 unchanged; no D8 exploration. Gradient tolerance fixed before results: atol=1e-5, rtol=1e-4. OLD/OLD/NEW_M0 fresh R0 forward+backward once each, TRAIN first synthetic microbatch. 4 fits max1280steps; no rescue. v1 geometry512 and occurrence projection reused with file hashes; all10240 cached examples are single-object.\n')
    print('PREFLIGHT_READY',stats,flush=True)

if __name__=='__main__':main()
