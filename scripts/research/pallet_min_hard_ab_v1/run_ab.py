"""Resume verified stages in isolated Python processes, without restarting fits."""
import subprocess
import sys
from . import common as C


STAGES=[('teacher','TEACHER_HARD_PREDICTION_LOCK.json'),('prepare_training','TRAIN_OCCURRENCE_LOCK.json'),
            ('test_training','PRETRAIN_TESTS.json'),('train:H_PSEUDO','FIT_H_PSEUDO.json'),
            ('train:H_MANUAL','FIT_H_MANUAL.json'),('infer','POSE_DECISIONS_LOCK.json'),
            ('evaluate','DECISION.json'),('final_report',None),('completion_audit',None),
            ('directive_audit',None)]


def verify_bindings(value):
    if isinstance(value,dict):
        if {'path','sha256'}<=set(value):C.verify(value)
        else:
            for v in value.values():verify_bindings(v)
    elif isinstance(value,list):
        for v in value:verify_bindings(v)


def stage_done(stage,artifact):
    if not artifact or not (C.DOC/artifact).exists():return False
    data=C.read(C.DOC/artifact);verify_bindings(data)
    if stage.startswith('train:'):
        assert data['complete'] and data['steps']==320 and len(data['epochs'])==5
    if stage=='test_training':assert data['passed']
    if stage=='evaluate':
        for n in ('RESULTS.json','TRANSITIONS.json','TRAIN_FIT.json','SOURCE_PRESERVATION.json','VERIFIED_VISIBLE.json'):
            assert (C.DOC/n).is_file(),f'Incomplete scoring artifact: {n}'
    return True


def main(allow_compute=True):
    with C.exclusive('ab_pipeline'):
        if not allow_compute and (C.DOC/'FINALIZATION_INPUT_LOCK.json').exists():
            from .finalization_inputs import verify
            verify()
        return run(allow_compute)


def run(allow_compute):
    label=C.read(C.DOC/'HARD_LABEL_LOCK.json');verify_bindings(label)
    events=[]
    for index,(stage,artifact) in enumerate(STAGES):
        if stage_done(stage,artifact):
            events.append(dict(stage=stage,action='VERIFIED_SKIP'));continue
        if artifact and not allow_compute:
            raise RuntimeError(f'Finalization forbids training/inference/scoring; missing stage: {stage}')
        if artifact:
            later=[name for _,name in STAGES[index+1:] if name and (C.DOC/name).exists()]
            if later:raise RuntimeError(f'Missing upstream marker {artifact} with completed downstream {later}; preserve artifacts, do not regenerate history')
        if stage.startswith('train:') and (C.RAW/'runs'/stage.split(':')[1]).exists():
            raise RuntimeError(f'Interrupted fit {stage}: preserve run; no silent restart or overwrite')
        if stage.startswith('train:'):
            arm=stage.split(':')[1];cmd=[sys.executable,'-u','-c',f'from scripts.research.pallet_min_hard_ab_v1.train import main; main({arm!r})']
        else:cmd=[sys.executable,'-u','-m','scripts.research.pallet_min_hard_ab_v1.'+stage]
        print('AB_STAGE',stage,flush=True);subprocess.run(cmd,cwd=C.ROOT,check=True)
        events.append(dict(stage=stage,action='EXECUTED'))
    C.save(C.OUT/'LAST_RESUME.json',dict(created_at=C.now(),allow_compute=allow_compute,events=events))
    return events


if __name__=='__main__':main()
