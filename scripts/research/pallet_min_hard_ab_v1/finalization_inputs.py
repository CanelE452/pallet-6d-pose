"""Protect completed scientific outputs while finishing presentation/tooling only."""
from . import common as C


def lock():
    path=C.DOC/'FINALIZATION_INPUT_LOCK.json'
    if path.exists():return C.read(path)
    names=('RESULTS.json','DECISION.json','FIT_H_MANUAL.json','FIT_H_PSEUDO.json',
           'SOURCE_PRESERVATION.json','VERIFIED_VISIBLE.json','TRAIN_FIT.json','TRANSITIONS.json',
           'HARD_LABEL_LOCK.json','TEACHER_HARD_PREDICTION_LOCK.json','TRAIN_OCCURRENCE_LOCK.json',
           'RAW_PREDICTIONS_LOCK.json','POSE_DECISIONS_LOCK.json','PRETRAIN_TESTS.json','SCORING_START.json')
    files={str(C.DOC/n):C.bind(C.DOC/n) for n in names}
    for n in names:
        d=C.read(C.DOC/n)
        for v in d.values():
            if isinstance(v,dict) and {'path','sha256'}<=set(v):files[v['path']]=v
    value=dict(created_at=C.now(),head=C.git('rev-parse','HEAD'),new_training_allowed=False,
               new_annotation_allowed=False,files=list({b['path']:b for b in files.values()}.values()))
    C.save(path,value,immutable=True);return value


def verify():
    data=lock()
    for b in data['files']:C.verify(b)
    return len(data['files'])


if __name__=='__main__':print('PROTECTED_SCIENTIFIC_ARTIFACTS',verify())
