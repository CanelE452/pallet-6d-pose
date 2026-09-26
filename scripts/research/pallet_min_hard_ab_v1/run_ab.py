"""Resume verified stages in isolated Python processes, without restarting fits."""
import subprocess
import sys
from . import common as C


def main():
    assert (C.DOC/'HARD_LABEL_LOCK.json').exists()
    stages=[('teacher','TEACHER_HARD_PREDICTION_LOCK.json'),('prepare_training','TRAIN_OCCURRENCE_LOCK.json'),
            ('test_training','PRETRAIN_TESTS.json'),('train:H_PSEUDO','FIT_H_PSEUDO.json'),
            ('train:H_MANUAL','FIT_H_MANUAL.json'),('infer','POSE_DECISIONS_LOCK.json'),
            ('evaluate','DECISION.json'),('final_report',None),('completion_audit',None)]
    for stage,artifact in stages:
        if artifact and (C.DOC/artifact).exists():continue
        if stage.startswith('train:'):
            arm=stage.split(':')[1];cmd=[sys.executable,'-u','-c',f'from scripts.research.pallet_min_hard_ab_v1.train import main; main({arm!r})']
        else:cmd=[sys.executable,'-u','-m','scripts.research.pallet_min_hard_ab_v1.'+stage]
        print('AB_STAGE',stage,flush=True);subprocess.run(cmd,cwd=C.ROOT,check=True)


if __name__=='__main__':main()
