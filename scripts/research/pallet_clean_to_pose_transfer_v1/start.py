import subprocess
from pathlib import Path
from . import common as C

def run():
    p=C.DOC/'START.json'
    if p.exists():print('START_EXISTS');return
    def git(*args):return subprocess.check_output(['git',*args],cwd=C.ROOT,text=True).strip()
    directive=Path('/home/minjae/Downloads/pallet_clean_to_pose_transfer_goal_plan_cli.txt')
    status=git('status','--short','--branch')
    C.save(C.RAW/'START_GIT_STATUS_PRIVATE.txt',status+'\n',True)
    modified=C.ROOT/'data/evaluation/pallet_eval_v1/reports/ANNOTATION_PROGRESS.md'
    C.save(p,dict(start_at=C.now(),head=git('rev-parse','HEAD'),branch=git('branch','--show-current'),
        remote_head=git('rev-parse','origin/main'),recent_commits=git('log','-10','--oneline').splitlines(),
        directive=C.bind(directive),git_status=C.bind(C.RAW/'START_GIT_STATUS_PRIVATE.txt'),
        preserve_user_change=C.bind(modified)),True)
    C.save(C.DOC/'GOAL_LOCK.md','''# Clean → natural occlusion pose transfer

현재 217장 중 모델 출력과 무관하게 외부 가림 없는 영상만 고정한다. accepted249는 감사하지만 기존에 실제 사용하지 않은 32장을 이번 primary TRAIN에 추가하지 않는다. SELF_OCCLUSION·TRUNCATION·HARD_VIEW는 외부 가림과 구분한다. 필터 통과나 confidence는 clean/정답 근거가 아니다.

자연 Plastic Moderate21+Severe78의 translation/full-rotation median/P90가 주 지표다. 성공 판정의 joint median 이득과 P90 손익을 함께 보고한다. Clean29 보존은 별도 공개한다. 평가 참조는 반복 DEV이며 독립 물리 GT로 승격하지 않는다.

Phase0–3 fit0. 적격 clean lock이 있을 때만 RAW/REF × CLEAR/random-OCC 네 학생을 각각320update로 비교한다. 기본 LR1e-5/pose+flow-only/R0/512source/512real 슬롯은 현재main을 따른다. 같은 seed의 order/base RGB/box/support를 검사하고 target 좌표 또는 추가 입력 가림 한 축만 다르게 한다.

조건부 selector 최대1, bridge 최대2cycle(각 RAW/REF pair), 유망 recipe의 실효 seed 반복만 허용한다. 학생fit 총10/GPU학습6시간 상한. 근거없는 sweep·새label·모델오차 기반 TRAIN 선택·기존 결과 덮어쓰기는 금지한다. 깨끗한 집합을 신뢰성 있게 확정하지 못하면 학습하지 않고 검토자료와 필요한 사용자 확인을 남긴다.

기존 annotation progress 수정과 무관한 미추적 자료는 보존한다. 공개에는 집계/코드/허용된 그림만 포함하고 원본 RGB·private 좌표·checkpoint는 포함하지 않는다.
''',True)
    C.save(C.DOC/'RESOURCE_LEDGER.json',dict(caps=dict(student_fits=10,selector_fits=1,GPU_training_seconds=21600),events=[],
        totals=dict(student_fits=0,selector_fits=0,GPU_training_seconds=0,optimizer_updates=0)),True)
    C.save(C.DOC/'STATE.json',dict(stage='PHASE_0_3_NO_FIT_AUDIT',at=C.now(),fits=0))
    print('START_LOCKED')

if __name__=='__main__':run()
