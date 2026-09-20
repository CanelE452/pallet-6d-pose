"""Audit the fixed replay ablation before considering any further run."""
import json
from pathlib import Path
from . import recovery_common as R
from .recovery_summary import detection_parity,clustered_pck_contrast
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM
C=R.C


def main():
    protocol=C.read(R.DOC/'pose_real_only/PROTOCOL.json')
    for b in protocol['sources']+protocol['inputs']:C.verify(b)
    base=[r for r in C.read(R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC']
    baseline={r['id']:r for r in base};metadata={r['id']:r['session'] for r in C.read(R.BASE_DOC/'EVAL_PROTOCOL.json')['records']}
    runs={};metrics={};sources=[]
    configs=[('pose_only','SYN_LR5'),('pose_only','RAW_LR5'),('pose_only','REF_LR5'),
        ('pose_real_only','RAW_REAL_ONLY'),('pose_real_only','REF_REAL_ONLY')]
    for phase,arm in configs:
        path=R.RAW/phase/f'SCREEN_{arm}.json';r=C.read(path);metrics[arm]=r['metrics']
        runs[arm]=dict(PCK20=100*r['symmetry']['PCK']['20'],median8=r['symmetry']['matched_pooled_corner8_median_px'],
            P90_8=r['symmetry']['matched_pooled_corner8_P90_px'],pose=r['pose'],
            damage=EM.damage([baseline[x['id']] for x in r['metrics']],r['metrics']))
        sources.append(C.bound(path))
        if phase=='pose_real_only':
            fit=C.read(R.DOC/phase/f'FIT_{arm}.json');C.verify(fit['checkpoint'])
            assert fit['optimizer_steps']==320 and fit['protected_state_exact']
            runs[arm]['detector_parity_194']=detection_parity(phase,arm)
    old,new=runs['REF_LR5'],runs['REF_REAL_ONLY']
    flags=dict(PCK20_above_previous_REF=new['PCK20']>old['PCK20'],PCK20_above_R0=new['PCK20']>100*EM.summary(base)['PCK']['20'],
        IoU3D_preserved=new['pose']['iou3d_median']>=old['pose']['iou3d_median'],matched8_P90_preserved=new['P90_8']<=old['P90_8'])
    for target in ['RAW','REF']:
        slots=(C.ROOT/protocol['datasets'][target]['train_list']['path']).read_text().splitlines()
        assert len(slots)==1024 and len(set(slots))==217 and all(Path(x).name.startswith('PLASTIC__') for x in slots)
    a,b=[C.read(R.DOC/'pose_real_only'/f'TRACE_{t}_REAL_ONLY.json') for t in ['RAW','REF']]
    assert a['image_tensor_sha256']==b['image_tensor_sha256']
    contrasts={name:clustered_pck_contrast(metrics['REF_REAL_ONLY'],other,metadata) for name,other in
        [('R0',base),('SYN_LR5',metrics['SYN_LR5']),('REF_LR5',metrics['REF_LR5']),('RAW_REAL_ONLY',metrics['RAW_REAL_ONLY'])]}
    report=dict(runs=runs,flags=flags,proceed_to_repeats=all(flags.values()),exploratory_clustered_contrasts=contrasts,
        unique_real_images=217,real_slots_per_epoch=1024,synthetic_train_slots=0,
        original_real_multiset_doubled=True,exposure_change_is_part_of_intervention=True,
        no_new_real_GT=True,existing_teacher_had_manual_real_supervision=True,full194_includes_teacher3=True,
        negative2689_not_run_in_this_screen=True,paper_fixed9_metrics_not_run_in_this_screen=True,
        independent_confirmation=False,auto_promote=False,
        sources=[C.bound(__file__),C.bound(R.DOC/'pose_real_only/PROTOCOL.json')]+sources)
    C.freeze(R.RAW/'pose_real_only/SCREEN_AUDIT.json',report)
    lines=['# 합성 replay 제거 통제 실험','',
        '기존 R0에서 독립 초기화, 검출·backbone·BN 동결, pose-only lr1e-5,5epochs/320steps. 기존217장과 코너 마스크·보정 좌표는 동일하다. 원래 실사512노출 multiset을 두 번 사용하여 실사1024, 합성0으로 바꿨다. 실사 노출 두 배 효과까지 포함한 비교이며 순수한 gradient 충돌만의 인과 증명은 아니다.','',
        '| 모델 | PCK20 %↑ | 대칭8점 median↓ | 대칭8점 P90↓ | IoU3D↑ |','|---|---:|---:|---:|---:|']
    for name,m in runs.items():lines.append(f"| {name} | {m['PCK20']:.3f} | {m['median8']:.3f} | {m['P90_8']:.3f} | {m['pose']['iou3d_median']:.5f} |")
    lines+=['','사전 반복 진행 조건: '+json.dumps(flags,ensure_ascii=False),
        '','반복 진행 조건 충족: '+str(report['proceed_to_repeats']),
        '','세션-cluster bootstrap은 반복 사용 DEV에 대한 탐색적 진단이고 검색 횟수 보정은 하지 않았다. full194에는 과거 보정기 학습3장도 포함된다. 독립 확인 또는 확정된 일반화 성능 주장은 없다.',
        '','```json',json.dumps(contrasts,ensure_ascii=False,indent=2),'```','',
        '전체194 검출 box/score exact parity와 동결 상태·320steps를 확인했다. 이 screen은 NEG2689 및 논문 fixed-index9점 지표를 새로 실행하지 않았다. 공식C2 평가, 기존 수도레이블, 최종 모델은 변경하지 않았다.','']
    C.write_text(R.RAW/'pose_real_only/REPORT_KO.md','\n'.join(lines))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
