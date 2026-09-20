"""Report the predeclared flip contrast, including failed arms and tail guards."""
import json
from . import recovery_common as R
from .recovery_summary import detection_parity
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM
C=R.C


def main():
    p=C.read(R.DOC/'pose_flip/PROTOCOL.json')
    for b in p['sources']+p['inputs']:C.verify(b)
    assert C.sha(p['installed_augmentation']['path'])==p['installed_augmentation']['sha256']
    base={r['id']:r for r in C.read(R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    arms={};bindings=[]
    for target in ['SYN','RAW','REF']:
        pair={}
        for phase,name in [('pose_only',target+'_LR5'),('pose_flip',target+'_FLIP')]:
            path=R.RAW/phase/f'SCREEN_{name}.json';r=C.read(path)
            pair[name]=dict(PCK20=100*r['symmetry']['PCK']['20'],median8=r['symmetry']['matched_pooled_corner8_median_px'],
                P90_8=r['symmetry']['matched_pooled_corner8_P90_px'],IoU3D=r['pose']['iou3d_median'],
                damage=EM.damage([base[x['id']] for x in r['metrics']],r['metrics']))
            bindings.append(C.bound(path))
        fit=C.read(R.DOC/'pose_flip'/f'FIT_{target}_FLIP.json');C.verify(fit['checkpoint'])
        assert fit['protected_state_exact'] and fit['optimizer_steps']==320
        pair['detector_parity_194']=detection_parity('pose_flip',target+'_FLIP')
        arms[target]=pair
    a,b=arms['REF']['REF_LR5'],arms['REF']['REF_FLIP']
    flags=dict(PCK20_above_previous_REF=b['PCK20']>a['PCK20'],IoU3D_preserved=b['IoU3D']>=a['IoU3D'],
        matched8_P90_preserved=b['P90_8']<=a['P90_8'],PCK20_above_R0=b['PCK20']>100*C.read(R.BASE_DOC/'RESULTS.json')['summaries']['R0']['PLASTIC']['PCK']['20'])
    raw_trace=C.read(R.DOC/'pose_flip/TRACE_RAW_FLIP.json');ref_trace=C.read(R.DOC/'pose_flip/TRACE_REF_FLIP.json')
    assert raw_trace['image_tensor_sha256']==ref_trace['image_tensor_sha256']
    result=dict(arms=arms,predeclared_flags=flags,proceed_to_repeats=all(flags.values()),
        pipeline_has_previous_teacher_real_supervision=True,full194_has_three_teacher_training_images=True,
        independent_holdout_confirmation=False,negative_inference_not_run_for_this_screen=True,
        matching_input_image_batch_RAW_REF=True,configuration_change_only_fliplr=True,auto_promote=False,
        sources=[C.bound(__file__),C.bound(R.DOC/'pose_flip/PROTOCOL.json')]+bindings)
    C.freeze(R.RAW/'pose_flip/SCREEN_AUDIT.json',result)
    lines=['# 좌우 반전 self-training 통제 실험','',
        '기존의 보정 수도레이블 및 필터는 그대로 두고 fliplr만 0→0.5로 바꿨다. 일반 플라스틱 고유217장, 합성512장, R0 초기화, pose-only lr1e-5, 고정5epoch/320steps. 반전 좌표·번호·ignore mask·이중 반전 복원 계약검사3개 통과.','',
        '앞선 준비 과정에서 중복 controls 키 때문에 프로토콜 생성이 실패했고, 첫 학습 호출도 프로토콜을 읽기 전에 종료했다. 수정 후 프로토콜을 고정하고 아래 세 학습만 수행했다. 실패한 호출에서 optimizer update는 없었다.','',
        '| 학습 대상 | 반전 | PCK20 %↑ | 대칭8점 median px↓ | 대칭8점 P90 px↓ | IoU3D↑ |','|---|---|---:|---:|---:|---:|']
    for target,pair in arms.items():
        for name in [target+'_LR5',target+'_FLIP']:
            q=pair[name];lines.append(f"| {target} | {'on' if name.endswith('FLIP') else 'off'} | {q['PCK20']:.3f} | {q['median8']:.3f} | {q['P90_8']:.3f} | {q['IoU3D']:.5f} |")
    lines+=['','## 사전 판정 조건','',json.dumps(flags,ensure_ascii=False),
        '',f'반복 실험 진행 조건 충족: {all(flags.values())}. 유리한 지표만 골라 채택하지 않는다.','',
        '전체194장 검출 box/score는 R0와 exact parity이고 보호된 가중치·buffer는 그대로다. 이 screen에서는 negative2689 추론 및 논문 fixed-index9점 지표를 새로 실행하지 않았다. 위 P90은 대칭8점 지표이며 논문9점 P90과 혼동하면 안 된다.','',
        '이 집합은 반복 사용 DEV이고 과거 보정기 학습3장이 포함된다. 독립 평가 성공 또는 최종 모델 교체 주장은 없다. 공식 C2 대칭, GT annotation, 기존 수도레이블은 바꾸지 않았다.','',
        'FixMatch는 변형된 입력에서 수도레이블 학습을 하는 동기일 뿐, 여기서 그 방법을 재현했다고 주장하지 않는다: https://arxiv.org/abs/2001.07685','']
    C.write_text(R.RAW/'pose_flip/REPORT_KO.md','\n'.join(lines))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
