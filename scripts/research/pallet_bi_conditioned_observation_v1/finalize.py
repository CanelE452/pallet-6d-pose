"""Close report prose, actual display review and cost metadata before pack audit."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
from PIL import Image
from scripts.research.pallet_risk_information_v1.common import read,write,sha,csv_write
from .artifacts import clean,number


def run(root,public,repo):
    root,public,repo=Path(root),Path(public),Path(repo)
    summary=read(root/'evaluation/SUMMARY.json');pairs=read(root/'evaluation/PAIRED.json');obs=read(root/'evaluation/OBSERVATION_SUMMARY.json')
    representative={state:{arm:next(v for v in obs if v['population']=='PRIMARY192' and v['arm']==arm and v['automatic_state']==state)
        for arm in ('OBS_COMMON_s20261018','OBS_BI_s20261018')} for state in ('B','I')}
    B=representative['B'];I=representative['I'];C='OBS_COMMON_s20261018';BI='OBS_BI_s20261018'
    interpretation='''판정은 **명시적 B/I 조건의 추가 가치 미확보, 새 scorer 채택 보류**다. 세 seed의 COMMON 대비 BI 성공 수 변화는0,0,−1장이며, 회복/손상은4/4,6/6,2/3이다. 대표 seed를 최고 seed로 바꾸지 않았다. N3·SubPix 및 현재 운영 기본값을 유지하고 논문을 자동 수정하지 않는다.

## 관측은 좋아졌고 자세도 좋아졌나

'''
    interpretation+=f'대표 seed의 B full 참조 {B[C]["full_point_n"]}점에서 q0 오차 평균은{number(B[C]["before"]["mean"])}px, COMMON {number(B[C]["after"]["mean"])}px, BI {number(B[BI]["after"]["mean"])}px였다. I {I[C]["full_point_n"]}점에서는 각각{number(I[C]["before"]["mean"])}→{number(I[C]["after"]["mean"])} / {number(I[BI]["after"]["mean"])}px다. BI의 I 오차는 COMMON보다 소폭 낮고 B는 소폭 높다. 자동 B/I 조건이 두 상태 모두에서 개선한 결과는 아니다. 부분선106코너는 full점 오차 분모에서 분리했다.\n\n'
    interpretation+='''가상 투영 참조에서 q0보다 좋은 후보는 B583/802점·I35/77점이었다. 같은 제한된 후보 안의 사후 최소 오차도 B10.101px·I13.745px이며, 큰 초기 대응/위치 오류까지 작은 국소 이동으로 복구할 수 있다는 근거는 없다. 이 최소값을 새 선택기의 검증 성능으로 쓰지 않는다. 대표 두 군의 적용좌표가 다른 영상111장 중 BI의 평균 코너 오차가 낮은 것은51장이고, 그중 자세 위치 오차가 높아진 것은11장, 회전 오차가 높아진 것은19장이다. 관측 위치 품질과6D 오차의 개선은 일치하지 않았다.

공통 모델에도 역할을 추론할 기하·마스크·두 cue가 있으므로 c는 새 센서 정보를 더하지 않는다. 이 설계에서 도움이 될 수 있는 것은 조건의 귀납적 편향이며, 이론적으로 성능 향상을 보장하지 않는다. 학습 목표는 개별 가상 코너의 위치 품질이고 공동6D 오차·물리 선 소속을 직접 감독하지 않는다. 이것들이 병목 후보라는 해석과 이번 데이터에서 확인한 수치를 구분한다.

## 무엇을 먼저 의심해야 하나

RAW matched와 N0는 모두108/192이지만14회복·14손상이고179개의 실제 자세가 달라졌다. 원좌표만 써도 공통 후단으로 바꾸면 위치 평균이11.306→14.149cm, 회전 평균이10.631→13.077°가 된다. 새 보정의 결과를 원 F와 같은 후단 위의 개선으로 해석할 수 없다. 대표 COMMON/BI의22fallback과 rank 분포는 같고4회복/4손상은 양쪽에서 수락된170장 안에서 생겼다. 따라서 fallback 수만 줄이면 해결된다고 결론 내리지 않는다.

합성 validation235에서 기하 역할 참조로 조건만 바꿔도44→44였다. 적용 태그가 달라진78장 중 실제 관측이 바뀐 것은3장이다. 투영 좌표만 교체하면44→69로 성공은 늘지만 위치 평균25.289→30.322cm·최대511.834→1105.873cm로 악화했다. 영상 밖의 참조점21건은 기존 guard가 거부했고 분모에 남겼다. 정확한 subset·순환성·C1·물리 주석 부재는 [DIAGNOSTIC_RESULTS](DIAGNOSTIC_RESULTS.md)에 명시했다. 실제 선 정답이나 완벽한 실사 상태를 준 실험으로 바꾸어 부르지 않는다.

다음 수정 우선순위를 한 가지로 좁히면 **원 관측을 전달했을 때도 손상이 생기는 공통 자세 후단의 수용·W/D 분기 안정성**이다. 관측 조건망을 확대하기 전에 이 경로의 손상을 분리해야 한다. 이는 저장 결과의 진단에서 나온 제안이며, 새 guard나 정책의 성능을 검증한 결과가 아니다. 이번 작업에서는 추가 학습·임계값 탐색·새 캠페인을 시작하지 않았다. [고정 출력 원인 분해](CAUSE_DECOMPOSITION.md)에 실제 family·부분선·rank·source와 예시 ID를 남겼다.

'''
    path=public/'REPORT_KO.md';text=path.read_text()
    marker='결과 판정과 남은 병목은 아래 해석 문단을 따른다. 원 계획·소스·전체 분모·실패를 보존했다.\n\n'
    if marker in text:text=text.replace(marker,interpretation,1)
    elif '## 관측은 좋아졌고 자세도 좋아졌나' not in text:raise ValueError('Unexpected report base')
    coverage='분할 전체768/235장 중 fullpoint 감독이 있는 프레임은532/171장이었다(합계703/1003). 나머지300장은 미관측 또는 고정 부분선 경로로 위치 점수망 loss를 만들지 않았으며, 자료 분모에서 삭제하거나0오차 target으로 채우지 않았다. 유효 후보 target은8048개이고 fixed partial504개는 이 loss에서 제외했다. 실제90epoch 원 곡선은 [TRAINING_CURVES.csv](TRAINING_CURVES.csv)에 있다.\n\n'
    if coverage not in text:text=text.replace('원 v2의 두 cue를',coverage+'원 v2의 두 cue를',1)
    path.write_text(text)
    readme=public/'README.md';text=readme.read_text()
    verdict='**결론: B/I 조건의 추가 가치 미확보.** PRIMARY192에서 BI의 COMMON 대비 회복/손상은 seed별4/4,6/6,2/3이었다. 6fit과 실제 자세 계산·독립 검산을 완료했고 모든 실패를 보존했다. 운영 기본값은 바꾸지 않았다.\n\n'
    if verdict not in text:text=text.replace('# B·I 조건부 국소 관측 보정\n\n','# B·I 조건부 국소 관측 보정\n\n'+verdict,1)
    readme.write_text(text)
    training_curves=[]
    for folder in sorted((root/'training').iterdir()):
        receipt=read(folder/'RUN_RECEIPT.json')
        for entry in read(folder/'HISTORY.json'):
            training_curves.append(dict(arm=receipt['arm'],seed=receipt['seed'],epoch=entry['epoch_index'],
                train_loss=entry['train']['loss'],validation_loss=entry['validation']['loss'],
                train_optimizer_loss=entry['training_optimizer_loss'],training_order_sha256=entry['training_order_sha256'],
                selected_checkpoint=entry['epoch_index']==receipt['best_epoch'],
                supervised_train_frames=entry['train']['supervised_frames'],full_train_frames=entry['train']['full_frame_denominator'],
                supervised_validation_frames=entry['validation']['supervised_frames'],full_validation_frames=entry['validation']['full_frame_denominator']))
    curve_path=public/'TRAINING_CURVES.csv'
    if not curve_path.exists():csv_write(curve_path,training_curves)
    costs=read(public/'EXECUTION_AND_COST.json')
    costs['technical_attempts']=clean([read(p) for p in sorted((root/'execution').glob('*/RECEIPT.json'))])
    costs['diagnostic_processes']=read(public/'DIAGNOSTIC_SUMMARY_PUBLIC.json')['cost']
    costs['diagnostic_seal']=clean(read(root/'diagnostics_v1/DIAGNOSTIC_SEAL.json'))
    costs['scientific_prediction_and_diagnostic_optimizer_calls']=(costs['blind_prediction_manifest']['actual_new_pose_optimizer_calls']+
        costs['diagnostic_seal']['actual_pose_optimizer_calls'])
    costs['technical_history']=clean(dict(extraction=[read(root/'TECHNICAL_CORRECTION_1.json'),read(root/'TECHNICAL_CORRECTION_2.json')],
        guard_repair=read(root/'solver_contract/PREDICT_SCIPY_GUARD_REPAIR.json'),display_repair=read(root/'GALLERY_DISPLAY_REPAIR.json')))
    write(public/'EXECUTION_AND_COST.json',costs,overwrite=True)
    # Only already-generated public display metadata is needed here.
    gallery=read(public/'gallery/GALLERY_MANIFEST.json')
    viewed=[f'gallery/contact_{i:02d}.jpg' for i in range(8)]+['gallery/crops/B.png','gallery/crops/I.png',
        *[f'gallery/details/{n}.png' for n in ('429c4a249f891bab','580b7210b7d37c93','c8aa9a43c196fa14','d59471a8a34e62a5')],
        'figures/six_training_curves.png','figures/pose_position_rotation_scatter.png','figures/reference_replacement_diagnostics.png']
    for name in viewed:
        with Image.open(public/name) as image:image.load()
    visual=dict(schema='BI_FINAL_VISUAL_REVIEW_V1',automatic_decode='PASS',root_model_pixel_view=True,
        actual_tool='view_image, actual final raster bytes',viewed_files={name:sha(public/name) for name in viewed},
        primary_images_reviewed_as_contact_thumbnails=192,contact_sheets=8,full_detail_reviewed=4,
        findings=['Caption overlap found in first display pass and corrected without scientific changes.',
            'Gray alternatives, cyan applied full observations, magenta partial normals, solid model projections are distinct.',
            'Large accepted errors and unchanged both-failure case retained; no masking of numeric extremes.',
            'General WD native-index projection repaired; exact1904 native projection audit passes.'],
        browser=dict(launcher='xdg-open existing browser session',HTTP_index=200,HTTP_visible_thumbnail_resources=200,
            actual_active_window_title='BI local observation gallery - Google Chrome',private_desktop_screenshot_sha256=sha(root/'visual_review/browser_open.png'),
            screenshot_uploaded=False,root_model_viewed_actual_browser_screenshot=True),
        human_review_performed=False,human_review_status='No independent human review claimed',RGB_rights='User confirmed all own-shot primary result overlays may be public',
        privacy_plan_sha256=gallery['privacy_plan_sha256'],display_only_redactions=True)
    write(public/'VISUAL_REVIEW.json',visual,overwrite=True)
    gallery['pixel_decode_status']='FINAL_MODEL_PIXEL_REVIEW_RECORDED_SEPARATELY_IN_VISUAL_REVIEW'
    write(public/'gallery/GALLERY_MANIFEST.json',gallery,overwrite=True)
    write(public/'RIGHTS_AND_PRIVACY.json',dict(public_RGB_derivatives_authorized=True,authorization='Explicit user reply: all directly shot, result panels may be public',
        original_RGB_uploaded=False,private_bank_or_annotation_or_weight_upload=False,primary_derivative_frames=192,
        redacted_frames=sum(bool(f['display_only_redactions']) for f in gallery['frames']),redaction_boxes=sum(len(f['display_only_redactions']) for f in gallery['frames']),
        inherited_privacy_plan_sha256=gallery['privacy_plan_sha256'],inference_image_changes=False,desktop_browser_screenshot_public=False),overwrite=True)
    comparison='''# 실행군의 정보·후단 차이

| 경로 | 초기점/이번 추가 학습 | 영상 관측 | 자세 후단 |
|---|---|---|---|
| N0_HISTORY | N3+SubPix 저장 출력 / 0 | 원 추론/기존 F 입력 | 기존 F의 실제 저장 자세 |
| RAW_OBSERVATION_MATCHED | 같은 N0 / 0 | 자동 V의 원q0, fullpoint만 | 공통 fresh점/부분normal adapter |
| LEGACY_RULE_MATCHED | 같은 N0 / 0 | 기존D1 규칙·recover stages·CAL/fixed FACE | 같은 공통 adapter |
| OBS_COMMON (3seed) | 같은 N0 / 합성768/235 | 두cue 합집합+MLP c=(.5,.5), fixed partial | 같은 공통 adapter |
| OBS_BI (3seed) | 같은 N0 / 같은 합성768/235 | 같은 후보/x/모델/cov; c만자동B/I | 같은 공통 adapter |
| E0_HISTORY | 과거 전체 시스템 / 이번0 | 상태별 보정·과거 전체 선택 경로 | 원E0 저장 자세 |
| H1_REP_HISTORY | 원H1 대표seed20261018 / 이번0 | 과거 최종 후보의 학습 선택 | 원H1 저장 자세 |
| PoseFix PF/YOLO anchor | 고정된 서로 다른 두 포트 경로 / 이번0 | 각고정PoseFix+SubPix 출력 | 각원후단의 저장 자세 |

주 대비는 같은 seed의 COMMON–BI뿐이다. 합성 자산 분할/C1·C2, CAL 실사 감독 이력, 양cue, 공통B/I noise처리를 공유했다. 다른 전체 시스템 간 차이를 c의 효과로 귀속하지 않는다. 모든 새 matched 경로가 N0로 시작하므로 이번 결과로 N3 단독/SubPix 단독 제거 효과를 새로 주장하지 않는다. 기존 비교군과 실패 기록을 모두 유지했다.
'''
    (public/'METHODS_AND_INFORMATION.md').write_text(comparison)
    links='\n[정보·후단 비교](METHODS_AND_INFORMATION.md) · [고정 출력 원인 분해](CAUSE_DECOMPOSITION.md) · [6fit 원 곡선](TRAINING_CURVES.csv) · [검산](VERIFICATION.json)\n'
    if links not in readme.read_text():readme.write_text(readme.read_text()+links)
    pointer=('# B·I 조건부 국소 관측 실험 — 완료\n\n'
        '[전체 보고서와192장 갤러리](_docs/experiments/pallet_bi_conditioned_observation_v1/README.md)\n\n'
        '같은 정보·후단에서 B/I 조건의 추가 가치는 미확보. 6fit·전체 평가·독립 검산을 공개했습니다.\n\n').encode()
    original=subprocess.run(['git','show','41ebc5a7f236abf6526eede5f6a8aa48176993b0:readme.md'],cwd=repo,check=True,stdout=subprocess.PIPE).stdout
    current=(repo/'readme.md').read_bytes()
    if current not in (original,pointer+original):raise ValueError('Preserve unexpected root README changes')
    (repo/'readme.md').write_bytes(pointer+original)


def main():
    p=argparse.ArgumentParser()
    for name in ('run-dir','public','repo'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.run_dir,a.public,a.repo)


if __name__=='__main__':main()
