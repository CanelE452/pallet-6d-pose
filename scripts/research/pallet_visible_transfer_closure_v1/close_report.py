"""Publish all contrasts without replacing historical main results."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-visible-transfer-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from . import common as C
from scripts.research.pallet_selftraining_paper_closure_v1.report import overlay,texescape

ARMS=(*C.ARMS,'RAW_NEW','REF_NEW')

def main():
    r=C.read(C.DOC/'RESULTS.json');f=r['full128']['ALL'];v=r['visible66']['ALL'];pair=r['visible_pairs']['ALL'];train=r['train_native_following']
    old=C.read(C.DOC/'TRAIN_TARGET_TRANSFER.json')['all']['corners']['residuals'];ref=C.read(C.DOC/'REFERENCE_AND_THRESHOLD_SENSITIVITY.json')
    table66=C.table(['모델','추가 update','PCK5','PCK10','PCK20','평균px','중앙값px','P90px','>20px'],[[a,0 if a=='R0' else 640 if a.endswith('NEW') else 320,*[f"{v[a]['PCK'][str(k)]['correct']}/66" for k in (5,10,20)],*[f"{v[a][k]:.3f}" for k in ('mean_px','median_px','p90_px')],v[a]['gt20']] for a in ARMS])
    table128=C.table(['모델','PCK10 %','ADDsym AUC','matched P90px','R med°','t med cm','검출/pose'],[[a,f"{100*f[a]['twoD']['PCK']['10']:.2f}",f"{f[a]['sixD']['ADDsym_AUC']:.5f}",f"{f[a]['twoD']['matched_pooled_corner8_P90_px']:.3f}",f"{f[a]['sixD']['rotation_deg']['median']:.3f}",f"{f[a]['sixD']['translation_cm']['median']:.3f}",f"128/{f[a]['sixD']['available']}"] for a in ARMS])
    tablepairs=C.table(['대조 (B-minus-A)','66점 진입/이탈','66점 PCK10 순변화','66점 평균오차 변화','128장 PCK10 Δpp','128장 AUC Δ'],[[k,f"{pair[k]['gains10']}/{pair[k]['losses10']}",pair[k]['correct_delta10'],f"{pair[k]['mean_error_delta']:+.3f}",f"{r['full128_pairs'][k]['PCK10_delta_pp']:+.3f}",f"{r['full128_pairs'][k]['ADDsym_AUC_delta']:+.5f}"] for k in pair])
    C.save(C.DOC/'TABLE_VISIBLE66.md',table66);C.save(C.DOC/'TABLE_FULL128.md',table128);C.save(C.DOC/'TABLE_CONTRASTS.md',tablepairs)
    tabletrain=C.table(['학생','raw타깃 평균px','corrected타깃 평균px'],[[a,f"{(train[a]['raw_target']['mean_px'] if a in train else old[a]['raw']['mean_px']):.3f}",f"{(train[a]['ref_target']['mean_px'] if a in train else old[a]['ref']['mean_px']):.3f}"] for a in ARMS[1:]])
    figs=C.DOC/'figures';figs.mkdir(exist_ok=True)
    fig,ax=plt.subplots(1,2,figsize=(10,3.6),layout='constrained');x=np.arange(4);names=['RAW320','REF320','RAW640','REF640']
    ax[0].bar(x,[v[a]['PCK']['10']['correct'] for a in ARMS[1:]],color=['#cba52e','#a6489c']*2);ax[0].set_xticks(x,names);ax[0].set(ylim=(0,66),ylabel='Correct /66',title='Verified PCK10: all frozen students')
    for i,key in enumerate(['PCK10','AUC']):
        y=[100*f[a]['twoD']['PCK']['10'] if key=='PCK10' else 100*f[a]['sixD']['ADDsym_AUC'] for a in ARMS[1:]]
        ax[1].plot(x,y,'o-',label='PCK10 (%)' if i==0 else 'D9 AUC x100')
    ax[1].set_xticks(x,names);ax[1].set_title('Full128 unchanged legacy reference');ax[1].legend();fig.savefig(figs/'bounded_comparison.png',dpi=170);plt.close(fig)
    # Only previously published example RGB; selection is explanatory and not used for fitting/scoring.
    allow={x['frame_id'] for x in C.read(C.P.DOC/'FIGURE_MANIFEST.json')['examples']}
    frames=C.read(C.RAW/'FRAME_METRICS.json');records={r['id']:r for r in C.P.records()};truth=C.read(C.P.TRUTH)
    predictions=C.read(C.P.RAW/'PREDICTIONS.json')
    for a in ('RAW_NEW','REF_NEW'):predictions[a]=C.read(C.RAW/f'EVAL_{a}.json')
    delta=lambda fid:float(np.mean(frames['REF_NEW'][fid]['errors'])-np.mean(frames['REF_LR5'][fid]['errors']))
    candidates=sorted(allow&set(records));selection=[('most_improved',min(candidates,key=lambda i:(delta(i),i))),('most_worsened',max(candidates,key=lambda i:(delta(i),i))),('least_changed',min(candidates,key=lambda i:(abs(delta(i)),i)))]
    manifest=[];images=''
    for tag,fid in selection:
        im=Image.open(C.ROOT/records[fid]['image']['path']);g=truth[fid];target={j:q for j,q in enumerate(g['gt'][:8]) if g['valid'][j]}
        fig,axes=plt.subplots(1,4,figsize=(15,4));colors=['#18c4f6','#df39a9','#f5d531','#ff6c35']
        for ax,a,col in zip(axes,('R0','REF_LR5','RAW_NEW','REF_NEW'),colors):
            overlay(ax,im,predictions[a][fid],target,f"{a}\nmean scored error {np.mean(frames[a][fid]['errors']):.2f}px",col)
        name='example_'+tag+'.png';fig.suptitle(f'{fid} | REF640 - REF320 {delta(fid):+.2f}px | native2D, not PnP',fontsize=10);fig.tight_layout();fig.savefig(figs/name,dpi=135);plt.close(fig)
        manifest.append(dict(file=name,id=fid,selection=tag,delta_px=delta(fid),source=records[fid]['image'],restriction='Within already published prior13 example IDs'))
        images+=f'\n### {tag}: {delta(fid):+.3f}px\n\n![{tag}](figures/{name})\n'
    C.save(C.DOC/'FIGURE_MANIFEST.json',dict(examples=manifest,rule='min/max/min-absolute REF_NEW minus REF_LR5 mean scored error, within prior published image IDs; examples not aggregate selection',all_reference='Legacy mixed provenance; green crosses, prediction lines native2D not PnP'))
    verdict=dict(diagnostic_status='MIXED_PARTIAL_TRAIN_FOLLOWING_AND_REFERENCE_SAMPLE_SENSITIVITY',experiment_status='COMPLETED',
        visible_transfer_result='MIXED',evidence_status='REUSED_DEV_ONLY',method_development_stopped=True,
        selected='A_budget',new_fits=2,raw_updates=640,ref_updates=640,pair_integrity='PASS: both first320 loss/LR CSV and EMA model bit-exact; shared images/support/boxes/init; protected state exact',
        old_ref_to_new_ref=pair['REF_NEW-minus-REF_LR5'],new_raw_to_new_ref=pair['REF_NEW-minus-RAW_NEW'],
        full128_new_difference=r['full128_pairs']['REF_NEW-minus-RAW_NEW'],independent_confirmation=False,
        unresolved=['Unknown physical accuracy of unlabeled TRAIN pseudo targets.','Augmented-tensor full trace unavailable.','Cause cannot be uniquely assigned to update shortage, replay/augmentation, target contradiction or frozen representation.','No certified independent evaluation or physical6D reference.'],
        corrected_target_added_value='Full128 reused legacy DEV: REF_NEW vs RAW_NEW +3.5533pp PCK10/+0.02508594 D9 AUC; verified66 PCK10 tied44/66. Not uniformly stronger than the original320-step contrast.',
        primary_explanation='Partial target following with limited marginal response to320extra updates; verified threshold transitions cancel. Subset composition already removes the legacy full-panel advantage before reference replacement.',
        competing_explanation='Pseudo-target error/contradiction, augmentation/replay, frozen representation and evaluation sampling remain unseparated; more updates in general are not ruled out.',
        user_action_required=False,additional_manual_or_RGB=0,teacher_selector_new_module_fits=0)
    C.save(C.DOC/'FINAL_DECISION.json',verdict)
    report='# 제한된 가시점 전달 실험 — 최종 보고\n\n## 최종 판단: 부분 전달은 확인, 학습량 연장만으로 가시점 우위는 확보하지 못함\n\n'
    report+='RAW와 REF 모두 320→640 update에서 검수 PCK10이 43/66→44/66으로 늘었지만 **두 학생의 동률은 유지**됐다. 기존 REF 대비 신규 REF는 정답 진입 1점/이탈 0점이며, 신규 RAW 대비는 진입 2점/이탈 2점이다. 신규 REF의 가시점 평균 오차는 기존보다 0.042px 감소했지만 중앙값은 7.097→7.120px로 소폭 악화됐다. PCK5/20은 26/63개로 그대로다.\n\n'
    report+='REF의 TRAIN corrected-target 평균 잔차는 3.070→2.994px로 0.076px만 줄었다. 128장에서는 기존 REF 대비 PCK10이 507/985→504/985(−0.305pp), matched P90이 42.085→42.160px로 악화됐고, D9 AUC는 0.35902→0.36189로 개선됐다. 반면 같은 640 update의 RAW 대비 REF는 PCK10 +3.553pp, AUC +0.02509로 여전히 우세하다. 보정 타깃의 기존 DEV 추가 가치는 유지되지만, 학습량 연장이 그 효과를 일관되게 키우지는 않았다.\n\n'
    report+='**판정은 혼합이며 기존 320-update 주 비교를 교체하지 않는다.** 이 결과는 추가 320 update로 해당 병목이 해소되지 않았음을 보일 뿐, 모든 학습량 가설을 기각하거나 다른 원인 하나를 증명하지 않는다. 독립 검증은 여전히 없으며, 새 방법 탐색 없이 원고를 닫는다.\n\n## 실행 요약\n\n기존 R0 / RAW_LR5 / REF_LR5와 Replay 9장·38코너를 고정했다. 추가 RGB·수동좌표·교사/선택기 학습은 0이다. TRAIN 근거로 A(학습량)만 골라 RAW_NEW와 REF_NEW 각 640 update, 총 1,280 update 한 쌍을 완료했다. 첫 320의 loss/LR 및 저장 EMA weight는 각기 기존 학생과 bit-exact이며 동결 backbone·검출부·모든 buffer도 보존했다. B 신뢰도 처리·재시도·best epoch 선택은 하지 않았다.\n\n'
    report+='## 1. 왜 기존에는 동률이었나\n\n같은 66점에서 RAW→REF 진입 3점과 이탈 3점이 상쇄됐다. Clean +2·Moderate −2·Severe 0이다. 연속 오차는 34점 개선/32점 악화, 평균 −0.461px여서 완전히 같은 출력은 아니다. 같은 66점의 legacy 참조는 RAW 45/REF 44, verified는 43/43이었다. 따라서 전체 128장의 RAW 468/REF 507 이득이 작은 66점에서 사라진 현상을 참조 오류만으로 설명할 수 없다. 그 subset은 참조 변경 전부터 REF가 우세하지 않았다. legacy↔verified 동일점 차이 중앙값은 2.828px, P90은 5.374px이며 자동 GT 수정은 0회다.\n\n'
    report+='## 2. 학습 타깃 전달은 어느 정도였나\n\n'+tabletrain+'\n217장의 1,627 유효 코너에 대한 증강 없는 원영상 좌표 오차다. 이미지 균등·실제 occurrence 가중 및 center 별도 결과는 JSON을 참조한다. 기존 RAW의 corrected 타깃 잔차 4.078px→기존 REF 3.070px, 방향 정렬 82.64%(보정량 ≥1px인 1,492점)로 **전달 0이 아니라 부분 전달**이었다. pseudo는 GT가 아니므로 이 표의 낮은 값은 타깃 모방일 뿐 실사 물리 정확도 증명이 아니다. 남은 잔차와 마지막 loss 감소는 학습량 가설을 시험할 근거지만 원인의 확증은 아니다.\n\n'
    report+='## 3. 검수 66점: 기존 REF 대비와 신규 RAW 대비를 분리\n\n'+table66+'\n'+tablepairs+'\n![고정 기존/신규 비교](figures/bounded_comparison.png)\n\nPCK10을 주 지표로 유지했다. PCK5/20, 평균/중앙값/P90과 전체 진입·이탈도 함께 공개한다. 자세한 난도/recording/corner·교사/학생 4범주와 recording LORO는 RESULTS.json에 있다. 66점은 16장에 묶여 있고 독립 66표본으로 검정하지 않았다. ORDER43/44를 새 초기화 seed 실험이라 부르지 않는다.\n\n'
    report+='## 4. 전체 128장: 같은 legacy·D9 계약\n\n'+table128+'\n분모는 985 supported corners, matched 120장/931점, 검출 128장이다. D9는 corner0..7으로 풀고 기존 center 포함 selector residual을 그대로 쓴다. GT 박스 매칭·oracle 후보·GEO 교체·난도별 모델 선택은 추론에 없다. 6D는 geometry-derived reference이며 독립 측정 GT가 아니다. 전체 난도/recording·회전/yaw/이동/IoU3D/axis는 RESULTS.json에 모두 있다.\n\n'
    report+='## 5. 실제 이미지: 개선·악화·작은 변화\n\n이전 보고서에서 이미 공개한 13개 예시 ID 안에서 동일 규칙(min/max/min-absolute 평균 오차 변화)으로 선정했다. 새로운 RGB 공개 범위를 넓히지 않았다. 전체 성능은 모든 128장/66점으로 계산하며 이 그림 선정은 수치에 영향이 없다. Green=legacy reference, 선=native 2D 예측이며 PnP 재투영이 아니다. 점수 동률은 동일 좌표를 의미하지 않는다.\n'+images
    source=C.read(C.DOC/'SOURCE_HOLDOUT.json');report+='\n## 6. 합성보류셋과측정범위\n\n원래framework synthetic val32를같은모델들로평가했으며 reliability calibration/새test로바꾸지않았다.\n\n'+C.table(['모델','Box AP50–95','Pose AP50–95'],[[a,f"{source['results'][a]['metrics/mAP50-95(B)']:.5f}",f"{source['results'][a]['metrics/mAP50-95(P)']:.5f}"] for a in ARMS])
    report+='\n## 7. 해석·한계·종료\n\n새 update 규칙의 효과(기존 REF→신규 REF)와 보정 좌표의 추가 효과(신규 RAW→신규 REF)를 구별했다. 하나의 문턱에서 한 점 늘거나 보조 지표가 하락했다고 전체 성공/실패로 자동 결론 내리지 않는다. 기존 주 표·LR4/order 민감도·teacher 50/66·학생 43/66 표를 지우지 않고 사후 보완 실험으로 추가했다.\n\n독립 확인 감사 결과를 재사용하며 현재 66/128은 REUSED_DEV_ONLY다. 예측→D9→hash lock→scoring 순서를 지켰지만 이미 본 DEV가 독립 TEST로 바뀌지 않는다. 기존 수동 감독 9장/38점은 teacher에, 66점은 평가에만 있다. 학생 학습 217장의 진짜 pseudo 정확도·숨은 점·새 재료/세션 일반화·물리 6D 정확도는 확인되지 않았다. 학습량·증강·source 혼합·표현·모순 타깃 중 단일 원인 확정은 불가다.\n\n**방법 개발 STOP. 추가 모듈/레이블/조건 탐색 없음. 사용자 추가 작업 없음.** 영문 원고 사후 진단절, 재현 문서, 빌드/검수까지 마무리한다. 기존 8페이지 release는 9238735 Git 이력으로 보존된다.\n'
    anchors=C.read(C.RAW/'ANCHOR_POINTS_NEW_PRIVATE.json')
    public=[{k:x[k] for k in ('frame_id','corner_id','severity','recording','errors','missing')} for x in anchors]
    C.save(C.DOC/'PAIRED_66_NEW_ROWS.json',public)
    worst=sorted(public,key=lambda x:x['errors']['REF_NEW']-x['errors']['REF_LR5'],reverse=True)[:3]
    report+='\n## 전체 66점 공개 및 주요 손상\n\nPAIRED_66_NEW_ROWS.json에는 모든 66점의 오류와 결측 여부를 같은 규칙으로 공개했다(정답/예측 좌표는 비공개). 아래는 기존 REF→신규 REF 연속 오차 증가 상위 3점이며, 사후 설명용이다. 이 점들만 평가하거나 제거하지 않았다.\n\n'+C.table(['이미지','코너','기존 REF px','신규 REF px','증가 px'],[[x['frame_id'],x['corner_id'],f"{x['errors']['REF_LR5']:.3f}",f"{x['errors']['REF_NEW']:.3f}",f"{x['errors']['REF_NEW']-x['errors']['REF_LR5']:+.3f}"] for x in worst])
    C.save(C.DOC/'REPORT_KO.md',report)
    C.save(C.DOC/'REPRODUCE.md','''# 재현 / 재개

저장소 root, 기존 `pallet-yolo26` Python/CUDA 환경을 사용한다. private RGB/좌표/checkpoint는 자동 공개되지 않으며 INPUT_BINDINGS의 일치 자료가 필요하다.

```bash
python -m scripts.research.pallet_visible_transfer_closure_v1.prepare
python -m scripts.research.pallet_visible_transfer_closure_v1.infer_train
python -m scripts.research.pallet_visible_transfer_closure_v1.diagnose
python -m scripts.research.pallet_visible_transfer_closure_v1.pre_fit
python -m scripts.research.pallet_visible_transfer_closure_v1.report_diagnosis
python -m scripts.research.pallet_visible_transfer_closure_v1.fit RAW_NEW
python -m scripts.research.pallet_visible_transfer_closure_v1.fit REF_NEW
python -m scripts.research.pallet_visible_transfer_closure_v1.evaluate_new freeze
python -m scripts.research.pallet_visible_transfer_closure_v1.evaluate_new score
python -m scripts.research.pallet_visible_transfer_closure_v1.evaluate_new source
python -m scripts.research.pallet_visible_transfer_closure_v1.close_report
python -m unittest scripts.research.pallet_visible_transfer_closure_v1.test_contract
```

이미완료된TRAIN추론/사전lock은 재실행하지말고해시검증한다(시각필드가있어새로잠그면기존lock과다르다). fit/평가완료파일이있으면비싼계산을재실행하지않는다. 중간fit실패는기존run/optimizer를보존하며 blind재시작하지않는다. 허용된총fit2/update1280상한을넘기지않는다. 평가점으로후보/문턱/seed를다시고르지않는다.

학습구현의effective lr0=1e-5다. INTERVENTION_LOCK.args는원래protocol template(lr0=1e-4)을보존한필드이며,실제lr0 override와10개epoch경로는learning_rates_by_epoch/fit.py/새run args.yaml·results.csv가명시한다. args.epochs=5는기존E2ELoss스케줄보존용,trainer.epochs=10이실제예산이다. 원래5epochcosine의epoch5 실제학습률을이후5epoch고정했다. 첫320 loss/LR/EMA parity를통과하지않으면실험전체중단하도록했다.

신규그림은실측좌표와이전공개예시RGB에서생성했다. 원본RGB·좌표·checkpoint와추론캐시는data하위private로유지하고공개문서에는집계·오차·출처해시만포함한다. CUDA no-opfallback없음/재부팅없음/환경변경없음/다른프로세스종료없음.

기존주결과는pallet_selftraining_paper_closure_v1에그대로있고본namespace는사후진단이다. 원고빌드는기존Tectonic을사용하고본namespace의로그/캐시/BUILD_RESULT에기록한다. 원래숫자표는덮어쓰지않는다.
''')
    print('FINAL_REPORT_READY',flush=True)

if __name__=='__main__':main()
