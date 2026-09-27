"""Decision-relevant measured figures and Korean report, no invented imagery."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-visible-transfer-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from . import common as C

def main():
    e=C.read(C.DOC/'REFERENCE_AND_THRESHOLD_SENSITIVITY.json');t=C.read(C.DOC/'TRAIN_TARGET_TRANSFER.json');v=t['all']['corners']
    figs=C.DOC/'figures';figs.mkdir(exist_ok=True)
    fig,ax=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    x=np.arange(3);names=['Clean','Moderate','Severe'];groups=['CLEAN','MODERATE_OCCLUSION','SEVERE_OCCLUSION']
    gains=[e['paired']['severity:'+g]['gains10'] for g in groups];losses=[e['paired']['severity:'+g]['losses10'] for g in groups]
    ax[0].bar(x-.17,gains,.34,label='wrong -> correct');ax[0].bar(x+.17,[-n for n in losses],.34,label='correct -> wrong');ax[0].set_xticks(x,names);ax[0].axhline(0,color='black',lw=.7);ax[0].set_title('Verified66: RAW -> REF at10px');ax[0].legend(fontsize=8)
    rows=C.read(C.DOC/'PAIRED_66_ROWS.json');a=[r['errors']['RAW_LR5'] for r in rows];b=[r['errors']['REF_LR5'] for r in rows]
    ax[1].scatter(a,b,s=20);ax[1].plot([0,45],[0,45],'k--',lw=.8);ax[1].axhline(10,color='gray',lw=.6);ax[1].axvline(10,color='gray',lw=.6);ax[1].set(xlabel='RAW error (px)',ylabel='REF error (px)',title='All66 points; no tail omitted')
    fig.savefig(figs/'paired66.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for i,target in enumerate(('raw','ref')):
        vals=[v['residuals'][a][target]['mean_px'] for a in C.ARMS]
        ax[0].bar(x+(i-.5)*.33,vals,.33,label=f'To {target} target')
    ax[0].set_xticks(x,['R0','RAW','REF']);ax[0].set(ylabel='Mean native residual (px)',title='TRAIN217 /1627 supported corners');ax[0].legend()
    pts=[p for p in C.read(C.DOC/'TRAIN_POINT_RESIDUALS.json') if p['corner']<8]
    ax[1].scatter([p['correction'] for p in pts],[p['student_change'] for p in pts],s=5,alpha=.3);ax[1].set(xlabel='Target correction norm (px)',ylabel='Student change norm (px)',title='Partial following, NOT GT accuracy')
    fig.savefig(figs/'train_transfer.png',dpi=170);plt.close(fig)
    scores=e['groups']['ALL'];old=e['legacy_same66'];res=v['residuals']
    report='# 검수 가시점 전달 진단 — 학습 전 결과\n\n'
    report+='## 결론\n\n보정 감독은 **부분적으로 전달**됐다. 학습217장의 공통 감독1627코너에서 corrected 타깃 잔차는 RAW4.078px→REF3.070px이다. 보정량이1px 이상인1492점 중 학생 변화의82.64%가 보정 방향과 양의 내적을 가지며, 투영 비율 중앙값은0.319다. 이는 완전 전달도, GT 정확도 증명도 아니다. 마지막 구간 loss 감소와 남은 잔차를 근거로 학습량 한 변수의640update 짝 실험만 잠근다. 원인 확정/필터 개발은 하지 않는다.\n\n'
    report+='## 66점 동률: 진입·이탈과 난도 상쇄\n\n'+C.table(['모델','PCK5','PCK10','PCK20','평균px','중앙값px','P90px','>20px'],[[a,*[f"{scores[a]['PCK'][str(k)]['correct']}/66" for k in (5,10,20)],*[f"{scores[a][k]:.3f}" for k in ('mean_px','median_px','p90_px')],scores[a]['gt20']] for a in ('R0','TEACHER','RAW_LR5','REF_LR5')])
    report+='\n'+C.table(['난도','점','진입','이탈','순변화'],[[g,e['paired']['severity:'+g]['n'],e['paired']['severity:'+g]['gains10'],e['paired']['severity:'+g]['losses10'],e['paired']['severity:'+g]['correct_delta10']] for g in groups])
    report+='\nRAW→REF 정답 진입3/이탈3, Clean+2·Moderate−2·Severe0을 원자료로 재확인했다. 연속 오차는34점 개선/32점 악화, 평균 변화−0.461px다. 동일PCK10은 동등성 증명이 아니다. PCK5/20은 설명용이며 주 문턱10px를 바꾸지 않았다. 양 학생 각각 교사만 정답10, 학생만 정답3, 둘 다 정답40, 둘 다 오답13이며 집합이 반드시 같은 것은 아니다. 교사가 평가점에서 맞았다는 것은 그 점으로 학생을 학습했다는 뜻이 아니다.\n\n![전체66 paired 변화](figures/paired66.png)\n\n'
    report+='## 참조와 표본을 분리\n\n'+C.table(['동일66점','legacy PCK10','verified PCK10','legacy 평균','verified 평균'],[[a,f"{old[a]['PCK']['10']['correct']}/66",f"{scores[a]['PCK']['10']['correct']}/66",f"{old[a]['mean_px']:.3f}",f"{scores[a]['mean_px']:.3f}"] for a in C.ARMS])
    report+=f"\n두 참조 좌표 차이 중앙값{e['reference_displacement']['median_px']:.3f}px, P90{e['reference_displacement']['p90_px']:.3f}px. 같은prediction·fixed ID·native 좌표·결측 처리·no-IoU-gate로 계산했다. 참조를 자동 수정하지 않았다. 전체128장과66점은 표본도 다르며 원래 전체 지표에는 매칭/대칭 계약도 있다. JSON의 full128_harmonized_fixedID_no_matching→legacy_same66→groups.ALL을 차례로 비교해야 한다. 전체 격차를 GT 오류 하나로 설명할 수 없다.\n\n"
    report+='## TRAIN 타깃 추종\n\n'+C.table(['모델','raw타깃 mean/med','ref타깃 mean/med','ref unique-image mean','ref occurrence mean'],[[a,f"{res[a]['raw']['mean_px']:.3f}/{res[a]['raw']['median_px']:.3f}",f"{res[a]['ref']['mean_px']:.3f}/{res[a]['ref']['median_px']:.3f}",f"{res[a]['ref']['weighted_mean']['unique_image_uniform']:.3f}",f"{res[a]['ref']['weighted_mean']['actual_occurrence_image_weighted']:.3f}"] for a in C.ARMS])
    report+='\n![실제TRAIN 타깃과 학생의 대응](figures/train_transfer.png)\n\n실사217 unique/epoch512 occurrence + 합성512 슬롯을 복원했다. 원래100px reflection padding과 저장학습RGB가 bit-exact, label→native roundtrip 오차<1e−5px, raw/ref 박스·support·원래confidence·center 보존을 확인했다. R0 재추론↔raw export 차이 평균<1e−6px이므로 현 추론 계약과 저장raw 좌표가 일치한다. 후보는 최고confidence만 사용했고 타깃에 가까운 후보로 교체하지 않았다. 코너·recording·bbox크기·유효support·반복노출별 집계는 TRAIN_TARGET_TRANSFER.json에 있다. Center는 별도 집계했다.\n\n실사 GT 없는 타깃 추종 진단이다. native 입력만 추론했으며 증강 tensor cache는 없으므로 증강된 모든 입력/학습 tensor의 전수 일치는 주장하지 않는다. CSV pose loss는 혼합real/source·면적/support정규화 값이고 pixel 오차가 아니다. 동일 구현이라도1:1 이미지 노출은 loss 기여량1:1을 뜻하지 않는다. true-ignore는 증강 후 v1에 적용되며 out-of-frame 변환은v0를 만들 수 있다.\n\n'
    report+='## 곡선과 한 개입의 근거\n\n'+C.table(['학생','epoch1..5 pose loss','최종 학습률'],[[a,', '.join(f"{r['train/pose_loss']:.5f}" for r in t['curves'][a]),t['curves'][a][-1]['lr/pg0']] for a in C.ARMS[1:]])
    report+='\n전체와 마지막 구간에서는 감소하지만 epoch3에서 반등하여 단조 감소가 아니다. 추가 최적화 가설을 시험할 관찰적 근거이지 학습량 부족의 확증이 아니다. 모순된pseudo·증강·혼합replay·동결표현·학습밖 전이가 경쟁 설명이다.\n\n640update를 각 학생R0부터 재실행하며 첫320의 기존cosine5 경로와 E2ELoss의one2many/one2one 경로를 유지한다. 321..640은 실제epoch5 학습률1.859423525e−6을 유지한다. 단순epochs10은 LR뿐 아니라 E2ELoss 스케줄도 바꾸므로 쓰지 않는다. 기존last는optimizer가 제거돼 이어학습으로 가장하지 않는다. 첫320의loss/LR CSV와 저장EMA weight를 기존과 비교하고 불일치 시 중단한다. last-only, 다른loss/필터/선택기/교사 변경0.\n\n'
    report+='## 선행 근거와 적용 한계\n\n[Soft Teacher §3.3](https://arxiv.org/pdf/2106.09018)는 jitter된 상자의 회귀 분산을 위치 pseudo 선별에 쓴다. [Unbiased Teacher v2 §3.3.2](https://arxiv.org/html/2206.09500)는 교사와 학생의 경계별 상대 불확실성으로 회귀 감독을 선택한다. box에서의 결과를 팔레트keypoint 효과로 간주하지 않는다. 이번에는 검증된 위치 calibration이 없고 TRAIN 기반 예산 가설을 선택했으므로 B는 실행하지 않는다. 원래confidence 보존은 corrected 위치 정확도 재측정이 아니며 안정적으로 틀릴 수 있다. 어려운 점을 제거하면 가림 감독도 줄어든다. [scikit-learn 데이터 누출 지침](https://scikit-learn.org/1.5/common_pitfalls.html)에 따라 평가 좌표는 fitting/calibration/threshold 선택에 쓰지 않는다. CVF 직접열기는403/오류여서 동일 저자의arXiv 원문을 확인했다.\n\n## 증거 범위와 종료\n\n66점은16장에 묶인 반복DEV이며독립66표본 검정하지 않는다. recording별 paired/LORO를 JSON에 남겼고 ORDER43/44는초기화seed 반복이 아닌 순서민감도다. LR4와 모든ORDER 결과도 보존하며 유리한 설정으로 주비교를 교체하지 않는다. 기존전체128장/teacher50점/학생43점 표는 각각다른질문이다. 별도독립확인 없음. 추가학습결과와 무관하게보고서·영문원고·재현문서까지닫고방법개발을종료한다.\n'
    C.save(C.DOC/'DIAGNOSTIC_REPORT_KO.md',report)
    C.save(C.DOC/'DECISION_BEFORE_FIT.md','# 학습 전 단일 개입 결정\n\nA_budget만 선택. 근거와 반론: DIAGNOSTIC_REPORT_KO.md / DECISION_BEFORE_FIT.json. 66점의 개별 손상점을 목표로 하이퍼파라미터를 정하지 않았다. 최대두fit·각640update; prefix불일치면중단하며 B로전환하지 않는다.\n')
    print('DIAGNOSTIC_REPORT_READY')

if __name__=='__main__':main()
