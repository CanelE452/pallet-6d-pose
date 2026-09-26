"""Public tables/charts plus deterministic improved and worsened DEV examples."""
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import common as C
from .infer import ARMS
from scripts.research.pallet_selector_recovery_v1 import common as F


def render():
    res=C.read(C.DOC/'RESULTS.json')['groups'];decision=C.read(C.DOC/'DECISION.json');anchor=C.read(C.DOC/'VERIFIED_VISIBLE.json')['groups'];src=C.read(C.DOC/'SOURCE_PRESERVATION.json')['groups'];fit=C.read(C.DOC/'TRAIN_FIT.json')['groups']
    figs=C.DOC/'figures';figs.mkdir(exist_ok=True)
    fm=C.read(C.RAW/'FRAME_METRICS.json');pred=C.read(C.RAW/'RAW_PREDICTIONS.json')['real']
    from scripts.research.pallet_recording_disjoint_transfer_v1 import common as V
    records=V.records();truth=C.read(V.E.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json')
    f=lambda x:'—' if x is None else f'{x:.4f}'
    counts=C.read(C.DOC/'HARD_COVERAGE_PUBLIC.json')['corner_counts']
    fig,ax=plt.subplots(figsize=(8,3));ax.bar([f'P{i}' for i in range(8)],[counts[str(i)] for i in range(8)])
    ax.set_title('User-confirmed direct clicks only (PnP-assisted annotation)');ax.set_ylabel('Points')
    fig.tight_layout();fig.savefig(figs/'04_manual_corner_coverage.png',dpi=150);plt.close(fig)
    lines=['# Minimal hard supervision A/B — 완료 결과','',f'## 1. 결론\n\n**{decision["primary"]}**. 기존 `S1 + GEO_LINEAR`와 hard8장 추가 학습을 비교했다. 두 새 모델 각 5epoch·320update 완료. 평가를 보고 학습량·seed·selector를 바꾸지 않았다. 기존 최종 모델은 자동 교체하지 않았다.','',
           '직접 클릭 감독은 효과가 있었다: 학습8장의 클릭36점은 10px 이내24→36점, 별도 hard 검증36점은19→24점이다. 그러나 배포 성능 개선으로 결론내릴 수는 없다. 중간/심한 난도의 PCK10·oracle 후보는 개선됐지만 최종 선택 pose는 악화됐다. **기존 S1을 유지하고, 추가 라벨링 대신 후보 선택 실패를 분석하는 것이 다음 한 단계다.**','',
           '## 2. 무엇을 비교했나','',
           '| 모델 | 초기화 | 실사/epoch | 합성/epoch | hard 좌표 |','|---|---|---|---:|---|',
           '| BASE | 기존 S1 재사용 | 기존 Clean10 반복512 | 512 | 없음 |',
           '| H_PSEUDO | 원래 R0 | clean448 + hard64 | 512 | 동결 TYPE_REPLAY_PIPELINE |',
           '| H_MANUAL | 원래 R0 | clean448 + hard64 | 512 | 직접 클릭한 36점 |','',
           '수동군과 수도레이블군은 같은 RGB·PnP 박스·augmentation·감독 마스크·occurrence 순서다. hard 슬롯에서 좌표값만 다르다. 기존 clean/합성 슬롯은 원본 S1 캐시 및 occlusion을 그대로 사용했다. hard에는 인공 가림을 추가하지 않았다. 각 hard 이미지는 총40회 노출했다. 마지막 checkpoint만 평가했다.','',
           '## 3. 난도 태깅과 주석','',
           '8031 RGB → 중복·노출 제외6821 → 모델 없이 태깅123장(Clean10/Moderate13/Severe9/Invalid91) → initial8장(4M/4S, 3recordings). reserve2는 사용하지 않았다. 추가 태깅·20/50장 확대는 하지 않는다.','',
           '![선정 원본 모음](figures/03_selected_hard_frames_contact.jpg)','',
           '직접 클릭36개만 감독. 자동 보완 코너28개와 중심점8개 제외. 사용자는 직접 찍은 점에 대해 “꽤 확실해”라고 확인했다. 사용자 요청으로 PnP 보조를 켰고, 추가 수동 bbox 대신 저장된 PnP 코너 외접 박스를 공통 association으로 사용하는 변경을 승인받았다. **HUMAN_PNP_ASSISTED_NOT_BLIND**이며 PnP 없는 독립 수동 GT라고 주장하지 않는다.','',
           '![코너별 직접 입력 수](figures/04_manual_corner_coverage.png)','',
           '교사 좌표 유효 coverage36/36. hard 학습 recording과 평가 recording 교집합0. hard 손실은 직접 클릭 support의 xy 및 기존 RLE 좌표 항만 사용하고 box/class/DFL/keypoint-objectness 직접 손실은0. 숨은 점에 직접 gradient가 없는지, hard가 없는 입력은 기존 손실과 bit-exact인지 실제 기울기 검사했다. shared backbone 업데이트로 다른 출력이 간접적으로 변하는 것은 가능하다.','',
           '## 4. 2D 평가 (고정 HELDOUT128)','',
           '| 난도(n) | 모델 | PCK5% | PCK10% | PCK20% | median px | P90 px | 검출/매칭 |','|---|---|---:|---:|---:|---:|---:|---|']
    groups=('CLEAN','MODERATE','SEVERE','ALL');ns=(29,21,78,128)
    for group,n in zip(groups,ns):
        for arm in ARMS:
            s=res[group][arm]['twoD'];lines.append(f'|{group}({n})|{arm}|'+ '|'.join(f(100*s['PCK'][str(t)]) for t in (5,10,20))+f'|{f(s["matched_pooled_corner8_median_px"])}|{f(s["matched_pooled_corner8_P90_px"])}|{s["detected"]}/{s["matched"]}|')
    lines+=['','PCK는 검출/매칭 실패 penalty 포함, median/P90은 기존 matched pooled corner8 지표다. 전체 물체 허용 대칭 정렬을 유지한다. verified visible 표는 대칭 최소화 없는 fixed-ID다.','',
            '## 5. 6D 및 후보/선택 분해','',
            '|난도|모델|CURRENT ADDsym AUC|ORACLE AUC|선택 손실|R med°|yaw med°|t med cm|IoU3D med|pose 수|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        for a in ARMS:
            s=res[g][a];p=s['current'];lines.append(f'|{g}|{a}|{f(p["ADDsym_AUC"])}|{f(s["oracle"]["ADDsym_AUC"])}|{f(s["selection_loss"])}|{f(p["rotation_deg"]["median"])}|{f(p["yaw_deg"]["median"])}|{f(p["translation_cm"]["median"])}|{f(p["IoU3D"]["median"])}|{p["available"]}/{p["frames"]}|')
    lines+=['','CURRENT=동일 frozen GEO_LINEAR, ORACLE=GT 사후 W/D 후보 최소오차 진단(배포 불가). D9 보조와 각 지표P90은 [전체 결과](RESULTS.json)에 포함. baseline2D/6D는 기존 공개 결과와 재현 일치 검사를 통과했다.','']
    fig,axs=plt.subplots(1,2,figsize=(12,4));x=np.arange(4)
    for j,a in enumerate(ARMS):
        axs[0].bar(x+(j-1)*.25,[res[g][a]['current']['ADDsym_AUC'] for g in groups],width=.25,label=a)
        axs[1].bar(x+(j-1)*.25,[100*res[g][a]['twoD']['PCK']['10'] for g in groups],width=.25,label=a)
    for ax,title in zip(axs,['Frozen GEO_LINEAR ADDsym AUC','PCK10 (%)']):ax.set_xticks(x,groups);ax.set_title(title);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(figs/'06_hard_ab_results.png',dpi=150);plt.close(fig)
    lines+=['![전체 비교](figures/06_hard_ab_results.png)','','## 6. Verified visible / source / 학습 적합도','',
            '|그룹|모델|PCK10 correct/total|median px|P90 px|','|---|---|---:|---:|---:|']
    for g in ('ALL','HARD','CLEAN','MODERATE','SEVERE'):
        for a in ARMS:
            s=anchor[g][a];p=s['PCK']['10'];lines.append(f'|ANCHOR {g}|{a}|{p["correct"]}/{p["total"]}|{f(s["median_px"])}|{f(s["p90_px"])}|')
    for group,values in [('SOURCE256',{a:src[a]['twoD'] for a in ARMS}),('TRAIN8 (일반화 아님)',{a:fit[a]['total'] for a in ARMS})]:
        for a,s in values.items():
            p=s['PCK']['10'];lines.append(f'|{group}|{a}|{p["correct"]}/{p["total"]}|{f(s["median_px"])}|{f(s["p90_px"])}|')
    lines+=['','SOURCE256 exact ADDsym AUC: '+', '.join(f'{a}={f(src[a]["pose"]["ADDsym_AUC"])}' for a in ARMS),
            '', '## 7. 촬영별 변화 및 손상/복구','', '|recording|모델|ΔPCK10 pp|ΔCURRENT AUC|ΔORACLE AUC|','|---|---|---:|---:|---:|']
    for g in sorted(set(res)-set(groups)):
        for a in ARMS[1:]:
            b=res[g]['BASE'];s=res[g][a];lines.append(f'|{g}|{a}|{f(100*(s["twoD"]["PCK"]["10"]-b["twoD"]["PCK"]["10"]))}|{f(s["current"]["ADDsym_AUC"]-b["current"]["ADDsym_AUC"])}|{f(s["oracle"]["ADDsym_AUC"]-b["oracle"]["ADDsym_AUC"])}|')
    lines+=['','[lost/gained correct10·20→10 복구·5→10 손상](TRANSITIONS.json) · [학습 적합도](TRAIN_FIT.json) · [verified visible](VERIFIED_VISIBLE.json)','',
            '## 8. 개선 및 악화 이미지','',
            '난도별 H_MANUAL−BASE의 정답10px 이내 코너 수 차이 상위2/하위2를 고정 규칙으로 선택했다. 동일 이미지는 중복하지 않는다. 초록=평가 reference, 노랑=각 모델 raw keypoints. **PnP 투영선이 아니다.** 사후 사례 선택이며 대표적인 빈도나 독립 일반화 증거가 아니다.']
    examples=[];edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    def correct(a,fid):return sum(e<=10 for e in fm[a][fid]['errors'])
    for g,sev in [('CLEAN','CLEAN'),('MODERATE','MODERATE_OCCLUSION'),('SEVERE','SEVERE_OCCLUSION')]:
        rr=[r for r in records if r['severity']==sev]
        rank=sorted(rr,key=lambda r:(correct('H_MANUAL',r['id'])-correct('BASE',r['id']),r['id']))
        chosen=[]
        for r in rank[:2]+rank[-2:]:
            if r['id'] not in {x['id'] for x in chosen}:chosen.append(r)
        for r in chosen:
            fid=r['id'];im=cv2.imread(str(C.ROOT/r['image']['path']));h,w=im.shape[:2];scale=450/w;ph=round(h*scale);panels=[]
            for a in ARMS:
                panel=cv2.resize(im,(450,ph));gt=np.asarray(truth[fid]['gt']);valid=np.asarray(truth[fid]['valid'],bool)
                def draw(q,color):
                    for j,k in edges:
                        if np.isfinite(q[[j,k]]).all() and not (q[[j,k]]==-1).all(1).any():cv2.line(panel,tuple(np.rint(q[j]*scale).astype(int)),tuple(np.rint(q[k]*scale).astype(int)),color,1,cv2.LINE_AA)
                    for point in q[:8]:
                        if np.isfinite(point).all() and not (point==-1).all():cv2.circle(panel,tuple(np.rint(point*scale).astype(int)),2,color,-1,cv2.LINE_AA)
                qgt=gt.copy();qgt[~valid]=np.nan;draw(qgt,(50,230,50));p=F.selected(pred[a][fid])
                if p:draw(np.array(p['keypoints_xy']),(20,220,255))
                header=np.full((42,450,3),25,np.uint8);cv2.putText(header,f'{a}  correct10: {correct(a,fid)}',(8,27),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
                panels.append(np.vstack([header,panel]))
            filename=f'example_{len(examples)+1:02d}.jpg';cv2.imwrite(str(figs/filename),np.hstack(panels),[cv2.IMWRITE_JPEG_QUALITY,85])
            delta=correct('H_MANUAL',fid)-correct('BASE',fid);examples.append(dict(id=fid,severity=g,correct10_delta=delta,figure=filename))
            lines+=['',f'**{g} · {fid} · correct10 변화 {delta:+d}점**',f'![{fid}](figures/{filename})']
    lines+=['','## 9. 해석과 한계','',
            '수동군은 수도레이블군보다 hard PCK10과 최종 AUC가 모두 높지만, 기존 S1의 hard 최종 AUC를 넘지 못했다. 중간난도 선택 손실은0.0653→0.1252, 심한난도는0.0686→0.0845로 커졌다. 좋은 W/D 후보를 최종 출력으로 선택하는 단계에서 이득이 손실되는 신호다. 다만 **오류 꼬리도 악화**됐다: 전체 matched P90은34.46→46.05px, 중간난도 P90은22.48→57.34px다. 그러므로 모든 실패를 selector 하나의 문제로 단정하지 않는다.','',
            'BASE→H_PSEUDO는 hard 노출·teacher 감독 패키지의 효과이고, H_PSEUDO→H_MANUAL은 같은 이미지·박스·support에서 좌표 source 변화의 효과다. 실제 delta를 비교해야 하며 비슷함의 임의 허용오차나 유의성 주장을 추가하지 않는다.','',
            '이미 반복 확인한 HELDOUT128 개발평가이며 독립 TEST가 아니다. 사람 태깅 sample은 전체 hard 빈도 추정이 아니다. 8장·single seed·일반 플라스틱만의 파일럿이다. PnP 보조 partial 수동 좌표는 독립 full6D GT가 아니다. Camera-facing ID와 물리 C2 동치는 다르다. Frozen GEO_LINEAR가 새 모델에 최적인지는 별개다. 추가 대량 annotation 대신 이번 고정 판정에 맞춰 다음 병목을 정한다.','',
            '실행 기록: H_PSEUDO 완료 저장 후 동일 프로세스에서 다음 학습으로 넘어가는 과정이 종료됐다. 완료된 fit은 반복하지 않았고 H_MANUAL은 별도 프로세스에서 최초 실행했다. 반복 thread-pool 초기화를 피하도록 실행부를 수정했다. 재부팅·드라이버 변경·타 GPU 프로세스 종료 없음.','',
            '[입력 lock](HARD_LABEL_LOCK.json) · [사용자 승인 프로토콜 변경](ANNOTATION_PROTOCOL_AMENDMENT.json) · [사전 검사](PRETRAIN_TESTS.json) · [실제 학습 짝 감사](PAIR_INTEGRITY_MANUAL_VS_PSEUDO.json) · [최종 감사](COMPLETION_AUDIT.json) · [판정](DECISION.json)']
    C.save(C.DOC/'GALLERY_SELECTION.json',examples)
    from .figures import render as render_figures
    from .report_layout import assemble
    render_figures()
    C.save(C.DOC/'REPORT_KO.md',assemble('\n'.join(lines)+'\n'))


if __name__=='__main__':render()
