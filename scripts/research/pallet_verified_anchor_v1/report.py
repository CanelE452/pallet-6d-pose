"""Local report and privacy-cropped scientific overlays, after labels are locked."""
import hashlib
from collections import Counter
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import common as C
from .review_saved_keypoints import REVIEW
from .evaluate import ARMS,PAIRS

FIG=C.DOC/'figures'

def figure(name):
    path=FIG/name;assert not path.exists()
    plt.tight_layout();plt.savefig(path,dpi=145);plt.close()

def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
        ['|'+'|'.join(str(v) for v in row)+'|' for row in rows])

def case(fid,tag,n,points,selected,truth):
    sel=selected[fid];im=cv2.imread(str(C.ROOT/sel['image']['path']));h,w=im.shape[:2]
    box=np.asarray(truth[fid]['box'],float);x0,y0=np.maximum(0,np.floor(box[:2])).astype(int);x1,y1=np.minimum([w,h],np.ceil(box[2:])).astype(int)
    assert x1>x0 and y1>y0
    roi=im[y0:y1,x0:x1].copy()
    detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    for x,y,fw,fh in detector.detectMultiScale(cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY),1.1,4,minSize=(20,20)):
        patch=roi[y:y+fh,x:x+fw];roi[y:y+fh,x:x+fw]=cv2.resize(cv2.resize(patch,(4,4)),(fw,fh),interpolation=cv2.INTER_NEAREST)
    rr=[r for r in points if r['frame_id']==fid];panels=[]
    for a in ('REFERENCE',*ARMS):
        view=roi.copy()
        def marker(xy,col,kind,size):
            if xy is None:return
            q=np.array(xy)-[x0,y0]
            if not (0<=q[0]<view.shape[1] and 0<=q[1]<view.shape[0]):return
            cv2.drawMarker(view,tuple(q.astype(int)),col,kind,size,1,cv2.LINE_AA)
        for r in rr:
            marker(r['verified_xy'],(0,255,0),cv2.MARKER_CROSS,12)
            marker(r['legacy_xy'],(255,0,255),cv2.MARKER_TILTED_CROSS,9)
            if a!='REFERENCE':marker(r['model_xy'][a],(0,230,255),cv2.MARKER_DIAMOND,9)
            q=np.array(r['verified_xy'])-[x0,y0]
            if 0<=q[0]<view.shape[1] and 0<=q[1]<view.shape[0]:
                cv2.putText(view,f'P{r["corner_id"]}',tuple((q+[4,-7]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,0),1,cv2.LINE_AA)
        canvas=np.full((360,450,3),24,np.uint8);scale=min(450/view.shape[1],275/view.shape[0])
        view=cv2.resize(view,(max(1,round(view.shape[1]*scale)),max(1,round(view.shape[0]*scale))))
        canvas[82:82+view.shape[0],:view.shape[1]]=view
        error=max(r['legacy_distance'] or 0 for r in rr) if a=='REFERENCE' else np.median([r['errors'][a] for r in rr])
        lines=[f'{tag} | {a}',f'{len(rr)} visible points | '+('legacy max' if a=='REFERENCE' else 'median')+f' {error:.2f}px',
               'green + human | magenta x legacy | yellow model']
        for i,line in enumerate(lines):cv2.putText(canvas,line,(6,20+i*24),cv2.FONT_HERSHEY_SIMPLEX,.42,(245,245,245),1,cv2.LINE_AA)
        panels.append(canvas)
    output=np.vstack([np.hstack(panels[:3]),np.hstack(panels[3:])]);name=f'case_{n:02d}_{tag}.jpg'
    assert not (FIG/name).exists();assert cv2.imwrite(str(FIG/name),output)
    return name

def main():
    res=C.read(C.DOC/'VERIFIED_RESULTS.json');comp=C.read(C.DOC/'MODEL_COMPARISON.json');agree=C.read(C.DOC/'REFERENCE_DISAGREEMENT.json')
    cov=C.read(C.DOC/'STATUS_COVERAGE.json');points=C.read(REVIEW/'SCORED_POINTS_PRIVATE.json');selection=C.read(REVIEW/'ANCHOR_SELECTION.json')
    selected={s['frame_id']:s for s in selection['frames']};truth=C.read(C.ROOT/'data/pallet/results/pallet_replay_clean19_v1/TRUTH_FOR_DISPLAY_ONLY.json')
    groups=['ALL',*C.SEVERITIES];names=['All66','Clean30','Moderate22','Severe14']
    recs=sorted({r['recording'] for r in selection['frames']});xx=np.arange(len(recs));bottom=np.zeros(len(recs))
    plt.figure(figsize=(9,4))
    for s in C.SEVERITIES:
        counts=[sum(r['recording']==g and r['severity']==s for r in selection['frames']) for g in recs]
        plt.bar(xx,counts,bottom=bottom,label=s.split('_')[0]);bottom+=counts
    plt.xticks(xx,recs);plt.ylabel('Selected frames');plt.title('Initial18, unchanged; 2 severe frames without saved keypoints');plt.legend(fontsize=8);figure('01_anchor_recording_severity.png')
    plt.figure(figsize=(8,4));plt.bar(np.arange(8),cov['corner_counts']);plt.xticks(range(8),[f'P{i}' for i in range(8)]);plt.ylabel('DIRECT_VISIBLE points');plt.title('66 manually confirmed points; P6 not covered');figure('03_visible_coverage.png')
    plt.figure(figsize=(8,4));plt.hist([r['legacy_distance'] for r in points],bins=np.arange(0,22,2));plt.xlabel('Verified vs legacy distance (px)');plt.ylabel('Corners');plt.title('All66 <=20px; 65/66 <=10px');figure('04_verified_vs_legacy_error.png')
    for file,kind in [('05_model_pck_verified.png','pck'),('06_model_median_verified.png','median')]:
        plt.figure(figsize=(10,4));xx=np.arange(4)
        for i,a in enumerate(ARMS):
            values=[res['groups'][g][a]['PCK']['10']['fraction']*100 if kind=='pck' else res['groups'][g][a]['median_px'] for g in groups]
            plt.bar(xx+(i-2)*.16,values,.16,label=a)
        plt.xticks(xx,names);plt.ylabel('PCK10 (%)' if kind=='pck' else 'Median error (px)');plt.legend();plt.title('Fixed native identity; same frozen predictions');figure(file)
    plt.figure(figsize=(10,4))
    for a in ARMS:
        yy=[res['per_corner'][str(i)][a]['median_px'] for i in range(8)];plt.plot(range(8),yy,'o-',label=a)
    plt.xticks(range(8),[f'P{i} n={cov["corner_counts"][i]}' for i in range(8)]);plt.ylabel('Median px');plt.legend();plt.title('No symmetry-min remapping; no P6 observations');figure('07_per_corner_error.png')
    plt.figure(figsize=(11,4));rr=comp['leave_one_recording_out'];xx=np.arange(len(rr))
    for i,(a,b) in enumerate(PAIRS):plt.plot(xx,[rr[g][f'{b}-minus-{a}']['PCK10_correct_delta'] for g in rr],'o-',label=f'{b}-{a}')
    plt.axhline(0,color='gray',lw=1);plt.xticks(xx,[f'exclude {g}' for g in rr],rotation=15);plt.ylabel('PCK10 correct count difference');plt.legend(fontsize=8);plt.title('Leave-one-recording-out sensitivity; not significance');figure('08_recording_sensitivity.png')
    # Actual disagreements >20 do not exist: label largest differences honestly.
    fids=sorted({r['frame_id'] for r in points});by={f:[r for r in points if r['frame_id']==f] for f in fids}
    largest=sorted(fids,key=lambda f:(-max(r['legacy_distance'] for r in by[f]),f))[:6]
    confirmed=[f for f in fids if any('MODEL_ERROR_CONFIRMED' in r['categories'].values() for r in by[f])]
    confirmed=sorted(confirmed,key=lambda f:(-max(max(r['errors'].values()) for r in by[f]),f))[:6]
    controls=sorted(fids,key=lambda f:hashlib.sha256(('20260923:'+f).encode()).hexdigest())[:6]
    cases=[]
    for tag,ff in [('largest_legacy_difference',largest),('confirmed_model_error',confirmed),('fixed_random_control',controls)]:
        for fid in ff:cases.append(dict(frame_id=fid,category=tag,file=case(fid,tag,len(cases)+1,points,selected,truth)))
    C.save_new(REVIEW/'CASE_MANIFEST_PRIVATE.json',cases)
    error_ids=Counter(r['corner_id'] for r in points if any(v=='MODEL_ERROR_CONFIRMED' for v in r['categories'].values()))
    decision=dict(status='COMPLETED_VISIBLE_ANCHOR_EVALUATION',primary='REFERENCE_STABLE_ENOUGH_FOR_NEXT_MODEL_WORK',
        secondary='MODEL_ERROR_CONFIRMED_WITH_VERIFIED_VISIBLE_POINTS',reference_disagreement_gt20=0,
        winner_claim='NO_GENERAL_WINNER; T1 PCK10 largest by only one point over R0/T0; T2 lowest median',
        confirmed_error_corner_counts=dict(error_ids),additional_anchor_images=0,
        more_training_labels_needed='NOT_ESTABLISHED_BY_THIS_PILOT',training_executed=False,
        next_one_experiment='DESIGN ONLY: audit the previously saved T2 TRAIN H36 residuals (remaining3 >20px), coordinate identity and native-to-model transforms before requesting any new training labels. Do not train or tune on this anchor.')
    C.save_new(C.DOC/'FINAL_DECISION.json',decision)
    allr=res['groups']['ALL'];a=agree['agreement']
    lines=['# 18장 코너 확인 후 frozen 모델 재평가','','## 결론','',
        '**이번 visible 코너에서 기존 기준이 크게 잘못됐다는 근거는 없습니다. 모델의 큰 오차는 일부 그대로 확인됐습니다.**',
        'PRIMARY: REFERENCE_STABLE_ENOUGH_FOR_NEXT_MODEL_WORK. SECONDARY: MODEL_ERROR_CONFIRMED_WITH_VERIFIED_VISIBLE_POINTS.',
        '추가 사진/재클릭 없이 이번 파일럿을 종료합니다. 신규 학습·추론·기존 GT 수정 모두 0회. 전체 성능 우승 모델은 정하지 않습니다.','',
        '## 진행 과정과 실제 작업량','',
        '모델 결과 없이 18장 선정 → 사용자 요청에 따라 기존 annotation.py + PnP로 키포인트 입력 → 직접 클릭의 상태만 별도 확인 → 입력 잠금 → legacy 기준의 QA 대상 확인 → 기존 R0/OLD_S1/T0/T1/T2 예측 hash 잠금 및 재사용 → fixed-identity visible-only 채점.',
        '**엄격한 blind annotation이 아닙니다.** 모델/이전 정답 좌표는 표시하지 않았지만 PnP 보조를 사용했습니다. 평가 기준은 사람의 직접 클릭+상태 판단이며 독립 6D GT가 아닙니다.','',
        table(['항목','수량'],[['선정 / 저장','18 / 16장'],['추가 선정','0장'],['미저장 심함','2장, 그대로 보존·미대체'],['저장된 직접 클릭 코너','75개 (클릭 이벤트 횟수와 다름)'],['화면 밖 직접 클릭','3개 제외'],['상태 검수','72개 완료'],['DIRECT_VISIBLE / EXTERNAL_OCCLUDED','66 / 6개'],['PnP 보완 / 직선 연장','52 / 1개, 주평가 제외'],['QA 재검토 / 변경','0 / 0'],['분/장','별도 측정하지 않음'],['전체144 상태 분류','미완료: 미입력·비수동 점 미분류로 제외']]),'',
        '![선정](figures/01_anchor_recording_severity.png)','','## 코너 정의와 coverage','',
        '![번호](figures/corner_index_reference.png)','','![가시점](figures/03_visible_coverage.png)','',
        '직접 보이는 점: CLEAN30 / MODERATE22 / SEVERE14, 16장·6 recording. 선택18장은7 recording이었지만 검증점은6 recording에만 있습니다. P6=0, 다른7개 ID는 각각3회 이상. 총60·난도별12·ID5종·recording3 기준을 충족해 추가6장을 요구하지 않습니다.',
        '사람이 입력할 수 있었던 가시점만 비교하므로 심한 가림 전체·숨은 점·저장 못한2장으로 일반화할 수 없습니다.','',
        '## 신규 클릭과 기존 기준의 차이','',
        f'중앙값 **{a["median_px"]:.2f}px**, P90 **{a["p90_px"]:.2f}px**. 5px 이내57/66, 10px 이내65/66, 20px 초과0/66. QA 조건(>20px/불확실 메모)에 해당하는 점0개. 기존 정답을 자동 수정하지 않았습니다.',
        '![기준차이](figures/04_verified_vs_legacy_error.png)','','## 모델 재평가: 직접 보이는66점, 번호 고정','',
        table(['모델','PCK5','PCK10','PCK20','중앙값 px','P90 px','>20px'],[
            [arm,f'{m["PCK"]["5"]["correct"]}/66',f'{m["PCK"]["10"]["correct"]}/66 ({100*m["PCK"]["10"]["fraction"]:.2f}%)',f'{m["PCK"]["20"]["correct"]}/66',f'{m["median_px"]:.2f}',f'{m["p90_px"]:.2f}',m['gt20']] for arm,m in allr.items()]),'',
        'T0: 추가 clean pseudo 학습, T1: 추가 hard pseudo 학습, T2: 같은 hard의 기존 수동 좌표 학습. OLD_S1은 이전 EASY→HARD 학생. 모두 저장된 예측 그대로이며 보정기·selector·confidence threshold를 바꾸지 않았습니다.',
        'T1의 PCK10이 가장 높지만 R0/T0보다 **1점** 더 맞습니다. T2는 중앙값이 가장 낮지만 PCK10은 T1보다2점 적습니다. 모델 선택이나 일반적인 우월성 결론에 쓰지 않습니다.',
        '![PCK](figures/05_model_pck_verified.png)','','![중앙값](figures/06_model_median_verified.png)','','## 난도별 PCK10','',
        table(['모델','CLEAN30','중간22','심함14'],[[arm]+[f'{res["groups"][s][arm]["PCK"]["10"]["correct"]}/{res["groups"][s][arm]["n"]}' for s in C.SEVERITIES] for arm in ARMS]),'',
        '심함에서 R0 7/14 → T1 10/14지만 작은 가시점 부분집합입니다. 중간에서는 R0 15/22가 가장 높아 일관된 hard 개선으로 부를 수 없습니다.','',
        '## 같은66점에서 legacy로 채점했을 때와 비교','',
        table(['모델','legacy PCK10','새 reference PCK10'],[[arm,res['legacy_reference_same_66_points']['ALL'][arm]['PCK']['10']['correct'],allr[arm]['PCK']['10']['correct']] for arm in ARMS]),'',
        '몇 픽셀 차이로 10px 경계의 정답 수와 순서가 달라집니다. 큰 GT 오류의 증거는 아니며, 소표본에서 한두 점 차이로 모델을 고르면 안 된다는 한계입니다. 과거 전체128장 결과와 이번66점 결과의 분모를 혼합하지 않습니다.','',
        '## 쌍별 변화와 recording 민감도','',
        table(['비교(뒤−앞)','PCK10 정답수 변화','중앙값 변화 px','프레임 win/loss/tie'],[[k,v['PCK10_correct_delta'],f'{v["median_error_delta_px"]:.3f}',str(v['frame_median_win_loss_tie'])] for k,v in comp['pairwise']['ALL'].items()]),'',
        '![ID별](figures/07_per_corner_error.png)','','![기록제외](figures/08_recording_sensitivity.png)','',
        '## 사례 이미지','',
        '초록 +: 신규 직접 클릭, 자홍 ×: legacy, 노랑 마름모: 모델. PnP wireframe이 아니라 실제 모델 키포인트입니다. 원본 대신 팔레트 ROI만 표시하고 검출된 얼굴은 모자이크합니다. 표시용 crop/resize는 평가 좌표에 영향을 주지 않습니다.',
        'legacy >20px 불일치 사례가 없으므로 아래 첫6장은 단순히 차이가 큰 순서입니다. 오류라고 단정하지 않습니다. 모델 오류 사례는 실제 해당 프레임만, 임의 대조6장은 고정 hash 순서로 표시합니다.','']
    for i,entry in enumerate(cases,1):lines += [f'### 사례 {i}: {entry["category"]}','',f'![사례{i}](figures/{entry["file"]})','']
    lines += ['## 다음 최소 행동과 한계','',
        '추가 어노테이션 필요성이 입증된 것은 아닙니다. 기존 기준과66점이 모두20px 이내여서 지금은 단순 RGB 수 증가나 대규모 GT 재작성보다, 이미 가진 학습 자료에서 큰 잔차가 남는 원인을 먼저 확인하는 것이 타당합니다.',
        '다음 한 실험은 **기존 T2 TRAIN H36에서 >20px로 남았던3점의 ID/좌표 변환/학습 입력 일치 감사**로 설계만 제안합니다. 새 학습·레이블링·튜닝은 실행하지 않았습니다. 이 anchor는 절대 TRAIN에 사용하지 않습니다.',
        '소규모·균형 진단이며 독립 TEST/운영분포 평가가 아닙니다. PnP 보조, 입력 가능한 점의 선택 편향, P6 부재, 미저장2장, 고유 recording6개, 재사용 DEV의 한계가 있습니다. 6D 정답을 새로 만들거나 기존6D 표에 섞지 않습니다.',
        '', '## 재현과 저장','',
        'FIRST_PASS_LOCK → VERIFIED_LABELS(private) → PREDICTIONS_LOCK → VERIFIED_RESULTS 순서로 고정. 좌표와 프레임 매핑은 private 로컬 파일에 보존. 공개용 JSON에는 집계만 남깁니다. 코드 `audit_completed_status.py`, `evaluate.py`, `report.py`를 참조하세요.',
        '이 보고서는 비교 표·차트·사례 이미지 26개와 함께 저장소에 포함합니다. 원본 RGB, 개인별 좌표 JSON, 정확한 프레임 매핑은 공개 대상에서 제외합니다.','']
    C.save_new(C.DOC/'EVALUATION_REPORT_KO.md','\n'.join(lines))
    print('REPORT_READY',len(cases),'case images; fixed points',len(points),decision)

if __name__=='__main__':main()
