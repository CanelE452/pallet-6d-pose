from collections import Counter
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.research.pallet_clean19_pose_mismatch_v1.render import table,f,draw_edges,projection
from . import common as E

SHORT=dict(zip(E.ARMS,['T0','T1','T2']))
def savefig(name):
    plt.tight_layout();plt.savefig(E.DOC/'figures'/name,dpi=160);plt.close()

def crop(r,box,width=440):
    """Display-only tight pallet ROI; do not publish full original backgrounds."""
    im=cv2.imread(str(E.ROOT/r['image']['path']));h,w=im.shape[:2];b=np.array(box,float)
    x0,y0=np.maximum(0,np.floor(b[:2])).astype(int);x1,y1=np.minimum([w,h],np.ceil(b[2:])).astype(int)
    if x1<=x0 or y1<=y0:raise ValueError('Invalid display ROI; do not expose full image fallback')
    roi=im[y0:y1,x0:x1];scale=width/roi.shape[1];view=cv2.resize(roi,(width,max(1,round(roi.shape[0]*scale))))
    # Faces within the pallet ROI are pixelated for display only, never evaluation.
    detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    for x,y,fw,fh in detector.detectMultiScale(cv2.cvtColor(view,cv2.COLOR_BGR2GRAY),1.1,4,minSize=(20,20)):
        patch=view[y:y+fh,x:x+fw];view[y:y+fh,x:x+fw]=cv2.resize(cv2.resize(patch,(4,4)),(fw,fh),interpolation=cv2.INTER_NEAREST)
    return view,np.array([x0,y0]),scale

def tile(image,lines,height=380,width=440):
    out=np.full((height,width,3),25,np.uint8);top=25*len(lines)+10;scale=min(width/image.shape[1],max(1,height-top)/image.shape[0]);im=cv2.resize(image,(max(1,round(image.shape[1]*scale)),max(1,round(image.shape[0]*scale))));out[top:top+im.shape[0],:im.shape[1]]=im
    for i,line in enumerate(lines):cv2.putText(out,line,(7,21+i*25),cv2.FONT_HERSHEY_SIMPLEX,.43,(240,240,240),1,cv2.LINE_AA)
    return out

def sheets(targets):
    c0=[r for r in targets if r['role']=='C0'];ee=sorted([r for r in targets if r['role']=='E'],key=lambda r:r['pair']);hh=sorted([r for r in targets if r['role']=='H'],key=lambda r:r['pair'])
    tiles=[]
    for r in c0:
        im,_,_=crop(r,r['bbox']);tiles.append(tile(im,['C0 | '+r['id'],'Shared clean teacher fitting image']))
    cv2.imwrite(str(E.DOC/'figures/02_C0_contact.jpg'),np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]))
    paired=[];teacher=[]
    for e,h in zip(ee,hh):
        row=[]
        for r in (e,h):
            im,_,_=crop(r,r['bbox']);row.append(tile(im,[r['role']+' | '+r['id'],r['severity'],'NOT same-pose before/after pair']))
        paired.append(np.hstack(row));im,offset,s=crop(h,h['bbox']);q=np.array(h['target']);g=np.array(h['manual']);mask=np.array(h['mask'])
        for j in np.flatnonzero(mask):
            u=tuple(((q[j]-offset)*s).astype(int));v=tuple(((g[j]-offset)*s).astype(int));cv2.circle(im,u,4,(255,220,0),-1);cv2.drawMarker(im,v,(0,255,0),cv2.MARKER_CROSS,11,2);cv2.putText(im,str(j),v,cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,0),1)
        teacher.append(tile(im,['H | '+h['id'],'Cyan=teacher; green=existing manual','Display reference; common channels only']))
    cv2.imwrite(str(E.DOC/'figures/03_EH_contact.jpg'),np.vstack(paired));cv2.imwrite(str(E.DOC/'figures/04_H_teacher_manual.jpg'),np.vstack([np.hstack(teacher[i:i+2]) for i in range(0,len(teacher),2)]))

def case(r,label,index,truth,preds,poses,fm,pm):
    fid=r['id'];base,offset,s=crop(r,truth[fid]['box']);meta=next(x for x in E.read(E.V.RAW/'INFERENCE_METADATA.json') if x['id']==fid);K=np.array(meta['K']);panels=[]
    for a in ('RGB',*E.ARMS):
        im=base.copy();lines=[a if a=='RGB' else SHORT[a],fid,label]
        if a=='RGB':lines+=['Tight ROI: display only','Yellow=2D; red=CURRENT pose','Cyan dashed=alternate W/D','Green=legacy reference','ORACLE POSTHOC NONDEPLOYABLE']
        else:
            q=E.D.points(preds[a][fid]);p=poses[a][fid];m=pm[a][fid]['current'];row=fm[a][fid]
            if q is not None:draw_edges(im,(q-offset)*s,(0,220,255))
            if p['current']['available']:draw_edges(im,(projection(p['current'],K)-offset)*s,(20,30,255))
            for h in p['hypotheses']:
                if h['name']!=p['current'].get('selected_hypothesis') and h['pose']['available']:draw_edges(im,(projection(h['pose'],K)-offset)*s,(255,220,0),True)
            for j in range(8):
                if truth[fid]['valid'][j]:cv2.drawMarker(im,tuple(((np.array(truth[fid]['gt'][j])-offset)*s).astype(int)),(0,255,0),cv2.MARKER_CROSS,10,1)
            lines += [f"PCK10 {sum(e<=10 for e in row['errors'])}/{row['corners']} | matched {row['matched']}", 'CURRENT ADDnorm '+f(m.get('ADDsym_normalized')), 'ORACLE '+f(pm[a][fid]['oracle'].get('ADDsym_normalized'))+' POSTHOC', 'Axis '+str(m.get('axis_correct')), 'R/yaw/t '+ '/'.join(f(m.get(k),1) for k in ('rotation_deg','yaw_deg','translation_cm'))]
        panels.append(tile(im,lines,height=580))
    name=f'case_{index:02d}_{label}.jpg';cv2.imwrite(str(E.DOC/'figures'/name),np.hstack(panels));return name

def main():
    E.immutable();split=E.read(E.DOC/'SPLIT_LOCK.json');pairs=E.read(E.DOC/'TARGET_AND_PAIR_AUDIT.json');res=E.read(E.DOC/'RESULTS.json')['groups'];train=E.read(E.DOC/'TRAIN_DIAGNOSTICS.json');source=E.read(E.DOC/'SOURCE_PRESERVATION.json');records=split['heldout'];targets=E.read(E.RAW/'TARGETS.json');trans=E.read(E.DOC/'TRANSITIONS.json')
    (E.DOC/'figures').mkdir(exist_ok=False)
    recs=sorted({r['recording_group'] for r in split['train']+records});fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(recs,[sum(r['recording_group']==k and r['old_role']=='TRAIN' for r in split['train']) for k in recs],label='old TRAIN retained')
    ax.bar(recs,[sum(r['recording_group']==k and r['old_role']=='EVAL' for r in split['train']) for k in recs],bottom=[sum(r['recording_group']==k and r['old_role']=='TRAIN' for r in split['train']) for k in recs],label='old EVAL reserved TRAIN pool')
    ax.bar(recs,[sum(r['recording_group']==k for r in records) for k in recs],label='new HELDOUT');ax.set(ylabel='Frames',title='Recording-disjoint fitting; reused DEV, not new TEST');ax.legend(fontsize=8);savefig('01_recording_split.png');sheets(targets)
    groups=['HELDOUT_CLEAN','HELDOUT_MODERATE','HELDOUT_SEVERE']
    for number,field,title in [(5,'PCK','PCK10'),(6,'current','CURRENT ADDsym AUC'),(7,'oracle','ORACLE ADDsym AUC — POSTHOC NONDEPLOYABLE')]:
        fig,ax=plt.subplots(figsize=(9,4));xx=np.arange(3)
        for i,a in enumerate(E.ARMS):ax.bar(xx+(i-1)*.25,[res[g][a]['twoD']['PCK']['10'] if field=='PCK' else res[g][a][field]['ADDsym_AUC'] for g in groups],.25,label=SHORT[a])
        ax.set(xticks=xx,xticklabels=['Clean29','Moderate21','Severe78'],ylabel=title,title='Frozen recording-heldout128');ax.legend();savefig(f'{number:02d}_{field}.png')
    fig,ax=plt.subplots(figsize=(9,4));names=['R0','OLD_S1',*E.ARMS];ax.bar(range(5),[source['twoD'][a]['PCK']['10']['fraction'] for a in names]);ax.set(xticks=range(5),xticklabels=['R0','old S1','T0','T1','T2'],ylabel='PCK10',title='Unchanged synthetic heldout256');savefig('08_source.png')
    fm=E.read(E.RAW/'FRAME_METRICS.json');pm=E.read(E.RAW/'POSE_METRICS.json');poses=E.read(E.RAW/'POSE_PREDICTIONS.json');truth=E.read(E.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json');preds={a:E.read(E.RAW/f'HELDOUT_{a}.json')['predictions'] for a in E.ARMS};selected=[]
    contrasts={}
    for left,right in zip(E.ARMS,E.ARMS[1:]):
        key=SHORT[right]+'-'+SHORT[left];contrasts[key]={}
        for g in groups:
            aa,bb=res[g][left],res[g][right];contrasts[key][g]=dict(PCK10_pp=100*(bb['twoD']['PCK']['10']-aa['twoD']['PCK']['10']),CURRENT_ADD_delta=bb['current']['ADDsym_AUC']-aa['current']['ADDsym_AUC'],ORACLE_ADD_delta=bb['oracle']['ADDsym_AUC']-aa['oracle']['ADDsym_AUC'])
        changes=[]
        for r in records:
            aa,bb=pm[left][r['id']]['current'],pm[right][r['id']]['current']
            if aa['available'] and bb['available']:changes.append((bb['ADDsym_normalized']-aa['ADDsym_normalized'],r))
        selected.extend((r,key+'_improved') for d,r in sorted(changes,key=lambda v:(v[0],v[1]['id'])) if d<0)
        selected=selected[:-max(0,len([x for x in selected if x[1]==key+'_improved'])-3)] if len([x for x in selected if x[1]==key+'_improved'])>3 else selected
        selected.extend((r,key+'_harmed') for d,r in sorted(changes,key=lambda v:(-v[0],v[1]['id']))[:3] if d>0)
    selected.extend((r,'hash_random_control') for r in sorted(records,key=lambda r:E.key(r['id']))[:6]);cases=[]
    for i,(r,label) in enumerate(selected):cases.append(dict(id=r['id'],label=label,file=case(r,label,i+1,truth,preds,poses,fm,pm)))
    # Report measured contrasts; no arbitrary success threshold or automatic next fit.
    hard1=[contrasts['T1-T0'][g] for g in groups[1:]];hard2=[contrasts['T2-T1'][g] for g in groups[1:]]
    inputsignal=all(v['CURRENT_ADD_delta']>0 and v['ORACLE_ADD_delta']>0 for v in hard1)
    targetsignal=all(v['CURRENT_ADD_delta']>0 and v['ORACLE_ADD_delta']>0 for v in hard2)
    decision=dict(status='COMPLETED_3_FITS',input_exposure_contrast='CONSISTENT_HARD_POSE_AND_CANDIDATE_GAIN' if inputsignal else 'MIXED_OR_NO_CONSISTENT_HARD_GAIN',target_coordinate_contrast='CONSISTENT_HARD_POSE_AND_CANDIDATE_GAIN' if targetsignal else 'MIXED_OR_NO_CONSISTENT_HARD_GAIN',contrasts=contrasts,uncertainty='single seed; 10 E/H pairs; reused DEV; legacy-reference heldout, no verified-manual channels; T2 not upper bound',next_one_experiment='고정 T0/T1/T2를 이미 보유한 별도 recording-disjoint 검증 자료에 적용하는 confirmation 설계. 먼저 수동 좌표 provenance와 recording 독립성을 확인하고, 검증 가능 자료가 없으면 새 성능 주장을 보류한다. 이번에는 실행하지 않음.',auto_followup=False)
    decision['next_one_experiment']='T2 고정 TRAIN H36채널 잔차 감사: 동일 last와 기존 occurrence에서 코너ID/좌표정의, native→model support, bbox-목표 관계를 확인해 남은3개 >20px 오차 원인 분리. 새 클릭·촬영·재학습·threshold 변경·평가GT 수정 없이 진단 설계만. 이번에는 실행하지 않음.'
    E.save(E.DOC/'DECISION.json',decision)
    lines=['# 기존 자료 EASY→HARD 전이 결정 실험','','## 결론과 데이터 계약','',f"**COMPLETED_3_FITS** — T0/T1/T2 각320 update, 총960. 신규 촬영0, 신규 수동 어노테이션0, 신규 교사 학습0. 입력 대조: `{decision['input_exposure_contrast']}`. 좌표 대조: `{decision['target_coordinate_contrast']}`.",'',f"TRAIN 기록: {split['train_recordings']}. HELDOUT 기록: {split['heldout_recordings']}. 66장의 TRAIN 기록 pool 중 실제 학생 고유 영상은 조건별20장(C0 10 + E 또는 H 10). 과거 EVAL에서56장이 TRAIN 기록 pool로 역할 변경됐지만, 실제 추가 학습 원본은 조건별10장이다. 세 조건 합집합으로는 E/H20장이다.",'','새 HELDOUT128 = Clean29 / Moderate21 / Severe78. 과거 평가300과 분모·recording 계약이 다르므로 이전 표에 끼워 넣거나 직접 향상률을 주장하지 않는다. 이미 열람된 DEV이며 새로운 TEST가 아니다. 목재 개선을 주장하지 않는다.','','![split](figures/01_recording_split.png)','','## 세 조건과 감독 예산','','T0: 공통 Clean10 + 추가 Clean10, 교사 pseudo. T1: 공통 Clean10 + 실제 가림10, 같은 교사 pseudo. T2: T1과 동일 RGB/bbox/증강/support에서 추가H 좌표만 기존 manual로 교체. H는 중간8/심함2. E/H는 같은 recording의 예산 대응이며 동일 자세의 가림 전후가 아니다. T1−T0는 시점·위치·자연가림이 함께 달라지는 분포 확장 총효과다.','',f"교사 fitting manual48좌표, T2 추가 고유 manual36좌표. replacement 직접 좌표 노출은 조건별4593개(키포인트 단위; scalar x/y는2배). provenance는 채널 선택에도 사용됐으므로 완전 무라벨 학습이 아니다. T2는 EXISTING_MANUAL_SUPERVISION_CONTROL이며 정확한 hidden GT 전체나 이론적 상한이 아니다.",'','![C0](figures/02_C0_contact.jpg)','','![E H not same pose](figures/03_EH_contact.jpg)','','![teacher manual](figures/04_H_teacher_manual.jpg)','','## 교사·분할 이력','','교사: 합성 PRIOR1 seed1/6000 → plastic Clean10/300step checkpoint 고정. 추가 real fitting 기록 없음. R0는 팔레트 추가 학습은 합성 전용이나 상위 COCO-pose 사전학습이 존재한다. 이 자료의 HELDOUT recording fitting과 구분한다. recording alias/partial-overlap/MAD2 근접중복은 예측 전에 통합했다. 원본319·기존 split/모델/표는 수정하지 않았다.','',f"사전 후보56개에서 교사 미검출 {pairs['teacher_no_detection']}개. 기존 self-visibility→visible refinement→hidden-PnP/fallback을 그대로 사용했다. 현재 Clean19 경로에는 별도 flip/LOO 탈락 필터가 없으므로 새 필터를 만들거나 통과율로 후보를 다시 고르지 않았다. 각 fallback 이유는 로컬 TRAIN_TEACHER_ONCE에 보존했다.",'','## 학습·증강 계약','','original R0부터 seed42, AdamW lr1e-4/lrf.1/cosine,640,batch/nbs16,5epoch. 실사2560=C01280+replacement1280, 합성2560. 합성512 source 순서·RGB·target과 C0 tensors는 기존 S1 그대로 재사용. 새 E/H는 실제 loader의 동일 seed/affine를 적용했다. S1의 고정 적용flag·fill·면적비·종횡비를 재사용하고 기존 사각형 중심을 bbox 정규화해 E/H 각 bbox로 옮겼다. 새 위치 탐색·visible≥4 표본 선별·S2 구조 배치는 없다. 크기·픽셀은 영상 bbox에 따라 다르며 정규화 정책만 동일하다. T1/T2는 RGB까지 exact. 세 조건 공통 transformed support를 사용하고 제외점 v=1 sentinel을 복구했다.','', '고정 last만 평가; validation은 학습 중 차단. 세 학생과 R0/old S1의 새 HELDOUT 예측을 먼저 freeze한 다음 GT 채점. 교사 bbox/class/confidence를 target 대조에서 변경하지 않았다.','','## 난도별 2D / 6D','']
    lines.insert(6,'이번 실험에서는 T0가 중간·심함 CURRENT ADD AUC에서 가장 좋았다. T1은 두 난도의 실제 자세·후보 품질을 악화했다. T2는 T1 대비 후보 oracle은 개선했지만 실제 pose는 중간만 개선되고 심함은 악화했다. T1 TRAIN pseudo36/36, T2 TRAIN manual33/36이10px 이내이며 T2 잔여3점은20px를 넘으므로 완전한 hard 정답 학습 성공을 전제하지 않는다. H는 기존 사람 난도 태그이며 저양각·모서리 모호성도 포함할 수 있다; 외부 가림 물체가 모든 H에 있다는 뜻은 아니다.')
    rows=[]
    for g in ['HELDOUT_ALL_PLASTIC',*groups]:
        for a in ('R0','OLD_S1',*E.ARMS):
            r=res[g][a];t=r['twoD'];p=r['current'];rows.append([g,SHORT.get(a,a),f(t['PCK']['10']),str(t['correct']['10'])+'/'+str(t['corners']),f(p['ADDsym_AUC']),f(r['oracle']['ADDsym_AUC']),f(r['selection_loss']),f(p['axis_accuracy'])])
    lines+=table(['집단','모델','PCK10','맞은점/분모','CURRENT AUC','ORACLE AUC 진단','선택손실','W/D parity'],rows)
    lines+=['','ORACLE는 POSTHOC GT 기반 NONDEPLOYABLE. W/D parity는 전체 회전 정확도가 아니다. 자세 reference는 geometry-derived이며 독립 실측 GT가 아니다. 새 HELDOUT은 직접 클릭 provenance가 확인되는 채널이0개이므로 **manual-only 평가는 N/A**다. legacy 전체 참조와 manual 표를 JSON에서 분리했고 unknown을 manual로 승격하지 않았다.','']
    for number,field in [(5,'PCK'),(6,'current'),(7,'oracle')]:lines+=['',f'![{field}](figures/{number:02d}_{field}.png)']
    lines+=['','### 두 대조의 변화량','']
    lines+=table(['대조','난도','PCK10 pp','CURRENT AUC Δ','ORACLE AUC Δ'],[[k,g,f(v['PCK10_pp'],3),f(v['CURRENT_ADD_delta'],6),f(v['ORACLE_ADD_delta'],6)] for k,gg in contrasts.items() for g,v in gg.items()])
    lines+=['','### 상세 분포','']
    lines+=table(['집단','모델','PCK5 / PCK20','매칭 median/P90 px','전체벌점 median/P90','R median/P90','yaw median/P90','t cm median/P90','IoU3D median/P90'],[[g,SHORT[a],f(res[g][a]['twoD']['PCK']['5'])+' / '+f(res[g][a]['twoD']['PCK']['20']),f(res[g][a]['twoD']['matched_pooled_corner8_median_px'],2)+' / '+f(res[g][a]['twoD']['matched_pooled_corner8_P90_px'],2),f(res[g][a]['twoD']['full_penalty_median_px'],2)+' / '+f(res[g][a]['twoD']['full_penalty_P90_px'],2),*[f(res[g][a]['current'][k]['median'],3)+' / '+f(res[g][a]['current'][k]['P90'],3) for k in ('rotation_deg','yaw_deg','translation_cm','IoU3D')]] for g in groups for a in E.ARMS])
    lines+=['','## source 보존·TRAIN fit','','![source](figures/08_source.png)','',f"고정 H 교사−manual 원영상 오차 median={train['teacher_manual_on_fixed_H']['median']:.3f}px, P90={train['teacher_manual_on_fixed_H']['P90']:.3f}px. 쌍 확정 이후 계산했으며 선택·가중치에는 사용하지 않았다.",'']
    lines+=table(['학생','부분','pseudo PCK10','manual PCK10','manual med/P90'],[[SHORT[a],role,f(v['pseudo']['PCK']['10']['fraction']),f(v['manual']['PCK']['10']['fraction']),f(v['manual']['median'],2)+' / '+f(v['manual']['P90'],2)] for a,rr in train['student'].items() for role,v in rr.items() if v['manual']['points']])
    lines+=['','TRAIN fit은 일반화 결과가 아니다. 숨은점의 독립검증 공백을 유지한다. source 2D 및 기존 검증 pose 평가는 [SOURCE_PRESERVATION.json](SOURCE_PRESERVATION.json)에 포함한다.','','## 개선·악화 및 hash-fixed 무작위 대조','','표시만 tight pallet ROI로 제한하고 얼굴 검출 부위는 픽셀화했다. 평가 입력은 원본 그대로다. 매칭실패는 숫자와 상태로 표시하며 긴 penalty 이동선은 그리지 않는다. 사례 개선/악화는 CURRENT normalized ADD 기준 사후 top3; 무작위6은 사전 hash 순서다.']
    for c in cases:lines+=['',f"### {c['label']} · {c['id']}",'',f"![{c['label']}](figures/{c['file']})"]
    lines+=['','## 한계·다음 한 실험','','단일 seed, 작은 E/H10쌍, 반복열람 DEV, 순수 occlusion 인과 대조 아님. 기록별 결과는 RESULTS.json에 모두 저장했다. 독립 기록은7개뿐이고 난도별 수가 더 적으므로 코너를 독립 표본으로 취급한 p-value나 정밀 CI를 만들지 않는다. T2의 일부 관측 코너만 감독한 결과로 단일RGB 불가능성 또는 완전 정답 hard 학습 실패를 주장하지 않는다.','',decision['next_one_experiment'],'','## 재현·실행 기록','',f"HEAD_BEFORE: `{split['head_before']}`. 새 namespace만 공개. checkpoint/캐시/원본 전체 영상은 로컬에 보존한다. 지시문 MD/TXT는 동일 내용 확인. 준비 중 괄호 문법 오류와 registry 타입 별칭을 수정한 뒤 진행했으며 optimizer 실행 전이었다. 학습 정책·loss·seed·예산 변경 및 재학습 없음.",'','[SPLIT_LOCK](SPLIT_LOCK.json) · [TARGET_AND_PAIR_AUDIT](TARGET_AND_PAIR_AUDIT.json) · [INPUT_LOCK](INPUT_LOCK.json) · [RESULTS](RESULTS.json) · [TRANSITIONS](TRANSITIONS.json) · [DECISION](DECISION.json)']
    E.save(E.DOC/'REPORT_KO.md','\n'.join(lines)+'\n');E.save(E.DOC/'SUMMARY_KO.md','\n'.join(lines[:10])+'\n\n[전체 이미지 보고서](REPORT_KO.md)\n');E.save(E.DOC/'CASE_MANIFEST.json',dict(cases=cases,random_rule='sha256(existing-hard-v1+frame_id) first6',display_only_crop=True))
    print('REPORT_READY',decision,flush=True)

if __name__=='__main__':main()
