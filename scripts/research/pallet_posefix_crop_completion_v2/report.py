"""Outcome-first illustrated report, including failures and fixed controls."""
from collections import defaultdict
import html
import math
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from . import common as C
from . import data as D
from .evaluate import peak_detail
EDGES=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

def main():
    lock=C.read(C.DOC/'INPUT_LOCK.json');freeze=C.read(C.DOC/'PREDICTION_LOCK.json');metrics=C.read(C.RAW/'FRAME_METRICS.json')
    results=C.read(C.DOC/'REAL_RESULTS.json');source=C.read(C.DOC/'SOURCE_RESULTS.json');fit=C.read(C.DOC/'TRAIN_FIT_RESULTS.json');heat=C.read(C.DOC/'HEATMAP_SUMMARY.json')
    diag=C.read(C.RAW/'HEATMAP_DIAGNOSTICS.json');sub=C.read(C.DOC/'SUBSET_LOCK.json');review=C.read(C.DOC/'REVIEW_STATUS.json');inputs=C.read(C.DOC/'TRAIN_INPUT_AUDIT.json')
    primary=results['summary']['PRIMARY_OCC96'];trans=results['transitions']['PRIMARY_OCC96'];pops=results['summary'];groups=results['subgroups']
    n_gain=groups['N14']['models']['C']['correct10']-groups['N14']['models']['A']['correct10'];overall=primary['C']['correct10']-primary['A']['correct10']
    p90='matched_pooled_corner8_P90_px'
    safety=dict(N14_gain_atleast1=n_gain>=1,primary_no_loss=overall>=0,BASE_loss_no_worse=trans['C']['BASE']['GB']<=trans['A']['BASE']['GB'],N2_loss_no_worse=trans['C']['N2']['GB']<=trans['A']['N2']['GB'],
        GREEN_no_loss=pops['GREEN150_MANUAL']['C']['correct10']>=pops['GREEN150_MANUAL']['A']['correct10'],
        CLEAN_no_loss=pops['CLEAN_NONCAD69']['C']['correct10']>=pops['CLEAN_NONCAD69']['A']['correct10'],
        source_clean_no_loss=source['results']['C']['clean']['COMMON_SUPPORT']['correct10']>=source['results']['A']['clean']['COMMON_SUPPORT']['correct10'],
        source_stress_no_loss=source['results']['C']['stress']['COMMON_SUPPORT']['correct10']>=source['results']['A']['stress']['COMMON_SUPPORT']['correct10'],
        P90_no_worse=primary['C'][p90]<=primary['A'][p90])
    system='NET_GAIN_WITH_PRESERVATION' if all(safety.values()) else 'TRADE_OFF' if n_gain>0 or overall>0 or trans['C']['A']['BG']>0 else 'NO_OBSERVED_GAIN'
    trainfit=fit['results'];before=trainfit['PRIOR1_150']['real_OCC'];after=trainfit['C']['real_OCC']
    fit_improves=after['COMMON_SUPPORT']['correct10']>before['COMMON_SUPPORT']['correct10'] and after['task_heatmap']+after['task_coordinate']<before['task_heatmap']+before['task_coordinate']
    h=heat['summary']['H163']['C'];decoder=h['top5_rescue_fraction_reachable']>=C.protocol()['decoder_route_fraction']
    mechanism='RECOVERY_OBSERVED' if n_gain>0 else 'ADAPTATION_UNRESOLVED' if not fit_improves else 'CANDIDATE_BUT_DECODE_FAIL' if decoder else 'SUPPORT_ONLY'
    if decoder:route='DECODER_UNCERTAINTY'
    elif n_gain>0 and not all(safety.values()):route='ADAPTIVE_SUPPORT'
    else:route='TARGET_DATA_AND_TRAIN_FIT'
    decision=dict(mechanism=mechanism,system=system,checks=safety,N14_net_gain=n_gain,primary_net_gain=overall,
        fit_improves_on_fixed_real_OCC=fit_improves,decoder_routing_fraction=h['top5_rescue_fraction_reachable'],next_primary_route=route,
        scope='single seed, repeatedly used DEV, descriptive pilot; no statistical or physical-coordinate guarantee',no_final_promotion=True)
    C.save(C.DOC/'DECISION.json',decision)
    # Use locked whole-object mapping for all mechanism heatmaps; top panels show official per-model mapping explicitly.
    rec={r['id']:r for r in lock['eval_records']};syms={r['object_type']:r['permutations'] for r in C.read(C.ROOT/lock['symmetry']['path'])['objects']}
    truth=C.read(C.B.E.V.RAW/'E1_FRAME_METRICS.json')['truth_for_display_only'];fp=C.read(C.F.RAW/'FRAME_METRICS.json')['FULL_PRESERVE']
    preds={a:C.read(C.ROOT/x['predictions']['path'])['predictions'] for a,x in freeze['arms'].items()};raw=C.read(C.B.E.V.RAW/'FROZEN_PREDICTIONS.json')['predictions']['R0']
    plan=C.read(C.DOC/'GALLERY_PLAN.json');selections=defaultdict(list)
    def add(group,rows):
        byframe=defaultdict(list)
        for r in rows:byframe[r['id']].append(r['canonical'])
        for fid,js in byframe.items():selections[group].append((fid,sorted(set(js))))
    add('N14_ALL',sub['groups']['N14']);add('U15_FIXED5',sub['groups']['U15'][:5]);add('RANDOM_HARD_FIXED',plan['random_hard'])
    cr=[r for r in C.read(C.RAW/'CORNER_ROWS.json') if r['frame_id'] in lock['populations']['PRIMARY_OCC96']]
    for name,predicate in [('A_TO_C_WINS',lambda r:r['errors']['A']>10 and r['errors']['C']<=10),('A_TO_C_DAMAGE',lambda r:r['errors']['A']<=10 and r['errors']['C']>10)]:
        chosen=sorted([r for r in cr if predicate(r)],key=lambda r:-abs(r['errors']['C']-r['errors']['A']))[:5]
        add(name,[dict(id=r['frame_id'],canonical=r['corner_id']) for r in chosen])
    for category in ('PEAK_PRESENT_DECODE_WRONG','NO_TESTED_PEAK','IDENTITY_SUSPECT'):
        rr=[r for r in diag if r['model']=='C' and r['fixed_set'] and (r['identity_suspect'] if category=='IDENTITY_SUSPECT' else r['category']==category)]
        add(category,rr[:5])
    for fid in plan['random_GREEN']:add('RANDOM_GREEN_FIXED',[dict(id=fid,canonical=int(np.flatnonzero(metrics['R0'][fid]['canonical_valid'])[0]))])
    figdir=C.DOC/'figures';figdir.mkdir(exist_ok=False);gallery=[];image_index=0;bindings=[]
    for group,frames in selections.items():
        for fid,js in frames:
            image_index+=1;C.verify(rec[fid]['image']);im=cv2.imread(str(C.ROOT/rec[fid]['image']['path']))[:,:,::-1];gt=np.array(truth[fid]['gt']);gv=np.array(truth[fid]['valid'])
            fig,axes=plt.subplots(1+math.ceil(len(js)/4),4,figsize=(16,4.5*(1+math.ceil(len(js)/4))),squeeze=False,facecolor='#101c26')
            box=C.B.E.P.top(raw[fid])['box_xyxy'];perms=syms[rec[fid]['object_type']]
            for col,a in enumerate(C.ARMS):
                ax=axes[0,col];ax.imshow(im);native=np.array(C.B.E.P.top(preds[a][fid])['keypoints_xy']);perm=perms[metrics[a][fid]['branch']];canonical=np.empty_like(native);canonical[perm]=native
                for x,y in EDGES:
                    if gv[x] and gv[y]:ax.plot(gt[[x,y],0],gt[[x,y],1],color='#65fc78',lw=.75,alpha=.8)
                    ax.plot(canonical[[x,y],0],canonical[[x,y],1],color='#f4d966',lw=.8)
                for j in js:
                    ax.scatter(*gt[j],marker='x',c='#65fc78',s=70);ax.scatter(*canonical[j],s=45,facecolors='none',edgecolors='#50deff');ax.text(*canonical[j],f'G{j}',color='#50deff',fontsize=8)
                for e,color in [(C.BASE_EXPANSION,'#ffb04c'),(C.EXPANSION,'#d190ff')]:
                    rect=D.transform_points(np.array([[0,0],[288,0],[288,384],[0,384]]),np.linalg.inv(D.matrix(box,e)));ax.add_patch(Polygon(rect,fill=False,color=color,lw=.7,ls='--'))
                errors=', '.join(f'G{j}:{metrics[a][fid]["canonical_errors"][j]:.1f}' for j in js)
                ax.set_title(f'{a}: train{1.25 if a in "AB" else 1.50:.2f}/infer{C.ARMS[a][1]:.2f}\nofficial branch {metrics[a][fid]["branch"]} | {errors}px',color='white',fontsize=9)
                ax.set_xlim(0,640);ax.set_ylim(480,0)
            cache=np.load(C.ROOT/freeze['arms']['C']['heatmaps'][fid]['path']);fixedperm=perms[fp[fid]['branch']]
            for k,j in enumerate(js):
                ax=axes[1+k//4,k%4];ch=fixedperm.index(j);v=peak_detail(cache,ch,gt[j]);m=cache['matrix'];rgb=cv2.warpAffine(im,m[:2],(288,384))
                ax.imshow(rgb);prob=np.exp(cache['logits'][ch]-cache['logits'][ch].max());ax.imshow(prob,extent=(-2,286,382,-2),cmap='inferno',alpha=.48,vmin=0,vmax=1)
                ax.scatter(*v['GT_crop'],marker='x',c='#65fc78',s=85);ax.scatter(*cache['expectation'][ch],c='#50deff',s=30);ax.scatter(*v['argmax_crop'],marker='+',c='#ff77ef',s=65)
                peaks=np.array(v['top5_crop']);ax.scatter(peaks[:,0],peaks[:,1],s=60,facecolors='none',edgecolors='yellow')
                ax.add_patch(Polygon([[0,0],[284,0],[284,380],[0,380]],fill=False,color='white',lw=.8))
                g=v['GT_crop'];ax.set_xlim(min(0,g[0]-10),max(288,g[0]+10));ax.set_ylim(max(384,g[1]+10),min(0,g[1]-10))
                oldmin=C.S.support(gt[j],D.matrix(box,C.BASE_EXPANSION))['minimum_distance_px']
                ax.set_title(f'C fixed P{ch}->G{j} (historical branch {fp[fid]["branch"]})\nmin support {oldmin:.1f}->{v["nearest_output_error"]:.1f}px\nE {v["expected_error"]:.1f}; arg {v["argmax_error"]:.1f}; top5 {v["top5_nearest_error"]:.1f}',color='white',fontsize=9)
            for k in range(len(js),4*(len(axes)-1)):axes[1+k//4,k%4].axis('off')
            for ax in axes.flat:ax.tick_params(colors='white',labelsize=6)
            fig.suptitle(f'{image_index:03d} {group} | green=legacy/manual reference; yellow=model | heatmap is posthoc diagnostic',color='white',fontsize=11);fig.tight_layout()
            path=figdir/f'comparison_{image_index:03d}.jpg';assert not path.exists();fig.savefig(path,dpi=100,facecolor=fig.get_facecolor());plt.close(fig)
            gallery.append(dict(group=group,path=f'figures/{path.name}',case=image_index));bindings.append(dict(id=fid,corners=js,group=group,path=str(path.relative_to(C.ROOT))))
    C.save(C.RAW/'GALLERY_SELECTION.json',bindings)
    doc=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>Crop completion A/B/C/D</title><style>body{background:#101c26;color:#eee;max-width:1700px;margin:auto;font:18px sans-serif}img{width:100%;margin:12px 0}a{color:#84daff}</style><h1>Crop completion A/B/C/D</h1><p>A/B 같은 기존 weight; C/D 같은 새 weight. 초록=기존 reference, 노랑=모델. 상단 공식 whole-object branch, 하단 기존 고정 branch의 native heatmap. GT는 추론 후 진단에만 사용.</p>']
    for g in gallery:doc+=['<h2>'+html.escape(g['group'])+'</h2>',f'<img loading="lazy" src="{g["path"]}" alt="{g["group"]}">']
    C.save(C.DOC/'GALLERY.html','\n'.join(doc)+'\n</html>\n')
    md=['# Crop completion v2 — 실제 학습·네 조건 비교','',f'신규학습 **C 1개 ×300update**. 기전 **{mechanism}**, 시스템 **{system}**. 과거 E0 실패는 변경하지 않았으며 이번은 별도로 승인된 탐색 실험이다.','',
        '| 조건 | primary ≤10 | N14 | I11 | E3 | U15 | R134 | FULL 정답손실 | N2 정답손실 | GREEN ≤10 | source clean ≤10 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in C.ARMS:
        md.append('| '+' | '.join([a,str(primary[a]['correct10'])]+[str(groups[g]['models'][a]['correct10']) for g in ('N14','I11','E3','U15','R134')]+[str(trans[a]['A']['GB']),str(trans[a]['N2']['GB']),str(pops['GREEN150_MANUAL'][a]['correct10']),str(source['results'][a]['clean']['COMMON_SUPPORT']['correct10'])])+' |')
    md+=['','A=기존FULL125/infer125, B=같은weight/infer150, C=PRIOR1부터 새FULL150/infer150, D=Cweight/infer125. B/D는 진단조건이며 DEV 최고값으로 최종 모델을 선택하지 않았다.',
        '',f'주 분모: primary {primary["A"]["total_frames"]}장/{primary["A"]["corners"]}코너. N14 등 이름은 기존 고정 집합이며 실제 counts={sub["counts"]}. GREEN은 수동좌표집합으로 분리. source는 기존1.25 고정{source["results"]["A"]["clean"]["COMMON_SUPPORT"]["n"]}코너, 새지원은 별도 표. 겹치는 모집단을 독립 재현으로 세지 않는다.',
        '', '## 핵심 질문과 판단','',
        f'- 범위를 열어 준 {groups["N14"]["n"]}개 중 실제 ≤10 정답: A {groups["N14"]["models"]["A"]["correct10"]}, B {groups["N14"]["models"]["B"]["correct10"]}, C {groups["N14"]["models"]["C"]["correct10"]}, D {groups["N14"]["models"]["D"]["correct10"]}. 도달가능과 실제복구를 구분한다.',
        f'- C−A 전체 정답 순변화 {overall:+d}개. A 정답손실 {trans["C"]["A"]["GB"]}, 새획득 {trans["C"]["A"]["BG"]}이며 BG−GB 일치.',
        f'- B−A는 같은 weight의 입력확대 효과. C−B는 동일확대입력에서 각자 PRIOR1부터 같은300step 학습한 weight 차이이며 FULL을 추가학습한 것이 아니다. C/D 차이는 아래 네조건 표에 그대로 보고한다.',
        f'- C의 고정 hard 분류: {h["categories"]}. reachable {h["reachable"]}개 중 top5 oracle rescue {h["top5_oracle_rescue"]}개 ({100*h["top5_rescue_fraction_reachable"]:.2f}%). argmax rescue {h["argmax_rescue"]}개. 다른채널 의심 {h["identity_suspect"]}개는 중복 보조태그이며 합산하지 않는다.',
        f'- 고정 real OCC TRAIN probe의 PRIOR1_150→C: PCK10 {100*before["COMMON_SUPPORT"]["PCK10"]:.2f}→{100*after["COMMON_SUPPORT"]["PCK10"]:.2f}%, task(CE+coord) {before["task_heatmap"]+before["task_coordinate"]:.4f}→{after["task_heatmap"]+after["task_coordinate"]:.4f}. 이것은 TRAIN 적합도이지 독립 일반화 성공이 아니다.',
        '- 다른채널 후보는 승인순열 대응 가능 여부와 모델 전체물체 branch 일치 여부를 구분했다. 자유 점별 재대응/새 selector/정답변경 없음. 후보가 없다고 RGB 정보가 없거나 아키텍처가 원천적으로 불가능하다고 결론내리지 않는다.',
        f'- 사람검토 {review["status"]}. 기존 U29 연결 {review["existing_cases"]}, 같은양식 추가 {review["additional_cases"]}코너. 새 GT/가시성 응답 생성 없음. 기존reference의 물리 좌표정확도와 external/self-occlusion subtype는 미확인이다.',
        '', '## 전체 성능','', '| 모집단 | 조건 | PCK5% | PCK10% | PCK20% | matched med | matched P90 | >20 수 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for pop,ss in pops.items():
        for a,s in ss.items():md.append(f'| {pop} | {a} | {100*s["PCK"]["5"]:.2f} | {100*s["PCK"]["10"]:.2f} | {100*s["PCK"]["20"]:.2f} | {s["matched_pooled_corner8_median_px"]:.2f} | {s[p90]:.2f} | {s["gross20_count"]} |')
    md+=['','검출/매칭/bbox/score/후보 identity 고정. 실패8장/54코너의800px penalty는 official PCK에서 유지하고 matched median/P90에서만 제외한다. 가림태그93장 정확도를 외부가림 코너 정확도로 부르지 않는다.',
        '', '## 고정 R0 band — primary','', '| band | 코너수 | A≤10 | B≤10 | C≤10 | D≤10 |','|---|---:|---:|---:|---:|---:|']
    for band,br in results['bands']['PRIMARY_OCC96'].items():md.append('| '+' | '.join([band,str(br['n'])]+[str(br['models'][a]['correct10']) for a in C.ARMS])+' |')
    md+=['','## 보존 판정','', '| 조건 | 충족 |','|---|---|']+[f'| {k} | {v} |' for k,v in safety.items()]
    md+=['','## Source: 고정분모와 신규지원 분리','', '| 조건 | 입력 | 집합 | n | ≤10 | PCK10% | PCK20% | median | P90 |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for a,ss in source['results'].items():
        for mode,parts in ss.items():
            for name,s in parts.items():md.append(f'| {a} | {mode} | {name} | {s["n"]} | {s["correct10"]} | {100*s["PCK10"]:.2f} | {100*s["PCK20"]:.2f} | {s["median"]:.2f} | {s["P90"]:.2f} |')
    md+=['','## TRAIN-fit (독립 평가 아님)','', '| 조건 | probe | common n | PCK10% | median | heatmap loss | coordinate loss | 신규감독 n |','|---|---|---:|---:|---:|---:|---:|---:|']
    for a,ss in trainfit.items():
        for mode,s in ss.items():md.append(f'| {a} | {mode} | {s["COMMON_SUPPORT"]["n"]} | {100*s["COMMON_SUPPORT"]["PCK10"]:.2f} | {s["COMMON_SUPPORT"]["median"]:.2f} | {s["task_heatmap"]:.4f} | {s["task_coordinate"]:.4f} | {s["NEW_SUPPORT"]["n"]} |')
    md+=['','## Heatmap 분해','', '| 집합 | 조건 | n | support fail | decoded correct | peak 있으나decode실패 | 시험한peak없음 | identity보조 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for pop,ss in heat['summary'].items():
        for a,v in ss.items():md.append('| '+' | '.join([pop,a,str(v['n'])]+[str(v['categories'].get(k,0)) for k in ('SUPPORT_FAIL','DECODED_CORRECT','PEAK_PRESENT_DECODE_WRONG','NO_TESTED_PEAK')]+[str(v['identity_suspect'])])+' |')
    md+=['','5×5 local maxima와 고정top5 규칙 유지. native↔canonical은 기전진단에서 과거FP branch 고정, 공식평가는 모델별whole-object branch. raw10px probability mass뿐 아니라 grid면적/균일기대mass 대비비와 log(grid수) 정규화entropy를 원본진단에 저장. top5 nearest-GT는 oracle이지 배포성능이 아니다.',
        '', '## 학습 계약·감독량·한계','', '| 구분 | 기존 감독 | 확대 감독 | 신규 | 손실 |','|---|---:|---:|---:|---:|']
    for key in ('real','source_train','real_exposures','source_exposures','source_heldout'):
        v=inputs['counts'][key];md.append(f'| {key} | {v["old"]} | {v["new"]} | {v["newly"]} | {v["lost"]} |')
    md+=['','실사253장 동일OCC 원영상/초기점/pseudo target/order, 합성동일row/order와 원영상교란. crop1.25 baseline 모든재구성tensor와 기존300step source교란hash 일치. source normal/stress 입력고정, 새감독코너가 RNG소비를 바꾸지 않음. PRIOR1 seed1, TFAdam1e-4,300step,real8+source8,micro2,FULL+BN통계/affine동결,preserveOFF,last300만. 중간평가·재학습구제 없음.',
        '', 'crop확대는 support만 바꾸지 않는다: 입력상물체scale5/6, 원영상grid1.2배, crop좌표loss의 원영상당 크기 및 신규감독량도 달라진다. 따라서 총crop경로 효과로만 해석한다. 단일seed, 반복DEV, 사용자대략clean 구간과 pseudo신뢰의 한계도 유지한다.',
        '', '학습 전 JSON 정수 직렬화 오류1회는 IO만 수정했다. 이전준비tensor/교란을 exact검증 재사용했고 학습은1회뿐이다.',
        '', '## 세션별 paired 정답 변화','', '| 세션 | n코너 | A | C | A→C 손실 | A→C 획득 |','|---|---:|---:|---:|---:|---:|']
    for sn,v in results['sessions']['PRIMARY_OCC96'].items():md.append(f'| {sn} | {v["corners"]} | {v["correct10"]["A"]} | {v["correct10"]["C"]} | {v["A_C"]["GB"]} | {v["A_C"]["BG"]} |')
    md+=['','## 다음 질문 하나','',f'**{route}**. 추가 실행 없이 다음 계획만 남겼다. 기존 최종 모델·논문표는 변경하지 않았다.','',
        '## 이미지','', '[전체 갤러리](GALLERY.html). N14 전부(프레임별 통합), 고정U15/무작위대조, 정답획득·손실 및 남은오류를 함께 제시한다. 상단의 모델별대칭branch와 하단의고정branch를 구분해야 하며, 서로다른branch의 같은G표시를 물리적으로같은native점의 이동이라고 단정하지 않는다.']
    for g in gallery:md+=['',f'### {g["case"]:03d} {g["group"]}','',f'![{g["group"]}]({g["path"]})']
    C.save(C.DOC/'REPORT_KO.md','\n'.join(md).rstrip()+'\n')
    C.save(C.DOC/'RESULTS.json',dict(real=results,source=source,train_fit=fit,heatmap=heat,decision=decision,
        supervision=inputs['counts'],review={k:review[k] for k in ('status','old29','existing_cases','additional_cases','additional_images','no_auto_answers','no_coordinate_substitution')}))
    C.save(C.DOC/'NEXT_STAGE_PLAN.md',f'# 다음 질문 하나: {route}\n\n기전={mechanism}, 시스템={system}. 새학습 실행 없음. '+
        ('정답근처 후보는 있으나 expectation이 실패하는 고정사례에서 GT선택 없이 일관된 mode를 고르는 근거가 있는가? 우선후보/불확실성 설계만 비교하고별도승인 전구현/학습 금지.\n' if route=='DECODER_UNCERTAINTY' else
         '얻은복구를 유지하면서 정상점손실을 줄일수있는 point-aware crop 한후보를 별도설계할 수 있는가? GT로crop마진튜닝금지,이번실행안함.\n' if route=='ADAPTIVE_SUPPORT' else
         '고정TRAIN과 실제DEV의 목표/입력오류분포 차이 중 무엇이 잔여후보생성 실패를 설명하는가? 현재teacher타깃과독립좌표정의의차이를 기존자료로감사하는계획만,새모델/GT자동생성 금지.\n'))
    print('REPORT_COMPLETE',len(gallery),decision,flush=True)

if __name__=='__main__':main()
