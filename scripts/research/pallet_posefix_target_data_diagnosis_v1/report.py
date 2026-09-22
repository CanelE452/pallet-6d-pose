"""Descriptive figures and complete decision report, no new inference."""
from collections import Counter
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from . import common as C
from .probe import controlled_occ,corrupt

def pct(x):return 'N/A (n=0)' if x is None else f'{100*x:.2f}%'

def main():
    s=C.read(C.DOC/'TRAIN_CORNER_CENSUS_SUMMARY.json');d=C.read(C.DOC/'TRAIN_DEV_DISTRIBUTION.json');h=C.read(C.DOC/'TRAIN_DEV_HEATMAP_COMPARE.json')
    p=C.read(C.DOC/'CONTROLLED_PROBE_RESULTS.json');n=C.read(C.DOC/'NATURAL_HARD_TRAIN_FIT.json');decision=C.read(C.DOC/'DECISION.json');cons=C.read(C.DOC/'PSEUDO_CONSISTENCY_AUDIT.json')
    ex=C.read(C.DOC/'TRAIN_EXPOSURE_AUDIT.json');tmp=C.read(C.DOC/'TEMPORAL_DIVERSITY_AUDIT.json');pr=C.read(C.DOC/'EVAL_PROVENANCE_AUDIT.json');m=C.read(C.DOC/'ERROR_MORPHOLOGY_AUDIT.json')
    train=C.read(C.RAW/'TRAIN_CORNER_CENSUS.json');dev=C.read(C.RAW/'DEV_CORNER_ROWS.json');frames=C.read(C.RAW/'TRAIN_FRAMES.json');protocol=C.read(C.DOC/'CONTROLLED_PROBE_PROTOCOL.json')
    tc={(r['index'],r['corner_id']):r for r in train};frozen=C.read(C.DOC/'PREDICTIONS_FROZEN.json');models={a:C.read(C.ROOT/b['path']) for a,b in frozen['models'].items()};full=models['FULL125']
    figs=C.DOC/'figures';figs.mkdir(exist_ok=False);plots=[]
    def finish(fig,name):
        fig.tight_layout();fig.savefig(figs/f'{name}.png',dpi=110);plt.close(fig);plots.append(name)
    names=['<=5','5-10','10-20','20-40','>40']
    for label,vals in [('train_clean',s['clean']['bands']),('train_occ',s['occ']['bands']),('dev_R0',d['DEV_MATCHED']['bands'])]:
        fig,ax=plt.subplots(figsize=(6,4));ax.bar(names,vals);ax.set(title=label+' (different reference provenance)',ylabel='corners');finish(fig,label)
    fig,ax=plt.subplots(figsize=(7,4))
    for label,values in [('TRAIN_OCC pseudo',[r['occ_error_norm'] for r in train]),('DEV reference',[r['error_norm'] for r in dev])]:
        a=np.sort(values);ax.plot(a,np.arange(1,len(a)+1)/len(a),label=label)
    ax.set(xlabel='error / raw bbox diagonal',ylabel='empirical CDF',xlim=(0,.6));ax.legend();finish(fig,'normalized_CDF')
    fig,ax=plt.subplots(figsize=(6,5));ax.imshow(s['transition'],cmap='Blues');ax.set_xticks(range(5),names);ax.set_yticks(range(5),names);ax.set(xlabel='OCC error',ylabel='CLEAN error')
    for i in range(5):
        for j in range(5):ax.text(j,i,str(s['transition'][i][j]),ha='center',va='center',color='orange' if s['transition'][i][j]>500 else 'black')
    finish(fig,'clean_OCC_transition')
    fig,ax=plt.subplots(figsize=(10,3));counts=Counter(r['display_index'] for r in train if r['occ_error_px']>20);ax.bar([f['display_index'] for f in frames],[counts[f['display_index']] for f in frames]);ax.set(xlabel='original display index (gaps = excluded)',ylabel='hard corners',title='One recording, not independent scenes');finish(fig,'hard_timeline')
    fig,ax=plt.subplots(figsize=(7,4));xx=np.arange(8)
    for k,(label,rr,key) in enumerate([('TRAIN',train,'occ_error_px'),('DEV',dev,'error')]):
        c=Counter(r['corner_id'] for r in rr if r[key]>20);ax.bar(xx+(k-.5)*.35,[c[j] for j in xx],width=.35,label=label)
    ax.set_xticks(xx);ax.legend();ax.set(title='Hard corner-ID distribution (symmetry conventions differ)',xlabel='corner ID');finish(fig,'corner_ids')
    fig,ax=plt.subplots(figsize=(6,4));ax.bar(names,[b['strict_count']/b['corners'] if b['corners'] else 0 for b in s['bands']['occ']]);ax.set(ylim=(0,1.1),ylabel='STRICT fraction',title='>40: n=0; all observed bands STRICT=1');finish(fig,'consistency_hardness')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,mode in zip(axs,('P1','P2')):
        for a in C.ARMS:
            curve=p['curves'][a][mode];rs=sorted(map(int,curve));ax.plot(rs,[curve[str(r)]['PCK10'] for r in rs],'-o',label=a)
        ax.set(xlabel='original-coordinate radius px',ylabel='recovery <=10',ylim=(0,1.05),title=mode+' TRAIN pseudo target');ax.legend(fontsize=8)
    finish(fig,'controlled_basin')
    fig,ax=plt.subplots(figsize=(6,4));ax.bar(['TRAIN strict hard','DEV hard','DEV reachable hard'],[h[k]['top5_present10'] for k in ('TRAIN_STRICT_HARD','DEV_MATCHED_HARD','DEV_reachable')]);ax.set(ylabel='own-channel top5 within10 rate',ylim=(0,1.05),title='FULL125; oracle diagnostic, not a selector');finish(fig,'candidate_gap')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for k,(label,v) in enumerate(m.items()):
        if label not in ('TRAIN','DEV'):continue
        hist=v['hard_count_histogram'];axs[0].plot(range(9),[hist.get(str(j),0)/v['frames'] for j in range(9)],'-o',label=label)
        axs[1].bar(label,v['translation_removed_RMSE_median'])
    axs[0].legend();axs[0].set(xlabel='hard corners / frame',ylabel='frame fraction');axs[1].set(ylabel='translation-removed RMSE px (hard frames)');finish(fig,'morphology')
    # Deterministic case examples: fixed order, do not duplicate cases to fill a quota.
    nat=[r for r in full if r['mode']=='NATURAL_OCC'];clean=[r for r in full if r['mode']=='NATURAL_CLEAN'];ctrl=[r for r in full if r['mode']=='P1' and r['radius']==30]
    hard=[r for r in nat if tc[(r['index'],r['corner_id'])]['occ_error_px']>20]
    def unique(rr,k,byframe=False):
        out=[];seen=set()
        for r in rr:
            key=r['index'] if byframe else (r['index'],r['corner_id'])
            if key not in seen:out.append(r);seen.add(key)
            if len(out)==k:break
        return out
    groups={'strict_TRAIN_hard':unique(hard,10,True),'TRAIN_easy':unique([r for r in clean if r['input_error']<=5],5,True),
        'controlled_r30_success':unique([r for r in ctrl if r['output_error']<=10],5),'controlled_r30_failure':unique([r for r in ctrl if r['output_error']>10],5),
        'natural_hard_recovered':unique([r for r in hard if r['output_error']<=10],5),'natural_hard_failed':unique([r for r in hard if r['output_error']>10],5)}
    rng=np.random.default_rng(20260922);random_frames=rng.choice(253,10,replace=False);groups['deterministic_random10']=[next(r for r in nat if r['index']==int(i)) for i in random_frames]
    selections=[];gallery=[]
    for group,rr in groups.items():
        for r in rr:
            i=r['index'];j=r['corner_id'];entry=C.pair(i);mode=r['mode'];item=entry['pair']['CLEAN' if mode in ('P1','NATURAL_CLEAN') else 'OCC']
            if mode=='P1':item=corrupt(item,j,r['radius'],r['angle'])
            rgb=np.clip(item['rgb'].transpose(1,2,0)+C.D.CORE.MEAN,0,255).astype(np.uint8)
            target=np.array(entry['metadata']['target_original']);orig_in=np.array(r['input_xy']);out=np.array(r['output_xy']);matrix=item['matrix']
            pts=C.D.transform_points(np.stack([target[j],orig_in,out]),matrix)
            fig,axs=plt.subplots(1,2,figsize=(8,5))
            for ax in axs:
                ax.imshow(rgb);ax.scatter(*pts[0],marker='x',c='lime',s=95,label='frozen pseudo (not GT)');ax.scatter(*pts[1],facecolors='none',edgecolors='orange',s=95,label='input');ax.scatter(*pts[2],marker='+',c='cyan',s=95,label='FULL125 output')
                ax.annotate('',xy=pts[2],xytext=pts[1],arrowprops=dict(arrowstyle='->',color='cyan'))
            axs[0].set_xlim(0,288);axs[0].set_ylim(384,0);axs[0].legend(fontsize=7)
            lo=np.min(pts,axis=0)-25;hi=np.max(pts,axis=0)+25;axs[1].set_xlim(lo[0],hi[0]);axs[1].set_ylim(hi[1],lo[1])
            fig.suptitle(f'{group} | P{j} | input {r["input_error"]:.2f}px -> output {r["output_error"]:.2f}px\nTRAIN pseudo target, not independent physical GT',fontsize=10)
            name=f'case_{len(gallery)+1:03d}';fig.tight_layout();fig.savefig(figs/f'{name}.png',dpi=100);plt.close(fig);gallery.append(dict(group=group,name=name));selections.append(dict(group=group,row=r))
    lock=C.read(C.P.DOC/'INPUT_LOCK.json');recs={r['id']:r for r in lock['eval_records']};bad=[r for r in dev if r['error']>20 and r['top5_error']>10][:5]
    for r in bad:
        record=recs[r['frame_id']];C.verify(record['image']);im=cv2.imread(str(C.ROOT/record['image']['path']))[:,:,::-1];points=np.array([r['target'],r['input_xy'],r['output_xy']]);fig,axs=plt.subplots(1,2,figsize=(10,4))
        for ax in axs:
            ax.imshow(im)
            for v,marker,color,label in zip(points,['x','o','+'],['lime','orange','cyan'],['legacy reference','R0','FULL125']):ax.scatter(*v,marker=marker,c=color,s=75,label=label)
        axs[0].legend(fontsize=7);lo=points.min(0)-30;hi=points.max(0)+30;axs[1].set_xlim(lo[0],hi[0]);axs[1].set_ylim(hi[1],lo[1]);fig.suptitle(f'DEV no own top5 | G{r["corner_id"]} | R0 {r["error"]:.1f}px; FULL {r["output_error"]:.1f}px\nReference physical identity unreviewed, not new GT',fontsize=10)
        name=f'case_{len(gallery)+1:03d}';fig.tight_layout();fig.savefig(figs/f'{name}.png',dpi=100);plt.close(fig);gallery.append(dict(group='DEV_no_candidate',name=name));selections.append(dict(group='DEV_no_candidate',row=r))
    C.save(C.RAW/'GALLERY_SELECTION.json',selections);C.save(C.DOC/'GALLERY_COUNTS.json',dict(Counter(r['group'] for r in gallery)))
    fc=p['curves']['FULL125'];df=d['DEV_MATCHED'];nf=n['FULL125']['STRICT_HALF_THRESHOLD']
    result=dict(STATUS='COMPLETE_NO_NEW_TRAINING',HEAD=C.read(C.DOC/'INPUT_BINDINGS.json')['HEAD'],NEW_TRAINING=0,OPTIMIZER_STEPS=0,
        TRAIN253=253,SUPERVISED_CORNERS=s['supervised_corners'],CLEAN_GT20=sum(s['clean']['bands'][3:]),OCC_GT20=s['hard20'],OCC_20_40=s['hard20_40'],OCC_GT40=s['hard40'],FRAMES_ANY_GT20=s['frames_any_hard'],
        STRICT_TOTAL=s['strict_corners'],STRICT_FRAMES=s['strict_frames'],STRICT_GT20=s['strict_hard20'],STRICT_20_40=s['strict_hard20_40'],STRICT_GT40=s['strict_hard40'],STRICT_HARD_FRAMES=s['strict_hard_frames'],
        MATCHED_CORNERS=df['n'],DEV_GT20=sum(df['bands'][3:]),DEV_20_40=df['bands'][3],DEV_GT40=df['bands'][4],TRAIN_DEV_HARD_RATE_RATIO=d['comparisons']['hard_rate_ratio'],
        EASY_TO_HARD20=s['easy_to_hard20'],EASY_TO_HARD40=s['easy_to_hard40'],
        R20_RECOVERY10=fc['P1']['20']['PCK10'],R30_RECOVERY10=fc['P1']['30']['PCK10'],R40_RECOVERY10=fc['P1']['40']['PCK10'],
        NATURAL_20_40_RECOVERY10=nf['20_40']['PCK10'],NATURAL_GT40_RECOVERY10=nf['gt40']['PCK10'],TRAIN_HARD_TOP5_PRESENT=h['TRAIN_STRICT_HARD']['top5_present10'],DEV_HARD_TOP5_PRESENT=h['DEV_MATCHED_HARD']['top5_present10'],TRANSFER_GAP_PP=h['gap_pp'],
        STRICT_FRACTION_EASY=cons['strict_fraction_easy'],STRICT_FRACTION_HARD=cons['strict_fraction_hard'],EXCLUDED11_REASONS=cons['excluded_reasons'],HIGHER_CONF_HARD=pr['higher_confidence_manual_source'],UNKNOWN_PROVENANCE_HARD=pr['unknown_provenance'],
        PRIMARY_BOTTLENECK=decision['primary'],SECONDARY_BOTTLENECK=decision['secondary'],NEXT_ONE_EXPERIMENT=decision['next_one_experiment'])
    C.save(C.DOC/'FINAL_OUTPUT.json',result)
    md=['# TARGET DATA / TRAIN FIT — 원인 분해 결과','',
        '**주 병목: HARD_SUPERVISION_SHORTAGE. 보조: TRANSFER_MORPHOLOGY_GAP_SIGNAL.** 현재 TRAIN에 큰 실사 입력 오류가 드물고, 학습한 TRAIN에서는 복구하지만 DEV로 같은 후보 생성이 전이되지 않는다. target/reference 신뢰도 차이와 한 recording의 memorization 가능성은 분리되지 않았다. 모델 용량이 충분하다거나, 데이터만 늘리면 해결된다는 결론은 아니다.','',
        '신규 학습 0 / optimizer step 0 / checkpoint 갱신 0 / GT·pseudo target 변경 0. 기존 모든 결과와 최종 모델 유지.','',
        '## 핵심 수치','',
        '|항목|결과|','|---|---|',f'|TRAIN 감독 코너|{s["supervised_corners"]} / 253장|',
        f'|CLEAN 입력 >20px|{sum(s["clean"]["bands"][3:])} (전부 ≤5px)|',f'|OCC 입력 >20px|{s["hard20"]} / 2024 = {pct(d["TRAIN_ALL"]["fraction_gt20"])}|',
        f'|OCC 20–40 / >40|{s["hard20_40"]} / {s["hard40"]}|',f'|hard 코너 보유 프레임|{s["frames_any_hard"]}; 2개 이상 {s["frames_2plus_hard"]}, 4개 이상 {s["frames_4plus_hard"]}|',
        f'|반복 노출|hard {ex["hard20"]} / 전체 {ex["total"]}; 고유 hard는 {ex["unique_hard_corners"]}개|',
        f'|STRICT|{s["strict_frames"]}장 전체, hard {s["strict_hard20"]}개 유지|',f'|DEV matched R0 >20|{sum(df["bands"][3:])}/{df["n"]} = {pct(df["fraction_gt20"])}|',
        f'|TRAIN/DEV hard 비율|{d["comparisons"]["hard_rate_ratio"]:.4f} (약 1/12)|',f'|easy→hard20 / hard40|{s["easy_to_hard20"]} / {s["easy_to_hard40"]}|',
        f'|자연 hard TRAIN FULL 복구|{nf["20_40"]["correct10"]}/{nf["20_40"]["n"]} = {pct(nf["20_40"]["PCK10"])}|',
        f'|top5 후보 TRAIN / DEV|{pct(h["TRAIN_STRICT_HARD"]["top5_present10"])} / {pct(h["DEV_MATCHED_HARD"]["top5_present10"])}|','',
        'TRAIN error는 frozen pseudo와의 거리, DEV error는 기존 reference와의 거리다. 독립 물리 GT가 같은 두 집단의 성능 비교로 해석하지 않는다. DEV 주 모집단93장 중 matched85장/659코너만 분포 비교하며, 기존 official 전체713코너의 실패 penalty는 변경하지 않는다.','',
        '## Controlled basin: 원영상 좌표 오류를 복구하는가','',
        f'TRAIN-only로 고른 {len(protocol["selected"])}개 코너, 반경당 네 방향 총512회. 전체 TRAIN census와 구분. P1/P2는 같은 CLEAN bbox·다른 점·선택점 교란을 유지하고 RGB만 바꿨다. P3 자연 OCC는 기존 OCC R0의 다른 점과 bbox도 다를 수 있어 P2−P3를 순수 point morphology 효과로 단정하지 않는다.','',
        '|모델|RGB|r20 ≤10|r30 ≤10|r40 ≤10|자연20–40 ≤10|','|---|---|---:|---:|---:|---:|']
    for a in C.ARMS:
        for mode in ('P1','P2'):md.append('|'+ '|'.join([a,mode]+[pct(p['curves'][a][mode][str(r)]['PCK10']) for r in (20,30,40)]+[pct(n[a]['STRICT_HALF_THRESHOLD']['20_40']['PCK10'])])+'|')
    md+=['','>40 자연 hard TRAIN은 n=0이므로 성공률을 0%로 쓰지 않는다. 통제 40px 진단은 자연 >40px 사례가 아니다. 높은 TRAIN 복구는 이 recording/pseudo 기준의 결과이며 generalization 또는 physical correctness 보장이 아니다.','',
        '## Morphology / 시간적 다양성','',
        f'한 recording에서 hard 프레임 {s["frames_any_hard"]}개, 연속 display-index run {tmp["hard"]["count"]}개, 최장 {tmp["hard"]["longest"]}개. 이 run 수를 독립 scene 수라고 부르지 않는다.',
        '',f'TRAIN hard는 두 코너 동시 오류가 {m["TRAIN"]["hard_count_histogram"].get("2",0)}/{m["TRAIN"]["hard_frames"]}장. DEV hard는 4개 이상 동시 오류가 {sum(int(v) for k,v in m["DEV"]["hard_count_histogram"].items() if int(k)>=4)}/{m["DEV"]["hard_frames"]}장이다. TRAIN도 coherent 방향 오류가 있으므로 “TRAIN은 random single-corner뿐”이라고 하지 않는다. 코너쌍·평균 이동·이동 제거 잔차·similarity 잔차는 별도 JSON에 저장했다.',
        '', '## Target consistency / excluded11 / reference 한계','',
        'STRICT 0.025에서 accepted253 전부 유지되어 이번 기준으로 hard target만 consistency가 약하다는 증거는 없다. 제외11장의 기록 reason은 모두 raw_confidence다. 이는 raw score/valid-keypoint 공통 초기 gate의 이름으로, 실제 값은 비공개 EXCLUDED11 표에서 따로 확인했다. 제외된 점의 정답 정확도나 final target은 추정하지 않았다.',
        '',f'PRIMARY hard {pr["primary_hard"]}개 중 출처 unknown {pr["unknown_provenance"]}, higher-confidence manual-source {pr["higher_confidence_manual_source"]}. 비교할 독립 확인된 hard subset이 없으므로 unknown에 실패가 집중됐다는 사실만으로 GT가 주 원인이라고 할 수 없다. GREEN manual-only는 기존 별도 population 유지. 물리/가상 identity와 가시성 독립 검토는 미완료.',
        '', '## 19개 필수 질문에 대한 답','']
    answers=[f'감독 {s["supervised_corners"]}코너.',f'CLEAN >20: {sum(s["clean"]["bands"][3:])}.',f'OCC >20: {s["hard20"]}.',f'20–40 / >40: {s["hard20_40"]}/{s["hard40"]}.',f'hard 프레임 {s["frames_any_hard"]}.',f'300step hard 반복 노출 {ex["hard20"]}, 고유 {ex["unique_hard_corners"]}.',f'STRICT hard {s["strict_hard20"]}, {s["strict_hard_frames"]}장.',f'TRAIN {pct(d["TRAIN_ALL"]["fraction_gt20"])} vs DEV {pct(df["fraction_gt20"])}.',f'easy→hard20/40={s["easy_to_hard20"]}/{s["easy_to_hard40"]}.',f'FULL controlled20/30/40='+', '.join(pct(fc['P1'][str(r)]['PCK10']) for r in (20,30,40))+'.',f'OCC RGB controlled20/30/40='+', '.join(pct(fc['P2'][str(r)]['PCK10']) for r in (20,30,40))+'.',f'FULL natural20–40={pct(nf["20_40"]["PCK10"])}; >40 n=0.',f'TRAIN/DEV top5={pct(h["TRAIN_STRICT_HARD"]["top5_present10"])} / {pct(h["DEV_MATCHED_HARD"]["top5_present10"])}.', 'TRAIN은 주로 두 코너의 동반 오류, DEV는 더 많은 코너 동시 오류와 큰 tail; 단순 방향 coherence는 양쪽 모두 존재.', '현재 STRICT는 모두 통과. 제외11은 초기 gate; 정답 난도에 대한 인과 근거 없음.', f'unknown reference source={pr["unknown_provenance"]}/{pr["primary_hard"]}; 오류라고 확정하지 않음.',decision['primary']+'.',decision['secondary']+'.','같은 frozen target·TRAIN 안에서 coupled hard-input dose를 바꾸는 통제 실험 한 개를 설계만 남김. 실행하지 않음.']
    md += [f'{i}. {a}' for i,a in enumerate(answers,1)]
    md+=['','## 분포·전이·후보 그래프','']
    for name in plots:md+=['',f'![{name}](figures/{name}.png)']
    md+=['','## 사례 갤러리','', '[이미지 연속 보기](GALLERY.html). 성공/실패 예시 수가 요청보다 적으면 있는 사례만 표시한다. natural hard failure는 1개뿐이다. 블라인드 검토 화면과 별도이며 사람 응답을 생성하지 않았다.']
    for g in gallery:md+=['',f'### {g["group"]}','',f'![{g["group"]}](figures/{g["name"]}.png)']
    md+=['','## 감사·실행 주의','',
        '초기 로더가 accepted-only253 manifest를 후보264로 잘못 가정한 것과 과거300장 role pool을 현재278장 집합과 직접 비교한 가정은 실제 저장 schema를 확인해 바로잡았다. 입력/프로토콜 변경 없음. GPU 진단 후 집계에서는 기존 evaluator의8-corner mask 및 null error 저장 형식을 처리하도록 분석 어댑터만 고쳤다. 완료된 TRAIN 집계는 exact 비교 재사용, 예측 재생성·추가 학습 없음. 실패 시도 메타데이터 보존.',
        '', '코드는 기존 로컬 연구 모듈·동결 체크포인트·비공개 데이터에 의존하므로 공개본만으로 standalone 재현 패키지는 아니다. 원본 prediction/좌표/검토매핑은 공개하지 않는다.']
    C.save(C.DOC/'RESULTS_KO.md','\n'.join(md)+'\n')
    htmlrows=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>TRAIN target diagnosis</title><style>body{font:18px system-ui;background:#11212b;color:white;margin:24px}img{max-width:100%}a{color:skyblue}</style><h1>TRAIN / DEV 원인 분해</h1><p>초록=기존 pseudo/reference (독립 GT 아님), 주황=입력, 하늘=FULL125 출력. 새 학습 없음.</p>']
    for g in gallery:htmlrows += [f'<h2>{html.escape(g["group"])}</h2><img loading="lazy" src="figures/{g["name"]}.png">']
    C.save(C.DOC/'GALLERY.html','\n'.join(htmlrows)+'</html>')
    sections={'TRAIN_HARDNESS_KO.md':'전체 감독2024, clean 전부≤5, OCC hard50/25장, >40 없음. 반복444를 고유50과 구분.','PSEUDO_CONSISTENCY_AUDIT.md':'0.025 기준253장 전부STRICT. accepted-only 선택 편향 존재. internal consistency는 정답성 보장이 아니다.','TRAIN_DEV_DISTRIBUTION_KO.md':f'TRAIN2.47% vs DEV29.74%, 비율{d["comparisons"]["hard_rate_ratio"]:.4f}. target provenance가 달라 p-value 원인 주장 금지.','CONTROLLED_PROBE_KO.md':'128 TRAIN 코너만 고정 선택. 각 반경512회. 전체 모델·P1/P2 결과는 CONTROLLED_PROBE_RESULTS.json. P1/P2 RGB 이외 동일. 자연 P3는 bbox/다른 코너가 다르다.'}
    for name,text in sections.items():C.save(C.DOC/name,'# '+name.removesuffix('.md')+'\n\n'+text+'\n\n[전체 결과 및 이미지](RESULTS_KO.md)\n')
    C.save(C.DOC/'NEXT_STAGE_PLAN.md','# 후속 실험 단 하나 — 설계만, 실행하지 않음\n\n질문: 같은 real TRAIN/pseudo target을 유지하면서 hard 입력의 노출량과 동반 코너 오류 구조를 바꾸면, 기존 natural TRAIN 적합을 보존하면서 recording-separated DEV로 복구가 전이되는가?\n\nFrozen target coupled-hard-dose FULL125 1개 vs 저장된 FULL125. 동일 PRIOR1/seed1/300step/source order+corruption/real253 order/손실/BN동결/crop1.25 유지. real batch8 중 사전 고정4개 슬롯만 점 입력을 변형하고 나머지4개 natural OCC 유지. 변형4개 중2개는 한 코너,2개는 수직 인접 코너쌍 (0,3),(1,2),(4,7),(5,6)에 같은 방향 이동. 목표 주변 원영상 반경20/40/60px·네 방향을 deterministic cycle로 배정. 입력 validity·bbox·RGB·center8·pseudo target 고정. TRUE GT/eval outcome으로 샘플/반경/쌍 선택 금지.\n\n이 실험은 hard-dose+구조를 묶은 하나의 개입이며 각 효과를 따로 증명하지 못한다. 기존 capacity는 simple perturbation에서 이미 높으므로 같은 단일점 jitter만 추가하는 것보다 미관측 큰 동반오류에 초점을 둔다. 단일 recording memorization/physical pseudo correctness는 여전히 별도 한계다.\n\nlast300만 기존 분모와 paired recovery/damage/source/green 안전지표로 비교. 최종 모델 자동 승격·실패 후 재학습·sweep 없음. 별도 실행 승인 필요. 이번 작업에서는 optimizer step0으로 종료.\n')
    print('REPORT_READY',len(plots),'plots',len(gallery),'cases',flush=True)

if __name__=='__main__':main()
