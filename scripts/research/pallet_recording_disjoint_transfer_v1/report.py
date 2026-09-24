"""Evidence-based interpretation and Korean report. No additional fits."""
import sys
from . import common as C

def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows]

def f(x,n=4):return '—' if x is None else f'{x:.{n}f}'

def decision():
    res=C.read(C.DOC/'RESULTS.json')['groups'];role=C.read(C.DOC/'H10_ROLE_PREVALENCE.json');anchor=C.read(C.DOC/'VERIFIED_ANCHOR_TRANSFER.json')
    deltas={}
    for g in ('ALL','CLEAN','MODERATE','SEVERE'):
        a,b=res[g]['S0'],res[g]['S1']
        deltas[g]=dict(PCK10_pp=100*(b['twoD']['PCK']['10']-a['twoD']['PCK']['10']),
            current_AUC=b['current']['ADDsym_AUC']-a['current']['ADDsym_AUC'],oracle_AUC=b['oracle']['ADDsym_AUC']-a['oracle']['ADDsym_AUC'],
            matched_P90_px=b['twoD']['matched_pooled_corner8_P90_px']-a['twoD']['matched_pooled_corner8_P90_px'],
            selection_loss=b['selection_loss']-a['selection_loss'])
    # Sign-based interpretation of this completed run, not a tuned acceptance threshold.
    assert all(deltas['SEVERE'][k]>0 for k in ('PCK10_pp','current_AUC','oracle_AUC'))
    assert deltas['MODERATE']['current_AUC']<0<deltas['MODERATE']['oracle_AUC']
    assert role['status']=='ROLE_MISMATCH_REPEATED'
    assert C.read(C.DOC/'RECORDING_DISJOINT_AUDIT.json')['status']=='RECORDING_DISJOINTNESS_VERIFIED'
    out=dict(primary='SEVERE_ONLY_TRANSFER_SIGNAL',secondary='CANDIDATE_GAIN_SELECTOR_LIMIT_IN_MODERATE',
        role_convention_status=role['status'],role_contract_warning=True,deltas=deltas,
        primary_bottleneck='Severity-dependent transfer: severe aggregate PCK/candidate/current improve, moderate current and clean preservation worsen. Severe matched P90 also worsens.',
        secondary_bottleneck='Moderate selector loses improved candidate quality; repeated training-H10 role mismatch limits identity/6D interpretation.',
        uniform_transfer_supported=False,hard_anchor_correct10_delta=anchor['correct10_delta']['HARD'],
        more_hard_labeling_justified='NOT_ESTABLISHED: teacher hard visible26/36 is better than S1 19/36; severe teacher8/14 and small biased anchor prevent a blanket conclusion.',
        selector_work_justified='YES as a moderate-specific signal, but after resolving the bounded role-contract warning; no selector changes in this run.',
        representation_change_justified='NOT_IDENTIFIED as the first necessary change by this frozen comparison.',
        next_one_experiment=dict(name='BOUNDED_ROLE_CONTRACT_NORMALIZATION_PILOT',design_only=True,
            reason='Directive CASE F: a second H10 frame meets the predeclared role-mismatch rule. Establish an explicit paper-facing role mapping before a new optimization experiment.',
            population='Existing fixed H10 only; current two flagged cases are diagnostics, not automatic relabel candidates. No expansion or new capture.',
            variable='One explicit deterministic conversion from documented annotation role convention to declared paper camera-facing contract; preserve rectangular physical C2 (180deg) equivalence separately.',
            controls='Frozen original annotations, predictions, checkpoints, split, PnP selector and metrics. Never choose per-model/per-point best correspondence to define GT.',
            evidence='Use already saved manual provenance and existing source-rule definitions; C4 best scores cannot establish semantic front. No new labeling or human review in this completed task.',
            stop='If existing provenance/semantic front cannot uniquely support a mapping, retain UNRESOLVED; no auto-GT edits and no transfer claim based on remapped scores.',
            output='Design for a versioned mapping proposal plus identity/C2 contract tests and unchanged-original comparison; not executed now.',
            training_steps=0),
        no_new_thresholds=True,not_independent_test=True,not_statistical_significance=True)
    C.save(C.DOC/'TRANSFER_DECISION.json',out)
    print('TRANSFER_DECISION',out['primary'],out['secondary'],flush=True)

def main():
    res=C.read(C.DOC/'RESULTS.json')['groups'];tr=C.read(C.DOC/'TRANSITIONS.json');role=C.read(C.DOC/'H10_ROLE_PREVALENCE.json')
    split=C.read(C.DOC/'RECORDING_DISJOINT_AUDIT.json');pair=C.read(C.DOC/'PAIR_INTEGRITY.json');prov=C.read(C.DOC/'CHECKPOINT_PROVENANCE.json')
    dec=C.read(C.DOC/'TRANSFER_DECISION.json');anchor=C.read(C.DOC/'VERIFIED_ANCHOR_TRANSFER.json');hist=C.read(C.DOC/'HISTORICAL_DIRECTION.json')['groups']
    source=C.read(C.DOC/'SOURCE_PRESERVATION.json')['arms'];cases=C.read(C.DOC/'CASE_MANIFEST.json')['cases'];bindings=C.read(C.DOC/'INPUT_BINDINGS.json')
    lines=['# Recording-disjoint CLEAN→NATURAL-HARD transfer','',
        '## 1. 한 줄 결론','',
        f"TRANSFER_DECISION: **{dec['primary']}**. ROLE_CONVENTION_STATUS: **{role['status']}**.",'',
        '**S1의 추가 랜덤 가림은 다른 recording의 심한 가림에서 평균적인 2D·후보·실제 자세 성능을 개선했지만, 중간 가림과 Clean까지 좋아지는 방법은 아니었다.** 심한 가림의 matched P90도 악화했으므로 모든 꼬리 오류가 줄었다고 하지 않는다.','',
        '기존 frozen 모델 재사용: 신규 학습0 / 재현 fit0 / 신규 신경망 추론0. 128장×3모델의 production D9 자세와 두 W/D 후보를 GT 없이 재계산·잠근 뒤 채점했다.','',
        '## 2. 왜 이 실험이 필요한가','',
        '기존 ALL300은 같은 training recording도 포함한 반복 DEV였다. 이번 비교는 기존 고정 HELDOUT128을 그대로 사용한다. 학습에 쓰지 않은 recording으로의 전이를 묻지만, 이미 열람된 재사용 DEV이며 독립 final TEST가 아니다. 원본319장·기존 결과·어노테이션·checkpoint를 고치지 않았다.','',
        '## 3. causal pair integrity','',
        f"상태 `{pair['status']}`. PLASTIC S0/S1은 original R0 초기값, Clean10 이미지, 교사 좌표·mask, synthetic512, epoch/slot/출처 순서, 초기 trainable inventory, AdamW/학습률/seed/업데이트를 공유한다. 각각5epoch·320update·batch/nbs16·640·seed42·lr1e-4/lrf0.1/cosine·last checkpoint다.",'',
        f"학습 실제 trace와 cache **{pair['actual_trace_cache_verified']}개**를 전수 재검증했다. S0/S1 RGB를 기존 코드로 재구성해 실제 입력 SHA와 대조했다. 변경은 실사2560회 중 **{pair['changed_real_inputs']}회({100*pair['changed_real_inputs']/2560:.2f}%)**에만 있었고 합성2560회는 같았다. 좌표·visibility mask·bbox는 같았다.",'',
        '중요한 구현 범위: S1은 감독점 coverage 조건과 기존 S2 placement와의 짝짓기 가능 조건을 통과한 위치에만 랜덤 사각형 fill을 넣는다. 따라서 **무조건적인 랜덤 erasing 전체의 효과가 아니라 이 고정 조건부 정책의 효과**다. S2를 새로 실행하거나 평가하지 않았다. 실제 저장 args도 name/save_dir 이외 동일함을 별도 감사했다.','',
        '![실제 동일 cache 입력의 유일한 차이](figures/02_s0_s1_input_contract.png)','',
        '[pair 감사](PAIR_INTEGRITY.json) · [checkpoint 해시](CHECKPOINT_PROVENANCE.json) · [실제 runtime args 감사](PAIR_RUNTIME_ARGS.json)','',
        '## 4. recording disjointness','']
    lines+=table(['역할','고유 이미지','recording'],[['학습',split['train_frames'],', '.join(split['train_recordings'])],['평가',split['heldout_frames'],', '.join(split['heldout_recordings'])]])
    lines+=['',f"recording 교집합 {len(split['recording_intersection'])}, image SHA 교집합 {len(split['image_sha_intersection'])}. 기존 grayscale64×48 MAD≤2/255 검사 재확인: 최소MAD {split['near_duplicate']['minimum']:.4f}, 겹침0. 근접중복 검사는 장면 독립성의 완전한 증명이 아니다.",'',
        '![학습10과 평가128 recording](figures/01_recording_disjoint_split.png)','',
        '## 5. H10 role prevalence','',
        '기존 H10, common/direct support 총36점만 검사했다. 사전 규칙은 n≥3 AND 같은ID 평균>20px AND 전체 C4 최소평균≤10px. 점별 free matching·threshold sweep·GT수정은 없다.','']
    rows=[]
    for r in role['rows']:
        if r['id'] not in role['strong_frames']:continue
        for a,m in r['models'].items():rows.append([r['id'],r['common_support_count'],a,f(m['same_ID_mean_px'],3),m['best_C4']['c4'],f(m['best_C4']['mean_px'],3),m['strong']])
    lines+=table(['프레임','공통점','모델','같은ID 평균px','전체 C4 최소','최소 평균px','strong'],rows)
    lines+=['',f"강한 신호 **{role['strong_count']}/10장**: {', '.join(role['strong_frames'])}. 020954는 teacher/T1에 신호가 있고 T2는 같은ID로 맞춘다. 그러므로 모든 모델·GT가 동일 원인이라고 단정할 수 없다. 이 표는 학습 진단 표본이지 전체 평가셋의 오류 빈도가 아니다.",'',
        '**180도 규약:** 직사각 팔레트의 물리적 C2(yaw180) 동치와 90/270 역할 좌표 변환(W/D swap)은 별개다. 앞서 A/B, C/D 각각180도 동치 쌍이라는 사용자 확인 내용을 유지한다. C4 사후 최소를 정답 수정이나 배포 성능으로 사용하지 않았다. 같은ID/6D 결과에는 convention warning을 남긴다.','',
        '![H10 동일ID와 C4 진단 차이](figures/01_h10_role_prevalence.png)','',
        '[H10 전수 표·metadata](H10_ROLE_PREVALENCE_KO.md)','']
    for n,g in [(6,'CLEAN'),(7,'MODERATE'),(8,'SEVERE')]:
        lines += [f'## {n}. Natural {g}','']
        rows=[]
        for a in C.ARMS:
            r=res[g][a];d=r['twoD'];p=r['current']
            rows.append([a,f"{p['frames']} / {d['corners']}",'/'.join(f(100*d['PCK'][k],2) for k in ('5','10','20')),f"{d['correct']['10']}/{d['corners']}",f(d['matched_pooled_corner8_median_px'],2)+' / '+f(d['matched_pooled_corner8_P90_px'],2),f(100*d['gross20'],2),f"{d['detected']}/{d['matched']}",f(p['ADDsym_AUC']),f(r['oracle']['ADDsym_AUC']),f(r['selection_loss'])])
        lines+=table(['모델','장 / 점','PCK5/10/20 %','PCK10 맞은점','matched med/P90 px','>20 %','검출/매칭','CURRENT AUC','ORACLE AUC*','선택손실'],rows)
        rows=[]
        for a in C.ARMS:
            p=res[g][a]['current'];rows.append([a,*[f(p[k]['median'],3)+' / '+f(p[k]['P90'],3) for k in ('rotation_deg','yaw_deg','translation_cm','IoU3D')],f"{p['axis_correct_count']}/{p['available']}",f"{p['available']}/{p['frames']}"])
        lines+=['']+table(['모델','R med/P90 °','Yaw med/P90 °','t med/P90 cm','IoU3D med/P90','W/D parity','pose coverage'],rows)
        t=tr[g]['canonical_GT_identity_aligned'];d=dec['deltas'][g]
        lines+=['',f"S1−S0: PCK10 **{d['PCK10_pp']:+.2f}%p**, CURRENT AUC **{d['current_AUC']:+.5f}**, ORACLE AUC **{d['oracle_AUC']:+.5f}**. 같은 정답ID 기준 ≤10→>10 손실 {t['lost_correct10']}점, >10→≤10 획득 {t['gained_correct10']}점, >20→≤10 복구 {t['recovery20_to10']}점, <5→>10 손상 {t['damage5_to10']}점.",'']
        if g=='CLEAN':
            lines+=['Clean에서 작은 정상 성능 손상을 지불했다. 추가 가림이 Clean을 보존했다고 단정하지 않는다. 합성source256 PCK10은 아래와 같이 개선됐지만 PCK5·PCK20은 조금 감소했다. 분모2028점은 source 유효점이며 실사와 합치지 않는다.','']
            lines+=table(['합성256','PCK5','PCK10','PCK20','median/P90 px'],[[a,*[f(100*source[a]['PCK'][k]['fraction'],2) for k in ('5','10','20')],f(source[a]['median'],2)+' / '+f(source[a]['P90'],2)] for a in ('S0','S1')])
            lines+=['','![Clean/source 보존](figures/07_clean_preservation.png)','']
        elif g=='MODERATE':
            lines+=['후보 oracle은 좋아졌지만 실제 자세는 악화했다. 선택손실이 0.04786→0.10490으로 증가하고 W/D parity는18/21→16/21이다. PCK10은 동률이며 P90은 개선됐다. **현재 선택기의 이득 전달 실패 신호**이지, oracle 성능을 달성한 것이 아니다.','']
        else:
            lines+=['2D PCK·candidate·current의 집계 방향은 모두 개선됐다. 다만 matched P90은48.45→54.31px로 악화했다. matched 집합도 변하므로 같은 점들 전체가 나빠졌다는 뜻은 아니며, [전체 벌점 포함 P90 및 매칭 분모](RESULTS.json)도 함께 보존했다. 통계적 유의성이나 모든 hard 이미지 개선을 주장하지 않는다.','']
    lines+=['![2D](figures/03_natural_pck_by_severity.png)','', '![실제6D](figures/04_current_pose_by_severity.png)','',
        '![후보품질 — 비배포용 oracle](figures/05_oracle_candidate_by_severity.png)','',
        '## 9. verified visible anchor','',
        '실제2점 QA를 완료한 FINAL_V2, HELDOUT 안의16장66개 DIRECT_VISIBLE만 사용했다. 미확인·PnP보완점을 정답으로 끼워 넣지 않았다. P0..P7 고정ID, symmetry-min 없음. PnP 보조 first pass와 선택 편향이 있는 작은 보조 표본으로 전체 운영분포나 독립6D GT가 아니다.','']
    rows=[]
    for g in ('ALL','CLEAN','MODERATE','SEVERE','HARD'):
        for a in (*C.ARMS,'TEACHER'):
            r=anchor['groups'][g][a];rows.append([g,a,*[f"{r['PCK'][k]['correct']}/{r['n']}" for k in ('5','10','20')],f(r['median_px'],3),f(r['p90_px'],3),r['gt20']])
    lines+=table(['집단','모델','PCK5','PCK10','PCK20','median px','P90 px','>20점'],rows)
    lines+=['','Hard visible36점에서 S0 20→S1 19점(10px 이내), teacher26점이다. S1은 median/P90/>20은 개선됐으나 PCK10은1점 감소했다. 작은 보조표에서 전반적 우월을 선언하지 않는다. Severe14점은8→8로 동률이다.','',
        '![최종 visible 보조 평가](figures/08_verified_visible_transfer.png)','',
        '## 10. historical ALL300와 비교','',
        '기존 RESULTS.json에서 자동으로 읽었다. 전체300에는 목재가 포함되므로 plastic184의 방향도 병기한다. recording-heldout128은 그 일부이자 반복 DEV이다. 아래는 **방향 비교**이지 새 데이터에서의 독립 재현이나 동일 분모 성능 차이가 아니다.','']
    rows=[]
    for g,h in hist.items():
        for label,n,key in [('historical',h['historical_frames'],'historical_delta'),('historical plastic',h['historical_plastic_frames'],'historical_plastic_delta'),('heldout',h['heldout_frames'],'recording_disjoint_delta')]:
            d=h[key];rows.append([g,label,n,f(d['PCK10_delta_pp'],3),f(d['ADDsym_delta'],5),h['status'] if label=='heldout' else '문맥용'])
    lines+=table(['난도','집단','장','S1−S0 PCK10 %p','S1−S0 CURRENT AUC','방향판정'],rows)
    lines+=['','중간 가림은 기존 PCK 개선이 heldout에서 사라졌고 실제 자세 악화 방향은 남았다. 심한 가림의 PCK·CURRENT 개선 방향은 다른 recording에서도 유지됐다. Clean은 기존의 개선 방향이 하락으로 바뀌었다. “방향 재현”에는 악화 방향의 재현도 포함하며 성공이라는 뜻이 아니다.','',
        '![과거와 분리평가 방향](figures/09_historical_vs_recording_disjoint.png)','',
        '## 11. 병목 분해','',
        'L1: 심한 가림 PCK 개선은 남지만 중간 PCK10은 동률, Clean은 손상. L2: 중간·심함의 oracle 후보 품질은 개선. L3: 심함에서는 이득이 전달되지만 중간에서는 선택손실 증가가 후보 이득을 상쇄한다. 따라서 모든 실패를 학생 표현력이나 selector 하나로 환원할 수 없다.','',
        '![선택손실](figures/06_selection_loss.png)','',
        '주2D는 기존 HELDOUT의 허용 대칭 whole-object 대응 계약을 유지했다. transition은 canonical GT ID 정렬과 native fixed-ID 무대칭을 둘 다 계산했고 **이번128장에서는 두 결과가 같다**. H10 C4진단은 이 평가에 적용하지 않았다. CURRENT PnP는 검출 bbox의 GT 매칭 gate를 사용하지 않는다. axis parity는 W/D extents 일치율이지 완전한 회전 정답률이 아니다. ORACLE은 GT로 두 후보 중 ADDnorm 최소를 고른 사후 진단이고 배포 불가다.','',
        '### Recording별 같은 S1−S0 대조','']
    lines+=table(['recording','장','PCK10 %p','CURRENT Δ','ORACLE Δ'],[[g,res[g]['S0']['current']['frames'],f(100*(res[g]['S1']['twoD']['PCK']['10']-res[g]['S0']['twoD']['PCK']['10']),2),f(res[g]['S1']['current']['ADDsym_AUC']-res[g]['S0']['current']['ADDsym_AUC']),f(res[g]['S1']['oracle']['ADDsym_AUC']-res[g]['S0']['oracle']['ADDsym_AUC'])] for g in split['heldout_recordings']])
    lines+=['','### 전체128장 집계 (R0 보조 / S1−S0 주대조)','']
    lines+=table(['모델','PCK5/10/20 %','PCK10 맞은점','matched med/P90 px','CURRENT AUC','ORACLE AUC','W/D parity','검출/매칭'],
        [[a,'/'.join(f(100*res['ALL'][a]['twoD']['PCK'][k],2) for k in ('5','10','20')),
          str(res['ALL'][a]['twoD']['correct']['10'])+'/'+str(res['ALL'][a]['twoD']['corners']),
          f(res['ALL'][a]['twoD']['matched_pooled_corner8_median_px'],2)+' / '+f(res['ALL'][a]['twoD']['matched_pooled_corner8_P90_px'],2),
          f(res['ALL'][a]['current']['ADDsym_AUC']),f(res['ALL'][a]['oracle']['ADDsym_AUC']),
          str(res['ALL'][a]['current']['axis_correct_count'])+'/128',
          str(res['ALL'][a]['twoD']['detected'])+'/'+str(res['ALL'][a]['twoD']['matched'])] for a in C.ARMS])
    lines+=['','### 사례: CURRENT normalized ADD로 사후 선택','',
        '중간 개선4·악화4, 심함 개선4·악화4, fixed SHA random6. 원본 전체가 아닌 팔레트 ROI만 표시했고 얼굴 감지 영역을 모자이크했다. 같은 이미지/ROI를 RGB·R0·S0·S1에 사용한다. **노랑 raw2D / 빨강 실제 PnP / 하늘 점선 대안 PnP / 초록 legacy GT**. 개선·악화 극단 사례는 모집단 평균의 대체 근거가 아니다.','']
    for r in cases:
        lines += [f"#### {r['group']} · {r['label']} · {r['id']}",'',f"S1−S0 CURRENT ADDnorm: {f(r['ADD_delta'],5)} (음수=개선).",'',f"![{r['group']} {r['label']}]({r['figure']})",'']
    lines+=['## 12. 객관적 판정','',f"PRIMARY: `{dec['primary']}`. SECONDARY: `{dec['secondary']}`.",'',
        '균일한 전이 성공은 아니다. Severe 집계 개선을 보존하되 Clean 손상·중간 selector 병목·visible PCK10 감소·H10 역할 불일치 반복을 함께 보고한다. H10 신호 때문에 전이 평가를 중단하지 않았고, 반대로 좋은 severe 평균으로 contract 문제를 덮지도 않았다.','',
        '- 추가 hard annotation의 필요성: 이번 결과만으로 확정하지 못한다. teacher hard26/36은 S1 19/36보다 낫지만 Severe teacher8/14는 한계가 있다.','- selector 후속 검토: 중간 난도에서 근거가 있다. 다만 아래 contract 분기 이후이며 이번에는 바꾸지 않았다.','- 새 representation: 현 결과로 최우선 변경이라고 특정할 수 없다.','',
        '## 13. 다음 딱 한 실험','',
        f"**{dec['next_one_experiment']['name']} — 설계만, 실행0.**",'',
        '지시문 CASE F에 따라 고정 H10의 기존 출처와 명시적인 paper camera-facing 규약 사이의 결정적 mapping을 좁게 검증한다. 물리적180도 C2 동치와 역할90/270 변환을 분리한 versioned mapping 제안·계약 테스트를 만들도록 설계한다. 현재 두 flagged 사례의 C4 minimum을 정답으로 채택하지 않는다.','',
        '원본 주석·frozen prediction·모델·split·PnP를 대조군으로 보존한다. 기존 provenance만으로 역할을 유일하게 뒷받침하지 못하면 UNRESOLVED로 멈춘다. 새 촬영·human review·GT 자동수정·추가 학습은 이번에 실행하지 않는다. 다른 실험 후보를 동시에 추가하지 않는다.','',
        '![판정과 단일 다음 분기](figures/10_transfer_routing.png)','',
        '## 14. 한계','',
        'single seed42, plastic-only pilot,7개 recording의 재사용 DEV, 이미 본 HELDOUT, 조건부 인공가림 정책, geometry-derived6D reference, 작은 visible anchor·PnP보조 first-pass·P6 부재·선택편향. H10 역할 신호는 학습 진단의2/10이지 evaluation 전체 GT 오류율이 아니다. CI/통계적 유의성·목재 일반화·독립 final TEST를 주장하지 않는다. synthetic256은 고정 기존 source이고 source6D는 추가 계산하지 않았다.','',
        '## 15. 재현','',f"HEAD_BEFORE: `{bindings['head_before']}`. branch: `{bindings['branch']}`. 새 commit은 이 보고서가 포함된 Git commit이다. `git log -1 --format=%H -- _docs/experiments/{C.NAME}/REPORT_KO.md`로 확인한다. push 검증 출력은 로컬 `outputs/{C.NAME}/COMMIT_PUSH_RESULT.txt`에 보존한다.",'']
    lines+=table(['모델','SHA256'],[[a,b['sha256']] for a,b in prov['checkpoints'].items()]+[['R0',prov['R0']['sha256']]])
    lines+=['','[모든 입력·코드·cache SHA](INPUT_BINDINGS.json) · [예측잠금](PREDICTIONS_LOCK.json) · [PnP잠금](POSE_PREDICTIONS_LOCK.json) · [결과JSON](RESULTS.json) · [전이점수](TRANSITIONS.json) · [자동 검사](AUDIT.json)','',
        '학습 checkpoint·원본 이미지·개별 좌표·private per-frame cache는 Git에 올리지 않는다. 보고서는 작은 ROI 그림과 집계값만 사용한다. 로컬 재현에는 해시로 묶인 기존 private 데이터/cache가 필요하다.','',
        '```bash','python -m scripts.research.pallet_recording_disjoint_transfer_v1.preflight',
        'python -m scripts.research.pallet_recording_disjoint_transfer_v1.role_scan',
        'python -m scripts.research.pallet_recording_disjoint_transfer_v1.infer',
        'python -m scripts.research.pallet_recording_disjoint_transfer_v1.evaluate',
        'python -m scripts.research.pallet_recording_disjoint_transfer_v1.report decision',
        'python -m scripts.research.pallet_recording_disjoint_transfer_v1.render',
        'python -m scripts.research.pallet_recording_disjoint_transfer_v1.report',
        'python -m scripts.research.pallet_recording_disjoint_transfer_v1.audit','```','',
        'preflight/infer/evaluate는 기존 완료 잠금을 덮어쓰지 않는다. 이미 완료한 작업의 확인은 audit만 실행한다.','']
    C.save(C.DOC/'REPORT_KO.md','\n'.join(lines))
    quick=['# Recording-disjoint transfer 요약','',dec['primary'],'',
        '기존 모델 재사용, 새 학습0. Clean29/Moderate21/Severe78, 다른recording128장 평가.','']
    quick+=table(['난도','S0 PCK10 %','S1 PCK10 %','S0 CURRENT','S1 CURRENT','S0 ORACLE','S1 ORACLE'],[[g,*[f(100*res[g][a]['twoD']['PCK']['10'],2) for a in ('S0','S1')],*[f(res[g][a][key]['ADDsym_AUC']) for key in ('current','oracle') for a in ('S0','S1')]] for g in ('CLEAN','MODERATE','SEVERE')])
    quick+=['','심함은 개선, 중간 실제자세·Clean은 악화. hard visible PCK10 20/36→19/36. H10역할 신호2/10; GT수정 없음.','',
        '![실제 자세](figures/04_current_pose_by_severity.png)','',
        f"![심함 개선 사례]({next(r['figure'] for r in cases if r['group']=='SEVERE' and r['label']=='improved')})",'',
        f"![중간 악화 사례]({next(r['figure'] for r in cases if r['group']=='MODERATE' and r['label']=='harmed')})",'',
        '[22개 사례와 전체 과정·수치 보고서](REPORT_KO.md)','']
    C.save(C.DOC/'SUMMARY_KO.md','\n'.join(quick))
    print('REPORT_COMPLETE',C.DOC/'REPORT_KO.md',flush=True)

if __name__=='__main__':
    decision() if len(sys.argv)>1 and sys.argv[1]=='decision' else main()
