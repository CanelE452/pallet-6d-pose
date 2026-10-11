"""Insert sealed aggregate results into the Korean report; no inference or fitting."""
from . import common as C


def replace_section(text, name, content):
    begin=f'<!-- FINAL_{name}_RESULTS -->';end=f'<!-- END_FINAL_{name}_RESULTS -->'
    left=text.index(begin)+len(begin);right=text.index(end)
    return text[:left]+'\n'+content.strip()+'\n'+text[right:]


def run():
    path=C.DOC/'RESULTS_KO.md';text=path.read_text()
    cc=C.read(C.DOC/'C_RESULTS.json');fits=C.read(C.DOC/'C_FITS.json')
    syn=C.read(C.DOC/'C_SYNTH_TEST_RESULTS.json');b1=C.read(C.DOC/'B1_RESULTS.json')
    metrics=C.read(C.DOC/'METRICS.json');paired=C.read(C.DOC/'PAIRED.json')
    risk=C.read(C.DOC/'E_PROBABILITY_RISK.json');oldrisk=C.read(C.DOC/'E_RISK_COVERAGE.json')
    lookup={r['method']:r for r in metrics['summaries'] if r['backbone']=='YOLO' and r['population']=='REAL173'}
    lines=['[확인] B1은 기존 N3 seed1의 입력 치수만 correct/mean/normalized-zero/TRAIN-shuffle로 바꾸고, 모든 조건에서 PnP 치수는 올바른 등록값으로 고정했다. 새로운 fitting이나 backbone 추론은0이며 OpenCV solve/refine 실제3,114회다. normalized-zero는 TRAIN-mean과 동일하다고 가정하지 않았다([B1_RESULTS.json](B1_RESULTS.json)).','',
      '| 기존 N3 입력 치수, REAL173 | 평균 코너 이동 px | 확률 L1 평균 | 후보 argmax 변경/1384 | W/D 선택 변경 | 5cm·5° % |',
      '|---|---:|---:|---:|---:|---:|']
    for name,r in b1['summaries'].items():
        lines.append(f"| {name} | {r['coordinate_delta_px']['mean']:.6f} | {r['probability_L1_mean']:.5f} | {r['argmax_changed']} | {r['WD_choice_switches']} | {r['metrics']['strict_success']*100:.3f} |")
    lines+=['','[확인] 기존 N3는 치수에 **반응한다**. 그러나 네 조건의 직접 축 오류는 모두43/170이고 W/D 선택 변경0이다. correct 성공77/173와 mean75/173,zero76/173,shuffle74/173의 차이는 작으며 치수의 안정적인 6D 이득으로 확정하지 않는다. 상태는 `SENSITIVE_BUT_BENEFIT_UNRESOLVED`다.']
    text=replace_section(text,'B1','\n'.join(lines))

    order=['OFFICIAL','TRAIN_PRIOR','D_LOGISTIC','G_LOGISTIC','V_LOGISTIC','V_MLP','G_MLP','G_V_MLP','G_V_D_MLP','V_SHUFFLED_LOGISTIC']
    lines=['[확인] **8fit 모두 완료**했다. C용 TRAIN4096/VAL1024/TEST1024의 renderer/scenario group 분리를 유지했고 실사 GT로 fitting·early stopping·threshold를 고르지 않았다. V는 동결 P3/P4의 예측 ROI2×2 공간 pooling(768차원)이다. 기존 P3/P4 cache를 predictor에 썼고, 누락된 현재 YOLO confidence 계약 확인을 위해 기존6144 합성 RGB를 실제 1회씩 동결 재추론했다. points/box/P3/P4 parity 최대 차이0, 동결 파라미터 no-grad를 확인했다. 새 합성 이미지 생성은0이다([C_EXTRACTION_RECEIPT.json](C_EXTRACTION_RECEIPT.json), [C_FITS.json](C_FITS.json)).','',
       '[확인] 모든 작은 분류기·shuffle·prior·공식 선택을 같은 REAL173/seed1·2·3으로 보고한다. 성공·기존W/D는173분모, 직접 축 분류170분모다. 각 행의 연속 통계는 seed 평균을 영상 내에 묶어 계산했다. 전체52집계의 평균·분산·SD·중앙값·P90·P95·최대값·95% CI는 [METRICS.json](METRICS.json)에 있다.','',
       '| Selector | 직접 축 오류 % | Balanced accuracy % | 기존 W/D % | 5cm·5° % | T 평균±SD cm | T P90 cm | R 평균±SD ° | R P90 ° | ADDsym AUC |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in order:
        a=cc['summaries'][name];r=lookup[name]['metrics'];t=r['translation_cm'];rot=r['rotation_deg']
        lines.append(f"| {name} | {100*a['REAL173']['axis_error_rate']:.3f} | {100*a['long_short_classification']['balanced_accuracy']:.3f} | {100*r['wd_confusion']['mean']:.3f} | {100*r['pose_success']['mean']:.3f} | {t['mean']:.2f}±{t['SD']:.2f} | {t['P90']:.2f} | {rot['mean']:.2f}±{rot['SD']:.2f} | {rot['P90']:.2f} | {lookup[name]['ADDsym_AUC_full']:.4f} |")
    lines+=['','[확인] 영상만 쓴 V_LOGISTIC의 합성 TEST 직접 분류 정확도는 **84.961%**, shuffle은51.465%여서 해당 합성 자료의 영상 특징에 분류 정보가 있다. 하지만 REAL173의 V_LOGISTIC balanced accuracy는46.355%, 직접 정확도45.882%, 성공24.470%로 **실사 전이가 실패**했다. G_MLP에 V를 추가하면 직접 축 오류25.686→29.412%, 기존W/D22.543→26.782%로 악화했고 성공47.206→46.628%다. 현재 파일럿에서 영상 특징의 실사 추가 가치는 확인되지 않았다([C_RESULTS.json](C_RESULTS.json), [C_SYNTH_TEST_RESULTS.json](C_SYNTH_TEST_RESULTS.json)).',
       '', '[확인] 동일 크기의 G+V+D는 G+V 대비 성공+.963pp와 기존W/D−.578pp의 작은 개선 방향이지만, 공식 선택 대비 성공+.963pp와 기존W/D+3.276pp의 **TRADEOFF**다. 치수의 연결 위치에 따라 반응·방향은 다르며 운영 우위를 확정할 증거가 없다. 전체8개 Holm p는 모두.3536 이상이고 임의 practical 합격선을 만들지 않았다.',
       '', '[확인] C 합성 TEST1024는 이번 C 분류기 fitting과 group 분리됐지만, 원 N3 TRAIN 노출926장과 겹치며 실제 renderer group은1개다. 전체 파이프라인의 독립 최종 test나 다양한 환경 일반화로 부르지 않는다. A3 signed-front 복원 이슈의 영향을 별도 명시한 채 raw 합성 결과를 보존한다([BC_INPUT_CONTRACT.json](BC_INPUT_CONTRACT.json), [C_SYNTH_TEST_RESULTS.json](C_SYNTH_TEST_RESULTS.json): `independent_final_test=false`).',
       '', '| Selector | Clean68 성공 % | Moderate42 성공 % | Severe63 성공 % |', '|---|---:|---:|---:|']
    for name in order:
        values=[cc['summaries'][name]['grade'][g]['strict_success']*100 for g in ['clean','moderate','severe']]
        lines.append('| '+name+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines+=['', '[확인] Severe를 포함한 모든 grade를 동일 방법으로 남겼다. material·session·reference-margin·고도 하위군은 [C_RESULTS.json](C_RESULTS.json)의 descriptive 수치이며 실사로 모델이나 threshold를 다시 고르지 않았다.',
       '', '![실제 성공·손상·큰 꼬리 사례](FIGURES/08_real_recovery_damage_and_tail_cases.png)',
       '', '[확인] 사례 그림은 **고정 seed1**에서 strict 회복 중 최대 R 감소, 직접 축 손상 중 최대 R 증가, 전체 공식 선택 중 최대 T, 모든 D good5→bad10 코너 손상을 정해진 기준으로 보여준다. 공개 범위는 실제 두 후보의 파생 T/R 수치이며 RGB·참조 좌표를 포함하지 않는다. 회복 `eval_pallet09:1778653889069260032`는 T22.811/R84.413→T2.081/R3.304로 성공을 회복했다. 축 손상 `eval_pallet09:1778653723571790336`는 R1.954→87.719로 악화하며 두 경로 모두 T>5cm라 strict는 모두 실패다. 가장 큰 T 사례도 양 방법에서 해결되지 않는다. 모든 전환 ID는 [CHANGED_SELECTIONS.csv](CHANGED_SELECTIONS.csv), 자세 수치는 [FIGURE_CASES.json](FIGURE_CASES.json)에 남겼다.']
    text=replace_section(text,'C','\n'.join(lines))

    gvd=next(r for r in risk['curves'] if r['method']=='G_V_D_MLP' and r['seed']==1)
    old=next(r for r in oldrisk['curves'] if r['method']=='N3_THEN_SUBPIX' and r['seed']==1 and r['confidence']=='official_weighted_score_gap')
    lines=['[확인] classifier confidence=`|2p_long−1|`의8arm+TRAINprior×3seed **27별도 곡선**과 synthetic VAL-only threshold의 실사 전이를 추가했다. 대표 G+V+D는 사전 주대비 arm이며 좋은 classifier를 사후 고른 것이 아니다. seed1 AURC **.27349**, 직접 축 오류 검출 ROC-AUC **.67094**로 기존 raw-gap seed1(.23220/.73088)보다 좋지 않다([E_PROBABILITY_RISK.json](E_PROBABILITY_RISK.json)).','',
       '| 실사 설명용 seed1 고정 coverage | 실제 처리율 % | 공식 gap accepted 실패 % | GVD accepted 실패 % | 공식 gap 성공/전체 % | GVD 성공/전체 % |','|---|---:|---:|---:|---:|---:|']
    for goal in [.95,.9,.8,.5]:
        a=next(r['actual'] for r in old['fixed_points'] if r['target_coverage']==goal)
        b=next(r['actual'] for r in gvd['fixed_points'] if r['target_coverage']==goal)
        assert a['coverage']==b['coverage']
        lines.append(f"| {goal*100:.0f}% | {100*a['coverage']:.3f} | {100*a['accepted_6D_failure']:.3f} | {100*b['accepted_6D_failure']:.3f} | {100*a['accepted_success_over_all']:.3f} | {100*b['accepted_success_over_all']:.3f} |")
    lines+=['','| GVD synthetic VAL 목표 처리율 % | VAL 실제 % | REAL 실제 % | REAL accepted 실패 % | REAL 성공/전체 % |','|---|---:|---:|---:|---:|']
    for r in gvd['VAL_only_threshold_transfer']:
        if r['target_VAL_coverage'] not in [.95,.9,.8,.5]:continue
        lines.append(f"| {100*r['target_VAL_coverage']:.0f} | {100*r['VAL_actual_coverage']:.3f} | {100*r['REAL_actual_coverage']:.3f} | {100*r['accepted_6D_failure']:.3f} | {100*r['accepted_success_over_all']:.3f} |")
    lines+=['','[확인] VAL50% threshold를 실사에 옮기면 실제26.012%만 처리하고 전체 성공 output은21.965%다. confidence는 probability calibration이나 환경 전이 후 안전 보장과 동일하지 않다. 이 threshold들은 VAL에서만 산출했으며 REAL risk를 맞추도록 재선택하지 않았다.']
    text=replace_section(text,'C2','\n'.join(lines))

    lines=['![전체 방법과8개 주대비](FIGURES/07_models_and_primary_paired_effects.png)','',
       '[확인] 그림(c)의1=GV−G,2=GVD−GV,3=D_JOINT_DIM→SubPix−CONTROL→SubPix,4=GVD−공식 선택이다. D의 **SubPix 경로는 실행 전 주대비로 고정**됐으며 Stage1은 별도 descriptive다. effect는 after−before(pp), W/D는 감소가 개선이고 성공은 증가가 개선이다. 동일9세션의10000draw를 모든 주대비에 사용했다([PAIRED.json](PAIRED.json), [BOOTSTRAP_LOCK.json](BOOTSTRAP_LOCK.json)).','',
       '| 사전 대비 | 지표 | 영상 가중 Δpp [95%CI] | 세션 동일 가중 Δpp | 원 p | Holm p |','|---|---|---:|---:|---:|---:|']
    for r in paired['results']:
        label='W/D confusion' if r['metric']=='wd_confusion' else '5cm·5°'
        lines.append(f"| {r['after']} − {r['before']} | {label} | {r['effect_pp']:+.3f} [{r['CI95'][0]:+.3f},{r['CI95'][1]:+.3f}] | {r['session_equal_effect_pp']:+.3f} | {r['p_raw']:.5f} | {r['p_Holm']:.5f} |")
    lines+=['','[확인] 이 p는 중심을 맞춘 paired bootstrap의 정확 integer tie/plus-one 정의이며, percentile CI를 역산해 p를 만들지 않았다. 양방향 CI와 p는 완전히 동일한 반전검정이 아니므로 CI의 끝점만 보고 확증 판정을 내리지 않는다. corrected family에서는 확증할 유의성이 없고, 이것을 효과0이라고 단정하지 않는다.',
      '', '[확인] 최종 독립 검산은10,726중복 제거 원행·52집계·8주가설을 다시 계산했다. 3,276통계 비교 최대 차이는1.455e−11, CI차이5.684e−14, ADDsym AUC차이2.220e−16이었다. 원행 join34,600건, 보호 FINAL 채점0, Severe63 보존, 최초D adapter 실패행 집계 제외를 확인했다([FINAL_AGGREGATE_VERIFICATION_FINAL.json](FINAL_AGGREGATE_VERIFICATION_FINAL.json): PASS).',
      '', '[확인] 총 실제 OpenCV solve/refine는**48,277/50,000**(A3 5,920/B1 3,114/C 30,748/D 8,495; 실패 호출 포함), classifier8fit, corrector2fit×1000updates다. 동결 YOLO 합성 재추론6144장과 N3cachedfeature6144forward를 기록했다. measured GPU active phase 합계4.401분/45분, peak allocated373,743,616bytes이며 새 REAL 고비용 RGB특징은0이다. CPU peak memory 미계측 범위도 그대로 명시한다([COMPUTE_RECEIPT.json](COMPUTE_RECEIPT.json)).',
      '', '[확인] D repair lock의 “three frames” 표현은 실제로 **같은 첫 영상의 세 route 예측**이었음을 별도 검산 파일에 정정했다. 원 lock SHA를 보존하고 silently rewrite하지 않았다([D_VERIFICATION.json](D_VERIFICATION.json): distinct_frames1, predictions3).',
      '', '[추정] 다음 행동은 **현재 좌표 복원 계약 수정→독립 관측 확보**다. C1 signed-front round-trip를 GT분기 없이 통과시키는 최소 회귀증거가 먼저 필요하다. 분류기8개를 추가 탐색하거나 실패 D의 설정을 반복 변경하지 않는다. GVD의 작은 strict 증분만으로 운영 selector를 교체하지 않고 현재 공식 경로를 유지한다. 영상 추가 가치를 다시 판단하려면 사전 고정된 selector가 독립 환경에서 기하 대비 W/D·strict·꼬리를 함께 개선하고 손상/처리율을 공개하는 최소 검증이 필요하다. 실제 plane/motion을 쓰려면 평가 GT와 독립적인 측정·연결 계약부터 확보한다.']
    text=replace_section(text,'PRIMARY','\n'.join(lines))
    first='허용된 REAL173에서 치수 입력은 반응했지만 영상·치수 융합의 일관된 6D 개선은 확인되지 않았고, C1 자세 복원 오류와 실사 전이 실패가 드러났다.'
    text=first+'\n'+text.split('\n',1)[1]
    text=text.replace('새로 fitting하는 계획이다.','새로 fitting했다.')
    text=text.replace('최종 질문에 대한 현재 답과 다음 판단','최종 질문에 대한 답과 다음 판단')
    text=text.replace('C/B1/C2 최종 검증 진행 중.','C8fit/B1/C2 최종 검증 완료.')
    text=text.replace('기존 B1 최종 결과 전에는 포괄 판정을 보류한다.','B1도 반응하지만 축 선택 변경0·작은 성공 차이여서 안정적인 이득은 미확인이다.')
    text=text.replace('C8fit 봉인·paired 검증 후 판정. 과거 다른 checkpoint 결과를 현재 특징의 증거로 대체하지 않는다.','합성 V정보는 확인됐으나 실사 전이 실패. GV는 G보다W/D악화·strict감소, Holm유의증분없음.')
    text=text.replace('먼저 C1 signed-front 복원 계약을 바로잡고, C확정paired결과 전에는 새selector운영을 보류.','먼저 C1 signed-front 복원 계약을 바로잡고, 새selector운영을 보류.')
    text=text.replace('완료 시 [EXPERIMENT_STATUS.csv]','[EXPERIMENT_STATUS.csv]')
    text=text.replace('실제 실행/재사용/차단 전체 상태와 비용·원격 게시 검증은 [EXPERIMENT_STATUS.csv]',
       '실제 실행/재사용/차단 전체 상태와 비용·재현은 [EXPERIMENT_STATUS.csv]')
    text=text.replace('PNG300dpi, 영어 label이며',
        '새 분석 PNG8개는300dpi, 별도 D 방법 그림은160dpi이며 모두 영어 label이다. 또한')
    unfavorable='- [확인] V_LOGISTIC은 합성 TEST 정확도84.961%에서 실사 직접 정확도45.882%로 전이 실패했다. 기하에 V를 추가한 주대비도 기존 W/D confusion이+4.239pp 악화했다. 작은 GVD 성공 증분만 남겨 영상 효과가 확인됐다고 주장하지 않는다([C_RESULTS.json](C_RESULTS.json), [PAIRED.json](PAIRED.json)).\n'
    if unfavorable not in text:
        at=text.index('## 정보 흐름과 평가 계약')
        text=text[:at]+unfavorable+'\n'+text[at:]
    if '[SOURCE_PRESERVATION.json]' not in text:
        text=text.replace('## 검산·재현·상태\n',
            '## 검산·재현·상태\n\n[확인] 원본 workspace의449개 기존 변경 상태와 tracked diff bytes를 보존했고,24개 역사 artifact의 SHA가 전후 일치했다([SOURCE_PRESERVATION.json](SOURCE_PRESERVATION.json)).\n')
    if '![D 국소 결합 방법]' not in text:
        text=text.replace('## D: 치수·영상 국소 결합의 실제 이득\n',
            '## D: 치수·영상 국소 결합의 실제 이득\n\n![D 국소 결합 방법](FIGURES/D_LOCAL_METHOD.png)\n')
    path.write_text(text)
    print('REPORT_FINAL',len(text.splitlines()),'lines','eight fixed hypotheses','all eight classifiers')


if __name__=='__main__':run()
