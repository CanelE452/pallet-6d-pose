"""Readable Korean prose for the frozen diagnostic report."""
PARAGRAPHS = {
    'FULL125는R0 대비2D': 'FULL125는 R0 대비 2D 중앙값을 10.692→8.566 px로 낮췄지만, T/R 이득은 −0.417 cm/−0.807°이며 T P90은 줄지 않았다. PRIOR1 대비 FULL125는 T −0.056 cm, R +0.170°의 tradeoff다. **FULL125가 PRIOR1보다 두 pose 축 모두 우수하다는 결론은 성립하지 않는다.**',
    'CPU R0와 기존 GPU': 'CPU R0와 기존 GPU 캐시 사이 최대 0.025 px 차이가 최초 0.01 px 재사용 smoke 기준을 넘었다. 기존 tolerance를 완화하지 않고 clean·가림 R0와 보정기 양쪽을 CPU로 재계산했다. 최초 clean smoke는 재사용했다. 따라서 E1 기존 GPU 캐시와 E3 CPU clean 값은 미세하게 다르며 이를 모델의 성능 개선으로 해석하지 않는다. affine roundtrip, invalid/center 보존, 모델 state 불변을 확인했다. [E3 결과](E3_SUMMARY.json), [CPU 정합 조치](E3_CPU_CONSISTENCY_ADDENDUM.json).',
    '학습은실사': '학습 구성은 실사 157×20, negative 259×6, 합성 12,000이며 validation은 합성 1,000장이다. 초기 모델도 R0 G38과 다르고 detector를 포함한 학습 구성이 다르다. 실제 이미지 SHA 중복은 0이지만 학습 recording과 평가 10/128장(자연 9/clean 1)이 겹친다. 겹치지 않는 natural90에서도 R0 T/R 11.919/5.045→REALFT_A 6.141/2.822로 개선한다. 이 90장도 반복 사용 DEV이며 새 독립 TEST가 아니다.',
    '기존 REALFT_A report': '기존 REALFT_A 보고서의 “epoch60 last” 설명은 요청된 best.pt 내부 args의 epochs40, stripped epoch−1, 기록된 train_results 16 epoch와 맞지 않는다. 이번에는 지정된 정확한 hash만 사용했고 유사 모델로 대체하지 않았다. 과거 정확한 checkpoint 선택 경위는 UNRESOLVED다. [계보](MODEL_LINEAGE.json), [checkpoint 내부 확인](REALFT_A_SELECTION_HISTORY.json), [중복 분리](E7_SUBGROUP_SUMMARY.json).',
    'natural99는6recording': 'natural99는 6 recording, clean29는 3 recording이다. paired recording bootstrap은 기존 2,000회/seed20260929를 재사용했고 같은 draw를 모든 paired 모델에 적용했다. undefined draw 수를 보존하며 이번 주 비교에서는 0회다. 각 recording의 N과 절대 T/R·차이는 [BY_RECORDING.json](BY_RECORDING.json), LORO는 각 결과 JSON에 있다. 가림·앙각·거리·bbox 크기·W/D·매칭 조건은 기존 metadata 범주 그대로 [층별 결과](E7_SUBGROUP_SUMMARY.json)에 있다. 작은 집단을 교차한 회귀는 추가하지 않았다.',
    '모든128장/1024코너': '모든 128장/1,024코너의 source 필드가 unknown이고 migration은 MANUAL_REVIEW_REQUIRED다. 이것이 1,024좌표가 틀렸다는 뜻은 아니다. manual이라는 파일명/객체 필드, 사람의 visible/occluded 판정, 직접 원클릭/투영점 출처, 독립적인 물리 축 정답은 서로 다르다. geometry-resolved 참조는 기존 2D·K·치수에서 유도되며 독립 장비로 실측한 6D가 아니다.',
    '**경험적noise floor': '**경험적 noise floor는 BLOCKED:** 독립 반복 원클릭·annotator/repeat 연결·공분산 자료를 찾지 못했다. 가정한 2 px 잡음이나 CI 폭으로 noise floor/MDE를 만들지 않았다. 새 재클릭, GT 수정, 물리 앵커 지표 추가도 하지 않았다. [참조 한계](E7_REFERENCE_LIMITS.json).',
    'Source256의image': 'Source256의 image/label/renderer 768개 hash 연결은 확인했다. 기존 SOURCE_BASE/FULL/SOURCE_ROWS는 오차 스칼라를 저장하며 operational xy가 없어 동일 출력의 T/R을 복원할 수 없다. E1/E3에서 현재 출력 계약을 직접 검증했으므로 1,024회 추가 source 추론은 SKIPPED했다. 과거 source R을 현재 C2로 복사하지 않았다. source 좌표 stress를 RGB 가림 실험이라고 부르지 않는다.',
    '[FixMatch]': '[FixMatch](https://arxiv.org/abs/2001.07685)는 약한 증강의 고신뢰 타깃을 강한 증강 입력과 연결하고, [Noisy Student](https://arxiv.org/abs/1911.04252)는 학생에 noise를 주면서 의사 라벨로 학습한다. 분류 결과는 팔레트 pose 전이의 보장이 아니다. 이번 E1/E3처럼 타깃 추종과 실제 정확도를 별도로 검증해야 한다.',
    '[PoseFix]': '[PoseFix](https://arxiv.org/abs/1812.03595)의 오류 분포를 반영한 refinement 원리를 E5에 연결했지만, 팔레트의 W/D·검출 실패를 사람 pose의 특정 오류 유형과 동일시하지 않는다. [OpenCV PnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)는 대응점으로 pose를 구하는 절차다. 작은 재투영 잔차가 올바른 물체/대응점을 보장하지 않는 것은 이번 박스 진단에서도 확인된다.',
    '[Self6D++]': '[Self6D++](https://arxiv.org/abs/2203.10339)는 가림을 고려한 자기지도 pose 개선의 근거다. 실제 CAD/K/mask가 확보되면 같은 후보의 렌더링·마스크/RGB 정합 점수가 참조에 가까운 후보를 구분하는지부터 검증할 수 있다. RGB-only 경로에 외부 depth가 필수라고 가정하지 않았고, 렌더러를 새로 구축하거나 논문을 재학습하지 않았다. 이는 이번에 선택한 후속 실험에 추가하는 묶음이 아니다.',
    '[SLAM-supported': '[SLAM-supported self-training](https://arxiv.org/abs/2203.04424)은 연속 관측과 카메라 운동 정보를 활용할 수 있다는 원리다. 해당 정보가 있을 때에만 후보의 다중 프레임 일관성과 T/R 관계를 검증한다. 동일한 90° 오답도 시간적으로 일관될 수 있다. SLAM을 구축하지 않았다.',
    '[JCGM101]': '[JCGM101](https://www.bipm.org/en/doi/10.59161/jcgm101-2008)의 분포 전파에는 정의된 입력 불확실성이 필요하다. 이번에는 실측 반복 클릭 분포가 없어 이를 가정으로 채우지 않았다. [Cameron–Miller](https://faculty.econ.ucdavis.edu/faculty/cameron/research/Cameron_Miller_JHR_2015_February.pdf)의 작은 cluster 수에 대한 주의를 반영해 6/3 recording 구간을 탐색적으로 표시하며 특정 검정의 보장이나 MDE로 해석하지 않는다.',
    '가림RGB대cleanRGB': '가림 RGB와 clean RGB를 같은 실제 pose·같은 qO에서 비교하는 짝 자료를 우선한다. 임시 예산은 새 6 recording×고정 2 pose×clean-before/occluded/clean-after=36장이다. 검정력으로 정한 표본 수가 아니다. 카메라·팔레트·K를 triplet 내에서 고정하고 occluder 위치를 출력 확인 전에 한 번 정한다. clean-after는 pose 고정 가정의 점검용이며 독립 표본으로 세지 않는다.',
    '고정PRIOR1/FULL125': '고정 PRIOR1/FULL125에 같은 qO·clean bbox·affine·confidence를 주고 RGB만 clean-before/occluded로 바꾼다. 동일 GEO와 held-W/D를 함께 보고, nativeOO·검출 실패·원래 12개 분모를 보존한다. clean 주석의 직접 클릭/투영 출처와 기하 참조의 한계를 명시한다. pose가 움직인 쌍은 그대로 표시하고 성공 쌍으로 대체하지 않는다.',
    '이비교에서RGB': '이 비교에서 RGB 가림 손실이 recording을 넘어 일관되면 다음 학습 요인은 오류 방향 상관성보다 촬영/가림 구성으로 좁힌다. 일관되지 않으면 학습을 보류하고 참조·후보·검출 설명을 유지한다. 이번 진단만으로 6 fit 또는 9 fit 학습을 동시에 제안하지 않는다. ST/보정기의 불가능이나 현재 99장의 일반화 성공을 선언하지 않는다.',
    '독립T/R 재계산': '독립 T/R 재계산 2,964행, 기존 R0/OLD_REF same-GEO 256행 parity, 원본 입력/code hash, source 768개 binding, 모델 state/BN 불변, 좌표 roundtrip·center/invalid 보존을 검증했다. [VALIDATION.json](VALIDATION.json), [RESOURCE_LEDGER.json](RESOURCE_LEDGER.json), [실행 명령](EXECUTION_COMMANDS.md).',
    '프레임·recording·모델': '프레임·recording·모델·조건·valid/matching·T/R·2D·기준 대비 변화는 [FRAME_RESULTS.csv](FRAME_RESULTS.csv), 후보별 전체 pose 오차는 [CANDIDATE_RESULTS.csv](CANDIDATE_RESULTS.csv)다. 전체 좌표/이미지 hash/변환/마스크/참조/후보 상세는 `data/pallet/results/pallet_pose_diagnosis_20260930_v1/`에 보존했다. 모든 산출물은 로컬이며 commit/push/외부 공유하지 않았다. 표에 필요한 판단 근거를 담았으므로 장식용 그림은 추가하지 않았다.',
}

KOREAN_STATUS = {
    'E0':'현재 ID·모델·좌표·평가 계약과 hash 일치 확인.',
    'E1':'동일 R0 입력에 identity/PRIOR1/FULL125, 같은 GEO와 W/D 고정 비교 완료.',
    'E1_target_accuracy':'natural99의 대응 clean view/실제 teacher 타깃이 없어 물리적 타깃 정확도는 미확정.',
    'E1_interpolation':'endpoint로 2D·pose·후보 효과를 구분했고 자연99 타깃이 없어 추가 보간 생략.',
    'E1_source':'768개 연결 확인. 출력 xy 부재와 의사결정 필요성에 따라 추가 1,024회 추론 생략.',
    'E2':'후보 전체 pose oracle과 모든 저장 박스 분석 완료. T 상위10장 중 매칭 실패8장.',
    'E2_wrong_instance':'박스 불일치만으로 다른 실물인지 물체 일부/배경인지 확정할 수 없음.',
    'E2_ROI':'박스 pool의 회복 가능3장/후보 누락5장이 구분돼 추가 crop 재추론 생략.',
    'E3':'29장×2마스크×2보정기의 2×2/nativeOO 완료. 실제 코너 중첩과 3 recording 한계 기록.',
    'E3_general_effect':'CO→OO 악화 신호는 있으나 recording CI가 0을 포함해 일반적 효과는 미확정.',
    'E4':'사람 검토174코너를 현재99에 연결. 교체 가능110+47코너로 두 선택 경로 비교.',
    'E5_A':'기존 TRAIN253의 hard50/2024, FULL125 복구49/50 재사용. 과거DEV93와 구분.',
    'E5_B':'새 고정 TRAIN29 마스크에서 >20px 오류가 0개여서 어려운 오류 전이는 미해결.',
    'E5_stress':'주변 오차 분포가 같은 4코너·30px 비교 완료. 복구79/116 대81/116.',
    'E6':'지정 REALFT_A 128장 및 동일 GEO 평가 완료. 중복 recording을 분리해도 개선 유지.',
    'E6_selection_history':'정확한 체크포인트 hash는 확인했지만 과거 epoch60 last 설명과 내부 기록 불일치.',
    'E7_statistics':'전체/실패 분모, paired recording bootstrap, LORO, 모집단별 recording·조건 집계 완료.',
    'E7_noise_floor':'독립 반복 클릭·annotator/repeat 연결·실측 공분산 자료 부재.',
    'next_intervention':'고정12pose의 clean→가림→clean36장 비교 하나로 계획을 좁힘. 실행하지 않음.',
}

def render(lines,cost):
    out=[]
    for line in lines:
        key=next((p for p in PARAGRAPHS if line.startswith(p)),None)
        if key:line=PARAGRAPHS[key]
        if line.startswith('신규 학습/optimizer update0'):
            line=(f"신규 학습/optimizer update 0회, GPU 시간 0초. CPU image-forward 총 1,027회 = E3 609(R0 87+보정기 522)+E5-B 174+E5 stress 116+REALFT_A 128이다. E1 추론은 0회다. 모델별 최초 smoke는 최종 clean 출력으로 재사용했다. CPU 측정 합은 {cost['CPU_seconds']:.1f}초({cost['CPU_seconds']/60:.1f} CPU분), stage wall 합은 {cost['summed_stage_wall_seconds']:.1f}초다. 병렬 단계 wall은 겹치므로 전체 경과시간으로 해석하지 않는다. 단일 process peak RSS는 {cost['peak_process_RSS_KiB']/1024:.1f} MiB이며 동시 총 메모리는 미측정이다. 조사·작성·웹 열람 시간은 stage 계측에 포함되지 않는다.")
        if line.startswith('| '):
            parts=line.split(' | ')
            if parts[0][2:] in KOREAN_STATUS:
                line=' | '.join(parts[:2]+[KOREAN_STATUS[parts[0][2:]]+' |'])
        for old,new in [('캐시512','캐시 512'),('CI는0','CI는 0'),('고정12pose','고정 12 pose'),('clean→가림→clean36장','clean→가림→clean 36장'),('FULL125의2D','FULL125의 2D'),('가림6D','가림 6D'),('교집합은92','교집합은 92'),('old-only1/current-only7','old-only 1/current-only 7'),('오차cm','오차 cm'),('frame의C2','frame의 C2'),('90°W/D','90° W/D'),('왜곡계수None','왜곡계수 None'),('중심8','중심 8'),('자연99의2D','자연99의 2D'),('실제T/R','실제 T/R'),('동일GEO','동일 GEO'),('코너2D','코너 2D'),('개선34','개선 34'),('악화27','악화 27'),('개선23','개선 23'),('개선15','개선 15'),('bootstrap95%','bootstrap 95%'),('구간은T','구간은 T'),('모두0','모두 0'),('clean29에서도2D','clean29에서도 2D'),('중앙값7.258','중앙값 7.258'),('달리T','달리 T'),('중앙값2.930','중앙값 2.930'),('실제GEO','실제 GEO'),('전체pose','전체 pose'),('전체(R,t)','전체 (R,t)'),('최소T·최소R','최소 T·최소 R'),('허용은1e','허용은 1e'),('상위10장 중8장','상위 10장 중 8장'),('기존IoU0.5','기존 IoU 0.5'),('않은3장','않은 3장'),('미매칭인5장','미매칭인 5장'),('나머지2장','나머지 2장'),('미검출은0','미검출은 0'),('P90은120','P90은 120'),('참조IoU','참조 IoU'),('T/R은11','T/R은 11'),('글자는RGB','글자는 RGB'),('둘째는좌표','둘째는 좌표'),('않았고58/58','않았고 58/58'),('박스IoU0.5','박스 IoU 0.5'),('단위는cm','단위는 cm'),('변화는+','변화는 +'),('중앙값은+','중앙값은 +'),('그러나3recording','그러나 3 recording'),('넓고0','넓고 0'),('avoid에서도+','avoid에서도 +'),('대비T','대비 T'),('controlledOO는T/R','controlledOO는 T/R'),('평균코너오차','평균 코너 오차'),('cover5.267px/avoid4.239px','cover 5.267 px/avoid 4.239 px'),('cover1장은0코너','cover 1장은 0코너'),('avoid3장은1코너','avoid 3장은 1코너'),('모든29장','모든 29장'),('참조코너','참조 코너'),('않는다.58개 조건은29개 원촬영','않는다. 58개 조건은 29개 원촬영'),('current99의8코너792슬롯','current99의 8코너 792슬롯'),('검토174=occluded125+visible49, unknown618','검토 174=occluded 125+visible 49, unknown 618'),('것은occluded110코너/74장, visible47코너/33장','것은 occluded 110코너/74장, visible 47코너/33장'),('old93의169','old93의 169'),('고정R','고정 R'),('제외PnP','제외 PnP'),('hard>20px는50/2024','hard>20 px는 50/2024'),('FULL125는49/50을≤10px','FULL125는 49/50을 ≤10 px'),('TRAIN20/30/40px','TRAIN 20/30/40 px'),('매칭91장/702코너 중>20px203개, >40px97개, hard보유60장','매칭 91장/702코너 중 >20 px 203개, >40 px 97개, hard 보유 60장'),('오류26장','오류 26장'),('타깃과DEV','타깃과 DEV'),('실측GT','실측 GT'),('TRAIN253을ID','TRAIN253을 ID'),('등간격29장','등간격 29장'),('nativeOO만174회','nativeOO만 174회'),('같은 객체지만>20px 입력오류는0','같은 객체지만 >20 px 입력 오류는 0'),('중앙오차','중앙 오차'),('identity1.867→FULL1250.640px','identity 1.867→FULL125 0.640 px'),('오차는0.638','오차는 0.638'),('같은4코너','같은 4코너'),('같은29개','같은 29개'),('큰4코너','큰 4코너'),('독립79/116 대 동반81/116','독립 79/116 대 동반 81/116'),('자연가림 RGB','자연 가림 RGB'),('체크포인트를128장CPU','체크포인트를 128장 CPU'),('매칭98/99','매칭 98/99'),('R0/보정기91/99','R0/보정기 91/99')]:
            line=line.replace(old,new)
        out.append(line)
    return '\n'.join(out)
