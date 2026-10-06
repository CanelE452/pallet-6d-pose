# README/CLAIM 작성용 원고 변경 근거 메모

[확인] 이 메모는 새 paper_updated의 실제 TeX 문장과 캡션을 가리켜. 최종 A 결과·손상·신규 A runtime을 실제 보충원고에 넣었고 논의·결론에도 제한된 부정 결과를 연결했어. 원래 N3 자료·숫자는 그대로야.

| 핵심 주장 범위 | 원고의 실제 문장 또는 수치 | 파일 |
|---|---|---|
| 조건부 국소 보정 | “본 연구는 등록 치수를 사용할 수 있고 초기 팔레트 코너가 이미 산출된 조건에서, 영상 특징과 치수에 따른 국소 키포인트 보정을 평가한다.” | abstract_ko.tex |
| DEV 분리 | “319장 개발자료에서 관측한 개선을 새로운 미사용 환경이나 실제 포크 삽입 성공으로 확대하지 않는다.” | sections/01_introduction.tex |
| 원인 단정 제한 | “작은 추가 이득의 원인을 모델 용량 부족으로 전제하지 않고, 후보에서의 선택 가능성, 실제 선택·학습, 독립 측정의 충분성을 별개 질문으로 둔다.” | sections/01_introduction.tex |
| 치수의 약한 추가 효과 | N2−N0 코너 중앙값 약 −0.1662 px. 치수 내용과 추가 파라미터 효과는 완전 분리하지 않았어. | sections/05_results.tex |
| 대칭의 혼합 효과 | N3−N2 코너 중앙값 약 +0.0004884 px. P90·위치·회전의 변화 방향이 같지 않아. 정사각형 119장 전체수동 모드도 N3−N2 +0.055 px, PCK −0.332 퍼센트포인트야. | sections/05_results.tex |
| 방법 전체 비교 | PoseFix 방식의 코너·위치·회전 중앙값 5.561 px/6.951 cm/2.028°가 N3 5.778 px/7.068 cm/2.070°보다 낮아. N3 코너 P90 42.134 px는 PoseFix 43.907 px보다 낮아. 동일 입력/손실/학습·선택 예산의 동등성을 주장하지 않아. | sections/05_results.tex, tables/comparator_results.tex, sections/07_discussion.tex |
| 이동 상한과 2D→6D | 실제 cap 도달 64/2445=2.62%; 같은 311장 중 2D 개선 후 위치 악화 seed별 98/78/100장이야. 용량 부족이나 이동 cap이 지배 원인이라는 결론으로 바꾸지 않았어. | sections/05_results.tex |
| 독립 참조 제한 | “자세 참조는 수동 코너와 물체 기하의 재구성에 의존한다.” “독립 물리 6D 실측 정확도를 주장하지 않는다.” | sections/07_discussion.tex |
| B 물리 pair 상태 | 8910 기록 영상+119 정사각형의 9029 관측에 연결된 독립 T/R 참조 0개. 실제 동일 상대 자세 clean/물리 가림 pair 평가 완료를 주장하지 않아. | sections/07_discussion.tex |
| 42→46 참고문헌 | 기존 42개를 실제 인용/주장과 대조하고 메타데이터 및 주장 수정 항목 총 15개. 신규 ResNet/Cawley/GUM/FactorVAE 4개로 본문 45개+보충 1개야. Simple Baselines는 유지했어. | REFERENCES_AUDIT.tsv, REFERENCES_AUDIT_SUMMARY.json |

[확인] 단위와 분모는 다음처럼 써야 해. 코너 오차는 복원된 원본 영상 px, PCK@10px는 전체 참조 코너에 대한 백분율(결측/매칭 실패도 분모에 남음), PCK 차이는 퍼센트포인트야. 위치 오차는 canonical translation[m]의 차이에 100을 곱한 cm, 회전 오차는 degree야. 319장 전체 참조 코너는 2499개이며 YOLO의 매칭된 코너 오차 분모는 2445개, 2D 개선/6D 손상 공동분석은 객체 매칭 311장이야. 전체 YOLO Base/N3 자세 산출과 위치·회전 headline 통계는 319/319장이야. 전체 자세 산출 분모와 2D 참조 매칭 분모를 섞으면 안 돼. 정사각형 119장은 한 촬영 세션·등록 치수[1.1,1.1,0.15] m이고 전체수동 602점/영상내 600점을 별도로 표시해. 해당 집합의 독립 6D 정확도는 x, 세션 구간은 NA야.

[확인] 등록 치수 입력 d=(W,D,H)와 정준 xyz=[W,H,D]는 서로 다른 순서야. C2={0,π}/C4={0,π/2,π,3π/2}는 정준 수직 Y축의 허용 proper rotation이고, W=D만으로 90도 외관·포크 작업 대칭을 승인하지 않아. 평가에서 허용한 전체 대응만 써.

[확인] 새 runtime은 같은 RTX 3080/pallet-yolo26 환경에서 native RAM BGR→전처리/검출/공유 특징→보정 head→prediction-only W/D+최종 PnP까지 잰 동기화 wall-clock ms야. 중앙값 Base/N3/PoseFix는 11.055908/14.469713/25.238526 ms, P90은 13.080935/17.169513/27.310544 ms이고 main 표 XIII에는 세 자리로 반올림했어. 13세션·26프레임·arm별 준비20·프레임당5반복, 유효 측정390+준비60=450 calls; parity 실패한 앞450을 폐기해 실제 총900 inference calls/optimizer0이야. 원출력 검산 max 차이는 Base/N3 0 px, PoseFix 0.000121719 px≤0.001 px야. decoding/loading 제외, process-resident memory는 독립 방법별 memory로 순위화하지 않아. 기존 별도 환경의 CUDA/head-only 값은 보충 표 SX에 남아 있고 wall-clock 표와 합치지 않아.

| 그림/표 | 실제 캡션이 명시하는 경계 |
|---|---|
| Fig.1 pipeline | 동일 초기 RGB 출력에서 시작하고 Base/보정 후에 같은 K와 치수를 사용해. 실선은 추론, 점선은 합성 정답의 보정 학습이야. |
| Fig.2 architecture | 두 수준 특징의 시각 점수+치수 점수, 이동 후보의 확률 가중평균과 별도 NoOp 집계. 치수 점수 조정은 기하 형상 강제와 달라. |
| Fig.3 symmetry | 전체 8코너+마스크의 일관된 대응이고 네 방향 그림은 허용 가능성을 설명해. 실제 합성 학습의 C4행은 0개라는 기존 사실을 보존해. |
| Fig.4 examples | seed1 그림, 사례 선택은 세 seed의 영상별 평균 변화. 수치는 영상별 8코너 평균이며 주표 풀링 중앙값과 달라. 코너를 옮기지 않고 큰 오류의 부분 감소와 소폭 악화를 모두 보존해. |
| Table V backbone | 같은 기반의 같은 초기 출력에 전후 보정. 조건부 자세 분모가 달라 기반 간 순위표가 아님을 명시해. |
| Table VII ablation | Base 대비 전체 보정과 N0/N2/N3의 작은 요소 차이를 분리하고 seed통계 평균을 ensemble로 부르지 않아. |
| Table VIII methods | 동일 319장·8코너 평가지만 입력·손실·감독이 다른 방법 전체 비교, P와 N0의 별도 가중치, PoseFix 우세 중앙값을 명시해. |
| Table XI tail | seed별 실제 개수, 코너 2445/영상311 분모, 2D 개선 후6D 악화를 분리해. |
| Table XIII matched runtime | 같은 GPU/환경의 최종 PnP 포함 wall-clock, 원래 수치 계약·원출력 parity, decoding/loading 제외·메모리 한계를 명시해. |

[확인] 실제 claim 근거는 main/supplement TeX와 RUNTIME_MATCHED/VERIFICATION, 기존 숫자 evidence야. A oracle/6fit/신규 A runtime은 최종 영수증을 검산해 별도의 탐색적 보충 결과로 넣었어. main 논의·결론은 ‘GEO의 실사 정규화 코너 오차는 N3/PoseFix보다 악화…RAW/PERM 대비 음의 구간도 좋은 코너 손상 때문에 고정 보존 조건 미충족…seed1GEO56.541ms 대N3 14.433ms’라는 정확한 제한을 명시해. 현재 PDF 화면 검토는 C_visual_review_KO.md와 C_VISUAL_REVIEW.json에 묶었고 main15/supplement14쪽의 clipping/폰트 material issue는 발견하지 못했어.

[확인] 신규 보충표SXVII는GEO/PERM6seed조건부코너median/P90(px),PCK10분율(0–1),전체319/319poseT(cm)/R(deg)야. SXVIII는GEOminusRAW/N3/PoseFix/PERM 손상코너seed별6/6/5·17/15/19·28/26/26·5/15/22개와pairedT/R평균 차이·세션CI를 넣었어. 손상은같은정준GT코너의대조<5px→GEO>10px이며프레임수가아니야. SXIX는NoOp/생성→최종W-D및같은bankADDsymgap(m),SXX는새실제A비용(ms)이야. PERM에일관된생성자세가없으므로W-D변경은NA로표시해.

[확인] 새조용한A패널Base/N3/FrozenGEO-J/GEO-J/PERM-J의wallmedian은11.0803685/14.432943/56.6632885/56.541317/56.6668995ms,P90은13.4283451/16.6356345/59.2348602/59.1736542/59.8373430ms야. 각130/130,준비100+측정650=750호출·좌표차모두0px/action/W-D/F가용성일치야. CPU집계와겹친첫750호출은DISCARDED_COST_PANEL로보존했지만보고표에서는제외했어. 기존Base/N3/PoseFix패널과별도실행임을명시했어. 현재실제빌드는main15/supplement14쪽,DONE,DOC스타일/그림포함81input계약을통과해.
