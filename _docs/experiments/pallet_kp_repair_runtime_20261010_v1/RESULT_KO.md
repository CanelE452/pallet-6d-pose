“가림 판단이 일부 틀려도 다른 유효 대응점으로 자세를 구하고 자기 가림 코너를 재투영했을 때, 기존 단순 대조보다 실제 위치·회전이 좋아졌는가?”

**아직 개선은 입증하지 못했다.** [실제로 실행한 상세 실험·이미지·원행·감독 복구 보고서](../pallet_kp_supervision_repair_20261010_v1/RESULT_KO.md)를 그대로 보존한다. 추가9000update 승인이 아직 없어 수정 모델의 학습·새 실사1914행·새 전체 경로600회 측정은 실행하지 않았다. 이 보완은 수정 모델이 생긴 뒤 사용할 **실제 전체 경로 시간 측정 연결과 사전 중단 검사**를 완성한 것이다. 준비 코드를 실행한 모델 성능이나 측정 latency로 보고하지 않는다.

## 왜 별도 측정 연결이 필요한가

원래 실험과 KP 후속 실험은 이미 실제 detector를 포함해 시간 측정을 했다. 그 기존 시간을 수정 모델의 시간으로 재사용하면 안 된다. 특히 후속 KP benchmark는 match-mass decoder·치수 prior 절제를 포함하는 반면, 수정 감독 비교는 원래66-way argmax·TLS·코너 교점 decoder를 그대로 유지한다. 따라서 **원래 observation benchmark를 재사용하고, 학습 체크포인트와 결과 출력 경로만 별도로 연결**한다. match-mass·새 prior·all-lines 절제·threshold 변경을 추가하지 않는다.

| 측정 경로 | 실제 측정 구간에 포함할 처리 | 한 경로 실행량 |
|---|---|---:|
| BASE | RAM영상→고정detector→Base좌표→기존초기자세 | warmup20+26영상×5=150 |
| N3_SUBPIX | 같은detector→기존N3→SubPix→기존자세 | 150 |
| N3_SUBPIX_GEOM_NOSELF_ROBUST | 같은detector→N3/SubPix→가림용초기자세→유한4점합의→H재투영 | 150 |
| 수정 IMAGE_ROLE | 같은detector/neck→원래소형head/66-way선택→역할용자세와가림용초기자세→유한4점합의→H재투영 | 150 |

합계600전체실행이며 measured520/warmup80이다. 동일26영상·13세션·경로 순서 회전·홀수block 역순을 유지한다. 모델·가중치 로딩, RGB 파일 decode, GT채점, parity 검사, durable journal 쓰기는 구간 밖이다. GPU synchronize 후 실제 perf-counter로 구간을 측정하며 cached 좌표나 cached fit를 재생하지 않는다. head를 위한 fresh 역할용자세와 final mask를 위한 별도 초기자세의 비용도 전체 구간에 포함한다.

## 실제 준비 검사와 현재 없는 결과

[RUNTIME_READINESS_CHECKS.json](RUNTIME_READINESS_CHECKS.json)은 실행한 준비 검사다. 완료 기록·가중치·새 sealed geometry가 없는 상태에서 detector/GPU import 전에 중단하는지, output이 과거311파일이나 원본/user checkout을 덮어쓰지 않는지, 고정600개 schedule이 정확한지 검사한다. [독립 코드 검토](CODE_REVIEW.json)는 이 adapter가 원래 실제 timing loop를 호출하는지 확인하며 모델·PnP 실행 증거를 대신하지 않는다. 실제 검사 command·exit·코드 hash·실행량을 기록한다. 독립 검토자는 별도8999-update 완료 fixture를 실제 CLI에서 제시했고, Torch/CV2/NumPy/model import 요청0회 상태에서 정확히 budget guard로 거부됐다. 이 fixture는 실제 학습 기록이 아니다. 실행한 startup/output 검사29개는 모두 통과했고, 원래 저장된600개 job ID/순서와 새 schedule이 일치했다. 아직 양성 완료 상태의 전체 guard 및600회 실제 benchmark는 실행하지 않았다.

현재 수정 모델 TRAINING_COMPLETION과 가중치는 없고, 추가학습 승인도 PENDING이다. **새 detector0/head0/optimizer0/PnP0/실제runtime0회**다. 새로운 평균 latency·새실사정확도·정식runtime PASS는 없다. 기존 실제600회 timing은 원래 보고서에 그대로 남아 있지만 이 phase의 결과로 복사하지 않는다.

측정 시작 전에는 고정 repair protocol·정식9000 update 완료 기록·사용자 승인 기록·세 마지막 checkpoint hash·reference를 읽기 전에 봉인한 IMAGE_ROLE geometry와 그 checkpoint 연계를 확인한다. 단순히 파일 이름이 같거나 원래 오래된 모델이 있다는 이유로 실행하지 않는다. 새 output/journal은 원래 결과와 분리하며, 중단된 실행은 보존하고 자동 재시작하지 않는다.

## 정확도와 시간 검산의 관계

측정 구간 밖에서 native raw좌표·후보선택·센터·box·score와 final좌표·R,t·status·H가 고정 sealed 출력과 일치하는지 확인한다. 큰 오차가 있어도 parity가 같으면 그 비용을 남기며, accuracy 성공 프레임만 시간 표에 남기지 않는다. parity 실패·경합·열상태 문제가 있으면 기록을 보존하고 정식 timing으로 표시하지 않는다. 원래 GT canary와 competing workload/온도 검사는 그대로다. 다른 작업을 종료하거나 경합 중 측정을 강행하지 않는다.

H 초기좌표는 final residual에 넣지 않으며, 새R,t가 있는 경우 H를 최종 재투영으로 실제 교체한다. 재투영 좌표를 다시 독립 PnP입력으로 쓰지 않는다. finite4-point subset·다중해·fallback·전체운용통계의 구현은 원래 고정 코드다. 이 runtime 준비에서 이를 새 solver 개선이라고 주장하지 않는다.

## 공개 확인·보존·남은 승인

이전311개 게시 파일은 [PRIOR_PUBLICATION_BINDINGS.json](PRIOR_PUBLICATION_BINDINGS.json)에서 SHA·크기를 보호한다. 현재 코드·검사·문서 hash는 [REVIEW_MANIFEST.json](REVIEW_MANIFEST.json), 공개 hash 재검산 결과는 [PUBLIC_REVIEW.json](PUBLIC_REVIEW.json), 게시 SHA와 원격 일치는 [PUBLICATION.json](PUBLICATION.json)에 있다. [재현 명령](REPRODUCE.md)에는 지금 실행 가능한 사전 검사와 아직 실행할 수 없는 실제600회 측정을 구분했다.

추가학습9000update 승인은 앞서 요청한 같은 질문으로 남아 있으며, 이번 준비를 새 승인으로 해석하지 않는다. [첨부 지시문 §12 발췌](ATTACHED_BUDGET_EXCERPT.md)의 “추가 비용이 필요하면 해당 블록만 멈추고 이미 완료한 것부터 게시한다”에 따라 독립적으로 가능한 준비와 실제 검사를 먼저 완료했다. 승인 후 같은3모델×3000update→전체319장×6경로→검산→실제600회측정→상세보고·research push를 수행할 준비이며, 실제 위치·회전 개선 달성은 여전히 미완료다. main push·main merge·force push는 수행하지 않는다.
