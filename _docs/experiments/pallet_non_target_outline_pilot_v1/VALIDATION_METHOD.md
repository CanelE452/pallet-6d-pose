# 독립 기능·산술 검산

실제 새 선택 출력이 봉인되고 별도 evaluator가 실행된 뒤, 신규 `independent_check.py`가 동일한 고정 입력에서 독립 검산을 수행했다. 실제 결과는 [INDEPENDENT_VALIDATION.json](INDEPENDENT_VALIDATION.json)에 있다. 검산 프로그램은 pilot의 `kernel.py`·`replay.py`, 기존 `rank_retained`, 기존 metric 함수를 import하지 않는다. 새로운 NN·PnP·최적화·후보·관측 생성 호출은0회다.

## 검사한 범위

- 고정6장 전체의 G10/G13·두 family에 저장된 retained351개와 최종18출력을 모두 읽었다. 두 성공 후보만 비교하는 검사가 아니다. `(id, stage, family, candidate_id)` 복합키와 봉인된 파일 SHA를 확인했다.
- 원 hull의 preclip vertex, canvas clipping 뒤 finite segment 및 표본의 exact 원순서를 복원했다. 508개 stage/family 부모 기록을 검사했다. 이는 영상별 같은127부모가 네 패킷에서 쓰인 기록이며 독립 물리 경계508개라는 뜻이 아니다.
- 동결된 별도 이진 마스크에서 `floor(x+0.5)`로 **preclip 원 끝점** membership을 다시 계산했다. canvas 절단점은 비대상 근거가 되지 않는다. 입력 없는3장을 포함해 새 경계·target mask 차집합·새 폐곡선을 만들지 않았다.
- 관측→모델은 원 표본 인덱스·원기여를 유지했다. 모델→관측은 각 후보의 원 표본에서 삭제 전 모든 원 finite segment까지의 최근접 대응과 원순서 동률을 별도로 계산했다. 남은 구간에 재대응하지 않았다. 서로 다른 후보의 model 표본 인덱스 차이를 짝짓지 않았다.
- 각 방향 원 normalizer·sampling/arc mass·기여 합, 유지 mass 비율, 원비용−삭제기여, 두 방향 비용 대체, MASS_MATCHED_CONTROL을 독립 함수로 확인했다. non-outline 원성분과 hard constraint도 유지됐다.
- 원 comparator의 stable order, G10 family 선택 뒤 해당 family의 G13 선택, 자기 fallback, 같은 후보의 native R/t·중심변환·physical Q를 독립 구현으로 확인했다. G10과 G13 총목적함수는 서로 비교하지 않았다.
- 무입력 재실행과 모든 실제 후보의 무입력 분기에서 scalar·전체 순위·최종 pose/metadata가 원 e와 exact하게 같았다. 실제 새 입력이 없는3장도 그대로 유지됐다.
- 공급기 실제 hook 계수 encoder3·decoder3과 no-prompt3장, 원 대안9개, 최대 quality/정확동률 첫순서, 모든 NPZ 이진 픽셀과9개 PNG0/255 표시 인코딩·파일 SHA를 재검사했다. semantic mask 정확도나 IoU를 검증한 것은 아니다.
- 실제 공급기에서 GT·target mask·후보·감사 파일 sentinel4개의 open이 차단됐다. 과학 자료의 금지 파일 읽기는0회였다. optional runtime의 `socket.bind`1회와 `subprocess.Popen`4회가 차단된 초기 실패 및 후처리 복구 기록을 보존하고 대조했다. 깨끗한 최초 종료로 바꾸지 않았다. NN·RGB 재읽기 없는 장부 복구, 유실된 시간값의 NA도 확인했다.
- selector의 실제 process read boundary와 금지 과학 함수 guard 기록을 검사했다. 이는 협조적 Python 코드의 경계이며 native C/CUDA 전체를 통제하는 운영체제 sandbox가 아니다.
- 예측 seal을 먼저 검사한 뒤 기존 참조를 읽었다. 중심 Euclidean 거리와 physical C2 반회전만으로18개 오차를 직접 계산했다.90° 동등성은 추가하지 않았다.12개 paired 전이, 평균·표본SD·최대·tail, 회복−손상=성공수 변화 및 전이군별 연속 T/R 오차 합을 확인했다.
- 원 RGB/target12개 파일, 기존 dirty6개 파일 SHA, 원 작업 트리의 NUL-delimited status·HEAD, 기존 모든 local branch ref가 시작 기록과 같았다.
- [별도 공개·가중치 검사](INDEPENDENT_PUBLICATION_CHECK.json)는 기존 publication inventory의 실제 파일 SHA와12개 원 RGB/target 멤버를 직접 대조했다. 모두 `prior_release_exact=true`다. MobileSAM vendor17개 runtime 파일과 checkpoint1개의 현재 SHA도 동결값과 같았다. 이는 작은 파생 그림의 기존 공개 이력을 확인하며 새로운 권리를 부여하는 절차가 아니다.

## 실제 검산 결과와 한계

산술·pose 최대 절대차는 `8.673617379884035e-18`, 별도 metric 최대차는 `1.4210854715202004e-14`로 모두 허용치 `1e-9` 이하다. 독립 검사 wall time은5.947602초, process CPU는3.893889초였다. 캐시 검산 시간이므로 RGB→pose 지연이나 FPS가 아니다.

공개·가중치 검사의 첫 시도는 isolated worktree의 readonly symlink 경로를 원 저장소의 문자상 하위경로로 취급하여 중단됐다. 실제 경로를 resolve한 뒤 동일 파일 바이트를 대조해 통과했다. 실패는 private `INDEPENDENT_PUBLICATION_CHECK_V1_FAILURE.json`에 남겼고, 성공 wall/CPU는0.193651/0.193609초다. 이 수정에서 모델 추론·과학 계산·입력 변경은0이다. 최초 산술 검산 당시 소스도 private `INDEPENDENT_SOURCE_AT_V1.py`로 보존했다.

원 e·SUPPORT_AWARE_RERANK·MASS_MATCHED_CONTROL은 각각 성공2/6이다. 새 두 경로의 회복·손상은 모두0이다. 최종 후보는 각각2장·3장에서 바뀌며 두 경로 모두 family가1장에서 바뀐다. 지속 실패4장의 연속 오차 합 변화는 주경로 `−61.0830cm/−75.1346°`, 대조군 `−82.8078cm/−74.5765°`다. 성공 조건 회복으로 바꿔 해석하지 않는다.

이 검사는 구현 계약·산술·기존 참조에 대한 오차 계산을 확인한다. 비대상 mask의 물리적 정답, 어두운 경계의 소유권, 참조/카메라의 독립적인 물리 정확성, 자동 공급기 또는 일반화 성능을 인증하지 않는다. 성공 반례2장은 모두 새 입력이 없어서 그대로 보존됐으므로, 양성 입력을 받은 성공 영상의 무손상을 입증하지 않는다. 전체238장 AUTO는 미실행이다.
