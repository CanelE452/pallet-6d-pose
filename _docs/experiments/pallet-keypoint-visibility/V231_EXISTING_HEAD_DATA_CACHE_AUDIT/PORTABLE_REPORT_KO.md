# V231: 기존 학습·공개 데이터·저장 특징 경로 점검

V230 이후 작성된 점검 원본 5개를 추가 공개합니다. **이번 V231은 코드와 실행·데이터 메타데이터를 대조한 기록이며, 새 학습이나 성능 개선 결과는 없습니다.** 최신 고정 실사 모델 비교는 [V229](../V229_FIXED_CONTEXT_REAL/PORTABLE_REPORT_KO.md), 실제 q/r crop 6개와 YCB 8,000행 관측표·검산 코드는 [V230](../V230_YCB_QUERY_OBSERVATION/PORTABLE_REPORT_KO.md)에 있습니다.

## 기존에 실제로 시도한 방법

| 방법 | 사용한 감독·입력 | 확인한 실행과 차이 |
|---|---|---|
| C10 / V110 | YCB associated-r 보임(D)/외부 가림(E) 직접 BCE, frozen ResNet18의 context/local96/center32 | 기존·합성 COVER/SHAM hinge를 함께 사용. 원 q의 표면 owner와 associated-r 상태는 다른 목표 |
| C11 / V111 | C10과 같은 YCB 라벨·loss, frozen DINO context 및 원 q 위치 patch 읽기 | 공간 query 읽기는 이미 시도했음. raw token을 sample한 뒤 Q 정규화 |
| C65 / V219 | 합성 point 6,096개와 공개 실사 finite-part weak FIT45/DEV12 | scratch CNN, crop 외부 RGB 없음. 이 recipe에는 natural YCB 직접 감독 pool이 없음 |
| V227 fixedQ / attention | 공개 MR6D 원 query의 role-weak 368개, FIT65/DEV303 | frozen DINO, 고정 Q와 학습 attention 비교 완료. token별 정규화 뒤 Q sample하여 C11과 순서도 다름 |

[비교 설명](audit/EXISTING_HEADS_KO.md) · [13개 코드·14개 메타데이터 SHA/라인 및 실행 근거](audit/EXISTING_HEADS.json)

YCB 입력 q는 GT reference r에 고정된 4px 이내 jitter를 더한 학습용 emulation입니다. 실제 초기 추론기의 오차 분포가 아니며, YCB logical slot의 팔레트 native 대응은 UNKNOWN입니다. 기존 associated-r S/U는 외부 가림 E로 합치지 않습니다.

따라서 YCB 직접 감독, RGB context, 원 q의 공간 특징 또는 COVER/SHAM을 새로 발견한 방법이라고 부르지 않습니다. 반대로 target·데이터·head·split이 서로 달라 모두 같은 실험인 것도 아닙니다. 여러 목표가 같은 scalar loss에 들어간 사실만으로 실패 원인이 확정되지는 않습니다. 이번에는 과거 gradient·특징·가중치·성능을 재실행하지 않았습니다.

## 이미 확보된 공개 Pallox 자료의 범위

기존 기록상 KU Leuven Pallox 자료(DOI: 10.48804/DJLDFS)는 camera·supplied pose·재구성 mesh·depth·SAM teacher와 284개 fully-joined frame을 갖고 있습니다. 원 역할 분리는 TRAIN_LAB168 / VAL_HALL116입니다. 실제 원 initializer의 q8을 보존한 V94 subset은 100frame·800slot이며 TRAIN44 / VAL56입니다. 전체를 새 TRAIN으로 합치지 않았습니다.

이 자료의 logical native slot과 의도한 물리적 부위의 의미 대응은 아직 UNKNOWN입니다. SAM mask가 없다는 사실을 외부 가림(E)으로 쓰거나, regional weak 신호를 정확한 native-point 정답으로 승격하지 않습니다. 이 불확실성이 모든 weak/AUX 학습을 금지한다는 뜻도 아닙니다. 수량·라이선스·파일 존재는 **기존 메타데이터 기록**을 확인한 것이며 raw payload나 geometry를 새로 검증한 결과가 아닙니다.

[Pallox의 출처·역할 분리·사용 가능한 인터페이스·제한](audit/PALLOX_READINESS.json)

## 저장 특징을 다시 사용할 때의 제한

C10/C11의 특징 cache는 각각 기존 3,652 unit과 추가 YCB 8,000 unit이 섞인 11,652 unit입니다. 기존 부분에 TRAIN/VALIDATION이 함께 있으므로 전체 배열을 먼저 열고 나중에 TRAIN mask를 적용하는 방식으로 재사용하면 안 됩니다. 이번에는 배열 header·값·RGB·가중치를 열지 않았습니다.

검토한 producer는 YCB raw96/context RGB와 전체 context patch token을 일시적으로 만들지만 해당 tensor를 모두 저장하지는 않습니다. 원 RGB 경로·SHA와 저장 descriptor가 있다는 사실을 RGB crop cache가 있다는 뜻으로 해석하지 않습니다. 700frame·1,000instance·8,000slot 역시 독립 component 수를 뜻하지 않습니다.

[캐시 경로 설명](audit/CACHE_ROUTE_KO.md) · [정확한 경로·feature slice·GT sidecar 구분](audit/CACHE_ROUTE.json)

## 다른 사람이 확인하는 방법

위 다섯 원본은 내용을 바꾸지 않고 복사했습니다. [원본 대응·SHA](ORIGINAL_AUDIT_BINDINGS.json)와 [게시 파일 manifest](PUBLICATION_MANIFEST.json)에 크기와 SHA256을 기록했습니다. 이 폴더에서 `python3 -B VERIFY_PUBLICATION.py`를 실행하면 게시된 파일의 바이트 일치를 확인할 수 있습니다. 원본 JSON에 기록된 절대경로는 당시 실행의 출처이며, 검산기는 그 경로를 따라가거나 원 데이터·모델을 열지 않습니다.

이번 게시 작업에서도 현재 평가 223장의 새 추론·학습·문턱 선택·PnP는 수행하지 않았습니다. V229의 미채택 결론과 V230의 관측 진단을 유지합니다.
