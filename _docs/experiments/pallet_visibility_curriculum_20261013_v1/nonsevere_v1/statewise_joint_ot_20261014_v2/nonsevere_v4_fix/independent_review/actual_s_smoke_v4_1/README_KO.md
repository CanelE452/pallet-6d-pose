# 실제 RGB에서 S 원시 좌표 불변성

**3개 실제 CAL 이미지의 원본/변경 쌍, 총 6회 full API 실행이 모두 통과했다.** 선택된 λ1·β1, 최종 CAL 모델, 현재 bias API SHA965bcf26…, 실제 Tiny/MobileSAM provider와 전체 56/24/250/256 search 제한을 사용했다. 단일 CUDA 프로세스와 CPU·interop 각 1 thread로 실행했다. 마스크·특징·분류기 주입은 없었다.

변경한 값은 고정된 초기 geometric S의 원시 좌표뿐이다. nonS 관측, 9번째 center, 이미지, K, native 모델과 크기, coarse 자세, Q 및 모든 설정은 유지했다. 원본 S metadata를 비교에서 제외하면 **전체 runtime이 정확히 동일했다.** 실제 provider의 두 마스크 내용 SHA, 프롬프트, quality, 상태와 query ID, B/I 좌표·관측/fit 공분산, O/T 영역, nonS scale, 최종 자세와 readout도 모두 같았다.

| CAL 사례 | S IDs | B/I | O/T | 최종 상태 |
|---|---|---:|---:|---|
| night08:1779449488068910592 | 6,7 | 2 | 3 | accepted, ambiguous family |
| outside:1778651519235162880 | 7 | 6 | 1 | accepted |
| wood_184309:000432 | 7 | 7 | 0 | accepted, native/reference 대응 미호환 사례 유지 |

각 호출의 S query·prompt·보정·measurement·영역·solver image/region term 0을 확인했다. 독립 후검산에서는 B/I 관측 30개, 개별 O 영역 8개, S readout 8개, 깊이 정점 48개, B/I 최종 모델 가시성·경계 30개가 통과했다. Q→물리 R, rigid projection 및 S readout 재계산의 최대 차이는 모두 0이었다. source 247개(실행 의존 로컬 코드·MobileSAM vendor 포함)의 현재/스냅샷 SHA, 입력·모델·이미지 21개 SHA, 사용자 보호6 SHA가 실행 전후 및 후검산에서 일치했다.

이 검증은 **현재 초기 geometric S 제외 경로**에 대한 것이다. 실제 human SELF를 B/I로 잘못 분류하는 한계, 모든 물리 O/T의 해 해결, 전역 유일성 또는 성능 목표 달성을 증명하지 않는다. 미호환 CAL 사례를 선택한 것은 stress inventory이며, runtime에 GT 대응 수정이나 호환성 분기를 넣지 않았다. annotation GT, TEST 파일·점수 및 자세 정확도 지표는 읽거나 계산하지 않았다.

실행 명령은 다음과 같다. 재실행 시 새 output 디렉터리를 지정해야 하며, 기록된 환경·선택 모델·입력 자료가 필요하다.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python \
_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/independent_review/run_actual_s_smoke_v4_1.py
```

증거는 `PROTOCOL.json`, `INPUT_RECORDS.json`, `code_snapshot/`, `PREDICTIONS.jsonl.gz`, `PREDICTION_SEAL.json`, `AUDIT.json`, `INDEPENDENT_POSTSEAL_CHECK.json`이다. 예측 봉인을 만든 뒤 쌍 비교 및 독립 scalar 후검산을 수행했다.
