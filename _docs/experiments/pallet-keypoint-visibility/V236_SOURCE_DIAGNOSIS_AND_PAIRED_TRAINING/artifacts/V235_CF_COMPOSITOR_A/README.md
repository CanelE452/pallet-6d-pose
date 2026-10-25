# V235 in-memory COVER / SHAM compositor

```python
from compositor import compose_counterfactual_pair
result = compose_counterfactual_pair(base, valid, donor_rgb, donor_mask, q, L, reference, row_id)
```

입력: `base uint8[3,96,96]`, `valid bool[96,96]`, `donor_rgb uint8[H,W,3]`, `donor_mask bool[H,W]`, 원 `q/reference float64[2]`, 양의 `L float64` scalar, 원 `row_id str`입니다. 파일·GT·모델을 열지 않는 NumPy 함수입니다. Caller가 SOURCE FIT D 부모와 허용된 FIT donor·다른 물체 조건을 검증해야 합니다. 이 helper는 픽셀로 D/instance/native/role을 추론하지 않습니다.

성공은 `success=True/status=CONSTRUCTED/masked=False`이며 `COVER/SHAM uint8[3,96,96]`, `alpha_COVER/alpha_SHAM bool[96,96]`, 조건부 `proposed_states={COVER:E,SHAM:D}`를 반환합니다. 실패는 `masked=True/failure_reason`이고 두 이미지·alpha·proposed_states는 모두 None입니다. 실패해도 caller는 원 BEFORE와 원 eligibility를 유지합니다. 원 q/L/valid/condition/native는 바꾸지 않으며 새로운 natural occlusion·physical native gold를 주장하지 않습니다.

`donor_plan`은 bbox·seed·size·후보 pool/그 SHA를, `teacherchecks`는 r-nearest3x3·mapped96·required taps와 role별 position/index/alpha pixels/changed RGB pixels/exact_base_RGB_same를 가집니다. `resized_donor_RGB/resized_donor_mask`는 동일 donor/size 검산용 배열입니다. Wrapper는 필요한 이미지/mask를 따로 저장하고 metadata의 NumPy scalar/array를 item()/tolist()로 JSON 변환할 수 있습니다. **큰 사례별 기록에는 전체 candidate_positions_xy를 생략하고 seed·SHA·selected index/position만 남겨도 재구성이 가능합니다.** Caller의 원 input/teacher authority는 별도로 기록해야 합니다.

고정 결정성: seed = `int.from_bytes(sha256(('V235:'+row_id).encode('utf-8')).digest(),'big') XOR 235`. NumPy PCG64로 먼저 max side index를 `[32,40,48,56]`에서 뽑습니다. Donor mask의 nonempty bbox(x1/y1 exclusive)를 crop하고, 긴 쪽을 선택 side에 맞추며 짧은 쪽은 `floor(original*side/max+.5)`/최소1로 반올림합니다. RGB는 half-pixel/border-clamp bilinear float64 후 round-half-up uint8, mask는 half-pixel nearest입니다. 정수 크기 반올림 외 aspect ratio를 유지합니다.

**Reference 검사 전에** COVER256후보와 SHAM256후보를 모두 생성합니다. 각 role마다 x256개 다음 y256개를 replacement로 뽑으며 전체 resized rectangle이96 안에 놓입니다. q/ref/valid는 RNG나 후보 좌표를 정하지 않습니다. 각 role의 첫 acceptance만 쓰고 크기 변경·retry/fallback 탐색은 없습니다.

Teacher math: 원 reference를 `np.rint`(nearest, ties-to-even)한 원본 좌표에서 dy→dx 순서3×3을 만들고 `(sample-q)/L*96+47.5`로 mapping합니다. 각 위치의 floor NW/NE/SW/SE 네 taps(가중치0인 이웃도 포함)가 모두96 안·원 valid여야 합니다. Opaque alpha는 translated donor mask AND 원 valid입니다. COVER는36개 tap 모두 alpha=True, SHAM은 모두False이고 effective alpha>=64입니다. **COVER에 별도의64 gate는 없습니다.** Valid가 다른 위치에서 donor를 자르면 effective alpha mass가 달라질 수 있어 두 role alpha count를 따로 기록합니다. 원 invalid RGB를 고치거나 zero화하지 않습니다.

Alpha pixels와 실제 changed RGB pixels를 구분합니다. Root clarification에 따라 zero changed RGB는 진단으로 기록하며 새로운 acceptance gate나 label 뒤집기를 추가하지 않습니다. 원 D에서 성공한 추가 front layer에 대한 synthetic associated-point supervision이며, original natural E/SELF/native 의미를 재정의하지 않습니다. ROI/GT reference는 teacher side에만 있습니다.

```sh
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B toy_compositor.py
```

Invented array 토이10항목은 same-seed 결과, 입력 보존, coverage/SHAM64, rounding/taps, ref-independent pool, OOF·invalid·empty·budget failure, zero-change 진단을 확인했습니다. 실제 SOURCE/donor/GT/model/보호 자료 읽기 및 학습은0입니다. 실제 조립은 Root의 별도 exact pixel entry 뒤에만 가능합니다.
