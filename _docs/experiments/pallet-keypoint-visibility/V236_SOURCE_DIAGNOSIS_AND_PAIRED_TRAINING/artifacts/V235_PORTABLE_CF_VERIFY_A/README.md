# V235 portable packed-alpha / teacher verifier

Python3 표준 라이브러리만 사용합니다. NumPy/PyTorch·RGB·원 GT·모델이 필요하지 않습니다.

```sh
python3 -B verify_cf_teacher.py \
  --metadata PAIR_METADATA.jsonl.gz \
  --alpha PAIR_ALPHA_PACKBITS.npy \
  --parent-diagnostics SAMPLE_DIAGNOSTICS.jsonl.gz \
  --ycb-reference YCB_KNOWN1521_LABEL_QUERY_MODEL_DIAGNOSTIC.csv
```

metadata/parent diagnostics는 JSONL 또는 gzip JSONL입니다. YCB reference CSV는 선택 사항이지만 제공하면 parent row_id/q/L/D 권위와 r 좌표까지 정확히 대조합니다. CSV가 없으면 YCB의 pair sidecar에 기록된 r만을 teacher 권위로 사용했다고 결과에 명시합니다. SYN r는 원 parent diagnostics와 맞아야 합니다.

고정 대상은5406 parent, 성공5367(SYN4537/YCB830), 실패39입니다. 원 diagnostics의 FIT/available/targetD 부모 집합과 pair_index0..5405를 전수 대조하며 row_id/family/q/L/pixel SHA/valid SHA 문자열을 확인합니다. 실패 pair는 alpha가 전부0이고 proposed_states=null이어야 하며 원 D를 유지합니다. 실패를 E로 바꾸지 않습니다.

Alpha NPY는 magic/version/struct 길이와 ast.literal_eval header로 읽습니다. v1/v2/v3, C-order uint8[5406,2,1152], 정확한 payload byte 수만 허용합니다. Bits는 index=y*96+x의 MSB-first입니다. 별도로 bool bytes0/1을 복원해 저장 alpha SHA와 픽셀 수를 맞춥니다.

원 r의 continuous4tap 및 floor(r+.5) nearest3×3 각각을 원 q/L로 `(point-q)/L*96+47.5`에 옮깁니다. 네 이웃 NW/NE/SW/SE가 모두96 안이어야 하며, 성공 COVER는 모두 opaque/SHAM은 모두 clear이고 SHAM alpha>=64입니다. 저장된 required tap 좌표를 그대로 믿어 검산하지 않습니다. 원 compositor는 np.rint(ties-even) 기록을 유지하지만, 이 checker는 원 teacher half-up과 continuous r를 독립 계산합니다. 검산 실패는 데이터 수정·재라벨을 뜻하지 않습니다.

**한계:** 원 RGB·원 valid bitmap·donor RGB/mask/depth·원 scene GT·모델을 다시 읽지 않습니다. pixel/valid SHA는 parent metadata 문자열과의 대조이며 원본 bytes 재hash가 아닙니다. Donor SHA 원본도 인증하지 않습니다. 원 valid 보존·changed RGB·원본 teacher correctness·natural occlusion·physical native gold·성능/인과를 이 도구가 재인증하지 않습니다. 이 도구가 확인하는 것은 저장된 synthetic opaque layer support와 teacher/input sidecar의 정합입니다.

열리는 파일은 CLI의 명시 인자뿐입니다. metadata 안 donor/원본 절대 경로를 따라가지 않습니다. v112(case-insensitive), pyc/pycache, .. 경로와 symlink를 열기 전에 거절합니다. Root→leaf O_NOFOLLOW를 사용합니다. 결과 JSON stdout/성공 exit0, 오류 JSON stderr/exit1이며 파일을 수정하지 않습니다.

```sh
python3 -B toy_verify_cf.py
```

토이는 임시 디렉터리에 struct로 만든 NPY와 가짜 JSONL/CSV만 사용합니다. 8개 검산군이 PASS했으며 실제 SOURCE pair/alpha/diagnostics/reference 읽기와 모델·학습은0입니다. 실제 패키지는 Root의 별도 승인 실행에서 검증합니다.
