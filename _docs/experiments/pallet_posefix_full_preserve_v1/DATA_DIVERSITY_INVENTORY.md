# 다음 분기용 데이터 다양성 감사

**CANDIDATE_AVAILABLE** — 같은 일반 플라스틱 종류의 single-session32 / multi-session32 후보 식별 가능: True. 학습/추론은 추가 실행하지 않았다.

기존 recording alias/partial overlap과 현재 평가 recording·이미지 SHA를 제외했다. manifest의 실제 후보 이미지 SHA를 다시 검증했다. session 전체 수는 현재 RGB 파일 수, hash/pass는 조사된 고정 pool 범위다. 프레임 번호는 timestamp로 해석하지 않았으며 실내외/주야간 미검증 항목은 UNKNOWN이다.

| session | recording | 전체 PNG | 조사 pool | unique/duplicate | filter pass | eval 제외 | teacher 노출 |
|---|---|---:|---:|---|---:|---|---|
| forklift_v4_173507 | REC_005 | 3729 | 250 | 250/0 | 193 | False | False |
| forklift_v4_174126 | REC_026 | 757 | 250 | 250/0 | 2 | False | False |
| forklift_v4_174342 | REC_008 | 2501 | 250 | 250/0 | 247 | False | False |
| forklift_v4_174925 | REC_013 | 1923 | 250 | 250/0 | 224 | False | False |
| real_unlabeled_day_20260830 | REC_001 | 29028 | 264 | 264/0 | 253 | False | True |
| capturenight01 | REC_019 | 1254 | 110 | 110/0 | 52 | False | False |
| capturenight02 | REC_024 | 782 | 82 | 82/0 | 12 | False | False |
| capturenight03 | REC_020 | 1219 | 108 | 108/0 | 35 | False | False |
| capturenight04 | REC_023 | 1075 | 80 | 80/0 | 6 | False | False |
| capturenight10 | REC_017 | 1474 | 120 | 120/0 | 29 | False | False |
| capturepallet01 | REC_049 | 42 | 9 | 9/0 | 0 | False | False |
| capturepallet10 | REC_028 | 613 | 133 | 133/0 | 21 | False | False |
| capturepallet11 | REC_011 | 1572 | 358 | 358/0 | 94 | False | False |

## 다음 실험 준비의 한계

DAY253은 Replay+self-occlusion PnP target이고 과거 다른 session pool은 Replay만 사용했다. 후보가 있다는 뜻이지 두 target recipe가 이미 matched라는 뜻은 아니다. 새 실험 전에 teacher/target/mask/augmentation/exposure를 통일해야 한다. 현재 고정 필터 통과는 정답 확인이나 clean/occlusion 판정이 아니다. 평가 GT로 session을 고르지 않았다. 초록을 섞어 diversity라고 주장하지 않고 일반 플라스틱끼리32장을 구성했다.

이 밤에는 subset 학습·새 pseudo 추론을 하지 않았으며 candidate identity만 로컬에 기록했다. private image hashes/manifests는 공개 보고서에 포함하지 않는다.
