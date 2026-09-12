# Cache와 GT 접근의 범위

선 teacher의 원본 예측은 이번 감사 전에 생성된 GT-free 캐시다:

- calibration: 이전 gain-selector의 `calibration_Q.json` 중 `raw_line`/`ambiguity`
- synth_val: corrected-v2의 `hough_seed1_synth_val.json` 중 같은 필드

두 원본의 checkpoint SHA와 prediction SHA는 PROTOCOL_LOCK에 고정했다.
새 pseudo-line 캐시는 이 원본의 선/ambiguity/available와 deterministic weight만
복사한다. 기존 Q point는 pseudo-keypoint로 사용하지 않는다.

새 캐시의 `GT_opened:false`는 **teacher prediction 생성 경로의 provenance**를
뜻한다. 캐시를 내보낸 mechanism 감사 프로세스 전체가 GT를 읽지 않았다는 뜻은
아니다. 감사 프로세스는 합성 GT를 명시적으로 열어 품질과 gradient를 평가했다.
캐시에 GT 좌표/label/GT-derived selection은 저장하지 않았다.

실사 GT annotation 접근은 별개로0이다. adaptation pool과 평가군의 중복 감사는
membership metadata와 RGB hash만 사용하고 GT annotation 경로는 열지 않았다.
학생 training 자체는 gate 실패로 실행하지 않았다.
