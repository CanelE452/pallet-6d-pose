# 실행 환경의 한계와 확인

CUDA는 기존 process-local userspace 경로에서 정상 사용했다. 시스템 설정,
드라이버 설치, 재부팅, 외부 프로세스 변경은 하지 않았다.

기존 환경의 optional Albumentations는 `ImageCompression(quality_range=...)`
API 불일치로 비활성화되어 있다. 별도 비보호 프로세스에서
`Albumentations(p=1.0).transform is None`을 확인했다.

이번 학습의 reserved-GT read guard는 bytes 파일명을 `Path`로 변환할 때
TypeError를 내는 보수적인 구현상의 한계가 있다. SciPy의 optional import
경로에서 추가 경고가 발생했으며 Albumentations는 마찬가지로 비활성화됐다.
이 경고를 CUDA 실패나 optimizer 실패로 해석하지 않는다. 학습 도중 코드를
바꾸거나 fit을 반복하지 않았다. 필수 stock 증강 및 RNG의 실질적 동일성은
매 업데이트의 실제 합성 tensor/box/keypoint SHA를 기존 같은 seed와 비교해
검증한다. 이 검사 결과는 TRAINING_AUDIT와 각 EXPOSURE/SAFETY_AND_PARITY에 있다.

cuBLAS strict determinism 설정은 기존 프로토콜 그대로이며 관련 경고가
발생한다. 입력 parity를 가중치의 bit-exact 재현성 보장과 혼동하지 않는다.

GT 접근 감사는 Python file-open audit와 source review를 결합한다.
OpenCV의 native image read는 고정한 RGB 이미지 경로만 사용한다.
임의 native I/O까지 감시하는 OS sandbox라고 주장하지 않는다.

최종 감사의 첫 호출은 공통 이름 `report`가 과거 실험 모듈로 resolve되어
중단됐다. 감사 코드에서 이번 파일을 명시적 경로로 import하도록 수정해
재검증했다. 위험도·학습·추론·metric 코드는 수정하거나 재실행하지 않았다.
