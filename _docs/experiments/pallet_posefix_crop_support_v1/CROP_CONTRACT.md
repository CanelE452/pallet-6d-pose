# Crop 계약

진행 상태: E0 gate 실패로 학습/scale-sanity/새 추론은 실행하지 않았다. 아래 real/source 재구성·corruption 보존은 C1 진행 시 필요한 계약 설계이며, 이번 실행에서 완료한 구현으로 주장하지 않는다. 실제 구현·검증한 범위는 공통 배율 설정과 E0 crop 기하이다.

실제 axis_aligned_crop_matrix(box,input_shape=(384,288),expansion=1.25). xyxy 중심 고정,3:4 aspect 조정 후배율. 새 wrapper의 EXPANSION=1.50이 유일한 실험값이며 기존 함수/전역값 수정 금지. 입력 H384×W288, logits K9×H96×W72. zero-based expectation×4, 연속 출력 지지영역 x[0,284],y[0,380]. inverse matrix로 원영상 변환.

GT crop mask는 x[0,288),y[0,384)로 출력 지지영역보다 조금 넓음. 바깥 target은 target_distribution에서 마스킹, edge clamp label 생성하지 않음. native channel 순서유지, center8와 invalid 입력·bbox·score pass-through. 평가만 허용 whole-object permutation 사용.

실사: 기존253 paired entry의 원영상 RGB/가림 plan/cleanR0/OCCR0/target을 재사용하고 clean/OCC support 교집합 재계산. source: 기존 prepared RGB/cache native GT와 R0 bbox를 같은 wrapper로 변환. corruption의 기존 원영상 identity/변위를 보존하고 crop변화로 newly-supervised 코너가 생겨도 RNG소비를 변경하지 않는다.
