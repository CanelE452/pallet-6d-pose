새 API fixture는 원래 물리 Q, 현재 nonS 크기→fit 공분산, 관측 q/Σ 불변성, S raw 삭제→특징→솔버→readout 연결을 검증하는 적절한 regression이다. S perturbation은 생성한 prompt audit까지 불변성을 확인한다.

다만 검은 합성 이미지, injected features/classifier, ones mask, 수제 모델, 제한된 starts를 사용한다. 실제 Tiny/SAM provider는 호출하지 않으며 실제 B/I 코너 교정, 새 learned CAL 적용, 실제 O/T joint 지역 활용 및 이미지 성능 검증을 대신하지 않는다. 현재 fixture를 fullfresh production RGB 증거로 부르면 안 된다.

숫자 연결 문제는 없다. API의 λ0 사전항 역할 표시와 domain0의 O/T solved flag는 잘못될 수 있어 root 소유 metadata 수정을 요청했다. 정상 CAL MODEL에 provenance 키가 없어서 API 모델 binding이 빈 객체가 되는 점도 보고했다. 동결 모델 파일은 변경할 필요 없다. 최종 전체 API 실행·실험 seals·controls·개선 목표 달성은 계속 별도 검증 대기다.
