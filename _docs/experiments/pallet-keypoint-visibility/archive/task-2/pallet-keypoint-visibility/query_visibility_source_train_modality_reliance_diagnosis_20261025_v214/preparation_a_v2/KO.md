V1 pre-execution root 검토에서 binding 전체 dict 비교가 size_bytes/kind 부가 표기 차이를 거부하는 문제가 발견됐다. 실제 payload·가중치를 열기 전의 metadata gate 문제다. V1은 모두 그대로 보존했다.

V2는 payload의 정확 경로와 SHA256만 identity로 비교하고 부가 annotation은 제외한다. root의 59개 whitelist 전체 목록은 그대로 요구한다. 서로 다른 부가 필드의 actual-shaped text fixture를 production payload_file에 통과시키고, 같은 경로의 잘못된 SHA 및 미등록 경로가 파일 접근 전에 거부되는 것을 확인했다. 선택/whitelist 파일은 V1 byteexact이고 다른 모든 기존 함수 AST는 동일하다. 새 output 경로는 source_train_modality_diagnosis_v2다.

7040/6096/944·weak44·6조건·원 q/L/native/U·라벨·127상수·factual replay1e-4와 sign exact·selected-TRAIN NPY 처리·원모델/threshold는 바꾸지 않았다. 실제 source/PNG/NPZ/weights/score 및 forward는 아직0이다.
