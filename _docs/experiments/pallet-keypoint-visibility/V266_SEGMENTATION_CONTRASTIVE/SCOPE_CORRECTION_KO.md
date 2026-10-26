# V266 실행 범위 정정

V266은 **키포인트 crop의 DINO 특징에 분할 모듈을 추가한 실험**입니다. 사용자가 제안한 **전체 이미지 CLIP 기반 분할은 실행하지 않았으며, 그 제안의 실패 근거로 해석할 수 없습니다.**

또한 V266의 target/foreign/background 소유 ID는 별도 진단용 합성440개 buffer에 있는 것입니다. 사용자의 원래 합성 데이터의 visible-pallet mask와 구분해야 합니다. 팔레트 visible mask만으로도 전체 이미지의 대상/비대상 이진 분할 학습은 가능합니다.

[범위·자료 출처 정정 상세](../V266_FULL_IMAGE_SCOPE_CORRECTION/README_KO.md)

이 정정으로 기존 모델·예측·수치·원 주석은 변경하지 않았고 새 학습은 없습니다.
