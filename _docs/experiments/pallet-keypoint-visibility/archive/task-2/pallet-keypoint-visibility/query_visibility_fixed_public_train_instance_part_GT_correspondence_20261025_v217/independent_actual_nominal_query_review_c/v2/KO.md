# V217 nominal 후보 projection 독립 QA

PASS. scalar Cramer inverse·4법선 cosine·pinhole·front/rear 순서를 독립 구현해 15 mapping과 120 JSON/CSV 전량을 대조했다. 원96행/U11/FIT64·DEV32와 후보 중복U17을 보존했다. 원 instance는 모두 미선택이며 새 라벨·학습 채택은 없다.

명목 reference와 available q의 변위는 103개 후보행 중 3px 이하 2개다. 이는 후보가 반복된 기술 통계이며 물리 좌표 정답·최적 instance 선택·모델 오류율이 아니다. 원 q를 이동하거나 입력을 recrop하지 않았다.

저장 정합 canvas scalar로 별도 in-bounds 기술만 추가했고 source PNG/원 GT/배열/모델은 열지 않았다. 실제 instance·part·visibility 및 nominal reference 불확실성은 별도 근거가 필요하다.
