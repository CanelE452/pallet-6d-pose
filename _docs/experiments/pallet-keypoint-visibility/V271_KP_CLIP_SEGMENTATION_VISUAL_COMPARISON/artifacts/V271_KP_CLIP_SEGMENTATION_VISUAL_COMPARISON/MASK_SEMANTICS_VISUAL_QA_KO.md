# 실제 mask·키포인트 사례 검토

실사 pixel segmentation GT 없이 저장 RGB·probability·binary mask·cutout과 원 query 점수를 대조했다. 아래는 네 사례의 육안 비교이며 전체 실사 segmentation 순위나 IoU를 뜻하지 않는다.

## F000 · CLEAR_WOOD

세 방법 모두 팔레트의 큰 영역을 찾는다. Surgery/DeCLIP은 내부 틈을 대부분 메운 단일 영역이다. OpenAI에는 굵은 내부 구멍과 우측 외곽 차이가 있다. pixel GT가 없으므로 그 차이를 실사 IoU로 순위화하지 않는다.

보임 G2/native2의 q에서 P(target): OpenAI .491, Surgery .857, DeCLIP .468. Surgery는 D, DeCLIP은 E. 팔레트가 화면상 대체로 분할되어도 개별 외곽점의 판정은 달라진다.

## F046 · DAY_CONE

DeCLIP은 콘과 맞닿는 부분을 더 크게 빼고 콘 하단이 별도 target으로 남지 않는다. Surgery는 콘 발판/하단에 떨어진 target 영역이 보인다. 이 육안의 외부 물체 혼입과 native E 정답은 다른 평가다.

가림 G370/native2는 세 CLIP 모두 E. P(target) OpenAI 9.60e-6, Surgery 6.64e-5, DeCLIP 1.30e-9. Surgery가 일부 콘을 포함해도 이 query는 비대상에 놓여 E가 나온다.

## F105 · SQUARE_CORNERS

DeCLIP은 오른쪽 배경 방향의 번짐이 적어 보이지만 얇은 팔레트 끝도 더 좁게 잡는다. Surgery는 오른쪽으로 더 퍼지며 두 native 외곽점을 포함한다.

보임 G841/native1, G842/native2: Surgery P(target) .976/.994→D, DeCLIP .361/.439→E. G840/native0은 둘 다 D(DeCLIP .997)이며 실패로 세지 않았다.

## F100 · NIGHT_CONE

DeCLIP cutout은 콘을 대부분 빼고 팔레트 쪽을 남긴다. Surgery는 팔레트에서 콘 쪽으로 연결된 넓은 영역을 남긴다. OpenAI도 작은 외곽/콘 혼입이 보인다.

가림 G800/native0는 OpenAI .00590/Surgery .409/DeCLIP 8.60e-7→모두 E proxy. 같은 native를 내는 V270 JOINT는 P(E).323→D라서 CLIP의 공간 마스크와 V270의 슬롯 편향을 구분할 수 있다.

## 공통 결론

CLIP 기반 모델은 팔레트의 큰 영역을 분리하는 데 쓰일 수 있는 공간 정보를 만들었다. DeCLIP은 여기의 콘 장면에서 콘 혼입이 줄지만, 보임 모서리의 mask sample이 낮아져 오판하기도 한다. Surgery가 일부 코너를 더 잘 포함하는 사례도 있으므로 모든 조건에서 DeCLIP이 우수하다고 말하지 않는다.

작은 native 외곽·내부 구멍·가림 경계는32×32 logit의 한계와 목표 차이를 별도로 봐야 한다. 원 query가 마스크 밖이라고 해서 외부 물체가 가렸다고 곧바로 해석할 수 없고, q 주변에 팔레트가 있다고 해서 목표 native가 보인다고 확정할 수도 없다.

V270의 하단 슬롯 E 편향은 별도 문제다. 공간 mask가 콘의 유무에 따라 달라지는 것과 달리 해당 native 슬롯은223장 모두 E였다. 좌표가 좋아 보이는 것만으로 가림 분류가 해결된 것은 아니다.

정량 segmentation GT가 있는 합성768/770에서는 예측들이 board gap을 대부분 채우고,770의 DeCLIP은 적재물/가운데 구멍도 채운다. 높은 영역 IoU가 세밀한 보이는 재료·모서리 경계의 충분한 정확도를 보장하지 않는 직접 예다. 새 학습이나 라벨 수정은 하지 않았다.
