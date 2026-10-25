# 주소 정량 해석과 후속 대조의 경계

48부모 중 COVER가 query central4를 비운 경우는22(FIT17/DEV5), SHAM이 central4에 닿은 경우는15다. 이는 finite ROI 감독과 query-pixel/native-point 상태가 서로 다른 축임을 보여준다. ROI centroid의 query중심 거리는 median17.882px지만, bag centroid는 material point나 GT 좌표오차가 아니다.

동일 FP32 condition32 아래 다른 ROI가 있는16 group은 **모두 같은 nominal witness의 jitter siblings**다. 모든 해당 group에서 기록된 RGB SHA는 서로 다르다. 따라서 이 결과를 같은 crop에서 임의로 다른 부위를 선택했다거나, image를 포함한 입력이 근본적으로 부족하다는 증명으로 읽을 수 없다. condition+valid까지 같은14 group도 RGB는 구분 가능성이 남는다. 알려진201행의 기록된 RGB-SHA+valid+condition signature에서 반대 D/E 충돌은0이다. PNG byte/decoded 중복을 새 검증한 것은 아니다.

ROI 주소/공간표현 문제와 point0 대 finite-part bag의 **타깃 의미 차이**는 별개다. 이번 SOURCE metadata만으로 condition-before-pooling과 separate point/part logit의 성능 우선순위는 확정할 수 없다. Root가 사전 고정한 후속 split-target strong-loss routing 대조는 후자의 의미/gradient 간섭 가설을 좁게 검사하는 데 적절하다. 성공해도 ROI 주소가 새로 주어지거나 원 initializer/native point가 인증된 것은 아니다. ROI를 입력에 추가하지 않고 동일 data·공통draw·양팔 architecture/init·1100 budget을 유지해야 해당 요인을 해석할 수 있다. 새 코드·학습은 이 주소 audit에서 하지 않았다.

접근 범위: A는 허용된 SOURCE 텍스트5개와 자체 생성 표만 읽었고 PNG/NPZ/PT/원GT/모델/보호값 접근0이다. 이는 **A audit에 한정**한다. 마감 시 root로부터 C의 held360 metadata eager-parse incident 존재를 통지받았다. 그 보호값은 A에게 전달되지 않았고 본인 모델 판단에 사용하지 않았으며, V223 전체의 protected-metadata access가0이었다고 주장하지 않는다.
