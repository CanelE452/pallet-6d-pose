# V196 구조 코드 검토

C56/C57은 8채널 96×96 입력을 96→48→48→24→24→12로 축소하고, 4×4 평균 pooling 후 64×3×3을 펼쳐 공통 64차원으로 압축합니다. Global average pooling으로 1×1이 되는 구조는 아닙니다. 3×3 위치 구획과 crop 상대 XY 채널은 남습니다.

입력은 RGB/255, valid, 고정 XY, SAM mask와 availability입니다. C56의 SAM 두 채널은 0입니다. 원 q/L은 crop 생성과 검증에 쓰이지만 실제 좌표값·native ID·끝점 enclosure·끝점 offset은 forward 입력이 아닙니다. 두 모델 모두 같은 encoder와 공통 64차원에서 support2, owner3, direct-end scalar를 출력합니다. 별도 dense 끝점 출력이나 끝점 위치에 맞춘 readout은 없습니다.

Root가 전달한 저장 메타데이터는 끝 enclosure 약 1×1.5 crop 좌표 픽셀, positive interpolation 참여 20–30/9216, q 중심에서 끝까지 0.2–32.4px, 15개 모두 COVER 변화·SHAM 변화 0입니다. 이 수치는 여기서 재측정하지 않았습니다. 보수적 enclosure 크기와 bilinear sample 참여 수는 서로 다른 양이며, 실제 RGB 대비·사람의 가시성·분류 가능한 신호량을 뜻하지 않습니다.

작고 위치가 달라지는 끝 단서가 stride 축소·평균 pooling·공통 특징 압축에서 주변 문맥과 섞일 수 있고, 모델이 query 중심과 다른 위치의 의도한 끝을 RGB에서 찾아야 한다는 가설은 코드와 양립합니다. SAM의 전체 물체 mask와 broad support/owner 과제는 해당 끝의 가시성과 다릅니다. 그러나 convolution이 축소 전에 작은 단서를 부호화할 수도 있고 3×3 위치 정보도 남으므로, 단서가 반드시 소실되거나 목표가 불가능하다는 결론은 나오지 않습니다.

학습된 activation·saliency·gradient·유효 수용영역, 끝 단서의 RGB 대비와 잡음, loss 간 간섭, SAM/pooling의 인과 효과는 측정하지 않았습니다. 체크포인트·배열·이미지·평가 결과를 열지 않았고 forward/fit도 수행하지 않았습니다. 원인 확정, 구조·문턱·모델 선택 또는 HOLDOUT에 따른 최적화는 없습니다.
