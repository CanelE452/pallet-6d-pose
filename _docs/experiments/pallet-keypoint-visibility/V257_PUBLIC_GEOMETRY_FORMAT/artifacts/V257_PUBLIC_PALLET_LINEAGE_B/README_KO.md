현재 근거로는 두 자료의 **공식 촬영 계보가 다르게 서술됨**을 확인했습니다. 정확한 촬영 원본 비중복 인증은 UNKNOWN입니다.

Pallox 공식 API/README는 KU Leuven Wu·Demeester의 284프레임 L515 촬영을 lab/hall·세 높이로 설명합니다. [공식 DOI](https://doi.org/10.48804/DJLDFS), [KU 저자 metadata](https://www.kuleuven.be/wieiswie/en/person/00137888). 보호45의 기존 identity receipt는 pinned MR6D의 val/000001:0..44로 연결됩니다. [MR6D 공식 논문](https://openaccess.thecvf.com/content/ICCV2025W/R6D/papers/Gouda_MR6D_Benchmarking_6D_Pose_Estimation_for_Mobile_Robots_ICCVW_2025_paper.pdf) §3.2는 validation을 D435i·30FPS·55-camera VICON 연구 홀 촬영으로 설명합니다. 이 문장은 validation 범위이며 MR6D 전체 센서를 뜻하지 않습니다.

이는 저장소 이름 차이보다 구체적인 서로 다른 촬영 선언입니다. 그러나 pinned 보호45와 Pallox284의 원본 capture-ID 대응표, 공통 알고리즘 checksum 교차표 또는 저자의 명시적 상호 비중복 선언은 확인하지 못했습니다. 따라서 공유 장면을 찾았다고도, 정확히 배제했다고도 주장하지 않습니다.

보호 원본 manifest·사례·RGB·depth·mask·pose·점수는 새로 읽지 않았습니다. 새 훈련 허가·새 라벨·모델/데이터/문턱 선택을 만들지 않았으며, 이 UNKNOWN을 모든 weak 학습의 일괄 금지로 확대하지 않습니다.
