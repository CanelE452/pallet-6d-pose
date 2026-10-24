# V219 학습 코드 독립 검토

**코드상 학습 차단 문제는 발견하지 않았습니다.** 실제 소스 배열·이미지·가중치·GT·보호 평가자료를 열지 않았으며 학습도 실행하지 않았습니다. `REVIEW.json`은 frozen runner `11c40cb6` / contract `fc0a161e` / protocol `f3b769da`에만 적용됩니다.

기존 `validate_targets`, `draw_step`, `five_losses`를 AST로 추출했을 때 필요한 전역값은 현재 주입되는 `np/torch/F/require`로 충족됩니다. 새 pool의 키와 클래스 순서는 기존 함수와 같습니다. 실제 순수 함수에 가상 NumPy 값을 연결해 기존 192행 분할(32/32/48/48/32), DEV·미지 계보 제외, synthetic owner의 마스크 적용 후 클래스별 재정규화를 확인했습니다. 16회 가상 추출에서 두 팔의 old192+point32 추출은 완전히 같았고 weak 추출은 별도 RNG를 썼습니다.

손실은 old `[:192]`의 다섯 손실 평균 + point `[192:224]`의 균형 BCE + treatment에만 weak `[224:]` 균형 BCE입니다. detach 없이 한 번의 backward로 연결됩니다. GroupNorm은 행별 계산이고 fitted RGB 통계나 BatchNorm은 없습니다. 이 항목은 코드의 미분 경로 검토이며 신경망 손실·gradient 값을 재현했다는 뜻은 아닙니다. 같은 C61 118359개 파라미터·seed1810 fresh init, RAW 입력, 기존 condition32를 유지합니다. public weak query8 자체가 GT-derived인 별도 SOURCE family라는 권위도 명시되어 있습니다.

실제 실행 전 A의 lineage QA가 SYNTH teacher7040 집합과 새 fit-mask nominal7040 집합의 동일성, PUBLIC/CF의 DEV4 전체 후손 제외, 원 initializer q8/META6를 확인해야 합니다. strong point target과 assistant expected-part weak target의 별도 root admission 및 root recipe/actual entry가 필요합니다. 이는 기존 runner의 검증 전제이며 분모·라벨을 바꾸라는 요청이 아닙니다.

**기록 한계:** probe의 `DEV_pixels_or_predictions_read=0`은 weak DEV PNG와 DEV forward에 관한 설명으로 한정해야 합니다. legacy PUBLIC/CF NPZ 전체를 unpack하므로 제외 행의 픽셀이 물리 메모리에는 들어올 수 있습니다. fit draw·owner 정규화·forward에서의 DEV 제외는 확인됐습니다. root에 이 표현의 한계를 알렸으며 원 producer는 수정하지 않았습니다.

probe는 팔당 최대32회·checkpoint/예측 저장 없이 끝나고, full fit은 다시 fresh init합니다. 두 final checkpoint가 모두 봉인된 뒤 weak DEV 예측을 냅니다. 기존 component role은 바꾸지 않습니다. 두 팔 모두 유지하며 모델·문턱·채택을 고르는 코드는 없습니다.

가상/정적 4그룹 검사: `310ace`, exit0. [검사 결과](INVENTED_NUMPY_AND_STATIC_CHECKS.json), [검사 코드](check_code_and_invented_numpy.py), [정확 바인딩과 한계](REVIEW.json).
