# 공개 SOURCE TRAIN 데이터의 실제 범위와 감독 권한

기존 공개 팔레트 입력은 **이미 MR6D**입니다. 원 다운로드는 [anas-gouda/mr6d](https://huggingface.co/datasets/anas-gouda/mr6d)의 커밋 `3c4ed8453312655499c3d3a3cf38b4929fae77ee`에 고정되어 있고, 이번 root 저장소 인덱스와 같은 리비전입니다. 새 데이터셋을 확보했다는 해석은 정정합니다. 새 인덱스는 출처를 다시 확인한 메타데이터입니다.

| 원 component TRAIN 풀 | 보존 행 / 입력 U | 원본 단위와 중복 | 학습 감독 |
|---|---:|---|---|
| SYNTH_AUG | 56,320 / 512 | 110 bases, 440 images, 110 components; SELF 52,800 + BEFORE anchor 3,520 | 기존 합성 support/owner와 별도 direct 약한 끝 감독. 이는 원 TRAIN 전체이며 V202 fit eligibility와 구분 |
| PUBLIC | 1,240 / 161 | 155 frames, 4 source prefixes, 103 components | owner 보조 known239행/81frames. PUBLIC support2/native D/E/quality 감독 없음 |
| CF | 3,816 / 360 | 159 pairs, 46 source frames, 477 variant images, 1 component | known owner314. COVER FOREIGN/SHAM TARGET; native external gold로 변환하지 않음 |
| DIRECT | 2,784 / 112 | known109triplets: synthetic82/41bases + public-real27/12frames; 합계42components | known327행(D218/E109). 실사81행은 SOURCE_RGB_SEMANTIC_END_WEAK |

PUBLIC 원본은 `val/000000` 65, `val/000002` 38, `val/000005` 24, `val/000006` 28프레임입니다. `val`은 원 데이터셋의 공개 경로 이름이며, 여기서 센 것은 기존 SOURCE_DEVELOPMENT 중 immutable component TRAIN만입니다. 보호된 기존 publicheld `val/000001`의 RGB·GT·score 본문이나 current 평가를 열지 않았습니다. Intake의 전체 경로 목록은 출처 메타데이터로만 읽었습니다.

155 TRAIN 프레임에는 intake manifest 기준 obj1 GT-list 관측 220개가 기록되어 있습니다. 이것은 프레임별 후보 인스턴스 관측 수이며, 서로 다른 물체 220개 또는 쿼리가 선택한 인스턴스 220개라는 뜻이 아닙니다. 현재 PUBLIC 입력 스냅샷에는 selected GT instance/obj_id/part 대응 필드가 없어서, 각 native query의 물리적 부위 대응은 UNKNOWN입니다. 직접 감독의 12 실사 프레임은 전부 PUBLIC 풀과 겹치며 한 연결 성분에 속합니다. 27triplets를 독립 자연 영상 27장으로 세지 않습니다.

YCB-V는 **59 cutouts/59 frames/59 declared instances/7 scenes/10 object IDs**의 외부 가림물 공여자입니다. 현재 파이프라인에서 팔레트 target 또는 native8 가시성 감독 데이터가 아닙니다. PUBLIC owner와 CF owner 보조 감독, public-real finite-end 약한 D/E/D 감독을 분리해야 합니다. native6/7 known direct0만으로 canonical 부위 누락이나 관련 D/E 실패를 단정하지 않습니다.

기존 로컬 원본 payload 경로는 V153 `full_source_v1/original/` 아래이며 원 receipt에 RGB/depth/full mask/visible mask/obj1 mesh 다운로드가 기록되어 있습니다. 선언된 payload namespace는 `models`와 `val`뿐입니다. 이번 감사는 원 asset 존재·해시·header·pixels를 새로 읽지 않았고, payload availability는 기존 receipt의 선언입니다. 입력·감독 캐시는 [기존 바인딩과 집계](SUMMARY.json), [TRAIN 프레임 목록](PUBLIC_TRAIN_FRAME_INVENTORY.csv), [공여자 목록](YCB_TRAIN_DONOR_INVENTORY.csv), [direct TRAIN 목록](KNOWN_DIRECT_TRAIN_ATTEMPTS.csv)에 보존했습니다.

추가 dynamic/mr/odyn과 기존 val의 상위 촬영 대응은 확인한 intake 계보에 없어 UNKNOWN입니다. 새 영상 도입 전 공유 촬영/중복을 확인해야 하며, 같은 저장소라는 사실로 독립성을 인정하지 않습니다. 이는 이미 사용 가능한 TRAIN155프레임에서 추가 약한 끝/상태 관찰을 준비하는 작업을 막는 조건은 아닙니다. 관측 권한과 기존 q/L/native/U를 유지해 감독을 보강할 수 있고, 완벽한 mesh/depth 인증을 새 전제조건으로 만들지 않습니다.

다음 안전한 실제 단계는 **기존 TRAIN 내에서 direct12프레임을 제외한 고정 프레임의 q8·끝·상태 대응을 독립 관찰**하는 것입니다. 새 선택은 root의 별도 prospective 규칙 뒤에 수행하며, 이번 감사에서 새 프레임·라벨·학습 admission을 정하지 않았습니다. 원 q/L/sampler·라벨·모델은 변경하지 않았습니다.

출처: [원 MR6D provenance](EXISTING_MR6D_PROVENANCE.json), [inventory COMPLETE](COMPLETE.json), [provenance COMPLETE](MR6D_PROVENANCE_COMPLETE.json). 행/이미지/증강/연결 성분은 구분했으며 연결 성분은 통계적 독립성 인증이 아닙니다.
