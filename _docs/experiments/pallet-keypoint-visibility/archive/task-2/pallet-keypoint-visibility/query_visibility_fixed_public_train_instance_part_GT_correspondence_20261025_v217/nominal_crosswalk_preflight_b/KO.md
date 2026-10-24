# V217 native·instance·부위 조인 코드/텍스트 검토

선언된 순서는 `dynamic[k] = fixed[perm[k]]`이며 한 번만 적용한다. 이미 dynamic인 q8/저장 좌표를 다시 재배열하거나 GT와 가까워지도록 permutation을 고르지 않는다. native0/1/4/5는 nominal +Z deck 쪽, 2/3/6/7은 −Z bottom-runner 쪽 역할이지만, 이는 실제 crop에서 그 부위가 보인다는 라벨이 아니다. rear라는 이름도 SELF를 뜻하지 않는다.

V153 `prepare()` 35–54행은 scene/frame의 모든 obj1 GT-list index와 각 mask 경로를 보존한다. `factual()` 168–182행은 `gt[frame_key][ix]`, info, camera/depth scale 및 같은 `ix`의 full/visible mask를 조인하고 native 의미를 `UNASSIGNED`로 남긴다. 검출된 대상이 어느 GT instance인지 선택하는 코드는 아니다. 같은 obj1의 다른 팔레트도 다른 instance이며, mask 파일의 GTID는 object class ID가 아니다.

A의 봉인 schema는 selected index null과 `UNKNOWN_NO_GT_BEST_OR_FIRST_INSTANCE_FALLBACK`을 유지한다고 전달됐다. 본 검토는 그 새 산출물을 재개방하지 않았다. 기존 3개 공유 텍스트만으로 고정12/96의 per-query intended-part 대응이 확정되지는 않는다.

다음 실제 점검에서는 원 row/source/native/q/L/U·FIT/DEV를 그대로 두고, (1) 명시적 instance 대응, (2) 한 번 적용한 nominal native→canonical과 expected part, (3) 원 q와 nominal reference의 crop 내 관계, (4) full/visible mask·depth support·겹침을 별도 열로 기록하면 된다. 전역 silhouette의 양성은 어느 target 표면일 뿐 intended part 증명이 아니며, 음성·depth hole·query OFF·다른 물체는 자동 E가 아니다. 같은 intended part의 양성 foreground 근거가 없는 경우 native 상태를 UNKNOWN으로 남긴다. 그렇다고 완벽한 mesh/코너 인증을 모든 약감독의 blanket gate로 요구하지 않는다.

보존된 nominal part 제안과 기존 mask 진단은 재사용할 수 있다. 아직 열지 않은 정확 포인터를 JSON에 기록했다. 이번 단계에서는 GT/카메라/마스크/depth/CAD/이미지/배열 파일의 stat·hash·read·decode, 실제 투영, 모델/점수·보호셋 접근, 새 라벨·admission·학습이 모두 0이다. 원 pose projection은 별도 actual entry 이후 불확실한 SOURCE nominal target 근거로 사용할 수 있으며, 배포 입력이나 q 교정으로 쓰지 않는다.

- [검토 JSON](REVIEW.json) · [8개 native 조인 표](NATIVE8_NOMINAL_CROSSWALK_CONTRACT.csv) · [가상/텍스트 검사](PURE_AND_TEXT_CHECKS.json)
- [현재 scope](../ROOT_PROSPECTIVE_FIXED12_PUBLIC_GT_CORRESPONDENCE_AUDIT_SCOPE.json)
- [V159 순서·부위 정의](../../query_visibility_native_part_support_feasibility_20261024_v159/native_definition_audit/DEFINITION_AND_FEASIBILITY.json)
- [V154 원 query/instance 약감독 경계](../../query_visibility_mr6d_source_supervision_bridge_20261024_v154/ROOT_SUPERVISION_SEAL.json)
- [V153 intake 코드](../../query_visibility_mr6d_full_public_source_20261024_v153/full_intake_preparation/intake_full_source.py)
