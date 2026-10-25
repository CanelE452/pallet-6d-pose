# V249 고정 SOURCE RGB-reference oracle 진단 — 준비만 완료

Root true entry 전에 실행하지 않는다. 입력은 원canonical 특징NPY, 원samples NPZ의5개 nonpixel member, 기존YCB1521CSV의14개 지정열뿐이다. 원RGB/GT/weight/model을 읽거나 새forward·학습을 하지 않는다.

계산 API는 `run_diagnostic(metadata_rows,features,samples)`이고 I/O나 Root 경로에 의존하지 않는다. `write_results(fresh_output_directory,result)`로 같은CPU결과를 저장할 수 있다. Root 공개wrapper는 explicit public-repo dependency map에서3개 SOURCE파일만 검증해 이 API를 호출하면 된다. publicwrapper는 실제진단의 새선행조건이 아니다.

SOURCEFIT available D의 저장descriptor를 sample_index 순서로FP64평균한 뒤L2한다. nominal objectID는 teacher-oracle이고, wrong은 숫자정렬FIT objectID의cyclicnext다. DEV397과3views/모든고정strata·object를 보고하고 지원부족은UNKNOWN으로 보존한다. E방향은−targetcos/−margin으로 사전고정하며 wrongcos는분포만보고한다. 원q/r/crop/라벨/역할은 변경하지 않는다.

Root scope는 `{approved:true,entry_contract:<완성entry에서root_scope만뺀사전>}`으로 만든다. true entry에그scope binding을넣는다. 이진단은배포가능reference 선택·nativegold·원initializer전이·causal증거·모델채택을뜻하지 않는다.
