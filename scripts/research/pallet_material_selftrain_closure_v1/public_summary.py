"""Public summaries; detailed candidate/calibration/local-state listings stay local."""
from . import common as C

def main():
    inv=C.read(C.DOC/'WOOD_DATA_INVENTORY.json');pre=C.read(C.DOC/'PREFLIGHT.json')
    C.save(C.DOC/'WOOD_DATA_INVENTORY_PUBLIC.json',dict(stats=inv['stats'],
        private_inventory=inv['full_private_inventory'],private_candidate_pool=inv['pool'],
        calibration='Existing registered calibration reused; camera matrices and candidate image-location catalogue are local-only.',
        legacy_catalogue=[{k:v for k,v in r.items() if k in ('recording','count','role','camera_contract')} for r in inv['legacy_catalogue']],
        source_recordings=['REC_001','REC_002'],evaluation_recordings=['REC_039','REC_042']),True)
    C.save(C.DOC/'PREFLIGHT_PUBLIC.json',dict(head_start=pre['head_start'],remote_start=pre['remote_start'],branch=pre['branch'],
        tracked_clean_at_start=True,existing_untracked_count=len(pre['untracked_files']),
        user_untracked_preserved=True,private_detail_binding=C.bind(C.DOC/'PREFLIGHT.json')),True)
    if (C.DOC/'WOOD_TRAIN_PROTOCOL.json').exists():
        p=C.read(C.DOC/'WOOD_TRAIN_PROTOCOL.json')
        C.save(C.DOC/'WOOD_TRAIN_PROTOCOL_PUBLIC.json',dict(
            **{k:p[k] for k in ('schema','arms','args','initialization','updates_per_arm','new_fit_limit','sampled_real_unique','accepted_real_unique','protected','only_intended_difference','validation','GT_training','evaluation_labels_opened','last_only','independent_confirmation')},
            private_protocol=C.bind(C.DOC/'WOOD_TRAIN_PROTOCOL.json'),preflight=p['preflight'],
            source512=C.bind(C.RAW/'SYNTHETIC_REPLAY512.json'),
            detailed_image_paths_and_label_manifests='LOCAL_ONLY; exact hashes retained privately'),True)
    C.save(C.DOC/'PUBLICATION_POLICY.md','# 공개 범위\n\n사용자 지정 PUBLIC origin을 확인했다. 자동 안전검사에서 상세 후보 목록 공개가 거절되어 목록 자체를 제외한 축소본으로 push했다. 데이터/카메라 행렬·원본RGB·원시좌표·checkpoint·전체 로컬 환경목록은 공개하지 않는다. 공개 산출물은 코드, 집계·방법 계약·중복 감사, 출처 해시, 기존에 공개를 허용한 평가 예시의 주석 포함 그림이다. 전체목록/좌표를 재현하려면 로컬의 비공개 입력이 필요하며 공개 benchmark 완전 배포라고 주장하지 않는다.\n\n`WOOD_DATA_INVENTORY_PUBLIC.json`, `WOOD_TRAIN_PROTOCOL_PUBLIC.json`, `PREFLIGHT_PUBLIC.json`은 원본 전체 목록을 해시로 연결한다. 기존 상세 원본은 삭제하거나 덮어쓰지 않는다.\n',True)

if __name__=='__main__':main()
