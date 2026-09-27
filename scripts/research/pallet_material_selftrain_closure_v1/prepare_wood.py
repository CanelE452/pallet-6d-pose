"""Lock material provenance and metadata-only Wood populations before inference."""
from collections import Counter
from . import common as C

def main():
    inv=C.read(C.RAW/'WOOD_INVENTORY_AGENT.json')
    for b in inv['sources']:C.verify(b)
    met={r['id']:r for r in C.read(C.P.META)}
    rows=inv['proposed_main_evaluation_records'];pool=inv['candidate_records']
    assert len(rows)==45 and len(pool)==1000
    for key in ('teacher_vs_main45','selected_candidates_vs_main45'):
        assert all(not v for v in inv['overlaps'][key].values())
    metadata=[dict(met[r['id']],image=r['image'],recording=r['recording'],recording_group=r['recording'],severity=r['severity'],session=r['session']) for r in rows]
    assert all(r['xyz']==[.8,.14,.59] for r in metadata)
    C.save(C.RAW/'EVAL_METADATA.json',metadata,True)
    C.save(C.DOC/'EVAL_POPULATION_LOCK.json',dict(status='LOCKED_BEFORE_NEW_PREDICTIONS_AND_SCORING',n=45,
        records=rows,metadata=C.bind(C.RAW/'EVAL_METADATA.json'),recordings=['REC_039','REC_042'],
        severity=inv['stats']['eligible_severity'],severe_available=False,
        rule=inv['population_rule'],excluded_recordings=['REC_001','REC_002'],excluded_images=71,
        independent_confirmation=False,reference='Existing legacy/mixed-provenance annotation. No trusted visible corners and no independent physical6D.',
        metadata_access='Inventory opened annotation containers only for camera/provenance fields; coordinates not used. New inference loads sanitized metadata only.'),True)
    C.save(C.DOC/'POOL.json',dict(records=pool,sources=inv['sources'],rule=inv['candidate_rule']),True)
    C.save(C.DOC/'WOOD_DATA_INVENTORY.json',dict(stats=inv['stats'],session_summaries=inv['session_summaries'],
        full_private_inventory=C.bind(C.RAW/'WOOD_INVENTORY_AGENT.json'),pool=C.bind(C.DOC/'POOL.json'),
        source_aliases=inv['session_aliases'],legacy_catalogue=[{k:v for k,v in r.items() if k!='images'} for r in inv['legacy_raw_catalogue']],
        hash_verification=inv['hash_verification'],sources=inv['sources']),True)
    C.save(C.DOC/'WOOD_PROVENANCE_AUDIT.json',dict(overlaps=inv['overlaps'],teacher_records=inv['teacher_records'],
        teacher_budget=dict(images=9,corners=38,wood_images=6,wood_corners=23,plastic_images=3,plastic_corners=15),
        accepted_pool_pending=True,train_candidate_eval_exact_ID_and_SHA_overlap_zero=True,
        teacher_eval_recording_overlap_zero=True,train_candidate_eval_recording_overlap_zero=True,
        teacher_candidate_recording_overlap=['REC_001','REC_002'],
        teacher_candidate_overlap_interpretation='Teacher and student share source recordings; not additional manual labels. Both RAW/REF receive identical teacher-based selection.',
        evidence_status='REUSED_DEV_ONLY',sources=inv['sources']),True)
    C.save(C.DOC/'WOOD_EVAL_ELIGIBILITY.md','# Wood 평가 적격성 고정\n\n기존 Wood116장 중 교사 감독 촬영 REC_001·REC_002의71장을 scoring 전에 제외했다. Plastic night도 Wood night와REC_002를 공유하므로 함께 제외한다. 성능에 따른 제외가 아니다.\n\n주 평가45장: REC_03925장, REC_04220장. Clean38·Moderate7이며 Severe는 없다. 교사/학생 후보와 평가의 exactID·SHA·recording 중복은0이다. 이들은 과거 연구에서 사용한 DEV이며 새 독립확인은 아니다.\n\n직접 클릭 출처를 검증할 수 있는 가시점은0개다. 따라서 Q1_WOOD는UNRESOLVED이고, 2D/6D 비교는 기존 legacy reference 조건에 한정한다. 동일 raw/corrected 학생 비교는 가능하나 실제 물리적6D 정답을 확보한 실험이라고 쓰지 않는다.\n\n모델과 무관한 기존 material review의 Wood13,908장 중 기존319장·교사와SHA가 겹치는80장을 제외한13,828장에서 후보1,000장을 선택한다. Day675/Night325를 비례 할당 후 정렬ID 중간점 간격으로 선택했다. 나머지 미사용 영상·평가 recording 원본·보정 불명확한 portrait225장은 추가 학습하지 않는다.\n',True)
    print('WOOD_PROVENANCE_LOCKED',len(rows),Counter(r['recording'] for r in rows),flush=True)

if __name__=='__main__':main()
