"""Record the root's RGB-only visual review, without consulting predictions."""
from collections import Counter
from . import common as C

def run():
    destination=C.DOC/'CLEAN_LOCK.json'
    if destination.exists():
        for b in C.read(destination)['inputs']:C.verify(b)
        print('CLEAN_ALREADY_LOCKED');return
    inventory=C.read(C.RAW/'CLEAN_MEMBERSHIP_PRIVATE.json')
    for b in inventory['bindings']:C.verify(b)
    rows=[]
    # Compact encoding of all individually displayed cells, not a session rule.
    marker_seen=set(range(1,65))|set(range(156,250))
    uncertain_person={135,136}
    for r in inventory['rows']:
        r=dict(r);index=int(r['review_id'][1:])
        if index in marker_seen:
            r.update(EXTERNAL_CLEAN='NO',marker_board='PRESENT',
                evidence='CLI_RGB_VISUAL: attached marker board is visible; excluded by explicit user clarification')
        elif index in uncertain_person:
            r.update(EXTERNAL_CLEAN='UNRESOLVED',marker_board='NOT_OBSERVED',
                evidence='CLI_RGB_VISUAL_NATIVE: person feet overlap or touch upper pallet silhouette; conservative exclusion')
        else:
            assert 65<=index<=134 or 137<=index<=155
            r.update(EXTERNAL_CLEAN='YES',marker_board='NOT_OBSERVED',
                evidence='CLI_RGB_VISUAL: no external foreground occluder or attached marker observed')
        r['reviewer']='ASSISTANT_RGB_ONLY_NOT_MANUAL_COORDINATE_REVIEW'
        r['HARD_VIEW']='LOW_ELEVATION_APPARENT'
        r['TRUNCATION']='NOT_EXHAUSTIVELY_TAGGED_NOT_A_SELECTION_CRITERION'
        r['SELF_OCCLUSION']='NOT_ASSESSED_NOT_EXCLUDED'
        r['eligible_for_primary']=r['used'] and r['EXTERNAL_CLEAN']=='YES'
        rows.append(r)
    locked=[r for r in rows if r['eligible_for_primary']]
    assert len(locked)==78 and len({r['recording'] for r in locked})==4
    assert sum(r['EXTERNAL_CLEAN']=='YES' for r in rows)==89
    policy=dict(user_reply='마커판이 붙은 영상도 clean에서 제외',
        review='All249 original RGB cells on16 sheets; native reinspection of ambiguous people frames. No model/GT/error overlay.',
        labels_are='Assistant visual tags, not user-manual tags or new coordinate annotation',
        eligibility='EXTERNAL_CLEAN=YES AND already in current217; attached markers excluded; two person-overlap ambiguities excluded',
        other_factors='No posthoc elevation/truncation selection; these are not all-visible/easy images. Self-occlusion is allowed.',
        no_prediction_based_selection=True,no_new_training_RGB=True,new_manual_coordinates=0)
    C.save(C.RAW/'CLEAN_REVIEW_PRIVATE.json',dict(rows=rows,policy=policy),True)
    C.save(C.RAW/'CLEAN_LOCKED_PRIVATE.json',dict(rows=locked,policy=policy),True)
    inputs=[C.bind(C.RAW/p) for p in ('CLEAN_MEMBERSHIP_PRIVATE.json','CLEAN_REVIEW_PRIVATE.json',
        'CLEAN_LOCKED_PRIVATE.json','CONTACT_SHEETS_LOCK.json')]+[C.bind(__file__)]
    lock=dict(locked_at=C.now(),locked_before_fit_and_new_scoring=True,inputs=inputs,policy=policy,
        USED217_CLEAN='MIXED',ACCEPTED249_CLEAN='MIXED',clean_locked_count=78,
        clean_locked_recordings=dict(Counter(r['recording'] for r in locked)),
        clean_locked_sessions=dict(Counter(r['session'] for r in locked)),
        accepted_status_counts=dict(Counter(r['EXTERNAL_CLEAN'] for r in rows)),
        used_status_counts=dict(Counter(r['EXTERNAL_CLEAN'] for r in rows if r['used'])),
        excluded_unused_clean=11,manual_coordinates_added=0,
        reference_accuracy='UNVERIFIED_PSEUDO_TARGETS; externally clean RGB does not certify coordinate correctness')
    C.save(destination,lock,True)
    public=C.read(C.DOC/'CLEAN_MEMBERSHIP_PUBLIC.json')
    public.update({k:lock[k] for k in ('USED217_CLEAN','ACCEPTED249_CLEAN','clean_locked_count',
        'clean_locked_recordings','clean_locked_sessions','accepted_status_counts','used_status_counts')})
    public.update(clean_lock=C.bind(destination),review_policy=policy)
    C.save(C.DOC/'CLEAN_MEMBERSHIP_PUBLIC.json',public)
    C.save(C.DOC/'CLEAN_POOL_AUDIT.md','''# Current217 clean 감사 — RGB-only 잠금 완료

1000 후보 →249 승인 →217 실제 TRAIN /512 real 슬롯을 복원했다. 기존123장 사람 난도tag와 승인249장 image SHA 교집합은 0이다. 촬영 세션·주야 meta는 clean 근거가 아니므로 assistant가 예측·GT·error·confidence·보정량 없이 원 RGB249장을 모두 확인했다. 이 판정은 사람이 직접 단 tag나 새 좌표 감독으로 부르지 않는다.

사용자 정정: **“마커판이 붙은 영상도 clean에서 제외”**. 부착 마커가 관찰된 accepted158장/used137장을 제외했다. 사람 하단과 팔레트 상단 윤곽의 겹침이 애매한 2장은 native RGB로 다시 확인한 뒤 UNRESOLVED로 두고 제외했다.

| 집합 | 외부 가림 없는 후보 | 마커판 제외 | 사람 겹침 불확실 | 합계 |
|---|---:|---:|---:|---:|
| accepted | 89 | 158 | 2 | 249 |
| 기존 실제 TRAIN | 78 | 137 | 2 | 217 |

새 primary는 기존 USED217 안의78장/4recording만 허용한다. 이전에 사용하지 않은 clean11장은 넣지 않는다. 세션별78장: capturenight03 29 / capturenight04 6 / capturenight10 27 / capturepallet10 16. 두 불확실 영상의 확인을 기다리지 않고 확실한 subset으로 진행할 수 있다.

**78장이 모두 쉬운 영상이라는 뜻은 아니다.** 저앙각이 많고 일부는 잘려 있다. EXTERNAL_CLEAN과 SELF_OCCLUSION/TRUNCATION/HARD_VIEW는 구분한다. 이번 입력선택 기준은 외부 가림·마커 없음이며, 잘림·시점별 전수수치 tag는 새로 만들지 않았다. 외부 가림 없는 RGB라고 기존 pseudo 좌표의 정확도가 검증된 것도 아니다.

RGB contact sheet30장(accepted16+used14)와 CSV는 private outputs에 유지한다. 공개되는 것은 집계/해시이며 원 RGB를 무심코 push하지 않는다. [CLEAN_LOCK.json](CLEAN_LOCK.json)의 정책·membership hash는 새 fit·평가 전에 잠갔다.
''')
    print('CLEAN_LOCKED',len(locked),lock['clean_locked_sessions'])

if __name__=='__main__':run()
