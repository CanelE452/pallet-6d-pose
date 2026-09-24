"""Versioned fixed-ID rescore after genuine human QA; frozen caches only."""
import copy
import math
from collections import Counter
import numpy as np
from . import common as C
from .complete_directive import QA
from .review_saved_keypoints import REVIEW
from .evaluate import ARMS, PAIRS, metrics, point, difference


def put(path, obj):
    if path.exists():
        assert C.read(path) == obj, f'Immutable output differs: {path}'
    else:
        C.save_new(path, obj)


def main():
    old_names = ('VERIFIED_RESULTS.json','MODEL_COMPARISON.json','REFERENCE_DISAGREEMENT.json',
                 'FINAL_DECISION.json','VERIFIED_LABELS_PUBLIC_SUMMARY.json','EVALUATION_REPORT_KO.md')
    historic = {n:C.sha(C.DOC/n) for n in old_names}
    lock = C.read(QA/'INPUT_LOCK.json')
    selection = C.read(REVIEW/'ANCHOR_SELECTION.json')
    assert C.sha(REVIEW/'ANCHOR_SELECTION.json') == lock['selection_sha256']
    first = C.read(REVIEW/'FIRST_PASS_SNAPSHOT.json')
    assert C.sha(REVIEW/'LABELS.json') == lock['first_pass']['labels_sha256']
    assert C.read(REVIEW/'LABELS.json') == first
    assert C.sha(C.DOC/'VERIFIED_RESULTS.json') == lock['original_results_sha256']
    qa = C.read(QA/'LABELS.json')
    queue = C.read(QA/'QA_QUEUE_PRIVATE.json')['points']
    expected = [[r['frame_index'],r['corner_id']] for r in queue]
    assert len(expected) == 2 and qa['review_queue'] == expected
    assert len(set(map(tuple,expected))) == 2
    final = copy.deepcopy(first)
    decisions = []
    for fi,ci in expected:
        old = first['frames'][fi]['corners'][ci]
        new = qa['frames'][fi]['corners'][ci]
        decision = new.get('qa_decision')
        assert decision in ('KEEP','CHANGE_STATUS','RECLICK'), 'WAITING_FOR_HUMAN_METADATA_QA'
        assert C.valid_corner(new,selection['frames'][fi]['hw'])
        if decision == 'KEEP':
            assert all(new.get(k)==old.get(k) for k in ('status','xy','input_xy'))
        final['frames'][fi]['corners'][ci] = copy.deepcopy(new)
        decisions.append(dict(qa_item=len(decisions)+1,corner=ci,decision=decision,
                              before=old['status'],after=new['status'],coordinate_changed=old['xy']!=new['xy']))
    for fi,frame in enumerate(first['frames']):
        assert qa['frames'][fi]['frame_id']==frame['frame_id']
        for ci,corner in enumerate(frame['corners']):
            if [fi,ci] not in expected:
                assert qa['frames'][fi]['corners'][ci] == corner
    for s in selection['frames']:
        assert C.sha(C.ROOT/s['image']['path']) == s['image']['sha256']
    final['reference_version']='VERIFIED_VISIBLE_ANCHOR_FINAL_V2'
    put(QA/'VERIFIED_LABELS_FINAL_PRIVATE.json',final)
    prediction_lock=C.read(C.DOC/'PREDICTIONS_LOCK.json')
    for b in prediction_lock['files']:
        assert C.sha(C.ROOT/b['path'])==b['sha256']
    teacher_lock=C.read(C.ROOT/'_docs/experiments/pallet_visible_refine_hidden_pnp_v1/PREDICTION_LOCK.json')
    for k in ('predictions','metadata'):
        b=teacher_lock[k];assert C.sha(C.ROOT/b['path'])==b['sha256']
    assert teacher_lock['predictions']==lock['teacher']
    base=C.ROOT/'data/pallet/results/pallet_existing_data_transfer_v1'
    refs=C.read(base/'REFERENCE_PREDICTIONS.json')
    preds={a:refs[a] for a in ARMS[:2]}
    for a,name in zip(ARMS[2:],('T0_EASY_PSEUDO','T1_HARD_PSEUDO','T2_HARD_MANUAL')):
        preds[a]=C.read(base/f'HELDOUT_{name}.json')['predictions']
    teacher=C.read(C.ROOT/teacher_lock['predictions']['path'])['predictions']['TYPE_REPLAY_PIPELINE']
    legacy_path=C.ROOT/'data/pallet/results/pallet_replay_clean19_v1/TRUTH_FOR_DISPLAY_ONLY.json'
    assert C.sha(legacy_path)==C.read(REVIEW/'QA_QUEUE.json')['legacy_sha256']
    legacy=C.read(legacy_path)
    rows=[]
    for fi,ci in final['review_queue']:
        frame=final['frames'][fi];corner=frame['corners'][ci];s=selection['frames'][fi]
        if corner['status']!='DIRECT_VISIBLE':continue
        assert corner['coordinate_source']=='manual_click' and C.valid_corner(corner,s['hw'])
        fid=frame['frame_id'];g=np.array(corner['xy']);old=legacy[fid];oldxy=np.asarray(old['gt'][ci])
        oldvalid=old['valid'][ci] and np.isfinite(oldxy).all()
        row=dict(frame_id=fid,corner_id=ci,severity=s['severity'],recording=s['recording'],
                 xy=g.tolist(),errors={},missing={},legacy_distance=float(np.linalg.norm(g-oldxy)) if oldvalid else None)
        for a,cache in {**preds,'TEACHER':teacher}.items():
            q=point(cache.get(fid,{}),ci)
            row['missing'][a]=q is None
            row['errors'][a]=float(np.linalg.norm(q-g)) if q is not None else math.hypot(*s['hw'])
        rows.append(row)
    put(QA/'SCORED_POINTS_FINAL_PRIVATE.json',rows)
    groups={'ALL':rows,**{s:[r for r in rows if r['severity']==s] for s in C.SEVERITIES}}
    recordings=sorted({r['recording'] for r in rows})
    groups.update({g:[r for r in rows if r['recording']==g] for g in recordings})
    scores={g:{a:metrics([r['errors'][a] for r in rr]) for a in ARMS} for g,rr in groups.items()}
    percorner={str(i):{a:metrics([r['errors'][a] for r in rows if r['corner_id']==i]) for a in ARMS} for i in range(8)}
    result=dict(reference=final['reference_version'],groups=scores,per_corner=percorner,fixed_identity=True,
                symmetry_remapping=False,missing_policy='native image diagonal penalty',new_training=0,new_inference=0)
    put(C.DOC/'VERIFIED_RESULTS_FINAL.json',result)
    compare=dict(pairwise={g:{f'{b}-minus-{a}':difference(rr,a,b) for a,b in PAIRS} for g,rr in groups.items()},
                 leave_one_recording_out={g:{f'{b}-minus-{a}':difference([r for r in rows if r['recording']!=g],a,b)
                                            for a,b in PAIRS} for g in recordings})
    old_results=C.read(C.DOC/'VERIFIED_RESULTS.json')
    delta={a:dict(PCK10_correct_delta=scores['ALL'][a]['PCK']['10']['correct']-old_results['groups']['ALL'][a]['PCK']['10']['correct'],
                  median_delta=scores['ALL'][a]['median_px']-old_results['groups']['ALL'][a]['median_px']) for a in ARMS}
    compare['provisional_to_final']=delta
    put(C.DOC/'MODEL_COMPARISON_FINAL.json',compare)
    agree=metrics([r['legacy_distance'] for r in rows if r['legacy_distance'] is not None])
    put(C.DOC/'REFERENCE_DISAGREEMENT_FINAL.json',dict(agreement=agree,GT_auto_edit=False,not_independent_6D_GT=True))
    teacher_result=dict(arm='TYPE_REPLAY_PIPELINE',supplementary_only=True,used_for_winner=False,
        lock=teacher_lock,groups={g:metrics([r['errors']['TEACHER'] for r in rr]) for g,rr in groups.items()},
        frame_coverage=dict(available=sum(fid in teacher for fid in {r['frame_id'] for r in rows}),total=len({r['frame_id'] for r in rows})),
        point_coverage=dict(available=sum(not r['missing']['TEACHER'] for r in rows),total=len(rows)),
        present_only=metrics([r['errors']['TEACHER'] for r in rows if not r['missing']['TEACHER']]),
        missing_policy='native image diagonal penalty; report present-only separately')
    put(C.DOC/'TEACHER_SUPPLEMENT_FINAL.json',teacher_result)
    status_counts=Counter(final['frames'][f]['corners'][c]['status'] for f,c in final['review_queue'])
    summary=dict(reference=final['reference_version'],selection=len(selection['frames']),
        saved_frames=sum(bool(f.get('keypoints_saved')) for f in final['frames']),direct_visible=len(rows),
        scored_frames=len({r['frame_id'] for r in rows}),recordings=len(recordings),status_counts=dict(status_counts),
        by_severity={s:len(groups[s]) for s in C.SEVERITIES},exact_coordinates_private=True,training_steps=0,
        protocol='PNP_ASSISTED_KEYPOINTS_FIRST_THEN_STATUS_THEN_METADATA_QA; not blind first pass')
    put(C.DOC/'VERIFIED_LABELS_FINAL_PUBLIC_SUMMARY.json',summary)
    qa_final=dict(status='HUMAN_METADATA_QA_COMPLETE',qa_points=len(expected),decisions_complete=len(decisions),
        decisions=decisions,old_visibility_is_authority=False,selection_unchanged=True,image_hashes_unchanged=True,
        historical_artifact_hashes=historic,final_reference_sha256=C.sha(QA/'VERIFIED_LABELS_FINAL_PRIVATE.json'),
        human_labels_sha256=C.sha(QA/'LABELS.json'),final_direct_visible=len(rows),
        provisional_to_final_point_delta=len(rows)-old_results['groups']['ALL']['R0']['n'],
        frozen_predictions_verified=True,teacher_cache_verified=True,no_new_training=True)
    put(C.DOC/'METADATA_QA_FINAL.json',qa_final)
    unchanged=all(scores[g]==old_results['groups'][g] for g in scores)
    decision=dict(status='COMPLETED_FINAL_VISIBLE_ANCHOR_EVALUATION',existing_conclusion_maintained=unchanged,
        primary='REFERENCE_STABLE_ENOUGH_FOR_NEXT_MODEL_WORK' if agree['gt20']==0 else 'REFERENCE_REVIEW_REQUIRED',
        secondary='NO_GENERAL_WINNER',teacher_supplement_not_ranked=True,more_labeling_needed_now=False,
        retraining_justified_now=False,next='Complete 011067 whole-C4 contract audit; no training',
        human_metadata_QA_complete=True,all_144_statuses_complete=False)
    put(C.DOC/'FINAL_DECISION_V2.json',decision)
    lines=['# verified anchor — 2점 QA 이후 최종 평가','',
        '실제 사람 결정 2/2를 반영한 별도 V2 reference입니다. 기존 잠정 결과는 보존했습니다.',
        'P4는 외부 가림→불확실로 바뀌어 계속 제외, P2는 직접 보임과 좌표를 유지했습니다.',
        f'선정 {summary["selection"]}장 / 저장 {summary["saved_frames"]}장 / 최종 직접 보임 {len(rows)}점. 기존 대비 점수 변화 없음: **{unchanged}**.',
        '', '| 모델 | PCK5 | PCK10 | PCK20 | median px | P90 px | >20 |','|---|---:|---:|---:|---:|---:|---:|']
    for a,m in {**scores['ALL'],'TEACHER (보조)':teacher_result['groups']['ALL']}.items():
        lines.append(f'|{a}|'+ '|'.join(f'{m["PCK"][str(k)]["correct"]}/{m["n"]}' for k in (5,10,20))+f'|{m["median_px"]:.4f}|{m["p90_px"]:.4f}|{m["gt20"]}|')
    lines += ['',f'교사 coverage: {teacher_result["frame_coverage"]} frames, {teacher_result["point_coverage"]} points.',
        '교사는 winner 선정에서 제외했습니다. 고정 ID, symmetry-min 없음, 결측은 원영상 대각선 벌점입니다.',
        f'새 visible reference와 legacy 거리: median {agree["median_px"]:.4f}px, P90 {agree["p90_px"]:.4f}px, >20px {agree["gt20"]}점.',
        '', '## 난도별 PCK10 / median px','', '| 난도 | R0 | OLD_S1 | T0 | T1 | T2 |','|---|---|---|---|---|---|']
    for s in C.SEVERITIES:
        lines.append('|'+s+'|'+'|'.join(f'{scores[s][a]["PCK"]["10"]["correct"]}/{scores[s][a]["n"]}; {scores[s][a]["median_px"]:.3f}' for a in ARMS)+'|')
    lines += ['', '## 해석과 한계','',
        '기존 다섯 모델 결론이 유지됩니다. T1의 PCK10 우위는 R0/T0 대비 1점이고 T2는 median이 가장 낮아, 전반적 우승 모델은 선언하지 않습니다.',
        '수량 충족은 전체 144개 상태 완료를 의미하지 않습니다. 미분류/미입력은 계속 제외했고 추가 annotation은 요구하지 않습니다.',
        'PnP 보조 first pass, 재사용 DEV, 관측 가능한 점의 선택 편향, P6 부재, 적은 recording 수의 한계를 유지합니다. 독립 6D GT가 아닙니다.',
        '', '![QA 전후와 최종 평가](../pallet_011067_corner_contract_v1/figures/01_verified_anchor_qa_before_after.png)',
        '', '## 근거','',
        '- [사람 QA 결과/보존 해시](METADATA_QA_FINAL.json)',
        '- [난도·recording·corner별 결과](VERIFIED_RESULTS_FINAL.json)',
        '- [frame win/loss/tie·leave-one-recording-out·잠정→최종 변화](MODEL_COMPARISON_FINAL.json)',
        '- [교사 보조 비교](TEACHER_SUPPLEMENT_FINAL.json)',
        '- [reference 비교](REFERENCE_DISAGREEMENT_FINAL.json)',
        '- [최종 결정](FINAL_DECISION_V2.json)',
        '- [다음 011067 contract 감사](../pallet_011067_corner_contract_v1/REPORT_KO.md)',
        '', '`python -m scripts.research.pallet_verified_anchor_v1.finalize_metadata_qa` — 새 학습/추론 없음.']
    path=C.DOC/'EVALUATION_REPORT_FINAL_KO.md'
    text='\n'.join(lines)+'\n'
    if path.exists():assert path.read_text()==text
    else:C.save_new(path,text)
    assert historic=={n:C.sha(C.DOC/n) for n in old_names}
    print(summary);print('RESULTS',scores['ALL']);print('TEACHER',teacher_result['groups']['ALL']);print('UNCHANGED',unchanged)


if __name__=='__main__':main()
