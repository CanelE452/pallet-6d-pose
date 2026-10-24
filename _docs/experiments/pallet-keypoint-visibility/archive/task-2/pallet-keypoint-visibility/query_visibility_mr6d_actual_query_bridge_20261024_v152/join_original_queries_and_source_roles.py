"""Post-seal factual join. Keep all80 queries and all104 source roles distinct."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np

V = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    sealpath = V / 'actual_mr6d_initializer_v1/PREDICTION_SEAL.json'
    assert sha(sealpath) == '041ba0669c2067dbae43c4885dc3b9c897fcd66c286482dfacc2d493ebb6db08'
    seal = read(sealpath)
    for item in seal['outputs']:
        assert sha(item['path']) == item['sha256']
    props = lines(V / 'actual_mr6d_initializer_v1/FROZEN_PROPOSALS_ALL10_INPUT_ONLY.jsonl')
    facts = lines(V / 'original_query_evidence/ALL80_ORIGINAL_QUERY_CROP_FACTS.jsonl')
    nativepath = V / 'native_reference_audit/SOURCE104_NATIVE_SEMANTIC_PROPOSALS_AGENT.jsonl'
    native = lines(nativepath)
    assert len(native) == 104 and len(facts) == 80 and len(props) == 10
    index = {(r['frame_index'], r['gt_instance_index'], r['native_slot']): r for r in native}
    assert len(index) == 104
    # Root personally viewed all10 fullscene/crop boards and three overviews.
    # These are explicit visual judgments, never confidence thresholds/GT argmin.
    judgments = [
        ('WRONG_ENTITY_WOOD_CRATE', None, '전경 목재 상자/문짝을 선택했다. 뒤의 세워진 팔레트와 다른 물체다.'),
        ('WRONG_ENTITY_WOOD_CRATE', None, '전경 목재 상자를 선택했다. 원 upright 팔레트 후보로 대체하지 않았다.'),
        ('NO_PALLET_IN_VIEW_FALSE_POSITIVE', None, '제공 팔레트 두 개 모두 화면 밖이다. 원 선택은 바닥·벽 경계의 오검출이다.'),
        ('STACK_COMPOSITE_ENTITY_UNRESOLVED', None, 'q8이 위 팔레트와 아래 전면·배경에 걸친다. 단일 팔레트 대응을 확정하지 않았다.'),
        ('STACK_COMPOSITE_ENTITY_UNRESOLVED', None, '전면 점들은 아래 팔레트, 후방 점들은 위 팔레트·배경으로 흩어진다. 단일 GT로 붙이지 않는다.'),
        ('WRONG_ENTITY_WOOD_CARGO', None, '팔레트 위 목재 화물의 얇은 윗부분을 선택했다. 팔레트 자체가 아니다.'),
        ('PALLET_ENTITY_SUPPORTED_NATIVE_MAPPING_PROVISIONAL', 0, '선택 box와 앞 모서리는 단일 팔레트에 대응한다. 후방 crop의 적재물과 위치 오차·self를 분리한다.'),
        ('WRONG_ENTITY_WOOD_CARGO', None, '팔레트 위 목재 상자를 선택했다. 실제 팔레트는 영상 아래쪽에 있다.'),
        ('PALLET_ENTITY_SUPPORTED_NATIVE_MAPPING_PROVISIONAL', 0, '선택 box와 앞 네 모서리는 팔레트에 대응한다. 후방은 적재물·self·좌표 오차를 분리한다.'),
        ('WRONG_ENTITY_SMALL_FLOOR_OBJECT', None, '팔레트 뒤 바닥의 작은 물체를 선택했다. 원 팔레트로 재선택하지 않았다.'),
    ]
    # This describes only factual crop contents, not whether the target role is D/E.
    local = {
        0: ['FOREIGN_WOOD_CRATE'] * 8,
        1: ['FOREIGN_WOOD_CRATE'] * 8,
        2: ['BACKGROUND_WALL_FLOOR'] * 8,
        3: ['STACK_WOOD', 'BACKGROUND_NEAR_WOOD_EDGE', 'STACK_WOOD_AND_FLOOR', 'OOB_PARTIAL_CROP',
            'BACKGROUND_NEAR_WOOD_EDGE', 'BACKGROUND_AND_WOOD_DECK', 'UPPER_WOOD_DECK', 'WOOD_END_AND_BACKGROUND'],
        4: ['LOWER_WOOD_END', 'WOOD_END_AND_FLOOR', 'LOWER_WOOD_END', 'OOB_PARTIAL_WOOD',
            'BACKGROUND', 'BACKGROUND_AND_WOOD_DECK', 'UPPER_WOOD_DECK', 'UPPER_WOOD_DECK'],
        5: ['FOREIGN_WOOD_CARGO'] * 8,
        6: ['PALLET_FRONT_CORNER', 'PALLET_FRONT_CORNER', 'PALLET_FRONT_CORNER', 'OOB_PARTIAL_WOOD',
            'BACKGROUND_AND_WOOD_EDGE', 'GRAY_CARGO', 'GRAY_CARGO', 'WOOD_DECK_NEAR_BLUE_CARGO'],
        7: ['BLUE_CARGO_AND_WOOD_CARGO', 'FOREIGN_WOOD_CARGO', 'FOREIGN_WOOD_CARGO', 'BLUE_CARGO_AND_WOOD_CARGO',
            'WALL_RACK_AND_WOOD_CARGO', 'WALL_AND_WOOD_CARGO', 'WALL_AND_WOOD_CARGO', 'FOREIGN_WOOD_CARGO'],
        8: ['PALLET_FRONT_CORNER', 'PALLET_FRONT_CORNER', 'PALLET_FRONT_CORNER', 'PALLET_FRONT_CORNER',
            'BACKGROUND_AND_WOOD_EDGE', 'GRAY_CARGO', 'GRAY_CARGO_WHITE_CARGO', 'WOOD_EDGE_AND_WHITE_CARGO'],
        9: ['SMALL_FLOOR_OBJECT_AND_FLOOR'] * 8,
    }
    frame_rows, joined = [], []
    for i, prop in enumerate(props):
        status, owner, reason = judgments[i]
        instance_ids = sorted({r['gt_instance_index'] for r in native if r['frame_index'] == i})
        frame_rows.append(dict(frame_index=i, source_id=prop['source_id'], selected_index=prop['selected_index'],
            candidate_count=len(prop['candidates']), source_GT_instance_ids=instance_ids, associated_GT_instance_id=owner,
            entity_status=status, reason_KO=reason, native_mapping='PROVISIONAL_SAME_CHANNEL_ONLY' if owner is not None else 'UNRESOLVED',
            source_scope='ALL10_FIXED_DEVELOPMENT_FRAMES_NOT_INITIALIZER_ACCURACY_ESTIMATE', human_gold=False))
        for slot in range(8):
            item = dict(facts[i * 8 + slot])
            assert item['frame_index'] == i and item['native_slot'] == slot
            q, side = np.asarray(item['original_query'], float), item['original_L']
            comparisons = []
            for instance in instance_ids:
                reference = index[i, instance, slot]
                ref = np.asarray(reference['reference_xy'], float)
                offset = q - ref
                comparisons.append(dict(gt_instance_index=instance, native_slot=slot, reference_xy=ref.tolist(),
                    nominal_same_channel_offset_xy=offset.tolist(), nominal_error_px=float(np.linalg.norm(offset)),
                    nominal_error_over_original_L=float(np.linalg.norm(offset) / side),
                    nominal_reference_inside_original_square=bool((np.abs(offset) <= side / 2).all()),
                    nominal_reference_inside_central32_of96=bool((np.abs(offset) <= side / 6).all()),
                    source_assistant_role_proposal=reference['proposal_state'], convex_self_visible_proxy_only=reference['convex_self_visible_proxy_only'],
                    NOT_an_exact_physical_pixel_gold=True, NOT_used_to_choose_owner_or_permutation=True))
            role = None if owner is None else index[i, owner, slot]
            item.update(associated_entity_id_or_U=owner, entity_status=status,
                native_slot_correspondence='PROVISIONAL_SAME_CHANNEL_ONLY' if owner is not None else 'UNRESOLVED',
                associated_role_semantic_state='U' if role is None else role['proposal_state'],
                associated_role_proposal_only=role is not None,
                associated_role_reason_KO=reason if role is None else role['reason_KO'],
                query_local_visual_evidence=local[i][slot], source_GT_native_comparisons=comparisons,
                wrong_entity_is_NOT_external_occlusion=True, no_GT_query_or_candidate_correction=True,
                source_assistant_review=True, human_gold=False, train_ready=False,
                supervised_quality_candidate='INCORRECT_TARGET_IDENTITY' if i in (0, 1, 2, 5, 7, 9) else 'UNRESOLVED_STACK' if i in (3, 4) else 'TARGET_ENTITY_SUPPORTED_POINT_QUALITY_SEPARATE')
            joined.append(item)
    out = V / 'root_saved_query_role_diagnosis'
    out.mkdir(exist_ok=False)
    rowsfile = out / 'ALL80_QUERY_IDENTITY_ROLE_AND_LOCAL_EVIDENCE.jsonl'
    rowsfile.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in joined))
    write(out / 'ALL10_FIRSTARGMAX_ENTITY_REVIEW.json', dict(completed=True, frames=frame_rows))
    accepted = [r for r in joined if r['associated_entity_id_or_U'] is not None]
    summary = dict(completed=True, code=bind(__file__), prediction_seal=bind(sealpath),
        source_proposals=bind(nativepath), source_proposals_read_after_prediction_seal=True,
        root_association_rule=bind(V / 'ROOT_ASSOCIATION_RULE_BEFORE_SAVED_PREDICTION_READ.json'),
        full_query_inventory=80, full_GT_native_inventory=104, source_frames=10,
        frame_identity_counts=dict(Counter(r['entity_status'] for r in frame_rows)),
        wrong_entity_or_no_visible_target_frames=6, stack_composite_unresolved_frames=2,
        supported_target_entity_frames=2, selected_entity_query_rows=16,
        selected_entity_role_proposal_counts=dict(Counter(r['associated_role_semantic_state'] for r in accepted)),
        all80_input_continuous_valid=sum(r['continuous_valid'] for r in joined),
        all80_input_nearest_valid=sum(r['nearest_valid'] for r in joined),
        all80_joined_inventory=bind(rowsfile), frames=frame_rows,
        source_sample_not_representative_not_an_initializer_accuracy_benchmark=True,
        proposed_quality_and_role_supervision_separated=True,
        all_failed80_inventory_retained_not_reduced_to16=True,
        exact3px_source_gold_created=False, semantic_human_gold_created=0, training_ready=False,
        actual_new_classifier_training_PnP_private_calls=0,
        next_action_KO='공개 자료의 오검출·여러 instance 혼합·같은 native 역할을 분리한 감독 정의를 고정하고, source만으로 query quality와 역할 visibility를 별도 출력하는 평가·학습을 설계한다.')
    write(out / 'RESULTS.json', summary)
    print(json.dumps(bind(out / 'RESULTS.json')))


if __name__ == '__main__':
    main()
