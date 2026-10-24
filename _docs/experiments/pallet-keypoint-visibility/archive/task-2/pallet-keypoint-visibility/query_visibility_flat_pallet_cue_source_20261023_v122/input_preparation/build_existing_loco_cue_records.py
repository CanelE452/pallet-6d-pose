"""Existing JSON-only cue/official-target join. No new labels, q, pixels or model reads."""
import hashlib
import json
from collections import Counter
from pathlib import Path

H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
CUES = P / 'query_visibility_loco_native_context_20261023_v74/public_reference_review/RGB_REFERENCE_PROPOSALS.jsonl'
TARGETS = P / 'query_visibility_loco_annotation_target_masks_20261023_v115/TARGET20_EXISTING_ANNOTATION_ENTITIES_GT_ONLY.jsonl'
REQUESTS = P / 'query_visibility_loco_annotation_target_masks_20261023_v115/SAM_REQUEST_PLAN40_METADATA_ONLY.jsonl'
CONTEXTS = P / 'query_visibility_target_context_rgb_20261023_v107/TARGET_CONTEXT_DEFINITIONS216_INPUT_ONLY.jsonl'


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def bind(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def inside(point, box):
    return box is not None and box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    scope = V / 'ROOT_SOURCE_SCOPE_BEFORE_NEW_LABELS.json'
    assert json.loads(scope.read_text())['fit_or_model_forward_admitted_yet'] is False
    cues, targets, requests, contexts = map(rows, (CUES, TARGETS, REQUESTS, CONTEXTS))
    assert len(cues) == 10 and len(targets) == 20 and len(requests) == 40
    input_rows, evidence, per_cue = [], [], []
    for ci, cue in enumerate(cues):
        matches = [(i, t) for i, t in enumerate(targets) if
                   t['official_image_id'] == cue['image_id'] and
                   t['existing_target_ann_id'] == cue['representative_annotation_id']]
        assert len(matches) == 1
        ti, target = matches[0]
        request_matches = [(i, r) for i, r in enumerate(requests) if
                           r['frame_index'] == target['frame_index'] and r['role'] == 'TARGET']
        assert len(request_matches) == 1
        ri, request = request_matches[0]
        witnesses = [w for w in target['visible_own_cue_witnesses'] or [] if w['cue_refID'] == cue['cue_refID']]
        assert len(witnesses) == 1 and all(witnesses[0][k] == value for k, value in cue.items())
        ctx_matches = [(i, r) for i, r in enumerate(contexts) if
                       r['source'] == 'LOCO' and r['original_input_id'] == target['input_id']]
        assert len(ctx_matches) == 1
        cxi, ctx = ctx_matches[0]
        assert request['ann_id'] == target['existing_target_ann_id']
        assert request['box_xyxy_raw'] == target['box_xyxy_raw']
        assert request['raw_hw'] == target['raw_hw'] == ctx['raw_hw']
        assert request['split'] == target['split']
        assert request['image_sha256'] == target['image_sha256'] == cue['image_sha256']
        assert request['image'] == cue['image']
        assert cue['state_proposal'] == 'VISIBLE_REFERENCE_CONDITIONAL_AI'
        box, ref, unc = target['box_xyxy_raw'], cue['reference_xy'], cue['uncertainty_bbox_half_open_xyxy']
        r_in = inside(ref, box)
        unc_in = inside(unc[:2], box) and inside(unc[2:], box)
        assert r_in
        old_box = ctx['target_box_xyxy']
        input_rows.append(dict(
            existing_cue_record_index=ci, cue_record_id=cue['cue_refID'], frame_index=target['frame_index'],
            input_id=target['input_id'], source='LOCO', split=target['split'], source_subset=target['source_subset'],
            official_image_id=target['official_image_id'], official_target_annotation_id=target['existing_target_ann_id'],
            full_RGB=request['canonical_decoded_RGB'], full_RGB_format='NPY_RGB_UINT8',
            decoded_RGB_sha256=request['decoded_RGB_pixels_sha256'], raw_hw=target['raw_hw'],
            original_JPEG=dict(path=request['image'], sha256=request['image_sha256']),
            official_target_bbox_xyxy=box, official_target_bbox_xywh=target['bbox_xywh_raw'],
            target_box_source='EXISTING_OFFICIAL_ANNOTATION_FOR_NEW_SOURCE_EMULATION_NOT_INITIALIZER',
            new_source_q8_L_not_constructed_yet=True, original_initializer_outputs_unchanged=True,
            initializer_to_annotation_association='UNKNOWN', native_semantic_mapping='UNKNOWN',
            official_target_provenance=dict(file=bind(TARGETS), row_index=ti),
            RGB_request_metadata_provenance=dict(file=bind(REQUESTS), row_index=ri),
            original_initializer_context_provenance=dict(file=bind(CONTEXTS), row_index=cxi),
        ))
        evidence.append(dict(
            existing_cue_record_index=ci, cue_record_id=cue['cue_refID'], frame_index=target['frame_index'],
            input_id=target['input_id'], split=target['split'],
            original_cue_proposal_unchanged=cue,
            reference_xy_unchanged=ref, uncertainty_bbox_half_open_xyxy_unchanged=unc,
            source_cue_provenance=dict(file=bind(CUES), row_index=ci),
            official_annotation_id_matches=True,
            cue_to_target_meaning='Existing explicit AI visual association to this fixed official annotation entity; weak region cue, not certified own pixels or native corner.',
            initializer_to_annotation_association='UNKNOWN',
            reference_inside_official_bbox=r_in, whole_uncertainty_inside_official_bbox=unc_in,
            uncertainty_clipped_or_modified=False, source_evidence_only=True,
            new_labels_created=False, new_training_admission=False,
            natural_external_or_self_label_generated=False,
        ))
        per_cue.append(dict(cue_record_id=cue['cue_refID'], frame_index=target['frame_index'],
                            split=target['split'], annotation_id=target['existing_target_ann_id'],
                            reference_inside_official_bbox=r_in, whole_uncertainty_inside_official_bbox=unc_in,
                            original_initializer_context_origin=ctx['box_origin'],
                            reference_inside_original_initializer_context=inside(ref, old_box)))
    H.mkdir(exist_ok=True)
    ip = H / 'EXISTING10_CUE_TARGET_INPUT_ONLY.jsonl'
    ep = H / 'EXISTING10_CUE_EVIDENCE_ONLY.jsonl'
    for path, data in ((ip, input_rows), (ep, evidence)):
        with path.open('x') as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    counts = dict(original_fixed_frames=20, existing_visible_cues=10,
                  unique_cue_frames=len({r['frame_index'] for r in input_rows}),
                  cue_frame_indices=sorted({r['frame_index'] for r in input_rows}),
                  cue_folds=dict(Counter(r['split'] for r in input_rows)),
                  cue_frame_folds=dict(Counter({r['frame_index']: r['split'] for r in input_rows}.values())),
                  exact_official_ann_image_RGB_identity_matches=10,
                  reference_inside_official_bbox=10,
                  whole_original_uncertainty_inside_official_bbox=sum(r['whole_uncertainty_inside_official_bbox'] for r in per_cue),
                  original_initializer_context_missing_cue_records=sum(r['original_initializer_context_origin'] == 'MISSING' for r in per_cue),
                  reference_inside_original_initializer_context=sum(r['reference_inside_original_initializer_context'] for r in per_cue),
                  natural_E_labels_available_in_these_proposals=0, new_q_rows_created=0, new_labels_created=0)
    report = dict(
        completed=True, metadata_only=True, root_scope=bind(scope), code=bind(Path(__file__)),
        inputs=bind(ip), evidence=bind(ep), source_JSON_bindings=[bind(p) for p in (CUES, TARGETS, REQUESTS, CONTEXTS)],
        counts=counts, per_cue=per_cue,
        finding_KO='기존10개 visible structural-cue weak proposals는 모두 official target annotation ID·image ID·RGB SHA로 정확히 연결됩니다. 원래 initializer context와의 관계는별도이며 UNKNOWN을 유지합니다. 따라서 official target bbox를 사용하는 새 source emulation은 이10개를출발점으로구성할수있습니다. 원uncertainty가bbox경계를넘는2개도수정·제거하지않습니다.',
        minimal_next_code_KO='이 input/evidence metadata join을그대로사용하고, 후속source geometry가고정되면 official bbox의4corner를포함하는7개rectangle anchors + fixed carrier q0=r+offset로새 q8을만듭니다. 새 q8의 unclipped minmax/context 및 V60 span L을동일하게계산합니다. 원actual initializer q8/L/bbox는변경하지않습니다. r/uncertainty는query-emulation/cover-certificate용 evidence 경로에만두고head에는image-derived inputs만전달합니다. 현재코드는q8·이미지·새labels를구성하지않습니다.',
        source_limits=['Only5 images/10 correlated cues,6TRAIN4VALIDATION; repeated source-development exposure.',
                       'Visible weak region/component cues, not per-query material membership, certified own pixels or native equality.',
                       'No natural external E label is present here. Future controlled E needs full original uncertainty coverage and matched cue-preserving sham.',
                       'GT-box source-emulation is explicit supervised input construction, not proof that the deployment initializer selected this instance.',
                       'V115 SAM request metadata is used only for existing RGB/annotation identity; no predicted masks or source model scores are read.'],
        actual_RGB_masks_depth_features_models_scores_checkpoint_current_V112_fit_PnP_reads=0,
    )
    write(H / 'SCHEMA_AND_MATCH_COUNTS.json', report)
    print(json.dumps(dict(summary=bind(H / 'SCHEMA_AND_MATCH_COUNTS.json'), inputs=bind(ip), evidence=bind(ep), counts=counts), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
