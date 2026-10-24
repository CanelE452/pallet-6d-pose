import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R = P / 'query_visibility_matched_query_scalar_20261023_v94'
O = R / 'source_metadata_physicalcue'
V66 = P / 'query_visibility_controlled_real_20261022_v66'
V85 = P / 'query_visibility_source_cue_corpus_20261023_v85'
V89 = P / 'query_visibility_public_cue_grid_20261023_v89'
V90 = P / 'query_visibility_public_cue_inputs_20261023_v90'
V91 = P / 'query_visibility_physical_cue_aux_20261023_v91'

FILES = {
    'scope': R / 'SCOPE_BEFORE_MATCHED_QUERY_CONTRACT.json',
    'teacher_contract': V91 / 'teacher_preparation/TEACHER_CONTRACT_BEFORE_BUILD.json',
    'cue_views604': V91 / 'teacher_preparation/corpus_build_v1/ALL604_CUE_VIEW_TEACHERS_GT_ONLY.jsonl',
    'cues302': V91 / 'teacher_preparation/corpus_build_v1/ALL302_CUES_SOURCE_IDS_GT_ONLY.jsonl',
    'teacher_summary': V91 / 'teacher_preparation/corpus_build_v1/GT_CORPUS_SUMMARY_AND_SCHEMA.json',
    'teacher_seal': V91 / 'teacher_preparation/corpus_build_v1/GT_CORPUS_SEAL.json',
    'proposed294': V85 / 'all49_completed_corpus/PROPOSED_294_CUE_AUXILIARY_GT_ONLY.jsonl',
    'canonical_summary': V85 / 'all49_completed_corpus/CORPUS_SUMMARY_KO.json',
    'canonical_seal': V85 / 'all49_completed_corpus/SEAL.json',
    'frames100': V89 / 'inputs_metadata/FRAMES_INPUTS_ONLY.jsonl',
    'grid_protocol': V89 / 'BEFORE_GRID_PROTOCOL.json',
    'old_q_inputs100': V66 / 'public_pallet/PAIRED_QUERIES/INPUTS_RGB_ONLY.jsonl',
    'old_q_recipe': V66 / 'PAIRED_QUERY_RECIPE.json',
    'old_q_interpretation': V66 / 'public_pallet/PAIRED_QUERIES/QUERY_RECIPE_INTERPRETATION.json',
    'old_q_seal': V66 / 'public_pallet/PAIRED_QUERIES/SEAL.json',
    'v90_protocol': V90 / 'INPUT_PRODUCER_PROTOCOL_BEFORE_PIXELS.json',
    'v90_code': V90 / 'producer/prepare_public_spatial_inputs.py',
    'v90_pack_seal': V90 / 'input_pack/INPUT_PACK_SEAL.json',
    'v90_frame_records': V90 / 'input_pack/FRAME_FULL_P3_RECORDS.jsonl',
    'r2b_initializer_protocol': P / 'query_visibility_cue_occlusion_20261023_v72/public_pallet_initializer/PROTOCOL.json',
    'loco_initializer_protocol': P / 'query_visibility_loco_native_context_20261023_v74/public_initializer/PROTOCOL.json',
}

def bind(p):
    b = p.read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b)}

def load(name):
    return json.loads(FILES[name].read_text())

def rows(name):
    return [json.loads(x) for x in FILES[name].read_text().splitlines() if x.strip()]

def write(p, obj):
    assert not p.exists(), f'Preserve existing output: {p}'
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def coverage(b, q, L):
    if b is None:
        return 'MISSING_REFERENCE_U'
    x0, y0, x1, y1 = b
    assert x1 > x0 and y1 > y0
    lo_x, lo_y = q[0] - L/2, q[1] - L/2
    hi_x, hi_y = q[0] + L/2, q[1] + L/2
    if x0 >= lo_x and y0 >= lo_y and x1 <= hi_x and y1 <= hi_y:
        return 'FULL'
    if min(x1, hi_x) > max(x0, lo_x) and min(y1, hi_y) > max(y0, lo_y):
        return 'PARTIAL'
    return 'MISS'

def main():
    policy = {
        'schema': 'V94_SOURCE_METADATA_PHYSICAL_CUE_AUDIT_SCOPE_V1',
        'created_UTC': datetime.now(timezone.utc).isoformat(),
        'authority': bind(FILES['scope']),
        'prior_exposure': 'Existing source cue/teacher/query metadata and earlier source visual reviews were known. This is a deterministic metadata reconciliation, not a blinded annotation.',
        'read_scope': 'Only listed public/source JSON/JSONL/code. No image bytes, PNG/NPY pixels, tensor arrays, model weights, current223 artifacts or current result values.',
        'arithmetic': 'All 604 cue-view x all eight old teacher-guided q preserved; float64/Python double continuous half-open windows. FULL is box containment, PARTIAL positive-area overlap, MISS zero-area overlap. Null remains MISSING_REFERENCE_U.',
        'not_a_label_packet': 'Availability never establishes target/cue binding or visibility; no best query, nearest cue assignment, coordinate movement or native label generation.',
        'input_bindings': {k: bind(p) for k, p in FILES.items()},
        'new_pixels_forward_fit_current_PnP': 0,
    }
    write(O / 'SCOPE_AND_ARITHMETIC_BEFORE_RECOMPUTATION.json', policy)
    cues, views, frames, old = rows('cues302'), rows('cue_views604'), rows('frames100'), rows('old_q_inputs100')
    proposed = rows('proposed294')
    assert len(cues) == 302 and len(views) == 604 and len(frames) == len(old) == 100 and len(proposed) == 294
    assert len({x['cue_id'] for x in cues}) == 302
    assert len({x['frame_input_id'] for x in frames}) == 100
    by_sha = {x['image_sha256']: x for x in old}
    assert len(by_sha) == 100
    by_frame = {x['frame_input_id']: x for x in frames}
    pair_old = defaultdict(list)
    for f in frames:
        q = by_sha[f['image_sha256']]
        assert f['image'] == q['image'] and f['raw_hw'] == q['raw_hw']
        assert q['query'] == q['query8'] and len(q['query']) == 8
        assert math.isfinite(q['side']) and q['side'] > 0
        pair_old[f['pair_id']].append(q)
    assert len(pair_old) == 50
    for pair, ab in pair_old.items():
        assert len(ab) == 2 and ab[0]['query'] == ab[1]['query'] and ab[0]['side'] == ab[1]['side'] and ab[0]['box'] == ab[1]['box']
    assert {x['native8_semantic_id'] for x in cues} == {None}
    counts, weighted, bysplit = Counter(), Counter(), defaultdict(Counter)
    avail, avail_known, avsplit = Counter(), Counter(), defaultdict(Counter)
    evidence_counts = Counter()
    csv_rows = []
    for v in views:
        label = int(v['teacher_label'])
        meaning = {0:'D_CONDITIONAL_AI', 1:'E_CONDITIONAL_AI', -1:'U'}[label]
        counts[meaning] += 1
        weighted[meaning] += v['duplicate_weight']
        bysplit[v['split']][meaning] += 1
        f = by_frame[v['frame_input_id']]
        qrec = by_sha[v['image_sha256']]
        assert v['image_sha256'] == f['image_sha256'] and v['raw_hw'] == f['raw_hw']
        b, r = v['uncertainty_bbox_xyxy_half_open_unchanged'], v['reference_xy_unchanged']
        assert (b is None) == (r is None)
        if r is not None:
            evidence_counts['views_with_nonnull_r_unc'] += 1
            assert all(math.isfinite(z) for z in b+r)
        evidence_counts['native8_unknown'] += int(v['native8_semantic_id'] is None)
        evidence_counts['human_gold_false'] += int(v['human_gold'] is False)
        for k, q in enumerate(qrec['query']):
            status = coverage(b, q, float(qrec['side']))
            avail[status] += 1
            if label != -1: avail_known[f'{meaning}|{status}'] += 1
            avsplit[v['split']][f'{meaning}|{status}'] += 1
            h,w = f['raw_hw']
            qvalid = all(math.isfinite(z) for z in q) and 0 <= q[0] < w and 0 <= q[1] < h
            q_in_unc = b is not None and b[0] <= q[0] < b[2] and b[1] <= q[1] < b[3]
            csv_rows.append(dict(cue_view_index=v['cue_view_index'], cue_id=v['cue_id'], pair_id=v['pair_id'], view=v['view'], split=v['split'], conditional_teacher=meaning, old_sampler_slot_logical_only=k, query_x=q[0], query_y=q[1], side=qrec['side'], query_center_in_native=qvalid, coverage=status, q_inside_uncertainty_DIAGNOSTIC_NOT_BINDING=q_in_unc, expected_native_cue_binding='UNKNOWN', training_admitted=False))
    assert len(csv_rows) == 4832
    summary = load('teacher_summary')
    assert counts == Counter({'D_CONDITIONAL_AI':261,'E_CONDITIONAL_AI':37,'U':306})
    assert weighted == Counter({'D_CONDITIONAL_AI':257.,'E_CONDITIONAL_AI':37.,'U':306.})
    assert summary['raw_view_label_counts'] == {'0':261,'-1':306,'1':37}
    with (O/'ALL4832_OLD_TEACHER_Q_CUE_AVAILABILITY_GT_ONLY.csv').open('x',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    v90 = rows('v90_frame_records')
    assert len(v90) == 100 and {x['image_sha256'] for x in v90} == set(by_sha)
    prohibited_query_fields = {'query','query8','keypoints','keypoints_xy','selected_index','candidates','candidate_scores','roi_side','side'}
    assert not any(set(x) & prohibited_query_fields for x in v90)
    geometric_known_full = sum(x['conditional_teacher'] != 'U' and x['coverage']=='FULL' for x in csv_rows)
    report = {
        'schema':'V94_SAVED_PHYSICAL_CUE_QUERY_LINEAGE_AND_AVAILABILITY_AUDIT',
        'PASS_metadata_reconciliation':True,
        'decision':'METADATA_READY_FOR_SEPARATE_ACTUAL_INITIALIZER_INPUT_SCOPE; NO_NEW_FIT_OR_LABEL_ADMISSION',
        'counts': {
            'physical_bin_instances':1,'pairs':50,'source_frames':100,'source_frames_by_split':dict(Counter(x['split'] for x in frames)),
            'original_cues':302,'canonical_nonseed_cues':294,'separate_seed_originals':8,'seed_linked_distinct_regions':6,
            'cue_views':604,'raw_view_conditional_states':dict(counts),'duplicate_weighted_view_states':dict(weighted),
            'view_states_by_split':{k:dict(v) for k,v in bysplit.items()},'evidence_metadata':dict(evidence_counts),
            'old_teacher_q_frames':100,'old_teacher_q_points':800,'old_teacher_q_pairs_shared_exact':50,
            'qcue_all_cross_combinations':4832,'all_combinations_availability':dict(avail),'known_teacher_availability_by_state':dict(avail_known),
            'known_teacher_FULL_combinations_not_binding':geometric_known_full,
            'all_combinations_by_split_state_coverage':{k:dict(v) for k,v in avsplit.items()},
            'q_inside_uncertainty_diagnostic_not_binding':sum(x['q_inside_uncertainty_DIAGNOSTIC_NOT_BINDING'] for x in csv_rows),
            'native_expected_cue_slot_bindings_known':0,'actual_initializer_q_L_frames_saved_in_reviewed_Pallox_manifests':0,
        },
        'answers_KO': {
            'a_actual_initializer': '검토한 Pallox100 저장 manifests에는 actual initializer q/L이 없다. V90은 같은 frozen backbone.predict를 실제100번 호출했지만 captured 후보/keypoints/선택index를 보존하지 않고 fullP3/affine와 V89 grid만 저장했다. 따라서 backbone forward가 있었음과 저장된 actual query 존재는 다르다. V72 actual48은 R2B, V74 actual20은 LOCO이며 Pallox가 아니다. 전체 파일시스템의 부재를 증명한 주장은 아니다.',
            'b_sampler_lineage': 'V66 q8은 paired SAM/depth conditional teacher evidence의 E-core/commonV/visible-boundary/teacherbox corners를 사전 sampler로 선택한 뒤 동일±3px jitter를 A/B에 적용했다. L은 q8 span의.25이며 teacher-union bbox와 jitter가 sourcebox를 만든다. k=0..7은 sampler logical slots이고 native8 semantic IDs가 아니다. V89/V90은 L128/256 grid이며 actual detector q도 아니다.',
            'c_instance_and_cue_support': '재추론을 별도 승인하면100원본 frame path/SHA/HW,50pair/view/split, 한 physical bin,302 cueID와151 nonnull r/uncertainty 후보를 그대로 join할 수 있다. 604중298 known D/E view records는 특정 wall/rim/runner cue의 before 고정+after 관찰과 문서 identity를 갖고 있다. 그러나 selected detection→이 bin 및 logical native slot→이 physical cue의 association은 현재 저장된 것으로 인증되지 않았다. r과 가장 가까운 candidate/slot을 GT-best로 골라 채우지 말아야 한다.',
            'd_denominators': 'ALL604 teacher JSONL은 label/uncertainty/duplicate mass의 권위있는 시작점이고 V89 frames100은 RGB identity join이다. 새 actual q/L이 저장되면 같은 half-open 기하로 모든 q×기존 cue의 FULL/PARTIAL/MISS/U를 계산할 수 있다. 지금 CSV4832는 기존 teacher-guided q에 대한 전조합 기하 진단일 뿐 같은-q scalar labels가 아니다. 특정 query의 expectedcue가 독립 정의되기 전 native supervised denominator는 UNKNOWN이며, AUX namespace는 별도 prospective binding/operator 계약으로 정의할 수 있다.'
        },
        'minimal_next_scope_not_execution': [
            'Pallox100 전체와 모든 no-detect/OOB를 보존하는 actual frozen initializer proposal metadata를 별도 사전 잠금·실행한다; GT-cue로 detection/slot/L을 선택하지 않는다.',
            '기존 part-slot은 view-relative wall/runner 의미이고 native8 UNKNOWN이다. Native label 승격 대신 분리된 AUX expected-cue task를 정의할 수 있으며 full CAD/native mapping을 모든 auxiliary fit의 필수조건으로 추가하지 않는다.',
            'Actual q/L과 source expectedcue binding이 있으면 fixed-cue label을 그대로 유지하고 availability를 GT-only로 따로 집계한다. Training/sourceVAL/deployment scalar operator는 동일해야 한다.',
            'Source cue identity가 불명확한 query는 UNKNOWN을 유지한다. FULL은 관측가능성/targetidentity/정답 보증이 아니고 MISS를 외부 E로 바꾸지 않는다.',
        ],
        'limits': {
            'domain':'Pallox is one tall wooden physical pallet-bin with wall/rim/runner cues; flat pallet deck/block/foot and square pallet distributions are different. Environment-heldout lab/hall is not independent-object/site validation.',
            'conditional_tier':'AI RGB-based part-line/corner-region observations, not human/native D/E/S gold. Natural cardboard evidence is conditional, no subpixel registration/opaque-alpha certificate.',
            'availability':'Source/eval r/uncertainty permit availability diagnostics only. Deployment cannot consultGT r to abstain.',
            'scope':'Metadata/code only; no pixels, mask/depth bodies, features/tensor arrays, weights, current reports or data, model/head forward, fitting, PnP.',
        },
        'binding_scope':{k:bind(p) for k,p in FILES.items()},
        'new_pixels_current_arrays_models_forward_fit_PnP':0,'training_admission':False,'fully_solved':False,'goal_status':'active',
    }
    write(O/'SOURCE_METADATA_PHYSICAL_CUE_AUDIT_KO.json',report)
    write(O/'SEAL.json',{'completed':True,'PASS':True,'scope':bind(O/'SCOPE_AND_ARITHMETIC_BEFORE_RECOMPUTATION.json'),'outputs':[bind(O/'SOURCE_METADATA_PHYSICAL_CUE_AUDIT_KO.json'),bind(O/'ALL4832_OLD_TEACHER_Q_CUE_AVAILABILITY_GT_ONLY.csv')],'code':bind(Path(__file__)),'original_files_modified':0,'pixels_model_forward_fit_current_PnP':0})
    print(json.dumps({'report':bind(O/'SOURCE_METADATA_PHYSICAL_CUE_AUDIT_KO.json'),'seal':bind(O/'SEAL.json'),'counts':report['counts']},ensure_ascii=False))

if __name__ == '__main__':
    main()
