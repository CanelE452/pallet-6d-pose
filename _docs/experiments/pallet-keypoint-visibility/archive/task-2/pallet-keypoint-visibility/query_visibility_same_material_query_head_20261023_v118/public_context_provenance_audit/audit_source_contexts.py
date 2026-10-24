"""PUBLIC source JSON/JSONL provenance arithmetic. No pixels/NPZ/model/current data."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import json
import numpy as np

O = Path(__file__).resolve().parent
V = O.parent
P = V.parent


def root(s):
    if s == 111:
        return P/'query_visibility_spatial_target_query_features_20261023_v111'
    matches=list(P.glob('query_visibility*20261023_v' + str(s)))
    assert len(matches)==1, matches
    return matches[0]


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s]


def binding(p):
    p = Path(p)
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def finite_points(points):
    result = []
    for q in points:
        if q is not None:
            a = np.asarray(q, np.float64)
            if a.shape == (2,) and np.isfinite(a).all() and not np.array_equal(a, [-1., -1.]):
                result.append(a)
    return np.asarray(result, np.float64).reshape(-1, 2)


def extent(points):
    q = finite_points(points)
    if not len(q):
        return None
    lo, hi = q.min(0), q.max(0)
    return np.r_[lo, hi].tolist() if (hi > lo).all() else None


def valid_box(box):
    if box is None:
        return False
    b = np.asarray(box, np.float64)
    return b.shape == (4,) and np.isfinite(b).all() and (b[2:] > b[:2]).all()


def uv(q, box):
    good = finite_points([q])
    if len(good) != 1 or not valid_box(box):
        return None
    b = np.asarray(box, np.float64)
    return ((good[0] - b[:2]) / (b[2:] - b[:2])).tolist()


def qstats(values):
    a = np.asarray(values, np.float64)
    return None if not len(a) else dict(zip(('min', 'p05', 'p25', 'median', 'p75', 'p95', 'max'),
                                          np.quantile(a, [0, .05, .25, .5, .75, .95, 1]).tolist()))


def geometry(q, box):
    x = uv(q, box)
    if x is None:
        return dict(u=None, v=None, inside=None, signed_edge_distance=None,
                    exact_boundary=None, one_half_token_edge_band=None, one_token_edge_band=None)
    u, v = x
    distance = min(u, v, 1-u, 1-v)
    return dict(u=u, v=v, inside=bool(distance >= 0), signed_edge_distance=distance,
                exact_boundary=bool(distance == 0),
                one_half_token_edge_band=bool(0 <= distance <= 1/32),
                one_token_edge_band=bool(0 <= distance <= 1/16))


def summarize(rr):
    good = [r for r in rr if r['inside'] is not None]
    return dict(N=len(rr), base_queries=len({r['base_query_key'] for r in rr}),
                contexts=len({r['context_key'] for r in rr}),
                origin_counts=dict(Counter(r['box_origin'] for r in rr)),
                state_counts=dict(Counter(r['state'] for r in rr)),
                geometry_available=len(good), missing=len(rr)-len(good),
                inside=sum(r['inside'] for r in good), outside=sum(not r['inside'] for r in good),
                exact_boundary=sum(r['exact_boundary'] for r in good),
                inside_within_half_context_token=sum(r['one_half_token_edge_band'] for r in good),
                inside_within_one_context_token=sum(r['one_token_edge_band'] for r in good),
                u=qstats([r['u'] for r in good]), v=qstats([r['v'] for r in good]),
                signed_edge_distance=qstats([r['signed_edge_distance'] for r in good]),
                q8_extent_counterfactual_inside=sum(r['q8_extent_inside'] is True for r in rr),
                q8_extent_counterfactual_exact_boundary=sum(r['q8_extent_exact_boundary'] is True for r in rr))


def main():
    assert not (O/'RESULT.json').exists(), 'No output overwrite'
    paths = dict(
        definitions=root(107)/'TARGET_CONTEXT_DEFINITIONS216_INPUT_ONLY.jsonl',
        old_contexts=root(108)/'context_feature_preparation/UNIT_CONTEXTS3652_INPUT_ONLY.jsonl',
        old_occurrences=root(96)/'actual_source_pair_input_pack/ALL31680_INPUT_LINEAGE.jsonl',
        old_GT=root(96)/'actual_query_anchored_image_intake/OCCURRENCES31680_GT_ONLY.jsonl',
        synthetic_occurrences=root(99)/'matching_feature_pack/INPUT_LINEAGE2304_INPUT_ONLY.jsonl',
        synthetic_GT=root(99)/'geometry_query_preparation/EXPECTED_CUE_TARGETS96_GT_ONLY.jsonl',
        surface_inputs=root(109)/'query_preparation/INPUTS1000_QUERY_EMULATION_ONLY.jsonl',
        surface_GT=root(109)/'corpus_build_v1/SLOT_EVIDENCE8000_GT_ONLY.jsonl',
        material_inputs=root(113)/'query_preparation/INPUTS400_QUERY_EMULATION_ONLY.jsonl',
        material_GT=root(113)/'corpus_build_v1/SLOT_EVIDENCE3200_GT_ONLY.jsonl',
        samewood_inputs=root(116)/'corpus_build_v1/OCCURRENCES924_INPUT_ONLY.jsonl',
        samewood_GT=root(116)/'corpus_build_v1/OCCURRENCES924_GT_ONLY.jsonl')
    definitions = rows(paths['definitions'])
    defs = {d['context_id']: d for d in definitions}
    context_records = []
    for d in definitions:
        context_records.append(dict(context_key=d['context_id'], family=d['source'],
                                    input_id=d['original_input_id'], box=d['target_box_xyxy'],
                                    box_origin=d['box_origin'], q8=d['original_query8_xy']))
    records = []

    def add(family, split, variant, state, input_id, slot, q, box, origin, q8, context_key,
            occurrence_key, public=True):
        geom = geometry(q, box)
        counterfactual = geometry(q, extent(q8))
        records.append(dict(family=family, split=split, variant=variant, state=state,
                            input_id=input_id, logical_slot=slot, q=q, box=box,
                            box_origin=origin, context_key=context_key,
                            base_query_key=input_id+'::'+str(slot), occurrence_key=occurrence_key,
                            public=public, **geom,
                            q8_extent_inside=counterfactual['inside'],
                            q8_extent_exact_boundary=counterfactual['exact_boundary']))

    old_contexts = rows(paths['old_contexts'])
    old, gt = rows(paths['old_occurrences']), rows(paths['old_GT'])
    assert len(old) == len(gt) == 31680 and len(old_contexts) == 3652
    for i, (r, t) in enumerate(zip(old, gt)):
        assert r['occurrence_index'] == t['occurrence_index'] == i
        assert r['occurrence_id'] == t['occurrence_id']
        c = old_contexts[r['input_unit_index']]
        d = defs[c['context_id']]
        assert c['box_xyxy'] == d['target_box_xyxy'] and c['query_xy'] == r['query_xy']
        state = {0:'D', 1:'E', -1:'U'}[t['candidate_aux_hidden_label_GT_only']]
        add(r['source'], r['raw_split'], r['variant'], state, r['input_id'],
            r['native_slot_logical_only'], r['query_xy'], c['box_xyxy'], d['box_origin'],
            d['original_query8_xy'], c['context_id'], r['occurrence_id'])
    syn, sgt = rows(paths['synthetic_occurrences']), rows(paths['synthetic_GT'])
    assert len(syn) == 2304 and len(sgt) == 96
    for i, r in enumerate(syn):
        assert r['occurrence_index'] == i
        c = old_contexts[1348+i]
        d = defs[c['context_id']]
        t = sgt[r['base_index']]
        assert t['base_id'] == r['base_id']
        state = t['stages'][r['variant']][r['aux_corner_index']]['state']
        add('V99_SYNTH', r['split'], r['variant'], state, r['base_id'], r['aux_corner_index'],
            r['query_xy'], c['box_xyxy'], d['box_origin'], d['original_query8_xy'],
            c['context_id'], 'V99::'+str(i), public=False)
    for inputs_key, labels_key, family, n in (
            ('surface_inputs','surface_GT','V109_YCB_BOX_SURFACE_EMULATION',1000),
            ('material_inputs','material_GT','V113_YCB_MATERIAL_SURFACE_EMULATION',400)):
        inputs, teacher = rows(paths[inputs_key]), rows(paths[labels_key])
        assert len(inputs) == n and len(teacher) == 8*n
        for j, r in enumerate(inputs):
            q8, box = r['initial_points'], r['query_box_xyxy']
            assert box == extent(q8), 'Existing source box exact q8minmax'
            cid = family+'::'+r['input_id']
            context_records.append(dict(context_key=cid, family=family, input_id=r['input_id'],
                                        box=box, box_origin='EMULATED_Q8_MINMAX', q8=q8))
            for k in range(8):
                t = teacher[8*j+k]
                assert t['slot_index'] == 8*j+k and t['input_id'] == r['input_id']
                assert t['logical_aux_slot'] == k
                add(family, r['split'], 'NATURAL', t['raw_surface_teacher_state'],
                    r['input_id'], k, q8[k], box, 'EMULATED_Q8_MINMAX', q8, cid,
                    r['input_id']+'::'+str(k))
    samewood, wgt = rows(paths['samewood_inputs']), rows(paths['samewood_GT'])
    assert len(samewood) == len(wgt) == 924
    for i, (r, t) in enumerate(zip(samewood, wgt)):
        assert r['occurrence_index'] == t['occurrence_index'] == i
        assert r['occurrence_id'] == t['occurrence_id']
        d = defs[r['source']+':'+r['input_id']]
        assert r['frozen_context_box_xyxy'] == d['target_box_xyxy']
        add('V116_'+r['source'], r['raw_split'], r['variant'],
            {0:'D',1:'E',-1:'U'}[t['candidate_aux_hidden_label_GT_only']],
            r['input_id'], r['native_slot_logical_only'], r['query_xy'],
            r['frozen_context_box_xyxy'], d['box_origin'], d['original_query8_xy'],
            d['context_id'], r['occurrence_id'])
    assert len(records) == 46108
    known = [r for r in records if r['state'] in ('D','E')]
    public = [r for r in records if r['public']]
    known_public = [r for r in known if r['public']]
    assert Counter(r['state'] for r in known_public) == Counter(D=1984, E=820)
    grouped = defaultdict(list)
    for r in known:
        grouped[(r['family'],r['split'],r['state'])].append(r)
    contexts_stats = []
    for c in context_records:
        a, x = c['box'], extent(c['q8'])
        item = dict(context_key=c['context_key'],family=c['family'],input_id=c['input_id'],
                    box_origin=c['box_origin'], source_box=a, q8_minmax=x,
                    finite_q8_count=len(finite_points(c['q8'])),
                    source_box_missing=not valid_box(a), q8_extent_missing=not valid_box(x),
                    exact_equal=a == x, source_box_q8_inside=None,
                    box_IoU=None, source_to_q8_width_ratio=None, source_to_q8_height_ratio=None,
                    max_absolute_edge_difference_px=None)
        if valid_box(a) and valid_box(x):
            a, x = np.asarray(a,np.float64), np.asarray(x,np.float64)
            overlap=np.maximum(np.minimum(a[2:],x[2:])-np.maximum(a[:2],x[:2]),0)
            inter=float(np.prod(overlap));area_a=float(np.prod(a[2:]-a[:2]));area_x=float(np.prod(x[2:]-x[:2]))
            item.update(box_IoU=inter/(area_a+area_x-inter),
                        source_to_q8_width_ratio=float((a[2]-a[0])/(x[2]-x[0])),
                        source_to_q8_height_ratio=float((a[3]-a[1])/(x[3]-x[1])),
                        max_absolute_edge_difference_px=float(abs(a-x).max()),
                        source_box_q8_inside=sum(geometry(q,a.tolist())['inside'] is True for q in c['q8']))
        contexts_stats.append(item)
    csumm = {}
    for family in sorted({c['family'] for c in contexts_stats}):
        cs=[c for c in contexts_stats if c['family']==family]
        usable=[c for c in cs if c['box_IoU'] is not None]
        csumm[family]=dict(contexts=len(cs),origins=dict(Counter(c['box_origin']for c in cs)),
                           source_missing=sum(c['source_box_missing']for c in cs),
                           q8_extent_missing=sum(c['q8_extent_missing']for c in cs),
                           both_present=len(usable), exact_equal_when_present=sum(c['exact_equal']for c in usable),
                           source_box_q8_inside=sum(c['source_box_q8_inside']for c in usable),
                           source_box_q8_total=sum(c['finite_q8_count']for c in usable),
                           box_IoU=qstats([c['box_IoU']for c in usable]),
                           width_ratio=qstats([c['source_to_q8_width_ratio']for c in usable]),
                           height_ratio=qstats([c['source_to_q8_height_ratio']for c in usable]),
                           max_edge_delta_px=qstats([c['max_absolute_edge_difference_px']for c in usable]))
    ledger=O/'KNOWN_PUBLIC2804_Q_CONTEXT_DIAGNOSTICS_GT_ONLY.jsonl'
    with ledger.open('x')as f:
        for r in known_public:
            f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
    contextfile=O/'CONTEXT1616_SOURCEBOX_VS_Q8_EXTENT_INPUT_ONLY.jsonl'
    with contextfile.open('x')as f:
        for r in contexts_stats:
            f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
    current_code=root(111)/'current_evaluation_preparation/current_adapter.py'
    current_contract=root(111)/'ROOT_CURRENT_COMPARISON_OPERATOR_BEFORE_INPUT_READ.json'
    c13_contract=V/'ROOT_CURRENT_SAVED_FEATURE_OPERATOR_BEFORE_C13_FIT.json'
    source_codes=[root(107)/'build_metadata_inventory.py',
                  root(111)/'feature_preparation/prepare_spatial_query_features.py',
                  root(109)/'producer/build_surface_source.py',
                  root(113)/'teacher_preparation/build_material_teacher.py',
                  root(116)/'feature_preparation_v1/prepare_same_material_spatial_features.py']
    result=dict(completed=True,PASS=True,blockers=[],
        created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Source input-only JSON/JSONL plus frozen source label metadata for denominators. Current CODE/operator contracts only; no current metadata/arrays/GT/scores/pixels, source pixels/features/weights/model calls/fit/PnP.',
        no_box_query_label_or_model_change=True,
        question='Does source context construction always match documented current q8minmax?',
        answer_KO='아니다. 기존 공개 LOCO/Pallox와 V116은 원 initializer selectedbbox를 쓰지만, V109/V113 YCB와 V99 합성은 q8minmax를 쓴다. 따라서 방식 차이는 source family에 따라 존재하며 전체 source에 보편적이지 않다.',
        current_documented_only=dict(mode='FROZEN_INITIAL_Q8_UNCLIPPED_UNEXPANDED_MINMAX',
            code_rule='FP64 finite non-sentinel original8q min/max; OOB finite points kept, no clipping/expansion/recenter; invalid extent=>missing.',
            contract_allows_verified_selected_box_in_principle=True,
            actual_adapter_code_forces_q8_fallback=True,
            actual_current_boxes_or_uv_not_read_or_measured=True,
            source_comparison_does_not_establish_cause_or_predictive_improvement=True),
        diagnostic_geometry=dict(u='(qx-xmin)/(xmax-xmin)',v='(qy-ymin)/(ymax-ymin)',
            q_inside='closed0<=u,v<=1, exact C11 token sampler support convention',
            boundary='d=min(u,v,1-u,1-v); d<0 outside,d=0 exactboundary',
            half_token='0<=d<=1/32 of16-token context grid; diagnostic only',
            one_token='0<=d<=1/16; diagnostic only',
            counterfactual='Metadata-only recomputation under original q8 extent; never substituted into features, labels or evaluation.',
            exposure='Existing metadata was already examined; no blind-pre-exposure claim and no source/current outcome tuning.'),
        population=dict(all_occurrences=len(records),public_occurrences=len(public),
                        all_state_counts=dict(Counter(r['state']for r in records)),
                        public_state_counts=dict(Counter(r['state']for r in public)),
                        known_public_occurrences=len(known_public),known_public=summarize(known_public),
                        known_synthetic_occurrences=sum(not r['public']for r in known)),
        context_box_comparisons_by_original_definition=csumm,
        known_by_family_fold_state=[dict(family=k[0],fold=k[1],state=k[2],**summarize(rr))for k,rr in sorted(grouped.items())],
        known_by_family=[dict(family=family,**summarize([r for r in known if r['family']==family]))for family in sorted({r['family']for r in known})],
        known_by_origin=[dict(box_origin=origin,**summarize([r for r in known_public if r['box_origin']==origin]))for origin in sorted({r['box_origin']for r in known_public})],
        limitations_KO=['q8minmax를 source에 재계산할 메타데이터는 있다. 이는 input normalization 후보의 실행 가능성만 뜻하며 채택/성능/정답 association의 증거가 아니다.',
            'YCB q8minmax의 수식은 current fallback과 같아도 q는 GT projection+jitter source emulation이다. 실제 initializer q 분포와 같다고 볼 수 없다.',
            '원 selectedbbox도 target identity/whole-target support를 보증하지 않는다. q가 selectedbbox밖이면 Q는0이지만 기존 C11 availability는 그대로 유지된다.',
            'known counts에는 같은 basequery의 before/cover/sham 및 여러 donor 반복이 포함된다. 별도 base/context 분모를 함께 기록했으며 IID 표본이 아니다.',
            'current UV/box distribution은 이번 범위에서 측정하지 않았다. source 차이가 current 오류를 일으켰다는 단독 원인 주장을 하지 않는다.',
            'S는 known geometric outside-task, U는 unresolved로 유지했다. Native expectedpart mapping UNKNOWN/human goldfalse/새 training admission없음.'],
        outputs=dict(known_PUBLIC_GT_only=binding(ledger),input_context_comparison=binding(contextfile)),
        source_metadata_bindings={k:binding(p)for k,p in paths.items()},
        code_bindings=[binding(p)for p in source_codes+[current_code]],
        current_contract_bindings=[binding(current_contract),binding(c13_contract)],own_code=binding(__file__))
    (O/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (O/'SEAL.json').write_text(json.dumps(dict(completed=True,PASS=True,result=binding(O/'RESULT.json'),
        files=[binding(ledger),binding(contextfile),binding(__file__)]),indent=2)+'\n')
    print(json.dumps(dict(result=binding(O/'RESULT.json'),known_public=result['population']['known_public'],
        families=result['known_by_family'],context_box_comparisons=csumm),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
