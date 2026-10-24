"""Source-only saved initial-logit/weak-role audit; no model, fit or image reads.

The current/private benchmark is deliberately absent from the explicit read list.
Ranking is a prospective algebraic diagnostic, not a fitted/admitted model.
"""
from pathlib import Path
import collections
import hashlib
import json
import numpy as np

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
OUT = Path(__file__).resolve().parent
V146 = P / 'query_visibility_public_native_source_expansion_20261024_v146'
V148 = P / 'query_visibility_public_native_supervision_control_20261024_v148'
POLICY = OUT.parent / 'ROOT_SOURCE_MECHANISM_POLICY_BEFORE_SAVED_ANALYSIS.json'
LABELS = V146 / 'ROOT_ADJUDICATED_SOURCE_NATIVE512_WEAK_LABELS.jsonl'
INITIAL = V148 / 'runs/C33_C34_PUBLIC_NATIVE_FIXED2000/C33_PUBLIC_NATIVE_WEIGHT0_FIXED2000/INITIAL_PUBLIC512_SCORES.npz'

def binding(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def write(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def quantiles(x):
    return dict(zip(('min', 'q25', 'median', 'q75', 'p90', 'max'),
                    map(float, np.quantile(x, (0, .25, .5, .75, .9, 1)))))

def main():
    # These source bindings are intentionally not recursively dereferenced.
    rows = [json.loads(s) for s in LABELS.read_text().splitlines()]
    with np.load(INITIAL, allow_pickle=False) as z:
        a = {k: z[k].copy() for k in z.files}
    assert len(rows) == 512
    assert np.array_equal(a['global_index'], np.arange(512))
    raw = a['raw_head_logit_before_U'].astype(np.float64)
    assert np.isfinite(raw).all()
    for i, r in enumerate(rows):
        assert i == r['global_index'] and r['frame_index'] == i // 8
        assert r['native_slot'] == i % 8 == int(a['native_slot'][i])
        assert r['id'] == str(a['frame_id'][i]) and r['source_id'] == str(a['source_id'][i])
        assert r['split'] == str(a['split'][i])
        assert r['id'] + ':native' + str(i % 8) == str(a['query_id'][i])
        assert np.array_equal(a['query_xy'][i], r['query_xy']) and float(a['L'][i]) == r['L']
        assert bool(a['query_available'][i]) == r['original_input_valid']
        assert r['root_adjudicated'] and not r['human_gold']
        assert not r['full_native8_ontology_certified'] and not r['full_goal_redefined_to_lower_roles']
    state = np.array([r['semantic_state'] for r in rows])
    split = a['split']; slot = a['native_slot']; available = a['query_available']
    train = split == 'TRAIN_LAB'
    di = np.flatnonzero(train & available & (state == 'D'))
    ei = np.flatnonzero(train & available & (state == 'E'))
    assert len(di) == 35 and len(ei) == 8
    assert set(slot[ei]) == {3}
    inventories = {}
    for name, mask in [('ALL', np.ones(512, bool)), ('TRAIN_LAB', train), ('VAL_HALL', ~train)]:
        inventories[name] = dict(rows=int(mask.sum()),
            states={s:int((mask & (state == s)).sum()) for s in 'DEU'},
            known_input_U=np.flatnonzero(mask & np.isin(state, ['D','E']) & ~available).tolist(),
            state_native={s:dict(collections.Counter(map(str, slot[mask & (state == s)]))) for s in 'DEU'})
    assert inventories['ALL']['states'] == dict(D=70, E=14, U=428)
    locality = {}
    for fold, mask in [('TRAIN_LAB', train), ('VAL_HALL', ~train)]:
        locality[fold] = {}
        for s in 'DE':
            ids = np.flatnonzero(mask & (state == s))
            ref = np.array([rows[i]['source_reference_xy'] for i in ids], np.float64)
            assert ref.shape == (len(ids), 2) and np.isfinite(ref).all()
            delta = a['query_xy'][ids] - ref
            error = np.linalg.norm(delta, axis=1)
            local = {}
            for name, halfspan in [('full96_sample_centres', 47.5 / 96), ('central32_sample_centres', 15.5 / 96)]:
                inside = (np.abs(delta) <= halfspan * a['L'][ids, None]).all(axis=1)
                outside = ids[~inside]
                local[name] = dict(reference_inside=int(inside.sum()), reference_outside=int((~inside).sum()),
                    outside_rows=[dict(global_index=int(i), source_id=rows[i]['source_id'], native_slot=int(slot[i]),
                        input_valid=bool(available[i]), q_error_px=float(np.linalg.norm(a['query_xy'][i]-np.array(rows[i]['source_reference_xy']))),
                        q_error_over_L=float(np.linalg.norm(a['query_xy'][i]-np.array(rows[i]['source_reference_xy']))/a['L'][i])) for i in outside])
            locality[fold][s] = dict(N=len(ids), q_to_virtual_reference_error_px=quantiles(error), **local)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    zd, ze = raw[di], raw[ei]
    gd = .5 / len(di) * sigmoid(zd)
    ge = .5 / len(ei) * (sigmoid(ze) - 1)
    bce = dict(formula='.5 mean_D softplus(zD) + .5 mean_E softplus(-zE)',
        mean_loss=float(.5*np.logaddexp(0,zd).mean()+.5*np.logaddexp(0,-ze).mean()),
        D=dict(N=len(di), native_counts=dict(collections.Counter(map(str,slot[di]))), logits=quantiles(zd), correct_at_zero=int((zd<0).sum()),
               bias_derivative_sum=float(gd.sum()), bias_derivative_by_slot={str(k):float(gd[slot[di]==k].sum()) for k in set(slot[di])}),
        E=dict(N=len(ei), native_counts=dict(collections.Counter(map(str,slot[ei]))), logits=quantiles(ze), correct_at_zero=int((ze>=0).sum()), bias_derivative_sum=float(ge.sum())),
        bias_derivative_total=float(gd.sum()+ge.sum()), D_abs_mass_over_E_abs_mass=float(gd.sum()/-ge.sum()),
        interpretation='Initial public term alone has a positive derivative on the common output bias; gradient descent moves that bias negative. This does not identify full-model/optimizer causation.')
    pair_results = {}
    for name, d in [('ALL_D35_x_E8', di), ('SAME_NATIVE3_D16_x_E8', di[slot[di]==3])]:
        diff = raw[d,None] - raw[ei][None,:]
        margin_arg = 1 + diff
        derivative = sigmoid(margin_arg) / diff.size
        violating = np.argwhere(margin_arg > 0)
        pairs = []
        for ix,jx in violating:
            i,j=int(d[ix]),int(ei[jx])
            pairs.append(dict(D_global_index=i,E_global_index=j,D_source_id=rows[i]['source_id'],E_source_id=rows[j]['source_id'],
                D_native=int(slot[i]),E_native=int(slot[j]),D_logit=float(raw[i]),E_logit=float(raw[j]),
                logit_difference_D_minus_E=float(diff[ix,jx]),softplus_margin1=float(np.logaddexp(0,margin_arg[ix,jx]))))
        pair_results[name] = dict(N=diff.size, D_N=len(d), E_N=len(ei), same_native_pairs=int((slot[d,None] == slot[ei][None,:]).sum()),
            cross_native_pairs=int((slot[d,None] != slot[ei][None,:]).sum()),
            same_frame_pairs=int((a['frame_index'][d,None] == a['frame_index'][ei][None,:]).sum()),
            misordered_D_ge_E=int((diff>=0).sum()), margin1_violations=int((margin_arg>0).sum()),
            hinge1_zero=int((margin_arg<=0).sum()), mean_softplus_margin1=float(np.logaddexp(0,margin_arg).mean()),
            max_softplus_margin1=float(np.logaddexp(0,margin_arg).max()), difference_quantiles=quantiles(diff),
            direct_common_output_bias_derivative_algebraic=0,
            numeric_summed_endpoint_derivative=float(derivative.sum()-derivative.sum()),
            margin1_violating_pairs=pairs)
    docs = {
        'V150_scope': POLICY,
        'V148_policy': V148/'ROOT_PUBLIC_NATIVE_SUPERVISION_POLICY_BEFORE_FIT.json',
        'V148_trainer': V148/'source_fit_preparation/train_public_native_control.py',
        'V146_labels': LABELS, 'initial_C32_public_saved_logits': INITIAL,
        'V109_YCB_corrected_policy': P/'query_visibility_ycb_real_surface_source_20261023_v109/ROOT_SOURCE_POLICY_V2_CORRECTED_INVENTORY_BEFORE_BULK.json',
        'V109_YCB_actual_summary': P/'query_visibility_ycb_real_surface_source_20261023_v109/corpus_build_v1/SUMMARY.json',
        'V109_YCB_producer': P/'query_visibility_ycb_real_surface_source_20261023_v109/producer/build_surface_source.py',
        'V106_static_AUX_contract': P/'query_visibility_ycb_surface_cues_20261023_v106/SURFACE_CANDIDATE_RESULT.json',
        'V110_actual_fit_design': P/'query_visibility_natural_surface_rgb_head_20261023_v110/ROOT_C10_DESIGN_BEFORE_FEATURE_FORWARD.json',
        'V110_actual_freeze_metadata': P/'query_visibility_natural_surface_rgb_head_20261023_v110/runs/C10_NATURAL_SURFACE_TARGET_RELATIVE_RGB/MODEL_FREEZE.json',
        'V110_actual_source_gates': P/'query_visibility_natural_surface_rgb_head_20261023_v110/runs/C10_NATURAL_SURFACE_TARGET_RELATIVE_RGB/SOURCE_GATE_RESULTS.json',
        'V95_counterfactual_scope': P/'query_visibility_same_q_counterfactual_20261023_v95/ROOT_SOURCE_SCOPE_BEFORE_ACTUAL_CONSTRUCTION.json',
        'V135_pair_trainer': P/'query_visibility_native_pair_supervision_control_20261024_v135/source_fit_preparation/train_native_pairs.py',
        'V133_legacy_source_loader': P/'query_visibility_native_geometry_control_20261024_v133/source_fit_preparation/train_matched_geometry.py',
        'V142_AUX_zero_wrapper': P/'query_visibility_native_target_gradient_control_20261024_v142/source_fit_preparation/train_native_gradient_control.py',
    }
    ycb = json.loads(docs['V109_YCB_actual_summary'].read_text())
    aux = json.loads(docs['V106_static_AUX_contract'].read_text())
    freeze = json.loads(docs['V110_actual_freeze_metadata'].read_text())
    gates = json.loads(docs['V110_actual_source_gates'].read_text())
    result = dict(completed=True,PASS=True,source_only=True,bindings={k:binding(p) for k,p in docs.items()},
        inventory=inventories, locality_diagnostic=locality, initial_public_BCE=bce, prospective_ranking=pair_results,
        YCB_existing_actual_source=dict(instances=ycb['all_instances'],slots=ycb['all_slots'],states=ycb['states'],VAL_instances=ycb['VAL_instances'],
            native_mapping=ycb['native_mapping'],construction=aux['construction_rule'],C10_selected_update=freeze['selected_update'],
            C10_failed_source_gates={k:v for k,v in gates['gates'].items() if not v['PASS']}),
        boundaries=dict(native_role_weak_bridge='Source-specific provisional integrated-assembly lower2/3 virtual footprint only; all upper/rear remain U. All E are cardboard/native3.',
            locality='Virtual-reference location relative to original saved q/L sample-centre support; not a new teacher or physical-pixel gold. No filtering/relabeling/correction.',
            rank='Loss is invariant to common additive logit offset. Native BCE, parameter decay and hidden-layer updates can still change absolute logits.',
            no_novelty_claim='V95 already used same-q public cutouts; V135 already used native matched E/D hinge pairs. Public partial-native ranking would test objective/calibration only, not new material/role coverage.',
            no_generalization_claim='Only one integrated wooden assembly with cardboard E, lower slot3; source VAL is development-exposed and not fresh human gold.'),
        calls=dict(model=0,optimizer_fit=0,feature_checkpoint_body=0,RGB_body=0,current_private_body_or_hash=0,forbidden_phase_access=0,PnP_or_q_L_change=0),
        selected_model=None,new_fit_authorized_or_run=False)
    write(OUT/'SOURCE_FIDELITY_AND_INITIAL_RANKING.json',result)
    write(OUT/'REVIEW_COMPLETE.json',dict(completed=True,PASS=True,code=binding(__file__),result=binding(OUT/'SOURCE_FIDELITY_AND_INITIAL_RANKING.json'),
        labels=binding(LABELS),initial_saved_source_logits=binding(INITIAL),source_only=True,current_private_body_or_hash=0,model_or_fit_calls=0,selected_model=None))
    print(json.dumps(dict(completed=True,PASS=True,receipt=binding(OUT/'REVIEW_COMPLETE.json'),initial_public_bias_derivative=bce['bias_derivative_total'],
        ranking={k:{s:v[s]for s in('N','misordered_D_ge_E','margin1_violations','mean_softplus_margin1')}for k,v in pair_results.items()}),indent=2))

if __name__=='__main__':
    main()
