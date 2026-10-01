"""Seal one fixed feature intervention before its four deterministic fits."""
from . import common as C


def main():
    review = C.read(C.DOC / 'PREFIT_REVIEW.json')
    assert review['complete'] and review['PASS']
    assert not (C.RAW / 'fits').exists()
    prior = C.read(C.ANCHOR_DOC / 'TRAIN_PROTOCOL.json')
    C.verify(C.read(C.ANCHOR_DOC / 'TRAIN_PROTOCOL_SHA.json'))
    source = C.read(C.ANCHOR_DOC / 'SOURCE_FEASIBILITY.json')
    assert source['complete'] and source['PASS'] and source['source_TRAIN_only']
    inputs = dict(prior['inputs'])
    inputs['prefit_review'] = C.bind(C.DOC / 'PREFIT_REVIEW.json')
    # Target/candidate feasibility is reused without new oracle selection.
    assert inputs['source_feasibility'] == C.bind(C.ANCHOR_DOC / 'SOURCE_FEASIBILITY.json')
    codes = [C.bind(C.HERE / n) for n in
             ('common.py', 'convex_train.py', 'evaluate_source.py', 'evaluate_real.py',
              'seal_training.py', 'prefit_review.py')]
    codes += prior['codes']
    codes = list({b['path']: b for b in codes}.values())
    p = dict(prior)
    p.update(schema='pallet_pose_anchor_context_training_v1', created_at=C.now(),
        raw_feature_dim=94, feature_dim=189,
        feature_map='normalized94_abs_anchor_delta94_identity1',
        initialization='All189 weights zero; no optimizer initialization seed search.',
        certificate_definition=prior['certificate_definition'].replace('w94', 'w189'),
        objective=prior['objective'].replace('w94', 'w189'),
        architecture='Shared linear189 with fixed R0 operational GEO context: concatenate normalized94, absolute normalized difference to the anchor94, and one exact anchor identity bit. Frozen candidates and refiners.',
        feature_transform='Exactly previous float32 subtract/divide using frozen TRAIN mean94/std94, promote to float64, then append abs(z_candidate-z_anchor) and identity. No additional normalization. Invalid candidate rows are all189 zeros.',
        changed_factor='Only the fixed input feature representation changes from94 to189. Preserve anchored target hashes, original candidate validity, full2598 denominator, CE+ridge coefficient, zero initialization, solver caps, certificate and all source/real gates.',
        anchor_rule='Frozen input-only R0 operational GEO identity, never the learned R0_ONLY choice. Original R0 slots0/1 cross-checked against candidate names. One all-invalid TRAIN row retains index-1, target-1 and zero CE over full2598. Any valid candidate without a valid R0 anchor is a contract error; no GT-derived or D-derived substitute.',
        embedding_control='Weights [old94, zeros95] reproduce the previous scores and CE+ridge objective on the same inputs. Extra features can alter context-dependent ordering; naive shared linear anchor subtraction alone cannot.',
        generalization_limit='Relative candidate and RGB context were previously studied with failures; this exact fixed189 map is a bounded ablation, not a claim of novelty or a guarantee of improvement.',
        inputs=inputs, codes=codes,
        evidence={'previous_anchored_protocol': C.bind(C.ANCHOR_DOC / 'TRAIN_PROTOCOL.json'),
                  'previous_anchored_report': C.bind(C.ANCHOR_DOC / 'REPORT_KO.md'),
                  'reused_real_oracle': C.bind(C.ANCHOR_DOC / 'REAL_FEASIBILITY.json')},
        evidence_access='Sealer-only rationale. Optimization opens only input/code bindings, never real-quality evidence.',
        stop='Max4 deterministic final fits; original1000 iteration/2000 objective-call limits and numerical certificate. Any failed certificate or any failed original45 source checks stops this method before learned real routing. No sweep, winner promotion, altered thresholds or dropped seeds.',
        method_success=False, goal_complete=False)
    assert p['target_rule'] == 'R0_GEO_PARETO_NONREGRESSION_FIXED_TRAIN_MINMAX'
    assert p['source_val'] == prior['source_val'] and p['real_evaluation'] == prior['real_evaluation']
    for b in [*inputs.values(), *codes, *p['evidence'].values()]:
        C.verify(b)
    C.save(C.DOC / 'TRAIN_PROTOCOL.json', p)
    C.save(C.DOC / 'TRAIN_PROTOCOL_SHA.json', C.bind(C.DOC / 'TRAIN_PROTOCOL.json'))
    print('ANCHOR_CONTEXT_TRAIN_PROTOCOL_SEALED', C.bind(C.DOC / 'TRAIN_PROTOCOL.json'))


if __name__ == '__main__':
    main()
