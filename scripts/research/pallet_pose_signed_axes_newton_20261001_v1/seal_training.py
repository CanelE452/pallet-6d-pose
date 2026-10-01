"""Seal a solver-only control of the unchanged signed physical T/R objective."""
from . import common as C
from . import convex_train as T


def main():
    review=C.read(C.DOC/'PREFIT_REVIEW.json')
    assert review['complete'] and review['PASS']
    assert not (C.RAW/'fits').exists()
    prior=C.read(C.SIGNED_DOC/'TRAIN_PROTOCOL.json')
    C.verify(C.read(C.SIGNED_DOC/'TRAIN_PROTOCOL_SHA.json'))
    rejected=C.read(C.SIGNED_DOC/'REJECTED_R0_ONLY.json')
    verification=C.read(C.SIGNED_DOC/'REJECTED_FIT_VERIFICATION.json')
    assert rejected['complete'] and not rejected['accepted'] and not rejected['certificate']['PASS']
    assert verification['complete'] and verification['PASS'] and not verification['accepted_checkpoint']
    assert verification['next_numerical_intervention']['status']=='PROPOSAL_ONLY_NOT_EXECUTED'
    basis=C.bind(C.RBF_DOC/'RBF_BASIS.json')
    assert review['basis_SHA_bind']==prior['basis_SHA_bind']==basis
    inputs=dict(prior['inputs'],prefit_review=C.bind(C.DOC/'PREFIT_REVIEW.json'))
    codes=[C.bind(C.HERE/name) for name in ('common.py','convex_train.py','evaluate_source.py',
        'evaluate_real.py','seal_training.py','prefit_review.py')]+prior['codes']
    codes=list({b['path']:b for b in codes}.values())
    p=dict(prior)
    p.update(schema='pallet_pose_signed_axes_newton_training_v1',created_at=C.now(),
        solver_rule=T.SOLVER_RULE,solver=T.solver_config(),
        certificate=dict(optimizer_success=True,gap_upper_bound_max=1e-6,gradient_linf_max=1e-8),
        changed_factor='Only the numerical solver changes: fixed block generalized Newton with Armijo backtracking replaces L-BFGS-B. Preserve the exact signed two-axis Huber objective, ridge, data/targets/features/scales/basis, zero initialization, runtime selection, per-fit iteration/call caps, and all source/real criteria.',
        solver_accounting='The initial objective and every line-search trial count toward the same2000-call cap. Only accepted points count toward the1000-iteration cap. Use the last accepted point and its already evaluated gradient. Each direction solves the two253-dimensional SPD generalized Hessian blocks; zero data curvature at |Huber residual|=1, plus unchanged lambda I. No extra damping, tolerance search, restart, rejected-weight warm start, or budget extension.',
        solver_success='Require finite gradient Linf<=1e-8 AND the original strong-convex gradient gap bound<=1e-6 at an accepted point. Stop on nonfinite values, failed solve, non-descent direction, step underflow or an exhausted original budget. Convergence does not certify T/R improvement.',
        stop='At most4 deterministic final fits from zero; each at most1000 accepted iterations and2000 total objective calls. A failed fit or any failed original source45 condition prevents real routing. Preserve every real stability criterion; do not select iterations/seeds or change thresholds after results.',
        inputs=inputs,codes=codes,
        evidence=dict(previous_signed_protocol=C.bind(C.SIGNED_DOC/'TRAIN_PROTOCOL.json'),
            previous_rejected_fit=C.bind(C.SIGNED_DOC/'REJECTED_R0_ONLY.json'),
            rejected_fit_verification=C.bind(C.SIGNED_DOC/'REJECTED_FIT_VERIFICATION.json'),
            previous_report=C.bind(C.SIGNED_DOC/'REPORT_KO.md'),
            original_goal_protocol=prior['evidence']['original_goal_protocol']),
        evidence_access='Only this sealer reads prior rejected-state evidence. Fitting starts from exact zero and opens the sealed inputs/codes; it does not load prior weights, source VAL quality or real-quality evidence.',
        method_success=False,goal_complete=False)
    unchanged=('source_val','real_evaluation','original_goal_criteria','real_stability_contract',
        'candidate_pool','inference_tie','no_valid_fallback','runtime_inputs','models','train_rows',
        'max_fits','lambda_l2','normalization','initialization','architecture','feature_transform',
        'objective','target','scale_definition','anchor_rule','runtime_safety','certificate_definition',
        'feature_dim','raw_feature_dim','context_dim','rbf_dim','output_dim','feature_map','input_rule',
        'target_rule','loss_rule','prediction_rule','huber_delta','basis_SHA_bind')
    for key in unchanged:assert p[key]==prior[key],key
    assert p['models']==list(C.MODEL_NAMES) and p['max_fits']==4
    assert p['solver']['maxiter']==prior['solver']['maxiter']==1000
    assert p['solver']['maxfun']==prior['solver']['maxfun']==2000
    for key,value in T.metadata().items():assert p[key]==review[key]==value,key
    for b in [*inputs.values(),*codes,*p['evidence'].values()]:C.verify(b)
    C.save(C.DOC/'TRAIN_PROTOCOL.json',p)
    C.save(C.DOC/'TRAIN_PROTOCOL_SHA.json',C.bind(C.DOC/'TRAIN_PROTOCOL.json'))
    print('SIGNED_AXES_NEWTON_PROTOCOL_SEALED',C.bind(C.DOC/'TRAIN_PROTOCOL.json'))


if __name__=='__main__':main()
