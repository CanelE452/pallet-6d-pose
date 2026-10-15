"""Postseal V33 LCC control scores with the unchanged fixed references/math.

The native line population and models remain V32. Only the separate C mask
observation is the LCC33 control. All pure pose/metric/statistic/reference
functions below are exact frozen scorer aliases, never relabelled CAL models.
ROOT score GO, source peer, complete238x5 prediction, native/e parity and the
same-prediction independent audit precede any numeric reference access.
"""
from copy import deepcopy
from pathlib import Path
import sys

from . import score_clean_fresh_BI_poses_v4_32 as P
from . import evaluate_largest_component_global_outline_poses_v4_33 as M

C = P.C
ARMS = M.ARMS
PARTITIONS = M.PARTITIONS
CONTROLLER_SHA256 = 'e997f2d1c33924989d8f5de599943717df9d0fc06001a2e053a1dc44747b83ca'
FROZEN_SCORER_SHA256 = 'ed00f94eccef5f55b5f0985a47e3ba53207e5809063c4c53d292565921849f1f'
SELECTION_PROFILE = M.SELECTION_PROFILE
ANCHOR, ANCHOR_SHA256 = P.ANCHOR, P.ANCHOR_SHA256
PUBLIC_MANIFEST, PUBLIC_MANIFEST_SHA256 = P.PUBLIC_MANIFEST, P.PUBLIC_MANIFEST_SHA256
REFERENCE_PATHS = P.REFERENCE_PATHS
METRIC, REF = P.METRIC, P.REF

# Exact original numerical functions, original native32 type checks and refs.
prediction_pose = P.prediction_pose
compact_predictions = P.compact_predictions
pose_metric = P.pose_metric
distribution = P.distribution
score_rows = P.score_rows
_reference_metadata = P._reference_metadata
_read_references = P._read_references


def _pins():
    if C.sha(P.__file__) != FROZEN_SCORER_SHA256 or C.sha(M.__file__) != CONTROLLER_SHA256:
        raise ValueError('Frozen metric scorer and reviewed LCC controller bytes required')
    M._pins()
    if ARMS != P.ARMS or PARTITIONS != P.PARTITIONS:
        raise ValueError('Original five arms and all fixed cohort denominators required')


def _sources():
    paths = [Path(__file__), Path(__file__).with_name('test_score_largest_component_global_outline_poses_v4_33.py'),
        Path(P.__file__), Path(M.__file__), Path(C.__file__), Path(METRIC.__file__),
        Path(REF.__file__), Path(REF.PG.__file__), Path(REF.SYM.__file__)]
    return M._union(M._source_map(), {M._relative(path):C.sha(path) for path in paths})


def _source_peer(binding):
    from . import declare_initial_only_observed_ray_v4_32 as D
    path = D._peer(binding, (P, M, sys.modules[__name__]))
    sources = D._source_bindings(C.read(path))
    test = Path(__file__).with_name('test_score_largest_component_global_outline_poses_v4_33.py')
    if sources.get(M._relative(test)) != C.sha(test):
        raise ValueError('Independent score peer must bind the exact new toy test source')
    return path


def _prediction_gate(output, expected_central, expected_prediction):
    """New33 closure only; no metrics or reference document is decoded here."""
    _pins()
    output = Path(output)
    protocol = M._load_prepared(output)
    seal = C.read(output/'PREDICTION_SEAL.json')
    links = dict(predictions_sha256='PREDICTIONS.jsonl.gz',protocol_sha256='PROTOCOL.json',
        input_seal_sha256='INPUT_SEAL.json',execution_protocol_sha256='PREDICTION_EXECUTION_PROTOCOL.json',
        row_index_sha256='PREDICTION_ROW_INDEX.json',separate_e_comparison_sha256='E_COMPARISON.jsonl.gz',
        native_and_e_parity_sha256='NATIVE_AND_E_PARITY_V1.json')
    if (C.sha(output/'PREDICTION_SEAL.json') != expected_prediction
            or seal['rows'] != 238 or seal['ordered_ids'] != protocol['ordered_ids']
            or len(set(seal['ordered_ids'])) != 238 or seal['arms'] != list(ARMS) or seal['arm_count'] != 5
            or seal['all_declared_arms_complete'] is not True or seal['no_population_drop'] is not True
            or seal['before_fixed_reference_scoring'] is not True
            or seal['default'] != 'SAME_FROZEN_E unchanged'
            or seal['outline_selection_profile'] != SELECTION_PROFILE
            or seal['native_CAL32_model_relabelled'] is not False
            or protocol['central_plan_sha256'] != expected_central or C.sha(M.DECLARATION) != expected_central
            or any(seal[key] != C.sha(output/name) for key,name in links.items())):
        raise ValueError('Complete same-central LCC33 sealed238x5 before reference access required')
    parity = C.read(output/'NATIVE_AND_E_PARITY_V1.json')
    if (parity['rows'] != 238 or parity['native_arm_count'] != 2
            or parity['native_arm_payload_equalities'] != 476 or parity['e_payload_equalities'] != 238
            or parity['failure_count'] != 0 or parity['numeric_tolerance'] != 0.):
        raise ValueError('Every238 exact native two-arm/e parity required before references')
    path = output/'independent_root_audit_v1/AUDIT.json'
    audit = C.read(path)
    aseal = C.read(path.with_name('AUDIT_SEAL.json'))
    if (audit['rows'] != 238 or audit['failure_count'] != 0 or audit['failures']
            or C.sha(path) != aseal['receipt_sha256']
            or audit['prediction_seal'] != dict(path=M._relative(output/'PREDICTION_SEAL.json'),sha256=expected_prediction)
            or audit['outline_selection_profile'] != SELECTION_PROFILE
            or audit['native_observation_family'] != M.M.G.OBSERVATION_FAMILY
            or audit['native_noise_population'] != M.M.G.POPULATION
            or audit['native_selection_profile'] != M.M.G.SELECTION_PROFILE):
        raise ValueError('Current independent zero-failure238 LCC audit must bind the exact prediction/population')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):
        M._verify(audit[key])
    return protocol


def score(*, root_score_GO=False, expected_central_plan_sha256=None,
          expected_prediction_seal_sha256=None, source_peer_binding=None, output=M.OUTPUT):
    if root_score_GO is not True:
        raise PermissionError('ROOT separate V33 score GO before all IO/reference access')
    if (not M._sha(expected_central_plan_sha256) or not M._sha(expected_prediction_seal_sha256)
            or not isinstance(source_peer_binding,dict) or set(source_peer_binding) != {'path','sha256'}
            or not isinstance(source_peer_binding['path'],str) or not M._sha(source_peer_binding['sha256'])):
        raise ValueError('Explicit central/prediction/source peer SHA bindings before IO')
    output = Path(output).resolve()
    names = ['SCORING_PROTOCOL.json','SCORE_SEAL.json']+[part+'_SCORE.json' for part in (*PARTITIONS,'PRIMARY')]
    if any((output/name).exists() for name in names):
        raise FileExistsError('Preserve every V33 score attempt')
    peer = _source_peer(source_peer_binding)
    protocol = _prediction_gate(output,expected_central_plan_sha256,expected_prediction_seal_sha256)
    compact = compact_predictions(C.rows(output/'PREDICTIONS.jsonl.gz'),C.rows(output/'E_COMPARISON.jsonl.gz'),protocol['ordered_ids'])
    bindings, refplan, inputs = _reference_metadata(protocol,compact)
    stage_files = ('PREDICTION_SEAL.json','PREDICTIONS.jsonl.gz','PROTOCOL.json','INPUT_SEAL.json',
        'PREDICTION_EXECUTION_PROTOCOL.json','PREDICTION_ROW_INDEX.json','E_COMPARISON.jsonl.gz',
        'NATIVE_AND_E_PARITY_V1.json','independent_root_audit_v1/AUDIT.json','independent_root_audit_v1/AUDIT_SEAL.json')
    inputs = M._union(inputs,{M._relative(output/name):C.sha(output/name) for name in stage_files},
        {M._relative(peer):C.sha(peer)})
    sources = M._union(protocol['source_sha256'],refplan['source_sha256'],_sources())
    protected = M._union(protocol['protected_user_sha256'],refplan['protected_user_sha256'])
    M._verify(sources,inputs,protected)
    C.write(output/'SCORING_PROTOCOL.json',dict(schema='FIXED_REFERENCE_ALL_ROW_LCC_OUTLINE_SCORING_PROTOCOL_V33',
        central_plan_sha256=expected_central_plan_sha256,prediction_seal_sha256=expected_prediction_seal_sha256,
        source_peer_binding=source_peer_binding,source_sha256=sources,input_sha256=inputs,protected_user_sha256=protected,
        reference_bindings=bindings,published_reference_anchor_sha256=ANCHOR_SHA256,
        outline_selection_profile=SELECTION_PROFILE,native_observation_family=M.M.G.OBSERVATION_FAMILY,
        native_noise_population=M.M.G.POPULATION,native_selection_profile=M.M.G.SELECTION_PROFILE,
        native_CAL32_models_relabelled=False,all238_all5_and_exact_e_verified_before_numeric_references=True,
        headline_joint_rule='translation_cm<=5 and rotation_deg<=5',missing_pose_is_failure_not_dropped=True,
        numeric_reference_decodes_before_this_protocol=0))
    reports = score_rows(compact,_read_references(compact,refplan))
    for part,report in reports.items():
        report['reference_binding'] = deepcopy(bindings if part=='PRIMARY' else bindings[part])
        report['outline_selection_profile'] = SELECTION_PROFILE
        report['native_line_population_and_CAL32_noise_unchanged'] = True
        report['original_TEST49_preservation_is_required_for_adoption'] = True
        C.write(output/(part+'_SCORE.json'),report)
    _prediction_gate(output,expected_central_plan_sha256,expected_prediction_seal_sha256)
    M._verify(sources,inputs,protected)
    C.write(output/'SCORE_SEAL.json',dict(schema='SEALED_FIXED_REFERENCE_LCC_OUTLINE_CONTROL_SCORES_V33',
        prediction_seal=M._binding(output/'PREDICTION_SEAL.json'),scoring_protocol_sha256=C.sha(output/'SCORING_PROTOCOL.json'),
        score_sha256={part:C.sha(output/(part+'_SCORE.json')) for part in reports},reference_bindings=bindings,
        reference_anchor_sha256=ANCHOR_SHA256,outline_selection_profile=SELECTION_PROFILE,
        native_CAL32_models_relabelled=False,all_numeric_references_decoded_only_after_all238x5_seal=True,
        all_rows_kept=True,default='SAME_FROZEN_E unchanged',new_parameter_or_default_selection_by_metrics=False))
    return M._binding(output/'SCORE_SEAL.json')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-score-GO',action='store_true')
    parser.add_argument('--central-plan-sha256',required=True)
    parser.add_argument('--prediction-seal-sha256',required=True)
    parser.add_argument('--source-peer',required=True)
    parser.add_argument('--source-peer-sha256',required=True)
    parser.add_argument('--output',type=Path,default=M.OUTPUT)
    args = parser.parse_args()
    print(score(root_score_GO=args.root_score_GO,expected_central_plan_sha256=args.central_plan_sha256,
        expected_prediction_seal_sha256=args.prediction_seal_sha256,
        source_peer_binding=dict(path=args.source_peer,sha256=args.source_peer_sha256),output=args.output))
