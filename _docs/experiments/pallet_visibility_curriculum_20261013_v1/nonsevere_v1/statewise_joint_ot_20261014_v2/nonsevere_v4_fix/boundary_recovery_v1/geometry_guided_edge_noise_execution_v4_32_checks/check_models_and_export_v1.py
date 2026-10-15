"""Owner post-fit artifact/export closure; no numerical likelihood refit."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scripts/research').is_dir())
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_visibility_curriculum_20261013_v1 import geometry_guided_edge_noise_calibration_v4_32 as M

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent / 'CAL_geometry_guided_edge_noise_v4_32'
META = HERE.parent / 'geometry_guided_native_evidence_v4_32_checks'
EXPECTED_SEAL = '4b7d20df18f62269360a873e530ee29f7535c9d3d4ec4c6b6f72e49b7be48ab7'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    checks = []
    def check(name, passes):
        checks.append(dict(check=name, passes=bool(passes)))
        if not passes:
            raise AssertionError(name)
    seal = read(MODEL / 'MODEL_SEAL.json')
    protocol = read(MODEL / 'PROTOCOL.json')
    prep = read(META / 'PREPARATION_SEAL.json')
    report = read(MODEL / 'CAL_LOGO_REPORT.json')
    check('exact model seal', sha(MODEL / 'MODEL_SEAL.json') == EXPECTED_SEAL)
    check('all seven fitted assets', len(seal['assets']) == 7)
    for relative, digest in seal['assets'].items():
        check('asset:' + relative, sha(MODEL / relative) == digest)
    check('prepared seal bound', seal['preparation_seal_sha256'] == sha(MODEL / 'PREPARATION_SEAL.json'))
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        mapping = protocol[key]
        if key in seal:
            check(key + ' model seal equality', seal[key] == mapping)
        for relative, digest in mapping.items():
            path = Path(relative)
            check(key + ':' + relative, sha(path if path.is_absolute() else ROOT / path) == digest)
    for relative, digest in prep['original_CAL_file_sha256'].items():
        check('CAL53:' + relative, sha(ROOT / relative) == digest)
    check('fixed inventories', seal['CAL_inventory'] == len(protocol['ids']) == 46
          and seal['compatible_inventory'] == 44 and len(prep['original_CAL_file_sha256']) == 53)
    check('strict generation', seal['observation_generation'] == M.OBSERVATION_FAMILY
          and seal['raw_observation_generation'] == M.RAW_OBSERVATION_GENERATION
          and seal['selection_profile'] == M.SELECTION_PROFILE and seal['noise_population'] == M.POPULATION
          and not seal['old_model_relabelled'] and not seal['default_adoption'])
    models, starts, unavailable, summary, exports = 0, 0, [], {}, {}
    optimizer_failures = []
    for policy in M.SOURCE_POLICIES:
        full_line = read(MODEL / policy / 'MODEL.json')
        full_tangent = read(MODEL / policy / 'TANGENT_MODEL.json')
        line_export = M.mixture_noise_parameters(full_line, source_policy=policy)
        tangent_export = M.tangent_variance_parameters(full_tangent, source_policy=policy)
        exports[policy] = dict(line=line_export, tangent=tangent_export)
        line_keys = ('fit_frame_count', 'fit_physical_edge_count', 'fit_alternative_count',
                     'fit_distinct_support_count', 'sigma_normal_px', 'sigma_sine',
                     'normal_variance_component_px2', 'angular_variance_component_sine2')
        tangent_keys = ('fit_frame_count', 'fit_tangent_count', 'gamma')
        summary[policy] = dict(line={key: full_line[key] for key in line_keys},
            tangent={key: full_tangent[key] for key in tangent_keys}, folds=[])
        inventory = [(None, dict(line=dict(status=report[policy]['line'], model=full_line),
                                 tangent=dict(status=report[policy]['tangent'], model=full_tangent)))]
        check(policy + ' four LOGO folds', len(report[policy]['LOGO_folds']) == 4)
        inventory += [(fold['heldout_session'], fold) for fold in report[policy]['LOGO_folds']]
        for held, block in inventory:
            fold_summary = dict(heldout_session=held, kinds={})
            for kind in ('line', 'tangent'):
                status, model = block[kind]['status'], block[kind]['model']
                label = f'{policy}/{held}/{kind}'
                if model is None:
                    unavailable.append(label)
                    check(label + ' unavailable no substitution', block[kind]['old_model_substituted'] is False)
                    continue
                models += 1
                check(label + ' CAL46/44', model['inventory_frame_count'] == 46
                      and model['compatible_inventory_count'] == 44 and model['fit_partition'] == 'CAL')
                check(label + ' heldout excluded', model['heldout_session'] == held
                      and (held is None or held not in model['fit_sessions'])
                      and not set(model['fit_ids']).intersection(model['excluded_heldout_ids']))
                check(label + ' final model digest', model['model_digest_sha256'] == M._digest(
                    {key: value for key, value in model.items() if key != 'model_digest_sha256'}))
                if kind == 'line':
                    exported = M.mixture_noise_parameters(model, source_policy=policy)
                    check(label + ' rho0', exported['normal_angle_correlation'] == 0.)
                    nll_key = 'training_mixture_nll'
                else:
                    exported = M.tangent_variance_parameters(model, source_policy=policy)
                    check(label + ' gamma/floor', exported['gamma'] >= 0 and exported['pixel_floor_variance_px2'] == 1.)
                    nll_key = 'training_NLL'
                searches = model['optimizer_searches']
                check(label + ' fixed three starts', len(searches) == 3)
                starts += len(searches)
                selected = model['selected_optimizer_start_index']
                check(label + ' selected stored minimum', 0 <= selected < 3
                      and abs(model[nll_key] - min(s[nll_key] for s in searches)) <= 1e-10)
                for index, search in enumerate(searches):
                    if not search['success']:
                        optimizer_failures.append(dict(model=label, start=index, saved_search=search))
                fold_summary['kinds'][kind] = dict(status=status, model_digest=model['model_digest_sha256'],
                    fit_frames=model['fit_frame_count'], optimizer_success=[s['success'] for s in searches],
                    selected_optimizer_start_index=selected,
                    sigma_normal_px=model.get('sigma_normal_px'), sigma_sine=model.get('sigma_sine'), gamma=model.get('gamma'))
            summary[policy]['folds'].append(fold_summary)
    check('all20 fitted models/60 starts', models == 20 and starts == 60 and not unavailable)
    result = dict(schema='V32_CAL_NOISE_OWNER_POSTFIT_ARTIFACT_EXPORT_CLOSURE_V1',
        UTC=datetime.now(timezone.utc).isoformat(), checks=len(checks), failures=0, check_results=checks,
        execution=dict(exec_session=50090, PID=462426, first_yield_ms=1000, exit_code=0,
            actual_fit_invocations=1, extra_fit_or_new_parameter_search=0),
        model_seal=dict(path=str((MODEL / 'MODEL_SEAL.json').relative_to(ROOT)), sha256=EXPECTED_SEAL),
        model_count=models, optimizer_start_count=starts, unavailable_models=unavailable,
        saved_optimizer_failures=optimizer_failures, summary=summary,
        current_maps=dict(source=len(protocol['source_sha256']), input=len(protocol['input_sha256']),
            protected=len(protocol['protected_user_sha256']), CAL_immutable=53),
        scope=dict(CAL_inventory=46, compatible=44, CAL_only_numeric_labels_in_authorized_fit=True,
            primary_or_TEST_REMAINDER_labels=0, new_RGB_NN_or_posefit=0,
            likelihood_or_gradient_independent_replay_in_this_check=False,
            frozen_sources_or_models_relabelled_or_modified=0, current_pipeline_adoption=0))
    receipt = HERE / 'FIT_POSTCHECK_RECEIPT_V1.json'
    if receipt.exists():
        raise FileExistsError('Preserve prior owner closure')
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    export = HERE / 'RUNTIME_PARAMETER_EXPORT_V1.json'
    export.write_text(json.dumps(dict(model_seal_sha256=EXPECTED_SEAL,
        observation_family=M.OBSERVATION_FAMILY, selection_profile=M.SELECTION_PROFILE,
        noise_population=M.POPULATION, policies=exports), ensure_ascii=False, indent=2) + '\n')
    (HERE / 'FIT_POSTCHECK_SEAL_V1.json').write_text(json.dumps(dict(receipt_sha256=sha(receipt),
        runtime_export_sha256=sha(export), audit_source_sha256=sha(Path(__file__)),
        model_seal_sha256=EXPECTED_SEAL), indent=2) + '\n')
    print(json.dumps(dict(checks=len(checks), failures=0, models=models, starts=starts,
        optimizer_failures=len(optimizer_failures), receipt_sha256=sha(receipt), export_sha256=sha(export),
        summary={policy: {key: value for key, value in block.items() if key != 'folds'} for policy, block in summary.items()})))


if __name__ == '__main__':
    main()
