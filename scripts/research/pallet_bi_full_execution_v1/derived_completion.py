"""ROOT-launched, serial derived outputs after the full S2/F final sequence.

This controller imports only metadata helpers. Children reuse the existing
postseal contrast, scalar-export and privacy-scoped display programs. It never
chooses F, trains, solves, scores a pose, or opens raw references/RGB itself.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import sys
import time

from . import final_sequence as F

NS = F.NS
SEEDS = F.SEEDS
PREFIXES = ['S1_UNION', 'S2_C0', 'S2_CP', 'S2_CB', 'S3_COMMON', 'S3_BI',
            'S4_PL', 'S4_CONTOUR', 'S4_IMAGE', 'F_COMMON', 'F_BI']
PAIRS = [('S2_C0', 'S2_CP', 'covariance_floor'), ('S2_CP', 'S2_CB', 'role_calibration'),
         ('S1_UNION', 'S3_COMMON', 'common_material_proposals'), ('S1_UNION', 'S3_BI', 'role_material_proposals'),
         ('S3_COMMON', 'S3_BI', 'S3_role_policy'), ('S1_UNION', 'S4_PL', 'PL_proposals'),
         ('S4_PL', 'S4_CONTOUR', 'contour_minus_PL'), ('S4_CONTOUR', 'S4_IMAGE', 'image_minus_contour'),
         ('S4_PL', 'S4_IMAGE', 'image_minus_PL'), ('F_COMMON', 'F_BI', 'final_matched_BI_effect')]
GALLERIES = [('S2', 'S2', ['S2_C0', 'S2_CP', 'S2_CB'], None),
             ('F', 'F', ['F_COMMON', 'F_BI'], None),
             ('F_BI_VS_COMMON', 'F', ['F_BI'], 'F_COMMON'),
             ('S2_CP_VS_C0', 'S2', ['S2_CP'], 'S2_C0'),
             ('S2_CB_VS_CP', 'S2', ['S2_CB'], 'S2_CP')]
STAGES = ['contrasts', 'export', *['gallery_'+g[0] for g in GALLERIES], 'scatter']
DEPENDENCIES = ['derived_completion.py', 'final_sequence.py', 'post_contrasts.py',
                'export_results.py', 'comparison_figures.py', 'stage_gallery.py',
                's1_analysis.py', 's1_union.py', 'f_core_resume.py', 'independent_check.py']


def expected_arms():
    return sorted(F.BASE + [a+'_s'+str(s) for a in PREFIXES for s in SEEDS])


def validate_contrast_plan(value):
    wanted = [dict(name=n+'_s'+str(s), before=a+'_s'+str(s), after=b+'_s'+str(s))
              for a, b, n in PAIRS for s in SEEDS]
    if (value['contrasts'] != wanted or value['populations'] != F.POPS or
            value['bootstrap_resamples'] != 10000 or value['recording_clusters_PRIMARY'] != 6):
        raise ValueError('All30 originally declared contrasts/populations/10000 draws required')


def scatter_pairs(previous):
    """Keep prior display choices, then cover every declared comparison."""
    result = deepcopy(previous)
    keys = {(p['before'], p['after'], p.get('before_seeded', True), p.get('after_seeded', True)) for p in result}
    def add(stage, name, before, after):
        key = before, after, True, True
        if key not in keys:
            result.append(dict(stage=stage, name=name, before=before, after=after));keys.add(key)
    for arm in PREFIXES:
        add('F' if arm.startswith('F_') else arm.split('_')[0], arm+'_vs_original', 'D1_H1_MLP', arm)
    for before, after, name in PAIRS:
        add('F' if after.startswith('F_') else after.split('_')[0], name, before, after)
    if len({(p['stage'], p['name']) for p in result}) != len(result):
        raise ValueError('Unique display stage/pair names required')
    return result


def prepare(sequence_plan, sequence_GO, previous_figures, contrast_plan, output, *, reference_projection=True):
    sequence_plan, sequence_GO, previous_figures, contrast_plan, output = map(
        Path, (sequence_plan, sequence_GO, previous_figures, contrast_plan, output))
    if output.exists():
        raise FileExistsError('Prepare a fresh derived-completion namespace')
    sequence = F.read(sequence_plan);go = F.read(sequence_GO)
    if (go.get('approved') is not True or go['plan_sha256'] != F.sha(sequence_plan) or
            go['controller_sha256'] != F.sha(F.__file__)):
        raise ValueError('Reviewed upstream sequence PLAN/GO required')
    F.guard(sequence)
    validate_contrast_plan(F.read(contrast_plan))
    previous = F.read(previous_figures)
    if len(previous['sources']) != 2:
        raise ValueError('Explicit completed S1 and S3/S4 source bundles required')
    folder = Path(__file__).resolve().parent
    sources = dict(sequence['sources'])
    for name in DEPENDENCIES:
        sources[str(folder/name)] = F.sha(folder/name)
    privacy_source = folder.parent/'pallet_boundary_action_repair_v1/display_privacy.py'
    sources[str(privacy_source)] = F.sha(privacy_source)
    fixed = {str(p.resolve()): F.sha(p) for p in (sequence_plan, sequence_GO, previous_figures, contrast_plan)}
    for name in ('RIGHTS_SCOPE_BINDING.json', 'DISPLAY_PRIVACY_PLAN.json', 'RIGHTS_RECEIPT.json'):
        p = Path(sequence['d1'])/name;fixed[str(p)] = F.sha(p)
    plan = dict(schema='FULL_CAMPAIGN_DERIVED_COMPLETION_PLAN_V1', controller_sha256=F.sha(__file__),
        sequence_plan=str(sequence_plan.resolve()), sequence_GO=str(sequence_GO.resolve()),
        sequence_output=sequence['output'], previous_figures=str(previous_figures.resolve()),
        contrast_plan=str(contrast_plan.resolve()), output=str(output.resolve()), run=sequence['run'],
        d1=sequence['d1'], reference_dir=sequence['reference_dir'], worktree=sequence['worktree'],
        python=sys.executable, sources=sources, fixed_inputs=fixed, stage_order=STAGES,
        requested_unique_arms=expected_arms(), scalar_arms=40, contrasts=30,
        scatter_pairs=scatter_pairs(previous['pairs']), reference_projection=reference_projection,
        resource=dict(sequential_child_workers=1, BLAS_OMP_MKL=1, launch_only_after_full_sequence=True),
        new_H1_fits=0, new_pose_solves=0, new_pose_metric_evaluations=0,
        display_scope='existing PRIMARY192 rights/privacy; all3seed transitions; no CAL RGB; no public poses/GT',
        child_programs_unchanged=True)
    F.write(output/'PLAN.json', plan)
    return plan


def guard(plan):
    F.verify_map(plan['sources']);F.verify_map(plan['fixed_inputs'])
    F.guard(F.read(plan['sequence_plan']))


def completed_upstream(plan):
    sequence = F.read(plan['sequence_plan'])
    return F.completed_sequence(plan['sequence_plan'], plan['sequence_GO'], sequence)


def source_bundle(evaluation, independent):
    evaluation, independent = Path(evaluation), Path(independent)
    es = F.read(evaluation/'EVALUATION_SEAL.json');check = F.read(independent/'INDEPENDENT_VERIFICATION.json')
    if (not es.get('complete') or not check.get('PASS') or check.get('errors') != [] or
            check['canonical_evaluation_seal_sha256'] != F.sha(evaluation/'EVALUATION_SEAL.json') or
            check['rows'] != 238 or check['arms'] != es['arms'] or check['populations'] != F.POPS or
            es['populations'] != F.POPS or check['reference_sha256'] != es['reference_sha256'] or
            check['prediction_bindings'] != es['prediction_bindings'] or
            check['success_classifications_exact'] is not True):
        raise ValueError('Full matching canonical/independent bundle required')
    for name in ('SCORED_ROWS.jsonl.gz', 'SUMMARY.json', 'PAIRED.json'):
        if F.sha(evaluation/name) != es['artifacts'][name]:
            raise ValueError('Canonical saved artifact changed')
    for row in es['prediction_bindings']:
        path = Path(row['path']);seal = F.read(path.parent/'PREDICTION_SEAL.json')
        if (F.sha(path) != row['sha256'] or F.sha(path.parent/'PREDICTION_SEAL.json') != row['seal_sha256'] or
                not seal.get('complete')):
            raise ValueError('Sealed evaluated prediction changed')
    return dict(evaluation_dir=str(evaluation), independent_dir=str(independent),
                independent_artifacts={n: F.sha(independent/n) for n in F.INDEPENDENT})


def gallery_manifest(bank, names, baseline=None):
    """Copy the full final bank declarations, including explicit method filters."""
    if any(a not in bank['arms'] for a in [*names, *([baseline] if baseline else [])]):
        raise ValueError('Gallery arm is absent from completed bank manifest')
    value = dict(schema='FULL_CAMPAIGN_POSTSEAL_DERIVED_GALLERY_V1',
        arms={a: deepcopy(bank['arms'][a]) for a in names},
        scope='Every PRIMARY recovery/damage ID over all3seeds plus existing normal/failure/tail rules')
    for arm, spec in value['arms'].items():
        if spec['backend'] != 'fullnormal' or 'generated_feature_seal' not in spec or 'result_root' not in spec:
            raise ValueError('Own fullnormal bank and generated core ancestry required')
    if baseline:
        value['comparison_baseline'] = dict(deepcopy(bank['arms'][baseline]), arm=baseline)
    return value


def generated_inputs(plan):
    out, run = Path(plan['output']), Path(plan['run']);sequence = F.read(plan['sequence_plan'])
    old = F.read(plan['previous_figures']);sources = []
    for source in old['sources']:
        actual = source_bundle(source['evaluation_dir'], source['independent_dir'])
        if actual != source:
            raise ValueError('Previously bound S1/S3/S4 independent bundle changed')
        sources.append(actual)
    for prefix in ('S2', 'F'):
        sources.append(source_bundle(run/(prefix+'_EVALUATION'), run/(prefix+'_INDEPENDENT')))
    arms = set();reference = None;bindings = {}
    for source in sources:
        es = F.read(Path(source['evaluation_dir'])/'EVALUATION_SEAL.json')
        arms.update(es['arms'])
        if reference is not None and reference != es['reference_sha256']:
            raise ValueError('Completed sources used different canonical reference hashes')
        reference = es['reference_sha256']
        for folder, names in ((Path(source['evaluation_dir']), ['EVALUATION_SEAL.json', 'SCORED_ROWS.jsonl.gz', 'SUMMARY.json', 'PAIRED.json']),
                              (Path(source['independent_dir']), F.INDEPENDENT)):
            bindings.update({str(folder/n): F.sha(folder/n) for n in names})
    if sorted(arms) != expected_arms():
        raise ValueError('Exactly all40 unique baseline and per-seed arms required')
    pairs = plan['scatter_pairs']
    rep = next(p for p in pairs if p['before'] == 'F_COMMON' and p['after'] == 'F_BI')
    manifest = out/'inputs/ALL_SOURCES.json'
    F.write(manifest, dict(schema='FULL_CAMPAIGN_COMPLETED_SOURCES_V1', sources=sources, pairs=pairs,
        representative=dict(stage=rep['stage'], pair=rep['name'], seed=SEEDS[0]),
        seeds='separate named arms; never pooled as independent frame samples'))
    bindings[str(manifest)] = F.sha(manifest)
    banks = {p: F.read(run/(p+'_POST_BANK_MANIFEST.json')) for p in ('S2', 'F')}
    finish = F.read(Path(sequence['F_finish'])/'FINISH_RECEIPT.json')
    if not finish['complete'] or finish['H1_fits'] != 6:
        raise ValueError('All own6Ffits must already be sealed')
    chosen = finish['selected_configuration']
    for arm, spec in banks['F']['arms'].items():
        if spec.get('extra_rows', []) != F.extra_declarations(sequence['F_finish'], spec['feature_seal']):
            raise ValueError('Gallery auxiliary methods differ from actual F union')
        admitted = {m for item in spec.get('extra_rows', []) for m in item['methods']}
        if chosen in ('S3_COMMON', 'S3_BI', 'S4_PL', 'S4_CONTOUR', 'S4_IMAGE'):
            if chosen not in admitted:
                raise ValueError('Chosen generic display stage was not an actual F union input')
            spec.update(support_root=str(run/'s3_s4_production/frames/real'),
                support_archive_seal=str(run/'s3_s4_archive_report/ROW_ARCHIVE_SEAL.json'), stage=chosen)
    gallery_paths = {}
    for key, prefix, names, baseline in GALLERIES:
        path = out/'inputs'/('GALLERY_'+key+'.json')
        value = gallery_manifest(banks[prefix], names, baseline)
        value['parent_manifest_sha256'] = F.sha(run/(prefix+'_POST_BANK_MANIFEST.json'))
        F.write(path, value);bindings[str(path)] = F.sha(path);gallery_paths[key] = path
    for prefix in ('S2', 'F'):
        for path in [run/(prefix+'_POST_BANK_MANIFEST.json'), run/(prefix+'_BANK_ANALYSIS')/'ANALYSIS_SEAL.json',
                     run/(prefix+'_BANK_ANALYSIS')/'INPUT_BINDINGS.json',
                     run/(prefix+'_BANK_ANALYSIS')/'BANK_COVERAGE_ROWS.jsonl.gz']:
            bindings[str(path)] = F.sha(path)
    return manifest, gallery_paths, bindings


def specifications(plan, manifest, galleries):
    output, run = Path(plan['output']), Path(plan['run']);python = plan['python']
    def command(module, *arguments):
        return [python, '-B', '-m', NS+module, *map(str, arguments)]
    specs = [dict(key='contrasts', folder=output/'ALL30_CONTRASTS', terminal='CONTRASTS_SEAL.json', manifest=manifest,
        argv=command('post_contrasts', '--manifest', manifest, '--plan', plan['contrast_plan'], '--output', output/'ALL30_CONTRASTS')),
        dict(key='export', folder=output/'PUBLIC_METRICS40', terminal='EXPORT_SEAL.json', manifest=manifest,
        argv=command('export_results', '--manifest', manifest, '--output', output/'PUBLIC_METRICS40'))]
    for key, prefix, names, baseline in GALLERIES:
        folder = output/('GALLERY_'+key)
        argv = command('stage_gallery', '--manifest', galleries[key], '--d1', plan['d1'],
            '--evaluation-dir', run/(prefix+'_EVALUATION'), '--coverage-dir', run/(prefix+'_BANK_ANALYSIS'),
            '--independent-dir', run/(prefix+'_INDEPENDENT'), '--output', folder, '--root-go')
        if plan['reference_projection']:
            argv += ['--reference-dir', plan['reference_dir']]
        specs.append(dict(key='gallery_'+key, folder=folder, terminal='GALLERY_MANIFEST.json',
                          manifest=galleries[key], argv=argv))
    specs.append(dict(key='scatter', folder=output/'FULL_SCATTER', terminal='FIGURE_MANIFEST.json', manifest=manifest,
        argv=command('comparison_figures', '--manifest', manifest, '--output', output/'FULL_SCATTER', '--root-go')))
    return specs


def native(plan, spec):
    folder, key = Path(spec['folder']), spec['key'];bindings = {}
    seal = F.bound(folder/spec['terminal'], bindings)
    module = 'stage_gallery' if key.startswith('gallery_') else dict(contrasts='post_contrasts', export='export_results', scatter='comparison_figures')[key]
    source = str(Path(__file__).parent/(module+'.py'))
    if seal.get('complete') is not True or seal['source_sha256'] != plan['sources'][source]:
        raise ValueError('Completed derived child source differs')
    artifacts = seal['artifact_sha256'] if module == 'stage_gallery' else seal['artifacts']
    for name, digest in artifacts.items():
        path = folder/name
        if F.sha(path) != digest:
            raise ValueError('Completed derived artifact changed')
        bindings[str(path)] = digest
    if key == 'contrasts':
        if (not seal['PASS'] or seal['contrasts'] != 30 or seal['populations'] != F.POPS or
                seal['manifest_sha256'] != F.sha(spec['manifest']) or seal['plan_sha256'] != F.sha(plan['contrast_plan']) or
                seal['new_pose_calls'] != 0 or seal['new_fits'] != 0):
            raise ValueError('Full30 declared paired contrast contract differs')
    elif key == 'export':
        if (seal['arms'] != expected_arms() or seal['populations'] != F.POPS or seal['summary_rows'] != 160 or
                seal['scalar_row_rows'] != 9520 or seal['identities'] != 238 or seal['metrics_recomputed'] or
                seal['raw_references_decoded'] != 0 or seal['solve_fit_forward_calls'] != 0 or
                not seal['all_outliers_and_failures_retained'] or
                F.read(folder/'INPUT_BINDINGS.json')['manifest_sha256'] != F.sha(spec['manifest'])):
            raise ValueError('Public40arm saved-scalar contract differs')
    elif key.startswith('gallery_'):
        display_names = ('stage_gallery.py', 's1_analysis.py', 's1_union.py', 'display_privacy.py')
        expected_display = {n: next(v for p, v in plan['sources'].items() if Path(p).name == n) for n in display_names}
        if (seal['input_bindings']['manifest_sha256'] != F.sha(spec['manifest']) or
                seal['display_source_closure'] != expected_display or seal['reference_projection_requested'] != plan['reference_projection'] or
                seal['CAL_RGB_opened'] != 0 or seal['raw_RGB_copied'] or seal['public_reference_pose_or_GT'] or
                any(seal[k] != 0 for k in ('new_cost_evaluations', 'new_pose_calls', 'new_fits'))):
            raise ValueError('Derived display privacy/input contract differs')
        if (seal['privacy_plan_sha256'] != F.sha(Path(plan['d1'])/'DISPLAY_PRIVACY_PLAN.json') or
                seal['rights_scope_sha256'] != F.sha(Path(plan['d1'])/'RIGHTS_SCOPE_BINDING.json')):
            raise ValueError('Existing display rights/privacy ancestry differs')
        approved = set(F.read(Path(plan['d1'])/'RIGHTS_SCOPE_BINDING.json')['approved_ids'])
        if len(approved) != 192 or any(c['id'] not in approved for c in seal['cases']):
            raise ValueError('Gallery must remain within approved PRIMARY192')
    else:
        if (seal['bindings']['manifest_sha256'] != F.sha(spec['manifest']) or
                seal['datasets'] != len(plan['scatter_pairs'])*3 or seal['PRIMARY_frames_per_dataset'] != 192 or
                not seal['all_outliers_retained'] or
                any(seal[k] != 0 for k in ('new_metric_evaluations', 'new_reference_decodes', 'new_RGB_reads', 'new_forwards', 'new_pose_calls', 'new_fits'))):
            raise ValueError('Full saved-error scatter contract differs')
    return bindings


def public_contrasts(plan):
    """Whitelist saved paired IDs/scalars/cluster results, without reanalysis."""
    out = Path(plan['output']);private = out/'ALL30_CONTRASTS';public = out/'PUBLIC_CONTRASTS30'
    source, sp = private/'INDEPENDENT_PAIRED_CONTRASTS.json', private/'CONTRASTS_SEAL.json'
    seal = F.read(sp)
    if not seal['complete'] or not seal['PASS'] or seal['artifacts'][source.name] != F.sha(source):
        raise ValueError('Authenticated all30 paired results required for public copy')
    terminal = public/'PUBLIC_CONTRASTS_SEAL.json'
    if terminal.exists():
        saved = F.read(terminal)
        if (saved['source_sha256'] != F.sha(__file__) or saved['paired_input_sha256'] != F.sha(source) or
                saved['paired_seal_sha256'] != F.sha(sp)):
            raise ValueError('Completed public contrast ancestry changed')
        bindings = {str(terminal): F.sha(terminal)}
        for name, digest in saved['artifacts'].items():
            if F.sha(public/name) != digest:
                raise ValueError('Completed public contrast artifact changed')
            bindings[str(public/name)] = digest
        return bindings
    if public.exists() and any(public.iterdir()):
        raise RuntimeError('Partial public contrast copy preserved; ROOT review required')
    def scalar(value):
        if value is not None and type(value) not in (str, int, float, bool):
            raise ValueError('Public paired scalar expected')
        return value
    def fields(value, names):
        return {n: scalar(value[n]) for n in names}
    def identifiers(value):
        if not isinstance(value, list) or any(type(v) is not str for v in value):
            raise ValueError('Saved paired ID list expected')
        return list(value)
    def ci(value):
        result = fields(value, [n for n in ('reason', 'cluster_count', 'valid_resamples', 'empty_resamples', 'seed', 'resamples', 'unit', 'method') if n in value])
        result['CI95'] = None if value['CI95'] is None else [scalar(v) for v in value['CI95']]
        if 'clusters' in value:result['clusters'] = identifiers(value['clusters'])
        if 'cluster_sizes' in value:result['cluster_sizes'] = {str(k): scalar(v) for k, v in value['cluster_sizes'].items()}
        return result
    source_value = F.read(source);declared = F.read(plan['contrast_plan'])['contrasts'];result = {}
    if set(source_value) != set(F.POPS):raise ValueError('All4 public paired populations required')
    for pop, n in F.POPS.items():
        if set(source_value[pop]) != {p['name'] for p in declared}:raise ValueError('Every public population requires all30 contrasts')
        result[pop] = {}
        for pair in declared:
            value = source_value[pop][pair['name']]
            if value['denominator'] != n or value['before'] != pair['before'] or value['after'] != pair['after']:
                raise ValueError('Public paired denominator/arm differs')
            current = fields(value, ('before', 'after', 'denominator', 'success_difference_count', 'success_difference_percentage_points'))
            categories = ('BOTH_SUCCESS', 'RECOVERY', 'DAMAGE', 'BOTH_FAILURE')
            current['counts'] = fields(value['counts'], categories)
            current['ids'] = {k: identifiers(value['ids'][k]) for k in categories}
            all_ids = [i for k in categories for i in current['ids'][k]]
            if len(all_ids) != n or len(set(all_ids)) != n or any(len(current['ids'][k]) != current['counts'][k] for k in categories):
                raise ValueError('Every paired ID/failure/outlier must be retained once')
            current['deltas'] = {}
            for axis in ('success_rate', 'translation_cm', 'rotation_deg'):
                delta = value['deltas'][axis]
                current['deltas'][axis] = fields(delta, ('mean', 'common_available_n', 'improved', 'worsened'))
                for key in ('recording_cluster_CI', 'named_session_cluster_sensitivity_CI'):
                    current['deltas'][axis][key] = ci(delta[key])
            current['recording_clusters'] = {}
            for cluster, record in value['recording_clusters'].items():
                saved = fields(record, ('n', 'before', 'after', 'success_difference_count', 'delta_T_mean_cm', 'delta_R_mean_deg'))
                saved.update({k: identifiers(record[k]) for k in ('recovery', 'damage')})
                current['recording_clusters'][cluster] = saved
            current['leave_one_cluster_out'] = {cluster: fields(record, ('n', 'success_difference_count',
                'success_difference_percentage_points', 'delta_T_mean_cm', 'delta_R_mean_deg'))
                for cluster, record in value['leave_one_cluster_out'].items()}
            result[pop][pair['name']] = current
    F.write(public/'ALL_POPULATION_PAIRED_CONTRASTS.json', result)
    F.write(public/'CONTRAST_SCHEMA.json', dict(schema='PUBLIC_SAVED_PAIRED_CONTRASTS_V1', populations=F.POPS,
        contrasts=30, seeds='distinct paired arms; no independent frame pooling',
        values='whitelist copy from authenticated saved independent paired results; no statistics recalculated',
        contents='all transition IDs/counts, paired scalar deltas, recording/session cluster bootstrap, per-recording and leave-one-recording results',
        omitted='poses,reference values,weights and arbitrary additional private fields', independent_real_test_n=0))
    F.write(terminal, dict(complete=True, source_sha256=F.sha(__file__), paired_input_sha256=F.sha(source),
        paired_seal_sha256=F.sha(sp), new_metric_or_statistical_calculations=0, public_poses_or_GT=False,
        artifacts={n: F.sha(public/n) for n in ('ALL_POPULATION_PAIRED_CONTRASTS.json', 'CONTRAST_SCHEMA.json')}))
    return {str(p): F.sha(p) for p in public.iterdir() if p.is_file()}


def artifact_sizes(plan, specs):
    """Report actual bytes; never discard an artifact to satisfy a package cap."""
    output = Path(plan['output']);folders = [s['folder'] for s in specs]+[output/'PUBLIC_CONTRASTS30']
    files = {str(p.relative_to(output)): p.stat().st_size for folder in folders for p in Path(folder).rglob('*') if p.is_file()}
    value = dict(files=files, total_bytes=sum(files.values()), operational_4MiB_exceedances={p: n for p, n in files.items() if n > 4*1024*1024},
                 no_artifacts_dropped_or_truncated=True, note='ROOT sets an honest operational packaging cap after reading actual sizes')
    path = output/'ARTIFACT_SIZES.json';F.write(path, value)
    return path


def run(plan_path, GO_path):
    started, CPU_started = time.monotonic(), time.process_time()
    plan, go = F.read(plan_path), F.read(GO_path)
    if (go.get('approved') is not True or go.get('plan_sha256') != F.sha(plan_path) or
            go.get('controller_sha256') != F.sha(__file__) or go.get('new_H1_fits_authorized') != 0 or
            go.get('new_pose_solves_authorized') != 0 or go.get('sequential_child_workers') != 1 or
            go.get('allow_PRIMARY_derived_display') is not True):
        raise PermissionError('Reviewed source/plan-bound derived-completion ROOT GO required')
    guard(plan);out = Path(plan['output']);terminal = out/'DERIVED_COMPLETION_SEAL.json'
    if terminal.exists():
        value = F.read(terminal)
        if (not value.get('complete') or value['plan_sha256'] != F.sha(plan_path) or value['ROOT_GO_sha256'] != F.sha(GO_path) or
                value['source_sha256'] != F.sha(__file__) or value['stage_order'] != STAGES or
                value['actual_command_ledger_sha256'] != F.sha(out/'ACTUAL_COMMAND_LEDGER.jsonl')):
            raise ValueError('Completed derived controller ancestry differs')
        completed_upstream(plan)
        upstream = Path(plan['sequence_output'])/'FINAL_SEQUENCE_SEAL.json'
        if value['upstream_sequence_seal_sha256'] != F.sha(upstream):
            raise ValueError('Completed upstream sequence changed')
        specs = {s['key']: s for s in specifications(plan, out/'inputs/ALL_SOURCES.json',
            {key: out/'inputs'/('GALLERY_'+key+'.json') for key, *_ in GALLERIES})}
        if set(value['checkpoints']) != {k+'.json' for k in STAGES}:
            raise ValueError('All8 derived stages required')
        for key in STAGES:
            cp = out/'checkpoints'/(key+'.json')
            if F.sha(cp) != value['checkpoints'][cp.name]:
                raise ValueError('Completed derived checkpoint changed')
            record = F.read(cp)
            if (not record['complete'] or record['identity']['input_bindings']['PLAN.json'] != F.sha(plan_path) or
                    record['identity']['input_bindings']['ROOT_GO'] != F.sha(GO_path)):
                raise ValueError('Completed derived checkpoint PLAN/GO differs')
            F.verify_map(record['completed_files'])
            F.verify_map({p: v for p, v in record['identity']['input_bindings'].items() if Path(p).is_absolute()})
            if native(plan, specs[key]) != record['completed_files']:
                raise ValueError('Completed derived native seal differs')
        F.verify_map(value['public_contrasts_files'])
        if public_contrasts(plan) != value['public_contrasts_files'] or F.sha(out/'ARTIFACT_SIZES.json') != value['artifact_sizes_sha256']:
            raise ValueError('Completed public packaging metadata changed')
        return value
    with (out/'OWNER.lock').open('a+') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        upstream = Path(plan['sequence_output'])/'FINAL_SEQUENCE_SEAL.json'
        F.wait_for(upstream, lambda: completed_upstream(plan))
        guard(plan)
        manifest, galleries, bindings = generated_inputs(plan)
        inputs = {**bindings, 'PLAN.json': F.sha(plan_path), 'ROOT_GO': F.sha(GO_path), str(upstream): F.sha(upstream)}
        F.append_ledger(out, dict(event='CONTROLLER_START', source_sha256=F.sha(__file__), epoch=time.time(),
            actual_argv=[v.decode() for v in Path('/proc/self/cmdline').read_bytes().split(b'\0') if v],
            sequential_child_workers=1, new_H1_fits=0, new_pose_solves=0))
        specs = specifications(plan, manifest, galleries)
        for spec in specs:
            guard(plan)
            action = F.stage(out, spec['key'], spec['argv'], inputs, lambda: native(plan, spec),
                spec['folder']/spec['terminal'], spec['folder'], lambda k, a: F.execute(k, a, out, plan['worktree']))
            F.append_ledger(out, dict(event='CHECKPOINT_ACTION', phase=spec['key'], action=action,
                checkpoint_sha256=F.sha(out/'checkpoints'/(spec['key']+'.json')), epoch=time.time()))
        guard(plan)
        metadata_start, metadata_cpu = time.monotonic(), time.process_time()
        public_was_complete = (out/'PUBLIC_CONTRASTS30/PUBLIC_CONTRASTS_SEAL.json').exists()
        F.append_ledger(out, dict(event='METADATA_START', phase='public_contrasts_and_sizes',
            function_call='derived_completion.public_contrasts + artifact_sizes', subprocess=False, epoch=time.time()))
        try:
            public_files = public_contrasts(plan)
            sizes = artifact_sizes(plan, specs)
        except Exception as error:
            F.append_ledger(out, dict(event='METADATA_TECHNICAL_FAILURE', phase='public_contrasts_and_sizes',
                error=repr(error), wall_seconds=time.monotonic()-metadata_start,
                controller_CPU_seconds=time.process_time()-metadata_cpu, epoch=time.time()))
            raise
        F.append_ledger(out, dict(event='METADATA_COMPLETE', phase='public_contrasts_and_sizes',
            function_call='derived_completion.public_contrasts + artifact_sizes', subprocess=False,
            action='VERIFY_EXISTING' if public_was_complete else 'WHITELIST_COPY',
            wall_seconds=time.monotonic()-metadata_start, controller_CPU_seconds=time.process_time()-metadata_cpu,
            new_metric_calculations=0, epoch=time.time()))
        value = dict(complete=True, schema='FULL_CAMPAIGN_DERIVED_COMPLETION_SEAL_V1',
            source_sha256=F.sha(__file__), plan_sha256=F.sha(plan_path), ROOT_GO_sha256=F.sha(GO_path),
            upstream_sequence_seal_sha256=F.sha(upstream), stage_order=STAGES,
            checkpoints={p.name: F.sha(p) for p in sorted((out/'checkpoints').glob('*.json'))},
            actual_command_ledger_sha256=F.sha(out/'ACTUAL_COMMAND_LEDGER.jsonl'),
            generated_inputs_sha256={p: v for p, v in bindings.items() if str(out/'inputs') in p},
            public_contrasts_files=public_files, artifact_sizes_sha256=F.sha(sizes),
            contrasts=30, scalar_arms=40, scalar_population_rows=160, scalar_frame_arm_rows=9520,
            gallery_groups=5, gallery_seed_count=3, scatter_datasets=len(plan['scatter_pairs'])*3,
            new_H1_fits=0, new_pose_solves=0, new_pose_metric_evaluations=0,
            raw_RGB_copied=False, public_poses_or_GT=False, all_outliers_and_failures_retained=True,
            controller_wall_seconds=time.monotonic()-started, controller_CPU_seconds=time.process_time()-CPU_started,
            controller_timing_scope='run entry through serial children and metadata; child wall/CPU separately recorded; prepare/import excluded',
            pixel_review='PENDING_ROOT_DERIVED_PIXEL_REVIEW', epoch=time.time())
        F.write(terminal, value)
        return value


if __name__ == '__main__':
    p = argparse.ArgumentParser();sub = p.add_subparsers(dest='mode', required=True)
    a = sub.add_parser('prepare')
    for name in ('sequence-plan', 'sequence-GO', 'previous-figures', 'contrast-plan', 'output'):
        a.add_argument('--'+name, required=True, type=Path)
    a.add_argument('--without-reference-projection', action='store_true')
    a = sub.add_parser('run');a.add_argument('--plan', required=True, type=Path);a.add_argument('--root-GO', required=True, type=Path)
    args = p.parse_args()
    if args.mode == 'prepare':
        prepare(args.sequence_plan, args.sequence_GO, args.previous_figures, args.contrast_plan, args.output,
                reference_projection=not args.without_reference_projection)
    else:
        print(run(args.plan, args.root_GO))
