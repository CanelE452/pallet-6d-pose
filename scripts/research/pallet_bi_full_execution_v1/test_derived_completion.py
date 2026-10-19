"""Metadata fixtures only: no real score rows, reference/RGB, poses or children."""
from copy import deepcopy
from pathlib import Path
import json

import pytest

from scripts.research.pallet_bi_full_execution_v1 import derived_completion as D

F = D.F


def put(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+'\n')


def forbidden(*args, **kwargs):
    raise AssertionError('Constructed test forbids actual child/reference/RGB/pose calls')


def contrasts():
    return dict(contrasts=[dict(name=n+'_s'+str(s), before=a+'_s'+str(s), after=b+'_s'+str(s))
                          for a, b, n in D.PAIRS for s in D.SEEDS], populations=F.POPS,
                bootstrap_resamples=10000, recording_clusters_PRIMARY=6)


def old_pairs():
    pairs = [dict(stage='S1', name='UNION', before='D1_H1_MLP', after='S1_UNION')]
    for arm in ('S3_COMMON', 'S3_BI', 'S4_PL'):
        pairs.append(dict(stage=arm[:2], name=arm+'_original', before='D1_H1_MLP', after=arm))
    for before, after in [('S3_COMMON', 'S3_BI'), ('S4_PL', 'S4_CONTOUR'),
                          ('S4_PL', 'S4_IMAGE'), ('S4_CONTOUR', 'S4_IMAGE')]:
        pairs.append(dict(stage=after[:2], name=before+'_'+after, before=before, after=after))
    for before in ('POSEFIX_SUBPIX_PF_ANCHOR', 'POSEFIX_SUBPIX_YOLO_ANCHOR'):
        pairs.append(dict(stage='S4', name=before, before=before, after='S4_PL', before_seeded=False))
    return pairs


def bank_spec(root, arm, extras=None):
    return dict(backend='fullnormal', features=str(root/arm/'features.jsonl.gz'),
        feature_seal=str(root/arm/'FEATURE_SEAL.json'), generated_feature_seal=str(root/arm/'CORE_FEATURE_SEAL.json'),
        result_root=str(root/arm/'real'), prediction_dir=str(root/arm/'prediction'),
        extra_rows=deepcopy(extras or []))


def test_all40_arms_and_all30_fixed_contrasts_without_seed_pooling():
    arms = D.expected_arms()
    assert len(arms) == len(set(arms)) == 40
    for arm in D.PREFIXES:
        assert sum(a.startswith(arm+'_s') for a in arms) == 3
    D.validate_contrast_plan(contrasts())
    for field, value in [('bootstrap_resamples', 10), ('recording_clusters_PRIMARY', 5)]:
        bad = contrasts();bad[field] = value
        with pytest.raises(ValueError, match='All30'):
            D.validate_contrast_plan(bad)
    bad = contrasts();bad['contrasts'].pop()
    with pytest.raises(ValueError, match='All30'):
        D.validate_contrast_plan(bad)


def test_full_scatter_preserves_static_baselines_and_all_declared_pairs():
    original = old_pairs();saved = deepcopy(original);pairs = D.scatter_pairs(original)
    assert original == saved and pairs[:10] == original and len(pairs) == 23
    assert sum(p.get('before_seeded') is False for p in pairs) == 2
    for before, after, name in D.PAIRS:
        assert any(p['before'] == before and p['after'] == after for p in pairs)
    for arm in D.PREFIXES:
        assert any(p['before'] == 'D1_H1_MLP' and p['after'] == arm for p in pairs)


def test_gallery_copies_methods_and_fullnormal_ancestry_with_correct_prefix(tmp_path):
    extras = [dict(path='PF_mixed_archive', seal='PF_seal', methods=['POSEFIX_SUBPIX_YOLO_ANCHOR', 'POSEFIX_SUBPIX_PF_ANCHOR'])]
    bank = dict(arms={a: bank_spec(tmp_path, a, extras) for a in ('F_COMMON', 'F_BI')})
    value = D.gallery_manifest(bank, ['F_BI'], 'F_COMMON')
    assert value['comparison_baseline']['arm'] == 'F_COMMON'
    assert value['arms']['F_BI']['extra_rows'] == extras
    assert value['comparison_baseline']['generated_feature_seal'] == bank['arms']['F_COMMON']['generated_feature_seal']
    value['arms']['F_BI']['extra_rows'][0]['methods'].append('changed_copy_only')
    assert bank['arms']['F_BI']['extra_rows'] == extras
    del bank['arms']['F_BI']['generated_feature_seal']
    with pytest.raises(ValueError, match='ancestry'):
        D.gallery_manifest(bank, ['F_BI'])


def test_exact_eight_serial_commands_have_no_fit_solver_or_choice(tmp_path):
    plan = dict(output=str(tmp_path/'out'), run=str(tmp_path), python='python',
                d1='baseline', contrast_plan='fixed30', reference_dir='private_display_refs', reference_projection=True)
    galleries = {k: tmp_path/(k+'.json') for k, *_ in D.GALLERIES}
    specs = D.specifications(plan, tmp_path/'ALL_SOURCES.json', galleries)
    assert [s['key'] for s in specs] == D.STAGES and len(specs) == 8
    assert sum('stage_gallery' in ' '.join(s['argv']) for s in specs) == 5
    for spec in specs:
        text = ' '.join(spec['argv'])
        assert all(token not in text for token in ('.heads ', 'f_finish', '.fullnormal_', 'evaluation score'))
        if spec['key'].startswith('gallery_'):
            assert '--reference-dir' in spec['argv'] and '--independent-dir' in spec['argv']
    plan['reference_projection'] = False
    assert all('--reference-dir' not in s['argv'] for s in D.specifications(plan, tmp_path/'m', galleries))


def bundle(root, arms):
    ev, ind, pred = root/'evaluation', root/'independent', root/'prediction'
    pred.mkdir(parents=True)
    (pred/'PREDICTIONS.jsonl.gz').write_bytes(b'opaque sealed predictions; no pose decode')
    put(pred/'PREDICTION_SEAL.json', dict(complete=True))
    pb = [dict(path=str(pred/'PREDICTIONS.jsonl.gz'), sha256=F.sha(pred/'PREDICTIONS.jsonl.gz'),
               seal_sha256=F.sha(pred/'PREDICTION_SEAL.json'))]
    for folder, names in ((ev, ['SCORED_ROWS.jsonl.gz', 'SUMMARY.json', 'PAIRED.json']), (ind, F.INDEPENDENT)):
        folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            (folder/name).write_bytes(b'opaque saved artifact; never numerically decoded')
    put(ev/'EVALUATION_SEAL.json', dict(complete=True, arms=arms, populations=F.POPS,
        reference_sha256='same_private_reference_hash', prediction_bindings=pb,
        artifacts={n: F.sha(ev/n) for n in ('SCORED_ROWS.jsonl.gz', 'SUMMARY.json', 'PAIRED.json')}))
    put(ind/'INDEPENDENT_VERIFICATION.json', dict(PASS=True, errors=[], rows=238, arms=arms, populations=F.POPS,
        canonical_evaluation_seal_sha256=F.sha(ev/'EVALUATION_SEAL.json'),
        reference_sha256='same_private_reference_hash', prediction_bindings=pb, success_classifications_exact=True))
    return D.source_bundle(ev, ind)


def test_bundle_authenticates_canonical_independent_and_prediction_bytes_only(tmp_path):
    source = bundle(tmp_path, ['CONSTRUCTED'])
    assert set(source['independent_artifacts']) == set(F.INDEPENDENT)
    path = Path(source['evaluation_dir'])/'SUMMARY.json';path.write_text('changed')
    with pytest.raises(ValueError, match='artifact changed'):
        D.source_bundle(source['evaluation_dir'], source['independent_dir'])


def test_prepare_reads_only_metadata_and_preserves_existing_output(tmp_path, monkeypatch):
    d1 = tmp_path/'d1'
    for name in ('RIGHTS_SCOPE_BINDING.json', 'DISPLAY_PRIVACY_PLAN.json', 'RIGHTS_RECEIPT.json'):
        put(d1/name, dict(constructed=True))
    sequence = dict(sources={}, run=str(tmp_path), output=str(tmp_path/'sequence'), d1=str(d1),
                    reference_dir='RAW_REFERENCE_PATH_MUST_NOT_BE_OPENED', worktree=str(Path.cwd()))
    pp, gp, previous, cp = [tmp_path/n for n in ('SEQUENCE_PLAN.json', 'SEQUENCE_GO.json', 'previous.json', 'contrasts.json')]
    put(pp, sequence);put(gp, dict(approved=True, plan_sha256=F.sha(pp), controller_sha256=F.sha(F.__file__)))
    put(previous, dict(sources=[{}, {}], pairs=old_pairs()));put(cp, contrasts())
    monkeypatch.setattr(F, 'guard', lambda plan: None)
    monkeypatch.setattr(F, 'wait_for', forbidden);monkeypatch.setattr(F.subprocess, 'call', forbidden)
    out = tmp_path/'fresh'
    plan = D.prepare(pp, gp, previous, cp, out)
    assert plan['scalar_arms'] == 40 and plan['contrasts'] == 30 and len(plan['scatter_pairs']) == 23
    assert plan['new_H1_fits'] == plan['new_pose_solves'] == 0
    assert len(list(out.iterdir())) == 1 and (out/'PLAN.json').exists()
    before = (out/'PLAN.json').read_bytes()
    with pytest.raises(FileExistsError, match='fresh'):
        D.prepare(pp, gp, previous, cp, out)
    assert (out/'PLAN.json').read_bytes() == before


def generated_fixture(tmp_path, monkeypatch):
    run, out = tmp_path/'run', tmp_path/'out'
    groups = [['S1_UNION'], ['S3_COMMON', 'S3_BI', 'S4_PL', 'S4_CONTOUR', 'S4_IMAGE'],
              ['S2_C0', 'S2_CP', 'S2_CB'], ['F_COMMON', 'F_BI']]
    sources = []
    for i, prefixes in enumerate(groups):
        source = bundle(tmp_path/str(i), F.BASE+[a+'_s'+str(s) for a in prefixes for s in D.SEEDS])
        sources.append(source)
    for prefix, source in zip(('S2', 'F'), sources[2:]):
        # Use the exact locations the controller consumes; all files remain
        # opaque metadata fixtures, even when their extensions are .gz.
        for source_dir, target in [(Path(source['evaluation_dir']), run/(prefix+'_EVALUATION')),
                                   (Path(source['independent_dir']), run/(prefix+'_INDEPENDENT'))]:
            import shutil
            shutil.copytree(source_dir, target)
    old = tmp_path/'old.json';put(old, dict(sources=sources[:2], pairs=old_pairs()))
    extra = [dict(path='PF_mixed', seal='PF_seal', methods=['POSEFIX_SUBPIX_PF_ANCHOR', 'POSEFIX_SUBPIX_YOLO_ANCHOR']),
             dict(path='material_S3_BI', seal='material_seal', methods=['S3_BI'])]
    for prefix, names in [('S2', groups[2]), ('F', groups[3])]:
        put(run/(prefix+'_POST_BANK_MANIFEST.json'), dict(arms={a: bank_spec(run, a, extra if prefix == 'F' else []) for a in names}))
        for n in ('ANALYSIS_SEAL.json', 'INPUT_BINDINGS.json', 'BANK_COVERAGE_ROWS.jsonl.gz'):
            put(run/(prefix+'_BANK_ANALYSIS')/n, dict(opaque=True))
    finish = run/'F_FINALIZE';put(finish/'FINISH_RECEIPT.json', dict(complete=True, H1_fits=6, selected_configuration='S3_BI'))
    seq = tmp_path/'sequence.json';put(seq, dict(F_finish=str(finish)))
    plan = dict(output=str(out), run=str(run), sequence_plan=str(seq), previous_figures=str(old), scatter_pairs=D.scatter_pairs(old_pairs()))
    monkeypatch.setattr(F, 'extra_declarations', lambda *args: deepcopy(extra))
    return plan, extra


def test_generated_full_sources_and_F_support_keep_actual_method_filters(tmp_path, monkeypatch):
    plan, extra = generated_fixture(tmp_path, monkeypatch)
    manifest, galleries, bindings = D.generated_inputs(plan)
    assert len(F.read(manifest)['sources']) == 4
    matched = F.read(galleries['F_BI_VS_COMMON'])
    assert matched['comparison_baseline']['arm'] == 'F_COMMON'
    for spec in [matched['arms']['F_BI'], matched['comparison_baseline']]:
        assert spec['extra_rows'] == extra and spec['stage'] == 'S3_BI'
        assert spec['support_root'].endswith('s3_s4_production/frames/real')
    assert F.read(galleries['S2_CP_VS_C0'])['comparison_baseline']['arm'] == 'S2_C0'
    assert F.read(galleries['S2_CB_VS_CP'])['comparison_baseline']['arm'] == 'S2_CP'
    assert len([p for p in bindings if 'GALLERY_' in p]) == 5


def test_missing_global_arm_cannot_be_omitted_or_zero_filled(tmp_path, monkeypatch):
    plan, extra = generated_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(D, 'expected_arms', lambda: ['intentionally_missing'])
    with pytest.raises(ValueError, match='all40'):
        D.generated_inputs(plan)


def test_ROOT_GO_rejected_before_guard_wait_or_children(tmp_path, monkeypatch):
    pp, gp = tmp_path/'PLAN.json', tmp_path/'GO.json';put(pp, {});put(gp, dict(approved=False))
    monkeypatch.setattr(D, 'guard', forbidden);monkeypatch.setattr(F, 'wait_for', forbidden)
    monkeypatch.setattr(F.subprocess, 'call', forbidden)
    with pytest.raises(PermissionError):
        D.run(pp, gp)


def run_fixture(tmp_path, monkeypatch):
    out = tmp_path/'controller';out.mkdir()
    upstream = tmp_path/'sequence/FINAL_SEQUENCE_SEAL.json';put(upstream, dict(complete=True))
    plan = dict(output=str(out), sequence_output=str(upstream.parent), worktree='constructed',
                scatter_pairs=D.scatter_pairs(old_pairs()))
    pp, gp = out/'PLAN.json', tmp_path/'GO.json';put(pp, plan)
    put(gp, dict(approved=True, plan_sha256=F.sha(pp), controller_sha256=F.sha(D.__file__),
        new_H1_fits_authorized=0, new_pose_solves_authorized=0, sequential_child_workers=1,
        allow_PRIMARY_derived_display=True))
    monkeypatch.setattr(D, 'guard', lambda plan: None)
    monkeypatch.setattr(D, 'completed_upstream', lambda plan: {'complete': True})
    monkeypatch.setattr(F, 'wait_for', lambda path, validator: validator())
    mp = out/'inputs/ALL_SOURCES.json';put(mp, dict(constructed=True))
    monkeypatch.setattr(D, 'generated_inputs', lambda plan: (mp, {}, {str(mp): F.sha(mp)}))
    specs = [dict(key=k, folder=out/k, terminal='SEAL.json', manifest=mp, argv=['metadata_child', k]) for k in D.STAGES]
    monkeypatch.setattr(D, 'specifications', lambda *args: specs)
    def native(plan, spec):
        p = spec['folder']/spec['terminal']
        assert F.read(p)['complete']
        return {str(p): F.sha(p)}
    monkeypatch.setattr(D, 'native', native)
    calls = []
    def execute(key, argv, output, worktree):
        calls.append(key)
        spec = next(s for s in specs if s['key'] == key)
        put(spec['folder']/spec['terminal'], dict(complete=True))
    monkeypatch.setattr(F, 'execute', execute)
    monkeypatch.setattr(F.subprocess, 'call', forbidden)
    def public(plan):
        path = out/'PUBLIC_CONTRASTS30/PUBLIC_CONTRASTS_SEAL.json'
        if not path.exists():put(path, dict(complete=True, constructed=True))
        return {str(path): F.sha(path)}
    monkeypatch.setattr(D, 'public_contrasts', public)
    return pp, gp, out, specs, calls


def test_waits_for_upstream_before_generated_inputs_or_ledger(tmp_path, monkeypatch):
    pp, gp, out, specs, calls = run_fixture(tmp_path, monkeypatch)
    def waiting(path, validator):
        raise RuntimeError('Constructed pending upstream')
    monkeypatch.setattr(F, 'wait_for', waiting)
    monkeypatch.setattr(D, 'generated_inputs', forbidden)
    with pytest.raises(RuntimeError, match='pending upstream'):
        D.run(pp, gp)
    assert calls == [] and not (out/'ACTUAL_COMMAND_LEDGER.jsonl').exists()


def test_all8_metadata_checkpoints_skip_without_new_calls_or_ledger_mutation(tmp_path, monkeypatch):
    pp, gp, out, specs, calls = run_fixture(tmp_path, monkeypatch)
    result = D.run(pp, gp)
    assert calls == D.STAGES and result['contrasts'] == 30 and result['scalar_arms'] == 40
    assert result['scatter_datasets'] == 69 and result['pixel_review'].startswith('PENDING')
    ledger = (out/'ACTUAL_COMMAND_LEDGER.jsonl').read_bytes()
    assert D.run(pp, gp) == result and calls == D.STAGES
    assert (out/'ACTUAL_COMMAND_LEDGER.jsonl').read_bytes() == ledger
    put(specs[0]['folder']/'SEAL.json', dict(complete=True, changed=True))
    with pytest.raises(ValueError, match='Frozen'):
        D.run(pp, gp)


def test_partial_derived_output_is_preserved_and_never_reexecuted(tmp_path, monkeypatch):
    pp, gp, out, specs, calls = run_fixture(tmp_path, monkeypatch)
    sentinel = specs[0]['folder']/'partial.dat';sentinel.parent.mkdir();sentinel.write_bytes(b'preserve')
    with pytest.raises(RuntimeError, match='Partial output preserved'):
        D.run(pp, gp)
    assert calls == [] and sentinel.read_bytes() == b'preserve'


@pytest.mark.parametrize('key', ['contrasts', 'export', 'scatter', 'gallery_S2'])
def test_native_completion_formats_and_denominator_privacy_guards(tmp_path, key):
    folder = tmp_path/'native';manifest = tmp_path/'manifest.json';put(manifest, dict(constructed=True))
    contrast_plan = tmp_path/'contrast_plan.json';put(contrast_plan, contrasts())
    source_names = ['post_contrasts.py', 'export_results.py', 'comparison_figures.py', 'stage_gallery.py',
                    's1_analysis.py', 's1_union.py']
    parent = Path(D.__file__).parent
    plan = dict(sources={str(parent/n): 'source:'+n for n in source_names}, contrast_plan=str(contrast_plan),
                scatter_pairs=D.scatter_pairs(old_pairs()), reference_projection=True, d1=str(tmp_path/'d1'))
    plan['sources']['/constructed/display_privacy.py'] = 'source:display_privacy.py'
    term = dict(contrasts='CONTRASTS_SEAL.json', export='EXPORT_SEAL.json', scatter='FIGURE_MANIFEST.json', gallery_S2='GALLERY_MANIFEST.json')[key]
    module = dict(contrasts='post_contrasts', export='export_results', scatter='comparison_figures', gallery_S2='stage_gallery')[key]
    put(folder/'artifact.json', dict(opaque_saved_metadata=True))
    seal = dict(complete=True, source_sha256='source:'+module+'.py', artifacts={'artifact.json': F.sha(folder/'artifact.json')})
    if key == 'contrasts':
        seal.update(PASS=True, contrasts=30, populations=F.POPS, manifest_sha256=F.sha(manifest),
                    plan_sha256=F.sha(contrast_plan), new_pose_calls=0, new_fits=0)
    elif key == 'export':
        put(folder/'INPUT_BINDINGS.json', dict(manifest_sha256=F.sha(manifest)))
        seal.update(arms=D.expected_arms(), populations=F.POPS, summary_rows=160, scalar_row_rows=9520,
                    identities=238, metrics_recomputed=False, raw_references_decoded=0, solve_fit_forward_calls=0,
                    all_outliers_and_failures_retained=True)
    elif key == 'scatter':
        seal.update(bindings=dict(manifest_sha256=F.sha(manifest)), datasets=69, PRIMARY_frames_per_dataset=192,
                    all_outliers_retained=True, **{k: 0 for k in ('new_metric_evaluations', 'new_reference_decodes',
                        'new_RGB_reads', 'new_forwards', 'new_pose_calls', 'new_fits')})
    else:
        rights = tmp_path/'d1/RIGHTS_SCOPE_BINDING.json';privacy = tmp_path/'d1/DISPLAY_PRIVACY_PLAN.json'
        put(rights, dict(approved_ids=[str(i) for i in range(192)]));put(privacy, dict(constructed=True))
        seal.update(input_bindings=dict(manifest_sha256=F.sha(manifest)),
            artifact_sha256=seal.pop('artifacts'), display_source_closure={n: 'source:'+n for n in
                ('stage_gallery.py', 's1_analysis.py', 's1_union.py', 'display_privacy.py')},
            reference_projection_requested=True, CAL_RGB_opened=0, raw_RGB_copied=False, public_reference_pose_or_GT=False,
            new_cost_evaluations=0, new_pose_calls=0, new_fits=0, cases=[dict(id='0')],
            privacy_plan_sha256=F.sha(privacy), rights_scope_sha256=F.sha(rights))
    put(folder/term, seal)
    spec = dict(key=key, folder=folder, terminal=term, manifest=manifest)
    assert str(folder/term) in D.native(plan, spec)
    if key == 'contrasts':seal['contrasts'] = 29
    elif key == 'export':seal['scalar_row_rows'] = 9519
    elif key == 'scatter':seal['all_outliers_retained'] = False
    else:seal['public_reference_pose_or_GT'] = True
    put(folder/term, seal)
    with pytest.raises(ValueError):D.native(plan, spec)


def paired_fixture(tmp_path):
    cp = tmp_path/'plan.json';put(cp, contrasts());results = {}
    for pop, n in F.POPS.items():
        results[pop] = {}
        ids = [f'{pop}_{i}' for i in range(n)]
        for pair in contrasts()['contrasts']:
            ci = dict(CI95=[0., 0.], cluster_count=1, reason='constructed metadata')
            delta = dict(mean=0., common_available_n=n, improved=0, worsened=0,
                         recording_cluster_CI=ci, named_session_cluster_sensitivity_CI=ci)
            results[pop][pair['name']] = dict(before=pair['before'], after=pair['after'], denominator=n,
                counts=dict(BOTH_SUCCESS=n, RECOVERY=0, DAMAGE=0, BOTH_FAILURE=0),
                ids=dict(BOTH_SUCCESS=ids, RECOVERY=[], DAMAGE=[], BOTH_FAILURE=[]),
                success_difference_count=0, success_difference_percentage_points=0.,
                deltas={k: deepcopy(delta) for k in ('success_rate', 'translation_cm', 'rotation_deg')},
                recording_clusters=dict(REC_001=dict(n=n, before=n, after=n, recovery=[], damage=[],
                    success_difference_count=0, delta_T_mean_cm=0., delta_R_mean_deg=0.,
                    private_extra=dict(actual_pose='MUST_NOT_EXPORT'))),
                leave_one_cluster_out=dict(REC_001=dict(n=0, success_difference_count=0,
                    success_difference_percentage_points=None, delta_T_mean_cm=None, delta_R_mean_deg=None)),
                poses={'R_physical': 'MUST_NOT_EXPORT'}, weights=[123])
    private = tmp_path/'ALL30_CONTRASTS'
    put(private/'INDEPENDENT_PAIRED_CONTRASTS.json', results)
    put(private/'CONTRASTS_SEAL.json', dict(complete=True, PASS=True,
        artifacts={'INDEPENDENT_PAIRED_CONTRASTS.json': F.sha(private/'INDEPENDENT_PAIRED_CONTRASTS.json')}))
    return dict(output=str(tmp_path), contrast_plan=str(cp)), results


def test_public_contrast_copy_retains_full_ids_clusters_and_whitelists_private_fields(tmp_path):
    plan, raw = paired_fixture(tmp_path)
    bound = D.public_contrasts(plan)
    public = tmp_path/'PUBLIC_CONTRASTS30/ALL_POPULATION_PAIRED_CONTRASTS.json'
    value = F.read(public)
    assert set(value) == set(F.POPS)
    assert 'MUST_NOT_EXPORT' not in public.read_text() and 'R_physical' not in public.read_text()
    for pop, n in F.POPS.items():
        assert len(value[pop]) == 30
        for name, result in value[pop].items():
            assert len(result['ids']['BOTH_SUCCESS']) == n
            assert result['recording_clusters']['REC_001']['n'] == n
            assert 'REC_001' in result['leave_one_cluster_out']
    before = {p: Path(p).read_bytes() for p in bound}
    assert D.public_contrasts(plan) == bound
    assert all(Path(p).read_bytes() == data for p, data in before.items())


def test_public_contrast_missing_id_is_rejected_and_large_files_reported_not_trimmed(tmp_path):
    plan, raw = paired_fixture(tmp_path)
    one = next(iter(raw['PRIMARY192'].values()));one['ids']['BOTH_SUCCESS'].pop()
    private = tmp_path/'ALL30_CONTRASTS';put(private/'INDEPENDENT_PAIRED_CONTRASTS.json', raw)
    put(private/'CONTRASTS_SEAL.json', dict(complete=True, PASS=True,
        artifacts={'INDEPENDENT_PAIRED_CONTRASTS.json': F.sha(private/'INDEPENDENT_PAIRED_CONTRASTS.json')}))
    with pytest.raises(ValueError, match='retained once'):D.public_contrasts(plan)
    file = tmp_path/'FULL_SCATTER/full.json';file.parent.mkdir();file.write_bytes(b'x'*(4*1024*1024+1))
    report = D.artifact_sizes(plan, [dict(folder=file.parent)])
    assert F.read(report)['operational_4MiB_exceedances'] == {'FULL_SCATTER/full.json': 4*1024*1024+1}
    assert file.stat().st_size == 4*1024*1024+1
