"""Fake filesystem/source-publication fixtures only; no experiment reads."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec=importlib.util.spec_from_file_location('publication10_closed_payload_v4',Path(__file__).with_name('prepare_closed_payload_v5.py'))
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)


def test_both_planning_and_finalGO_precede_IO(monkeypatch):
    def no(*args,**kwargs):raise AssertionError('No IO beforeGO')
    monkeypatch.setattr(P,'candidates',no);monkeypatch.setattr(P,'sha',no)
    with pytest.raises(PermissionError):P.draft()
    with pytest.raises(PermissionError):P.build()
    with pytest.raises(ValueError,match='before IO'):P.build(root_full_result_publication_GO=True)


@pytest.mark.parametrize('name',['x/INPUTS.jsonl.gz','x/E_COMPARISON.jsonl.gz','x/PREDICTIONS.jsonl.gz',
    'x/OUTLINE_OBSERVATIONS.jsonl.gz','x/rows/01.json','x/rows_REST/001.json','x/prediction_spool/one.json',
    'x/row_spool/one.json','x/LIVE_PROGRESS.json','x/input.npz'])
def test_raw_large_runtime_input_paths_fail_closed(name):
    assert not P.permitted(name)


def test_compact_archives_allowed_but_other_gzip_is_not():
    assert P.permitted('closed/POSES.jsonl.gz')and P.permitted('closed/SELECTED_POSE_COMPONENTS.jsonl.gz')
    assert not P.permitted('closed/untyped.jsonl.gz')


def test_source_HEAD_exact_reuse_and_missing_dependency_selection(tmp_path,monkeypatch):
    monkeypatch.setattr(P,'ROOT',tmp_path);monkeypatch.setattr(P,'WT',tmp_path/'wt')
    central=Path('central.json');monkeypatch.setattr(P,'CENTRAL',central);monkeypatch.setattr(P,'OBS',Path('obs'))
    for name,body in [('exact.py','oldexact'),('new.py','newsource')]:
        (tmp_path/name).write_text(body)
    P.write(central,dict(source_sha256={name:P.sha(name)for name in ('exact.py','new.py')},protected_user_sha256={}))
    def git(args,**kwargs):
        return SimpleNamespace(returncode=0,stdout=b'oldexact')if args[-1]=='HEAD:exact.py'else SimpleNamespace(returncode=128,stdout=b'')
    monkeypatch.setattr(P.subprocess,'run',git)
    status=P.source_publication_status()
    assert status['count']==2 and status['exact_HEAD_reuse']==1 and status['missing_or_different_HEAD']==1
    assert [r['path']for r in status['rows']if r['supplement_current_source']]==['new.py']


def test_unique_historical_code_kept_duplicate_snapshots_omitted(tmp_path,monkeypatch):
    for name,value in [('ROOT',tmp_path),('B',Path('B')),('S',Path('S')),('PUB',Path('publication')),
            ('PIXEL32',Path('pixel')),('PIXEL32_INITIAL',Path('pixel_initial'))]:monkeypatch.setattr(P,name,value)
    (tmp_path/'B').mkdir();(tmp_path/'S').mkdir()
    (tmp_path/'S/test_v4_33.py').write_text('current')
    group=tmp_path/'B/largest_component_fake_v4_33/code_snapshot';group.mkdir(parents=True)
    (group/'test_v4_33.py').write_text('current')
    historical=group/'changed_v4_33.py';historical.write_text('old source')
    (tmp_path/'S/changed_v4_33.py').write_text('new source')
    rows=P.file_rows(P.candidates())
    names=[r['path']for r in rows]
    assert str(historical.relative_to(tmp_path))in names
    assert str((group/'test_v4_33.py').relative_to(tmp_path))not in names
    assert next(r for r in rows if r['path']==str(historical.relative_to(tmp_path)))['classification'].startswith('historical')


def test_other_namespace_current_source_snapshot_is_not_false_historical(tmp_path,monkeypatch):
    for name,value in [('ROOT',tmp_path),('B',Path('B')),('S',Path('S')),('PUB',Path('publication')),
            ('PIXEL32',Path('pixel')),('PIXEL32_INITIAL',Path('pixel_initial'))]:monkeypatch.setattr(P,name,value)
    (tmp_path/'B').mkdir();(tmp_path/'S').mkdir()
    live=tmp_path/'other/module.py';live.parent.mkdir();live.write_text('unchanged other namespace')
    snapshot=tmp_path/'B/largest_component_fake_v4_33/code_snapshot/other/module.py'
    snapshot.parent.mkdir(parents=True);snapshot.write_bytes(live.read_bytes())
    assert snapshot.relative_to(tmp_path)not in P.candidates()
    # A current dependency SHA is also sufficient even for a flat snapshot.
    flat=snapshot.parent.parent/'flat.py';flat.write_bytes(live.read_bytes())
    status=dict(rows=[dict(path='other/module.py',sha256=P.sha('other/module.py'),supplement_current_source=False)])
    assert flat.relative_to(tmp_path)not in P.candidates(status)


def test_unclosed_probe_and_partial_renderer_cannot_enter_final_runtime_payload(tmp_path,monkeypatch):
    monkeypatch.setattr(P,'ROOT',tmp_path);monkeypatch.setattr(P,'B',Path('B'))
    for group in ('largest_component_probe_v4_33','largest_component_process_v4_33_v1'):
        folder=tmp_path/'B'/group;folder.mkdir(parents=True)
        (folder/'image.png').write_bytes(b'synthetic image bytes')
        (folder/'PROCESS_DATA.json').write_text('{}')
        selected={Path('B')/group/'image.png',Path('B')/group/'PROCESS_DATA.json'}
        with pytest.raises(ValueError,match='Unbound'):
            P.validate_runtime_closure(selected,{})
        bindings={str(path):P.sha(path)for path in selected}
        with pytest.raises(ValueError,match='final group seal'):
            P.validate_runtime_closure(selected,bindings)


def test_explicit_seal_assets_required_for_png_process_and_unknown_payload(tmp_path,monkeypatch):
    monkeypatch.setattr(P,'ROOT',tmp_path);monkeypatch.setattr(P,'B',Path('B'))
    group=Path('B/largest_component_process_v4_33_v1');folder=tmp_path/group;folder.mkdir(parents=True)
    image=group/'image.png';process=group/'PROCESS_DATA.json';seal=group/'RENDER_SEAL.json'
    (tmp_path/image).write_bytes(b'synthetic PNG');(tmp_path/process).write_text('{}')
    P.write(seal,{'assets':{'image.png':P.sha(image),'PROCESS_DATA.json':P.sha(process)}})
    selected={image,process,seal};bindings={str(path):P.sha(path)for path in selected}
    assert len(P.validate_runtime_closure(selected,bindings))==3
    P.write(seal,{'assets':{'image.png':P.sha(image)}});bindings[str(seal)]=P.sha(seal)
    with pytest.raises(ValueError,match='allowlist'):
        P.validate_runtime_closure(selected,bindings)


def test_external_user_display_requires_actual_asset_provenance(tmp_path,monkeypatch):
    monkeypatch.setattr(P,'ROOT',tmp_path);monkeypatch.setattr(P,'B',Path('D/B'));monkeypatch.setattr(P,'D',Path('D'))
    group=Path('D/show_191cm_case_v1');(tmp_path/group).mkdir(parents=True)
    image=group/'image.png';report=group/'REPORT_KO.md';provenance=group/'PROVENANCE.json'
    (tmp_path/image).write_bytes(b'original saved display');(tmp_path/report).write_text('posthoc display')
    with pytest.raises(ValueError,match='Unbound'):
        P.validate_runtime_closure({image,report},{})
    P.write(provenance,dict(assets={'image.png':P.sha(image),'REPORT_KO.md':P.sha(report)},predeclaration_claim=False))
    paths={image,report,provenance};bindings={str(path):P.sha(path)for path in paths}
    result=P.validate_runtime_closure(paths,bindings)
    assert len(result)==3 and all(row['seal_asset_match']for row in result if row['path']!=str(provenance))
    # A group name or caller hash never certifies a partial image/probe.
    P.write(provenance,dict(assets={'REPORT_KO.md':P.sha(report)}));bindings[str(provenance)]=P.sha(provenance)
    with pytest.raises(ValueError,match='allowlist'):
        P.validate_runtime_closure(paths,bindings)


@pytest.mark.parametrize('relative',['D/B/largest_component_probe_checks_v4_33/RESULT.json',
    'D/other_display_v1/unknown.png'])
def test_unknown_check_folder_json_and_nonshow_external_image_cannot_bypass_closure(tmp_path,monkeypatch,relative):
    monkeypatch.setattr(P,'ROOT',tmp_path);monkeypatch.setattr(P,'B',Path('D/B'));monkeypatch.setattr(P,'D',Path('D'))
    path=Path(relative);(tmp_path/path).parent.mkdir(parents=True);(tmp_path/path).write_text('{}')
    with pytest.raises(ValueError,match='Unbound'):
        P.validate_runtime_closure({path},{})
    assert P.runtime_group(Path('D/B/largest_component_probe_checks_v4_33/SOURCE_OWNER_V1.json'))is None
