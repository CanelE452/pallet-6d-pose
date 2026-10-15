"""V10 closed-evidence publication planning V4; never changes Git or experiments.

``draft`` scans paths and opaque bytes only. ``build`` is a separate ROOT GO
after full V43 scores/recount/image closure; it checks small sealed metadata,
but never decodes INPUTS/E/PRED/OBS, masks, references or model values.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ast
import hashlib
import json
import re
import subprocess


ROOT=Path('/home/minjae/Documents/github/pallet-pose')
WT=Path('/tmp/pallet-visibility-main-publication-20261015-v1')
D=Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1')
B=D/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
S=Path('scripts/research/pallet_visibility_curriculum_20261013_v1')
PUB=D/'publication_followup_20261015_v10'
EVAL=B/'largest_component_global_outline_pose_eval_v4_33'
OBS=B/'largest_component_global_outline_observations_v4_33'
RECOUNT=B/'independent_largest_component_global_outline_score_review_v4_33'
PIXEL32=B/'clean_fresh_BI_containment_pixel_diagnosis_v4_32_r2'
PIXEL32_INITIAL=B/'clean_fresh_BI_containment_pixel_diagnosis_v4_32_v1'
CENTRAL=B/'largest_component_global_outline_method_plan_v4_33/PREDECLARATION.json'
CENTRAL_SHA256='44e8b7835be40246997d76603e96a8b28ec80826fc16afd3be72d57478d290de'
PIXEL32_SEAL_SHA256='5034084086512d0d495afc14791524d05465673c31f9c5422eb30035cc8c4814'
ARMS=('SAME_FROZEN_E','CLEAN_FRESH_BI_ALL_BI_NATIVE','CLEAN_FRESH_BI_ALL_BI_CONTAINMENT',
      'CLEAN_FRESH_BI_SOURCE_NATIVE','CLEAN_FRESH_BI_SOURCE_CONTAINMENT')
FORBIDDEN_PARTS={'code_snapshot','code_snapshot_v2','source_snapshot','final_source_snapshot',
    '__pycache__','row_spool','prediction_spool','frames','rows','rows_REST'}
ALLOWED_GZIP={'POSES.jsonl.gz','SELECTED_POSE_COMPONENTS.jsonl.gz'}
FORBIDDEN_NAMES={'INPUTS.jsonl.gz','E_COMPARISON.jsonl.gz','PREDICTIONS.jsonl.gz',
    'OBSERVATIONS.jsonl.gz','CAL_OBSERVATIONS.jsonl.gz','REST_OBSERVATIONS.jsonl.gz',
    'OUTLINE_OBSERVATIONS.jsonl.gz','CAL_OUTLINE_OBSERVATIONS.jsonl.gz','REST_OUTLINE_OBSERVATIONS.jsonl.gz',
    'LIVE_PROGRESS.json','CAL_LIVE_PROGRESS.json','REST_LIVE_PROGRESS.json'}
MAX_FILE_BYTES=50*1024*1024
FINAL_NAMES={'CLOSED_PAYLOAD_MANIFEST_V1.json','PAYLOAD_VALIDATION_V1.json',
    'CLOSED_FILES_TO_PUBLISH_V1.txt','FORCE_ADD_FILES_V1.txt'}


def sha(path):
    h=hashlib.sha256()
    with (ROOT/path).open('rb')as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads((ROOT/path).read_text())
def write(path,value):(ROOT/path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def permitted(path):
    path=Path(path)
    return (not path.is_absolute() and '..'not in path.parts
        and not any(part in FORBIDDEN_PARTS for part in path.parts)
        and path.name not in FORBIDDEN_NAMES and path.suffix not in ('.pyc','.pending','.npz','.npy','.log')
        and (path.suffix!='.gz' or path.name in ALLOWED_GZIP))


def is_snapshot(path):
    return any(part.startswith(('code_snapshot','source_snapshot','final_source_snapshot'))for part in Path(path).parts)


def source_publication_status():
    """Only source maps/files and read-only Git HEAD/WT are inspected."""
    mapping={};anchors=[];protected=set()
    for path in (CENTRAL,OBS/'FINAL_PROTOCOL.json',EVAL/'PROTOCOL.json'):
        if not(ROOT/path).is_file():continue
        metadata=read(path);anchors.append(dict(path=str(path),sha256=sha(path)))
        protected.update(metadata.get('protected_user_sha256',{}))
        for name,digest in metadata['source_sha256'].items():
            if name in mapping and mapping[name]!=digest:raise ValueError('Source closure conflict '+name)
            mapping[name]=digest
    rows=[]
    for name,digest in sorted(mapping.items()):
        if name in protected:continue
        if sha(name)!=digest:raise ValueError('Immutable current source changed '+name)
        result=subprocess.run(['git','show','HEAD:'+name],cwd=WT,capture_output=True)
        head=None if result.returncode else hashlib.sha256(result.stdout).hexdigest()
        target=WT/name
        current=hashlib.sha256(target.read_bytes()).hexdigest()if target.is_file()else None
        rows.append(dict(path=name,sha256=digest,published_HEAD_sha256=head,current_publication_WT_sha256=current,
            published_HEAD_exact=head==digest,publication_WT_exact=current==digest,
            supplement_current_source=head!=digest))
    return dict(metadata_anchors=anchors,rows=rows,count=len(rows),
        exact_HEAD_reuse=sum(row['published_HEAD_exact']for row in rows),
        missing_or_different_HEAD=sum(row['supplement_current_source']for row in rows),
        excluded_protected_user_paths=sorted(protected),git_mutations=False)


def candidates(source_status=None):
    """Planning candidates, not a claim that unfinished stages are closed."""
    result=set((S/p.name)for p in (ROOT/S).glob('*v4_33*.py'))
    folders=[p for p in (ROOT/B).iterdir()if p.is_dir() and 'largest_component'in p.name and 'v4_33'in p.name]
    folders += [ROOT/PIXEL32,ROOT/PIXEL32_INITIAL,ROOT/PUB]
    snapshots=[]
    for folder in folders:
        if folder.is_dir():
            for p in folder.rglob('*'):
                if not p.is_file():continue
                relative=p.relative_to(ROOT)
                if is_snapshot(relative):
                    if p.suffix=='.py':snapshots.append(relative)
                elif permitted(relative)and p.name not in FINAL_NAMES:result.add(relative)
    # Retain one exact changed historical source by SHA; omit identical live
    # dependencies and repeated whole-tree source snapshots.
    seen={sha(path)for path in result if path.suffix=='.py'}
    if source_status is not None:
        seen.update(row['sha256']for row in source_status['rows'])
    for path in sorted(snapshots):
        digest=sha(path);live=S/path.name
        # Whole-tree snapshots preserve their original repository-relative
        # source path; flat historical snapshots use the live S basename.
        for index,part in enumerate(path.parts):
            if part.startswith(('code_snapshot','source_snapshot','final_source_snapshot')):
                original=Path(*path.parts[index+1:])
                if (ROOT/original).is_file():live=original
                break
        if digest in seen or (ROOT/live).is_file()and sha(live)==digest:continue
        result.add(path);seen.add(digest)
    if source_status is not None:
        result.update(Path(row['path'])for row in source_status['rows']if row['supplement_current_source'])
    return sorted(path for path in result if permitted(path)or(path.suffix=='.py'and is_snapshot(path)))


def file_rows(paths):
    rows=[]
    for path in sorted(set(paths)):
        target=ROOT/path
        historical=path.suffix=='.py'and is_snapshot(path)
        if ((not permitted(path)and not historical)or target.is_symlink()or not target.is_file()
                or not target.resolve().is_relative_to(ROOT)or target.stat().st_size>=MAX_FILE_BYTES):
            raise ValueError('No live/raw/unsafe/large payload: '+str(path))
        superseded=path.name.endswith('_V1.json')and target.with_name(path.name.replace('_V1.json','_V2.json')).exists()
        historical=historical or any(part.startswith(('initial_source_attempt','initial_syntax_attempt'))for part in path.parts)
        rows.append(dict(path=str(path),bytes=target.stat().st_size,sha256=sha(path),
            classification='historical_frozen_source_not_current_runtime'if historical else
                'superseded_owner_metadata_preserved_not_current_source_validation'if superseded else'current_source_or_closed_evidence'))
    return rows


def draft(*,root_packaging_plan_GO=False):
    if root_packaging_plan_GO is not True:raise PermissionError('ROOT source/metadata packaging-plan GO before IO')
    status=source_publication_status();paths=candidates(status)
    # Avoid a self-referential draft list hash. The immutable draft manifest
    # records only source/metadata available at this planning instant.
    generated={PUB/'PACKAGING_PLAN_V4.json',PUB/'CANDIDATE_INVENTORY_DRAFT_V4.json',
        PUB/'CANDIDATE_FILES_DRAFT_V4.txt',PUB/'SOURCE_PUBLICATION_STATUS_DRAFT_V4.json'}
    if any((ROOT/path).exists()for path in generated):raise FileExistsError('Preserve every planning attempt')
    rows=file_rows(paths)
    write(PUB/'CANDIDATE_INVENTORY_DRAFT_V4.json',dict(schema='V10_PLANNING_CANDIDATES_NOT_FINAL_CLOSED_PAYLOAD',
        UTC=datetime.now(timezone.utc).isoformat(),files=rows,count=len(rows),bytes=sum(x['bytes']for x in rows),
        actual_metric_pose_mask_reference_or_model_decodes=0,full_result_publication_GO_received=False,
        limitations='Available path/hash inventory only. Mutable/pending V43 stages must close before final build; no result/accuracy claim.'))
    write(PUB/'SOURCE_PUBLICATION_STATUS_DRAFT_V4.json',status)
    (ROOT/PUB/'CANDIDATE_FILES_DRAFT_V4.txt').write_text(''.join(str(p)+'\n'for p in paths))
    plan=dict(schema='V10_APPEND_ONLY_CLOSED_PUBLICATION_PLAN',UTC=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p):sha(p)for p in (Path(__file__).resolve().relative_to(ROOT),
            PUB/'test_prepare_closed_payload_v4.py')},
        planned_groups=['All newV43 live source/tests and design/owner/independent/root reviews/GO records',
            'Central33 and closedCAL46+REST192/full outline small protocols/seals/indexes and independent pixel audits',
            'Full238x5 prediction/audit/score/recount seals and small reports, selected scalar components/compact final poses if available',
            'Closed actual V43 case process PNG/report/provenance and source plan when handed off',
            'Previously unpublished closed V42 containment pixel diagnosis and preserved initial failed schema attempt',
            'Explicit user-requested V42 191cm display/source and after-display asset provenance; no predeclaration claim'],
        required_before_final_build=['ROOT full-result publication GO','caller-pinned final report/score seal/independent audit/recount',
            'V43 whole238/native2+e exact invariant, all46+192 pixel audits, fixed denominators',
            'explicit closed image/compact/probe bindings if included, protected6 current bytes, link/AST/file hash check'],
        original_union_control='Published V42R2 retained unchanged; original native2/e results must remain identical',
        numerical_control='Uniform LCC8 mask observation only; oldC25/coefficient/native32population/noise/freshPnP/guards remain fixed',
        semantic_limit='Pixel transformation and algorithm audit do not certify semantic pallet/occluder identity or physical pose accuracy',
        prior_release_does_not_contain_new_V42_V43_raw_archives=True,
        raw_inputs_E_predictions_observations_spools_or_duplicate_snapshots_selected=False,
        report_numbers_pending_fullscore_ROOT_GO=True,git_index_WT_commit_push_operations=False)
    write(PUB/'PACKAGING_PLAN_V4.json',plan)
    return dict(candidates=len(rows),candidate_bytes=sum(x['bytes']for x in rows),
        plan_sha256=sha(PUB/'PACKAGING_PLAN_V4.json'),actual_data_decodes=0)


def _full_result_gate(score_sha,audit_sha,recount_sha,report_binding,extra_closed_bindings):
    # Small sealed metadata only; score rows and pose/mask datasets stay opaque.
    for path,digest in ((EVAL/'SCORE_SEAL.json',score_sha),(EVAL/'independent_root_audit_v1/AUDIT.json',audit_sha),
            (RECOUNT/'AUDIT.json',recount_sha),(CENTRAL,CENTRAL_SHA256),
            (PIXEL32/'PROVENANCE_SEAL_V2.json',PIXEL32_SEAL_SHA256)):
        if sha(path)!=digest:raise ValueError('Exact closed result/provenance binding required: '+str(path))
    score=read(EVAL/'SCORE_SEAL.json');pred=read(EVAL/'PREDICTION_SEAL.json')
    audit=read(EVAL/'independent_root_audit_v1/AUDIT.json');recount=read(RECOUNT/'AUDIT.json')
    if (pred['rows']!=238 or pred['arms']!=list(ARMS)or pred['arm_count']!=5
            or pred['all_declared_arms_complete']is not True or pred['no_population_drop']is not True
            or score['prediction_seal']['sha256']!=sha(EVAL/'PREDICTION_SEAL.json')or score['all_rows_kept']is not True
            or audit['rows']!=238 or audit['failure_count']!=0 or audit['failures']
            or audit['prediction_seal']['sha256']!=sha(EVAL/'PREDICTION_SEAL.json')
            or recount['failure_count']!=0 or recount['failures']or not recount['exact_native_and_e_score_invariance']):
        raise ValueError('Complete fivearm238 zeroaudit/recount/native-e parity required')
    for folder,receipt in ((EVAL/'independent_root_audit_v1',audit_sha),(RECOUNT,recount_sha)):
        if read(folder/'AUDIT_SEAL.json')['receipt_sha256']!=receipt:raise ValueError('Audit receipt seal mismatch')
    parity=read(EVAL/'NATIVE_AND_E_PARITY_V1.json')
    if (parity['rows']!=238 or parity['native_arm_payload_equalities']!=476
            or parity['e_payload_equalities']!=238 or parity['failure_count']!=0):raise ValueError('Native/e invariant missing')
    for folder,count in ((OBS/'independent_mask_outline_audit_v1',46),(OBS/'independent_REST_mask_outline_audit_v1',192)):
        # The REST operator/auditor must use the explicitly named closed path;
        # a later path-contract correction needs an additive source version.
        result=read(folder/'AUDIT.json');seal=read(folder/'AUDIT_SEAL.json')
        if result['rows']!=count or result['failure_count']!=0 or result['failures']or seal['report_sha256']!=sha(folder/'AUDIT.json'):
            raise ValueError('Both complete independent pixel/outline audits required')
    for part,digest in score['score_sha256'].items():
        if sha(EVAL/(part+'_SCORE.json'))!=digest:raise ValueError('Saved score changed')
    bindings={**extra_closed_bindings,report_binding['path']:report_binding['sha256']}
    for path,digest in bindings.items():
        if not permitted(path)or sha(path)!=digest:raise ValueError('Caller-pinned closed report/image/compact binding changed')
    return score,audit,recount



FINAL_GROUP_SEALS={'RENDER_SEAL.json','SEAL.json','SCORE_SEAL.json','OUTLINE_OBSERVATION_SEAL.json',
    'AUDIT_SEAL.json','PROVENANCE_SEAL_V2.json','PROVENANCE.json'}
KNOWN_SMALL_OUTPUTS={'PLAN.json','SUMMARY.json','PROTOCOL.json','FINAL_PROTOCOL.json','PREPARATION.json',
    'REST_PREPARATION.json','EXECUTION_START.json','REST_EXECUTION_START.json','CAL_ROW_INDEX.json',
    'REST_ROW_INDEX.json','OUTLINE_ROW_INDEX.json','REST_SUMMARY.json','INPUT_SEAL.json',
    'PREDICTION_EXECUTION_PROTOCOL.json','PREDICTION_ROW_INDEX.json','PREDICTION_SEAL.json',
    'NATIVE_AND_E_PARITY_V1.json','SCORING_PROTOCOL.json','CAL_SCORE.json','TEST_SCORE.json',
    'REMAINDER_SCORE.json','PRIMARY_SCORE.json','AUDIT.json','REPORT_KO.md','EXECUTION_RECEIPT_V1.json'}


def runtime_group(path):
    path=Path(path)
    try:parts=path.relative_to(B).parts
    except ValueError:
        # Runtime displays/probes outside the experiment tree obey the same
        # asset contract; a folder name cannot bypass completion checks.
        try:parts=path.relative_to(D).parts
        except ValueError:return None
        if path.is_relative_to(PUB):
            return PUB if path.suffix.lower()in ('.png','.jpg','.jpeg','.gz')or path.name.startswith('PROCESS')else None
        return D/parts[0]if len(parts)>=2 and path.suffix!='.py'else None
    if len(parts)<2:return None
    group=B/parts[0]
    if group==PIXEL32_INITIAL:
        # Explicit historical schema-failure evidence; never a successful run.
        if path.name not in {'PLAN.json','PLAN_SEAL.json','FAILED_ATTEMPT_V1.json','render_stored_containment.py'}:
            raise ValueError('Unknown historical failure output '+str(path))
        return None
    if path.suffix=='.py':return None
    source_metadata=path.suffix=='.json'and(path.name.startswith(('SOURCE_','OWNER_'))
        or path.name.startswith('ROOT_')and any(token in path.name for token in ('GO','SOURCE','REVIEW','GATE'))
        or path.name in {'PREDECLARATION.json','PLAN.json','PLAN_SEAL.json'})
    if source_metadata:return None
    return group


def _sealed_assets(document,folder,sealname):
    result={}
    for key in ('assets','asset_sha256','artifact_sha256'):
        for name,digest in document.get(key,{}).items():
            if isinstance(digest,str)and re.fullmatch('[0-9a-f]{64}',digest):
                relative=Path(name)
                path=relative if relative.is_relative_to(D)or relative.is_relative_to(S)else folder/relative
                result[str(path)]=digest
    # Fixed scalar metadata names used by our closed diagnostic/audit seals.
    names={'report_sha256':'REPORT_KO.md','evidence_sha256':'EVIDENCE.json','summary_sha256':'SUMMARY.json',
        'components_sha256':'SELECTED_POSE_COMPONENTS.jsonl.gz','result_sha256':'RESULT.json',
        'provenance_evidence_sha256':'PROVENANCE_EVIDENCE_V2.json',
        'provenance_report_sha256':'PROVENANCE_ADDENDUM_KO_V2.md','base_report_sha256':'REPORT_KO.md',
        'PNG_sha256':'STORED_C_SAMPLES_AND_FINAL_HULLS.png','previous_execution_seal_sha256':'EXECUTION_SEAL_V1.json'}
    if sealname.startswith('EXECUTION_SEAL'):names['receipt_sha256']='EXECUTION_RECEIPT_V1.json'
    elif sealname=='AUDIT_SEAL.json':names['receipt_sha256']='AUDIT.json'
    for key,name in names.items():
        digest=document.get(key)
        if isinstance(digest,str)and re.fullmatch('[0-9a-f]{64}',digest):result[str(folder/name)]=digest
    return result


def validate_runtime_closure(selected,bindings):
    """Every discovered actual output is caller-bound and group-seal linked.

    Source/design/owner/peer/GO histories remain planning evidence. Runtime
    PNG/PROCESS/compact/probe files never enter via a broad folder glob alone.
    """
    groups={}
    for path in selected:
        group=runtime_group(path)
        if group is not None:groups.setdefault(group,[]).append(Path(path))
    covered=[]
    for group,paths in groups.items():
        explicit={Path(name):digest for name,digest in bindings.items()if Path(name).is_relative_to(group)}
        if any(str(path)not in bindings for path in paths):
            raise ValueError('Unbound discovered runtime output; caller must bind every file: '+str(group))
        final=[path for path in explicit if path.parent==group and path.name in FINAL_GROUP_SEALS]
        if not final:raise ValueError('Missing explicit final group seal: '+str(group))
        assets={};seals=[]
        for path,digest in explicit.items():
            if not re.fullmatch('[0-9a-f]{64}',str(digest))or sha(path)!=digest:
                raise ValueError('Explicit closed runtime bytes changed '+str(path))
            if ('SEAL'in path.name or path.name=='PROVENANCE.json')and path.suffix=='.json':
                assets.update(_sealed_assets(read(path),path.parent,path.name));seals.append(path)
        # Explicit linked receipts can themselves carry the asset allowlist.
        for path in paths:
            if path.name.endswith('RECEIPT_V1.json')and assets.get(str(path))==bindings[str(path)]:
                assets.update(_sealed_assets(read(path),path.parent,path.name))
        for path in paths:
            digest=bindings[str(path)]
            named_metadata=(path.name in KNOWN_SMALL_OUTPUTS or 'SEAL'in path.name or path.name=='PROVENANCE.json'
                or path.name.endswith('.md')and 'REPORT'in path.name)
            if assets.get(str(path))!=digest and not named_metadata:
                raise ValueError('Runtime file not in sealed asset allowlist: '+str(path))
            # All image/PROCESS/compact/unknown payloads need an actual seal
            # asset match; a caller hash alone is not a completion certificate.
            if (path.suffix.lower()in ('.png','.jpg','.jpeg','.gz')or path.name.startswith('PROCESS'))and assets.get(str(path))!=digest:
                raise ValueError('Unsealed image/process/compact output '+str(path))
            covered.append(dict(path=str(path),sha256=digest,group=str(group),
                seal_asset_match=assets.get(str(path))==digest,known_named_metadata=named_metadata,
                explicit_group_seals=[str(p)for p in final]))
    return covered


def build(*,root_full_result_publication_GO=False,expected_score_seal_sha256=None,
          expected_audit_sha256=None,expected_recount_sha256=None,report_binding=None,extra_closed_bindings=None):
    if root_full_result_publication_GO is not True:raise PermissionError('Separate ROOT completed-result packaging GO before IO')
    for value in (expected_score_seal_sha256,expected_audit_sha256,expected_recount_sha256):
        if not isinstance(value,str)or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('Explicit closed result SHA before IO')
    if (not isinstance(report_binding,dict)or set(report_binding)!={'path','sha256'}
            or not isinstance(extra_closed_bindings,dict)):raise ValueError('Explicit final report and extra closed image/compact bindings required')
    if any((ROOT/PUB/name).exists()for name in FINAL_NAMES):raise FileExistsError('Preserve every final payload attempt')
    _full_result_gate(expected_score_seal_sha256,expected_audit_sha256,expected_recount_sha256,report_binding,extra_closed_bindings)
    source_status=source_publication_status()
    selected=set(candidates(source_status))|{Path(p)for p in extra_closed_bindings}|{Path(report_binding['path'])}
    runtime_closure=validate_runtime_closure(selected,extra_closed_bindings)
    generated={PUB/name for name in FINAL_NAMES};allpaths=selected|generated
    (ROOT/PUB/'CLOSED_FILES_TO_PUBLISH_V1.txt').write_text(''.join(str(p)+'\n'for p in sorted(allpaths)))
    result=subprocess.run(['git','check-ignore','--stdin'],cwd=ROOT,text=True,capture_output=True,
        input=''.join(str(p)+'\n'for p in sorted(allpaths)))
    if result.returncode not in (0,1):raise ValueError(result.stderr)
    (ROOT/PUB/'FORCE_ADD_FILES_V1.txt').write_text(result.stdout)
    selected|=generated-{PUB/'CLOSED_PAYLOAD_MANIFEST_V1.json',PUB/'PAYLOAD_VALIDATION_V1.json'}
    rows=file_rows(selected);paths={row['path']for row in rows}|{str(p)for p in generated}
    failures=[];AST=0;links=[]
    for row in rows:
        p=ROOT/row['path']
        if p.suffix=='.py':ast.parse(p.read_text());AST+=1
        if p.suffix=='.md':
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
                if '://'in target or target.startswith('#'):continue
                targetpath=(p.parent/target.split('#')[0]).resolve();relative=str(targetpath.relative_to(ROOT))
                covered=relative in paths
                if not covered and targetpath.is_file()and(WT/relative).is_file():
                    tracked=subprocess.run(['git','ls-files','--error-unmatch','--',relative],cwd=WT,capture_output=True).returncode==0
                    covered=tracked and hashlib.sha256((WT/relative).read_bytes()).hexdigest()==sha(relative)
                if not covered:failures.append('Missing unpublished local link '+relative)
                links.append(dict(source=row['path'],target=target,covered=covered))
    protected=read(CENTRAL)['protected_user_sha256']
    for path,digest in protected.items():
        if sha(path)!=digest:failures.append('Protected bytes changed '+path)
    if len(protected)!=6:failures.append('Original protected6 missing')
    manifest=dict(schema='V10_CLOSED_LCC8_AND_PRIOR_CONTAINMENT_DIAGNOSIS_PAYLOAD_V1',
        UTC=datetime.now(timezone.utc).isoformat(),count=len(rows),bytes=sum(row['bytes']for row in rows),files=rows,
        all_paths_to_publish=len(allpaths),protected_user_sha256=protected,report_binding=report_binding,
        source_publication_status=source_status,runtime_group_file_closure=runtime_closure,
        self_reference_exclusions=['CLOSED_PAYLOAD_MANIFEST_V1.json','PAYLOAD_VALIDATION_V1.json'],
        raw_INPUTS_E_PRED_OBS_spools_or_duplicate_snapshots_included=False,new_raw_in_prior_release_claim=False,
        root_copy_commit_push_owner=True,algorithm_audit_is_not_mask_semantic_or_accuracy_certification=True)
    write(PUB/'CLOSED_PAYLOAD_MANIFEST_V1.json',manifest)
    write(PUB/'PAYLOAD_VALIDATION_V1.json',dict(manifest_sha256=sha(PUB/'CLOSED_PAYLOAD_MANIFEST_V1.json'),
        failures=failures,source_AST_files=AST,local_links=links,forced_add_paths=result.stdout.splitlines(),
        maximum_file_bytes=max(row['bytes']for row in rows),numeric_dataset_decodes=0,git_index_WT_mutations=False))
    if failures:raise ValueError('Preserve failed publication attempt: '+str(len(failures)))
    return dict(paths=len(allpaths),bytes=manifest['bytes'],manifest_sha256=sha(PUB/'CLOSED_PAYLOAD_MANIFEST_V1.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('draft',))
    parser.add_argument('--root-packaging-plan-GO',action='store_true')
    args=parser.parse_args();print(draft(root_packaging_plan_GO=args.root_packaging_plan_GO))
