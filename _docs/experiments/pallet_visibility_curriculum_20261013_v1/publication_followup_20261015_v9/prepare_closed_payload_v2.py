"""Curated V9 closed evidence inventory only. No Git/index/WT mutations.

Explicit V9 names are used; prior manifests are never renamed/substituted.
The root report and earlier draft/bindings are immutable payloads. Large input,
E-comparison, full prediction and raw observation journals/spools are omitted.
"""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,re,subprocess

ROOT=Path('/home/minjae/Documents/github/pallet-pose')
WT=Path('/tmp/pallet-visibility-main-publication-20261015-v1')
D=Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1')
B=D/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
S=Path('scripts/research/pallet_visibility_curriculum_20261013_v1')
PUB=D/'publication_followup_20261015_v9'
EVAL=B/'clean_fresh_BI_pose_eval_v4_32_r2'
REPORT_SHA='a6c4253587005161426ab16f3cad46f3d8b86d3d992d0fd9f0979b46ceb064c7'
MANIFEST_NAME='CLOSED_PAYLOAD_MANIFEST_V2.json'
VALIDATION_NAME='PAYLOAD_VALIDATION_V2.json'
FILELIST_NAME='CLOSED_FILES_TO_PUBLISH_V2.txt'
FORCE_NAME='FORCE_ADD_FILES_V2.txt'


def sha(path):
    h=hashlib.sha256()
    with (ROOT/path).open('rb')as f:
        for v in iter(lambda:f.read(4*1024*1024),b''):h.update(v)
    return h.hexdigest()


def read(p):return json.loads((ROOT/p).read_text())
def write(p,x):(ROOT/p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def permitted(p):
    return not(any(z in p.parts for z in('code_snapshot','source_snapshot','final_source_snapshot','__pycache__','row_spool','prediction_spool','frames'))
        or p.name in('INPUTS.jsonl.gz','E_COMPARISON.jsonl.gz','PREDICTIONS.jsonl.gz','OBSERVATIONS.jsonl.gz','CAL_OBSERVATIONS.jsonl.gz','REST_OBSERVATIONS.jsonl.gz','LIVE_PROGRESS.json')
        or p.suffix in('.pending','.pyc'))


def main():
    if (ROOT/PUB/MANIFEST_NAME).exists():raise FileExistsError('Preserve V9 attempt')
    oldroot={str(p.relative_to(ROOT)):sha(p.relative_to(ROOT))for p in(ROOT/PUB).iterdir()if p.is_file()}
    if sha(PUB/'REPORT_KO_FINAL_V1.md')!=REPORT_SHA or read(PUB/'ROOT_REPORT_BINDING_V1.json')['report_sha256']!=REPORT_SHA:
        raise ValueError('Exact ROOT final report binding')
    selected={Path(name)for name in oldroot}
    for name in'''CAL_SCORE.json TEST_SCORE.json REMAINDER_SCORE.json PRIMARY_SCORE.json SCORING_PROTOCOL.json SCORE_SEAL.json
PROTOCOL.json INPUT_SEAL.json PREDICTION_EXECUTION_PROTOCOL.json PREDICTION_ROW_INDEX.json PREDICTION_SEAL.json
OWNER_POSITIVE_PREPARATION_RECEIPT_V1.json OWNER_POSITIVE_PREPARATION_SEAL_V1.json OWNER_PREDICTION_COMPLETION_RECEIPT_V1.json OWNER_PREDICTION_COMPLETION_SEAL_V1.json
ROOT_FIXED_REFERENCE_SCORE_GO_V1.json ROOT_INDEPENDENT_ALL238_AUDIT_GO_V1.json'''.split():selected.add(EVAL/name)
    folders=[EVAL/'independent_root_audit_v1',B/'independent_clean_fresh_BI_score_review_v4_32',
        B/'clean_fresh_BI_fixed_initial_pose_diagnosis_v4_32_v1',B/'clean_fresh_BI_full_result_process_v4_32_r2_v1',
        B/'clean_fresh_BI_full_result_process_plan_v4_32_r2_v1',B/'clean_fresh_BI_full_result_process_source_checks_v4_32_r2_v1',
        B/'clean_fresh_BI_fixed_failure_diagnosis_v4_32_v1',B/'clean_fresh_BI_compact_pose_archive_v4_32_r2_v1',
        B/'clean_fresh_BI_compact_pose_archive_plan_v4_32_r2_v1',B/'clean_fresh_BI_compact_pose_archive_checks_v4_32_r2_v1',
        B/'clean_fresh_BI_initial_guard_cost_probe_v4_32_v1']
    for folder in folders:
        if not(ROOT/folder).is_dir():raise FileNotFoundError(folder)
        selected.update(p.relative_to(ROOT)for p in(ROOT/folder).rglob('*')if p.is_file()and permitted(p.relative_to(ROOT)))
    for name in('render_clean_fresh_BI_full_result_process_v4_32_r2_v1','archive_compact_clean_fresh_BI_poses_v4_32_r2_v1'):
        selected.update((S/(name+'.py'),S/('test_'+name+'.py')))
    selected.add(S/'render_saved_full_result_roi_v4_32_r2_v1.py')
    selected.add(S/'probe_clean_initial_guard_cost_v4_32.py')
    if sha(EVAL/'PREDICTION_SEAL.json')!='eff80efdb1cb9c17f7eb13df50b953467f82bbe3415c8385b9aa146456948b48':raise ValueError('Exact final PRED')
    if sha(EVAL/'SCORE_SEAL.json')!='7bcc9bce3e8b14a87021955c513e3061dcb561cc309fabd61964e06efbcc6c80':raise ValueError('Exact final SCORE')
    if sha(B/'clean_fresh_BI_fixed_failure_diagnosis_v4_32_v1/CLOSURE_RECEIPT_V1.json')!='a17bb24c691f4dc6c2dfa304c49656ef3280eff0189d0bb09dacacdbed08feb6':raise ValueError('Closed fixed failure diagnostic')
    if sha(B/'clean_fresh_BI_initial_guard_cost_probe_v4_32_v1/SEAL.json')!='eaae76cdb2bc51c2e18c9878f065d4001dacfc74a301ac13a0872e12c35a9d73':raise ValueError('Closed initial guard probe')
    generated={PUB/n for n in(FILELIST_NAME,FORCE_NAME,MANIFEST_NAME,VALIDATION_NAME)}
    allpaths=selected|generated
    (ROOT/PUB/FILELIST_NAME).write_text(''.join(str(p)+'\n'for p in sorted(allpaths)))
    ignored=subprocess.run(['git','check-ignore','--stdin'],cwd=ROOT,input=''.join(str(p)+'\n'for p in sorted(allpaths)),text=True,capture_output=True)
    if ignored.returncode not in(0,1):raise ValueError(ignored.stderr)
    (ROOT/PUB/FORCE_NAME).write_text(ignored.stdout)
    selected|=generated-{PUB/MANIFEST_NAME,PUB/VALIDATION_NAME}
    rows=[]
    for p in sorted(selected):
        path=ROOT/p
        if not permitted(p)or path.is_symlink()or not path.is_file()or path.stat().st_size>=50*1024*1024:raise ValueError('Unsafe/live/large path '+str(p))
        rows.append(dict(path=str(p),bytes=path.stat().st_size,sha256=sha(p)))
    protected=read(D/'nonsevere_v1/remainder_after_test49_v1/repair_campaign_20261015_v1/ACTIVE_GOAL_PROTOCOL_V1.json')['protected_user_sha256']
    write(PUB/MANIFEST_NAME,dict(schema='CURATED_CLOSED_FULL_R2_RESULT_PAYLOAD_V9_V2',UTC=datetime.now(timezone.utc).isoformat(),
        count=len(rows),bytes=sum(r['bytes']for r in rows),files=rows,all_paths_to_publish=len(allpaths),
        self_reference_exclusions=[MANIFEST_NAME,VALIDATION_NAME],
        root_final_report_sha256=REPORT_SHA,existing_root_files_before_package_sha256=oldroot,
        explicit_exclusions=['large INPUTS/E_COMPARISON/PREDICTIONS','raw observation journals/spools','all238 frames/full worker costplan spools',
                             'pending VISUAL C-point overlay'],
        compact_archive_scope='all238x5 stored actual_pose/topWD/status/source flags/selected scalar; no seed/start/fullcostplan/trajectory reconstruction',
        original_code_models_results_V8_index_WT_or_root_report_changed=False,protected_user_sha256=protected))
    checks=[];links=[];ast_count=0;historical_syntax=[]
    paths={r['path']for r in rows}|{str(p)for p in generated}
    for r in rows:
        p=ROOT/r['path'];checks.append((r['path']+':SHA',sha(Path(r['path']))==r['sha256']))
        if p.suffix=='.py':
            try:ast.parse(p.read_text());ast_count+=1
            except SyntaxError as e:
                if 'initial_syntax_attempt_v1'in p.parts and p.name=='render_clean_fresh_BI_full_result_process_v4_32_r2_v1.py':
                    historical_syntax.append(dict(path=r['path'],expected_preserved_error=str(e)))
                else:raise
        if p.suffix=='.md':
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
                if '://'in target or target.startswith('#'):continue
                resolved=(p.parent/target.split('#')[0]).resolve();rel=str(resolved.relative_to(ROOT))
                exists=resolved.exists()or rel==str(PUB/VALIDATION_NAME);covered=rel in paths
                if exists and not covered and(WT/rel).is_file():
                    tracked=subprocess.run(['git','ls-files','--error-unmatch','--',rel],cwd=WT,capture_output=True).returncode==0
                    covered=tracked and hashlib.sha256((WT/rel).read_bytes()).hexdigest()==sha(Path(rel))
                links.append(dict(source=r['path'],target=target,exists=exists,included_or_identical_public_WT=covered));checks.append((r['path']+':'+target,exists and covered))
    for p,h in oldroot.items():checks.append(('old_ROOT_pub9 '+p,sha(Path(p))==h))
    for p,h in protected.items():checks.append(('protected '+p,sha(Path(p))==h))
    # Small sealed outputs and images only: check all asset bytes without reopening raw inputs.
    for folder,sealname,assetkey in((EVAL,'SCORE_SEAL.json','score_sha256'),(EVAL/'independent_root_audit_v1','AUDIT_SEAL.json','asset_sha256'),
        (B/'clean_fresh_BI_compact_pose_archive_v4_32_r2_v1','SEAL.json','assets')):
        seal=read(folder/sealname)
        if folder==EVAL:
            for name,h in seal[assetkey].items():checks.append(('score '+name,sha(folder/(name+'_SCORE.json'))==h))
        elif assetkey in seal:
            for name,h in seal[assetkey].items():checks.append(('asset '+name,sha(folder/name)==h))
    validation=dict(schema='CURATED_V9_CLOSED_PAYLOAD_VALIDATION',UTC=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=sha(PUB/MANIFEST_NAME),checks=len(checks),failures=[n for n,ok in checks if not ok],
        AST_files=ast_count,expected_preserved_historical_syntax=historical_syntax,local_links=links,payload_files=len(rows),payload_bytes=sum(r['bytes']for r in rows),
        all_paths_to_publish=len(allpaths),forced_add_paths=ignored.stdout.splitlines(),protected6_exact=True,
        maximum_file_bytes=max(r['bytes']for r in rows),root_DRAFT_bindings_FINAL_unchanged=True,
        pending_diagnostics_not_included=True,raw_large_or_runtime_inputs_excluded=True,index_WT_commit_push_mutations=False)
    write(PUB/VALIDATION_NAME,validation)
    print(json.dumps({k:validation[k]for k in('checks','failures','payload_files','payload_bytes','all_paths_to_publish','maximum_file_bytes')},ensure_ascii=False))
    if validation['failures']:raise ValueError('Preserve failed payload validation for additive closure')


if __name__=='__main__':main()
