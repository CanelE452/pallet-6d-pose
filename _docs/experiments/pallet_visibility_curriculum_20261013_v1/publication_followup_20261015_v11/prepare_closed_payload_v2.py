"""User-authorized V11 closed publication; no Git mutations or new inference.

Reuses frozen V10/V5 source and core result gates. Explicitly adds the closed
gain10 experiments, their small raw input/results/masks, and static MD galleries.
Large full-population raw archives and duplicated per-frame dumps stay outside
main; their separate inventory/Release is handled by the raw-evidence owner.
"""
from pathlib import Path
from datetime import datetime,timezone
import ast
import hashlib
import importlib.util
import json
import re
import subprocess
from urllib.parse import unquote

ROOT=Path('/home/minjae/Documents/github/pallet-pose')
D=Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1')
PUB=D/'publication_followup_20261015_v11'
PREVIOUS=D/'publication_followup_20261015_v10'
S=Path('scripts/research/pallet_visibility_curriculum_20261013_v1')
spec=importlib.util.spec_from_file_location('frozen_publication_v10_v5',ROOT/PREVIOUS/'prepare_closed_payload_v5.py')
OLD=importlib.util.module_from_spec(spec);spec.loader.exec_module(OLD)
B=OLD.B
NEW=B/'new_gain10_paired_BI_v2'
FAILED=B/'new_gain10_paired_BI_v1'
PAIR=B/'raw_yolo_vs_N3_subpix_TEST49_paired_v1'
INPUT=B/'raw_ns_BI_gain10_input_v1'
RENDER=B/'new_gain10_paired_BI_process_v1'
GROUPS=(D/'show_191cm_case_20261015_v1',D/'raw_n3_subpix_gain10_contact_v1',PAIR,INPUT,NEW,FAILED,RENDER,
    B/'gain10_common_BI_pipeline_source_audit_v1',B/'raw_ns_BI_gain10_score_source_checks_v1',
    B/'raw_yolo_N3_subpix_gain10_process_v1',B/'raw_yolo_N3_subpix_gain10_process_source_checks_v1',
    B/'raw_yolo_N3_subpix_gain10_process_peer_review_v1')
FINALS={'CLOSED_PAYLOAD_MANIFEST_V1.json','PAYLOAD_VALIDATION_V1.json','CLOSED_FILES_TO_PUBLISH_V1.txt','FORCE_ADD_FILES_V1.txt'}
PIN=dict(prediction='7739a8a3ae263ca29781ec92c39de12b33273efb3a039f11bfe2fd12e5e55d13',
    score='7c98c362f5ded2bb377bf5d0de30c3ff35268f0440a6484e24af46c8743146e0',
    audit='d51b0a943b1b23d6847f735bdc52d7f095513608ecd04c8e026a8a4bd1dc9537',
    recount='9f8b0a8801c0be139a1fbfd5b23ce297322a9eb0a2114295358a1a98f637fe39',
    paired_prediction='f57cc4e978f92b066d2179e1b5e5394c608685c3573a641fc297a29d788a7147',
    paired_score='87715871b675dfcc01ec32925926b4706b28a40b1f80047a2c2920594560bb97')


def sha(p):
    h=hashlib.sha256()
    with (ROOT/p).open('rb')as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads((ROOT/p).read_text())
def write(p,x):(ROOT/p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rel(p):return str(p.relative_to(ROOT))


def raw_exception(path):
    return (path==INPUT/'INPUTS.json' or path in {NEW/'PREDICTIONS.jsonl.gz',NEW/'OBSERVATIONS.jsonl.gz'}
        or path.parent==NEW/'masks'and path.suffix=='.npz')


def allowed(path):
    if path.is_absolute()or '..'in path.parts:return False
    if raw_exception(path):return True
    return OLD.permitted(path)or path.suffix=='.py'and OLD.is_snapshot(path)


def group_files(group):
    result=set()
    for file in (ROOT/group).rglob('*'):
        if not file.is_file()or'__pycache__'in file.parts:continue
        path=file.relative_to(ROOT)
        if OLD.is_snapshot(path):
            live=S/path.name
            if file.suffix!='.py'or(ROOT/live).is_file()and sha(path)==sha(live):continue
        if allowed(path):result.add(path)
    return result


def verify_closed_results():
    OLD._full_result_gate(PIN['score'],PIN['audit'],PIN['recount'],
        dict(path=str(PREVIOUS/'REPORT_KO_FINAL_V1.md'),sha256='d97212bb5f0a070ef8230aef6046c2ada3a4b35637e425ca2f196964b9044b9b'),{})
    for path,digest in ((NEW/'PREDICTION_SEAL.json',PIN['paired_prediction']),(NEW/'SCORE_SEAL.json',PIN['paired_score'])):
        if sha(path)!=digest:raise ValueError('Paired closed seal changed')
    pred=read(NEW/'PREDICTION_SEAL.json');score=read(NEW/'SCORE_SEAL.json')
    if (pred['rows']!=10 or len(pred['arms'])!=4 or pred['before_scoring']is not True or pred['GT_inputs']is not False
            or pred['predictions_sha256']!=sha(NEW/'PREDICTIONS.jsonl.gz')
            or score['prediction_seal_sha256']!=PIN['paired_prediction']or score['arm_rows']!=40
            or score['score_sha256']!=sha(NEW/'SCORE.json')):raise ValueError('Paired all-ten/four closure required')
    failed=read(FAILED/'FAILED_ATTEMPT_V1.json')
    if failed['state_mask_BI_forward_calls']!=0:raise ValueError('Historical failed-attempt scope changed')
    qa=read(RENDER/'OWNER_COMPLETION_AND_VISUAL_QA_V1.json')
    if qa.get('failure_count',0)!=0 or qa.get('failures'):raise ValueError('Closed paired image QA required')


def source_status():
    status=OLD.source_publication_status();mapping={r['path']:r['sha256']for r in status['rows']}
    metadata=[NEW/'PROTOCOL.json',RENDER/'ANALYSIS_PLAN.json',
        B/'raw_yolo_N3_subpix_gain10_process_v1/ANALYSIS_PLAN.json',
        B/'largest_component_global_outline_process_v4_33_v1/display_v2/RENDER_SEAL.json',
        D/'raw_n3_subpix_gain10_contact_v1/RENDER_SEAL.json']
    for path in metadata:
        doc=read(path)
        for name,digest in doc.get('source_sha256',{}).items():
            if name in mapping and mapping[name]!=digest:raise ValueError('Current source closure conflict '+name)
            mapping[name]=digest
        if 'source_binding'in doc:mapping[doc['source_binding']['path']]=doc['source_binding']['sha256']
    for file in (ROOT/S).glob('*gain10*.py'):mapping[rel(file)]=sha(file.relative_to(ROOT))
    for file in (ROOT/S).glob('render_single_191cm_case*.py'):mapping[rel(file)]=sha(file.relative_to(ROOT))
    rows=[]
    for name,digest in sorted(mapping.items()):
        path=Path(name)
        if path.is_absolute():path=path.relative_to(ROOT)
        if sha(path)!=digest:raise ValueError('Closed current source changed '+str(path))
        result=subprocess.run(['git','show','HEAD:'+str(path)],cwd=OLD.WT,capture_output=True)
        head=hashlib.sha256(result.stdout).hexdigest()if result.returncode==0 else None
        rows.append(dict(path=str(path),sha256=digest,published_HEAD_sha256=head,published_HEAD_exact=head==digest,supplement_current_source=head!=digest))
    return dict(rows=rows,count=len(rows),exact_HEAD_reuse=sum(r['published_HEAD_exact']for r in rows),
        supplement_current_source=sum(r['supplement_current_source']for r in rows),read_only_HEAD=True)


def asset_proofs(paths):
    """Every runtime payload is linked to closed seal assets, not glob trust."""
    proofs={};docs=[]
    for path in paths:
        if path.suffix=='.json'and('SEAL'in path.name or path.name=='PROVENANCE.json'):
            doc=read(path);docs.append(path)
            proofs.update(OLD._sealed_assets(doc,path.parent,path.name))
            scalar={'inputs_sha256':'INPUTS.json','manifest_sha256':'SCORING_REFERENCE_MANIFEST.json',
                'predictions_sha256':'PREDICTIONS.jsonl.gz','observations_sha256':'OBSERVATIONS.jsonl.gz',
                'score_sha256':'SCORE.json','summary_sha256':'STATE_AND_BI_MOVEMENT_SUMMARY_V1.json'}
            for key,name in scalar.items():
                if isinstance(doc.get(key),str):proofs[str(path.parent/name)]=doc[key]
            if path.name=='API_FACTS_SEAL_V1.json':proofs[str(path.parent/'API_FACTS_V1.json')]=doc['receipt_sha256']
    # The existing V32 diagnostic execution seal binds a receipt whose
    # asset_sha256 map carries SPATIAL_SUMMARY and the remaining exact assets.
    for path in paths:
        if path.name.endswith('RECEIPT_V1.json')and proofs.get(str(path))==sha(path):
            proofs.update(OLD._sealed_assets(read(path),path.parent,path.name))
    # Mask bindings are immutable metadata inside the already SHA-sealed paired
    # predictions. Only their path/hash fields are consumed; no pose math/ref IO.
    import gzip
    with gzip.open(ROOT/NEW/'PREDICTIONS.jsonl.gz','rt')as stream:
        for line in stream:
            row=json.loads(line)
            for branch in row['branches'].values():
                binding=branch['mask_binding'];proofs[binding['path']]=binding['sha256']
    return proofs,docs


def planning_metadata(path):
    name=path.name
    return (path.suffix=='.py'or path.is_relative_to(PUB)or path.is_relative_to(PREVIOUS)
        or name.startswith(('SOURCE_','OWNER_','ROOT_'))or'OWNER'in name
        or name in {'PLAN.json','PLAN_SEAL.json','ANALYSIS_PLAN.json','SOURCE_PLAN.json','PREDECLARATION.json','PROVENANCE.json',
            'PROTOCOL.json','FINAL_PROTOCOL.json','PREPARATION.json','REST_PREPARATION.json','EXECUTION_START.json',
            'REST_EXECUTION_START.json','SUMMARY.json','REST_SUMMARY.json','CAL_ROW_INDEX.json','REST_ROW_INDEX.json',
            'OUTLINE_ROW_INDEX.json','INPUT_SEAL.json','PREDICTION_ROW_INDEX.json','PREDICTION_EXECUTION_PROTOCOL.json',
            'NATIVE_AND_E_PARITY_V1.json','SCORING_PROTOCOL.json','SCORING_REFERENCE_MANIFEST.json','AUDIT.json',
            'CAL_SCORE.json','TEST_SCORE.json','REMAINDER_SCORE.json','PRIMARY_SCORE.json','FAILED_ATTEMPT_V1.json'}
        or'SEAL'in name or path.suffix=='.md')


def build():
    if any((ROOT/PUB/n).exists()for n in FINALS):raise FileExistsError('Preserve V11 final attempt')
    verify_closed_results();status=source_status()
    oldstatus=OLD.source_publication_status();paths=set(OLD.candidates(oldstatus))
    for group in GROUPS:paths|=group_files(group)
    paths|={Path(r['path'])for r in status['rows']if r['supplement_current_source']}
    # The raw-evidence owner is independently writing repro_inventory_*;
    # publish its final receipt in ROOT's later commit, never snapshot it live.
    paths|={p.relative_to(ROOT)for p in (ROOT/PUB).iterdir()if p.is_file()and p.name not in FINALS
        and not p.name.startswith('repro_inventory_')and allowed(p.relative_to(ROOT))}
    # Protected user files remain byte-verified but are never published here.
    protected=read(OLD.CENTRAL)['protected_user_sha256']
    paths-={Path(p)for p in protected}
    for name,digest in protected.items():
        if sha(Path(name))!=digest:raise ValueError('Protected user bytes changed')
    proof,seals=asset_proofs(paths);rows=[]
    for path in sorted(paths):
        file=ROOT/path
        if not allowed(path)or file.is_symlink()or file.stat().st_size>=100*1024*1024:raise ValueError('Unsafe/large file '+str(path))
        digest=sha(path);metadata=planning_metadata(path)
        if not metadata and proof.get(str(path))!=digest:raise ValueError('Unsealed runtime payload '+str(path))
        rows.append(dict(path=str(path),sha256=digest,bytes=file.stat().st_size,
            classification='explicit_small_gain10_raw_exception'if raw_exception(path)else'historical_source'if OLD.is_snapshot(path)else'source_or_named_metadata'if metadata else'seal_linked_runtime_asset',
            seal_asset_match=proof.get(str(path))==digest))
    generated={PUB/name for name in FINALS};allpaths=paths|generated
    (ROOT/PUB/'CLOSED_FILES_TO_PUBLISH_V1.txt').write_text(''.join(str(p)+'\n'for p in sorted(allpaths)))
    result=subprocess.run(['git','check-ignore','--stdin'],cwd=ROOT,text=True,capture_output=True,input=''.join(str(p)+'\n'for p in sorted(allpaths)))
    if result.returncode not in (0,1):raise ValueError(result.stderr)
    (ROOT/PUB/'FORCE_ADD_FILES_V1.txt').write_text(result.stdout)
    for name in ('CLOSED_FILES_TO_PUBLISH_V1.txt','FORCE_ADD_FILES_V1.txt'):
        p=PUB/name;rows.append(dict(path=str(p),sha256=sha(p),bytes=(ROOT/p).stat().st_size,classification='final_publication_list',seal_asset_match=False))
    failures=[];links=[];ast_count=0
    selected={r['path']for r in rows}|{str(p)for p in generated}
    for row in rows:
        file=ROOT/row['path']
        if file.suffix=='.py':ast.parse(file.read_text());ast_count+=1
        if file.suffix not in ('.md','.html'):continue
        text=file.read_text()
        targets=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text)if file.suffix=='.md'else re.findall(r'(?:href|src)=[\"\x27]([^\"\x27]+)',text)
        for target in targets:
            if target.startswith(('#','data:','mailto:'))or'://'in target:continue
            targetpath=(file.parent/unquote(target.split('#')[0])).resolve()
            if not targetpath.is_relative_to(ROOT):failures.append('Link escapes repo '+target);continue
            name=rel(targetpath);covered=name in selected
            if not covered and targetpath.is_file()and(OLD.WT/name).is_file():
                tracked=subprocess.run(['git','ls-files','--error-unmatch','--',name],cwd=OLD.WT,capture_output=True).returncode==0
                covered=tracked and hashlib.sha256((OLD.WT/name).read_bytes()).hexdigest()==sha(Path(name))
            if not covered:failures.append('Unpublished local link '+name)
            links.append(dict(source=row['path'],target=target,covered=covered))
    # Recheck selected bytes after all link/AST reads; no source/record rewriting.
    for row in rows:
        if sha(Path(row['path']))!=row['sha256']:failures.append('Selected bytes changed '+row['path'])
    manifest=dict(schema='USER_REQUESTED_ALL_CLOSED_V33_AND_PAIRED_GAIN10_PUBLICATION_V11',UTC=datetime.now(timezone.utc).isoformat(),
        files=rows,count=len(rows),bytes=sum(r['bytes']for r in rows),all_paths_to_publish=len(allpaths),
        source_publication_status=status,closed_result_sha256=PIN,explicit_groups=[str(p)for p in GROUPS],
        runtime_seal_documents=[str(p)for p in seals],protected_user_sha256=protected,protected_files_excluded=True,
        raw_exceptions=[r['path']for r in rows if r['classification']=='explicit_small_gain10_raw_exception'],
        duplicated_gain10_frames_and_large_full32_33_raw_excluded=True,large_raw_delta_release_separate=True,
        old_V10_reports_and_all_experimental_results_unchanged=True,git_mutations=False,
        self_reference_exclusions=['CLOSED_PAYLOAD_MANIFEST_V1.json','PAYLOAD_VALIDATION_V1.json'])
    write(PUB/'CLOSED_PAYLOAD_MANIFEST_V1.json',manifest)
    write(PUB/'PAYLOAD_VALIDATION_V1.json',dict(manifest_sha256=sha(PUB/'CLOSED_PAYLOAD_MANIFEST_V1.json'),
        failures=failures,failure_count=len(failures),source_AST_files=ast_count,local_links=links,
        image_files=sum(Path(r['path']).suffix.lower()in ('.png','.jpg','.jpeg')for r in rows),
        HTML_files=sum(Path(r['path']).suffix=='.html'for r in rows),maximum_file_bytes=max(r['bytes']for r in rows),
        user_requested_build_GO=True,new_inference_reference_metric_pose_or_mask_computation=0,git_mutations=False))
    if failures:raise ValueError('Preserve failed payload attempt '+str(len(failures)))
    return dict(paths=len(allpaths),bytes=manifest['bytes'],manifest_sha256=sha(PUB/'CLOSED_PAYLOAD_MANIFEST_V1.json'),validation_sha256=sha(PUB/'PAYLOAD_VALIDATION_V1.json'))


if __name__=='__main__':print(build())
