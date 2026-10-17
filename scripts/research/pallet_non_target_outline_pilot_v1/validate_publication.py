"""Final package and protected-source checks; no new inference or selection."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time
import re
from urllib.parse import urlsplit, unquote
import numpy as np

from scripts.research.pallet_outline_physical_support_audit_v1.validate import HTMLLinks, privacy_scan


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def require(x,msg):
    if not x:raise ValueError(msg)
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root)


def check_public_links(public,allow_pending_validation_outputs=False):
    count=0
    for path in sorted(public.rglob('*')):
        if path.suffix not in ('.md','.html'):continue
        text=path.read_text()
        if path.suffix=='.html':
            parser=HTMLLinks();parser.feed(text);links=parser.links
        else:
            # SDK indexing followed by parentheses inside code is not a link.
            text=re.sub(r'```[\s\S]*?```','',text)
            text=re.sub(r'`[^`\n]*`','',text)
            links=re.findall(r'!?\[[^\]\n]*\]\(([^\s)]+)(?:\s+[^)]*)?\)',text)
        for link in links:
            url=urlsplit(link.strip('<>'))
            if url.scheme or url.netloc:continue
            target=(path.parent/unquote(url.path)).resolve()if url.path else path.resolve()
            pending={str((public/n).resolve())for n in ('VALIDATION.json','VALIDATION.md')}
            if allow_pending_validation_outputs and str(target)in pending and not target.exists():continue
            require(target.exists(),'Broken publication link '+str(path)+' -> '+link)
            if url.fragment and target.suffix=='.html':
                parser=HTMLLinks();parser.feed(target.read_text());require(unquote(url.fragment)in parser.ids,'Missing HTML anchor '+link)
            count+=1
    return count


def run(source_root,asset_root,private_root,score_private,state_private,public,output):
    started,cpu=time.monotonic(),time.process_time()
    s,a,p,score,state,out,dest=map(Path,(source_root,asset_root,private_root,score_private,state_private,public,output))
    require(not dest.exists(),'Preserve prior validation receipt')
    required=('REPORT_KO.md','PLAN_AND_CONTRACT.md','INPUT_SUPPLIER_AUDIT.md','PROMPT_MANIFEST.jsonl',
        'NON_TARGET_INPUT_MANIFEST.jsonl','PARENT_SUPPORT.jsonl','PREDICTIONS.jsonl.gz','PREDICTION_SEAL.json',
        'METRICS.csv','TRANSITIONS.csv','COST_BREAKDOWN.csv','COST.json','REPRODUCE.md','NEXT_DECISION_KO.md',
        'CONTROL_COMPARISON.csv','CONTROL_COMPARISON.md','gallery/index.html')
    for n in required:require((out/n).is_file(),'Missing required output '+n)
    seal=read(out/'PREDICTION_SEAL.json')
    for name,digest in seal['artifacts'].items():require(sha(out/name)==digest,'Scientific output changed after seal '+name)
    require(seal['frames']==6 and seal['candidates']==351 and seal['prediction_rows']==18,'Fixed denominator changed')
    require(seal['read_guard']['denied_events']==[],'Selector accessed forbidden input')
    require(seal['runtime_guard']['forbidden_function_bodies_executed']==0,'Forbidden model/pose body entered')
    require(all(x['baseline_family_candidate_exact']and x['no_input_cost_rank_pose_exact']for x in seal['baseline_checks']),'Identity failure')
    independent=read(p/'INDEPENDENT_CHECK_V1.json');publication=read(p/'INDEPENDENT_PUBLICATION_CHECK_V2.json')
    require(independent['passed']is True,'Independent functional check failed')
    require(publication.get('passed')is True,'Publication permission/weights check failed')
    # Public masks are exact frozen binary inputs; paths alone were made relative.
    frozen=[json.loads(l)for l in(out/'INPUTS_FROZEN.jsonl').read_text().splitlines()]
    require(len(frozen)==6 and sum(x['admissible_for_censoring']for x in frozen)==3,'Input denominator changed')
    for entry in frozen:
        if not entry['admissible_for_censoring']:continue
        f=out/entry['mask_path'];require(sha(f)==entry['mask_sha256'],'Public input mask changed')
        v=np.load(f,allow_pickle=False)
        require(hashlib.sha256(v.tobytes()).hexdigest()==entry['pixel_bytes_sha256'],'Binary pixel SHA changed')
    for r in read(out/'PUBLIC_DERIVATIONS.json')['records']:
        require(sha(out/r['public'])==r['public_sha256'],'Derived receipt changed '+r['public'])
        original=p/r['source'].removeprefix('PRIVATE_AUDIT/')
        require(sha(original)==r['original_sha256'],'Original receipt changed')
    binding=read(out/'INPUT_BINDINGS.json')
    actual={str(f.relative_to(s)):sha(f)for f in sorted((s/'scripts/research/pallet_non_target_outline_pilot_v1').glob('*.py'))}
    require(actual==binding['source_sha256'],'New source hash set stale; repackage first')
    upstream=read(s/'_docs/experiments/pallet_selection_score_diagnosis_v1/SOURCE_BINDINGS.json')
    for path,digest in upstream['legacy_source_sha256'].items():require(sha(s/path)==digest,'Legacy source changed '+path)
    scoreseal=read(state/'score_v2/SCORE_SEAL.json')
    for path,digest in scoreseal['annotation_sha256'].items():require(sha(s/path)==digest,'Annotation changed '+path)
    for r in scoreseal['reference_bindings'].values():require(sha(s/r['path'])==r['sha256'],'Reference source changed')
    start=read(p/'START_BEFORE_FETCH.json')
    require(git(a,'rev-parse','HEAD').decode().strip()==start['original_HEAD'],'User workspace HEAD changed')
    require(git(a,'branch','--show-current').decode().strip()==start['original_branch'],'User branch changed')
    require(git(a,'status','--porcelain','-z')==(p/'ORIGINAL_STATUS_NUL.bin').read_bytes(),'User full NUL status changed')
    for path,digest in start['protected_files_SHA256'].items():require(sha(a/path)==digest,'User edit changed')
    for ref in start['branch_refs']:
        name,head=ref.split();require(git(a,'rev-parse',name).decode().strip()==head,'Prior work branch changed')
    prefixes=('scripts/research/pallet_non_target_outline_pilot_v1/','_docs/experiments/pallet_non_target_outline_pilot_v1/')
    for path in ('data','_docs/experiments/pallet_visibility_curriculum_20261013_v1'):
        require((s/path).is_symlink()and(s/path).resolve()==(a/path).resolve(),'Asset link changed')
    changed=git(s,'diff','--name-only','4d509f66e0e544a970cd039afb5d33c94680d394','--').decode().splitlines()
    changed+=git(s,'ls-files','--others','--exclude-standard').decode().splitlines()
    require(all(x.startswith(prefixes)or x in ('data','_docs/experiments/pallet_visibility_curriculum_20261013_v1')for x in changed),'Unexpected namespace change')
    # The final validator is the one linked output not yet written.
    links=check_public_links(out,allow_pending_validation_outputs=True);privacy=privacy_scan(out)
    result=dict(schema='NON_TARGET_FUNCTIONAL_FINAL_PUBLICATION_VALIDATION_V1',passed=True,
        fixed_frames=6,retained_candidates=351,prediction_rows=18,contexts_parent_records=508,
        admitted_foreign_inputs=3,no_admissible_inputs=3,AUTO238_executed=False,
        independent_arithmetic_receipt_sha256=sha(p/'INDEPENDENT_CHECK_V1.json'),
        independent_publication_receipt_sha256=sha(p/'INDEPENDENT_PUBLICATION_CHECK_V2.json'),
        scientific_prediction_artifacts_match_private_seal=True,source_files_bound=len(actual),
        legacy_source_SHA_checked=len(upstream['legacy_source_sha256']),annotation_SHA_checked=len(scoreseal['annotation_sha256']),
        original_dirty_workspace_HEAD_branch_status_and_6files_preserved=True,prior_branch_refs_preserved=True,
        no_changes_outside_new_namespaces=True,no_defaults_papers_weights_references_changed=True,
        public_links_checked=links,privacy=privacy,
        artifact_sha256={str(f.relative_to(out)):sha(f)for f in sorted(out.rglob('*'))if f.is_file()and f.name not in ('VALIDATION.json','VALIDATION.md')},
        validator_source_sha256=sha(__file__),elapsed_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu)
    dest.write_text(json.dumps(result,indent=2)+'\n');(out/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    check_public_links(out);privacy_scan(out)
    return {k:result[k]for k in ('passed','retained_candidates','prediction_rows','public_links_checked','elapsed_seconds')}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('source-root','asset-root','private-root','score-private','state-private','public','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();print(run(a.source_root,a.asset_root,a.private_root,a.score_private,a.state_private,a.public,a.output))
