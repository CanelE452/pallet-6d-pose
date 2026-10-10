"""Own-frozen public byte closure and saved report transcription; stdlib only.

No metric, bootstrap, timing, pose or raw-row arithmetic is repeated. Existing
independent statistics/runtime receipts retain that authority. Explicit
external declarations remain metadata and are never opened as public assets.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re

REPO=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_factor_subset_consensus_20261011_v15'
PRIVATE=Path('/tmp/pallet-factor-subset-consensus-private-20261011-v15')
FILES=('PUBLIC_REPORT_PROTOCOL.json','PUBLIC_REPORT_STARTED.json','PUBLIC_REPORT_CHECKS.json')
EXCLUDE=FILES+('PUBLIC_COPY_PROTOCOL.json','PUBLIC_COPY_STARTED.json','PUBLIC_COPY_CHECKS.json','PUBLIC_COPY_MANIFEST.json','PUBLIC_REVIEW_KO.md')
METHODS=('BASE','N3_SUBPIX','ROLE_BOUNDARY_H_ROBUST','FINITE_EDGE_CONSENSUS_STABLE_INCIDENCE','FACTOR_SUBSET_FINITE_EDGE_CONSENSUS')
SHORT=dict(zip(METHODS,('BASE','N3','POINT','STABLE','SUBSET')));STRATA=('combined','easy','medium')
METRICS=('translation_cm','rotation_deg','ADDsym_cm');MOMENTS=('mean','sample_variance','sample_std','median','P90','max')
LIMITS=dict(new_GT_model_fit_PnP_J_SVD_seed_draw_timing_calls=0,new_raw_metric_or_CI_arithmetic_calls=0,
    independent_checks_delegated_to_saved_ENVELOPE_and_RUNTIME_authorities=True,
    external_declared_metadata_is_not_current_asset_verification=True,physical_ownership_certified=False,
    hardware_or_remote_publication_certified=False,automatic_retry=False)

def require(value,message):
    if not value:raise ValueError(message)

def safe(path):
    p=Path(path).absolute();require(not any(q.is_symlink() for q in (p,*p.parents)),'symlink '+str(p));return p

def read(path):return json.loads(safe(path).read_text())

def bind(path):
    p=safe(path);require(p.is_file(),'missing '+str(p));h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())

def same(a,b):return all(a[k]==b[k] for k in ('bytes','sha256'))

def write(path,value):
    with safe(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def origin_authority(args):
    original=args.repo_root/CODE.relative_to(REPO)/'public_review.py'
    failure_path=args.input/'FIRST_PUBLIC_METADATA_FAILURE.json';failure=read(failure_path)
    require(bind(original)['sha256']=='1f5c280166778224ccb5bd30e095d7e726c618386c7f9eea5e8d1ec1cf6d400a'
        and same(bind(original),failure['original_checker']),'exact original public checker preserved')
    require(failure['schema']=='factor_subset_public_metadata_first_failure_v15' and failure['complete'] is False
        and failure['passed'] is False and failure['failed_before_PROTOCOL'] is True and failure['failed_before_STARTED'] is True
        and failure['own_report_transcription_calls']==failure['new_GT_models_fits_PnP_J_SVD_seed_timing_calls']==0
        and failure['original_checker_preserved'] is True and failure['file_existence_heuristic_allowed'] is False,
        'actual original metadata-only failure preserved')
    before={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(original.read_text()).body if isinstance(n,ast.FunctionDef)}
    after={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
    unchanged=[name for name in before if name!='closure']
    require(all(before[name]==after[name] for name in unchanged),'all original non-closure functions unchanged')
    return dict(original_checker=bind(original),original_failure=bind(failure_path),review_checker=bind(__file__),
        unchanged_original_nonclosure_functions=unchanged,transcription_arithmetic_AST_changes=0,
        legacy_origins=['experiment','immutable_baseline_dependency'],
        explicit_external_alias_prefixes=['fresh_v7:source:','fresh_v7:baseline:'],
        legacy_classification_requires_origin_path_SHA_and_bytes=True,missing_file_heuristic_added=False,
        original_failed_before_protocol_and_STARTED=True,original_checker_or_fixed_reports_changed=False)

def closure(args):
    files={};skipped=[];declarations=[];seen=set();queue=[];legacy={}
    origin_review=origin_authority(args)
    rp=read(args.input/'RUNTIME_PROTOCOL.json')
    for alias,value in rp['inputs'].items():
        prefixes=('fresh_v7:source:','fresh_v7:baseline:')
        prefix=next((x for x in prefixes if alias.startswith(x)),None)
        if prefix and value.get('origin')=='external_readonly_dependency':
            name=alias[len(prefix):];p=Path(name)
            if not p.is_absolute() and '..' not in p.parts and p.parts and p.parts[0] in ('scripts','_docs'):
                for path in (name,value['path']):
                    q=Path(path)
                    require(not q.is_absolute() and '..' not in q.parts,'explicit external declaration path')
                    legacy[(path,value['bytes'],value['sha256'])]=dict(alias=alias,declaration=value)
    def public(value):
        if value.get('origin')=='external_readonly_dependency':return None
        if value.get('origin') in ('experiment','immutable_baseline_dependency') and (value['path'],value['bytes'],value['sha256']) in legacy:return None
        p=Path(value['path'])
        if p.is_absolute():
            if not p.is_relative_to(args.recorded_repo_root):return None
            p=p.relative_to(args.recorded_repo_root)
        if not p.parts or p.parts[0] not in ('scripts','_docs') or '..' in p.parts:return None
        return safe(args.repo_root/p)
    def add(path):
        p=safe(path);require(p.is_relative_to(args.repo_root),'public input outside current tree')
        name=str(p.relative_to(args.repo_root));require(p.is_file(),'missing declared public input '+name)
        if name not in files:files[name]=p;queue.append(p)
    def inspect(value,context):
        if isinstance(value,dict):
            if {'path','bytes','sha256'}<=set(value):
                p=public(value)
                if p is None:skipped.append(dict(context=context,declaration=value));return
                require(same(bind(p),value),'public declaration differs '+context);declarations.append(context);add(p)
            else:
                for key,child in value.items():inspect(child,context+'/'+str(key))
        elif isinstance(value,list):
            for index,child in enumerate(value):inspect(child,context+'/'+str(index))
    for tree in (args.repo_root/CODE.relative_to(REPO),args.input):
        for p in sorted(tree.rglob('*')):
            safe(p)
            if '__pycache__' in p.parts or p.suffix=='.pyc' or p.parent==args.input and p.name in EXCLUDE:continue
            if p.is_file():
                allowed=p.suffix=='.py' if tree!=args.input else p.suffix in ('.json','.md','.gz','.csv','.png','.txt') or p.name=='.gitignore' or bool(re.fullmatch(r'.+\.part\d{3}',p.name))
                require(allowed,'undeclared asset in own tree '+str(p));add(p)
    links=[];images=[]
    for name in ('RESULT_KO.md','README.md'):
        for match in re.finditer(r'(!?)\[[^\]]*\]\(([^)]+)\)',(args.input/name).read_text()):
            target=match.group(2)
            if target.startswith(('http://','https://','#')):continue
            p=safe(args.input/target.partition('#')[0]).resolve();add(p);links.append(str(p.relative_to(args.repo_root)))
            if match.group(1):images.append(str(p.relative_to(args.repo_root)))
    while queue:
        p=queue.pop()
        if p.suffix=='.json' and p not in seen:
            seen.add(p);data=read(p)
            if isinstance(data,dict):
                for key,value in data.items():
                    if key in ('inputs','model_bindings') or isinstance(value,dict) and {'path','bytes','sha256'}<=set(value):
                        inspect(value,str(p.relative_to(args.repo_root))+'/'+key)
    files=dict(sorted(files.items()));return files,{k:bind(p) for k,p in files.items()},dict(
        public_files=len(files),declared_public_bindings=len(declarations),external_metadata_bindings_not_opened=skipped,
        legacy_external_alias_declarations=[dict(path=k[0],**v) for k,v in legacy.items()],local_links=links,image_links=images,
        excluded_late_paths=list(EXCLUDE),classification_uses_file_existence_heuristic=False,
        declared_inputs_and_direct_byte_refs_followed_source_protection_inventory_not_asset_dependencies=True,
        public_origin_review=origin_review)

def authority(args):
    p=args.input;m=read(p/'METRICS.json');v=read(p/'ENVELOPE_VERIFICATION.json');r=read(p/'RUNTIME_RECEIPT.json');rc=read(p/'RUNTIME_CHECKS.json')
    require(m['complete'] is True and m['methods']==list(METHODS) and m['actual_arithmetic']==dict(scored_rows=1225,new_rows=245,posthoc_rows=245,methods=5,strata=3,moment_distributions=135,moment_scalars=810,paired_groups=27,metric_CIs=81,new_bootstrap_draws=0),'fixed1225/810/27/81 statistics')
    require(v['complete'] is True and v['passed'] is True and v['failure_count']==0 and v['actual_counts']['saved_scored_rows_streamed']==1225 and v['actual_counts']['moment_scalars_checked']==810 and v['actual_counts']['metric_CI_slots_checked']==81,'actual saved statistics PASS')
    require(same(v['protocol'],bind(p/'ENVELOPE_STATISTICS_PROTOCOL.json')) and v['inputs']==read(p/'ENVELOPE_STATISTICS_PROTOCOL.json')['inputs'],'actual statistics own authority')
    for marker,flag in (('public_binding_review','public_enumeration_AST_changes'),('STARTED_envelope_review','envelope_AST_insertions')):
        require(v[marker][flag]==1 and v[marker]['statistic_formula_AST_changes']==0 and v[marker]['full_canonical_original_restore_proven'] is True,'unchanged statistics review formulas')
    require(r['complete'] is True and r['statistics_official'] is True and r['serialized_rows']==150 and rc['complete'] is True and rc['passed'] is True and rc['failure_count']==0 and rc['rows']==150 and rc['measured']==130 and rc['summary_slots']==56,'actual fresh runtime and saved independent PASS')
    require(read(p/'CPU_CHECKS.json')['passed'] is True and read(p/'CPU_CHECKS.json')['tests_run']==12,'actualCPU12')
    require(read(p/'VALIDATION_CHECKS.json')['passed'] is True and read(p/'VALIDATION_CHECKS.json')['failure_count']==0,'actual full geometry scalar PASS')
    figure,visual=read(p/'FIGURE_BINDINGS.json'),read(p/'ROOT_VISUAL_REVIEW.json')
    require(figure['complete'] is True and figure['PNG_files']==8 and figure['panels']==24 and figure['display_records']==1225 and visual['passed'] is True and visual['actual_image_tool_views']==8,'actual8figure/24panel visual authority')
    for row in figure['figures']:
        path=Path(row['path']);path=args.repo_root/path.relative_to(args.recorded_repo_root) if path.is_absolute() else args.repo_root/path
        require(same(bind(path),row),'actual figure bytes')
    for name in ('STATISTICS_METADATA_FIRST_FAILURE.json','REVIEW_ENVELOPE_FIRST_FAILURE.json'):
        failure=read(p/name);require(failure['complete'] is False and failure['failed_before_STARTED'] is True and failure['own_statistic_arithmetic_calls']==failure['new_GT_models_fits_projection_SVD_seed_calls']==0,'preserved zero-arithmetic verifier failure')
    require(read(p/'BUILD_LEDGER.json')['goal_achieved']==m['primary_both_T_R_means_better_than_N3_on_full245'],'dynamic truthful goal transcription')
    return m,r

def tables(args,m,r):
    counts=Counter();seen=set();maximum=0.
    def num(text,expected):
        nonlocal maximum
        if expected is None:require(text=='None','transcribed None');return
        actual=float(text);require(math.isfinite(actual) and math.isfinite(expected),'finite scalar')
        delta=abs(actual-expected);maximum=max(maximum,delta);require(delta<=1e-12+1e-12*abs(expected),'saved scalar transcription mismatch')
    def once(kind,key):require((kind,key) not in seen,'duplicate table row');seen.add((kind,key));counts[kind]+=1
    reverse={v:k for k,v in SHORT.items()}
    for raw in (args.input/'RESULT_KO.md').read_text().splitlines():
        if not raw.startswith('|'):continue
        cells=[x.strip() for x in raw.strip('|').split('|')]
        if cells[0] in STRATA:
            s=m['strata'][cells[0]]
            if len(cells)==7 and cells[1] in reverse and cells[2] in ('operational','new_pose','fallback'):
                d=s['methods'][reverse[cells[1]]]['metrics'][cells[2]];once('moment',(cells[0],cells[1],cells[2]));num(cells[3],d['translation_cm']['n'])
                for cell,key in zip(cells[4:],METRICS):
                    values=cell.split('<br>');require(len(values)==6,'six moment scalar lines')
                    for text,field in zip(values,MOMENTS):num(text,d[key][field]);counts['moment_scalars']+=1
            elif len(cells)==12 and cells[1].startswith('SUBSET−'):
                comparator=reverse[cells[1].split('−',1)[1]];d=s['contrasts'][METHODS[-1]+'_minus_'+comparator][cells[2]];once('paired',(cells[0],comparator,cells[2]))
                for index,key in enumerate(('denominator','common_frames','candidate_new_pose_in_pairs','comparator_new_pose_in_pairs','bootstrap_nonempty_resamples','bootstrap_empty_resamples'),3):num(cells[index],d[key])
                for cell,key in zip(cells[9:],METRICS):
                    delta,ci=cell.split(';',1);num(delta.strip(),d['metrics'][key]['mean_delta']);actual=ast.literal_eval(ci.strip());expected=d['metrics'][key]['CI95']
                    require((actual is None)==(expected is None),'CI empty scalar slot')
                    if actual is not None:
                        require(len(actual)==2,'CI scalar pair')
                        for a,b in zip(actual,expected):num(str(a),b)
                    counts['CI_slots']+=1
            elif len(cells)==9 and cells[1] in reverse:
                d=s['methods'][reverse[cells[1]]];once('operational',(cells[0],cells[1]))
                for cell,key in zip(cells[2:8],('denominator','operational','new_pose','fallback','no_pose','fixed_control')):num(cell,d[key])
                require(json.loads(cells[8])==d['statuses'],'saved operating statuses')
            elif len(cells)==5 and cells[1] in s['diagnostics'][METHODS[-1]]['mask_and_pose_groups_vs_fixed_N3']:
                d=s['diagnostics'][METHODS[-1]]['mask_and_pose_groups_vs_fixed_N3'][cells[1]];once('mask',(cells[0],cells[1]))
                for cell,key in zip(cells[2:],('n','new_pose','fallback')):num(cell,d[key])
            elif len(cells)==6 and cells[1] in ('DIRECT_VISIBLE','SELF_OCCLUDED','ACTUALLY_REPROJECTED'):
                d=s['diagnostics'][METHODS[-1]][cells[1]];error=d['errors'][cells[2]];once('corner',(cells[0],cells[1],cells[2]));num(cells[3],error['n'])
                require(len(cells[4].split('<br>'))==6,'six corner moments')
                for text,key in zip(cells[4].split('<br>'),MOMENTS):num(text,error[key])
                if cells[1]=='DIRECT_VISIBLE':num(cells[5],d['damage_5px_to_over10px'])
                else:require(cells[5]=='not_measured','no invented hidden damage')
            elif len(cells)==9 and cells[1] in ('all_candidate_boundary_quality','adopted_boundary_quality','displayed_boundary_quality'):
                d=s['diagnostics'][METHODS[-1]][cells[1]];once('boundary',(cells[0],cells[1]));num(cells[2],d['candidates'])
                for cell,key in zip(cells[3:5],('CORRECT_WITHIN8PX','INCORRECT_OVER8PX')):num(cell,d['candidate_proxy_quality_counts'].get(key,0))
                for cell,key in zip(cells[5:8],('improved','worsened','unchanged')):num(cell,d[key])
                require(len(cells[8].split('<br>'))==6,'six boundary moments')
                for text,key in zip(cells[8].split('<br>'),MOMENTS):num(text,d['candidate_error_px'][key])
        elif len(cells)==9 and cells[0] in r['summaries']:
            d=r['summaries'][cells[0]];once('runtime',cells[0])
            for cell,key in zip(cells[1:],('n','mean','sample_variance','sample_std','median','P90','maximum','ddof')):
                num(cell,d[key]);counts['timing_slots' if key!='ddof' else 'timing_ddof_metadata']+=1
    require(counts['moment']==45 and counts['moment_scalars']==810 and counts['paired']==27 and counts['CI_slots']==81 and counts['operational']==15 and counts['runtime']==8 and counts['timing_slots']==56
        and counts['mask']==48 and counts['corner']==27 and counts['boundary']==9,'complete810/81/56 and saved diagnostic display tables')
    return dict(table_rows=dict(counts),scientific_moment_or_CI_recomputation=0,max_transcription_difference=maximum,
        timing_reported_statistic_slots=56,timing_n_is_in56_slots_ddof_is_additional_metadata=True)

def guard(args,stage):
    out=safe(args.output);require(out==DOC or out.is_relative_to(PRIVATE),'new DOC/private own output only');out.mkdir(parents=True,exist_ok=True)
    for name in FILES if stage=='freeze' else FILES[1:]:require(not (out/name).exists(),'preserve '+name)
    return out

def freeze(args):
    out=guard(args,'freeze');authority(args);_,b,scope=closure(args)
    write(out/FILES[0],dict(schema='factor_subset_report_public_protocol_v15',inputs=b,scope=scope,limits=LIMITS,
        checker=bind(__file__),frozen_before_own_report_scalar_transcription=True))
    print('V15_PUBLIC_REPORT_FROZEN',len(b),flush=True)

def run(args):
    out=guard(args,'run');own=read(out/FILES[0]);_,b,scope=closure(args);require(own['inputs']==b and own['scope']==scope and same(own['checker'],bind(__file__)),'own public freeze differs')
    write(out/FILES[1],dict(protocol=bind(out/FILES[0]),inputs=b,limits=LIMITS))
    result=dict(schema='factor_subset_report_public_checks_v15',complete=False,passed=False,protocol=bind(out/FILES[0]),inputs=b,scope=scope,limits=LIMITS,checker=bind(__file__),automatic_retry=False)
    try:
        m,r=authority(args);result.update(tables(args,m,r));require(len(scope['image_links'])==8 and len(set(scope['image_links']))==8,'eight actual report images')
        require(closure(args)[1:]==(b,scope),'public inputs changed');result.update(complete=True,passed=True,existing_statistics_rows=1225,existing_statistics_moments=810,existing_metric_CIs=81,existing_runtime_rows=150)
    except Exception as error:result['error']=dict(type=type(error).__name__,message=str(error))
    write(out/FILES[2],result);print('V15_PUBLIC_REPORT',result['passed'],len(b),flush=True)
    if not result['passed']:raise SystemExit(1)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--input',type=Path,default=DOC);p.add_argument('--output',type=Path,default=DOC)
    p.add_argument('--repo-root',type=Path,default=REPO);p.add_argument('--recorded-repo-root',type=Path,default=Path('/tmp/pallet-kp-publication-20261010-v6'))
    args=p.parse_args();args.input=safe(args.input);args.repo_root=safe(args.repo_root);(freeze if args.stage=='freeze' else run)(args)

if __name__=='__main__':main()
