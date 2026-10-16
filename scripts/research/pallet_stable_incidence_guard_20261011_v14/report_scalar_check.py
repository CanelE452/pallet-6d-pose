"""Own-frozen stdlib review of V14 report transcription and public evidence.

No statistics, CIs, timing, truth, projection, fit or model is recomputed here.
Raw saved scalar/status/ID joins and public byte closure are checked separately
from the existing independent 972/81 and 56-slot arithmetic authorities.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

REPO=Path(__file__).resolve().parents[3]
DOC=REPO/'_docs/experiments/pallet_stable_incidence_guard_20261011_v14'
PRIVATE=Path('/tmp/pallet-stable-incidence-guard-private-20261011-v14')
FILES=('REPORT_SCALAR_PROTOCOL.json','REPORT_SCALAR_STARTED.json','REPORT_SCALAR_CHECKS.json','REPORT_PUBLIC_FRAME_SCALARS.csv')
METHODS=dict(BASE='BASE',N3='N3_SUBPIX',POINT='ROLE_BOUNDARY_H_ROBUST',LOCAL='ROLE_BOUNDARY_LOCAL_POINT_LINE',
    FINITE='FINITE_EDGE_CONSENSUS',STABLE='FINITE_EDGE_CONSENSUS_STABLE_INCIDENCE')
STRATA=('combined','easy','medium');SCOPES=('operational','new_pose','fallback')
PAIRED=('common_operational','candidate_new_pose','both_new_pose')
METRICS=('translation_cm','rotation_deg','ADDsym_cm')
MOMENTS=('mean','sample_variance','sample_std','median','P90','max')
TIMES=('full','detector','N3_correction','initial_pose','boundary_observation',
    'endpoint_validation_and_filtered_local','final_pose_reprojection_and_metadata','return')
EVIDENCE=('RESULT_KO.md','README.md','BUILD_LEDGER.json','METHOD_CONTRACT_KO.md','RUNTIME_CONTRACT_KO.md',
    'PROTOCOL.json','GEOMETRY_SEAL.json','GEOMETRY_SEALED.jsonl.gz','CONTROL_RECEIPT.json','CONTROL_LEDGERS.jsonl.gz',
    'VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json','SCORING_RECEIPT.json',
    'SCORING_DISPATCH_FIRST_FAILURE.json','SCORING_DISPATCH_REVIEW_PROTOCOL.json','SCORING_DISPATCH_REVIEW_STARTED.json',
    'SCORING_AUTHORITY_SUPPLEMENT.json','STATISTICS_STARTED.json','STATISTICS_AUTHORITY_SUPPLEMENT.json',
    'METRICS.json','STATISTICS_VERIFICATION_PROTOCOL.json','STATISTICS_VERIFICATION_STARTED.json','VERIFICATION.json',
    'PREDICTIONS.jsonl.gz','POSTHOC_ROWS.jsonl.gz','CPU_CHECKS.json','REVIEW_CPU_CHECKS.json','PACKAGE_CPU_CHECKS.json',
    'ROLE_ORDER_AUDIT_FIRST_FAILURE.json','REVIEW_ROLE_ORDER_AUDIT_CHECKS.json',
    'CORRESPONDENCE_DIAGNOSTIC_PROTOCOL.json','CORRESPONDENCE_DIAGNOSTIC_CHECKS.json','CORRESPONDENCE_DIAGNOSTIC_ROWS.jsonl.gz',
    'RUNTIME_PROTOCOL.json','RUNTIME_RECEIPT.json','RUNTIME_ROWS.jsonl.gz','RUNTIME_RESOURCES.json',
    'RUNTIME_CHECK_PROTOCOL.json','RUNTIME_CHECK_STARTED.json','RUNTIME_CHECKS.json',
    'FIGURE_PROTOCOL.json','FIGURES_STARTED.json','FIGURE_BINDINGS.json','FIGURE_CASE_ROWS.jsonl.gz','ROOT_VISUAL_REVIEW.json')
LIMITS=dict(new_model_GT_score_fit_projection_SVD_seed_draw_timing_calls=0,
    moments_CIs_or_timing_recomputed=False,raw_scalars_are_saved_proxy_values=True,
    private_assets_opened=0,physical_truth_or_ownership_certified=False,
    hardware_or_publication_certified=False,preserved_original_failures=True)

def require(value,message):
    if not value:raise ValueError(message)

def safe(path):
    p=Path(path).absolute();require(not any(q.is_symlink() for q in (p,*p.parents)),'symlink '+str(p));return p

def read(path):return json.loads(safe(path).read_text())

def binding(path):
    p=safe(path);require(p.is_file(),'missing '+str(p));h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())

def same(a,b):return all(a[k]==b[k] for k in ('bytes','sha256'))

def write(path,value):
    with safe(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def rows(path):
    with gzip.open(safe(path),'rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)

def finite(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
    if isinstance(value,list):return [finite(v) for v in value]
    return value

def public_path(args,value):
    p=Path(value['path'])
    if p.is_absolute():
        if not p.is_relative_to(REPO):return None
        p=p.relative_to(REPO)
    if not p.parts or p.parts[0] not in ('scripts','_docs') or '..' in p.parts:return None
    return safe(args.repo_root/p)

def closure(args):
    files={n:safe(args.input/n) for n in EVIDENCE};files['checker']=safe(Path(__file__))
    declarations=[];skipped=[];unique={}
    def inspect(value,context):
        if isinstance(value,dict):
            if {'path','bytes','sha256'}<=set(value):
                p=public_path(args,value)
                if p is None:skipped.append(context)
                else:
                    key=str(p.relative_to(args.repo_root));require(key not in unique or same(unique[key][1],value),'conflicting public binding '+key)
                    unique[key]=(p,value);declarations.append(context)
            else:
                for k,v in value.items():inspect(v,context+'/'+str(k))
        elif isinstance(value,list):
            for i,v in enumerate(value):inspect(v,context+'/'+str(i))
    for n,p in files.copy().items():
        if p.suffix=='.json':inspect(read(p),n)
        elif p.suffix=='.md':
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
                if target.startswith(('https://','http://','#')):continue
                q=safe(args.input/target.partition('#')[0]).resolve()
                require(q.is_relative_to(args.repo_root) and q.is_file(),'missing/nonpublic frozen link '+target)
                files['linked:'+str(q.relative_to(args.repo_root))]=q
    if args.public_protocol:
        files['late_public_protocol']=safe(args.public_protocol);inspect(read(args.public_protocol),'late_public_protocol')
    for key,(p,value) in unique.items():
        require(same(binding(p),value),'public byte mismatch '+key);files['public:'+key]=p
    return files,{k:binding(p) for k,p in files.items()},dict(declared_public_bindings=len(declarations),
        unique_public_files=len(unique),external_metadata_bindings_not_opened=len(skipped))

def tables(text,m,runtime,diagnostic,result):
    count=Counter();seen={k:set() for k in ('operational','moment','CI','runtime','mask','corner','correspondence')}
    def num(value,expected,label):
        if expected is None:require(value=='None','None '+label);return
        actual=float(value);require(math.isfinite(actual) and math.isfinite(expected),'finite '+label)
        d=abs(actual-expected);result['max_numeric_difference']=max(result['max_numeric_difference'],d)
        require(d<=1e-12+1e-12*abs(expected),'transcription '+label)
    def once(group,key):require(key not in seen[group],'duplicate '+group+' '+str(key));seen[group].add(key)
    correspondence=dict(NEW='NEW',fallback='기본 N3 반환',final_bad_q_eq1='최종 inlier에 bad q 1개',final_bad_q_eq2='최종 inlier에 bad q 2개',
        final_bad_q_gt2='최종 inlier에 bad q 3개 이상',accepted_correct_q_ge4='실제 fit에 correct q 4개 이상',
        correct_q_ge4_but_both_worse='correct q 4개 이상인데 N3보다 T/R 모두 나쁨',all_fit_q_known_correct='비어 있지 않은 fit q가 전부 correct',
        all_fit_q_known_correct_but_both_worse='fit q가 전부 correct인데 N3보다 T/R 모두 나쁨',
        accepted_point_count_lt4_with_lines='fit 점 4개 미만+선',accepted_correct_q_lt4_with_lines='correct fit 점 4개 미만+선',
        line_only_NEW='점 없이 선만으로 NEW',point_only_NEW='선 없이 점만으로 NEW')
    for line in text.splitlines():
        if not line.startswith('|'):continue
        c=[v.strip() for v in line.strip('|').split('|')]
        if len(c)==11 and c[0] in STRATA and c[1] in METHODS:
            key=(c[0],METHODS[c[1]]);once('operational',key);b=m['strata'][key[0]]['methods'][key[1]]
            require([int(v) for v in c[2:8]]==[b[k] for k in ('denominator','operational','new_pose','fallback','no_pose','fixed_control')],'operational counts')
            for v,k in zip(c[8:],METRICS):num(v,b['metrics']['operational'][k]['mean'],str(key)+k)
            count['operational_rows']+=1
        elif len(c)==7 and c[0] in STRATA and c[1] in METHODS and c[2] in SCOPES:
            key=(c[0],METHODS[c[1]],c[2]);once('moment',key);b=m['strata'][key[0]]['methods'][key[1]]['metrics'][key[2]]
            for cell,k in zip(c[4:],METRICS):
                values=cell.split('<br>');require(len(values)==6 and int(c[3])==b[k]['n'],'six moments and n')
                for v,f in zip(values,MOMENTS):num(v,b[k][f],str(key)+k+f);count['moment_scalars']+=1
            count['moment_rows']+=1
        elif len(c)==12 and c[0] in STRATA and c[1] in ('STABLE−N3','STABLE−FINITE','STABLE−POINT') and c[2] in PAIRED:
            left,right=c[1].split('−');key=(c[0],METHODS[left]+'_minus_'+METHODS[right],c[2]);once('CI',key)
            b=m['strata'][key[0]]['contrasts'][key[1]][key[2]]
            require([int(v) for v in c[3:9]]==[b['denominator'],b['metrics'][METRICS[0]]['n'],b['candidate_new_pose_in_pairs'],
                b['comparator_new_pose_in_pairs'],b['bootstrap_nonempty_resamples'],b['bootstrap_empty_resamples']],'paired counts')
            for cell,k in zip(c[9:],METRICS):
                mean,ci=cell.split(';',1);num(mean.strip(),b['metrics'][k]['mean_delta'],str(key)+k)
                expected=b['metrics'][k]['CI95'];actual=json.loads(ci.strip()) if ci.strip()!='None' else None
                if expected is None:require(actual is None,'empty CI');count['None_CI_slots']+=1
                else:
                    require(isinstance(actual,list) and len(actual)==2,'two CI endpoints')
                    for a,e in zip(actual,expected):num(str(a),e,str(key)+k+'CI')
                    count['nonempty_CI_slots']+=1
                count['CI_slots']+=1
            count['paired_rows']+=1
        elif len(c)==9 and c[0] in TIMES:
            key=c[0];once('runtime',key);b=runtime['summaries'][key];require(int(c[1])==b['n'] and int(c[8])==b['ddof'],'timing n/ddof')
            for v,k in zip(c[2:8],(*MOMENTS[:-1],'maximum')):num(v,b[k],'runtime'+key+k)
            count['runtime_rows']+=1;count['runtime_n_moments']+=7
        elif len(c)==6 and c[0] in STRATA and c[1] in ('WRONG_ON_KNOWN','MATCHES_ON_KNOWN','MASK_NOT_APPLIED','NO_KNOWN_VISIBILITY'):
            key=(c[0],c[1]+'__'+c[2]);once('mask',key);b=m['strata'][key[0]]['diagnostics'][METHODS['STABLE']]['mask_and_pose_groups_vs_fixed_N3'][key[1]]
            require([int(v) for v in c[3:]]==[b[k] for k in ('n','new_pose','fallback')],'mask counts');count['mask_rows']+=1
        elif len(c)==7 and c[0] in STRATA and c[1] in ('DIRECT_VISIBLE','SELF_OCCLUDED','ACTUALLY_REPROJECTED'):
            key=(c[0],c[1]);once('corner',key);b=m['strata'][key[0]]['diagnostics'][METHODS['STABLE']][key[1]]
            require(int(c[2])==b['corners'] and int(c[3])==b.get('damage_5px_to_over10px',0),'corner count/direct-damage sentinel')
            for cell,version in zip(c[4:],('N3','input','output')):
                n,rest=cell.split(';',1);values=rest.strip().split('<br>');require(int(n)==b['errors'][version]['n'] and len(values)==6,'corner n/six moments')
                for v,k in zip(values,MOMENTS):num(v,b['errors'][version][k],str(key)+version+k);count['corner_scalars']+=1
            count['corner_rows']+=1
        elif len(c)==2 and c[0] in correspondence.values():
            key=next(k for k,v in correspondence.items() if v==c[0]);once('correspondence',key)
            require(int(c[1])==diagnostic['groups'][key]['count'],'saved correspondence count');count['correspondence_rows']+=1
    require(all(count[k]==v for k,v in dict(operational_rows=18,moment_rows=54,moment_scalars=972,paired_rows=27,CI_slots=81,
        nonempty_CI_slots=72,None_CI_slots=9,runtime_rows=8,runtime_n_moments=56,mask_rows=48,corner_rows=9,corner_scalars=162,correspondence_rows=13).items()),'complete fixed tables')
    return dict(count)

def inspect(args,files,result):
    m=read(files['METRICS.json']);ledger=read(files['BUILD_LEDGER.json']);runtime=read(files['RUNTIME_RECEIPT.json'])
    for n in ('VALIDATION_CHECKS.json','VERIFICATION.json','RUNTIME_CHECKS.json','CORRESPONDENCE_DIAGNOSTIC_CHECKS.json','ROOT_VISUAL_REVIEW.json'):
        x=read(files[n]);require(x['complete'] is True and x['passed'] is True,'actual PASS '+n)
    require(m['complete'] is True and m['methods']==list(METHODS.values()) and m['actual_arithmetic']['scored_rows']==1470
        and m['primary_both_T_R_means_better_than_N3_on_full245']==ledger['goal_achieved'],'fixed statistics/goal transcription')
    require(read(files['CPU_CHECKS.json'])['passed'] is False and len(read(files['CPU_CHECKS.json'])['errors'])==3
        and read(files['REVIEW_CPU_CHECKS.json'])['tests_run']==0 and read(files['PACKAGE_CPU_CHECKS.json'])['passed'] is True,'preserved CPU attempts')
    failure=read(files['SCORING_DISPATCH_FIRST_FAILURE.json']);require(failure['GT_score_calls']==failure['scored_rows']==0 and failure['failed_before_SCORING_STARTED'] is True,'preserved pre-GT dispatch failure')
    verified=read(files['VERIFICATION.json'])['actual_counts'];require(verified['moment_scalars_checked']==972 and verified['metric_CI_slots_checked']==81,'independent972/81 authority')
    checked=read(files['RUNTIME_CHECKS.json']);require(checked['checks']==51778 and checked['measured']==130 and checked['summary_slots']==56
        and runtime['complete'] is True and runtime['statistics_official'] is True and runtime['serialized_rows']==150,'actual150/130/56 authority')
    visual=read(files['ROOT_VISUAL_REVIEW.json']);require(visual['actually_viewed_with_view_image'] is True and visual['PNG_files']==8
        and visual['case_panels']==24 and visual['aggregate_plots']==2,'actual root8image review')
    text=files['RESULT_KO.md'].read_text();result['tables']=tables(text,m,runtime,read(files['CORRESPONDENCE_DIAGNOSTIC_CHECKS.json']),result)
    require(all(word in text for word in ('아직 아니다','2164','9000','GEOMETRIC_PROXY','174.7568271983788','51778')),'negative result/history/proxy/large error disclosure')
    links=[];images=set()
    for n in ('RESULT_KO.md','README.md','METHOD_CONTRACT_KO.md','RUNTIME_CONTRACT_KO.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',files[n].read_text()):
            if target.startswith(('https://','http://','#')):continue
            name,_,fragment=target.partition('#');p=safe(args.input/name).resolve()
            require(p.is_relative_to(args.repo_root) and p.is_file(),'missing/nonpublic local link '+target)
            if fragment.startswith('L') and fragment[1:].isdigit():require(int(fragment[1:])<=len(p.read_text().splitlines()),'source line fragment')
            links.append(target)
            if n=='RESULT_KO.md' and name.startswith('figures/'):images.add(name)
    require(images=={'figures/01_all_operational.png','figures/02_same_frame_pairs.png',*('figures/case_%02d.png'%i for i in range(1,7))},'eight fixed images linked')
    result['local_links']=len(links);result['image_links']=len(images)
    cohort=read(public_path(args,m['inputs']['cohort']));ids=set(cohort['ids']);require(len(ids)==245,'full245 cohort')
    saved={};csvrows=[]
    for alias in ('new_scores','finite_scores','old_local_scores','point_scores','fixed_scores'):
        path=public_path(args,m['inputs'][alias])
        for row in rows(path):
            key=(row['method'],row['id']);require(key not in saved and key[0] in METHODS.values() and key[1] in ids,'raw saved score identity')
            saved[key]={k:row[k] for k in ('id','method','session','native_points','pose','pose_available','output_status','new_pose_estimated','fallback_used')}
            pose=row['pose'];require(row['pose_available']==pose['available'],'raw pose availability')
            for metric in ('translation_cm','rotation_deg','ADDsym_m'):
                if pose['available']:require(isinstance(pose[metric],(int,float)) and math.isfinite(pose[metric]),'finite saved scalar')
            csvrows.append(dict(id=row['id'],session=row['session'],method=row['method'],output_status=row['output_status'],
                available=pose['available'],new_pose_estimated=row['new_pose_estimated'],fallback_used=row['fallback_used'],
                **{k:pose.get(k) if pose['available'] else None for k in ('translation_cm','rotation_deg','ADDsym_m')}))
    require(len(saved)==1470 and all({fid for method,fid in saved if method==mth}==ids for mth in METHODS.values()),'all six saved populations')
    for stratum in STRATA:
        subset=ids if stratum=='combined' else {f['id'] for f in cohort['frames'] if f['label']==('clean' if stratum=='easy' else 'moderate')}
        for method in METHODS.values():
            selected=[saved[method,f] for f in subset]
            counts=dict(denominator=len(subset),operational=sum(r['pose_available'] for r in selected),
                new_pose=sum(r['new_pose_estimated'] and r['pose_available'] for r in selected),
                fallback=sum(r['fallback_used'] and r['pose_available'] for r in selected),
                no_pose=sum(not r['pose_available'] for r in selected),
                fixed_control=sum(r['output_status']=='FRESH_FIXED_CONTROL' for r in selected),
                statuses=dict(Counter(r['output_status'] for r in selected)))
            b=m['strata'][stratum]['methods'][method];require(all(counts[k]==b[k] for k in counts),'raw status-to-report join')
    observed=[]
    for row in rows(files['GEOMETRY_SEALED.jsonl.gz']):
        observed.append(row['id'])
        s=saved[METHODS['STABLE'],row['id']]
        require(all(finite(row[k])==finite(s[k]) for k in ('id','method','native_points','output_status','new_pose_estimated','fallback_used')),'geometry-to-score join')
    require(observed==cohort['ids'],'ordered full245 geometry')
    for name in ('CONTROL_LEDGERS.jsonl.gz','POSTHOC_ROWS.jsonl.gz','CORRESPONDENCE_DIAGNOSTIC_ROWS.jsonl.gz'):
        require([row['id'] for row in rows(files[name])]==cohort['ids'],'ordered full245 '+name)
    panels=list(rows(files['FIGURE_CASE_ROWS.jsonl.gz']));require(len(panels)==24,'fixed24 saved panels')
    for panel in panels:
        s=saved[panel['method'],panel['id']];d=panel['display']
        require(all(finite(d[k])==finite(s[k]) for k in ('id','method','native_points','pose','output_status','new_pose_estimated','fallback_used')),'saved figure scalar/output join')
    calls=Counter(row['phase'] for row in rows(files['RUNTIME_ROWS.jsonl.gz']));require(calls==Counter(warmup=20,measured=130),'saved150 runtime phase population')
    result.update(saved_scored_rows_joined=1470,case_panels_joined=24,runtime_saved_rows=150,raw_scalar_units={'translation_cm':'cm','rotation_deg':'degrees','ADDsym_m':'m'},
        raw_scalar_CSV_is_transcription_only=True)
    return csvrows

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--input',type=Path,default=DOC)
    p.add_argument('--output',type=Path,default=DOC);p.add_argument('--repo-root',type=Path,default=REPO);p.add_argument('--public-protocol',type=Path)
    args=p.parse_args();args.input=safe(args.input);args.output=safe(args.output);args.repo_root=safe(args.repo_root)
    require(args.input.is_relative_to(args.repo_root) and args.input.is_dir() and args.output.is_dir()
        and (args.output==args.repo_root/DOC.relative_to(REPO) or args.output.is_relative_to(PRIVATE)),'existing public input and exclusive V14 output required')
    for n in FILES if args.stage=='freeze' else FILES[1:]:require(not (args.output/n).exists(),'preserve '+n)
    files,b,public=closure(args)
    if args.stage=='freeze':
        write(args.output/FILES[0],dict(schema='stable_incidence_report_public_protocol_v14',inputs=b,public_closure=public,limits=LIMITS,
            expected_moment_scalars=972,expected_CI_slots=81,expected_timing_n_moments=56,expected_raw_score_rows=1470,frozen_before_own_checks=True));return
    own=read(args.output/FILES[0]);require(own['schema']=='stable_incidence_report_public_protocol_v14' and own['inputs']==b
        and own['public_closure']==public and own['limits']==LIMITS,'own freeze/code/evidence differs')
    write(args.output/FILES[1],dict(protocol=binding(args.output/FILES[0]),inputs=b,limits=LIMITS))
    result=dict(schema='stable_incidence_report_public_checks_v14',complete=False,passed=False,protocol=binding(args.output/FILES[0]),
        inputs=b,public_closure=public,limits=LIMITS,max_numeric_difference=0.)
    try:
        values=inspect(args,files,result)
        with (args.output/FILES[3]).open('x',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
        require(closure(args)[1:]==(b,public),'report/code/public bytes changed')
        result.update(complete=True,passed=True,scalar_CSV=binding(args.output/FILES[3]))
    except Exception as error:result['error']=dict(type=type(error).__name__,message=str(error))
    write(args.output/FILES[2],result);print('V14_REPORT_PUBLIC',result['passed'],flush=True)
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
