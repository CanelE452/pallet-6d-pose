"""Verify exported Newton/source evidence, historical images and real nonexecution."""
import argparse
import ast
import csv
import json
from pathlib import Path
import re
from urllib.parse import unquote
import numpy as np
from PIL import Image
from . import common as C


def all_bindings(value):
    if isinstance(value,dict):
        if {'path','sha256'}<=value.keys():C.verify(value)
        for item in value.values():all_bindings(item)
    elif isinstance(value,list):
        for item in value:all_bindings(item)


def csv_rows(path):
    assert b'\r' not in path.read_bytes()
    with path.open(newline='') as handle:return list(csv.DictReader(handle))


def scalar_equal(text,value):
    if value is None:return text==''
    if isinstance(value,(int,float)) and not isinstance(value,bool):return float(text)==value
    return text==str(value)


def verify_source_exports(source):
    C.verify(source['metrics'])
    lock=C.read(C.DOC/'SOURCE_VAL_ROUTING_LOCK.json');C.verify(lock['choices'])
    choices=C.read(C.ROOT/lock['choices']['path'])
    rows=csv_rows(C.DOC/'SOURCE_VAL_FRAME_RESULTS.csv')
    index={(r['model'],r['id']):r for r in rows}
    assert len(rows)==len(index)==8192
    with np.load(C.ROOT/source['metrics']['path'],allow_pickle=False) as z:
        for model in z['models'].tolist():
            for i,fid in enumerate(z['ids'].tolist()):
                row=index[model,fid]
                assert float(row['T_cm'])==z[model][i,0] and float(row['R_deg'])==z[model][i,1]
                assert row['available']=='True' and row['split']=='VAL'
                pick=choices['records'].get(model,{}).get(fid,{})
                assert row['candidate']==pick.get('candidate_name','fixed_GEO')
                assert row['fallback']==str(pick.get('fallback',False))
    checks=csv_rows(C.DOC/'SOURCE_VAL_CHECKS.csv')
    expected=[dict(model=m,baseline=b,criterion=k,PASS=v) for m,g in source['comparisons'].items()
        for b,comp in g.items() for k,v in comp['checks'].items()]
    assert len(checks)==len(expected)==45
    for row,raw in zip(checks,expected):assert row=={k:str(v) for k,v in raw.items()}
    return dict(frame_rows=8192,checks=45,passed_checks=43,failed_checks=2)


def verify_training_exports(report):
    fits={};logs=[];exports={}
    for model in C.MODEL_NAMES:
        fit=C.read(C.DOC/f'FIT_{model}.json');all_bindings(fit)
        ck=C.read(C.ROOT/fit['checkpoint']['path']);assert ck['certificate']['PASS']
        assert ck['schema']=='pallet_pose_signed_axes_newton_linear253x2_v1'
        assert np.asarray(ck['weight']).shape==(253,2) and ck['solver_rule']=='BLOCK_GENERALIZED_NEWTON_ARMIJO'
        public=C.DOC/'model_parameters'/f'{model}.json'
        assert public.read_bytes()==(C.ROOT/fit['checkpoint']['path']).read_bytes()
        exports[model]=dict(local=fit['checkpoint'],published=C.bind(public))
        raw=[json.loads(line) for line in (C.ROOT/fit['trace']['path']).read_text().splitlines()]
        logs.extend(dict(model=model,**row) for row in raw)
        fits[model]=fit
    assert report['exports']==exports
    assert {p.name for p in (C.DOC/'model_parameters').glob('*.json')}=={f'{m}.json' for m in C.MODEL_NAMES}
    counts={}
    for event,filename,total in [('objective','TRAINING_OBJECTIVE_LOG.csv',528),('iteration','TRAINING_ITERATION_LOG.csv',94)]:
        expected=[row for row in logs if row['event']==event];actual=csv_rows(C.DOC/filename)
        assert len(expected)==len(actual)==total
        # Initial and trial rows have optional fields. CSV uses their union.
        keys=set().union(*(row.keys() for row in expected))
        for row,raw in zip(actual,expected):
            assert set(row)==keys,(event,set(row)^keys)
            for key in keys:assert scalar_equal(row[key],raw.get(key)),(event,key)
        counts[event]=total
    return fits,logs,counts


def verify_figure_values(report,source,training,logs,fits):
    prior=C.read(C.RBF_DOC/'SOURCE_VAL_GATE.json');values={}
    for model in C.MODEL_NAMES:
        rows=[r for r in logs if r['model']==model and r['event']=='objective']
        values[f'train_{model}']=dict(calls=[r['call'] for r in rows],objective=[r['objective'] for r in rows],
            Huber=[r['Huber'] for r in rows],gap_ratio_to_limit=[r['gradient_gap_upper_bound']/1e-6 for r in rows],
            linf_ratio_to_limit=[r['gradient_linf']/1e-8 for r in rows],armijo_accepted=[r['armijo_accepted'] for r in rows],
            final_accepted_call=fits[model]['final_accepted_call'])
    for axis in ('translation_cm','rotation_deg'):
        for q in ('median','P90'):
            values[f'source_{axis}_{q}']=dict(
                previous_rbf_margin=[prior['summaries'][m]['full_population'][axis][q] for m in C.MODEL_NAMES],
                current_signed_newton=[source['summaries'][m]['full_population'][axis][q] for m in C.MODEL_NAMES],
                fixed_R0_GEO=source['summaries']['R0_GEO']['full_population'][axis][q])
    for axis in ('T','R'):
        values[f'train_nonanchor_{axis}']={metric:[training['models'][m]['regression_statistics']['nonanchor_valid_candidates']['axes'][axis][metric] for m in C.MODEL_NAMES] for metric in ('MAE','RMSE')}
    values['train_unsafe_count']=dict(
        previous_rbf_margin=[training['models'][m]['previous_fixed_rbf']['statistics']['anchor_violations']['either']['count'] for m in C.MODEL_NAMES],
        current_signed_newton=[training['models'][m]['statistics']['anchor_violations']['either']['count'] for m in C.MODEL_NAMES])
    assert report['figure_values']==values
    return len(values)


def verify_tables(md,source,training,fits):
    lines=md.splitlines();counts={}
    def table(header,rows):
        start=lines.index(header)+2;found=[]
        while start<len(lines) and lines[start].startswith('|'):
            found.append(lines[start]);start+=1
        assert found==rows,(header,found,rows)
        counts[header]=len(rows)
    rows=[]
    for m in C.MODEL_NAMES:
        r=training['models'][m];q=r['independent_recompute'];fit=fits[m]
        rows.append(f"| {m} | {fit['objective_calls']} | {fit['iterations']} | {r['newton_trace']['rejected_Armijo_trials']} | {q['Huber']:.9f} | {q['objective']:.9f} | {q['gradient_linf']:.3e} | {q['certified_gap_upper_bound']:.3e} |")
    table('| 모델 | objective 호출 | 승인 반복 | 거부 trial | Huber | J | gradient Linf | gap 상한 |',rows)
    old=C.read(C.SIGNED_DOC/'REJECTED_R0_ONLY.json')['certificate'];current=fits['R0_ONLY']['certificate']
    table('| 같은 R0_ONLY Huber 목적함수 비교 | 직전 L-BFGS 거부 상태 | 이번 zero-init Newton 인증 상태 |',[
        f"| J | {old['objective_value']:.15g} | {current['objective_value']:.15g} |",
        f"| gradient-gap 상한 | {old['gradient_l2_squared_over_2lambda']:.15g} | {current['gradient_l2_squared_over_2lambda']:.15g} |",
        f"| 호출 / 반복 | {old['objective_calls']} / {old['iterations']} | {current['objective_calls']} / {current['iterations']} |"])
    assert f"{old['objective_value']-current['objective_value']:.12g}" in md
    rows=[]
    for m in C.MODEL_NAMES:
        r=training['models'][m]['regression_statistics']['nonanchor_valid_candidates'];t,rot=r['axes']['T'],r['axes']['R']
        rows.append(f"| {m} | {r['candidate_pairs']} | {t['MAE']:.6f} | {t['RMSE']:.6f} | {100*t['sign_accuracy']:.3f}% | {rot['MAE']:.6f} | {rot['RMSE']:.6f} | {100*rot['sign_accuracy']:.3f}% |")
    table('| 모델 | 비-anchor 후보 수 | T MAE | T RMSE | T 부호 정확도 | R MAE | R RMSE | R 부호 정확도 |',rows)
    rows=[]
    for m in C.MODEL_NAMES:
        r=training['models'][m];old=r['previous_fixed_rbf'];parts=[]
        for axis in ('T','R','either'):parts.append(f"{old['statistics']['anchor_violations'][axis]['count']} → {r['statistics']['anchor_violations'][axis]['count']}")
        for k in ('safe_improvement','anchor'):parts.append(f"{old['risk_statistics']['classes'][k]['count']} → {r['risk_statistics']['classes'][k]['count']}")
        rows.append('| '+m+' | '+' | '.join(parts)+' |')
    table('| 모델 | T 위반 이전→현재 | R 위반 이전→현재 | 한 축 이상 위반 이전→현재 | 안전 개선 이전→현재 | anchor 선택 이전→현재 |',rows)
    rows=[]
    for m,s in source['summaries'].items():
        v=s['full_population']
        values=[v[a][q] for a,q in [('translation_cm','median'),('rotation_deg','median'),('translation_cm','P90'),('rotation_deg','P90')]]
        rows.append('| '+m+' | '+' | '.join(f'{v:.6f}' for v in values)+f" | {s['failed_pose']} |")
    table('| 모델 | T 중앙값 cm | R 중앙값 ° | T P90 cm | R P90 ° | 실패 |',rows)
    diagnostic=C.read(C.DOC/'SOURCE_TRANSFER_DIAGNOSTIC.json');rows=[]
    for m in ('UNION_s1','UNION_s2','UNION_s3'):
        r=diagnostic['models'][m];old=r['previous_RBF_actual_selection']['classes'];current=r['actual_source_selection']['classes'];known=r['known_safe_opportunities']
        rows.append(f"| {m} | {old['safe_improvement']} → {current['safe_improvement']} | {old['unsafe']} → {current['unsafe']} | {current['anchor']} | {known['frames_with_demonstrated_opportunity']} | {known['miss_lower_bound']} |")
    table('| 모델 | 안전 개선 이전 RBF→현재 | unsafe 이전 RBF→현재 | 현재 anchor 유지 | 알려진 기회 | 놓친 기회 하한 |',rows)
    for item in source['failed_checks']:assert '- `'+item+'`' in md
    return counts


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--visual-reviewed',action='store_true',required=True)
    args=parser.parse_args();assert args.visual_reviewed
    report=C.read(C.DOC/'REPORT_DATA.json');all_bindings(report)
    md=(C.DOC/'REPORT_KO.md').read_text()
    for name in ('PREFIT_REVIEW.json','TRAIN_CONVERGENCE.json','SOURCE_VAL_VERIFICATION.json'):
        audit=C.read(C.DOC/name);assert audit['complete'] and audit['PASS']
    training=C.read(C.DOC/'TRAIN_CONVERGENCE.json');complete=C.read(C.DOC/'TRAINING_COMPLETE.json')
    source=C.read(C.DOC/'SOURCE_VAL_GATE.json');not_run=C.read(C.DOC/'REAL_EVALUATION_NOT_RUN.json')
    diagnostic=C.read(C.DOC/'SOURCE_TRANSFER_DIAGNOSTIC.json');all_bindings(diagnostic)
    assert diagnostic['complete'] and diagnostic['PASS'] and not diagnostic['method_success'] and not diagnostic['goal_complete']
    assert diagnostic['known_candidate_errors_are_partial'] and diagnostic['no_new_selector_policy_evaluated']
    assert diagnostic['next_single_intervention']['status']=='PROPOSAL_ONLY_NOT_EXECUTED'
    for key in ('new_argmin_computations','new_fits','raw_GT_reads','real_reference_reads','new_real_routes','threshold_sweeps'):
        assert diagnostic[key]==0,key
    diagnostic_counts={}
    with np.load(C.ROOT/source['metrics']['path'],allow_pickle=False) as z:
        anchor=z['R0_GEO']
        for model in C.MODEL_NAMES:
            delta=z[model]-anchor
            safe=int(((delta<=0).all(1)&(delta<0).any(1)).sum())
            unsafe=int((delta>0).any(1).sum())
            actual=diagnostic['models'][model]['actual_source_selection']['classes']
            assert actual['safe_improvement']==safe and actual['unsafe']==unsafe,model
            diagnostic_counts[model]=dict(safe_improvement=safe,unsafe=unsafe)
    assert complete['complete'] and complete['all_certified'] and complete['fit_count']==4
    assert complete['total_objective_calls']==528 and complete['total_iterations']==94
    assert source['complete'] and not source['PASS'] and not source['real_routing_authorized']
    assert source['checks_passed']==43 and source['checks_total']==45
    assert source['failed_checks']==['UNION_s3/R0_ONLY/translation_cm_median_strict','UNION_s3/R0_GEO/translation_cm_median_strict']
    assert not_run['complete'] and not_run['status']=='NOT_RUN_SOURCE_GATE_FAILED'
    assert not_run['learned_real_routes']==not_run['real_metric_calls']==not_run['raw_real_reference_reads']==0
    for path,absent in not_run['absence_checks'].items():assert absent and not (C.ROOT/path).exists(),path
    assert report['status']=='CERTIFIED_TRAIN_SOURCE_GATE_FAILED'
    assert report['actual_new_fits']==report['actual_training_attempts']==report['certified_models']==4
    assert report['source_VAL_evaluated'] and report['source_VAL_checks_passed']==43 and report['source_VAL_checks_total']==45
    assert not report['learned_real_evaluated'] and not report['current_real_TR_effect_measured']
    assert not report['stable_joint_improvement_achieved'] and not report['goal_complete'] and not report['method_success']
    assert report['objective_calls']==528 and report['iterations']==94
    assert report['baseline_panels']==6 and report['current_learned_real_panels']==0
    assert report['new_metric_calls']==report['new_image_forwards']==report['new_PnP_solves']==0
    assert report['zero_initialization_verified'] and not report['previous_rejected_weight_warmstart']
    assert report['csv_rows']=={'SOURCE_VAL_FRAME_RESULTS.csv':8192,'SOURCE_VAL_CHECKS.csv':45,'TRAINING_OBJECTIVE_LOG.csv':528,'TRAINING_ITERATION_LOG.csv':94}
    source_review=verify_source_exports(source)
    fits,logs,trace_counts=verify_training_exports(report)
    rejected=sum(r['event']=='objective' and r['phase']=='trial' and not r['armijo_accepted'] for r in logs)
    assert report['rejected_Armijo_trials']==rejected==528-94-4
    figure_count=verify_figure_values(report,source,training,logs,fits)
    tables=verify_tables(md,source,training,fits)
    from .verify_gallery import verify
    gallery=verify(report)
    images=[]
    for p in sorted((C.DOC/'figures').iterdir()):
        assert p.suffix in ('.png','.jpg')
        with Image.open(p) as im:w,h=im.size;im.verify()
        assert w>500 and h>300 and 'figures/'+p.name in md
        images.append(dict(binding=C.bind(p),width=w,height=h))
    assert len(images)==6 and sum(i['binding']['path'].endswith('.jpg') for i in images)==3
    pending={'PUBLIC_REVIEW.json','PUBLIC_REVIEW_KO.md','PUBLICATION_MANIFEST.json'}
    checkout=Path('/tmp/pallet-pose-github-review-20260930');links=0
    for p in C.DOC.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if link.startswith(('https://','http://','#')):continue
            target=(p.parent/unquote(link.split('#')[0])).resolve()
            assert target.is_relative_to(C.ROOT),(p,link)
            if target.parent==C.DOC and target.name in pending:pass
            elif target.is_relative_to(C.DOC) or target.is_relative_to(C.HERE):assert target.exists(),(p,link)
            else:assert (checkout/target.relative_to(C.ROOT)).exists(),(p,link)
            links+=1
    files=[C.ROOT/'.gitignore',C.ROOT/'readme.md']
    for folder in (C.DOC,C.HERE):
        files += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in pending]
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text())
    review=dict(complete=True,PASS=True,created_at=C.now(),code=C.bind(__file__),
        source_gate_PASS=False,source_checks_passed=43,source_checks_total=45,source_export=source_review,
        trace_counts=trace_counts,rejected_Armijo_trials=rejected,certified_models=4,parameter_count=2024,
        exact_figure_value_groups=figure_count,verified_tables=tables,figures=images,gallery=gallery,
        visual_review_completed=True,markdown_links_checked=links,learned_real_evaluated=False,
        current_method_real_T_R_effect_measured=False,method_success=False,goal_complete=False,
        fixed_source_diagnostic=C.bind(C.DOC/'SOURCE_TRANSFER_DIAGNOSTIC.json'),
        independently_checked_source_selection_counts=diagnostic_counts,
        reviewed_artifacts=[C.bind(p) for p in sorted(set(files))],new_fits_by_audit=0,new_reference_metric_calls=0)
    C.save(C.DOC/'PUBLIC_REVIEW.json',review)
    C.save(C.DOC/'PUBLIC_REVIEW_KO.md','# 공개 결과 검산\n\n공개 기록 검산 PASS입니다. 네 모델의 수렴 인증과 source 43/45·전체 gate FAIL, 실사 미실행 판정을 구분했습니다.\n\n'
        '- 실제 최종 모델4개의 공개 파라미터가 원본 checkpoint와 byte 단위로 일치합니다.\n'
        '- objective528행·승인반복94행과 source8192행·45조건을 원본에 대조했습니다.\n'
        '- 그래프11개 수치 묶음과 상세 표6개를 독립 대조했습니다.\n'
        '- 그래프3개와 기존R0 사진6장·치수의 그림3개를 직접 검토했습니다. 현재 모델의 실사 결과는 없습니다.\n\n'
        '[검산 JSON](PUBLIC_REVIEW.json) · [상세 보고서](REPORT_KO.md)\n')
    print('PUBLIC_REVIEW_PASS',len(files),len(images),links)


if __name__=='__main__':main()
