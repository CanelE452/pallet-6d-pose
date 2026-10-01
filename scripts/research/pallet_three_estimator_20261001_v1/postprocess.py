"""CPU-only completion watcher and manuscript postprocessing, never experiment execution.

Use --wait to wait for the existing serial experiment queue. Without --wait,
missing or unfinished results fail immediately. Only fixed completed reports
populate the revision. PDF visual review and GitHub publication stay pending.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = ROOT / '_docs/experiments' / HERE.name
RAW = ROOT / 'data/pallet/results' / HERE.name
PAPER = ROOT / '_docs/paper/sensors_dope_extension_20261001_v1'
PYTHON = '/home/minjae/anaconda3/envs/pallet-yolo26/bin/python'
READY = 'EXPERIMENT_REPORTS_COMPLETE_PUBLICATION_PENDING'
EXPERIMENTS = {'DOPE':'pallet_dope_refiner_20261001_v1',
               'ResNet-18':'pallet_resnet18_refiner_20261001_v1'}
STOP = False
CHILD = None


def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def bind(path):
    path = Path(path).resolve()
    assert path.is_relative_to(ROOT.resolve()), path
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return dict(path=str(path.relative_to(ROOT)), sha256=h.hexdigest(), bytes=path.stat().st_size)
def verify(value):
    actual = bind(ROOT / value['path'])
    assert actual['sha256'] == value['sha256'], value['path']
    assert 'bytes' not in value or actual['bytes'] == value['bytes'], value['path']
    return ROOT / value['path']
def write(path, value, *, frozen=False):
    path = Path(path)
    assert path.resolve().is_relative_to(DOC.resolve()) or path.resolve().is_relative_to(RAW.resolve())
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if frozen and path.exists():
        assert path.read_text() == text, ('Immutable receipt differs', path)
        return
    pending = path.with_name(path.name + '.pending')
    pending.write_text(text); pending.replace(path)


def signal_stop(signum, frame):
    global STOP
    STOP = True
    if CHILD is not None and CHILD.poll() is None:
        CHILD.send_signal(signal.SIGTERM)


def status(state, stage, **extra):
    write(RAW / 'POSTPROCESS_STATUS.json', dict(state=state, stage=stage, time=now(),
        pid=os.getpid(), **extra))


def wait_for_queue(wait, poll):
    path = RAW / 'QUEUE_STATUS.json'
    previous = None
    while True:
        if STOP:
            raise RuntimeError('Postprocessing stopped while waiting')
        queue = read(path) if path.exists() else None
        state = None if queue is None else queue.get('state')
        if state == READY:
            return queue
        if state in ('STOPPED_ON_ERROR','STOPPED','CANCELLED'):
            raise RuntimeError('Experiment queue stopped; no results are manufactured: ' + str(queue))
        if not wait:
            raise RuntimeError('Experiment queue is not complete: ' + str(state))
        marker = (state, None if queue is None else queue.get('stage'))
        if marker != previous:
            status('WAITING_EXPERIMENT_QUEUE', str(marker), no_results_generated=True)
            print('POSTPROCESS_WAIT', marker, flush=True); previous = marker
        time.sleep(poll)


def verify_report_receipt(path):
    value = read(path); assert value['complete'] is True
    for binding in [value['code'],value['report'],*value['sources'],*value['figures'],*value['data_files']]:
        verify(binding)
    return value


def prerequisites(queue):
    assert queue['state'] == READY and queue['stage'] == 'all_scheduled_experiments'
    plan = read(DOC / 'QUEUE_PLAN.json')
    assert plan['model_count'] == 3 and plan['serial_GPU'] and plan['no_outcome_based_retries']
    assert plan['original_stable_TR_goal_achieved'] is False
    assert plan['code_sha256'] == bind(HERE / 'run_queue.py')['sha256']
    history = queue['completed_stages']
    assert [r['name'] for r in history] == [r['name'] for r in plan['pipeline']]
    bindings = [bind(DOC/'QUEUE_PLAN.json'), bind(RAW/'QUEUE_STATUS.json')]
    for row, stage in zip(history,plan['pipeline']):
        path = verify(row['receipt'])
        assert path.resolve() == (ROOT/stage['receipt']).resolve()
        receipt = read(path)
        assert receipt.get('complete') is True or receipt.get('PASS') is True
        bindings.append(bind(path))
    for label, name in EXPERIMENTS.items():
        folder = ROOT / '_docs/experiments' / name
        verify_report_receipt(folder/'REPORT_COMPLETE.json')
        audit = read(folder/'TRAINING_ARTIFACT_AUDIT.json')
        assert audit['complete'] is True and audit['PASS'] is True
        assert audit['fits'] == 6 and audit['total_head_updates'] == 36000
        assert verify(audit['training_complete']).resolve() == (folder/'TRAINING_COMPLETE.json').resolve()
        verify(audit['protocol']); verify(audit['code'])
        results = read(folder/'DEV_RESULTS.json')
        assert results['complete'] and results['role'] == 'REUSED_DEV'
        assert (results['positive_frames'],results['sessions'],results['gt_denominator']) == (319,13,2818)
        assert results['geometry_derived_reference_not_independent_physical_metrology'] is True
        runtime = read(folder/'RUNTIME.json')
        assert runtime['complete'] and runtime['PASS'] and runtime['measured_calls'] == 390
        assert runtime['all_cache_replays_PASS'] is True
        selection = read(folder/'SELECTION.json')
        assert selection['complete'] and selection['real_selection'] is False
        bindings += [bind(folder/n) for n in ('REPORT_COMPLETE.json','TRAINING_ARTIFACT_AUDIT.json',
            'DEV_RESULTS.json','DEV_PAIRED_RESULTS.json','RUNTIME.json','SELECTION.json','SYNTHETIC_HELDOUT.json')]
    return sorted({b['path']:b for b in bindings}.values(),key=lambda b:b['path'])


def commands():
    dope = ROOT / '_docs/experiments' / EXPERIMENTS['DOPE']
    resnet = ROOT / '_docs/experiments' / EXPERIMENTS['ResNet-18']
    return [
        ('joint_report', HERE/'report.py', [], DOC/'REPORT_COMPLETE.json'),
        ('insert_dope', PAPER/'update_dope_results.py', ['--results-dir',str(dope),'--runtime',str(dope/'RUNTIME.json')], PAPER/'DOPE_RESULTS_BINDINGS.json'),
        ('insert_resnet18', PAPER/'update_resnet_results.py', ['--results-dir',str(resnet),'--runtime',str(resnet/'RUNTIME.json')], PAPER/'RESNET_RESULTS_BINDINGS.json'),
        ('build_pdf', PAPER/'build.py', ['--render'], PAPER/'DRAFT_BUILD.json'),
    ]


def verify_insert(path, prefix):
    receipt = read(path)
    assert receipt['status'] == prefix.upper()+'_RESULTS_AND_RUNTIME_INSERTED', receipt['status']
    assert receipt['runtime_results_inserted'] is True
    assert receipt['independent_TEST_available'] is False
    assert receipt['new_training_or_inference'] == 0
    verify(receipt['code'])
    assert receipt['evidence'] and receipt['generated_tables']
    for binding in receipt['evidence'] + receipt['generated_tables'] + receipt.get('generated_assets',[]):
        verify(binding)
    if 'formatter_code' in receipt: verify(receipt['formatter_code'])
    if prefix == 'resnet':
        assert receipt['third_estimator_pending'] is False and receipt['third_estimator_outcomes_inserted'] is True
        assert receipt['three_fixed_estimator_measurements_complete'] is True
        assert receipt['stable_physical_TR_improvement_established'] is False
        assert receipt['all_estimators_improve_inferred'] is False
    assert all(Path(b['path']).name.startswith(prefix+'_') for b in receipt['generated_tables'])
    return receipt


def verify_stage_receipt(name, path):
    value = read(path)
    if name == 'joint_report':
        assert value['complete'] and value['models'] == 3
        assert value['original_stable_TR_goal_achieved'] is False
        assert value['independent_TEST'] is False and value['publication_verified'] is False
        for binding in [value['report'],value['code'],*value['evidence'],*value['outputs']]: verify(binding)
    elif name in ('insert_dope','insert_resnet18'):
        verify_insert(path, 'dope' if name == 'insert_dope' else 'resnet')
    elif name == 'build_pdf':
        assert value['complete'] and value['scope'] == 'TYPESETTING_ONLY'
        assert value['scientific_success_inferred'] is False and value['author_approval'] is False
        assert value['visual_review'] == 'PENDING' and value['not_submitted'] is True
        assert value['results_insert_status']=='DOPE_RESULTS_AND_RUNTIME_INSERTED'
        assert value['resnet_results_insert_status']=='RESNET_RESULTS_AND_RUNTIME_INSERTED'
        for binding in value['insertion_receipts'].values():verify(binding)
        for binding in [value['code'],value['compiler'],*value['inputs']]: verify(binding)
        assert {doc['document'] for doc in value['documents']} == {'manuscript','supplementary'}
        for doc in value['documents']:
            assert doc['pages'] > 0 and len(doc['rendered_pages']) == doc['pages']
            for binding in [doc['pdf'],doc['text'],doc['build_log'],*doc['rendered_pages']]: verify(binding)
    return value


def run_stage(name, script, args, receipt, dependency_bindings, script_binding):
    global CHILD
    checkpoint = RAW/'postprocess'/(name+'_COMPLETE.json')
    verify(script_binding)
    if checkpoint.exists():
        completed = read(checkpoint)
        assert completed['complete'] and completed['command'] == [PYTHON,'-B',str(script),*args]
        assert completed['script'] == script_binding and completed['inputs'] == dependency_bindings
        for binding in completed['inputs']: verify(binding)
        assert verify(completed['receipt']).resolve() == receipt.resolve()
        verify_stage_receipt(name,receipt)
        return completed
    log = RAW/'postprocess'/(name+'.log'); log.parent.mkdir(parents=True,exist_ok=True)
    command = [PYTHON,'-B',str(script),*args]
    environment = dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        MPLBACKEND='Agg',MPLCONFIGDIR='/tmp/pallet-postprocess-mpl')
    for binding in dependency_bindings: verify(binding)
    started = now()
    with log.open('a',buffering=1) as output:
        output.write(json.dumps(dict(event='START',time=started,command=command,script=script_binding))+'\n')
        status('RUNNING_CPU_POSTPROCESS',name,log=str(log.relative_to(ROOT)))
        CHILD = subprocess.Popen(command,cwd=ROOT,env=environment,stdout=output,stderr=subprocess.STDOUT)
        code = CHILD.wait(); CHILD = None
        output.write(json.dumps(dict(event='EXIT',time=now(),returncode=code))+'\n')
    if code != 0:
        raise RuntimeError('Postprocessing stage failed: %s exit=%s log=%s' % (name,code,log))
    assert not STOP, 'Stopped after child termination'
    verify(script_binding)
    for binding in dependency_bindings: verify(binding)
    verify_stage_receipt(name,receipt)
    completed = dict(complete=True,name=name,started=started,finished=now(),command=command,
        script=script_binding,inputs=dependency_bindings,receipt=bind(receipt),log=bind(log),returncode=0)
    write(checkpoint,completed,frozen=True)
    return completed


def mean(values):
    return None if any(v is None for v in values) else sum(values)/len(values)
def num(value, digits=3, scale=1):
    if value is None: return '--'
    assert math.isfinite(float(value))
    return f'{float(value)*scale:.{digits}f}'
def interval(value):
    if value.get('status') != 'COMPLETE': return 'unavailable'
    return f"{num(value['delta'])} [{num(value['low'])}, {num(value['high'])}]"


def check_joint_csv():
    historical = ROOT/'_docs/experiments/pallet_sensors_submission_v1/UNIFIED_DEV_RESULTS.json'
    sources = [bind(historical)]; expected=[]
    datasets = [('YOLO',read(historical)['methods'],'R0')]
    for label,name in EXPERIMENTS.items():
        path = ROOT/'_docs/experiments'/name/'DEV_RESULTS.json'
        datasets.append((label,read(path)['methods'],'DOPE' if label=='DOPE' else 'RESNET18')); sources.append(bind(path))
    for label,methods,base in datasets:
        for arm in ('baseline','D','P'):
            values = [methods[base]] if arm=='baseline' else [methods[arm+str(s)] for s in (1,2,3)]
            average = lambda key: mean([v[key] for v in values])
            pose = lambda key: mean([v['pose'].get(key) for v in values])
            expected.append(dict(estimator=label,arm=arm,median_px=average('median_px'),p90_px=average('p90_px'),
                matched_frames=average('matched_frames'),PCK10_percent=100*mean([v['ALL_GT_PCK']['10'] for v in values]),
                T_median_cm=pose('translation_median_cm'),R_median_deg=pose('rotation_median_deg'),
                T_p90_cm=pose('translation_p90_cm'),R_p90_deg=pose('rotation_p90_deg'),
                pose_coverage_percent=100*pose('coverage')))
    with (DOC/'RESULTS.csv').open(newline='') as stream: actual=list(csv.DictReader(stream))
    assert len(actual)==len(expected)==9
    checked=0
    for row,want in zip(actual,expected):
        assert set(row)==set(want)
        for key,value in want.items():
            if isinstance(value,str): assert row[key]==value
            elif value is None: assert row[key]=='', (key,row[key])
            else:
                assert math.isclose(float(row[key]),value,rel_tol=1e-13,abs_tol=1e-12),(key,row[key],value)
                checked+=1
    return dict(PASS=True,rows=9,numeric_values=checked,sources=sources,csv=bind(DOC/'RESULTS.csv'))


def tex_rows(text):
    inside=False;rows=[]
    for line in text.splitlines():
        if r'\midrule' in line: inside=True;continue
        if r'\bottomrule' in line: inside=False
        if inside and ' & ' in line:
            line=line.strip(); assert line.endswith(r'\\'),line
            rows.append([cell.strip() for cell in line[:-2].split(' & ')])
    return rows


def expected_numeric_tables(folder, base, prefix):
    r=read(folder/'DEV_RESULTS.json'); pair=read(folder/'DEV_PAIRED_RESULTS.json'); runtime=read(folder/'RUNTIME.json')
    shownbase='ResNet18' if base=='RESNET18' else base
    primary=[]
    for label,value in [(shownbase,r['methods'][base]),(shownbase+'+D mean',r['seed_mean']['D']),(shownbase+'+P mean',r['seed_mean']['P'])]:
        p=value['pose']
        primary.append([label,num(value['median_px']),num(value['p90_px']),num(value['ALL_GT_PCK']['10'],scale=100),
            num(value['matched_frames'],1),num(p['translation_median_cm']),num(p['rotation_median_deg']),num(p['coverage'],scale=100)])
    full=[]
    for arm in (base,'D1','D2','D3','P1','P2','P3'):
        v=r['methods'][arm];p=v['pose']
        full.append([shownbase if arm==base else arm,num(v['median_px']),num(v['p90_px']),str(v['matched_frames']),str(v['supervised_points']),
            str(2818-v['supervised_points']),num(v['ALL_GT_PCK']['10'],scale=100),str(p['n']),num(p['translation_median_cm']),num(p['rotation_median_deg'])])
    paired=[]
    for key,label in [('P_minus_'+base,'P--'+shownbase),('D_minus_'+base,'D--'+shownbase),('P_minus_D','P--D')]:
        v=pair['results'][key]
        paired.append([label,interval(v['conditional_keypoint_median']['session']),
            interval(v['pose']['translation_error_cm']),interval(v['pose']['rotation_error_deg'])])
    cost=[]
    for arm,label in [(base,shownbase),('D1',shownbase+'+D1'),('P1',shownbase+'+P1')]:
        v=runtime['results'][arm]
        cost.append([label,num(v['keypoints_ms']['median']),num(v['pose_ms']['median']),num(v['full_ms']['median']),
            num(v['full_ms']['p90']),str(v['pose_status_counts'].get('OK',0))])
    heldout=read(folder/'SYNTHETIC_HELDOUT.json');source=[]
    for arm in (base,'D1','D2','D3','P1','P2','P3'):
        v=heldout['results'][arm]
        assert v['frames']==1985 and v['gt9']==v['observed9']+v['missing9']
        source.append([shownbase if arm==base else arm,num(v['median_px']),num(v['p90_px']),
            num(v['pck10_all_gt'],scale=100),str(v['observed9']),str(v['missing9'])])
    return {prefix+'_results.tex':primary,prefix+'_full.tex':full,prefix+'_paired.tex':paired,
            prefix+'_runtime.tex':cost,prefix+'_source.tex':source}


def check_baseline_curve():
    source=ROOT/'_docs/experiments'/EXPERIMENTS['ResNet-18']/'BASELINE_TRAINING_COMPLETE.json'
    baseline=read(source)
    assert baseline['complete'] and baseline['epochs']==60 and baseline['updates']==209940
    path=PAPER/'generated_tables/RESNET_BASELINE_CURVE.csv'
    with path.open(newline='') as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==len(baseline['curves'])==60
    checked=0
    for epoch,(row,original) in enumerate(zip(rows,baseline['curves']),1):
        assert original['epoch']==epoch and original['training_frames']==55980
        expected={k:original[k] for k in ('epoch','step','lr','training_frames','training_loss')}
        expected.update(calibration_frames=original['calibration']['frames'],calibration_loss=original['calibration']['loss'],
            calibration_supervised_channels=original['calibration']['supervised_channels'])
        assert set(row)==set(expected)
        for key,value in expected.items():
            assert math.isclose(float(row[key]),value,rel_tol=1e-13,abs_tol=1e-12),(epoch,key)
            checked+=1
    return dict(PASS=True,rows=60,numeric_values=checked,source=bind(source),csv=bind(path))


def normalized_text(text):
    for old in ('\u2212','\u2013','\u2014'): text=text.replace(old,'-')
    return re.sub(r'\s+',' ',text).strip().replace('--','-')


def row_in_pdf(row, text):
    cells=[normalized_text(cell.replace(r'\%','%')) for cell in row]
    # The published table has numeric cells in fixed row order. Whitespace and
    # PDF minus/dash encoding may differ; values, signs and NA are unchanged.
    pattern=r'(?<![A-Za-z0-9.])'+r'\s+'.join(re.escape(cell).replace(r'\ ',r'\s+') for cell in cells)+r'(?![0-9.])'
    return re.search(pattern,normalized_text(text)) is not None


def audit_numeric_and_pdf():
    joint=check_joint_csv();build=verify_stage_receipt('build_pdf',PAPER/'DRAFT_BUILD.json')
    texts={};documents=[]
    for document in build['documents']:
        path=verify(document['pdf'])
        output=subprocess.check_output(['pdftotext','-layout',str(path),'-'],text=True)
        stored=verify(document['text']).read_text()
        assert output==stored,'Extracted PDF text differs from build receipt'
        assert '??' not in output and 'Lorem ipsum' not in output
        lowered=normalized_text(output).lower()
        for stale in ('DOPE measurement status: pending','ResNet18 measurement status: pending',
                      'A third estimator remains pending','No ResNet18 outcome is inserted in this draft',
                      'DOPE results remain pending in this revision','ResNet18 results remain pending in this revision'):
            assert stale.lower() not in lowered,('Stale pending result claim',stale)
        texts[document['document']]=output
        documents.append(dict(document=document['document'],pdf=bind(path),text=bind(verify(document['text'])),
            pages=document['pages'],rendered_pages=document['rendered_pages']))
    checks=[]
    for label,name in EXPERIMENTS.items():
        base='DOPE' if label=='DOPE' else 'RESNET18';prefix='dope' if label=='DOPE' else 'resnet'
        folder=ROOT/'_docs/experiments'/name
        for filename,expected in expected_numeric_tables(folder,base,prefix).items():
            path=PAPER/'generated_tables'/filename
            actual=tex_rows(path.read_text());assert actual==expected,('Numeric table source mismatch',filename,actual,expected)
            located=[]
            for row in expected:
                matches=[key for key,text in texts.items() if row_in_pdf(row,text)]
                assert matches,('Published numeric row missing from PDFs',filename,row)
                located.append(matches)
            checks.append(dict(table=bind(path),rows=len(expected),source_values_exact=True,pdf_rows_found=located,
                sources=[bind(folder/n) for n in ('DEV_RESULTS.json','DEV_PAIRED_RESULTS.json','RUNTIME.json','SYNTHETIC_HELDOUT.json')]))
    return dict(PASS=True,joint_csv=joint,baseline_curve=check_baseline_curve(),tables=checks,documents=documents,
        source_numbers_reestimated=False,new_evaluation=False,visual_review_performed=False,
        limitation='Numeric table text and hashes verified; page layout, legibility and scientific wording still require visual/author review.')


def execute(wait, poll):
    RAW.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
    with (RAW/'postprocess.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        final_path=DOC/'POSTPROCESS_COMPLETE.json'
        if final_path.exists():
            previous=read(final_path);assert previous['complete'] and previous['PASS']
            for binding in previous['reviewed_artifacts']: verify(binding)
            print('POSTPROCESS_ALREADY_COMPLETE',bind(final_path)['sha256'],flush=True);return
        queue=wait_for_queue(wait,poll);inputs=prerequisites(queue)
        rows=commands();scripts={name:bind(script) for name,script,_,_ in rows}
        plan=dict(schema='three_estimator_cpu_postprocess_plan_v1',code=bind(__file__),inputs=inputs,
            stages=[dict(name=n,script=scripts[n],args=a,receipt=str(r.relative_to(ROOT))) for n,_,a,r in rows],
            GPU_allowed=False,experiment_execution_allowed=False,publication_allowed=False)
        write(DOC/'POSTPROCESS_PLAN.json',plan,frozen=True)
        completed=[]
        for name,script,args,receipt in rows:
            if STOP:raise RuntimeError('Postprocessing stopped')
            dependencies=inputs+[r['receipt'] for r in completed]
            completed.append(run_stage(name,script,args,receipt,dependencies,scripts[name]))
        status('VERIFYING_PDF_NUMBERS','numeric_and_text_audit')
        audit=audit_numeric_and_pdf()
        # Current binding records supersede the older pending draft, but visual
        # review receipts for that draft are not promoted to this new PDF.
        reviewed={b['path']:b for b in inputs+[bind(__file__),bind(DOC/'POSTPROCESS_PLAN.json')]}
        for stage in completed:
            for binding in [stage['script'],stage['receipt'],stage['log']]:reviewed[binding['path']]=binding
        for path in [DOC/'RESULTS.csv',DOC/'REPORT_KO.md']:
            b=bind(path);reviewed[b['path']]=b
        for check in audit['tables']:
            b=check['table'];reviewed[b['path']]=b
        b=audit['baseline_curve']['csv'];reviewed[b['path']]=b
        build=read(PAPER/'DRAFT_BUILD.json')
        for binding in build['inputs']:reviewed[binding['path']]=binding
        for filename in ('DOPE_RESULTS_BINDINGS.json','RESNET_RESULTS_BINDINGS.json'):
            insertion=read(PAPER/filename)
            for binding in insertion['generated_tables']+insertion.get('generated_assets',[]):
                reviewed[binding['path']]=binding
        for document in audit['documents']:
            for binding in [document['pdf'],document['text'],*document['rendered_pages']]:reviewed[binding['path']]=binding
        receipt=dict(schema='three_estimator_cpu_postprocess_complete_v1',complete=True,PASS=True,
            completed_at=now(),stages=completed,numerical_and_pdf_audit=audit,
            actual_experiment_results_complete=True,PDFs_generated=True,
            PDF_visual_review='PENDING_FOR_CURRENT_PDF_HASHES',author_review='PENDING',GitHub_publication='PENDING',
            original_stable_TR_goal_achieved=False,independent_TEST=False,goal_complete=False,
            manuscript_submitted=False,acceptance_claimed=False,new_fits=0,new_inference=0,new_evaluation=0,GPU_calls=0,
            reviewed_artifacts=sorted(reviewed.values(),key=lambda b:b['path']))
        write(final_path,receipt,frozen=True)
        status('ACTUAL_RESULTS_AND_PDFS_COMPLETE_REVIEW_PUBLICATION_PENDING','done',receipt=bind(final_path))
        (DOC/'STATUS_KO.md').write_text('# 세 기반 추정기 실험·원고 후처리 상태\n\n'
            '실제 YOLO·DOPE·ResNet-18 결과를 연결한 종합 보고서와 원고·보충자료 PDF를 생성했습니다. '
            '완료된 원본 지표, 삽입 표, PDF 텍스트와 파일 해시를 검산했습니다.\n\n'
            '**현재 PDF의 페이지별 시각 검토, 저자 검토, GitHub 게시가 대기 중입니다.** '
            '과거 초안의 PDF 검토는 새 PDF에 승계하지 않습니다.\n\n'
            '실사 T/R의 안정적 공동 개선과 독립 TEST 일반화는 입증되었다고 표시하지 않습니다. '
            '논문 투고·채택이나 전체 목표 완료를 뜻하지 않습니다.\n\n'
            '[종합 결과](REPORT_KO.md) · [후처리 검산](POSTPROCESS_COMPLETE.json)\n')
        print('POSTPROCESS_COMPLETE_REVIEW_AND_PUBLICATION_PENDING',bind(final_path)['sha256'],flush=True)


def selfcheck():
    text='\\begin{tabular}{lr}\nArm & Value\\\\\\midrule\nDOPE & 1.000\\\\\nP--DOPE & --\\\\\n\\bottomrule\\end{tabular}\n'
    assert tex_rows(text)==[['DOPE','1.000'],['P--DOPE','--']]
    assert row_in_pdf(['DOPE','1.000'],'DOPE    1.000')
    assert row_in_pdf(['P--DOPE','--'],'P\u2013DOPE  \u2013')
    assert not row_in_pdf(['DOPE','1.000'],'DOPE    1.001')
    assert num(None)=='--' and num(-1.23456)=='-1.235'
    print(json.dumps(dict(PASS=True,synthetic_only=True,actual_results_read=0,subprocesses_launched=0,
        files_written=0,GPU_calls=0)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--poll-seconds',type=float,default=30.)
    parser.add_argument('--selfcheck',action='store_true')
    args=parser.parse_args();assert 1 <= args.poll_seconds <= 60
    if args.selfcheck:selfcheck();return
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    try:execute(args.wait,args.poll_seconds)
    except BaseException as error:
        status('STOPPED_ON_ERROR','postprocess',error_type=type(error).__name__,error=str(error))
        raise

if __name__=='__main__':main()
