"""Check public records of an optimization rejection, not method success."""
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


def verify_bindings(value):
    if isinstance(value,dict):
        if {'path','sha256'}<=value.keys():C.verify(value)
        for item in value.values():verify_bindings(item)
    elif isinstance(value,list):
        for item in value:verify_bindings(item)


def csv_rows(path):
    assert b'\r' not in path.read_bytes()
    with path.open(newline='') as handle:return list(csv.DictReader(handle))


def equal_scalar(text,value):
    if isinstance(value,(int,float)) and not isinstance(value,bool):return float(text)==value
    return text==str(value)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--visual-reviewed',action='store_true',required=True)
    args=parser.parse_args();assert args.visual_reviewed
    report=C.read(C.DOC/'REPORT_DATA.json');md=(C.DOC/'REPORT_KO.md').read_text()
    verify_bindings(report)
    for name in ('PREFIT_REVIEW.json','REJECTED_FIT_VERIFICATION.json','EVALUATION_NOT_RUN.json'):
        item=C.read(C.DOC/name);assert item['complete'] and item['PASS'],name
    rejected=C.read(C.DOC/'REJECTED_R0_ONLY.json');verify_bindings(rejected)
    cert=rejected['certificate'];assert rejected['complete'] and not rejected['accepted']
    assert not cert['PASS'] and not cert['optimizer_success']
    assert cert['iterations']==1000 and cert['objective_calls']==1108
    assert cert['gradient_l2_squared_over_2lambda']>cert['max_gap_upper_bound']==1e-6
    assert np.asarray(rejected['final_weight']).shape==(253,2)
    assert report['status']=='FIT_REJECTED'
    assert report['actual_training_attempts']==report['rejected_attempts']==1 and report['certified_models']==0
    assert not report['source_VAL_evaluated'] and not report['learned_real_evaluated'] and not report['goal_complete']
    assert report['exports']=={}
    assert report['rejected_fit']==C.bind(C.DOC/'REJECTED_R0_ONLY.json')
    assert report['certificate']==cert and report['solver']==rejected['solver']
    assert report['rejected_exports']['R0_ONLY']['deployable'] is False
    assert (C.DOC/'rejected_optimizer_state/R0_ONLY.json').read_bytes()==(C.DOC/'REJECTED_R0_ONLY.json').read_bytes()
    logs=[json.loads(line) for line in (C.ROOT/rejected['trace']['path']).read_text().splitlines()]
    counts={}
    for event,filename,number in [('objective','TRAINING_OBJECTIVE_LOG.csv',1108),('iteration','TRAINING_ITERATION_LOG.csv',1000)]:
        raw=[r for r in logs if r['event']==event];exported=csv_rows(C.DOC/filename)
        assert len(raw)==len(exported)==number
        for row,record in zip(exported,raw):
            expected=dict(model='R0_ONLY',**record)
            assert set(row)==set(expected)
            for key,value in expected.items():assert equal_scalar(row[key],value),(event,key)
        counts[event]=len(exported)
    assert len(logs)==2108
    calls=[r for r in logs if r['event']=='objective']
    iterations=[r for r in logs if r['event']=='iteration']
    figure_values=dict(objective_calls=[r['call'] for r in calls],
        objective=[r['objective'] for r in calls],Huber=[r['Huber'] for r in calls],
        gradient_gap_upper_bound=[r['gradient_l2']**2/(2*cert['lambda_l2']) for r in calls],
        certificate_limit=cert['max_gap_upper_bound'],iteration=[r['iteration'] for r in iterations],
        iteration_objective_calls=[r['objective_calls'] for r in iterations],
        iteration_objective=[r['objective'] for r in iterations],
        iteration_gradient_gap_upper_bound=[r['gradient_l2']**2/(2*cert['lambda_l2']) for r in iterations])
    assert report['figure_values']==figure_values
    fmt=lambda v:f'{v:.12g}'
    for key in ('objective_value','Huber','L2_penalty','gradient_l2','gradient_l2_squared_over_2lambda'):
        assert fmt(cert[key]) in md,key
    assert f"{cert['gradient_l2_squared_over_2lambda']/cert['max_gap_upper_bound']:.6f}배" in md
    independent=C.read(C.DOC/'REJECTED_FIT_VERIFICATION.json')
    independent_rows={
        'Huber + ridge J':f"{independent['independent_recompute']['objective']:.15g}",
        'gradient L2':f"{independent['independent_recompute']['gradient_l2']:.15g}",
        '목적함수 gap 상한':f"{independent['independent_recompute']['certified_gap_upper_bound']:.15g}",
        '506×506 Hessian 최소 고유값':f"{independent['Hessian']['min']:.12g}",
        'Hessian 최대 고유값':f"{independent['Hessian']['max']:.12g}",
        'Hessian condition number':f"{independent['Hessian']['condition_number']:.9f}",
        '정확히 Huber 경계인 잔차 수':str(independent['Hessian']['exact_kink_count']),
    }
    for label,value in independent_rows.items():assert f'| {label} | {value} |' in md,label
    assert {p.name for p in (C.RAW/'fits').iterdir()}=={'R0_ONLY'}
    for path in [C.DOC/'TRAINING_COMPLETE.json',C.RAW/'fits/R0_ONLY/final.json',
                 C.DOC/'SOURCE_VAL_GATE.json',C.RAW/'SOURCE_VAL_CHOICES.json',
                 C.DOC/'SOURCE_VAL_FRAME_RESULTS.csv',C.DOC/'REAL_PROTOCOL.json',
                 C.RAW/'REAL_CHOICES.json',C.DOC/'REAL_RESULTS.json',C.DOC/'REAL_FRAME_RESULTS.csv']:
        assert not path.exists(),str(path)
    assert not list(C.DOC.glob('FIT_*.json'))
    assert not (C.DOC/'model_parameters').exists()
    from .verify_gallery import verify
    gallery=verify(report)
    images=[]
    for p in sorted((C.DOC/'figures').iterdir()):
        assert p.suffix in ('.png','.jpg')
        with Image.open(p) as im:
            w,h=im.size;im.verify()
        assert w>500 and h>300 and 'figures/'+p.name in md
        images.append(dict(binding=C.bind(p),width=w,height=h))
    assert len(images)==5 and sum(i['binding']['path'].endswith('.jpg') for i in images)==3
    pending={'PUBLIC_REVIEW.json','PUBLIC_REVIEW_KO.md','PUBLICATION_MANIFEST.json'}
    checkout=Path('/tmp/pallet-pose-github-review-20260930');links=0
    for p in C.DOC.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if link.startswith(('https://','http://','#')):continue
            rel=unquote(link.split('#')[0]);target=(p.parent/rel).resolve()
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
    result=dict(complete=True,PASS=True,created_at=C.now(),code=C.bind(__file__),status='TRAIN_CONVERGENCE_FAILED',
        objective_rows=counts['objective'],iteration_rows=counts['iteration'],new_fit_attempts=1,certified_models=0,
        source_VAL_evaluated=False,learned_real_evaluated=False,current_method_T_R_effect_measured=False,
        figures=images,gallery=gallery,visual_review_completed=True,markdown_links_checked=links,
        exact_figure_value_groups=len(figure_values),rejected_certificate_values_checked=6,
        independent_recomputed_table_values_checked=len(independent_rows),
        reviewed_artifacts=[C.bind(p) for p in sorted(set(files))],new_fits_by_audit=0,new_reference_metric_calls=0,
        method_success=False,goal_complete=False)
    C.save(C.DOC/'PUBLIC_REVIEW.json',result)
    C.save(C.DOC/'PUBLIC_REVIEW_KO.md','# 공개 결과 검산\n\n공개 기록의 일치 검산은 PASS입니다. 학습 수렴 인증은 FAIL이며 현재 방법의 T/R 효과는 미측정입니다.\n\n'
        '- 실제 한 번의 학습 시도, objective1,108행과 iteration1,000행을 원본과 대조했습니다.\n'
        '- 공개한506개 값은 거부된 최적화 종료 상태이며 사용할 모델 checkpoint가 아닙니다.\n'
        '- 사진6장과 치수는 기존 R0 입력 예시이며 현재 모델의 실사 결과가 아닙니다.\n'
        '- 후속3fit·합성VAL·실사평가 산출물 부재를 확인했습니다.\n\n'
        '[검산 상세](PUBLIC_REVIEW.json) · [결과 보고서](REPORT_KO.md)\n')
    print('PUBLIC_REVIEW_PASS',len(files),len(images),links)


if __name__=='__main__':main()
