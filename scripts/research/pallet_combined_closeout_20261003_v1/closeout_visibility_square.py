"""Read-only static visibility and separate square-panel closeout.

No training, inference, annotations, original receipts, PDF, or external writes.
Whole-object symmetry is selected once on the unchanged full reference per frame;
visibility strata never select their own symmetry branch.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_n3_completion_v3 import common as C, evaluation as E, metrics as M, reuse as R
from scripts.research.pallet_n3_completion_v3 import square as S, square_yolo as Y
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math
from scripts.research.pallet_static_registry_review_20261003_v1 import open_corner_visibility as V
from scripts.research.pallet_static_registry_review_20261003_v1 import open_existing_square150 as G
from scripts.research.pallet_static_registry_review_20261003_v1 import audit as A

OUT = ROOT/'_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/visibility_square'
CATEGORIES = ('DIRECT_VISIBLE', 'EXTERNAL_OCCLUDED', 'SELF_OCCLUDED', 'OUT_OF_FRAME', 'UNKNOWN')
INPUTS = {}
CHECKS = {}


def bind(path):
    path = Path(path).resolve()
    entry = dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                 sha256=C.sha256(path), bytes=path.stat().st_size)
    INPUTS[str(path)] = entry
    return entry


def read(path):
    bind(path)
    return C.read(path)


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(A.clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def table(name, rows):
    if not rows:
        return
    keys = list(rows[0])
    with (OUT/(name+'.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    def fmt(value):
        if value is None: return 'NA'
        return f'{value:.6f}' if isinstance(value, float) else str(value)
    lines = ['| '+' | '.join(keys)+' |', '|'+'|'.join(['---']*len(keys))+'|']
    lines += ['| '+' | '.join(fmt(row[key]) for key in keys)+' |' for row in rows]
    (OUT/(name+'.md')).write_text('\n'.join(lines)+'\n')


def merge_visibility():
    context = V.load_context()
    native = V.VisibilityReview(context, V.OUT/'STATIC_CORNER_VISIBILITY_INPUTS.json')
    read(V.MANIFEST); read(V.SEVERITY); read(native.store_path)
    for case in context['frames'].values():
        bind(ROOT/case['image']['path']); bind(ROOT/case['annotation']['path'])
    replay = {}
    for event in native.store['history']:
        if event.get('undone'):
            continue
        fid = event['case_id']
        if replay.get(fid, {}) != event['previous_corners']:
            raise RuntimeError('Visibility history does not replay: '+fid)
        replay[fid] = copy.deepcopy(event['current_corners'])
    CHECKS['visibility_history_replays_current_records'] = all(
        replay.get(fid, {}) == row['corners'] for fid,row in native.store['records'].items())
    if not CHECKS['visibility_history_replays_current_records']:
        raise RuntimeError('Visibility history replay/current records mismatch')
    labels, rows, sources = {}, [], Counter()
    for fid,case in context['frames'].items():
        for point in case['corners']:
            if not point['metric_reference']:
                continue
            ci = point['corner_id']; status = native.state_for(fid, ci)
            if status not in CATEGORIES:
                raise RuntimeError('Visibility still unresolved: '+fid+':'+str(ci))
            source = 'locked_prior_approved_visibility' if point['locked'] else 'new_explicit_human_visibility_sidecar'
            labels[(case['population'],case['frame_id'],ci)] = status
            sources[source] += 1
            entry = native.store['records'].get(fid,{}).get('corners',{}).get(str(ci))
            if point['locked']:
                if entry is not None:
                    raise RuntimeError('New sidecar overlaps a protected locked point')
            elif (entry.get('human_input_status') not in
                    ('HUMAN_EXPLICIT_VISIBILITY_INPUT','HUMAN_CONFIRMED_GEOMETRY_PROPOSAL')
                    or entry.get('input_action') not in
                    ('human_keyboard','human_keyboard_geometry_confirmation','human_button',
                     'human_button_geometry_confirmation')):
                raise RuntimeError('Visibility lacks explicit human action: '+fid)
            rows.append(dict(population=case['population'], frame_id=case['frame_id'], case_id=fid,
                corner_id=ci, category=status, reference_xy=point['xy'],
                reference_in_frame=point['reference_in_frame'], reference_source=point['reference_source'],
                label_source=source, label_evidence=point['source'] if point['locked'] else entry,
                image_sha256=case['image']['sha256'], annotation_sha256=case['annotation']['sha256'],
                source_coordinates_replaced=False, independent_reference_claim=False))
    counts = Counter(r['category'] for r in rows)
    population_counts = {pop:dict(Counter(r['category'] for r in rows if r['population']==pop))
                         for pop in ('DEV319','GREEN0918')}
    CHECKS['exact_3030_new_plus_71_locked'] = (
        len(rows)==3101 and sources['new_explicit_human_visibility_sidecar']==3030
        and sources['locked_prior_approved_visibility']==71 and native.counts()['pending']==0)
    CHECKS['visibility_all_438_frames_present'] = len({r['case_id'] for r in rows})==438
    if not all(CHECKS.values()): raise RuntimeError('Static visibility denominator mismatch')
    artifact = dict(schema='static_visibility_merge_closeout_20261006_v1',
        new_human_states=3030, protected_prior_states=71, reference_corner_total=3101,
        missing_states=0, frames=438, counts=dict(counts), population_counts=population_counts,
        input_bindings=context['bindings'], history_events=len(native.store['history']),
        history_replayed=True, locked_states_not_overwritten=True, reference_coordinates_not_changed=True,
        actor=native.store['actor'], prior_model_prediction_exposure='NOT_CONFIRMED',
        independent_6D_reference_created=False, rows=rows)
    write('STATIC_VISIBILITY_MERGE_AUDIT.json',artifact)
    return context,labels,artifact


def attach_normalized(truth, payload):
    normalized = E.normalize_prediction_payload(payload)
    if not normalized['complete'] or not normalized['coordinate_ok']:
        raise RuntimeError('Incomplete/non-original-coordinate fixed prediction')
    rows = copy.deepcopy(truth)
    for row in rows:
        row.update(predictions={}, matched={}, detected={})
        for method,records in normalized['methods'].items():
            pred = records[row['id']]
            row['predictions'][method] = pred['points']
            row['detected'][method] = pred['detected']
            row['matched'][method] = bool(pred['detected'] and E._iou(pred['bbox'],row['box'])>=.5)
    for method in ('n3_seed1','n3_seed2','n3_seed3'):
        for fid,base in normalized['methods']['base'].items():
            reason = E._preservation_error(base,normalized['methods'][method][fid])
            if reason: raise RuntimeError('Fixed N3 detection-preservation drift: '+reason)
    return rows


def canonical_points(rows, method, scored, labels, population, backbone, mode):
    result = []
    for row,score in zip(rows,scored):
        if row['id'] != score['id']: raise RuntimeError('Score order mismatch')
        if not score['evaluable']: continue
        pred = row['predictions'][method]
        p = np.full((9,2),np.nan) if pred is None else np.asarray(pred,float)
        pv = np.isfinite(p).all(-1) & ~(p==-1).all(-1) & bool(score['matched'])
        permutation = np.asarray(row['permutations'],int)[score['branch']]
        observed = np.zeros(8,bool)
        for prediction_id,canonical_id in enumerate(permutation[:8]):
            observed[canonical_id] = bool(pv[prediction_id] and score['canonical_valid'][canonical_id])
        observed_values = [score['canonical_errors'][i] for i in range(8) if observed[i]]
        if sorted(observed_values) != sorted(score['observed_errors']):
            raise RuntimeError('Canonical observed-mask reconstruction disagrees with locked scorer')
        for ci,valid in enumerate(score['canonical_valid']):
            if not valid: continue
            result.append(dict(population=population, backbone=backbone, mode=mode, method=method,
                frame_id=row['id'], session=row['session'], corner_id=ci,
                category=labels[(population,row['id'],ci)], error_px=score['canonical_errors'][ci],
                observed=bool(observed[ci]), matched=score['matched'], detected=score['detected'],
                whole_object_branch=score['branch']))
    return result


def summarize_points(points, category):
    full = np.asarray([p['error_px'] for p in points],float)
    observed = np.asarray([p['error_px'] for p in points if p['observed']],float)
    return dict(category=category, frames=len({p['frame_id'] for p in points}),
        reference_corners=len(points), observed_corners=len(observed),
        unobserved_reference_corners=len(points)-len(observed),
        median_px=float(np.median(observed)) if len(observed) else None,
        P90_px=float(np.quantile(observed,.9)) if len(observed) else None,
        PCK10_percent=100*float(np.mean(full<=10)) if len(full) else None,
        metric_status='MEASURED' if len(full) else 'NA_EMPTY_CATEGORY')


def family_name(method):
    if method in ('R0','base'): return 'Base'
    if method.startswith('OLD_P'): return 'P'
    if method.startswith('N2'): return 'N2'
    if method.startswith(('N3','n3')): return 'N3'
    if method.startswith('N0'): return 'N0_BASE_REPLAY'
    return method


def family_rows(rows, grouping):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in grouping)+(family_name(row['method']),)].append(row)
    output=[]
    for key,items in buckets.items():
        result=dict(zip(grouping,key[:-1]),method=key[-1],seeds=1 if key[-1]=='Base' else len(items),
                    aggregation='single fixed base' if key[-1]=='Base' else 'mean of per-seed statistics')
        for field in items[0]:
            if field in (*grouping,'method','seed'): continue
            values=[item[field] for item in items]
            if all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in values):
                result[field] = values[0] if len(set(values))==1 else float(np.mean(values))
            elif all(v==values[0] for v in values): result[field]=values[0]
        output.append(result)
    return output


def visibility_metrics(context, labels):
    dev,dev_sources=R.load_dev_context(ROOT,include_pose=False)
    for entry in dev_sources.values(): bind(ROOT/entry['path'])
    yolo_methods=['R0']
    for arm in ('OLD_P','N2_DIM_ONLY','N3_DIM_SYM'):
        for seed in (1,2,3):
            method,sources=R.attach_dcp(dev,ROOT,arm,seed)
            yolo_methods.append(method)
            for entry in sources.values(): bind(ROOT/entry['path'])
    frozen_yolo=read(ROOT/'data/pallet/results/pallet_n3_static_closeout_v1/YOLO_SCORES.json')
    fixed={('DEV319','yolo','locked_2499'):(dev,yolo_methods,frozen_yolo)}
    for backbone in ('dope','resnet18'):
        payload=read(C.RAW/f'predictions/{backbone}_DEV319.json')
        cached=read(C.RAW/f'evaluation/{backbone}.json')
        fixed[('DEV319',backbone,'locked_2499')]=(attach_normalized(dev,payload),list(E.METHODS),cached)
    square_modes={mode:S.truth_rows(in_frame_only=mode=='manual_in_frame')
                  for mode in ('manual_declared','manual_in_frame')}
    yraw=read(C.RAW/'SQUARE_YOLO_PREDICTIONS.json')
    Y.validate_final_predictions(yraw['predictions'],Y._records(read(S.SNAPSHOT)))
    for mode,truth in square_modes.items():
        yrows=copy.deepcopy(truth)
        for row in yrows: row.update(predictions={},matched={},detected={})
        for method,predictions in yraw['predictions'].items():
            candidates={p['id']:R._record_candidate(p) for p in predictions}
            R._attach(yrows,method,candidates)
        fixed[('GREEN0918','yolo',mode)]=(yrows,list(Y.METHODS),None)
        for backbone in ('dope','resnet18'):
            payload=read(C.RAW/f'predictions/{backbone}_GREEN0918_119.json')
            fixed[('GREEN0918',backbone,mode)]=(attach_normalized(truth,payload),list(E.METHODS),None)
    point_rows,summary_rows,square_scores=[],[],{}
    for (population,backbone,mode),(rows,methods,cache) in fixed.items():
        for method in methods:
            scored=M.score_corner_rows(rows,method)
            if cache is not None:
                previous=(cache[method]['corner_scores'] if backbone=='yolo' else
                          cache['methods'][method]['result']['corner_rows'])
                normalize=lambda values:[{k:v for k,v in r.items() if k!='occlusion'} for r in values]
                equal=normalize(scored)==normalize(previous)
                CHECKS[f'raw_regression:{population}:{backbone}:{method}']=equal
                if not equal: raise RuntimeError('Raw fixed-score regression failed: '+backbone+'/'+method)
            if population=='GREEN0918': square_scores[(backbone,mode,method)]=scored
            points=canonical_points(rows,method,scored,labels,population,backbone,mode)
            point_rows.extend(points)
            for category in (*CATEGORIES,'ALL'):
                selected=points if category=='ALL' else [p for p in points if p['category']==category]
                summary_rows.append(dict(population=population,backbone=backbone,mode=mode,method=method,
                    seed='single' if method in ('R0','base') else method[-1],
                    **summarize_points(selected,category)))
            expected=2499 if population=='DEV319' else (602 if mode=='manual_declared' else 600)
            if len(points)!=expected: raise RuntimeError('Visibility denominator differs across fixed methods')
            summary=eval_math.summary(scored)
            if (sum(p['observed'] for p in points)!=summary['observed_corners']
                    or summarize_points(points,'ALL')['median_px']!=summary['matched_pooled_corner8_median_px']):
                raise RuntimeError('Conditional metric pooled-ALL regression failed')
    families=family_rows(summary_rows,('population','backbone','mode','category'))
    write('VISIBILITY_RESULTS.json',dict(schema='static_visibility_metrics_closeout_20261006_v1',
        rows=summary_rows,family_rows=families,
        contract=dict(symmetry='Unchanged full-reference whole-object branch; then canonical GT identity stratification',
                      conditional='Finite predicted reference corners on matched frames only',
                      full_PCK10='All eligible reference corners; detection/corner misses retain image-diagonal penalties',
                      seeds='Arithmetic mean of per-seed statistics; no prediction averaging'),
        prior_prediction_exposure='NOT_CONFIRMED', source_coordinates='Frozen existing reference; not new independent GT'))
    table('VISIBILITY_RESULTS',families);table('VISIBILITY_PER_SEED',summary_rows)
    with (OUT/'PER_POINT_SCORES.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(point_rows[0]));writer.writeheader();writer.writerows(point_rows)
    return square_modes,square_scores


def square_panel(modes,scores):
    old=read(ROOT/'_docs/experiments/pallet_n3_static_closeout_v1/SQUARE_AUDIT.json')
    snapshot=read(S.SNAPSHOT);read(S.SYMMETRY)
    per_seed=[]
    for (backbone,mode,method),scored in scores.items():
        summary=eval_math.summary(scored)
        if backbone=='yolo':
            family=old['results'][backbone][mode][{'Base':'R0','P':'OLD_P','N2':'N2_DIM_ONLY','N3':'N3_DIM_SYM'}[family_name(method)]]
            previous=family['single'] if method=='R0' else family['seeds'][method[-1]]
            equal=summary==previous
        else:
            headline=E._headline(dict(corner=M._corner_summary(scored),pose=None))
            equal=headline==old['results'][backbone][mode][method]
        CHECKS[f'square_regression:{backbone}:{mode}:{method}']=equal
        if not equal: raise RuntimeError('Square frozen regression failed: '+backbone+mode+method)
        per_seed.append(dict(backbone=backbone,mode=mode,method=method,
            seed='single' if method in ('R0','base') else method[-1],frames=summary['total_frames'],
            reference_corners=summary['corners'],observed_corners=summary['observed_corners'],
            detected_frames=summary['detected'],matched_frames=summary['matched'],
            median_px=summary['matched_pooled_corner8_median_px'],P90_px=summary['matched_pooled_corner8_P90_px'],
            PCK10_percent=100*summary['PCK']['10'],translation_cm='x',rotation_deg='x',CI='NA_SINGLE_SESSION'))
    families=family_rows(per_seed,('backbone','mode'))
    comparisons=[]
    for mode in modes:
        for backbone in ('yolo','dope','resnet18'):
            panel={r['method']:r for r in families if r['backbone']==backbone and r['mode']==mode}
            for before in (('Base','N2') if backbone=='yolo' else ('Base',)):
                a,b=panel[before],panel['N3']
                comparisons.append(dict(backbone=backbone,mode=mode,before=before,after='N3',
                    median_difference_px=b['median_px']-a['median_px'],
                    P90_difference_px=b['P90_px']-a['P90_px'],PCK10_difference_percentage_points=b['PCK10_percent']-a['PCK10_percent'],
                    statistic='mean(seed median after - median before); not median(frame after-before)'))
    outside=[]
    inframe={row['id']:row for row in modes['manual_in_frame']}
    for row in modes['manual_declared']:
        for ci in range(8):
            if row['valid'][ci] and not inframe[row['id']]['valid'][ci]:
                outside.append(dict(id=row['id'],corner_id=ci,xy=row['gt'][ci],hw=row['hw']))
    CHECKS['square_exact_outside_two_corners']=outside==old['outside_declared_corners']
    write('SQUARE119_RESULTS.json',dict(schema='square119_dual_mode_closeout_20261006_v1',
        frames=119,sessions=1,manual_declared=602,manual_in_frame=600,
        rows=per_seed,family_rows=families,comparisons=comparisons,outside_declared_corners=outside,
        reference_6D='x: no independent canonical physical reference',
        intervals='NA: single correlated session; no generalization CI',
        dimension_effect='Not identifiable: all119 share1.1x1.1x0.15m; package transfer only',
        source_snapshot=bind(S.SNAPSHOT),source_prior_audit=bind(ROOT/'_docs/experiments/pallet_n3_static_closeout_v1/SQUARE_AUDIT.json')))
    table('SQUARE119_RESULTS',families);table('SQUARE119_PER_SEED',per_seed);table('SQUARE119_COMPARISONS',comparisons)
    return families


def historical_square150():
    ctx=G.load_context();native=G.ExistingSquareReview(ctx,G.OUT/'GREEN150_SEVERITY_INPUTS.json')
    snapshot=read(G.SNAPSHOT);read(native.store_path)
    labels={row['frame_id'].removeprefix('GREEN150::'):row['severity'] for row in native.store['records'].values()}
    if len(labels)!=150 or native.counts()!=dict(clean=103,moderate=44,severe=3,unreviewed=0):
        raise RuntimeError('Historical150 severity membership mismatch')
    for row in snapshot['records']:
        bind(ROOT/row['image']['path']);bind(ROOT/row['annotation']['path'])
    path=ROOT/'data/pallet/results/final_dimension_v1/green150_saved_labels_v1/METRICS.json'
    old=read(path);C.verify(old['predictions']);bind(ROOT/old['predictions']['path'])
    payload=read(ROOT/old['predictions']['path']);C.verify(payload['dataset']);C.verify(payload['model'])
    bind(ROOT/payload['dataset']['path']);bind(ROOT/payload['model']['path'])
    rows=[]
    for mode,entry in old['modes'].items():
        for method,scores in entry['rows'].items():
            if {r['id'] for r in scores}!=set(labels): raise RuntimeError('Historical150 score identity mismatch')
            CHECKS[f'historical150_summary:{mode}:{method}']=eval_math.summary(scores)==entry['summary'][method]
            for group in ('all','clean','moderate','severe'):
                selected=scores if group=='all' else [r for r in scores if labels[r['id']]==group]
                summary=eval_math.summary(selected)
                rows.append(dict(population='HISTORICAL_GREEN150',mode=mode,severity=group,method=method,
                    seed='single' if method=='R0' else method[-1],frames=summary['total_frames'],reference_corners=summary['corners'],
                    observed_corners=summary['observed_corners'],matched_frames=summary['matched'],
                    median_px=summary['matched_pooled_corner8_median_px'],P90_px=summary['matched_pooled_corner8_P90_px'],
                    PCK10_percent=100*summary['PCK']['10'],translation_cm='x',rotation_deg='x'))
    families=family_rows(rows,('population','mode','severity'))
    artifact=dict(schema='historical_square150_separate_closeout_20261006_v1',frames=150,
        sessions=len({r['session'] for r in snapshot['records']}),severity_counts=dict(Counter(labels.values())),
        reference_contract=old['protocol'],manual_corner_denominator=681,all_known_proxy_denominator=1200,
        rows=rows,family_rows=families,N3='x: no same-contract fixed N3 predictions found for this150 cohort',
        dope='x: no same-contract frozen DOPE predictions',resnet18='x: no same-contract frozen ResNet predictions',
        independent_6D='x',CI='NA: correlated capture families; no independent-cluster claim',
        must_not_merge_with_square119=True,source_snapshot=bind(G.SNAPSHOT),source_metrics=bind(path))
    write('HISTORICAL_SQUARE150.json',artifact)
    table('HISTORICAL_SQUARE150',families);table('HISTORICAL_SQUARE150_PER_SEED',rows)
    return artifact


def main():
    started=time.monotonic()
    for module in (sys.modules[__name__],C,E,M,R,S,Y,V,G,A,eval_math):
        bind(module.__file__)
    context,labels,merged=merge_visibility()
    print('VISIBILITY_MERGE',merged['counts'],flush=True)
    modes,scores=visibility_metrics(context,labels)
    families=square_panel(modes,scores)
    historical=historical_square150()
    for path,binding in INPUTS.items():
        if C.sha256(path)!=binding['sha256']: raise RuntimeError('Input changed during closeout: '+path)
    if not all(CHECKS.values()): raise RuntimeError('Closeout regression failed')
    receipt=dict(schema='visibility_square_execution_20261006_v1',status='PASS',
        executed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
        command=f'{sys.executable} -m scripts.research.pallet_combined_closeout_20261003_v1.closeout_visibility_square',
        input_bindings=list(INPUTS.values()),checks=CHECKS,checks_passed=len(CHECKS),
        newly_computed='Visibility strata on existing fixed predictions; square dual-mode regression; historical150 severity strata',
        new_training_runs=0,optimizer_updates=0,new_inference_runs=0,new_PnP_runs=0,
        raw_annotations_changed=False,frozen_receipts_changed=False,PDF_generated=0,pushes=0,external_uploads=0)
    write('EXECUTION_VALIDATION.json',receipt)
    text=['# 가시성·정사각형 CLI 마감', '',
        '기존 사람 입력3030점과 보호된71점을 합쳐3101개 참조 코너의 가시성 분류를 검산했다. 누락0점이다. 좌표·원본 주석·고정 예측은 변경하지 않았다.', '',
        '| 분류 | 참조 코너 |','|---|---:|']
    text += [f'| {name} | {merged["counts"].get(name,0)} |' for name in CATEGORIES]
    text += ['', '가시성별 표는 프레임마다 전체 물체 대칭을 먼저 선택한 뒤 원래 GT 코너ID로 나눴다. 각 가시성 집단에서 대칭을 다시 선택하지 않았다. DOPE의 매칭 성공 프레임 내부 결측 코너도 PCK 분모와 벌점에 유지하고 조건부 중앙값/P90에서 제외했다.', '',
        '[가시성 결과](VISIBILITY_RESULTS.md) · [seed별 결과](VISIBILITY_PER_SEED.md) · [3101점 출처](STATIC_VISIBILITY_MERGE_AUDIT.json) · [모든 코너 원시 지표](PER_POINT_SCORES.csv)', '',
        '정사각형119장은 manual_declared602점과 manual_in_frame600점을 모든 방법에 각각 동일하게 적용했다. 두 모드를 섞지 않았으며 기존 고정 결과에 대한 전수 회귀검산을 통과했다. N3 대Base와 N3 대N2 비교는 별도 열로 보존했다.', '',
        '[119장 두 모드 결과](SQUARE119_RESULTS.md) · [Base/N2 대비 N3 변화](SQUARE119_COMPARISONS.md) · [119장 지표·분모·제한](SQUARE119_RESULTS.json)', '',
        '119장은 단일 촬영 세션이다. 독립6D 참조가 없으므로 이동·회전 정확도는x이며 물리적6D 정답을 재구성하지 않았다. 모든 영상의 치수가 같아 치수 입력의 인과 효과는 식별할 수 없다. 세션 일반화 신뢰구간은NA로 둔다.', '',
        '기존 정사각형150장은7개 세션의 가림 없음103/중간44/어려움3으로 별도 재집계했다. 과거 고정R0/N0/N2 결과만 재사용했다. 직접 클릭681점과 PnP 포함1200점 proxy를 별도 모드로 유지했으며119장과 합산하지 않았다. 이150장에 대한 동일 계약N3/DOPE/ResNet 결과는x다.', '',
        '[기존150장 별도 결과](HISTORICAL_SQUARE150.md) · [출처와 제한](HISTORICAL_SQUARE150.json)', '',
        f'실행 검산{len(CHECKS)}개PASS, 실제 경과{receipt["elapsed_seconds"]:.3f}초. 새 학습·optimizer update·추론·PnP·PDF·업로드·push는0회다. 새 표의 원고 반영 여부는 별도 원고 작업의cell map에서 확인해야 한다.', '',
        '[실행 명령·입력 SHA-256·회귀검산](EXECUTION_VALIDATION.json)']
    (OUT/'REPORT_KO.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(status='PASS',checks=len(CHECKS),seconds=receipt['elapsed_seconds'],
        visibility_counts=merged['counts'],square_family_rows=len(families),historical_counts=historical['severity_counts']),ensure_ascii=False),flush=True)


if __name__=='__main__': main()
