"""Consume existing human labels and frozen scores into a new closeout namespace.

No image/model inference, PnP, training, source annotation writes or publication.
The native labels are human-entered grades whose criterion remains unconfirmed.
"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from scripts.research.pallet_static_registry_review_20261003_v1 import audit as A
from scripts.research.pallet_n3_completion_v3 import common as C, evaluation as E, metrics as M, square as S

ROOT = A.ROOT
DOC = ROOT/'_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/static'
RAW = ROOT/'data/pallet/results/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/static'
MANIFEST = ROOT/'_docs/experiments/pallet_static_registry_review_20261003_v1/review/STATIC_REVIEW_MANIFEST.json'
NATIVE = ROOT/'data/pallet/results/pallet_static_registry_review_20261003_v1/native_severity_review'
SEMANTICS = ROOT/'data/pallet/results/pallet_combined_closeout_20261003_v1/human_review/SEVERITY_CRITERION_CONFIRMATION.json'
SEEDS = (1, 2, 3)


def binding(path):
    return A.bind(Path(path), root=ROOT)


def verify(entry):
    path = ROOT/entry['path']
    if A.sha256(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
        raise ValueError('Immutable binding mismatch: '+str(path))


def labels_and_audit():
    store_path=NATIVE/'STATIC_SEVERITY_INPUTS.json'
    groups_path=NATIVE/'STATIC_SEVERITY_GROUPS.json'
    manifest=A.read(MANIFEST);store=A.read(store_path);groups=A.read(groups_path)
    cases={c['case_id']:c for c in manifest['cases']}
    records=store['records']
    checks={}
    checks['438_unique_cases']=len(cases)==len(manifest['cases'])==438
    checks['manifest_binding']=store['input_bindings']['static_manifest_sha256']==A.sha256(MANIFEST)
    checks['source_binding_inventory']=store['input_bindings']['source_bindings']==manifest['source_bindings']
    for entry in manifest['source_bindings']:verify(entry)
    checks['original_source_bindings']=True
    for case in cases.values():
        verify(case['image']);verify(case['annotation'])
    checks['438_images_and_annotations_unchanged']=True
    checks['native_record_ids']=set(records)<=set(cases) and all(k==r['frame_id'] for k,r in records.items())
    checks['actual_human_inputs']=all(r['label_source']=='HUMAN_DIRECT_CLASS'
        and r['input_action'] in ('human_keyboard','human_mouse_button')
        and r['human_identity_confirmed'] is False and r['actor_id']==store['actor']['id'] for r in records.values())
    checks['record_identity_image_population']=all(r['image_sha256']==cases[k]['image']['sha256']
        and r['dataset_population']==cases[k]['population'] and r['session_id']==cases[k]['session'] for k,r in records.items())
    checks['prior_provenance_unchanged']=all(r['prior_approved_severity']==cases[k]['frame_severity'] for k,r in records.items())
    checks['labels_valid']=all(r['severity'] in ('clean','moderate','severe') for r in records.values())
    checks['view_timing']=all(datetime.fromisoformat(r['selected_at'].replace('Z','+00:00'))>=
        datetime.fromisoformat(r['started_at'].replace('Z','+00:00')) and r['elapsed_view_seconds']>=0 for r in records.values())
    replay={}
    for event in store['history']:
        if event['action']=='explicit_class_selection':replay[event['frame_id']]=event['current']
        elif event['action']=='human_reset_to_unreviewed':replay.pop(event['frame_id'],None)
        else:raise ValueError('Unexpected native history action')
    checks['history_replay_exact']=replay==records
    effective=[]
    for key,case in cases.items():
        current=records.get(key);prior=case['frame_severity']
        grade=current['severity'] if current else prior['status'] if prior['locked'] else 'unreviewed'
        effective.append(dict(case_id=key,id=case['frame_id'],population=case['population'],
            session=case['session'],material=case['material'],severity=grade,
            source='actual_native_human_selection' if current else 'existing_locked_approved_label',
            native_record_sha256=A.sha256(NATIVE/'severity_annotations'/case['population']/
                (Path(case['image']['path']).stem+'.json')) if current else None,
            image=case['image'],annotation=case['annotation'],prior_approved=prior,
            human_identity_confirmed=False,classification_semantics='NOT_CONFIRMED'))
    checks['all_grades_available']=all(r['severity'] in ('clean','moderate','severe') for r in effective)
    expected={(r['population'],r['severity'],r['case_id']) for r in effective}
    actual=[(p,g,r['frame_id']) for p,gg in groups['groups'].items() for g,rr in gg.items() for r in rr]
    checks['native_group_exact_unique']=len(actual)==len(set(actual))==438 and set(actual)==expected
    for row in effective:
        if row['source']=='actual_native_human_selection':
            p=NATIVE/'severity_annotations'/row['population']/(Path(row['image']['path']).stem+'.json')
            checks['individual_native_records_exact']=checks.get('individual_native_records_exact',True) and A.read(p)['record']==records[row['case_id']]
    counts={p:dict(Counter(r['severity'] for r in effective if r['population']==p)) for p in ('DEV319','GREEN0918')}
    checks['expected_latest_counts']=counts=={'DEV319':{'clean':153,'moderate':92,'severe':74},'GREEN0918':{'clean':3,'moderate':85,'severe':31}}
    if not all(checks.values()):raise ValueError(checks)
    criterion=A.read(SEMANTICS) if SEMANTICS.is_file() else {'criterion':'NOT_CONFIRMED'}
    return effective,dict(schema='pallet_native_static_labels_closeout_v1',status='PASS',checks=checks,
        counts=counts,new_native_records=len(records),reused_approved_labels=438-len(records),
        criterion=criterion,classification_semantics_status='NOT_CONFIRMED' if criterion['criterion']=='NOT_CONFIRMED' else 'USER_CONFIRMED',
        selection_status='USER_ENTERED_GRADES; grade counts preserved; not externally verified occlusion truth',
        sources=[binding(MANIFEST),binding(store_path),binding(groups_path)]+manifest['source_bindings'],rows=effective)


def aggregate_panel(methods, families, labels, checks, prefix, *, pose=True):
    out={}
    for display,names in families.items():
        per_seed={}
        for name in names:
            payload=methods[name]
            corners=A._relabeled(payload['corner_scores'],labels)
            poses=A._relabeled(payload.get('pose_scores',[]),labels)
            if len({r['id'] for r in corners})!=len(corners) or set(r['id'] for r in corners)!=set(labels):
                raise ValueError('Frame ID population mismatch: '+prefix+'/'+name)
            if pose and set(r['id'] for r in poses)!=set(labels):raise ValueError('Pose denominator drift: '+prefix+'/'+name)
            result=A._summaries(corners,poses)
            before=A._headline(payload['corner_scores'],payload.get('pose_scores',[]))
            checks[prefix+'/'+name]=dict(exact_equal=A.clean(before)==A.clean(result['all']),before=before,after=result['all'])
            if not checks[prefix+'/'+name]['exact_equal']:raise ValueError('Overall metric drift')
            per_seed[name]=result
        mean={key:A._mean_dict([per_seed[n][key] for n in names]) for key in A.GROUP_ORDER}
        cross=set.intersection(*(set(per_seed[n]['material_x_severity']) for n in names))
        mean['material_x_severity']={k:A._mean_dict([per_seed[n]['material_x_severity'][k] for n in names]) for k in sorted(cross)}
        out[display]=dict(raw_methods=names,result=mean,per_seed=per_seed,
            aggregation='single frozen prediction path' if len(names)==1 else 'arithmetic mean of per-seed statistics; predictions not averaged')
    return out


def square_corner_headline(rows):
    result=M._corner_summary(rows);den=result['denominators'];met=result['metrics']
    return dict(frames=den['frames'],full_supervised_corners=den['full_supervised_corners'],observed_corners=den['observed_corners'],
        matched_frames=den['matched_frames'],corner_median_px=met['matched_pooled_corner8_median_px'],corner_P90_px=met['matched_pooled_corner8_P90_px'],
        PCK10_fraction=met['full_PCK10_fraction'],translation_median_cm=None,translation_P90_cm=None,rotation_median_deg=None,
        rotation_P90_deg=None,yaw_median_deg=None,yaw_P90_deg=None,IoU3D_median=None,ADDsym_AUC_full=None,
        pose_available_frames=None,pose_failed_frames=None,pose_coverage=None)


def square_groups(rows):
    out={g:square_corner_headline(rows if g=='all' else [r for r in rows if r['occlusion']==g]) for g in A.GROUP_ORDER}
    out['material_x_severity']={'plastic::'+g:out[g] for g in A.GROUP_ORDER[:-1]}
    return out


def aggregate_square(labels, inputs, checks):
    cached_path=C.RAW/'SQUARE_YOLO_METRICS.json';cached=A.read(cached_path)
    old_path=ROOT/'_docs/experiments/pallet_n3_static_closeout_v1/SQUARE_AUDIT.json';old=A.read(old_path)
    inputs.extend([binding(cached_path),binding(old_path),binding(S.SNAPSHOT),binding(S.SYMMETRY)])
    verify(cached['predictions']);verify(cached['snapshot']);verify(cached['symmetry_contract'])
    output={};score_export={}
    for mode in ('manual_declared','manual_in_frame'):
        truth=S.truth_rows(in_frame_only=mode=='manual_in_frame');normalized={};original={}
        output[mode]={'manual_corner_denominator':602 if mode=='manual_declared' else 600,'backbones':{}}
        score_export[mode]={}
        original['yolo']={k:[dict(r,material='plastic',occlusion='unclassified') for r in rows] for k,rows in cached['modes'][mode]['rows'].items()}
        for backbone in ('dope','resnet18'):
            pred_path=C.RAW/'predictions'/f'{backbone}_GREEN0918_119.json';payload=A.read(pred_path)
            if mode=='manual_declared':inputs.append(binding(pred_path))
            preds=E.normalize_prediction_payload(payload)['methods'];rows=[]
            if set(preds)!=set(E.METHODS):raise ValueError('Incomplete frozen square methods')
            for target in truth:
                row=dict(target,predictions={},matched={},detected={})
                for method,records in preds.items():
                    if set(records)!=set(labels):raise ValueError('Square population drift')
                    pred=records[target['id']]
                    if method!='base' and E._preservation_error(preds['base'][target['id']],pred):raise ValueError('N3 detection preservation drift')
                    row['predictions'][method]=pred['points'];row['detected'][method]=pred['detected']
                    row['matched'][method]=bool(pred['detected'] and E._iou(pred['bbox'],target['box'])>=E.MATCH_IOU)
                rows.append(row)
            original[backbone]={method:M.score_corner_rows(rows,method) for method in E.METHODS}
            for method,corner_rows in original[backbone].items():
                headline=E._headline(dict(corner=M._corner_summary(corner_rows)))
                previous=old['results'][backbone][mode][method]
                checks[f'square_old_regression/{backbone}/{mode}/{method}']={
                    'exact_equal':headline==previous,'before':previous,'after':headline}
                if headline!=previous:raise ValueError('Square prior headline drift: '+backbone+'/'+method)
        for backbone,methods in original.items():
            score_export[mode][backbone]={};per_all={}
            families=({'Base':['R0'],'P':[f'OLD_P_seed{s}' for s in SEEDS],
                'N2':[f'N2_DIM_ONLY_seed{s}' for s in SEEDS],'N3':[f'N3_DIM_SYM_seed{s}' for s in SEEDS]} if backbone=='yolo' else
                {'Base':['base'],'N3':[f'n3_seed{s}' for s in SEEDS]})
            for name,rows in methods.items():
                relabeled=A._relabeled(rows,labels)
                for row in relabeled:row['material']='plastic'
                score_export[mode][backbone][name]=relabeled;per_all[name]=square_groups(relabeled)
                before=square_corner_headline(rows);after=per_all[name]['all']
                checks[f'square_label_invariance/{backbone}/{mode}/{name}']=dict(exact_equal=before==after,before=before,after=after)
                if before!=after:raise ValueError('Square grade changes overall metrics')
            for display,names in families.items():
                means={g:A._mean_dict([per_all[n][g] for n in names]) for g in A.GROUP_ORDER}
                means['material_x_severity']={'plastic::'+g:means[g] for g in A.GROUP_ORDER[:-1]}
                output[mode]['backbones'].setdefault(backbone,{})[display]=dict(raw_methods=names,result=means,
                    per_seed={n:per_all[n] for n in names},aggregation='arithmetic mean of per-seed statistics' if len(names)>1 else 'single frozen path')
        for backbone,methods in output[mode]['backbones'].items():
            for result in methods.values():
                if result['result']['all']['full_supervised_corners']!=output[mode]['manual_corner_denominator']:raise ValueError('Square denominator drift')
    return dict(schema='pallet_native_square_grade_reaggregation_v1',population='GREEN0918_119',frames=119,sessions=1,
        label_counts=dict(Counter(labels.values())),modes=output,classification_semantics_status='NOT_CONFIRMED',
        physical_pose_reference=False,pose_status='x: no independent canonical 6D reference',generalization_CI='NA: one capture session',
        contrast_definitions={'N3_minus_Base':'whole trained correction package','N3_minus_N2':'added symmetry under the same dimension-input family'},
        new_training=0,new_inference=0,new_PnP=0),score_export


def emit_csv(payload, path, *, square=False):
    columns=['population','mode','backbone','method','seed_statistic','group']
    metrics=['frames','full_supervised_corners','observed_corners','matched_frames','corner_median_px','corner_P90_px','PCK10_fraction',
        'translation_median_cm','translation_P90_cm','rotation_median_deg','rotation_P90_deg','yaw_median_deg','yaw_P90_deg',
        'IoU3D_median','ADDsym_AUC_full','pose_available_frames','pose_failed_frames','pose_coverage']
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(columns+metrics)
        panels=payload['modes'] if square else {'locked_dev319':payload}
        for mode,panel in panels.items():
            for backbone,methods in panel['backbones'].items():
                for method,p in methods.items():
                    stats={'mean_or_single':p['result'],**p.get('per_seed',{})}
                    for seed,groups in stats.items():
                        flat={g:groups[g] for g in A.GROUP_ORDER};flat.update(groups['material_x_severity'])
                        for group,headline in flat.items():
                            writer.writerow([payload['population'],mode,backbone,method,seed,group]+[headline.get(m) for m in metrics])


def main():
    started=time.monotonic();DOC.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    effective,label_audit=labels_and_audit()
    labels={r['id']:r['severity'] for r in effective if r['population']=='DEV319'}
    square_labels={r['id']:r['severity'] for r in effective if r['population']=='GREEN0918'}
    artifact,base_checks=A.aggregate_backbones(ROOT,labels)
    checks={f'dev319/{backbone}/{method}':v for backbone,mm in base_checks['checks'].items() for method,v in mm.items()}
    yolo_path=ROOT/A.YOLO_SCORES_REL;yolo=A.read(yolo_path)
    extra=aggregate_panel(yolo,{'N0':[f'N0_BASE_REPLAY_seed{s}' for s in SEEDS],
        'N1':[f'N1_SYM_ONLY_seed{s}' for s in SEEDS]},labels,checks,'dev319/yolo')
    artifact['backbones']['yolo'].update(extra)
    artifact['schema']='pallet_native_static_grade_reaggregation_v1'
    artifact['classification_semantics_status']=label_audit['classification_semantics_status']
    artifact['label_source']=binding(NATIVE/'STATIC_SEVERITY_INPUTS.json')
    artifact['source_note']='latest native user-entered grades, existing approved labels reused; original external-occlusion panel remains separate'
    reused_path=C.RAW/'reuse/PER_FRAME_SCORES.json';reused=A.read(reused_path)
    inputs=list(artifact['inputs'].values())+label_audit['sources']+[binding(reused_path)]
    comparator=aggregate_panel(reused['comparators'],{f:[f'H_{f}_seed{s}' for s in SEEDS] for f in ('D','L','PoseFix')},labels,checks,'comparators319')
    comparator_artifact=dict(schema=artifact['schema'],population='DEV319',label_counts=artifact['label_counts'],backbones={'yolo':comparator},
        classification_semantics_status=label_audit['classification_semantics_status'],budget_equivalence='not established; previous fixed predictions reused')
    student_ids={r['id'] for r in reused['students_safe128']['R0']['corner_scores']}
    student_labels={k:labels[k] for k in student_ids}
    student=aggregate_panel(reused['students_safe128'],{'Base':['R0'],'source_only_update':['SYN_LR5'],
        'raw_pseudo_student':['RAW_LR5'],'corrected_pseudo_student':['REF_LR5'],'N3':[f'N3_DIM_SYM_seed{s}' for s in SEEDS]},student_labels,checks,'students128')
    student_artifact=dict(schema=artifact['schema'],population='SAFE_COMMON128',frames=128,label_counts=dict(Counter(student_labels.values())),
        backbones={'yolo':student},classification_semantics_status=label_audit['classification_semantics_status'],
        denominator_note='All methods, including Base and N3, use the same safe non-exposed128; not mixed with DEV319.',frame_ids=sorted(student_ids))
    square_artifact,square_scores=aggregate_square(square_labels,inputs,checks)
    old_static=ROOT/'_docs/experiments/pallet_combined_closeout_20261003_v1/static/STATIC_REAGGREGATION.json'
    previous=A.read(old_static)
    for entry in previous['inputs'].values():verify(entry)
    for backbone,methods in previous['backbones'].items():
        for method,p in methods.items():
            now=artifact['backbones'][backbone][method]['result']['all']
            old=p['result']['all'];checks[f'previous319/{backbone}/{method}']=dict(exact_equal=old==now,before=old,after=now)
            if old!=now:raise ValueError('Prior combined overall319 regression drift')
    inputs.append(binding(old_static))
    unique={x['path']:x for x in inputs}
    protected=list(unique.values());protected_before={x['path']:x['sha256'] for x in protected}
    # Store new derivations only, including raw score caches needed by visibility analysis.
    for name,payload in [('STATIC_REAGGREGATION.json',artifact),('COMPARATOR_REAGGREGATION.json',comparator_artifact),
            ('STUDENT128_REAGGREGATION.json',student_artifact),('SQUARE_REAGGREGATION.json',square_artifact),
            ('LABEL_PROVENANCE_AUDIT.json',label_audit)]:
        A.write_json(DOC/name,payload)
    A.write_json(RAW/'SQUARE_PER_FRAME_SCORES.json',square_scores)
    A.write_json(RAW/'EFFECTIVE_FRAME_LABELS.json',label_audit)
    emit_csv(artifact,DOC/'DEV319_HEADLINE_AND_SEED.csv')
    emit_csv(comparator_artifact,DOC/'COMPARATOR319_HEADLINE_AND_SEED.csv')
    emit_csv(student_artifact,DOC/'STUDENT128_HEADLINE_AND_SEED.csv')
    emit_csv(square_artifact,DOC/'SQUARE119_HEADLINE_AND_SEED.csv',square=True)
    input_unchanged=all(A.sha256(ROOT/p)==h for p,h in protected_before.items())
    if not input_unchanged or not all(x['exact_equal'] for x in checks.values()):raise ValueError('Regression validation failed')
    invariant=dict(status='PASS',all_exact=True,checks=checks,check_count=len(checks),source_inputs_unchanged=input_unchanged,
        original_images_annotations_verified=438,frozen_score_input_bindings_verified=True,
        new_training=0,optimizer_updates=0,new_inference=0,new_PnP=0)
    A.write_json(DOC/'STATIC_INVARIANCE_CHECK.json',invariant)
    sources=dict(schema='native_static_closeout_bindings_v1',inputs=protected,
        metric_code=[binding(A.__file__),binding(M.__file__),binding(E.__file__),binding(S.__file__),binding(__file__)],
        raw_derivations=[binding(RAW/'SQUARE_PER_FRAME_SCORES.json'),binding(RAW/'EFFECTIVE_FRAME_LABELS.json')])
    A.write_json(DOC/'SOURCE_BINDINGS.json',sources)
    elapsed=time.monotonic()-started
    execution=dict(status='PASS',executed_at=datetime.now(timezone.utc).isoformat(),wall_seconds=elapsed,
        command='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_static_registry_review_20261003_v1.reaggregate_native_closeout_20261006',
        outputs=[],new_training=0,optimizer_updates=0,new_inference=0,new_PnP=0,
        source_inputs_unchanged=input_unchanged,checks=len(checks),paper_insertion_status='NOT_INSERTED_BY_THIS_TASK')
    lines=['# 최신 사람 입력 등급 재집계','',
        '**실제 재집계·검산 완료.** 직사각형 153/92/74, 정사각형 3/85/31의 사람 입력 등급을 사용했다. 분류 기준은 미확인으로 유지하며 확정 외부 가림 판정으로 승격하지 않는다.','',
        f'- 전체 및 과거 결과 회귀검산 {len(checks)}/{len(checks)} PASS; 원본 438장 이미지·참조·출처 해시 유지.',
        '- YOLO Base/P/N0/N1/N2/N3 및 DOPE/ResNet Base/N3, 각 3seed 통계와 재질×등급을 재집계했다.',
        '- D/L/PoseFix는 같은319장, 학생 대안은 Base/N3까지 같은128장 별도 패널이다.',
        '- 정사각형 두 모드는 각각602/600점이며 모든 방법에 같은 분모를 썼다. 독립6D참조가 없어 T/R는 x, 한 세션 일반화 CI는 NA다.',
        f'- 새 학습·optimizer update·모델 추론·PnP 0회. 실행 wall {elapsed:.3f}초.',
        '- 이 작업에서는 원고를 수정하지 않았다. 별도 원고 작업이 결과를 삽입한다.','',
        '[319장](STATIC_REAGGREGATION.json) · [정사각형](SQUARE_REAGGREGATION.json) · [비교군](COMPARATOR_REAGGREGATION.json) · [학생128](STUDENT128_REAGGREGATION.json)',
        '[검산](STATIC_INVARIANCE_CHECK.json) · [사람 입력 출처](LABEL_PROVENANCE_AUDIT.json) · [해시 연결](SOURCE_BINDINGS.json)','',
        '```bash',execution['command'],'```','']
    (DOC/'REPORT_KO.md').write_text('\n'.join(lines),encoding='utf-8')
    execution['outputs']=[binding(p) for p in sorted(DOC.iterdir()) if p.is_file() and p.name!='EXECUTION_RECEIPT.json']
    A.write_json(DOC/'EXECUTION_RECEIPT.json',execution)
    print(json.dumps(dict(status='PASS',counts=label_audit['counts'],checks=len(checks),wall_seconds=elapsed,
        doc=str(DOC),raw=str(RAW)),ensure_ascii=False))


if __name__=='__main__':main()
