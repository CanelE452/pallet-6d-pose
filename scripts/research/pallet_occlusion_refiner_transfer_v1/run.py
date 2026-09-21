"""Read-only historical inputs; immutable E1 outputs; fail-closed E2 gate."""
import argparse
import copy
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_posefix_large_error_v1 import evaluate as O
from scripts.research.pallet_posefix_replay_v1 import core as N
from scripts.research.pallet_cad_refiner_comparison_v1 import self_occlusion as S
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as M

NAME = 'pallet_occlusion_refiner_transfer_v1'
DOC = ROOT / '_docs/experiments' / NAME
RAW = ROOT / 'data/pallet/results' / NAME
OUT = ROOT / 'outputs' / NAME
DCP = ROOT / 'data/pallet/results/pallet_dim_conditioned_p_v1'
CAD = ROOT / 'outputs/pallet_cad_refiner_comparison_v1'
read = lambda p: json.loads(Path(p).read_text())


def bind(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return dict(path=str(path.relative_to(ROOT)), sha256=h.hexdigest(), bytes=path.stat().st_size)


def verify(b):
    assert bind(ROOT / b['path'])['sha256'] == b['sha256'], b['path']


def freeze(path, value):
    path = Path(path).resolve()
    assert any(path.is_relative_to(p) for p in (DOC, RAW, OUT))
    data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text() == data, ('Completed artifact overwrite prohibited', str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: f.write(data)


def predictions_only(p):
    return copy.deepcopy({k: p[k] for k in ('candidates', 'selected_index')})


def inputs():
    records = read(ROOT / '_docs/experiments/pallet_type_selftrain_v1/EVAL_PROTOCOL.json')['records']
    return [r for r in records if r['kind'] in ('PLASTIC', 'GREEN')]


def condition_map():
    rows = list(csv.DictReader((ROOT / 'data/evaluation/pallet_eval_v1/manifests/frames.csv').open()))
    # Prefer exact image path; duplicates with conflicting conditions are not silently resolved.
    return rows, {'data/evaluation/pallet_eval_v1/' + r['image_path']: r for r in rows}


def preflight():
    records = inputs()
    old = read(ROOT / '_docs/experiments/pallet_cad8_selftrain_v1/PROTOCOL.json')
    checkpoints = {'R0': old['initialization'], 'POSEFIX_SYNTH': bind(N.C.PRIOR_CK),
                   'REPLAY': read(N.DOC / 'FIT.json')['checkpoint']}
    for arm in ('N2_DIM_ONLY', 'N3_DIM_SYM'):
        checkpoints[arm] = read(ROOT / f'_docs/experiments/pallet_dim_conditioned_p_v1/fits/{arm}_seed1.json')['checkpoint']
    for b in checkpoints.values(): verify(b)
    assert checkpoints['POSEFIX_SYNTH']['sha256'] == read(N.C.PRIOR_DOC / 'PRIOR_SELECTION.json')['checkpoints']['1']
    document_names = ['pallet_clean_pseudo_stac_v1/README.md', 'pallet_cad8_occlusion_v1/RESULTS_KO.md',
        'pallet_posefix_replay_v1/RESULTS_KO.md', 'pallet_posefix_other_audit_v1/REPORT_KO.md',
        'pallet_posefix_green_audit_v1/REPORT_KO.md', 'pallet_n2_coarse_fine_v1/RESULTS_KO.md',
        'pallet_posefix_utility_selector_v1/RESULTS_KO.md']
    code_names = ['pallet_cad_refiner_comparison_v1/self_occlusion.py', 'pallet_posefix_large_error_v1/core.py',
        'pallet_dim_conditioned_p_v1/refiner.py', 'pallet_final_ml_contribution_test_v1/generic_point_refiner.py',
        'pallet_sensors_submission_v1/prior_model.py', 'pallet_posefix_replay_v1/train.py',
        'pallet_posefix_replay_v1/source.py', 'pallet_dim_conditioned_p_v1/eval_math.py']
    sources = [ROOT / '_docs/experiments' / p for p in document_names] + [ROOT / 'scripts/research' / p for p in code_names]
    sources += [ROOT / '_docs/experiments/pallet_type_selftrain_v1/POOL.json',
                ROOT / '_docs/experiments/pallet_type_selftrain_v1/EVAL_PROTOCOL.json',
                ROOT / 'data/evaluation/pallet_eval_v1/manifests/frames.csv',
                N.DOC / 'INPUT_LOCK.json', N.DOC / 'PROTOCOL.json', N.C.DOC / 'TRAIN_SUPPORT.json',
                ROOT / 'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json']
    freeze(DOC / 'INPUT_BINDINGS.json', dict(status='[확인]', checkpoints=checkpoints,
           inputs=[bind(p) for p in sources], code=bind(Path(__file__)),
           branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
           HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
           compared_to='73bfe38a259b3c846e49a98fb82c44578f6e2248',
           related_diff=subprocess.check_output(['git','diff','73bfe38a','--',*[str(p.relative_to(ROOT)) for p in sources]],text=True)))
    rows, conditions = condition_map()
    teacher = read(ROOT / '_docs/experiments/pallet_large_error_refiner_v1/SPLIT.json')['train']
    teacher_sessions = sorted({r['session'] for r in teacher})
    evalids = {r['id'] for r in records}
    populations = {'PLASTIC194': [r for r in records if r['kind']=='PLASTIC'],
                   'GREEN150_MANUAL': [r for r in records if r['kind']=='GREEN'],
                   'DEV72_REFERENCE_UNKNOWN': read(ROOT / '_docs/experiments/pallet_large_error_refiner_v1/SPLIT.json')['evaluation'],
                   'PRIMARY_OCC96': [r for r in records if r['id'] in old['primary_occlusion_ids']],
                   'CLEAN_NONCAD69': [r for r in old['eval_records'] if r['occlusion']=='none'],
                   'CAD18': [r for r in records if r['session']=='eval_cad'], 'SELECTED_CLEAN8': old['train_records']}
    metadata = {}
    for name, rr in populations.items():
        metadata[name] = dict(frames=len(rr), sessions=dict(Counter(r.get('session',r['id'].split(':')[0]) for r in rr)),
                             ids=[r['id'] for r in rr],
                             teacher_session_overlap=sorted({r.get('session',r['id'].split(':')[0]) for r in rr}&set(teacher_sessions)),
                             role='audit/evaluation only in this new experiment',
                             gt_provenance='manual-only scoring' if name=='GREEN150_MANUAL' else 'legacy reference; per-point manual/PnP/unknown counted during scoring',
                             prior_exposure='Reused DEV; selected CAD8 used in earlier student training, not new independent test')
    freeze(DOC / 'DATA_ROLE_MANIFEST.json', dict(status='[확인]', populations=metadata, records=records,
           teacher_manual_train=teacher, teacher_sessions=teacher_sessions, eval_GT_not_training=True,
           condition_fields=['occlusion','truncation','usage_role','session_id','image_sha256'],
           selected8_NOT_automatically_promoted_to_training=True))
    # Audit the existing disjoint candidate pool plus previous raw-RGB scouting, not the eval-selected CAD8.
    pool = read(ROOT / '_docs/experiments/pallet_type_selftrain_v1/POOL.json')['records']
    for name in ('SCOUT.json', 'EXPANDED.json'):
        p = ROOT / 'outputs/pallet_easy_rgb_review_v1' / name
        if p.exists(): pool += read(p)['records']
    unique = {}
    for r in pool: unique.setdefault(r['image']['sha256'], r)
    protected = read(ROOT / '_docs/experiments/pallet_type_selftrain_v1/EVAL_PROTOCOL.json')['records'] + teacher
    protected_hash = {r['image']['sha256'] for r in protected}
    protected_roots = {str(Path(r['image']['path']).parent.parent) for r in protected}
    groups = read(ROOT / 'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json')
    edges = [{s['session_key'] for s in g['sessions']} for g in groups['groups'] if not g['is_collection']]
    edges += [{p['session_a'],p['session_b']} for p in groups['partial_overlap_pairs']]
    while True:
        n = len(protected_roots)
        for edge in edges:
            if edge & protected_roots: protected_roots |= edge
        if n == len(protected_roots): break
    byhash = {}
    for r in rows: byhash.setdefault(r['image_sha256'], []).append(r)
    audit=[]
    for h,r in sorted(unique.items()):
        verify(r['image'])
        found=byhash.get(h,[])
        labels={(x['occlusion'],x['truncation']) for x in found}
        overlap = h in protected_hash or r['session'] in protected_roots
        clean = labels == {('none','none')}
        audit.append(dict(id=r.get('id',h),image=r['image'],session=r['session'],
            verified_conditions=[dict(occlusion=x['occlusion'],truncation=x['truncation'],usage_role=x['usage_role']) for x in found],
            overlap=overlap,verified_clean=clean,eligible=clean and not overlap,
            reason='protected_evaluation_or_teacher_recording' if overlap else 'eligible' if clean else 'no_verified_clean_condition',
            previous_status=r.get('status','POOL_RECORD_WITHOUT_CONDITION_FIELDS')))
    eligible=[r for r in audit if r['eligible']]
    freeze(DOC/'E1_PSEUDOLABEL_POOL.json',dict(status='[확인]',source_rows=len(pool),unique_candidate_images=len(audit),
        eligible_before_inference=len(eligible),reasons=dict(Counter(r['reason'] for r in audit)),records=audit,
        verified_metadata_scope='workspace frames.csv + prior pool/scout/expanded schemas; no unreviewed image assumed clean',
        frozen_teacher_path='R0 -> existing flip/medianLOO/confidence -> frozen Replay -> existing self-occlusion PnP -> frozen corrected all8 medianLOO',
        new_thresholds=False,GT_correctness_selection=False))
    freeze(DOC/'PSEUDOLABEL_MANIFEST.json',dict(status='[확인] NOT_GENERATED' if not eligible else '[확인] REQUIRES_GENERATION',
        records=[],reason='No eligible verified clean image' if not eligible else 'Run frozen pipeline first',
        original_confidence_name='R0 keypoint confidence, not post-correction confidence',eligible_unique=len(eligible)))
    freeze(DOC/'E1_PSEUDOLABEL_POOL.md',f'# E1-C pool audit\n\n[확인] 기존 후보/시각검토 {len(pool)}행 → SHA 중복 제거 {len(audit)}장. 기존 메타데이터로 clean이며 평가·teacher 학습 세션과 분리된 후보 {len(eligible)}장.\n\n[확인] unknown/미검토를 clean으로 간주하지 않았다. CAD8은 이번 지시에서 평가 세션 제외 조건에 걸리므로 자동 재사용하지 않는다. GT 정확도로 프레임을 선택하지 않았다.\n\n[확인] 실제 이유별 수: `{dict(Counter(r["reason"] for r in audit))}`. 8장 미만이면 E2 STOP.\n')
    freeze(DOC/'PREFLIGHT_AUDIT.md',f'''# Preflight audit

[확인] 현재 main/HEAD는 INPUT_BINDINGS에 고정했다. 이전 기준 73bfe38a 관련 tracked diff도 기록했다. 기존 파일 변경/checkout 없음.

[확인] R0/N2/N3/synthetic-only PoseFix/Replay 5개 checkpoint path/SHA를 실제 파일과 대조했다. PoseFix는 PRIOR1 마지막6000step, Replay는300step이다. 기존 Replay protocol은 seed1/TFAdam1e-4/real8+source8/micro2/300update, BN running statistics 고정이다.

[확인] N2/N3는 R0 P3/P4 feature, 초기9점, predicted bbox, validity, input shape, canonical [W,D,H] 기반5차원 context를 받는다. N3는 whole-object symmetry-aware supervision, N2는 해당 supervision 없이 학습했다. 추론에서는 GT branch를 받지 않는다. 기본 local 후보222개, 중심점 보존.

[확인] PoseFix/Replay는 predicted bbox의288×384 RGB crop와 초기9점 Gaussian map/validity를 받는다. RGB evidence를 쓰는 ResNet152 포즈 보정기다. 치수·카메라·GT가 model inference 인자가 아니다. center/invalid points/bbox/score/candidate identity와 원래 R0 confidence를 보존한다.

[확인] 자기 가림 PnP는 예측 pose/등록치수/K로 hidden을 추정하고 visible+confidence≥.5+in-frame6점 이상으로 refit한다. hidden set 변화 시 fallback. 외부 가림 분류기가 아니며 geometry consistency는 영상 정답 보장이 아니다.

[확인] PCK는 whole-object symmetry 한 분기, 전체 유효GT8코너 분모, miss는 이미지 대각선 벌점이다. median/P90은 매칭 관측 코너만; 중심점 제외. canonical GT identity를 맞춰 복구·손상을 센다. offscreen/confidence 처리 차이는 기존 input/target 계약 그대로 유지한다.

[확인] teacher 수동 학습 세션: {teacher_sessions}. 평가 집합/세션/역사적CAD8 학습 노출은 DATA_ROLE_MANIFEST에 구분했다. PRIMARY_OCC96은 해당 teacher 세션 제외. GREEN은 수동 코너만 평가한다.

[확인] clean pool 감사 결과 eligible={len(eligible)}. 현재 조사한 paired 관련 산출물은 method-paired statistics/동일 이미지 모델 비교이며 camera/pallet movement parity가 확인된 physical clean/occlusion pair manifest는 발견하지 못했다.

[추정] 따라서 external physical occlusion recovery를 직접 입증할 수 없다. E1-B는 CAD18 자기 가림 sanity만 수행한다. PnP-derived 가능 GT의 정확도를 독립 측정처럼 주장하지 않는다.
''')
    print('PREFLIGHT',len(audit),'unique',len(eligible),'eligible',flush=True)


def cache():
    """Copy only prediction objects, never GT, before new GPU inference/scoring."""
    records=inputs();bindings=read(DOC/'INPUT_BINDINGS.json')
    for b in bindings['checkpoints'].values():verify(b)
    basepath=ROOT/'data/pallet/results/pallet_type_selftrain_v1/EVAL_PREDICTIONS_R0.json'
    base=read(basepath);verify(base['checkpoint'])
    predictions={'R0':{r['id']:predictions_only(r['prediction']) for r in base['records'] if r['id'] in {x['id'] for x in records}}}
    sources=[basepath]
    for arm,name in [('N2','N2_DIM_ONLY_seed1'),('N3','N3_DIM_SYM_seed1')]:
        path=DCP/f'predictions/REAL_DEV/{name}.json';d=read(path);verify(d['checkpoint']);sources.append(path)
        predictions[arm]={r['id']:predictions_only(r) for r in d['records'] if r['id'] in predictions['R0']}
    for arm,path in [('POSEFIX_SYNTH',N.C.RAW/'EXISTING_PREDICTIONS.json'),('REPLAY',N.RAW/'PREDICTIONS.json')]:
        sources.append(path);d=read(path);predictions[arm]={}
        for dataset in ('DEV72','GREEN150'):
            for r in d[dataset]:
                O.assert_same(predictions['R0'][r['id']],r['predictions']['R0'],'R0 cache parity',atol=1e-4)
                predictions[arm][r['id']]=predictions_only(r['predictions']['POSEFIX_RAW'])
                if dataset=='GREEN150':predictions['N2'][r['id']]=predictions_only(r['predictions']['A_N2'])
    path=CAD/'REPLAY_PREDICTIONS.json';d=read(path);sources.append(path)
    predictions['REPLAY'].update({fid:predictions_only(p) for fid,p in d['predictions']['REPLAY_RAW'].items()})
    # Every available baseline preserves the exact R0 detection contract (coordinates may differ).
    from scripts.research.pallet_posefix_replay_v1.evaluate import assert_preserved
    for arm,pp in predictions.items():
        for fid,p in pp.items():assert_preserved(predictions['R0'][fid],p)
    freeze(RAW/'CACHED_PREDICTIONS.json',dict(predictions=predictions,sources=[bind(p) for p in sources],
        GT_input=False,N3_GREEN='MISSING_PREEXISTING_PREDICTION; not replaced by a different checkpoint',
        records=[dict(id=r['id'],image=r['image'],session=r['session'],kind=r['kind']) for r in records]))
    print('CACHED', {k:len(v) for k,v in predictions.items()},flush=True)


def infer():
    import torch
    from scripts.research.pallet_cad8_selftrain_v1.run import gpu
    from scripts.research.pallet_posefix_replay_v1.evaluate import assert_preserved
    N.setup();gpu();assert torch.cuda.is_available()
    data=read(RAW/'CACHED_PREDICTIONS.json');ck=read(DOC/'INPUT_BINDINGS.json')['checkpoints']
    for arm in ('POSEFIX_SYNTH','REPLAY'):
        dest=RAW/f'NEW_{arm}.json'
        if dest.exists():continue
        verify(ck[arm]);saved=torch.load(ROOT/ck[arm]['path'],map_location='cpu',weights_only=False)
        model=N.C.PoseFixPallet9();model.load_state_dict(saved['model_state_dict']);model=model.cuda().eval().requires_grad_(False);del saved
        pp={};parity=None
        with torch.inference_mode():
            for r in data['records']:
                fid=r['id']
                if fid in data['predictions'][arm] and parity is not None:continue
                verify(r['image']);im=cv2.imread(str(ROOT/r['image']['path']));base=data['predictions']['R0'][fid]
                before=copy.deepcopy(base);q=N.C.predict(model,im,base,None);assert base==before;assert_preserved(base,q)
                if fid in data['predictions'][arm]:
                    O.assert_same(q,data['predictions'][arm][fid],'frozen model reproduction',atol=.01);parity=fid
                else:pp[fid]=q
                if len(pp)%30==0:print('INFER',arm,len(pp),gpu(),flush=True)
        freeze(dest,dict(predictions=pp,checkpoint=ck[arm],GT_input=False,reference_parity_frame=parity,parity_atol_px=.01))
        del model;torch.cuda.empty_cache()
        print('INFER_DONE',arm,len(pp),flush=True)


def freeze_predictions():
    data=read(RAW/'CACHED_PREDICTIONS.json');pp=data['predictions']
    files=[RAW/'CACHED_PREDICTIONS.json']
    for arm in ('POSEFIX_SYNTH','REPLAY'):
        f=RAW/f'NEW_{arm}.json';pp[arm].update(read(f)['predictions']);files.append(f)
    # Reuse existing GT-free PnP outputs only on CAD18 where all input contracts were audited.
    p=CAD/'self_occlusion/PREDICTIONS.json';files.append(p)
    for arm,rows in read(p)['predictions'].items():
        label={'R0':'R0_PNP','N3_DIM_SYM_seed1':'N3_PNP','REPLAY_RAW':'REPLAY_PNP'}[arm]
        pp[label]={fid:predictions_only(v) for fid,v in rows.items()}
    from scripts.research.pallet_posefix_replay_v1.evaluate import assert_preserved
    for arm,rows in pp.items():
        for fid,p in rows.items():assert_preserved(pp['R0'][fid],p)
    freeze(RAW/'FROZEN_PREDICTIONS.json',dict(predictions=pp,records=data['records'],GT_input=False))
    freeze(DOC/'E1_PREDICTIONS_LOCK.json',dict(predictions=bind(RAW/'FROZEN_PREDICTIONS.json'),sources=[bind(f) for f in files],
        checkpoints=read(DOC/'INPUT_BINDINGS.json')['checkpoints'],all_outputs_locked_before_GT_scoring=True))
    print('PREDICTIONS_LOCKED',flush=True)


def best_candidate(rows):
    # A single whole-object candidate maximizes PCK10 numerator, then minimizes mean error.
    # This rule is frozen in code before scoring; diagnostic oracle only.
    return min(rows,key=lambda a:(-sum(x<=10 for x in rows[a].get('errors',[])),rows[a].get('frame_mean_px',float('inf')),a))


def summarize(rows,baseline):
    s=M.summary(rows);s['gross20_count']=sum(x>20 for r in rows for x in r.get('errors',[]))
    s['recovery_damage']=O.recovery_damage(baseline,rows)
    s['recovery_damage_matched_only']=O.recovery_damage(baseline,rows,True)
    return s


def score():
    lock=read(DOC/'E1_PREDICTIONS_LOCK.json');verify(lock['predictions'])
    data=read(RAW/'FROZEN_PREDICTIONS.json');pred=data['predictions'];records=inputs()
    pe,pop=O.population_metadata();targets={item.frame_id:(item,meta) for item,meta in pop}
    groups={r['object_type']:r['permutations'] for r in read(N.C.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']}
    metrics={a:{} for a in pred};truth={};provenance=Counter()
    for r in records:
        verify(r['annotation']);ann=read(ROOT/r['annotation']['path'])
        for point in ann['objects'][0]['keypoint_annotations'][:8]:provenance[(r['kind'],point.get('source','MISSING'))]+=1
        ds='GREEN150' if r['kind']=='GREEN' else 'DEV72'
        gt,box,modes,obj=O.read_targets(ds,r,[480,640],pe,targets)
        valid=modes[0][1];truth[r['id']]=dict(gt=gt.tolist(),valid=valid.tolist())
        for arm,pp in pred.items():
            if r['id'] in pp:metrics[arm][r['id']]=O.score_prediction(pp[r['id']],gt,box,valid,groups[obj],[480,640],r)
    roles=read(DOC/'DATA_ROLE_MANIFEST.json')['populations'];reports={};selections={}
    for name,population in roles.items():
        ids=population['ids'];available=[a for a in pred if set(ids)<=set(pred[a])]
        assert 'R0' in available
        rows={a:[metrics[a][fid] for fid in ids] for a in available}
        selections[name]={fid:best_candidate({a:metrics[a][fid] for a in available}) for fid in ids}
        rows['B_ORACLE']=[dict(metrics[selections[name][fid]][fid],oracle_arm=selections[name][fid]) for fid in ids]
        summaries={a:summarize(rr,rows['R0']) for a,rr in rows.items()}
        best_actual=max(summaries[a]['PCK']['10'] for a in available)
        delta=100*(summaries['B_ORACLE']['PCK']['10']-best_actual)
        reports[name]=dict(status='[확인]',available_candidates=available,missing_candidates=[a for a in pred if a not in available],summary=summaries,
            oracle_gain_vs_best_actual_pp=delta,oracle_gain_vs_R0_pp=100*(summaries['B_ORACLE']['PCK']['10']-summaries['R0']['PCK']['10']),
            headroom='SMALL_HEADROOM' if delta<3 else 'MODERATE_HEADROOM' if delta<8 else 'LARGE_HEADROOM',
            headroom_reference='Best actual whole-object method PCK10 on same full population',
            oracle_rule='frame PCK10 count max; then frame mean min; then lexicographic name. Single whole-object output only; diagnostic GT oracle.')
    freeze(RAW/'E1_FRAME_METRICS.json',dict(metrics=metrics,oracle=selections,truth_for_display_only=truth))
    freeze(DOC/'E1_CANDIDATE_HEADROOM.json',dict(status='[확인]',populations=reports,
        provenance_counts={str(k):v for k,v in provenance.items()},predictions_lock=bind(DOC/'E1_PREDICTIONS_LOCK.json'),
        independent_confirmation=False,GT_only_after_prediction_lock=True,failed_frames_removed=0))
    lines=['# E1-A candidate headroom','', '[확인] 모든 후보 좌표를 고정한 후 GT 채점했다. GREEN은 수동 코너만, 나머지는 기존 reference이다. 실패 프레임은 분모에 남겼다. 학습 없음.', '',
           '[확인] Oracle은 frame별 PCK10 정답 수 최대 → 평균 오차 최소 → 이름 순으로 whole-object 후보 하나만 선택한다. 실사용 가능한 선택기가 아니다. headroom 분류는 동일 집합 최고 실제 후보 대비 PCK10 차이다.','']
    for name,r in reports.items():
        lines += ['## '+name,'',f'[확인] {r["headroom"]}; 최고 실제 후보 대비 oracle +{r["oracle_gain_vs_best_actual_pp"]:.2f}pp, R0 대비 +{r["oracle_gain_vs_R0_pp"]:.2f}pp.', '',
                  '| [확인] 방법 | PCK10 % | PCK20 % | median px | P90 px | >20 점 | 복구 / hard | 손상 / good | 매칭 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for a,s in r['summary'].items():
            d=s['recovery_damage'];lines.append(f'| {a} | {100*s["PCK"]["10"]:.2f} | {100*s["PCK"]["20"]:.2f} | {s["matched_pooled_corner8_median_px"]:.2f} | {s["matched_pooled_corner8_P90_px"]:.2f} | {s["gross20_count"]} | {d["recovered"]}/{d["hard"]} | {d["damaged"]}/{d["good"]} | {s["matched"]}/{s["total_frames"]} |')
        lines += ['', '[확인] 미제공 후보: '+', '.join(r['missing_candidates'])+'. 없는 후보를 R0로 채우지 않았으며, 모든 프레임에 존재하는 후보끼리만 oracle 비교했다.','']
    freeze(DOC/'E1_CANDIDATE_HEADROOM.md','\n'.join(lines)+'\n')
    print('SCORED',{k:(v['headroom'],v['oracle_gain_vs_best_actual_pp']) for k,v in reports.items()},flush=True)


def geometry():
    cv2.setNumThreads(1);tests=S.tests()
    saved=read(CAD/'self_occlusion/PREDICTIONS.json');base=read(CAD/'CACHED_PREDICTIONS.json');base.update(read(CAD/'REPLAY_PREDICTIONS.json')['predictions'])
    poses=read(CAD/'POSE_PREDICTIONS.json');decisions=read(CAD/'filter_audit/DECISIONS.json')['rows']
    print('GEOMETRY_SCHEMA',list(decisions[0]),flush=True)
    # Use the same archived camera extraction as original self-occlusion code.
    meta={r['id']:r for r in decisions};results=[]
    for arm in S.ARMS:
        for fid,candidate_prediction in base[arm].items():
            c=O.top(candidate_prediction);original=np.array(c['keypoints_xy'])
            K=np.array(meta[fid]['K'])
            initial=poses[arm][fid]
            q,info=S.correct(c,initial,K)
            np.testing.assert_allclose(q,O.top(saved['predictions'][arm][fid])['keypoints_xy'],atol=1e-8)
            variations=[]
            for axis in range(3):
                for scale in (.95,1.05):
                    altered=copy.deepcopy(initial)
                    if initial['available']:altered['cf_extents'][axis]*=scale
                    z,detail=S.correct(c,altered,K)
                    variations.append(dict(axis=axis,scale=scale,applied=detail['applied'],reason=detail['reason'],
                        hidden=detail['hidden'],max_output_shift_px=float(np.linalg.norm(z-q,axis=1).max()),
                        visible_fit_mean_px=detail.get('visible_fit_mean_px')))
            used=info['used'];points=original[used] if used else np.empty((0,2))
            spread=float(cv2.contourArea(cv2.convexHull(points.astype(np.float32)))) if len(used)>=3 else 0.
            results.append(dict(id=fid,arm=arm,**info,usable_points=len(used),visible_hull_area_px2=spread,
                sensitivity=variations,baseline_reproduced=True))
    freeze(DOC/'E1_GEOMETRY_RECOVERABILITY.json',dict(status='[확인]',scope='CAD18 self-occlusion sanity only',
        paired_physical_occlusion_GT='UNAVAILABLE_IN_AUDITED_MANIFESTS',GT_used_in_PnP=False,
        GT_caveat='Hidden corner GT may be PnP-derived; no independent physical recovery claim',tests=tests,rows=results,
        failure_categories=dict(Counter(r['reason'] for r in results)),
        dimensions_test='Each camera-facing axis independently ±5%; keep original initial pose for visibility, refit visible prediction points; not GT-axis tuning'))
    freeze(DOC/'E1_GEOMETRY_RECOVERABILITY.md',f'# E1-B geometry sanity\n\n[확인] 検証된 physical clean/occlusion pair가 없어 CAD18×3후보 자기 가림 출력54개만 재현했다. GT-visible external recovery oracle은 미실행이다.\n\n[확인] 실패/적용 이유: `{dict(Counter(r["reason"] for r in results))}`. 각 camera-facing 치수축 ±5%의6변형에 대해 출력 이동, fit residual, hidden set을 기록했다. 사용점 수·convex hull 면적도 기록했다.\n\n[확인] 이상적 cuboid regression: `{tests}`. 이는 실제 팔레트 camera/GT contract의 완전 검증이 아니다.\n\n[추정] PnP 유래 가능 hidden GT와의 일치는 물리적 복구를 독립 입증하지 못한다.\n')
    print('GEOMETRY_DONE',Counter(r['reason'] for r in results),flush=True)


def gate():
    pool=read(DOC/'E1_PSEUDOLABEL_POOL.json');count=pool['eligible_before_inference']
    passed=count>=8 and len(read(DOC/'PSEUDOLABEL_MANIFEST.json')['records'])>=8
    freeze(DOC/'E1_TO_E2_GATE.json',dict(status='[확인]',eligible_unique_clean=count,pass_gate=False,
        blocker='VERIFIED_DISJOINT_CLEAN_POOL_LT8' if count<8 else 'ADDITIONAL_PARITY_AND_PSEUDOLABEL_AUDIT_REQUIRED',
        trained=False,reason='Do not silently promote CAD evaluation or unknown condition frames',
        unexecuted_tests=['E2 clean/occluded target parity','actual occluded R0 re-inference','four-arm exposure/update parity']))
    assert not passed,'Eligible pool available: implement and test full E2 protocol before enabling training'
    for name in ('E2_PROTOCOL.json','E2_INPUT_LOCK.json','E2_FITS.json','E2_REAL_RESULTS.json','E2_SOURCE_RESULTS.json','E2_DECISION.json'):
        freeze(DOC/name,dict(status='[확인] NOT_RUN_E1_GATE_STOP',reason='Verified disjoint clean pseudo-label pool gate not passed',training_updates=0,not_a_FAIL_E2_result=True))
    freeze(DOC/'NEXT_STAGE_PLAN.md','# Next stage\n\n[확인] E2 무결성 진입 gate 미통과로 E2 및 E3~E6 실행하지 않았다. PASS/PARTIAL/FAIL_E2 중 하나를 성능 결과처럼 부여하지 않는다.\n\n[확인] 필요한 사용자 선택: 평가·Replay TRAIN 세션과 분리된 기존 실사에서 최소8장의 clean 조건을 확인해 고정하거나, 별도 지시로 기존CAD 세션의 데이터 역할 변경을 명시적으로 승인해야 한다. 후자는 이번 제외 규칙 변경이며 독립 검증이 아니다. 새 코너 좌표 annotation을 요구하는 것은 아니다.\n\n[추정] 학습 전 pool 부족은 representation 실패의 증거가 아니다. E1 headroom과 clean pool 확인을 분리하여 판단해야 한다.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','cache','infer','freeze','score','geometry','gate']);args=p.parse_args()
    {'preflight':preflight,'cache':cache,'infer':infer,'freeze':freeze_predictions,'score':score,'geometry':geometry,'gate':gate}[args.stage]()
