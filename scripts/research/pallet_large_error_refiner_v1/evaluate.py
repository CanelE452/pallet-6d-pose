"""Paired untouched recording evaluation: raw results and GT-free fallback gates."""
import copy
import math
import time
import cv2
import numpy as np
import torch
from scripts.research.pallet_large_error_refiner_v1 import run as R
from scripts.research.pallet_large_error_refiner_v1.model import WideRefiner, corrupt
from inference import load_head, predict_captured, preservation, serial
from point_inference import replace_selected
from dev_evaluate import population_metadata, iou
from eval_math import measure, summary
from pseudo_label_filters import geometry_scores

NAMES = ('R0', 'A_N2', 'B_CAP32', *R.ARMS)
PERM_FLIP = np.array([1, 0, 3, 2, 5, 4, 7, 6, 8])


def recovery_damage(base, new):
    assert [r['id'] for r in base] == [r['id'] for r in new]
    a, b = [], []
    for x, y in zip(base, new):
        if not x['evaluable']: continue
        assert y['evaluable'] and x['canonical_valid'] == y['canonical_valid']
        for i, v in enumerate(x['canonical_valid']):
            if v: a.append(x['canonical_errors'][i]); b.append(y['canonical_errors'][i])
    a, b = np.asarray(a), np.asarray(b)
    hard, good = a > 20, a < 5
    recovered = int((hard & (b <= 10)).sum()); damaged = int((good & (b > 10)).sum())
    return dict(hard=int(hard.sum()), recovered=recovered, recovery_rate=recovered/int(hard.sum()) if hard.any() else None,
        good=int(good.sum()), damaged=damaged, damage_rate=damaged/int(good.sum()) if good.any() else None)


def top(p):
    idx = p['selected_index']
    return None if idx is None else p['candidates'][idx]


def gate(p, flip, width, dims, K, lock):
    a, b = top(p), top(flip)
    if a is None or b is None: return dict(accepted=False, reason='missing detection')
    q = np.asarray(a['keypoints_xy']); valid = (np.asarray(a['keypoints_conf']) >= lock['keypoint_validity']['kp_conf_threshold']) & np.isfinite(q).all(-1)
    if a['score'] < lock['TAU_BOX'] or valid[:8].sum() < lock['keypoint_validity']['min_valid_corners']:
        return dict(accepted=False, reason='confidence/valid corners')
    fq = np.array(b['keypoints_xy']); fq[:, 0] = width - 1 - fq[:, 0]; fq = fq[PERM_FLIP]
    fv = (np.asarray(b['keypoints_conf'])[PERM_FLIP] >= lock['keypoint_validity']['kp_conf_threshold']) & np.isfinite(fq).all(-1)
    try:
        s = geometry_scores(q, valid, np.asarray(K), dict(x=dims[0], y=dims[2], z=dims[1]), fq, fv)
        scores = {k: float(s[k]) if s[k] is not None and np.isfinite(s[k]) else None for k in ('s_reproj','s_remove','s_flip')}
        threshold = lock['geometry_thresholds']
        accepted = all(scores[k] is not None and scores[k] <= threshold[t] for k, t in [('s_remove','tau_remove'),('s_flip','tau_flip')])
        return dict(accepted=accepted, reason='accepted' if accepted else 'geometry/flip', **scores)
    except cv2.error as exc:
        return dict(accepted=False, reason='PnP error', error=str(exc))


def load_heads():
    heads = {}
    for name in R.ARMS:
        fit = R.E.read(R.DOC / f'FIT_{name}.json'); R.F.verify(fit['checkpoint'])
        ck = torch.load(R.ROOT/fit['checkpoint']['path'], map_location='cpu', weights_only=False)
        assert ck['complete'] and ck['step'] == R.STEPS
        assert ck['protocol_sha'] == R.E.sha(R.DOC/'PROTOCOL.json')
        head = WideRefiner().cuda().eval(); head.load_state_dict(ck['state']); head.requires_grad_(False)
        heads[name] = head
    return heads


@torch.no_grad()
def synthetic_stress(heads):
    from data import PaperData
    data = PaperData(); rows = np.load(R.RAW/'TRAIN_ORDERS.npz')['held']
    rng = torch.Generator(device='cuda').manual_seed(219)
    results = {mode: {name: [] for name in ('R0', *heads)} for mode in ('clean','stress')}
    changed_results = {name: [] for name in ('R0', *heads)}
    for start in range(0, len(rows), 16):
        clean = data.batch(rows[start:start+16], 'N2_DIM_ONLY')
        stress = corrupt(clean, rng)
        changed = (stress['points'][:, :8] != clean['points'][:, :8]).any(-1)
        for mode, b in [('clean',clean),('stress',stress)]:
            mask = b['gt_valid'][:, :8].bool() & b['point_valid'][:, :8].bool()
            for name in ('R0', *heads):
                q = b['points'] if name == 'R0' else heads[name](b)
                err = (q[:, :8] - b['gt_points'][:, :8]).norm(dim=-1)
                results[mode][name].extend(err[mask].cpu().tolist())
                if mode == 'stress': changed_results[name].extend(err[mask & changed].cpu().tolist())
    summarize = lambda v: dict(corners=len(v), mean_px=float(np.mean(v)), median_px=float(np.median(v)), P90_px=float(np.quantile(v,.9)), PCK10=float((np.array(v)<=10).mean()))
    payload = dict(units='network input pixels, not original real-image pixels',
        populations={mode: {k:summarize(v) for k,v in rows.items()} for mode,rows in results.items()},
        perturbed_corners_only={k:summarize(v) for k,v in changed_results.items()}, heldout_rows=256)
    R.freeze(R.DOC/'SYNTHETIC_STRESS.json', payload)


@torch.no_grad()
def evaluate():
    R.verify(); R.E.gpu(); begin = time.monotonic()
    settings = R.F.checked_lock(); lock = R.E.read(R.FILTER)
    norm = R.E.read(R.E.DOC/'DIM_NORMALIZATION_LOCK.json')
    groups = {r['object_type']:r for r in R.E.read(R.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']}
    original, _ = load_head('N2_DIM_ONLY', 1); heads = load_heads()
    synthetic_stress(heads)
    extractor = R.E.old('features').FrozenYoloFeatures(R.E.R0)
    preserves = 0

    def predict(cap, dims, order, hw):
        nonlocal preserves
        out = {'R0':dict(candidates=serial(cap['candidates']), selected_index=cap['selected_index'])}
        for name, fraction in [('A_N2',.01),('B_CAP32',.04)]:
            rule = dict(settings['decode_rule'], max_move_image_diagonal_fraction=fraction)
            out[name], _ = predict_captured(original, 'N2_DIM_ONLY', cap, dims, order,
                settings['temperatures']['N2_DIM_ONLY_seed1'], rule, hw, norm)
            preserves += 1
        batch, inputs = R.input_batch(cap, dims, order)
        for name, head in heads.items():
            after = cap['candidates'] if batch is None else replace_selected(cap['candidates'],cap['selected_index'],
                inputs['points'],head(batch)[0].cpu().numpy(),inputs['gain'],1.)
            preservation(cap['candidates'],after,cap['selected_index']); preserves += 1
            out[name] = dict(candidates=serial(after), selected_index=cap['selected_index'])
        return out

    all_scores = {ds:{mode:{name:[] for name in NAMES} for mode in ('ungated','gated')}
                  for ds in ('DEV72','GREEN150_MANUAL','GREEN150_ALL_KNOWN_PROXY')}
    predictions = {'DEV72':[], 'GREEN150':[]}; all_gates = {'DEV72':[], 'GREEN150':[]}
    pe, pop = population_metadata(); by_id = {item.frame_id:(item,meta) for item,meta in pop}
    records = [('DEV72',r) for r in R.E.read(R.DOC/'SPLIT.json')['evaluation']]
    records += [('GREEN150',r) for r in R.E.read(R.GREEN)['records']]
    olddev = {r['id']:r for r in R.E.read(R.E.RAW/'predictions/REAL_DEV/N2_DIM_ONLY_seed1.json')['records']}
    oldgreen = {r['id']:r for r in R.E.read(R.F.RAW/'green150_saved_labels_v1/PREDICTIONS.json')['predictions']['N2_DIM_ONLY_seed1']}
    try:
        for i, (dataset,r) in enumerate(records):
            R.F.verify(r['image']); R.F.verify(r['annotation']); R.F.verify(r['camera'])
            im = cv2.imread(str(R.ROOT/r['image']['path'])); assert im is not None
            if dataset == 'DEV72':
                R.F.verify(r['cache'])
                cached = torch.load(R.ROOT/r['cache']['path'], map_location='cpu', weights_only=False)
                cap = cached['captured']
                for k in ('p3','p4'): cap[k] = cap[k].cuda()
                dims, order = cached['dimensions'], cached['order']
                obj = r['object_type']; K = r['K']; old = olddev[r['id']]
            else:
                cap = extractor.predict(im); dims, order = r['canonical_WDH_m'], 4
                obj = 'plastic_standard_110x110x15'; K = r['source_K']; old = oldgreen[r['id']]
            preds = predict(cap, dims, order, im.shape[:2])
            if top(preds['A_N2']) is not None:
                assert np.allclose(top(preds['A_N2'])['keypoints_xy'], top(old)['keypoints_xy'], atol=1e-5, rtol=0), r['id']
            flipcap = extractor.predict(np.ascontiguousarray(im[:,::-1]))
            flips = predict(flipcap, dims, order, im.shape[:2])
            gates = {name:gate(preds[name],flips[name],im.shape[1],dims,K,lock) for name in NAMES}
            # Frame population is unchanged: fail -> frozen N2, never omit.
            gated = {name:preds[name] if gates[name]['accepted'] else preds['A_N2'] for name in NAMES}
            all_gates[dataset].append(dict(id=r['id'], gates=gates))
            predictions[dataset].append(dict(id=r['id'],image=r['image'],session=r['session'],raw_hw=list(im.shape[:2]),
                ungated=preds,gated=gated))
            # Targets are read only AFTER all normal/flip predictions and filter choices are fixed.
            if dataset == 'DEV72':
                item,meta = by_id[r['id']]; t = pe.E._legacy_forbidden_target(item)
                gt, box = np.asarray(t.keypoints_xy), np.asarray(t.box_xyxy)
                modes = [('DEV72',np.asarray(t.keypoint_supervision_mask))]
            else:
                doc = R.E.read(R.ROOT/r['annotation']['path'])
                gt, valid = R.annotation_arrays(doc); _, manual = R.annotation_arrays(doc,True)
                h,w = im.shape[:2]; inside = valid & (gt[:,0]>=0) & (gt[:,0]<w) & (gt[:,1]>=0) & (gt[:,1]<h)
                box = np.r_[gt[inside].min(0),gt[inside].max(0)]
                modes = [('GREEN150_MANUAL',manual),('GREEN150_ALL_KNOWN_PROXY',valid)]
            for mode,valid in modes:
                for policy,pp in [('ungated',preds),('gated',gated)]:
                    for name,p in pp.items():
                        c = top(p); matched = c is not None and iou(c['box_xyxy'],box)>=.5
                        points = np.full((9,2),np.nan) if c is None else c['keypoints_xy']
                        metric = measure(points,gt,valid,groups[obj]['permutations'],im.shape[:2],matched,c is not None)
                        all_scores[mode][policy][name].append(dict(id=r['id'],session=r['session'],**metric))
            if (i+1)%10 == 0: R.status('EVALUATING', done=i+1,total=len(records),elapsed_seconds=time.monotonic()-begin,gpu=R.E.gpu())
    finally: extractor.close()
    results = {}
    for dataset, policies in all_scores.items():
        results[dataset] = {}
        for policy, scores in policies.items():
            # Fixed raw-R0 denominator even when evaluating gated outputs.
            reference = policies['ungated']['R0']
            results[dataset][policy] = {name:dict(**summary(rows), **recovery_damage(reference,rows)) for name,rows in scores.items()}
    verdicts = {}
    for policy in ('ungated','gated'):
        verdicts[policy] = {}
        for name in ('B_CAP32',*R.ARMS):
            checks = {}
            for dataset in ('DEV72','GREEN150_MANUAL'):
                a,b = results[dataset][policy][name],results[dataset][policy]['A_N2']
                estimable = a['recovery_rate'] is not None and a['damage_rate'] is not None
                checks[dataset] = dict(estimable=estimable,
                    delta_recovery_pp=None if not estimable else 100*(a['recovery_rate']-b['recovery_rate']),
                    delta_damage_pp=None if not estimable else 100*(a['damage_rate']-b['damage_rate']),
                    delta_PCK10_pp=100*(a['PCK']['10']-b['PCK']['10']))
                c = checks[dataset]
                c['pass'] = bool(estimable and c['delta_recovery_pp']>=10 and c['delta_damage_pp']<=.5 and c['delta_PCK10_pp']>=-.5)
            verdicts[policy][name] = dict(pass_all=all(c['pass'] for c in checks.values()),checks=checks)
    acceptance = {ds:{name:dict(accepted=sum(r['gates'][name]['accepted'] for r in rows),total=len(rows),
        rate=sum(r['gates'][name]['accepted'] for r in rows)/len(rows)) for name in NAMES} for ds,rows in all_gates.items()}
    R.freeze(R.RAW/'PREDICTIONS.json',predictions)
    R.freeze(R.RAW/'PER_FRAME_METRICS.json',all_scores)
    R.freeze(R.RAW/'GATES.json',all_gates)
    R.freeze(R.DOC/'RESULTS.json',dict(complete=True,results=results,verdicts=verdicts,acceptance=acceptance,
        all_preservation_checks=preserves,original_N2_prediction_parity=True,elapsed_seconds=time.monotonic()-begin,
        independent_test=False,auto_promoted=False,GT_input=False,negative_FP='Not rerun: R0 boxes/scores/candidates unchanged',
        verdict='POSITIVE_SCREEN_REQUIRES_REPLICATION' if any(v['pass_all'] for p in verdicts.values() for v in p.values()) else 'NO_PRE_REGISTERED_PASS'))
    report(); R.F.checked_lock()
    R.status('COMPLETE', verdict=R.E.read(R.DOC/'RESULTS.json')['verdict'],gpu=R.E.gpu())


def report():
    result = R.E.read(R.DOC/'RESULTS.json'); split = R.E.read(R.DOC/'SPLIT.json')
    lines = ['# 큰 오차 보정 — 단일 seed bounded screen', '',
        f"판정: {result['verdict']}. 원본 최종 모델/논문 표/라벨은 변경하지 않음.", '',
        f"실사 후보 24장 중 출처 확인된 {split['train_frames']}장/{split['manual_corners']}개 수동 클릭 코너만 E 학습에 사용. PnP/출처 미확인/화면 밖 코너 제외.", '',
        '후보 24장 전체 촬영 그룹을 평가에서 제외한 DEV72 + 학습 제외 GREEN150. 재사용 개발 데이터이며 독립 최종 검증 아님.', '',
        'C/D/E 동일 초기값, seed1, 2,000 step, batch16, 마지막 checkpoint만 평가. A 기존 N2, B 이동 상한만 8→32px(640×480 기준). 새 모델은 bbox 대각선 20%까지 공동 보정.', '',
        'recovery: 같은 R0 >20px 코너가 <=10px로 복구. damage: 같은 R0 <5px 코너가 >10px로 악화. 허용된 전체 물체 대칭만 사용하며 코너별 임의 GT 대응은 허용하지 않음.', '']
    for dataset, policies in result['results'].items():
        for policy, rows in policies.items():
            lines += [f'## {dataset} / {policy}', '', '| 모델 | median px | P90 px | PCK10 % | 복구 / hard | 훼손 / good |', '|---|---:|---:|---:|---:|---:|']
            for name,s in rows.items():
                lines.append(f"| {name} | {s['matched_pooled_corner8_median_px']:.3f} | {s['matched_pooled_corner8_P90_px']:.3f} | {100*s['PCK']['10']:.3f} | {s['recovered']}/{s['hard']} | {s['damaged']}/{s['good']} |")
            lines.append('')
    lines += ['## 필터 수용률', '', '| 데이터 | 모델 | accepted / 전체 |', '|---|---|---:|']
    for dataset,arms in result['acceptance'].items():
        for name,r in arms.items(): lines.append(f"| {dataset} | {name} | {r['accepted']}/{r['total']} |")
    lines += ['', '## 해석 제한', '',
        '- 필터 실패 시 기존 A로 복귀하며 평가 이미지는 버리지 않음. gated R0도 실패 시 A인 혼합 시스템이므로 원래 R0는 ungated 행임.',
        '- 정상 코너를 보호하면서 큰 오차를 복구한다는 사전 기준을 두 실사 집단 모두에서 만족해야 positive screen. 단일 예제 개선만으로 통과시키지 않음.',
        '- 실사 supervision은 소수 촬영의 9장뿐이며 E의 일반화 실패가 모든 실사 학습 가능성의 부정을 의미하지 않음.',
        '- DEV 라벨은 기존 camera-facing 2D/부분 출처 미확인 제약, GREEN manual-only는 직접 클릭점; all-known은 PnP 포함 proxy.',
        '- 두 데이터 모두 과거 연구에서 재사용됨. GREEN 관련 촬영의 역사적 개발 노출도 있으므로 독립 확인·논문 최종 성능 주장 불가.',
        '- 코너 역할 뒤바뀜은 이번 큰 위치 오차 학습 범위 밖. 예제의 큰 고정-index 오차를 모두 위치 오차라고 해석하지 않음.',
        '- confidence/geometry/flip은 정확성 보증이 아님. 같은 잘못된 예측이 두 방향에서 일관될 수 있음.',
        '- C 대비 D가 큰 입력 오류 합성의 효과, D 대비 E는 합성 절반을 직접 실사 supervision으로 교체한 효과. A 대비 C는 구조/학습량 등이 함께 달라지는 시스템 비교.',
        '- R0 검출 박스/score/후보 수/centroid 보존. negative 오검출 개선 실험이 아니며 AP 향상 주장하지 않음.', '']
    (R.DOC/'RESULTS_KO.md').write_text('\n'.join(lines))
