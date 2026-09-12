"""Read-only independent replay audits and correction report generation."""
import argparse
import ast
import csv
import subprocess

import numpy as np
import torch

from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.util import immutable_json, read_json, sha256
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.data import ObservationDataset
from scripts.research.pallet_symmetry_dht_local_v2.tests.test_wls_correction import numpy_reference
from . import geometry
from .correction import REPO, PACKAGE, OLD, RAW, DOC, EXPORT, NAMES


def replay_audit():
    path = 'scripts/research/pallet_symmetry_dht_local_v2/symdht_local_v2/geometry.py'
    source = subprocess.check_output(['git','show',f'187589a:{path}'],cwd=REPO,text=True)
    tree = ast.parse(source)
    ns = {'torch':torch,'Tensor':torch.Tensor,'EDGES':geometry.EDGES,'clip_vector_norm':geometry.clip_vector_norm}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='single_mode_wls']
    exec(compile(ast.Module(body=body,type_ignores=[]),'<pinned buggy v2>', 'exec'),ns)
    data = list(ObservationDataset(EXPORT/'synth_val.json',targets=False))
    reports = []
    for name in NAMES[1:]:
        old = read_json(OLD/f'predictions/{name}_synth_val.json')
        fixed = read_json(RAW/f'posthoc/predictions/{name}_synth_val.json')
        original = {r['frame_id']:r for r in old['records']}; repaired = {r['frame_id']:r for r in fixed['records']}
        old_error=ref_error=0.; exact_fields=True; contracts=True
        for item in data:
            before = original[item['frame_id']]; after = repaired[item['frame_id']]
            exact_fields &= all(before[k]==after[k] for k in ('raw_line','utility','point_valid','anchor_index','mode_mass','ambiguity'))
            args = (item['base_points'][None],item['point_valid'][None],item['point_sigma'][None],torch.tensor(before['raw_line'])[None],torch.tensor(before['utility'])[None], .005*item['image_hw'].norm()[None],item['image_hw'][None])
            buggy, _ = ns['single_mode_wls'](*args)
            old_error = max(old_error,float(np.max(np.abs(buggy[0].numpy()-np.asarray(before['points'])))))
            arrays = tuple(x.numpy().astype(bool if x.dtype==torch.bool else np.float64) for x in args)
            reference = numpy_reference(*arrays)[0]
            ref_error = max(ref_error,float(np.max(np.abs(reference-np.asarray(after['points'])))))
            q=np.asarray(after['points']); delta=np.asarray(after['correction'])
            contracts &= np.array_equal(q[8],item['base_points'][8].numpy())
            contracts &= bool(np.linalg.norm(delta[:8],axis=-1).max() <= .01*float(item['image_hw'].norm())+1e-4)
        # Original GPU solve / CPU replay and raw-pixel float32 rounding differ
        # slightly. Tolerance is 0.001px, far below the 0.25px utility threshold.
        row={'name':name,'unchanged_saved_fields':exact_fields,'center_and_cap':contracts,
             'buggy_CPU_replay_vs_saved_GPU_max_px':old_error,'fixed_CPU_vs_independent_float64_lstsq_max_px':ref_error,
             'tolerance_px':.001,'PASS':exact_fields and contracts and old_error<.001 and ref_error<.001}
        reports.append(row)
    result={'scope':'all 6x512 saved predictions; original function from pinned Git AST and independent NumPy absolute-coordinate least squares',
            'neural_forward_count':0,'optimizer_updates':0,'GT_opened':False,'reports':reports,'PASS':all(r['PASS'] for r in reports)}
    immutable_json(DOC/'OFFLINE_REFERENCE_AUDIT.json',result)
    print(result,flush=True)
    assert result['PASS']


def report():
    result=read_json(DOC/'RESULT_SUMMARY.json'); replay=read_json(DOC/'OFFLINE_WLS_REPLAY.json')
    oracle=read_json(DOC/'ORACLE_CONTROLLED.json')['summary']
    point=result['point']['primary_frame_mean_over_raw_diagonal']
    columns=('primary_frame_mean_over_raw_diagonal','pooled_symmetric_median_px','pooled_symmetric_p90_px','good_point_damage_rate')
    lines=['# v2 WLS 구현 오류 정정 재현', '',
        '기존 187589a의 실행과 저장 성능은 보존한다. 기존 `DHT_LOCAL_TRACK_CLOSED`는 잘못 구현된 실행의 역사적 판정이며 정상 v2 방법의 결론으로 사용하지 않는다. 기존 oracle은 `INVALID_FOR_CAUSAL_DECOMPOSITION_DUE_TO_WLS_BUG`로 해석 보류한다.', '',
        '실제 6개 START의 경로·SHA·1792개 순서를 승인 train manifest와 대조했다. INIT_PARITY의 512장 기록은 사전 감사 호출 오류였으며, 확인된 START는 train 1792장에 일치한다. 사후 정정 parity는 학습 전 감사가 올바르게 수행됐다는 소급 증명이 아니다.', '',
        '수정은 single_mode_wls에서 signed와 force를 각 corner별로 계산하도록 옮긴 두 줄뿐이다. loss/utility/decoder/lattice/threshold/seed/batch/학습량과 나머지 학습 코드는 그대로다. 새 6개 run을 각각 2000 step 실행했으며 마지막 checkpoint만 평가했다.', '',
        f'정정 재학습의 고정 gate 판정: `{result["scientific_verdict"]}`. Real DEV 실행: `{result["real_DEV_executed"]}`. FINAL은 열지 않았다.', '',
        '| 모델 | seed | primary | Point 대비 개선 % | median px | P90 px | good damage % |',
        '|---|---:|---:|---:|---:|---:|---:|',
        f'| Point | — | {point:.9f} | 0 | {result["point"][columns[1]]:.4f} | {result["point"][columns[2]]:.4f} | 0 |']
    for arm in ('direct','hough'):
        for r in result[arm]:
            lines.append(f'| {arm} | {r["seed"]} | {r[columns[0]]:.9f} | {100*(point-r[columns[0]])/point:+.3f} | {r[columns[1]]:.4f} | {r[columns[2]]:.4f} | {100*r[columns[3]]:.3f} |')
    if all(r[columns[0]] >= point for r in result['hough']):
        lines += ['', '정상 WLS로 재학습한 Hough도 세 seed 모두 Point의 primary를 넘지 못했다. 따라서 WLS 오류만으로 이전 열세가 모두 설명되지는 않는다. 이 고정 예산·구조의 DHT local fusion 트랙은 종료한다. GT 선 oracle의 개선은 현재 보정식에 진단상 여지가 남음을 보여주며, 예측선과 utility 선택의 결합이 여전히 병목이라는 해석을 지지한다.']
    lines += ['', '양의 변화율이 개선이다. 세 Hough seed가 모두 고정 gate를 통과해야 실사 DEV 평가가 허용된다.', '',
              '| Hough seed | buggy primary | fixed-WLS posthoc | 정상 WLS 재학습 |', '|---|---:|---:|---:|']
    old_rows={r['name']:r for r in replay['records']}
    for r in result['hough']:
        old=old_rows[f'hough_seed{r["seed"]}']
        lines.append(f'| {r["seed"]} | {old["buggy"][columns[0]]:.9f} | {old["fixed_posthoc"][columns[0]]:.9f} | {r[columns[0]]:.9f} |')
    lines += ['', 'posthoc는 기존 raw_line/utility를 그대로 사용하는 GT-free 재결합이다. 기존 buggy gradient로 학습한 가중치의 실행 영향이며 정상 학습을 대체하지 않는다.', '',
              'Oracle은 공통 eligible-edge 교집합과 동일한 gain>0.25px GT 선택 규칙으로 재계산했다. 선이 다르면 실제 선택된 edge도 달라진다. 따라서 아래 차이는 순수 격자/예측 오차의 인과 기여율이 아니다.', '',
              '| Oracle 가지 | primary | baseline 대비 개선 % |','|---|---:|---:|']
    for key,val in oracle.items():lines.append(f'| {key} | {val:.9f} | {100*(oracle["point"]-val)/oracle["point"]:.3f} |')
    lines += ['', '예측선 oracle의 출처는 기존 v1 Hough seed1이다. GT-assisted 선택 결과를 배포 정확도나 달성 가능한 성능 상한으로 해석하지 않는다.', '',
              '| Hough seed | utility AUROC | AUPRC | signed gain Spearman |', '|---|---:|---:|---:|']
    for r in result['hough']:
        u=r['utility_calibration'];lines.append(f'| {r["seed"]} | {u["AUROC"]:.4f} | {u["AUPRC"]:.4f} | {u["spearman_utility_vs_gain"]:.4f} |')
    lines += ['', 'Utility 통계는 선택된 단일 edge의 unit-weight counterfactual gain>0.25px 사건에 대한 것이다. 여러 edge를 예측 utility로 동시에 결합한 최종 보정의 calibration이나 기대 순이익과 같지 않다. 정상 WLS에서는 단일 edge의 감독·실행 동작 일치를 회귀검사로 확인했다.', '',
              '고정 프로토콜 내 정정 재현의 결과만 논문 결론에 사용한다. 정확한 GT 선 oracle의 개선은 선 기하에 진단상 여지가 있음을 보이나, GT 선택을 포함하므로 예측선/utility 각각의 병목 기여도는 분리하지 못한다. DHT 전체의 불가능성이나 미래 성공을 확정하지 않는다.', '',
              '원본 v1/v2 결과 JSON·checkpoint·예측·GT 보존 해시 검사는 PRESERVATION_VERIFIED.json에, 수식과 전체 저장 예측 독립 검사는 REGRESSION_TESTS.json/OFFLINE_REFERENCE_AUDIT.json에 기록한다. 정정 재학습의 전체 7×512 frame 좌표 지표와 WLS 독립 검사는 FINAL_INDEPENDENT_AUDIT.json에 기록한다.']
    text='\n'.join(lines)+'\n'
    (DOC/'CORRECTION.md').write_text(text)
    cohorts={}
    for name in NAMES:
        ev=read_json(RAW/f'evaluations/{name}_synth_val.json')
        rows=ev['records']; difficulty=np.array([r['base_frame_mean_px'] for r in rows]); gain=np.array([r['gain_frame_px'] for r in rows])
        cohorts[name]=[{'quartile':i+1,'count':len(indices),'mean_baseline_px':float(difficulty[indices].mean()),'mean_final_gain_px':float(gain[indices].mean())} for i,indices in enumerate(np.array_split(np.argsort(difficulty,kind='stable'),4))]
    immutable_json(DOC/'BASELINE_DIFFICULTY_COHORTS.json',cohorts)
    print(text,flush=True)


def final_audit():
    """Recompute primary/quantiles/damage from coordinates without assessment helpers."""
    data=list(ObservationDataset(EXPORT/'synth_val.json',targets=True))
    audits=[]
    for name in NAMES:
        prediction=read_json(RAW/f'predictions/{name}_synth_val.json')
        ev=read_json(RAW/f'evaluations/{name}_synth_val.json')
        saved={r['frame_id']:r for r in prediction['records']}
        assert len(saved)==512 and prediction['GT_opened'] is False
        normalized=[];pooled=[];good=damage=covered=catastrophic=0;ref_error=0.
        for item in data:
            r=saved[item['frame_id']];q=np.asarray(r['points']);p=item['base_points'].numpy().astype(float)
            target=item['target_points'].numpy().astype(float);valid=item['target_valid'].numpy()
            pred_valid=np.asarray(r['point_valid']);base_valid=item['point_valid'].numpy();D=float(item['image_hw'].norm())
            choices=[]
            for perm in item['symmetry_permutations'].numpy():
                mask=valid[perm][:8];gt=target[perm][:8]
                b=np.full(8,D);m=np.full(8,D)
                bv=mask & base_valid[:8];mv=mask & pred_valid[:8] & np.isfinite(q[:8]).all(-1)
                b[bv]=np.linalg.norm(p[:8][bv]-gt[bv],axis=-1)
                m[mv]=np.linalg.norm(q[:8][mv]-gt[mv],axis=-1)
                choices.append((float(b[mask].mean()) if mask.any() else D,float(m[mask].mean()) if mask.any() else D,b,m,mask,mv))
            bm=min(choices,key=lambda x:x[0]);mm=min(choices,key=lambda x:x[1])
            normalized.append(mm[1]/D);pooled.extend(mm[3][mm[4]] if mm[4].any() else [D]*8)
            g=bm[4] & (bm[2]<=5);good+=int(g.sum());damage+=int((g & (bm[3]>10)).sum());covered+=int(bm[5].sum())
            catastrophic+=int(bm[0]<=10 and mm[1]>50)
            assert np.array_equal(q[8],p[8])
            if name!='point':
                args=(p[None],base_valid[None],item['point_sigma'].numpy().astype(float)[None],np.asarray(r['raw_line'])[None],np.asarray(r['utility'])[None],np.array([.005*D],dtype=np.float32).astype(float),item['image_hw'].numpy().astype(float)[None])
                ref=numpy_reference(*args)[0]
                ref_error=max(ref_error,float(np.max(np.abs(ref-q))))
                assert np.linalg.norm(np.asarray(r['correction'])[:8],axis=-1).max()<=.01*D+1e-4
        computed={'primary_frame_mean_over_raw_diagonal':float(np.mean(normalized)),
            'pooled_symmetric_median_px':float(np.median(pooled)),'pooled_symmetric_p90_px':float(np.percentile(pooled,90)),
            'good_point_count_base_le_5px':good,'good_point_damage_count_method_gt_10px':damage,
            'coverage_count':covered,'new_catastrophic_frame_count':catastrophic}
        errors={key:abs(val-ev[key]) for key,val in computed.items()}
        assert max(errors.values())<1e-9, errors
        assert ref_error<.001, ref_error
        audits.append({'name':name,'metrics':computed,'metric_max_abs_error':max(errors.values()),'numpy_WLS_max_px':ref_error,'PASS':True})
    training=read_json(DOC/'TRAINING_AUDIT.json')
    for run in training['runs']:
        start=run['START'];end=run['COMPLETION']
        checkpoint=torch.load(end['checkpoint'],map_location='cpu')
        assert checkpoint['steps']==2000 and all(torch.isfinite(t).all() for t in checkpoint['state_dict'].values())
        assert start['config_sha256']==sha256(start['config'])
        assert start['manifest_sha256']==sha256(EXPORT/'train.json')
        assert end['checkpoint_sha256']==sha256(end['checkpoint'])
    immutable_json(DOC/'FINAL_INDEPENDENT_AUDIT.json',{'PASS':True,'runs':audits,
        'neural_forward_count':0,'optimizer_updates':0,'scope':'independent coordinate metrics and absolute-coordinate NumPy WLS, 7x512 frames; finite final states and source/config/manifest hashes'})
    print('FINAL INDEPENDENT AUDIT PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['replay','report','final']);a=p.parse_args()
    {'replay':replay_audit,'report':report,'final':final_audit}[a.phase]()
