"""Contract evidence, independent coordinate audit, and final Korean report."""
import argparse
import subprocess
import sys

import numpy as np
import torch

from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.data import ObservationDataset
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.util import read_json, sha256, canonical_sha
from scripts.research.pallet_symmetry_dht_local_v2.symdht_local_v2.runner import seed_all, sequence
from .core import GainSelector
from .experiment import ROOT, PACKAGE, DOC, RAW, EXPORT, SPLITS, CHECKPOINT, TEST_Q, save, preserved, state_hash


def tests():
    proc=subprocess.run([sys.executable,'-m','pytest',str(PACKAGE/'tests'),'-q'],cwd=ROOT,capture_output=True,text=True)
    save(DOC/'REGRESSION_TESTS.json',dict(exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (PACKAGE/'tests').glob('test_*.py')}))
    print(proc.stdout,flush=True);assert proc.returncode==0,proc.stderr


def independent_targets():
    results=[]
    for split,name in SPLITS.items():
        data=ObservationDataset(EXPORT/(name+'.json'),targets=True)
        cache=torch.load(RAW/f'{split}_observations.pt',map_location='cpu')
        recorded=read_json(RAW/f'{split}_targets.json');max_error=0.;ref_rows=[]
        for i,item in enumerate(data):
            p=cache['p'][i].numpy().astype(float);q=cache['q'][i].numpy().astype(float)
            assert np.array_equal(p,item['base_points'].numpy()) and np.array_equal(p[8],q[8])
            assert cache['frame_ids'][i]==item['frame_id']==recorded[i]['frame_id']
            diag=float(item['image_hw'].norm());valid=item['point_valid'].numpy()
            best=None
            for choice,perm in enumerate(item['symmetry_permutations'].numpy()):
                mask=item['target_valid'].numpy()[perm][:8];gt=item['target_points'].numpy().astype(float)[perm][:8]
                pe=np.full(8,diag);qe=np.full(8,diag)
                pv=mask & valid[:8] & np.isfinite(p[:8]).all(-1);qv=mask & valid[:8] & np.isfinite(q[:8]).all(-1)
                pe[pv]=np.sqrt(((p[:8][pv]-gt[pv])**2).sum(-1));qe[qv]=np.sqrt(((q[:8][qv]-gt[qv])**2).sum(-1))
                pm=float(pe[mask].mean()) if mask.any() else diag
                qm=float(qe[mask].mean()) if mask.any() else diag
                if best is None or pm<best['pm']:
                    best=dict(pm=pm,qm=qm,pe=pe,qe=qe,mask=mask,choice=choice,pv=pv,qv=qv,d=diag)
            r=recorded[i];gain=best['pm']-best['qm']
            assert r['choice']==best['choice']
            max_error=max(max_error,abs(gain-r['gain_px']),abs(gain/diag-r['gain_normalized']))
            ref_rows.append(best)
        assert max_error<1e-12
        results.append(dict(split=split,frames=len(data),max_gain_error=max_error,PASS=True))
        if split=='test':test_rows=ref_rows
    return results,test_rows


def coordinate_metrics(rows,pick):
    values=[];pool=[];cover=damage=good=cat=0
    for r,choose_Q in zip(rows,pick):
        e=r['qe'] if choose_Q else r['pe'];mean=r['qm'] if choose_Q else r['pm'];mask=r['mask']
        values.append(mean/r['d']);pool.extend(e[mask] if mask.any() else [r['d']]*8)
        cover+=int((r['qv'] if choose_Q else r['pv']).sum())
        g=mask & (r['pe']<=5);good+=int(g.sum());damage+=int((g & (e>10)).sum())
        cat+=int(r['pm']<=10 and mean>50)
    return dict(primary=float(np.mean(values)),median_px=float(np.median(pool)),p90_px=float(np.percentile(pool,90)),
        coverage=cover,good_point_damage_rate=damage/max(good,1),catastrophic=cat)


def final():
    protocol=read_json(DOC/'PROTOCOL_LOCK.json');summary=read_json(DOC/'RESULT_SUMMARY.json')
    for path,digest in protocol['source_sha256'].items():assert sha256(ROOT/path)==digest
    assert sha256(CHECKPOINT)==protocol['frozen_checkpoint_sha256'] and sha256(TEST_Q)==protocol['test_Q_sha256']
    targets,rows=independent_targets();test=torch.load(RAW/'test_observations.pt',map_location='cpu')
    cal=read_json(DOC/'CALIBRATION_SELECTION.json');runs=read_json(DOC/'TRAINING_AUDIT.json')['runs']
    checked=[]
    for entry,run in zip(summary['selectors'],runs):
        seed=entry['seed'];ck=torch.load(ROOT/run['checkpoint'],map_location='cpu');assert ck['steps']==2000
        assert sha256(ROOT/run['checkpoint'])==run['checkpoint_sha256']
        assert canonical_sha(sequence(1792,128000,seed))==run['order_sha256']
        seed_all(seed);initial=GainSelector(ck['dimension'],ck['state_dict']['mean'],ck['state_dict']['std'])
        assert state_hash(initial)==run['initial_sha256']
        initial.load_state_dict(ck['state_dict']);initial.eval()
        assert state_hash(initial)==run['final_sha256'] and all(torch.isfinite(t).all() for t in ck['state_dict'].values())
        saved=torch.load(RAW/f'seed{seed}/test_inference.pt',map_location='cpu');assert saved['GT_opened'] is False
        with torch.no_grad():gh=initial(test['x'])/1000.
        assert torch.equal(gh,saved['predicted_gain_normalized'])
        chosen=saved['selected_Q'];c=cal[seed-1]
        expected=torch.zeros_like(chosen) if c['force_Point'] else torch.isfinite(gh)&(gh*test['diagonal']>c['tau_px'])
        assert torch.equal(chosen,expected)
        assert torch.equal(saved['points'],torch.where(chosen[:,None,None],test['q'],test['p']))
        assert torch.equal(saved['points'][:,8],test['p'][:,8])
        m=coordinate_metrics(rows,chosen.numpy());error=max(abs(v-entry['metrics'][k]) for k,v in m.items());assert error<1e-12
        checked.append(dict(seed=seed,independent_metric_max_error=error,exact_action=True,init_and_order_reproduced=True,updates=2000,PASS=True))
    for name,pick in [('Point',np.zeros(512,bool)),('Always_Q',np.ones(512,bool))]:
        m=coordinate_metrics(rows,pick);assert max(abs(v-summary[name][k]) for k,v in m.items())<1e-12
    oracle=read_json(DOC/'ORACLE_HEADROOM.json');m=coordinate_metrics(rows,np.array([r['pm']>r['qm'] for r in rows]))
    assert max(abs(v-oracle['Oracle_P_or_Q'][k]) for k,v in m.items())<1e-12
    original=read_json(DOC/'PRESERVED_ARTIFACT_SHA.json');assert preserved()==original
    save(DOC/'FINAL_INDEPENDENT_AUDIT.json',dict(PASS=True,target_audits=targets,selector_audits=checked,
        original_artifacts_preserved=len(original),frozen_checkpoint_unchanged=True,original_test_Q_unchanged=True,
        inference_actions_exact=True,center8_exact=True,source_hashes_locked=True,oracle_independently_verified=True))
    print('FINAL INDEPENDENT AUDIT PASS',flush=True)


def report():
    result=read_json(DOC/'RESULT_SUMMARY.json');oracle=read_json(DOC/'ORACLE_HEADROOM.json')
    diagnostic=read_json(DOC/'SELECTOR_DIAGNOSTICS.json');training=read_json(DOC/'TRAINING_AUDIT.json')
    audit=read_json(DOC/'FINAL_INDEPENDENT_AUDIT.json');assert audit['PASS']
    lines=['# Frozen Hough gain selector v1 — 최종 보고','',
        '## 관찰','',
        f'판정: `{result["verdict"]}`. 고정 corrected Hough seed1의 Q와 Point P를 그대로 사용했다. selector seed1/2/3은 각각2000 optimizer updates를 실제 수행했다. upstream updates는0이고 frozen checkpoint/기존 Q와 원본 산출물 '+str(audit['original_artifacts_preserved'])+'개의 보존을 확인했다.', '',
        f'P가 고른 공통 whole-object symmetry로 target과 지표를 평가했다. oracle P-or-Q는 primary {100*oracle["relative_primary_gain"]:.3f}% 개선, Q 선택률 {100*oracle["oracle_selected_Q_rate"]:.3f}%, 전체 평균 gain {oracle["oracle_mean_gain_px"]:.6f}px로 사전 headroom gate를 통과했다. GT-assisted 진단이며 배포 성능이 아니다.', '',
        '| 방법 | primary mean/diag | median px | P90 px | coverage | good damage % | catastrophic |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in [('Point',result['Point']),('Always-Q',result['Always_Q']),('Oracle P-or-Q',oracle['Oracle_P_or_Q'])]+[(f'S_gain seed{r["seed"]}',r['metrics']) for r in result['selectors']]:
        lines.append(f'| {name} | {m["primary"]:.9f} | {m["median_px"]:.6f} | {m["p90_px"]:.6f} | {m["coverage"]} | {100*m["good_point_damage_rate"]:.4f} | {m["catastrophic"]} |')
    lines += ['',f'Selector parameter count: {training["runs"][0]["parameter_count"]}. 입력: FEATURE_SCHEMA.json. frozen ROI spatial/local evidence + P/Q coordinates, delta, lines, endpoint relations. normalization은 train만 사용했다.', '',
        '| seed | tau px | Q 선택 % | G_hat MAE px | Spearman | selected-Q mean gain px | false-harm % | 전체 mean gain px |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in diagnostic:
        fmt=lambda x:'N/A' if x is None else f'{x:.6f}'
        lines.append(f'| {r["seed"]} | {r["tau_px"]} | {100*r["selected_Q_rate"]:.3f} | {r["G_hat_MAE_px"]:.6f} | {fmt(r["correlation"]["Spearman"])} | {fmt(r["selected_Q_realized_mean_gain_px"])} | {fmt(None if r["false_harm_rate_among_selected"] is None else 100*r["false_harm_rate_among_selected"])} | {r["population_realized_mean_gain_px"]:.6f} |')
    lines += ['', 'tau는 calibration256의 사전 고정 grid와 제약에서만 선택했다. tau가 null이면 가능한 후보가 없어 사전 고정 always-P fallback을 적용한 것이다. 선택되지 않은 frame gain은0이며 selected subset의 conditional gain과 전체 평균 gain을 구별한다.', '',
        '## 해석','']
    if result['verdict']=='GAIN_SELECTOR_SYNTHETIC_FAIL':
        lines += ['현재 frozen P/Q 후보에는 GT oracle에서 제한적인 선택 여지가 있었지만, 이 단일 gain-regression selector/고정 예산으로 3 seed 모두 안정적인1% 이득을 회수하지 못했다. 현재 논문은 검증된 기존 결과와 failure analysis를 바탕으로 마무리한다. 새 loss/architecture/threshold/seed 탐색을 시작하지 않는다.']
    else:
        lines += ['사전 고정 합성 평가 조건을 세 seed 모두 통과했다. 이는 이 frozen 후보와 분할에서의 선택 성능이며, real DEV 및 6D 일반화와 구별해야 한다.']
    lines += ['', '## 미확정','',
        'train1792는 upstream Hough의 학습 frame이기도 하므로 selector train target의 낙관 편향 가능성이 있다. frame/이미지/hash는 train/cal/test 사이에 분리됐지만 G38/P0/TEX session/source 범주가 겹친다. session-held-out 일반화, 예측선과 selector 각각의 독립 인과 기여도, Hough 일반의 우열/불가능성은 주장할 수 없다.', '',
        f'real DEV 실행: {result["real_DEV_executed"]}; FINAL 열람: {result["FINAL_opened"]}. real DEV가 실행되지 않았다면 실사2D/6D 개선은 미평가다.', '',
        '13개 회귀검사는 REGRESSION_TESTS.json, 공통 assignment target과 exact inference 및 전체 좌표 지표 독립 검사는 FINAL_INDEPENDENT_AUDIT.json, source/session/difficulty별 target 및 상관은 GAIN_TARGET_AUDIT.json에 기록했다. GT line error는 진단에서만 쓰며 selector 입력에는 없다.', '',
        '기존 v2 평가기는 각 방법의 최적 symmetry를 따로 고르므로 Always-Q primary가 과거 보고값과 조금 다를 수 있다. 본 실험에서는 action/target 일치를 위해 baseline assignment를 공통 고정했다. 기존 결과를 수정하지 않았다.', '',
        '대형 observation cache, selector checkpoint와 inference tensor는 data/pallet/results/pallet_hough_gain_selector_v1에 보관하고 기존 Git ignore 정책을 따른다. 코드와 작은 결과/감사 파일만 main에 commit/push한다.']
    (DOC/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['tests','final','report']);args=p.parse_args();globals()[args.phase]()
