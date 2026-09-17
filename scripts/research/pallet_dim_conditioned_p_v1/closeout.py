"""Require every requested stage before publishing an execution-complete report."""
import subprocess,io,unittest
import numpy as np
import torch
import dcp_env as E
from eval_math import contrast,damage,summary

def secondary():
    outputs={}
    for prefix,pairs in [('SQUARE',[('S1_SYM','S0_FIXED'),('S2_META_SYM','S1_SYM'),('S2_META_SYM','S0_FIXED')]),
      ('MIXED_SYNTH',[('M1_META_SYM','M0_DIM_SYM')]),('MIXED_DEV',[('M1_META_SYM','M0_DIM_SYM')]),('MIXED_SQUARE',[('M1_META_SYM','M0_DIM_SYM')])]:
        rows=E.read(E.RAW/f'{prefix}_METRICS.json');outputs[prefix]={}
        for new,base in pairs:
            b=[[r for r in rows[f'{base}_seed{s}'] if r['evaluable']] for s in [1,2,3]];n=[[r for r in rows[f'{new}_seed{s}'] if r['evaluable']] for s in [1,2,3]]
            assert all([r['id'] for r in x]==[r['id'] for r in b[0]] for x in [*b,*n])
            c=contrast([[r['E_sym'] for r in x] for x in n],[[r['E_sym'] for r in x] for x in b],None if prefix=='MIXED_SYNTH' else [r['session'] for r in b[0]])
            d=[damage(x,y) for x,y in zip(b,n)];c['damage']=d
            gross=float(np.mean([summary(y)['gross20']-summary(x)['gross20'] for x,y in zip(b,n)]))
            safe=sum(r['good5_to_bad10']-r['reverse_good5_to_bad10'] for r in d)<=0 and gross<=0
            c.update(safety_pass=bool(safe),gross20_delta=gross)
            c['verdict']='WORSENED' if c['CI95'][0]>0 else 'SUPPORTED_DIAGNOSTIC_ONLY' if c['CI95'][1]<0 and c['improved_seeds']>=2 and safe else 'UNRESOLVED'
            outputs[prefix][new+'__minus__'+base]=c
    E.write(E.DOC/'SECONDARY_STATISTICS.json',outputs)
    for title,file,populations,limitation in [('Square real-supervised secondary','SQUARE_SECONDARY_REPORT.md',['SQUARE'],
      'Dimensions are constant within this cohort: dimension utility is not identifiable. Real-supervised engineering evidence, not synthetic-only main evidence. 696 original train images, matched usable subset; 96,000 exposures per fit. DEV155 reused/interleaved; no early stopping. Train probes at1000/3000/6000 are fixed and never select checkpoints.'),
      ('Mixed-domain routing diagnostic','MIXED_ROUTING_DIAGNOSTIC.md',['MIXED_SYNTH','MIXED_DEV','MIXED_SQUARE'],
      'DIAGNOSTIC_ONLY: each batch8 synthetic +8 square real; domain/group confounding remains. Populations are never pooled into a success statistic. Feasible conditioning is not evidence that dimensions discover physical symmetry.')]:
        lines=['# '+title,'',limitation,'']
        for pop in populations:
            lines+=['## '+pop,'','| arm | E_sym | median px | P90 px | PCK10 |','|---|---:|---:|---:|---:|']
            source=E.read(E.DOC/f'{pop}_RESULTS.json')['summary'];arms=E.SQUARE_ARMS if pop=='SQUARE' else E.MIXED_ARMS
            for a in arms:
                avg=lambda k:np.mean([source[f'{a}_seed{s}'][k] for s in [1,2,3]])
                pck=np.mean([source[f'{a}_seed{s}']['PCK']['10'] for s in [1,2,3]])
                lines.append(f'| {a} | {avg("E_sym"):.8f} | {avg("matched_pooled_corner8_median_px"):.4f} | {avg("matched_pooled_corner8_P90_px"):.4f} | {pck:.4f} |')
            lines+=['']+[f'- {k}: Δ={v["delta"]:.8g}, 95% CI {v["CI95"]}; {v["verdict"]}.' for k,v in outputs[pop].items()]+['']
        (E.DOC/file).write_text('\n'.join(lines))

def main():
    required=['PAPER_TRAINING_COMPLETE','SQUARE_TRAINING_COMPLETE','MIXED_TRAINING_COMPLETE','SYNTH_HELDOUT_RESULTS','REAL_DEV_RESULTS','PAIRED_STATISTICS','DAMAGE_DECOMPOSITION','METADATA_SENSITIVITY','RUNTIME_AND_MEMORY','PAPER_POSE_RESULTS','SQUARE_POSE_RESULTS','MIXED_POSE_RESULTS','SQUARE_RESULTS','MIXED_SYNTH_RESULTS','MIXED_DEV_RESULTS','MIXED_SQUARE_RESULTS']
    for name in required:assert E.read(E.DOC/(name+'.json'))['complete'],name
    adapter=E.read(E.DOC/'ACTUAL_ADAPTER_TESTS.json');assert adapter['PASS']
    correction_path=E.DOC/'CANONICAL_DIMENSION_CORRECTION.json'
    correction=E.read(correction_path) if correction_path.exists() else None
    if correction:assert correction['complete'] and correction['metadata_conditioned_main_updates_before_correction']==0
    else:
        # A clean reproduction uses corrected prepare.py from the outset and
        # must not manufacture this execution's historical correction.
        provenance=E.read(E.DOC/'DIMENSION_INPUT_PROVENANCE.json')
        assert provenance['G38_rows_reconstructed']==40000 and provenance['inference_receives_fixed_XYZ_only']
        E.verify([provenance['fixed_dimension_source'],provenance['fixed_dimension_builder']])
    valid_smoke=E.read(E.DOC/'SMOKE_COMPLETE.json')['updates'];assert valid_smoke==200
    invalid_smoke=correction['archived_invalid_N4_smoke_updates'] if correction else 0
    cpu_updates=adapter['diagnostic_CPU_optimizer_updates']+(correction['archived_CPU_adapter_updates'] if correction else 0)
    total_updates=180000+valid_smoke+invalid_smoke+cpu_updates
    E.verify(E.read(E.DOC/'SOURCE_BINDINGS.json')['files']);assert E.sha(E.R0)==E.R0_SHA
    square=E.read(E.DOC/'SQUARE_CACHE_COMPLETE.json');E.verify([square['corrected_target_view'],square['source_membership'],*square['cache_bindings']])
    E.verify(E.read(E.DOC/'SECONDARY_TRAINING_CODE_LOCK.json')['files'])
    E.verify(E.read(E.DOC/'DEV_CACHE_COMPLETE.json')['metadata_bindings'])
    for k,s in E.read(E.DOC/'SOURCE_BINDINGS.json')['cache_stat'].items():
        p=E.ROOT/s['path'];assert p.stat().st_size==s['bytes'] and p.stat().st_mtime_ns==s['mtime_ns'],k
    fits=[]
    for arm in [*E.ARMS,*E.SQUARE_ARMS,*E.MIXED_ARMS]:
        for seed in [1,2,3]:
            c=E.read(E.DOC/f'fits/{arm}_seed{seed}.json');assert c['complete'] and c['steps']==6000 and not c['smoke'];assert E.sha(E.ROOT/c['checkpoint']['path'])==c['checkpoint']['sha256']
            ck=torch.load(E.ROOT/c['checkpoint']['path'],map_location='cpu',weights_only=False);assert ck['step']==6000 and ck['complete'];assert all(torch.isfinite(v).all() for v in ck['model_state_dict'].values());fits.append(c)
            for b in ck['bindings']:
                path=E.ROOT/b['path']
                if E.sha(path)!=b['sha256']:
                    assert arm in ['N0_BASE_REPLAY','N1_SYM_ONLY'] and path.parent==E.DOC
                    archive=E.DOC/'history/TRAIN_PROTOCOL_LOCK_BEFORE_CANONICAL_CORRECTION.json' if path.name=='TRAIN_PROTOCOL_LOCK.json' else E.DOC/'history/camera_facing_dimension_correction'/path.relative_to(E.DOC)
                    assert E.sha(archive)==b['sha256'],b
    assert len(fits)==30;E.write(E.DOC/'TRAINING_COMPLETE.json',dict(complete=True,fits=30,main_updates=180000,valid_smoke_updates=valid_smoke,archived_invalid_smoke_updates=invalid_smoke,total_smoke_updates=valid_smoke+invalid_smoke,discarded_CPU_adapter_test_updates=cpu_updates,all_optimizer_updates=total_updates,exposures=2880000,checkpoints=[c['checkpoint'] for c in fits],R0_hash=E.sha(E.R0),OLD_P_preserved=True))
    replay={}
    for seed in [1,2,3]:
        old=torch.load(E.C.BRAW/f'runs/seed{seed}/last.pt',map_location='cpu',weights_only=False)
        new=torch.load(E.RAW/f'runs/N0_BASE_REPLAY_seed{seed}/last.pt',map_location='cpu',weights_only=False)
        assert set(old['model_state_dict'])==set(new['model_state_dict'])
        deltas={k:float((old['model_state_dict'][k].double()-v.double()).abs().max()) for k,v in new['model_state_dict'].items()}
        replay[str(seed)]=dict(bit_exact=all(v==0 for v in deltas.values()),maximum_parameter_abs_delta=max(deltas.values()),per_tensor_max_abs_delta=deltas,
          old=E.bound(E.C.BRAW/f'runs/seed{seed}/last.pt'),new=E.bound(E.RAW/f'runs/N0_BASE_REPLAY_seed{seed}/last.pt'))
    E.write(E.DOC/'N0_STATE_REPRODUCTION.json',dict(seeds=replay,source_order_optimizer_exact=True,bit_determinism_not_assumed=True,
      metric_tolerance=E.read(E.DOC/'TRAIN_PROTOCOL_LOCK.json')['parity_tolerance']))
    for family in [E.ARMS,E.SQUARE_ARMS,E.MIXED_ARMS]:
        for seed in [1,2,3]:
            a=[c for c in fits if c['arm'] in family and c['seed']==seed];assert len({c['initial_base_sha256'] for c in a})==1 and len({c['batch_order_sha256'] for c in a})==1
    stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover(str(E.HERE),pattern='test_*.py'));assert tests.wasSuccessful()
    E.write(E.DOC/'REGRESSION_TESTS.json',dict(PASS=True,tests=tests.testsRun,log=stream.getvalue(),final=True));E.write(E.DOC/'TEST_RESULTS.json',E.read(E.DOC/'REGRESSION_TESTS.json'))
    secondary();stats=E.read(E.DOC/'PAIRED_STATISTICS.json');pose=E.read(E.DOC/'PAPER_POSE_RESULTS.json')['summary'];runtime=E.read(E.DOC/'RUNTIME_AND_MEMORY.json')['summary']
    lines=['# Dimension-Conditioned Symmetry-Aware P v1 — 완료 보고','',
      f'R0는 완전 고정했다. 기존 P의222 후보·반경0.08 bbox diagonal·center8·검출 후보/box/score/order/index를 유지했다. 새 refiner30회 ×6000 =180,000 updates. 유효 smoke{valid_smoke}, 보존된 무효 smoke{invalid_smoke}, 폐기한 CPU adapter{cpu_updates} one-step updates는 별도다(총 optimizer 호출{total_updates:,}).','',
      ('중요한 계약 정정: 참고 exporter의 G38 dimensions_m은 canonical이 아니라 camera-facing였다. 치수 없는 N0/N1 6회는 유지하고, DIM 본학습은 한 step도 실행하기 전에 고정 renderer XYZ로 정정·전수 검증했다. 과거 입력·normalization·smoke와 중단 경위는 CANONICAL_DIMENSION_CORRECTION.json 및 history에 보존했다. GT pose를 보고 보이는 W/D를 입력하는 방식은 사용하지 않는다.' if correction else '처음부터 수정된 prepare.py의 고정 renderer XYZ를 사용한 재실행이다. 과거 실행의 무효 smoke·정정 이력을 새로 만들거나 학습 budget에 포함하지 않는다.'),'',
      '치수 입력은 canonical [W,D,H]에서 만든 [logW,logD,logH,log(W/D),log(H/sqrt(WD))], paper TRAIN55,980행 mean/std로 표준화했다. META는 C1/C2/C4 one-hot을 추가한다. G38/legacy는 고정 renderer XYZ를 사용했고 G38은 원본 builder와 40,000행 전수 대조했다. raw dimensions_m은 camera-facing라 입력에서 배제했다. 실사는 registry만 사용했다. **실사 object_type이 외부에서 알려져 있다는 조건부 결과**이며 unknown-type deployment는 지원/측정하지 않았다. GT pose 기반 W/D swap·추론 GT branch는 없다.','',
      'Loss target은 frozen R0 phase에 가장 가까운 허용 whole-object GT tuple을 TRAIN에서만 선택한다. 같은 tuple의 좌표와 mask가 함께 움직이며, corner별 자유매칭은 없다. Calibration은 모든 arm에 동일한 기존 fixed-index Gaussian target CE를 사용했다. 원본 P lambda/cap 고정, temperature만 synthetic calibration에서 선택했다.','']
    for split in ['SYNTH_HELDOUT','REAL_DEV']:
        s=E.read(E.DOC/f'{split}_RESULTS.json')['summary'];lines+=['## '+split,'','3-seed 지표 평균. median/P90은 matched corner8, E_sym/PCK는 missing penalty를 포함한 전체 GT 분모.','',
          '| arm | median px | P90 px | E_sym | PCK10 | translation median cm |','|---|---:|---:|---:|---:|---:|']
        for arm in ['OLD_P',*E.ARMS]:
            avg=lambda k:np.mean([s[f'{arm}_seed{i}'][k] for i in [1,2,3]])
            pk=np.mean([s[f'{arm}_seed{i}']['PCK']['10'] for i in [1,2,3]]);tr=np.mean([pose[split][f'{arm}_seed{i}']['translation_cm']['median'] for i in [1,2,3]])
            lines.append(f'| {arm} | {avg("matched_pooled_corner8_median_px"):.4f} | {avg("matched_pooled_corner8_P90_px"):.4f} | {avg("E_sym"):.8f} | {pk:.4f} | {tr:.4f} |')
        lines+=['','| contrast | delta | CI95 | improving seeds | verdict |','|---|---:|---|---:|---|']
        for k,c in stats['contrasts'][split].items():lines.append(f'| {k} | {c["delta"]:.8g} | {c["CI95"]} | {c["improved_seeds"]}/3 | {c["verdict"]} |')
        lines+=['',f'OLD_P 재현 사전 tolerance 통과: {stats["OLD_P_reproduction"][split]["within_prelocked_tolerance"]}. CUDA grid_sample backward 비결정성 때문에 bit-exact 재학습을 주장하지 않는다.','']
    real=stats['contrasts']['REAL_DEV'];syn=stats['contrasts']['SYNTH_HELDOUT'];dam=E.read(E.DOC/'DAMAGE_DECOMPOSITION.json')['contrasts']
    pack=real['N4_META_SYM__minus__N0_BASE_REPLAY'];code=real['N4_META_SYM__minus__N3_DIM_SYM']
    dimension_damage=dam['REAL_DEV']['N2_DIM_ONLY__minus__N0_BASE_REPLAY']
    lines+=['## 판정 요약과 UNRESOLVED의 뜻','',
      f'실사 N4−N0: {pack["verdict"]}, ΔE_sym={pack["delta"]:.8g}, CI95={pack["CI95"]}, {pack["improved_seeds"]}/3 seed 개선. 알려진 object_type 조건에서의 작고 제한적인 개선이며 재사용 DEV 결과다.',
      f'실사 N4−N3: {code["verdict"]}. 따라서 explicit symmetry code의 필수성이나 보편적 우월성을 주장하지 않는다. Target-only 증분도 각 contrast를 따로 해석한다.',
      'UNRESOLVED는 실행 실패가 아니다. CI가0을 포함하거나, CI가 개선 방향이어도 사전 안전 기준을 통과하지 못한 경우다.',
      f'실사 N2−N0의 canonical-aligned good<5→bad>10 합계는 {sum(r["good5_to_bad10"] for r in dimension_damage)}건, 역방향은 {sum(r["reverse_good5_to_bad10"] for r in dimension_damage)}건이다. 평균 개선과 별개로 이 안전 조건을 확인해야 한다.',
      f'합성 N4−N0 gross20 변화는 {syn["N4_META_SYM__minus__N0_BASE_REPLAY"]["gross20_delta"]*100:+.5f} percentage points다. 따라서 평균/CI만으로 전체 성공을 선언하지 않는다.',
      '6D 표는 기술통계이며 paired superiority 검정을 추가로 주장하지 않는다. Translation/rotation/ADD와 IoU3D가 같은 방향인지 각 지표를 확인해야 한다.','',
      '## Metadata 민감도 — 실사 3-seed 평균','',
      '| 진단 | ΔE_sym vs correct | 평균 좌표 변화 px | top1 변화율 |','|---|---:|---:|---:|']
    sensitive=E.read(E.DOC/'METADATA_SENSITIVITY.json')['results']['REAL_DEV']
    for mode,label in [('D1','TRAIN mean dims'),('D2','frozen shuffled dims'),('D3','wrong group'),('D4','zero standardized metadata')]:
        rr=[sensitive[f'N4_META_SYM_seed{s}_{mode}'] for s in [1,2,3]]
        lines.append(f'| {label} | {np.mean([r["delta_E_sym"] for r in rr]):+.8f} | {np.mean([r["coordinate_delta_px"] for r in rr]):.4f} | {np.mean([r["top1_changed_fraction"] for r in rr]):.4f} |')
    locality=E.read(E.DOC/'LOCAL_TARGET_RANGE_AUDIT.json')['paper_train'];square_local=E.read(E.DOC/'SQUARE_TARGET_RANGE_AUDIT.json')
    counts=E.read(E.DOC/'SYMMETRY_CONTRACT_AUDIT.json')['counts']['train']
    lines+=['','치수 perturbation은 실제 출력 변화 진단이지 인과 증명은 아니다. Wrong-G가 나빠지지 않는 결과도 그대로 보존했다.','',
      '## TRAIN target locality','',
      f'Paper TRAIN 계약 수: C1={counts.get("C1",0)}, C2={counts.get("C2",0)}, C4={counts.get("C4",0)}. 사용 가능 matched C2 중 nonidentity phase는 {sum(v for k,v in locality["C2"]["branches"].items() if k!="0")}/{locality["C2"]["frames"]}.',
      f'반경 내 코너 비율: C1 {locality["C1"]["after"]["fraction_in_radius"]*100:.3f}%, C2 {locality["C2"]["after"]["fraction_in_radius"]*100:.3f}%. Paper에서는 전후 비율이 동일하다.',
      f'Square C4 matched TRAIN={square_local["C4_frames"]}; branch counts={square_local["branch_counts"]}. 반경 내 비율 {square_local["residuals"]["before"]["fraction_in_radius"]*100:.3f}%→{square_local["residuals"]["after"]["fraction_in_radius"]*100:.3f}%. 반경은 확대하지 않았다.',
      'Square 세부 결과: [SQUARE_SECONDARY_REPORT.md](SQUARE_SECONDARY_REPORT.md). Mixed 세 집단별 결과: [MIXED_ROUTING_DIAGNOSTIC.md](MIXED_ROUTING_DIAGNOSTIC.md).','']
    lines+=['## 비용·해석 제한','',
      'P 기본18,962 params; DIM20,259(+1,297,6.84%); META20,307(+1,345,7.09%).',
      f'동일 RTX3080 full pipeline median: N0 {runtime["N0_BASE_REPLAY"]["full_ms"]["median"]:.3f}ms, N4 {runtime["N4_META_SYM"]["full_ms"]["median"]:.3f}ms. 메모리는 두 head와R0가 함께 상주한 process scope이며 arm별 독립 측정이라고 부르지 않는다.',
      'Metadata perturbation 수치는 METADATA_SENSITIVITY.json: correct 대비 mean/shuffle/wrong-G/zero 변화는 민감도 진단이지 인과적 증명이 아니다. N4−N3가 unresolved이면 explicit code가 필수라고 주장하지 않는다.',
      '합성 frame bootstrap 의존성 한계와 scenario-cluster 부 분석을 같이 보존했다. 실사319는 reused DEV13세션, 다중 비교 보정 confirmatory claim은 없다. Geometry-reconstructed pose GT는 독립 물리 계측이 아니다. Source C1의 physical front phase 모호성은 남는다.',
      'Square와mixed는 별도 보고서다. Square 안에서는 치수가 상수이므로 치수 효용을 검증하지 않는다. Mixed에서는 synthetic/real domain과 C2/C4가 얽혀 있으므로 기능적 routing 진단으로만 해석한다.',
      '원본 R0/P/D/L/PoseFix·데이터·cache·논문/배포를 수정하지 않았고 FINAL/sealed 열람, 재부팅/드라이버 변경, 새로운 sweep이나 seed 추가는 없다.',
      f'최종 회귀검사 {tests.testsRun}/{tests.testsRun} PASS. 성능 성공과 무결성 PASS는 별개다.','']
    (E.DOC/'FINAL_REPORT_KO.md').write_text('\n'.join(lines))
    E.write(E.DOC/'FINAL_AUDIT.json',dict(complete=True,status='COMPLETE',fits=30,main_updates=180000,valid_smoke_updates=valid_smoke,archived_invalid_smoke_updates=invalid_smoke,discarded_CPU_adapter_test_updates=cpu_updates,tests=tests.testsRun,
      required_files=[E.bound(E.DOC/(n+'.json')) for n in required],R0_hash_unchanged=True,OLD_P_preserved=True,source_bindings_unchanged=True,
      source_cache_bytes_mtime_unchanged=True,original_files_written=False,FINAL_access=False,real_selection=False,user_unrelated_changes_preserved=True,
      independent_confirmation=False,unknown_object_type_deployment_supported=False,checkpoint_selection='final6000',missing=[]))
    print('FINAL_COMPLETE 30 fits 180000 main updates',flush=True)
if __name__=='__main__':main()
