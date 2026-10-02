"""Independent numeric checks, provenance chain and local-only closeout report."""
import csv,datetime,hashlib,inspect,json,re,subprocess,time
from collections import Counter
from pathlib import Path
import numpy as np
from .compute import C,R,M,ROOT,DOC,RAW,SOURCE,write,bind,clean
from .paper import md,fmt

def independently_check_scores(scores,expected):
    c=scores['corner_scores'];p=scores['pose_scores'];obs=np.array([e for r in c if r['evaluable'] for e in r['observed_errors']]);full=np.array([e for r in c if r['evaluable'] for e in r['errors']]);valid=[r for r in p if r['available']]
    actual={'matched_pooled_corner8_median_px':np.median(obs),'matched_pooled_corner8_P90_px':np.quantile(obs,.9),'full_PCK10_fraction':np.mean(full<=10),
        'full_penalty_P90_px':np.quantile(full,.9),'observed_corners':len(obs),'full_supervised_corners':len(full),'pose_available_frames':len(valid),'pose_coverage':len(valid)/len(p)}
    for f in ['translation_cm','rotation_deg','yaw_deg']:
        for name,q in [('median',.5),('P90',.9)]:actual['pose_'+f+'_'+name]=np.quantile([r[f] for r in valid],q)
    actual['pose_IoU3D_median']=np.median([r['IoU3D'] for r in valid]);a=np.array([r['ADDsym_normalized'] if r['available'] else np.inf for r in p]);thresholds=np.linspace(0,.1,1001)
    actual['pose_ADDsym_AUC_full']=np.trapz((a[:,None]<=thresholds).mean(0),thresholds)/.1
    checks={k:abs(float(v)-float(expected[k]))<=1e-12 for k,v in actual.items()}
    assert all(checks.values()),checks
    return checks

def main():
    started=time.time();initial=C.read(DOC/'INITIAL_STATE.json');core=C.read(DOC/'CORE_RESULTS.json');scored=C.read(RAW/'YOLO_SCORES.json');checks={}
    for method,s in scored.items():
        if method=='R0':h=core['methods']['R0']['mean']['headline']
        else:arm,seed=method.rsplit('_seed',1);h=core['methods'][arm]['per_seed'][seed]['headline']
        checks[method]=independently_check_scores(s,h)
    # Verify same raw hash and measurement aggregation for reused and new runtimes.
    runtime={};bindings=[]
    for b in ['dope','resnet18','yolo']:
        receipt=C.read(DOC/'RUNTIME_YOLO_SEED1.json') if b=='yolo' else C.read(C.DOC/f'RUNTIME_{b.upper()}_SEED1.json')
        path=RAW/'RUNTIME_YOLO_RAW.json' if b=='yolo' else C.RAW/f'runtime/{b}_seed1.json'
        assert C.sha256(path)==receipt['raw']['sha256'];raw=C.read(path)
        samples=raw['measurements'];rc={}
        assert len(samples)==390 and len(raw['warmup'])==60
        for name in ['base_e2e','n3_seed1_e2e','n3_seed1_only']:
            rows=[r for r in samples if r['path']==name];assert len(rows)==130
            # Historical per-call schema uses timing fields at top level.
            values=[r.get('timing',r)['cuda_event_ms'] for r in rows]
            h=receipt['summary'][name]['cuda_event_ms']
            rc[name]=bool(np.median(values)==h['median_ms'] and np.quantile(values,.9)==h['p90_ms'])
        assert all(rc.values());runtime[b]=rc;bindings.extend([bind(path)])
    # All six existing head receipts/checkpoints; no optimizer or training invoked.
    heads=[]
    for b in ['dope','resnet18']:
        for seed in (1,2,3):
            p=C.DOC/f'TRAIN_{b.upper()}_SEED{seed}.json';r=C.read(p)
            assert r['complete'] and r['steps']==6000 and r['exposures']==96000 and r['dimension_input_to_N3'] and r['symmetry_supervision']
            checkpoint=C.verify(r['checkpoint']);heads.append({'backbone':b,'seed':seed,'receipt':bind(p),'checkpoint':bind(checkpoint),'base_sha256':r['base_checkpoint_sha256']})
    write(DOC/'TRAINING_REUSE_VERIFICATION.json',{'heads':heads,'new_training':0,'optimizer_updates':0})
    from scripts.research.pallet_n3_completion_v3 import evaluation as E
    preservation={}
    for b in ['dope','resnet18']:
        for population in ['DEV319','GREEN0918_119']:
            payload=C.read(C.RAW/f'predictions/{b}_{population}.json')
            methods=E.normalize_prediction_payload(payload)['methods'];largest=0.;checked=0
            for seed in (1,2,3):
                for fid,base in methods['base'].items():
                    after=methods[f'n3_seed{seed}'][fid]
                    assert E._preservation_error(base,after) is None
                    if base['points'] is not None:
                        left=np.array(base['points']);right=np.array(after['points']);finite=np.isfinite(left[:8]).all(-1)&np.isfinite(right[:8]).all(-1)
                        if finite.any():
                            move=np.linalg.norm(right[:8][finite]-left[:8][finite],axis=1)
                            assert float(move.max())<=.01*np.linalg.norm(base['hw'])+1e-4
                            largest=max(largest,float(move.max()))
                    checked+=1
            preservation[b+':'+population]={'frames_times_seeds':checked,'box_score_center_valid_preserved':True,'original_diagonal_1pct_cap':True,'max_move_px':largest}
    write(DOC/'PRESERVATION_RECHECK.json',preservation)
    # Only inspect already approved metadata hashes; never open held-out images.
    snapshot=C.read(ROOT/'_docs/experiments/pallet_green0918_dimension_audit_v1/DATASET_SNAPSHOT.json');source=C.read(C.SOURCE)
    green={r['image']['sha256'] for r in snapshot['records']};exposure={}
    for partition in ['train','calibration','selection','heldout']:
        hashes={r['image_sha256'] for r in source['records'] if r['partition']==partition}
        exposure[partition]={'manifest_rows':sum(r['partition']==partition for r in source['records']),'exact_hash_overlap_with_GREEN':len(hashes&green)}
    write(DOC/'SQUARE_EXPOSURE_AUDIT.json',{'source_manifest':bind(C.SOURCE),'metadata_only':True,'sealed_images_opened':0,'hash_comparisons':exposure,
        'limitations':'Hashes describe prepared synthetic images, not perceptual duplicates or complete upstream training history. Base fitting histories and N3 synthetic-only receipts reused; prior S0/S1 session/identity correspondence unresolved.'})
    # Corrected effective record includes prospective-selection limitations and tested parity.
    effective=C.read(DOC/'RESNET_EFFECTIVE_PROTOCOL.json');effective['base_choice_history']='Epoch10 was fixed within CONSTANT arm. Existing N3 protocol documents substitution for collapsed60epoch RGB. No independent prospective proof that the choice of baseline family was blind to all DEV observations; do not claim independent test selection.'
    effective['folding_equivalence']='Existing actual-checkpoint unit test rerun: zero-context conditioned vs folded RGB logits rtol2e-5/atol2e-6, features exactly equal; PASS in initial40 passed tests.'
    inp=effective['recipe']['input']
    inp['decoder']=effective['actual_training']['decoder']
    inp['checkpoint_selection']='fixed final epoch10; step34990 within CONSTANT arm'
    inp.pop('mse',None);inp['objective']=effective['actual_training']['objective']
    inp['confidence_threshold']=None;inp['threshold_inclusive']='NA: DSNT expectation decoder does not gate channels by peak confidence'
    inp['runtime_validity']='All nine finite spatial expectations; box hull of corner0..7'
    write(DOC/'RESNET_EFFECTIVE_PROTOCOL.json',effective)
    # Explicit frame-level co-occurrence evidence for the large severe-group median change.
    traces=C.read(RAW/'PNP_TRACES.json');changes=C.read(RAW/'POSE_HYPOTHESIS_CHANGES.json');summary=[];examples=[]
    for b,methods in traces.items():
        base='R0' if b=='yolo' else 'base'
        for seed in (1,2,3):
            after=f'N3_DIM_SYM_seed{seed}' if b=='yolo' else f'n3_seed{seed}'
            for group in ['clean','moderate','severe','unclassified']:
                left=[r['metric']['rotation_deg'] for r in methods[base] if r['occlusion']==group and r['pose']['available']]
                right=[r['metric']['rotation_deg'] for r in methods[after] if r['occlusion']==group and r['pose']['available']]
                selected=[r for r in changes if r['backbone']==b and r['seed']==seed and r['occlusion']==group and r['delta_rotation_deg'] is not None]
                summary.append([b,seed,group,len(left),float(np.median(left)),float(np.median(right)),float(np.median(right)-np.median(left)),sum(r['WD_hypothesis_changed'] for r in selected),sum(r['evaluation_symmetry_changed'] for r in selected),sum(r['delta_rotation_deg']< -1e-9 for r in selected),sum(r['delta_rotation_deg']>1e-9 for r in selected)])
                if group=='severe':examples.extend(sorted(selected,key=lambda r:(-abs(r['delta_rotation_deg']),r['id']))[:5])
    write(DOC/'POSE_HYPOTHESIS_ANALYSIS.json',{'summary':summary,'largest_absolute_change_examples':examples,'posthoc':True,'new_threshold':False,'all_rows':bind(RAW/'POSE_HYPOTHESIS_CHANGES.json'),'raw_poses':bind(RAW/'PNP_TRACES.json')})
    (DOC/'POSE_HYPOTHESIS_ANALYSIS.md').write_text('# 자세 가설 변화 대조\n\n아래는 같은 성공 프레임의 회전 중앙값과 가설 전이이다. 가설 변화와 오차 감소의 동반 관측이며 원인 효과로 단정하지 않는다. 일부 큰 변화가 분포 중앙의 순위를 바꿀 수 있고, 중앙값 감소만큼 모든 프레임이 개선된 것은 아니다. 대칭군·축·임계값은 변경하지 않았다. 전체 frame R/T/yaw·가설·오차는 PNP_TRACES에 남겼다.\n\n'+md(['Backbone','Seed','Group','Success','Before R med','After R med','Difference','WD changes','Eval symmetry changes','Improved frames','Worse frames'],summary)+'\n\n설명용 상세 사례는 심한 가림에서 절대 회전 변화 순 상위5개/seed를 사후 선택했으며 정량 평가 집단에는 영향이 없다. 새 큰 오류 임계값을 만들지 않았다.\n')
    # Source bindings include concrete owner locations; no excluded artifact traversal.
    explicit=[C.DEV,C.SOURCE,M.POSE_SOURCE,ROOT/'scripts/paper/pose_metric_closure_v1/symmetry_aware_pose_metrics.py',ROOT/'scripts/research/pallet_dim_conditioned_p_v1/eval_math.py',C.DOC/'PROTOCOL.json',C.DOC/'REUSE_RESULTS.json',C.RAW/'reuse/PER_FRAME_SCORES.json']
    for b in ['dope','resnet18']:explicit += [C.RAW/f'predictions/{b}_DEV319.json',C.RAW/f'predictions/{b}_GREEN0918_119.json',C.RAW/f'evaluation/{b}.json']
    explicit += list((ROOT/'scripts/research/pallet_n3_static_closeout_v1').glob('*.py'))
    explicit += [DOC/p for p in ['EXECUTE_N3_NON_LIFTER_CLOSEOUT_KO.txt','pallet_n3_non_lifter_cli_20261002.txt']]
    bindings.extend(bind(p) for p in explicit)
    write(DOC/'SOURCE_BINDINGS.json',{'direct':bindings,'core_raw_reference_bindings':core['sources'],'comparators_and_students':C.read(DOC/'RAW_COMPARATOR_STUDENT_REGRESSION.json')['sources'],
        'head_bindings':heads,'label_bindings':C.read(DOC/'STATIC_LABEL_AUDIT.json')['sources'],'square_bindings':C.read(DOC/'SQUARE_AUDIT.json')['annotation_bindings'],
        'execution_worktree':str(ROOT),'raw_owner':str(SOURCE),'remote_reference':'a7fb68050e2207a5d8c379a1c1704f4e694b6675','remote_access_performed':False})
    original=C.DOC/'contract/source_context/manuscript_ko_v3.md';paper=DOC/'manuscript_ko_v3_static_closeout.md';original_text=original.read_text();new=paper.read_text()
    counts=np.load(RAW/'BOOTSTRAP_SESSION_COUNTS.npy');validation=C.read(DOC/'BOOTSTRAP_VALIDATION.json')
    validation['draws_with_at_most_one_unique_session']=int(((counts>0).sum(1)<=1).sum())
    validation['conditional_pose_draws_with_at_most_one_success_session']={}
    sessions=C.read(DOC/'PAIRED_POSE_ANALYSIS.json')['contract']['sessions']
    for b,methods in traces.items():
        for name,rows in methods.items():
            supported={r['session'] for r in rows if r['pose']['available']}
            keep=np.array([s in supported for s in sessions])
            validation['conditional_pose_draws_with_at_most_one_success_session'][b+':'+name]=int(((counts[:,keep]>0).sum(1)<=1).sum())
    write(DOC/'BOOTSTRAP_VALIDATION.json',validation)
    # Original x positions remain traceable even where table layout expanded.
    original_gaps=[]
    for match in re.finditer(r'<div id="([^"]+)">(.*?)</div>',original_text,re.S):
        label,content=match.groups();method='';ri=0
        for line in content.splitlines():
            if not line.startswith('|'):continue
            ri+=1;columns=[x.strip() for x in line.strip('|').split('|')]
            for ci,value in enumerate(columns):
                if not re.search(r'(?<![A-Za-z])x(?![A-Za-z])',value):continue
                status='INSERTED';reason='Verified table or directly related replacement panel'
                if label=='tab:case' or 'lifter' in label.lower() or '주행' in line:status='OUT_OF_SCOPE_USER';reason='Excluded; original text/row preserved'
                elif label=='tab:update_alternative':status='BLOCKED_CONTRACT';reason='Original319 panel has no common non-exposed student cohort; separate128 panel inserted'
                elif label=='tab:type_results' and ri>=11 and ci in [6,7]:status='BLOCKED_REFERENCE';reason='Square6D reference absent; x retained'
                original_gaps.append({'table_label':label,'original_markdown_row':ri,'column':ci,'original_value':value,'original_row':line,'status':status,'reason':reason,'new_source_map':'PAPER_CELL_MAP.json'})
    write(DOC/'ORIGINAL_X_DISPOSITION.json',original_gaps)
    # Do not count excluded or undefined values as experimental failure.
    remaining=[
        ['Square119 T/R/yaw/IoU3D/ADDsym','x','BLOCKED_REFERENCE','No independent canonical6D reference; no pseudo truth generated'],
        ['Unknown191 external occlusion','x','NEED_REVIEW','No direct-human label provenance; copied66 split labels not promoted'],
        ['Unknown2428 corner visibility','x','NEED_REVIEW','No approved point status; includes one explicit uncertain point; no valid=visible inference'],
        ['Square119 occlusion/visibility','x','NEED_REVIEW','Existing square validity/manual source is not a reviewed visibility grade'],
        ['Original319 student panel','x','BLOCKED_CONTRACT','Shared non-exposed cohort only128; substitute table clearly separate'],
        ['Square session generalization CI','NA','NOT_APPLICABLE','One session'],
        ['Historical S0/S1 to GREEN identity/exposure','x','BLOCKED_PROVENANCE','No complete identity/session linkage; do not claim independent test'],
        ['Same-environment D/L/PoseFix runtime ranking','x','NOT_CLAIMED','Accuracy comparison only; no matched latency superiority claim'],
        ['Matching Korean v3 LaTeX source','x','NEED_SOURCE','Markdown copy updated; exact-label LaTeX fragments ready, not inserted into unavailable source'],
        ['Original static architecture figure assets','x','NEED_SOURCE','Contract bundle contains Markdown only; original three diagram images absent'],
        ['All lifter work','x','OUT_OF_SCOPE_USER','No raw reads/analysis/inference/control; excluded chapter/table unchanged'],
    ]
    (DOC/'REMAINING_X.md').write_text('# 남은 x / NA / 실제0 / 제외\n\n'+md(['Item','Value','Status','Reason'],remaining)+'\n실제0: 신규 학습·optimizer update·새 self-training·PDF·외부업로드·push 모두0. 모든 기반/seed의 신규 자세 실패·복구0. ResNet 비항등 대칭 선택0과C4 학습행0은 기존 측정값이다. x로 대체하지 않는다.\n')
    # Independent check that rendered numeric fields match CSV and cell-map values.
    cells=C.read(DOC/'PAPER_CELL_MAP.json');cell_checks=[]
    for cell in cells:
        p=Path(cell['new_table_path']);assert C.sha256(Path(cell['source_path']))==cell['source_sha256']
        with p.with_suffix('.csv').open() as f:rows=list(csv.reader(f))
        ci=rows[0].index(cell['metric_and_unit']);value=rows[cell['row_index']+1][ci]
        if isinstance(cell['value'],(float,int)):cell_checks.append(abs(float(value)-cell['value'])<=1e-12)
        elif cell['value'] is None:cell_checks.append(value=='')
        else:cell_checks.append(value==cell['value'])
        if cell['markdown_inserted']:assert '<div id="'+cell['table_label']+'">' in new
    assert all(cell_checks)
    protected={p:C.sha256(ROOT/p)==sha for p,sha in initial['protected_hashes'].items()};assert all(protected.values())
    assert original_text[original_text.index('# 기록된 리프터'):original_text.index('# 논의')]==new[new.index('# 기록된 리프터'):new.index('# 논의')]
    assert original_text[original_text.index('# 참고문헌'):]==new[new.index('# 참고문헌'):]
    initial_tests=Path('/tmp/pallet-n3-static-tests.log').read_text();retests=Path('/tmp/pallet-n3-static-square-tests.log').read_text();assert '40 passed' in initial_tests and '3 passed' in retests
    (DOC/'TESTS_INITIAL.txt').write_text(initial_tests);(DOC/'TESTS_AFTER_INPUT_PATH_REPAIR.txt').write_text(retests)
    verify={'status':'PASS_WITH_DECLARED_REVIEW_AND_SOURCE_BLOCKS','independent_numpy_checks':checks,'runtime_reaggregation':runtime,'table_numeric_checks':len(cell_checks),'table_numeric_all_pass':all(cell_checks),
        'protected_original_files':len(protected),'protected_hashes_unchanged':all(protected.values()),'lifter_chapter_unchanged':True,'reference_list_unchanged':True,
        'tests':{'passed_total':43,'initial_failed':3,'failure_reason':'Missing worktree-local square annotations; raw owner files existed and hashes matched. Restored119 symlinks, no coordinate edits.','resolved_by_retest':3},
        'raw_comparator_student_regression':C.read(DOC/'RAW_COMPARATOR_STUDENT_REGRESSION.json')['checks'],'bootstrap_replication':C.read(DOC/'BOOTSTRAP_VALIDATION.json'),'backbone_preservation':preservation,
        'square_mode_regression':C.read(DOC/'SQUARE_AUDIT.json')['checks'],'training_runs':0,'optimizer_updates':0,'push':False,'PDF':False,'lifter_execution':False,
        'final_verification_seconds':time.time()-started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(DOC/'VERIFY_STATIC_RESULTS.json',verify)
    costs={k:C.read(DOC/p).get('seconds') for k,p in [('core','CORE_RESULTS.json'),('paired_pose','PAIRED_POSE_ANALYSIS.json'),('pose_traces','POSE_TRACE_AND_TAIL.json'),('extra_uncertainty','PAIRED_2D_UNCERTAINTY.json'),('comparator_student_raw_regression','RAW_COMPARATOR_STUDENT_REGRESSION.json')]}
    costs['YOLO_runtime_wall_seconds']=C.read(DOC/'RUNTIME_YOLO_SEED1.json')['elapsed_seconds'];costs['overall_wall_minutes']=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(initial['started_utc'])).total_seconds()/60
    costs['CPU_process_seconds']='NA: not instrumented; phase wall times reported, not processor consumption';costs['new_GPU_measured_calls']=390;costs['new_GPU_warmup_calls']=60;costs['new_untimed_base_calls_for_N3_only']=150;costs['new_unique_runtime_frames']=26;costs['reused_GPU_measured_calls']=780
    write(DOC/'EXECUTION_COST.json',costs)
    status_rows=[['N0/N1 6D','RECOMPUTE + VERIFIED + INSERTED','1914 frame-method rows,6 seeds; same PnP'],['R0/P/N2/N3','REUSE + RAW REGRESSION','26 original score arrays identical'],['ResNet model','CORRECTED + INSERTED','Actual checkpoint10epoch/34990steps; fold parity passes'],['Pose uncertainty','RECOMPUTE + INSERTED','13 session paired10000 draws; all3seeds and ablations'],['Occlusion/material','REUSE + REAGGREGATE + INSERTED','same reviewed128 andunknown191 across backbones'],['Visibility','PARTIAL_LABELS + INSERTED','visible66/external5/unknown2428; old coordinates retained'],['Square','VERIFIED + INSERTED','both600/602 modes;6D x'],['D/L/PoseFix','REUSE + RAW REGRESSION + INSERTED','319/8corners'],['Student alternatives','REUSE + RAW REGRESSION + INSERTED','128 separate cohort'],['Runtime','REUSE2 + MEASURE1 + INSERTED','DOPE/ResNet reused;YOLO new;environments separated'],['LaTeX','READY_TO_INSERT','Exact-label fragments; matching source absent'],['Lifter','OUT_OF_SCOPE_USER','No execution; chapter unchanged']]
    (DOC/'TASK_STATUS.md').write_text('# 실행 상태\n\n'+md(['Task','Status','Evidence / remaining input'],status_rows))
    paired=C.read(DOC/'PAIRED_POSE_ANALYSIS.json');n0=core['methods']['N0_BASE_REPLAY']['mean']['headline'];n1=core['methods']['N1_SYM_ONLY']['mean']['headline']
    report='# N3 비리프터 마감 결과\n\n계산 가능한 비리프터 항목을 실제 계산·검산하고 국문v3 Markdown 복사본의 표와 관련 본문에 반영했다. 모든 논문 미확정 항목을 해결한 것은 아니다. 새 학습0회,optimizer0회이며 리프터 작업·PDF 생성·외부업로드·push는 하지 않았다.\n\n'
    report+='## 바로 확인할 파일\n\n- [결과 반영 원고](manuscript_ko_v3_static_closeout.md)\n- [표/본문 변경 목록](PAPER_RESULTS_PATCH_KO.md) · [칸별 상태표](PAPER_GAP_MATRIX.md) · [원래 x 위치](ORIGINAL_X_DISPOSITION.json)\n- [검증 결과](VERIFY_STATIC_RESULTS.json) · [원시 출처/해시](SOURCE_BINDINGS.json) · [남은 x](REMAINING_X.md)\n\n'
    report+='## 실제 완료한 계산과 원고 반영\n\n'+md(['Task','Status','Evidence'],status_rows)+'\n'
    report+='N0/N1 누락은 reuse.py의CORE_ARMS 조건이었다. 원본 코드를 전역 변경하지 않고 별도 경로에서 같은 evaluator를 호출했다. 기존 R0/P/N2/N3의26개 프레임 점수 배열, 비교군/학생 원시 좌표 재평가,정사각형18개 모드·방법 검산이 일치했다. 별도NumPy 집계와 표CSV/JSON값 대조도 통과했다.\n\n'
    report+=md(['Method','T median/P90 cm','R median/P90 deg','Yaw median/P90 deg','IoU3D','ADDsym full AUC','Pose'],[[name,f"{h['pose_translation_cm_median']:.3f} / {h['pose_translation_cm_P90']:.3f}",f"{h['pose_rotation_deg_median']:.3f} / {h['pose_rotation_deg_P90']:.3f}",f"{h['pose_yaw_deg_median']:.3f} / {h['pose_yaw_deg_P90']:.3f}",h['pose_IoU3D_median'],h['pose_ADDsym_AUC_full'],'319/319'] for name,h in [('N0',n0),('N1',n1)]])
    report+='\n## 결과가 지지하는 주장과 한계\n\n세 기반의 T·R 중앙값 평균은 개선됐다. 아래는N3−Base,단위cm/도,13세션 bootstrap95% 구간이다. ResNet T 중앙값 구간은0을 포함한다. DOPE 회전P90과ResNet 이동P90은 악화되었다. 이 구간은 반복 사용한DEV의 사후 불확실성이며 독립 물리6D 검증이 아니다.\n\n'
    pr=[]
    for b,d in paired['backbones'].items():
        for metric in ['translation_cm_median','translation_cm_P90','rotation_deg_median','rotation_deg_P90']:
            h=d['mean_of_seed_statistics'][metric];pr.append([b,metric,h['mean_seed_delta'],f"[{h['CI95'][0]:.3f}, {h['CI95'][1]:.3f}]"])
    report+=md(['Backbone','Metric','Delta','CI95'],pr)
    report+='\n![Pose uncertainty](figures/pose_uncertainty.png)\n\n![Within-backbone results](figures/backbone_results.png)\n\nN3가N2/PoseFix보다 항상 우수하지 않다. 정사각형에서YOLO N3−R0는 선언602점 기준중앙값−0.522px,PCK+4.042pp지만N3−N2는중앙값+0.055px,PCK−0.332pp로 악화했다. PoseFix의조건부 중앙값5.561px는N3의5.778px보다낮고,P90은N3가낮다. 강건성은세 기반에각각학습한적용근거로제한한다.\n\nResNet은10epoch CONSTANT-fold RGB/DSNT이며60epoch RGB/argmax가아니다. 치수조건은N3에들어간다. 비항등대칭목표선택0회,C4학습행0은실제관측이다. 치수logits민감도는경로활성증거이며치수의독립정확도효과가아니다. [명세정정](PROTOCOL_CORRECTIONS.md)에서원본해시와유효값을대조할수있다.\n\n'
    report+='## 주석·정사각형·실패의 처리\n\n가림변화는`eval_pallet09:1778653661653195264` 한ID의중간→심함차이로확인했다. 개수에맞추어주석을수정하지않았다. 직접가시66점과외부가림5점의기존사람검수를찾아상태만재사용했다. 재클릭좌표와DEV참조가달라기존좌표를유지했고,PnP보조검수이력·예측노출미확인을명시했다. 미분류191장과2428점의가시성은여전히검수가필요하다. [주석감사](STATIC_LABEL_AUDIT.md) · [모델정보없는검수자료](review/index.html).\n\n정사각형두화면밖점은029710/코너0(-42,289),029844/코너4(-7,310)이다.119장한세션·한치수로600/602를분리했고독립6D참조부재로T/R은x다. 원시평가성공ID는전후동일하며신규실패/복구0이다. DOPE109실패는전체319장분모에남겼다. [프레임별pose/가설분석](POSE_HYPOTHESIS_ANALYSIS.md)에서중앙값변화와가설변화를구분한다.\n\n![Occlusion curves](figures/occlusion_pck.png)\n\n![Static examples](figures/static_examples.png)\n\n설명예시는seed1·확정등급ID에서개선/무차이/악화별ID사전순첫사례를선택했다. 사후설명용이며정량집단을바꾸지않았다. 코너예측과참조이며PnP재투영점이아니다.\n\n'
    report+='## 삽입 대기와 남은 입력\n\nLaTeX 표 조각은 준비됐지만 matching국문v3 `.tex`가 없어 그 소스에 삽입하지 않았다. 다른 원고는 수정하지 않았다. 추가 기반별 가시성 세부결과는 CROSS_BACKBONE_SUBGROUPS.json에 있고 주표에는YOLO 가시성만 반영했다. 원래세정적도식이미지파일은전달된bundle에없어복구대기다. 새결과그림4개는생성했다. 리프터장과참고문헌42편은그대로보존했다.\n\n'+md(['Item','Value','Status','Reason'],remaining)+'\n'
    report+='## 실행·검증·실제 비용\n\n'+md(['Phase','Wall seconds / count'],[[k,v] for k,v in costs.items()])
    report+='\n새GPU본측정은YOLO3경로×130=390회,준비60회,보정단독경로용별도untimed기반호출150회다. DOPE/ResNet의기존780본측정은재사용했다. 같은GPU이나YOLO라이브러리환경이다르므로절대속도순위를주장하지않는다. 새YOLO 전체중앙시간은Base11.195ms/N3 15.138ms,보정만3.432ms였다.\n\n검증테스트는최초40통과·3실패(격리worktree주석경로부재),원시소유checkout의동일해시주석119개를읽기용symlink로연결한뒤3개모두통과했다. 수치나테스트허용오차는바꾸지않았다. 새스크립트개발중세seed키명과정규화payload구조를맞추는오류가있었고저장된완료계산은재사용했다. 학습/측정을결과가나쁘다는이유로반복하지않았다.\n\n[실행 명령과 재시작 방법](README_RUN.md) · [상세 비용](EXECUTION_COST.json). CPU processor시간은별도계측하지않아NA이며위값은phase벽시계시간이다.\n'
    (DOC/'FINAL_REPORT_KO.md').write_text(report)
    readme='# 실행 명령\n\n실행 checkout은 `/tmp/pallet-pose-n3-v3` (a7fb680 기반)이다. 원시 자료 소유 checkout은 `/home/minjae/Documents/github/pallet-pose`이며 그main의기존변경은보존했다. 이작업은새namespace만쓴다. 기존통합all/verify는리프터를포함하므로호출하지않는다.\n\n```bash\ncd /tmp/pallet-pose-n3-v3\n'
    for module in ['compute','paired','audits','pose_trace','uncertainty_extra','subgroups','paper','figures','finish']:readme+=f'/home/minjae/anaconda3/envs/pallet-pose/bin/python -m scripts.research.pallet_n3_static_closeout_v1.{module}\n'
    readme+='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_n3_static_closeout_v1.runtime_yolo\n```\n\n실제실행은compute→paired/audits→runtime_yolo→pose_trace→uncertainty_extra→subgroups→paper→figures→finish순이었다. 파일이완료되어있으면재실행할필요가없다. compute와runtime_yolo는완료결과를재사용한다. 그밖의감사/내보내기는결과재생성명령이므로수정이나검증필요가있을때만실행한다. 신규학습/optimizer호출은없다.\n\n검증: 기존test_metrics/test_evaluation/test_reuse/test_square/test_square_yolo와test_adapters::ConstantCheckpointTests를실행했다. 최초로그와경로복원뒤실패3개재실행로그를함께저장했다. GPU는승인된기존장비에서단일프로세스로계측했으며의존성을설치/업그레이드하지않았다.\n'
    (DOC/'README_RUN.md').write_text(readme)
    from .report_ko import write_report
    write_report()
    print('Verification complete',len(cells),'cells',len(protected),'protected originals',flush=True)

if __name__=='__main__':main()
