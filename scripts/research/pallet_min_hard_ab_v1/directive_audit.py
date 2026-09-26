"""Named section19 checks with evidence and explicit user-approved amendments."""
import ast
from collections import Counter
import re
import unittest
import numpy as np
import yaml
from . import common as C
from . import policy
from .labels import validate_frame,coverage
from .figures import FIGURES
from .finalization_inputs import verify as verify_science


def main():
    read=lambda n:C.read(C.DOC/n)
    tests=[]
    def check(name,ok,evidence,method='artifact recheck',amended=False):
        tests.append(dict(test=name,status=('AMENDED_USER_APPROVED' if amended else 'PASS') if bool(ok) else 'FAIL',evidence=evidence,method=method))
    bindings=read('INPUT_BINDINGS.json');changed=[]
    for i,b in enumerate(bindings['files']):
        if C.sha(C.ROOT/b['path'])!=b['sha256']:changed.append(b['path'])
        if i%1500==0:print('FINAL_UPSTREAM_HASH',i,len(bindings['files']),flush=True)
    check('test_head_recorded',len(bindings['head'])==40,'INPUT_BINDINGS.json')
    check('test_input_hashes',not changed,f"{len(bindings['files'])} upstream hashes; changed={changed}")
    check('test_existing_results_unchanged',verify_science()==35,'FINALIZATION_INPUT_LOCK.json: 35 completed scientific artifacts; all original upstream hashes also checked')
    prep=read('PREPARATION_TESTS.json');selectionaudit=read('HARD_SELECTION_AUDIT.json')
    C.verify(selectionaudit['previous_full_queue_MAD_audit'])
    inventory=read('CANDIDATE_POOL_AUDIT.json');C.verify(inventory['private_inventory'])
    rows=C.read(C.ROOT/inventory['private_inventory']['path'])['rows'];ex=C.read(C.RAW/'EXCLUSION_IDENTITIES_PRIVATE.json')
    sha={r['image']['sha256'] for r in rows};forbidden={b['sha256'] for b in ex['protected_images']+ex['current_training_images']+ex['historical_pool_images']}
    check('test_adaptation_pool_only',prep['tests']['adaptation_pool_only'] and set(inventory['eligible_recordings'])<=set(inventory['source_recordings']),'PREPARATION_TESTS.json + CANDIDATE_POOL_AUDIT.json','historical inventory audit + current manifest')
    for name in ('eval','anchor','clean10','h10','final_reserved'):
        check(f'test_{name}_excluded',not sha&forbidden and not {r['recording'] for r in rows}&set(ex['heldout_recordings']+ex['reserved_recordings']),'EXCLUSION_IDENTITIES_PRIVATE.json; SHA + entire heldout/reserved recordings')
    check('test_sha_overlap_zero',not sha&forbidden and len(sha)==len(rows),'current eligible pool vs protected SHA sets')
    check('test_near_duplicate_zero',prep['tests']['full_queue_near_duplicate_zero_against_protected_and_each_other'] and inventory['near_duplicate_overlap_after_exclusion']==0,'PREPARATION_TESTS.json: full queue368 MAD audit; bound by HARD_SELECTION_AUDIT.json','verified historical full-queue MAD audit; not claimed rerun')
    queue=C.queue();tags=read('DIFFICULTY_TAG_LOCK.json');sel=read('HARD_SELECTION_LOCK.json');C.verify(tags['snapshot']);C.verify(sel['private_selection'])
    check('test_queue_built_without_model_outputs',inventory['model_outputs_opened']==inventory['coordinate_GT_opened']==0,'CANDIDATE_POOL_AUDIT.json','historical sampling provenance')
    modules=[]
    for name in ('prepare.py','policy.py','tag_difficulty.py'):
        tree=ast.parse((C.ROOT/'scripts/research'/C.NAME/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):modules.extend(a.name for a in node.names)
            if isinstance(node,ast.ImportFrom):modules.append(node.module or '')
    check('test_gui_no_model_overlay',not any(any(k in m for k in ('torch','ultralytics','teacher','evaluate')) for m in modules),'raw difficulty GUI import graph; later annotation PnP is a separately approved amendment','static imports + historical GUI smoke')
    check('test_fixed_round_order',queue==policy.temporal_queue(rows) and all(r['round']==r['bin']%3+1 for r in queue),'queue reconstruction from frozen eligible pool')
    ll=read('HARD_LABEL_LOCK.json');tl=read('TEACHER_HARD_PREDICTION_LOCK.json');rl=read('RAW_PREDICTIONS_LOCK.json');pl=read('POSE_DECISIONS_LOCK.json');sc=read('SCORING_START.json')
    check('test_human_tags_locked_before_model_open',tags['created_at']<sel['created_at']<ll['created_at']<tl['created_at'] and tags['model_outputs_opened']==0,'tag -> selection -> label -> teacher locks')
    selected=C.read(C.ROOT/sel['private_selection']['path'])['rows'];initial=[r for r in selected if r['assignment']=='INITIAL']
    responses=C.read(C.ROOT/tags['snapshot']['path']);check('test_hard_only_from_human_tags',all(r['tag'] in ('MODERATE','SEVERE') and responses[r['frame_id']]['tag']==r['tag'] for r in selected),'locked human tags, not prediction scores')
    check('test_three_recordings',len({r['recording'] for r in initial})>=3,'HARD_SELECTION_LOCK.json')
    check('test_max3_per_recording',max(Counter(r['recording'] for r in selected).values())<=3,'initial + reserve max3')
    check('test_initial8_reserve2',len(initial)==8 and len(selected)==10,'HARD_SELECTION_LOCK.json')
    check('test_selection_locked_before_teacher_or_student_output',sel['created_at']<ll['created_at']<tl['created_at']<rl['created_at'],'ordered immutable locks')
    C.verify(ll['labels']);frames=C.read(C.ROOT/ll['labels']['path'])['frames'];amend=read('ANNOTATION_PROTOCOL_AMENDMENT.json');C.verify(amend['approval']);C.verify(amend['point_confirmation'])
    valid=all(not any(validate_frame(f)) for f in frames.values())
    check('test_manual_bbox_present_role_confident',valid and all(f['bbox_source']=='PNP_PROJECTED_CORNERS_CLIPPED' for f in frames.values()),'ANNOTATION_PROTOCOL_AMENDMENT.json: user approved shared PnP association box instead of manual visible-envelope bbox',amended=True)
    check('test_visible_xy_required',valid and all(p['xy'] is not None for f in frames.values() for p in f['corners'] if p['status']=='DIRECT_VISIBLE'),'private frozen labels; range/duplicate/LR/TB QA')
    check('test_nonvisible_xy_none',all(p['xy'] is None for f in frames.values() for p in f['corners'] if p['status']!='DIRECT_VISIBLE'),'private frozen labels; unobserved physical visibility not inferred')
    check('test_role_uncertain_excluded',coverage(frames,selected)['usable_frames']==8 and all(f['role']=='ROLE_CONFIDENT' for f in frames.values()),'user confirmation; ROLE_UNCERTAIN exclusion also covered by unit test')
    check('test_P8_not_primary',all(len(f['corners'])==8 for f in frames.values()),'labels contain corners0..7 only; tensor mask rechecked below')
    check('test_no_PnP_completion',all(p['source']=='manual_click' for f in frames.values() for p in f['corners'] if p['xy'] is not None),'User enabled PnP display/fill, but no completed corner enters manual support; original PnP-free UI condition amended',amended=True)
    public=['HARD_LABEL_LOCK.json','HARD_PROVENANCE_PUBLIC.json','HARD_COVERAGE_PUBLIC.json']
    check('test_public_no_private_coordinates',all('"xy"' not in (C.DOC/n).read_text() for n in public),'public label provenance/coverage vs private coordinate snapshot')
    check('test_teacher_after_label_lock',ll['created_at']<tl['created_at'],'HARD_LABEL_LOCK.json / TEACHER_HARD_PREDICTION_LOCK.json')
    C.verify(tl['R0']);C.verify(tl['refiner']);check('test_teacher_checkpoint_frozen',True,'R0 and refiner hashes rechecked')
    occ=read('TRAIN_OCCURRENCE_LOCK.json');C.verify(occ['occurrences']);plan=C.read(C.ROOT/occ['occurrences']['path'])
    same=True;different=0;auto_ignored=True
    for r in plan:
        if not r['hard']:continue
        C.verify(r['hard_cache'])
        with np.load(C.ROOT/r['hard_cache']['path']) as z:
            a,b=z['H_MANUAL'],z['H_PSEUDO'];same &= np.array_equal(a[...,2],b[...,2]);different+=int(not np.array_equal(a[...,:2],b[...,:2]));auto_ignored &= bool(np.all(a[:,8,2]==1))
            assert z['img'].shape==(3,640,640) and np.isfinite(z['bboxes']).all()
    for name in ('test_same_support_manual_pseudo','test_manual_pseudo_same_mask'):
        check(name,same and auto_ignored,'all320 shared hard-cache masks + P8 true-ignore')
    for name in ('test_same_bbox_manual_pseudo','test_manual_pseudo_same_aug'):
        check(name,same,'one shared RGB/box/affine cache per hard occurrence, not separate arm transforms')
    pair=read('PAIR_INTEGRITY_MANUAL_VS_PSEUDO.json');fits={a:read(f'FIT_{a}.json') for a in ('H_MANUAL','H_PSEUDO')}
    args={a:yaml.safe_load((C.RAW/'runs'/a/'args.yaml').read_text()) for a in fits}
    check('test_same_init_hash',all(f['initial_state_verified'] for f in fits.values()),'train on_start compared every initial-state digest to original S1','saved runtime assertion')
    check('test_same_trainable_inventory',fits['H_MANUAL']['trainable_inventory']==fits['H_PSEUDO']['trainable_inventory'],'actual named trainable inventories')
    check('test_same_optimizer',pair['same_optimizer_args'] and all(a['optimizer']=='AdamW' and not a['resume'] for a in args.values()),'same args and fresh optimizer construction; no separate initial optimizer-state snapshot was saved','runtime configuration + implementation')
    check('test_seed42',all(a['seed']==42 for a in args.values()),'actual args.yaml')
    check('test_320_updates',all(f['steps']==320 for f in fits.values()),'optimizer post-step hooks in FIT records')
    check('test_5epochs',all(len(f['epochs'])==5 for f in fits.values()),'FIT records')
    for name,kind,n in [('test_512synthetic_per_epoch','SYNTH',512),('test_hard64_per_epoch','HARD',64),('test_clean448_per_epoch','CLEAN',448)]:
        check(name,all(sum(r['kind']==kind for r in plan if r['epoch']==e)==n for e in range(5)),'frozen5120 occurrence plan + completion actual trace audit')
    check('test_512real_per_epoch',all(sum(r['real'] for r in plan if r['epoch']==e)==512 for e in range(5)),'frozen occurrence plan')
    check('test_hard_total320',sum(r['hard'] for r in plan)==320,'frozen occurrence plan')
    for name in ('test_synthetic_replay_identical','test_manual_pseudo_same_RGB'):
        check(name,pair['passed'] and pair['unchanged_original_S1']==4800,'completion audit compares every actual RGB/box/target trace digest')
    check('test_manual_pseudo_coordinate_source_only_difference',same and different==320 and pair['same_RGB_box'],'all320 target arrays plus actual5120 trace audit')
    check('test_last_checkpoint_only',all(f['checkpoint_selection']=='last' and f['checkpoint']['path'].endswith('/last.pt') for f in fits.values()),'FIT/checkpoint paths')
    from pathlib import Path
    validation=[]
    for a in args.values():
        dataset=yaml.safe_load(Path(a['data']).read_text());validation.extend(Path(dataset['val']).read_text().splitlines())
    validation_ok=len(validation)==64 and all('/datasets/g38_legacy_v1v2_p0_tex20k/images/val/' in p for p in validation)
    check('test_no_real_DEV_validation',all(a['val'] is False for a in args.values()) and validation_ok,'val=False; both actual val lists checked: 32 synthetic images each; no DEV; last checkpoint only','actual args + validation-list paths')
    check('test_RAW_lock_before_GT',rl['created_at']<sc['created_at'],'raw lock before first scoring-stage reference read')
    check('test_POSE_lock_before_GT',pl['created_at']<sc['created_at'],'pose lock before scoring-stage reference read')
    C.verify(pl['scorer']);result=read('RESULTS.json')
    check('test_same_frozen_GEO_LINEAR_all_arms',result['same_frozen_GEO_LINEAR'],'single frozen scorer hash rechecked')
    check('test_D9_supplement_only',all('D9' in v for arms in result['groups'].values() for v in arms.values()),'RESULTS.json: separate D9 metrics; primary is current')
    for name,g,n in [('test_clean29','CLEAN',29),('test_moderate21','MODERATE',21),('test_severe78','SEVERE',78)]:
        check(name,all(v['twoD']['total_frames']==n for v in result['groups'][g].values()),'fixed DEV group counts')
    anchor=read('VERIFIED_VISIBLE.json');C.verify(anchor['binding']);check('test_anchor_FINAL_V2',anchor['reference']=='VERIFIED_VISIBLE_ANCHOR_FINAL_V2','verified anchor hash + reference_version')
    check('test_oracle_posthoc_only',not pl['oracle_used'] and not pl['GT_input'] and result['oracle']=='GT POSTHOC NONDEPLOYABLE','pose locks and scoring contract')
    dec=read('DECISION.json');protocol=read('PROTOCOL_LOCK.json')
    check('test_primary_BASE_vs_HMANUAL',all(np.isclose(dec['deltas'][g]['current'],result['groups'][g]['H_MANUAL']['current']['ADDsym_AUC']-result['groups'][g]['BASE']['current']['ADDsym_AUC']) for g in dec['deltas']),'decision deltas independently recomputed')
    check('test_decision_rule_predeclared',protocol['created_at']<ll['created_at'] and 'LOCALIZATION_SELECTOR_LIMIT' in protocol['decision_rules'],'original immutable PROTOCOL_LOCK; named case maps to frozen DECISION')
    check('test_no_more_labeling_on_no_gain',dec['stop_more_hard_labeling'] and len(frames)==8,'stop_more_hard_labeling; reserves not activated')
    check('test_no_threshold_sweep',set(p.name for p in (C.RAW/'runs').iterdir() if p.is_dir())==set(fits),'only the two fixed arm run directories; seed42/5epoch/320updates','run inventory + fixed args; no broader historical claim')
    suite=unittest.defaultTestLoader.loadTestsFromNames(['scripts.research.pallet_min_hard_ab_v1.test_workflow','scripts.research.pallet_min_hard_ab_v1.test_resume'])
    runner=unittest.TextTestRunner(verbosity=1).run(suite)
    for fig in FIGURES:assert (C.DOC/'figures'/fig).is_file(),fig
    for b in read('FIGURE_MANIFEST.json')['figures']:C.verify(b)
    headers=re.findall(r'^## (\d+)\.',(C.DOC/'REPORT_KO.md').read_text(),re.M)
    assert headers==[str(i) for i in range(1,17)]
    required=read('DIRECTIVE_REQUIREMENTS.json')['required_checks']
    assert len(tests)==len(required) and {r['test'] for r in tests}==set(required),('Missing named checks',set(required)-{r['test'] for r in tests})
    status=Counter(r['status'] for r in tests)
    value=dict(created_at=C.now(),status='PASS_WITH_USER_APPROVED_AMENDMENTS' if not status['FAIL'] and runner.wasSuccessful() else 'FAIL',tests=tests,
               counts=dict(status),unit_tests=runner.testsRun,unit_tests_passed=runner.wasSuccessful(),upstream_hashes=len(bindings['files']),
               figures=len(FIGURES),new_training=False,new_annotations=False,new_evaluation=False,
               optional_bootstrap='NOT_RUN (optional; not a completion blocker)',
               evidence_limit='Historical GUI/inventory and training assertions are explicitly identified; no retrospective full file-access trace or independent manual-GT accuracy guarantee.')
    C.save(C.DOC/'DIRECTIVE_AUDIT.json',value)
    lines=['# 지시문 항목별 최종 검증','',f"상태: **{value['status']}**. {len(tests)}개 명명 항목: {dict(status)}. 단위검사 {runner.testsRun}개.",'',
           'PASS는 아래 기재한 증거 범위의 통과다. 과거 실행의 모든 파일 접근을 소급 추적했다는 뜻은 아니다. 사용자 승인으로 바뀐 수동 bbox·PnP 조건은 원문 그대로 통과했다고 표시하지 않는다.','',
           '|원문 검사명|상태|검증 방법|근거|','|---|---|---|---|']
    lines.extend(f'|{r["test"]}|{r["status"]}|{r["method"]}|{r["evidence"]}|' for r in tests)
    lines+=['','## 별도 마무리 확인','',f'- 지시문 지정 그림14개 존재 및 manifest 기록.','- REPORT_KO.md 1~16절 + 부록에 표/이미지/해석 정리.','- 원래 실험 결과·checkpoint·raw prediction·pose decision35개 보존.','- CLI final output 항목 및 중간 상태 resume 회귀검사.','- 추가 학습0, 추론0, 어노테이션0.','- 선택 항목인 bootstrap 미실행; 유의성 주장 없음.','', '[상세 JSON](DIRECTIVE_AUDIT.json) · [재개 테스트](RESUME_TESTS.json) · [그림 manifest](FIGURE_MANIFEST.json)']
    C.save(C.DOC/'DIRECTIVE_CHECKLIST.md','\n'.join(lines)+'\n')
    assert value['status']!='FAIL',value['counts']
    from .status_output import render
    C.save(C.DOC/'FINAL_CLI_OUTPUT.txt',render()+'\n')
    print('DIRECTIVE_AUDIT',len(tests),dict(status),flush=True)


if __name__=='__main__':main()
