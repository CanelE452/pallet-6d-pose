"""Audit historical paired experiments and freeze scope BEFORE new scoring."""
from collections import Counter
from pathlib import Path
import subprocess
import numpy as np
from . import common as C

def main():
    if (C.DOC/'PAPER_CLOSURE_PROTOCOL.json').exists():
        for b in C.read(C.DOC/'INPUT_BINDINGS.json')['files']: C.verify(b)
        print('ALREADY_LOCKED'); return
    rr=C.records(); ids={r['id'] for r in rr}; hashes={r['image']['sha256'] for r in rr}; recs={r['recording_group'] for r in rr}
    assert len(ids)==128
    p=C.read(C.REC/'pose_only/PROTOCOL.json')
    accepted=[r for r in C.read(C.CACHE/'PSEUDO_ACCEPTED.json') if r['kind']=='PLASTIC']
    assert len(accepted)==249
    files={C.SPLIT,C.FINAL,C.TRUTH,C.META,C.REC/'RECOVERY_AUDIT.json',C.OLD/'POOL.json',C.OLD/'DATA_VERIFICATION.json',C.CACHE/'PSEUDO_ACCEPTED.json',C.OLD/'PSEUDO_PROTOCOL.json'}
    paired={}; train_ids=set(); supports=Counter()
    for target in ('RAW','REF'):
        dataset=p['datasets'][target]
        paired[target]=[Path(x) for x in (C.ROOT/dataset['train_list']['path']).read_text().splitlines()]
        files.add(C.ROOT/dataset['train_list']['path'])
    assert [x.name for x in paired['RAW']]==[x.name for x in paired['REF']]
    changes=0
    for a,b in zip(paired['RAW'],paired['REF']):
        assert C.sha(a)==C.sha(b)
        la=a.parent.parent/'labels'/a.with_suffix('.txt').name
        lb=b.parent.parent/'labels'/b.with_suffix('.txt').name
        aa=np.array(la.read_text().split(),float);bb=np.array(lb.read_text().split(),float)
        assert np.array_equal(aa[:5],bb[:5])
        qa=aa[5:].reshape(-1,3);qb=bb[5:].reshape(-1,3)
        assert np.array_equal(qa[:,2],qb[:,2])
        ignored=qa[:,2]!=2; assert np.array_equal(qa[ignored],qb[ignored])
        files.update([la,lb,a.resolve()])
        if a.name.startswith('PLASTIC__'):
            train_ids.add(a.stem); supports['exposed_valid_points_per_epoch']+=int((qa[:,2]==2).sum())
            changes+=int(np.any(qa!=qb))
        else: assert np.array_equal(aa,bb)
    assert len(train_ids)==217 and len(paired['RAW'])==1024
    train=[r for r in accepted if r['id'] in train_ids]
    assert not hashes & {r['image']['sha256'] for r in train}
    assert not recs & {r['recording_id'] for r in train}
    teacher=C.ROOT/'_docs/experiments/pallet_posefix_replay_v1'
    support=C.ROOT/'_docs/experiments/pallet_posefix_large_error_v1/TRAIN_SUPPORT.json'
    manual=C.read(support); assert manual['manual_corners']==38
    teacher_ids=C.read(teacher/'INPUT_LOCK.json')['real_ids']
    assert not set(teacher_ids)&ids
    # Explicit teacher recording provenance: three plastic-night points/images,
    # six wood-day images; these sessions are absent from HELDOUT128.
    registry=C.read(C.ROOT/'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json')
    files.add(C.ROOT/'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json')
    teacher_sessions=sorted({x.split(':')[0] for x in teacher_ids})
    assert not set(teacher_sessions)&{r['session'] for r in rr}
    files.update([support,teacher/'INPUT_LOCK.json',teacher/'PROTOCOL.json',teacher/'FIT.json',C.ROOT/C.read(teacher/'FIT.json')['checkpoint']['path'], C.ROOT/p['initialization']['path']])
    fits={}; predcoverage={}
    for arm in C.ARMS:
        path=C.prediction_path(arm); payload=C.read(path)
        predcoverage[arm]=len(ids & {r['id'] for r in payload['records']})
        assert predcoverage[arm]==128
        files.add(path)
        if arm=='R0': continue
        fp=C.REC/C.phase(arm)/f'FIT_{arm}.json'; f=C.read(fp)
        assert f['complete'] and f['optimizer_steps']==320 and f['protected_state_exact'] and f['exact_R0_initialization']
        C.verify(f['checkpoint']);C.verify(f['protocol'])
        fits[arm]={k:f[k] for k in ('checkpoint','optimizer_steps','exact_R0_initialization','protected_state_exact')}
        files.update([fp,C.ROOT/f['checkpoint']['path'],C.ROOT/f['protocol']['path']])
    traces={}
    for order in (43,44):
        pair=[C.REC/'pose_repeat'/f'TRACE_{a}_ORDER{order}.json' for a in ('RAW','REF')]
        aa,bb=map(C.read,pair);assert aa['image_tensor_sha256']==bb['image_tensor_sha256']
        traces[order]=aa['image_tensor_sha256'];files.update(pair)
    assert traces[43]!=traces[44]
    for r in rr: C.verify(r['image']);files.add(C.ROOT/r['image']['path'])
    files.update(Path(__file__).parent.glob('*.py'))
    for n in ('recovery_pose.py','recovery_pose_trainer.py','recovery_repeat.py','pseudo.py','train.py'):
        q=C.ROOT/'scripts/research/pallet_type_selftrain_v1'/n
        if q.exists():files.add(q)
    audit=dict(status='MATCHABLE_BY_REEVALUATION',paired_targets='EXACT_MATCH except supervised xy',
        initialization=p['initialization'],args=p['args'],real_unique=217,accepted=249,pool=1000,
        real_slots_per_epoch=512,synthetic_slots_per_epoch=512,epochs=5,updates=320,
        real_exposures=2560,synthetic_exposures=2560,source_unique=512,source_negatives=0,
        identical_order_images_boxes_masks=True,coordinate_changed_real_slots=changes,support=dict(supports),
        train_recordings=sorted({r['recording_id'] for r in train}),eval_recordings=sorted(recs),
        train_eval_recording_overlap=[],train_eval_RGB_overlap=[],teacher_manual_images=9,teacher_manual_corners=38,
        teacher_sessions=teacher_sessions,teacher_eval_exact_ID_overlap=[],
        teacher_recording_check='Session aliases cross-checked separately against SOURCE_RECORDING_GROUPS; see TEACHER_RECORDING_AUDIT.json',
        fit_records=fits,first_batch_trace_hashes=traces,
        stochasticity='Order43/44 are order repeats, NOT independent initialization seeds. First-batch trace, not a full-run tensor trace.',
        R1_NAIVE_status='NOT_COMPARABLE: different images, SGD,900 updates,1440 source images',
        CLEAN19_status='DIFFERENT_BUT_INTERPRETABLE: different teacher/manual budget; no raw-label student control',
        independent_test=False)
    C.save(C.DOC/'CORE_COMPARABILITY_AUDIT.json',audit,True)
    C.save(C.DOC/'INPUT_BINDINGS.json',dict(files=[C.bind(f) for f in sorted(files)]),True)
    rolemap={'R0':'MAIN_BASELINE','R1_NAIVE/P43/P44':'NOT_COMPARABLE','Replay teacher last300 (9/38)':'MAIN_CORRECTED_PSEUDO',
        'pose_only RAW_LR4/RAW_LR5':'MAIN_RAW_PSEUDO','pose_only REF_LR4/REF_LR5':'MAIN_CORRECTED_STUDENT',
        'pose_repeat RAW/REF ORDER43/44':'MAIN_ORDER_SENSITIVITY','pose_only/repeat SYN':'SOURCE_UPDATE_CONTROL',
        'Clean19 OLD_STUDENT/S0/S1/S2':'EXTENSION_OCCLUSION','H_MANUAL hard8':'EXTENSION_MANUAL_HARD',
        'HMAN_SPECIFIC_GEO_LINEAR':'EXTENSION_SELECTOR','ALL300':'DIAGNOSTIC_ONLY','FINAL_V2 visible66':'MAIN_Q1_FIXED_ID',
        'HELDOUT128':'MAIN_REUSED_DEV','source256':'DIAGNOSTIC_ONLY','old reserves':'PROVENANCE_AUDIT_AFTER_FREEZE'}
    C.save(C.DOC/'EVIDENCE_INVENTORY.json',dict(roles=rolemap,prediction_coverage=predcoverage,prior_experiment_roots=[str(C.REC.relative_to(C.ROOT)),str(teacher.relative_to(C.ROOT))]),True)
    C.save(C.DOC/'EXPERIMENT_ROLE_MAP.md','# Experiment roles\n\n'+C.table(['Artifact family','Role'],rolemap.items()),True)
    C.save(C.DOC/'REUSE_MATRIX.json',dict(new_fits=0,reused_student_fits=12,new_frozen_teacher_inference='HELDOUT128 only if needed',reevaluation='Same128 / fixed common D9 / 2D support unchanged',not_reused=['R1_NAIVE as causal control','Clean19 teacher as quality evidence for old Replay students']),True)
    C.save(C.DOC/'GOAL_LOCK.md','# Paper claim lock\n\nRaw versus corrected pseudo-label self-training under an explicit real supervision budget. Close Q1 label quality, Q2 matched student value, Q3 value over R0, Q4 scope and final6D. Stop new method development; negative results are valid closure. No new refiner, selector, loss, annotation or sweep.\n\nHistorical paired pose-only fits reused; new fits=0. Both original learning-rate strata and all order repeats will be reported, not selected on the current128 results. LR5 is the previously DEV-selected configuration, transparently identified, with LR4 sensitivity. No claim of annotation-cost superiority.\n',True)
    C.save(C.DOC/'CLAIM_MATRIX.json',{f'C{i}':dict(question=q,status='UNRESOLVED') for i,q in enumerate(['corrected label quality','corrected versus raw student','corrected versus R0','final6D','real supervision budget','DEV versus independent'],1)},True)
    C.save(C.DOC/'PAPER_CLOSURE_PROTOCOL.json',dict(created_at=C.now(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        arms=C.ARMS,primary_configuration='Previously selected LR5; selection occurred on reused194 DEV. Report LR4 and both order repeats without dropping any.',
        primary_2D='PCK10 all supported corners, with PCK5/20 and median/P90/tail',main_pose='Fixed original D9 selection + unchanged corner0..7 solver; no oracle selection',
        pseudo_teacher='Replay last300 fc9b3d7b; not Clean19 teacher',Q1='FINAL_V2 visible66 in16 images, fixed native identity, missing diagonal penalty, no GT selection',
        population='HELDOUT128 plastic only: Clean29 Moderate21 Severe78; reused7recordingsDEV',
        new_fits=0,manual_budget=dict(images=9,corners=38),claim_rules='Positive direction descriptive only; mixed6D => conditional scope. If raw/corrected difference unsupported, report NOT_SUPPORTED. Never search for a rescue.',
        independent='Audit only after table/claim freeze; no accessible certified independent set => DEV-only draft, not invented confirmation',
        prior_evidence_read=True,prospective_preregistration=False,stage='LOCKED_BEFORE_NEW_REEVALUATION'),True)
    C.save(C.DOC/'CORE_COMPARABILITY_REPORT_KO.md','# 핵심 비교 공정성\n\n기존 pose-only RAW/REF 비교를 재사용한다. 두 learning rate와 두 order 반복을 전부 보고한다. 새 학습은 0회. 같은 217 RGB, 2560 real/2560 synthetic 노출, 320 update, R0 초기값, 박스·마스크·증강 설정을 공유하고 감독 좌표만 다르다. Order 반복 첫 batch RGB tensor hash도 같다(전체 실행 tensor 추적이라는 뜻은 아님).\n\n보정기는 수동 9장·38점으로 학습했다. Q1도 반드시 그 보정기로 계산하며, Clean19 보정기 수치를 대신 넣지 않는다. R1_NAIVE와 Clean19는 핵심 causal 비교로 섞지 않는다. 128장은 반복 DEV이고 독립 test가 아니다.\n',True)
    print('LOCKED',audit['train_recordings'], 'manual',audit['teacher_sessions'],flush=True)

if __name__=='__main__': main()
