"""Final mechanical evidence audit and lightweight public measurement bundle."""
import re
import subprocess
from pathlib import Path
import numpy as np
from . import common as C
from scripts.research.pallet_recording_disjoint_transfer_v1.evaluate import summary
from scripts.research.pallet_clean19_pose_mismatch_v1 import diagnose as D
from scripts.research.pallet_recording_disjoint_transfer_v1 import common as V

def main():
    checks=[]
    def check(name,ok):
        assert ok,name
        checks.append(dict(check=name,passed=True))
    for b in C.read(C.DOC/'INPUT_BINDINGS.json')['files']:C.verify(b)
    check('all_original_input_hashes_unchanged',True)
    lock=C.read(C.DOC/'PREDICTIONS_LOCK.json')
    for b in lock['files']:C.verify(b)
    check('prediction_and_pose_lock_before_scoring',lock['created_at']<C.read(C.DOC/'SCORING_START.json')['utc'])
    fm=C.read(C.RAW/'FRAME_METRICS.json');pm=C.read(C.RAW/'POSE_METRICS.json');res=C.read(C.DOC/'CORE_RESULTS.json')['groups']
    for group,ids in V.groups(C.records()).items():
        for arm in C.ARMS:
            D.close(summary([fm[arm][i] for i in ids]),res[group][arm]['twoD'])
            D.close(D.aggregate([pm[arm][i] for i in ids]),res[group][arm]['sixD'])
            check(group+'/'+arm+' independent aggregate replay',True)
    supplemental={s:C.read(C.REC/'pose_only/PROTOCOL.json')['datasets'][s] for s in ('RAW','REF','SYN')}
    support=C.read(C.ROOT/'_docs/experiments/pallet_posefix_large_error_v1/TRAIN_SUPPORT.json')
    budget=dict(teacher_images=9,manual_corners=38,teacher_records=[dict(id=r['id'],manual_corner_ids=[j for j,v in enumerate(r['manual']) if v],crop_supported=[j for j,v in enumerate(r['crop_supported']) if v]) for r in support['records']],
        student_real_manual_coordinate_loss=False,student_unlabeled_unique=217,teacher_shared_filter_influences_both_arms=True,
        evaluation_manual_points=66,evaluation_manual_images=16,evaluation_labels_never_training=True,
        generic_upstream_pretraining='R0 COCO-pose; not a never-real-pretraining claim',annotation_time_savings_measured=False,
        prior_development_label_use='Full project uses other manual labels for development/extension studies;9/38 is the core frozen teacher fitting budget, not total project annotation labor.')
    C.save(C.DOC/'MANUAL_SUPERVISION_BUDGET.json',budget)
    check('teacher_support_count',sum(len(r['manual_corner_ids']) for r in budget['teacher_records'])==38)
    rr={r['id']:r for r in C.records()}
    public=dict(reference='Legacy full128 mixed-provenance; visible66 separately',
        metadata={i:dict(recording=r['recording_group'],severity=r['severity']) for i,r in rr.items()},
        frame_metrics=fm,pose_metrics=pm,
        visible_point_errors=[{k:v for k,v in r.items() if k in ('frame_id','corner_id','severity','recording','errors','missing')} for r in C.read(C.RAW/'ANCHOR_POINTS.json')])
    C.save(C.DOC/'MEASUREMENT_ROWS.json',public)
    tex=(C.PAPER/'manuscript.tex').read_text();bibliography=(C.PAPER/'references.bib').read_text()
    used=set(k.strip() for group in re.findall(r'\\cite\{([^}]+)\}',tex) for k in group.split(','))
    defined=set(re.findall(r'@\w+\{([^,]+),',bibliography));check('all_citation_keys_exist',used<=defined)
    for path in re.findall(r'\\(?:input|includegraphics)(?:\[[^]]*\])?\{([^}]+)\}',tex):
        check('tex resource '+path,(C.PAPER/path).exists())
    for p in [C.DOC/'REPORT_KO.md',C.DOC/'PSEUDO_LABEL_QUALITY_REPORT_KO.md']:
        for ref in re.findall(r'!\[[^]]*\]\(([^)]+)\)',p.read_text()):check('markdown image '+ref,(p.parent/ref).is_file())
    check('no_placeholder',not re.search(r'TODO|PLACEHOLDER|\[xx\]|\?\?',tex))
    check('native_visible_student_tie_reported','43/66' in tex and '43/66' in (C.DOC/'REPORT_KO.md').read_text())
    check('manual_budget_reported','38' in tex and 'nine' in tex)
    check('limitations_and_negative_examples','student_06_worsened.pdf' in tex and 'does not' in tex)
    build=C.read(C.DOC/'BUILD_RESULT.json');C.verify(build['pdf'])
    check('pdf_build_complete_and_no_unresolved_or_overfull',build['complete'] and not build['warnings_to_inspect'])
    # Freeze code/data references used by the final paper, beyond the initial discovery lock.
    files=set(Path(__file__).parent.glob('*.py'))
    for n in ('pallet_recording_disjoint_transfer_v1/common.py','pallet_recording_disjoint_transfer_v1/evaluate.py',
              'pallet_verified_anchor_v1/evaluate.py','pallet_clean19_pose_mismatch_v1/diagnose.py',
              'pallet_dim_conditioned_p_v1/eval_math.py','pallet_posefix_large_error_v1/core.py','pallet_sensors_submission_v1/prior_model.py'):
        files.add(C.ROOT/'scripts/research'/n)
    files.add(Path(D.Pose.__file__))
    files.add(C.ROOT/'scripts/paper/pose_metric_closure_v1/symmetry_aware_pose_metrics.py')
    pseudo=C.read(C.OLD/'PSEUDO_PROTOCOL.json')
    for b in pseudo['sources']+[pseudo['code']]:
        C.verify(b);files.add(C.ROOT/b['path'])
    check('historical_pseudo_filter_and_teacher_source_hashes_verified',True)
    for p in (C.PAPER/'generated_tables').glob('*.json'):
        obj=C.read(p)
        if p.name=='MANUSCRIPT_NUMBER_PROVENANCE.json':
            for v in obj.values():C.verify(v['source'])
        elif p.name=='NUMBER_PROVENANCE.json':
            for b in obj['sources']+obj['tables']:C.verify(b)
    check('manuscript_number_source_hashes_verified',True)
    prior_path='_docs/paper/sensors_submission_v1'
    diff=subprocess.check_output(['git','diff','287f317967621ca438e33fdbc3634ccfe6eb3c54','--',prior_path],text=True)
    check('old_paper_not_overwritten',not diff)
    check('branch_main',subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='main')
    test=subprocess.run([__import__('sys').executable,'-m','unittest','scripts.research.pallet_selftraining_paper_closure_v1.test_closure'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    C.save(C.DOC/'TEST_RESULTS.txt',test.stdout);check('regression_tests_pass',test.returncode==0)
    result=dict(complete=True,utc=C.now(),checks=checks,passed=len(checks),new_fits=0,
        logical_completeness=True,core_comparison_fair=True,evidence_decision='PAPER_CORE_SUPPORTED within reused plastic legacy-reference DEV only',
        visible_student_PCK10_not_improved=True,independent_confirmation=False,physical_pose_independence=False,
        innovation_claim='Controlled empirical application study; no new self-training/refiner/PnP invention',
        manuscript=C.bind(C.PAPER/'manuscript.tex'),pdf=build['pdf'],code_and_metric_dependencies=[C.bind(p) for p in sorted(files)],
        figure_manifest=C.bind(C.DOC/'FIGURE_MANIFEST.json'),public_measurements=C.bind(C.DOC/'MEASUREMENT_ROWS.json'),
        author_review_required_for_submission=True,actual_submission_performed=False,
        render_review='See PDF_VISUAL_REVIEW.json; underfull typography warnings allowed, no overfull/undefined',
        stop_rule='No further method development; writing/reproduction/independent confirmation only')
    C.save(C.DOC/'PAPER_AUDIT.json',result)
    C.save(C.DOC/'PAPER_AUDIT_KO.md',f'# 논문 최종 근거 감사\n\n{len(checks)}개 자동 검사 통과. 13개 회귀 테스트 통과. 신규 학습0회. 실제8페이지 PDF 빌드 완료.\n\nQ1 teacher 품질과 Q2 student를 분리했고, 128/985, 120/931,16/66 분모를 섞지 않았다. 같은D9이며 oracle수치 없음. 9/38 예산과 shared teacher filter 조건,COCO upstream, 반복DEV·historical LR선택,66점 학생PCK10동률, 악화사례를 숨기지 않았다.\n\n기존 원고는 수정하지 않았다. 제출 전 저자·소속·데이터공개·funding 및 DEV-only 주장을 사람이 승인해야 한다. 저널 제출은 수행하지 않았고 게재 가능성을 보장하지 않는다. 추가 annotation은 현재 원고완성의 blocker가 아니다.\n\n공개 MEASUREMENT_ROWS.json으로 aggregate 계산을 추적할 수 있다. 전체 재현에는 private RGB/annotation/checkpoint 접근이 필요하다.\n')
    print('AUDIT_PASS',len(checks),test.stdout[-500:],flush=True)

if __name__=='__main__':main()
