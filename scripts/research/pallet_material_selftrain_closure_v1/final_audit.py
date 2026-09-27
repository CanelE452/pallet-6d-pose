"""Execution and paper audits; no new fitting, selection or scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from unittest.mock import patch
import numpy as np
from . import common as C
from .infer_eval import verify_final_lock
from .train_pair import parity_signature,assert_pair
from .wood_inventory_audit import overlap

def build():
    from scripts.research.pallet_selftraining_paper_closure_v1 import package
    with patch.object(package.C,'DOC',C.DOC),patch.object(package.C,'RAW',C.RAW):package.build()

def main():
    tests=['scripts/research/pallet_material_selftrain_closure_v1/test_pair.py',
           'scripts/research/pallet_material_selftrain_closure_v1/test_eval.py',
           'scripts/research/pallet_type_selftrain_v1/test_contract.py',
           'scripts/research/pallet_type_selftrain_v1/test_recovery_pose.py']
    run=subprocess.run([sys.executable,'-m','pytest','-q',*tests],cwd=C.ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    C.save(C.RAW/'TEST_OUTPUT.txt',run.stdout);assert run.returncode==0,run.stdout
    count=int(re.search(r'(\d+) passed',run.stdout).group(1))
    checks={};add=lambda k,v:checks.__setitem__(k,bool(v))
    for b in C.read(C.DOC/'INPUT_BINDINGS.json')['files']:C.verify(b)
    add('original_core_inputs_unchanged',True)
    inv=C.read(C.RAW/'WOOD_INVENTORY_AGENT.json');accepted=C.read(C.RAW/'WOOD_ACCEPTED_SHARED.json')
    evals=inv['proposed_main_evaluation_records'];teacher=inv['teacher_records']
    accepted_overlap=overlap(accepted,evals);teacher_overlap=overlap(teacher,evals)
    add('train_eval_exact_ID_SHA_recording_overlap_zero',all(not v for v in accepted_overlap.values()))
    add('teacher_eval_exact_ID_SHA_recording_overlap_zero',all(not v for v in teacher_overlap.values()))
    add('teacher_train_recording_overlap_disclosed',set(r['recording'] for r in accepted)==set(r['recording'] for r in teacher))
    protocol=C.read(C.DOC/'WOOD_TRAIN_PROTOCOL.json')
    for b in protocol['inputs']+protocol['sources']+[protocol['initialization'],protocol['preflight']]:C.verify(b)
    sig={t:parity_signature((C.ROOT/protocol['datasets'][t]['train_list']['path']).read_text().splitlines()) for t in ('RAW','REF')}
    assert_pair(sig['RAW'],sig['REF']);add('RAW_REF_RGB_order_support_box_source_parity',True)
    fits=[C.read(C.DOC/f'FIT_{a}.json') for a in ('WOOD_RAW_LR5','WOOD_REF_LR5')]
    for fit in fits:C.verify(fit['checkpoint']);C.verify(fit['protocol'])
    add('two_new_fits_only',len(list((C.RAW/'runs').iterdir()))==2)
    add('each320updates',all(f['optimizer_steps']==320 for f in fits))
    add('same_R0_initialization_and_protected_state_exact',all(f['exact_R0_initialization'] and f['protected_state_exact'] for f in fits))
    add('training_did_not_open_evaluation_labels',all(f['evaluation_labels_opened'] is False for f in fits))
    add('last_only',all(f['checkpoint_selection']=='final last.pt only' for f in fits))
    add('true_ignore_loss_gradient',C.read(C.DOC/'WOOD_TRUE_IGNORE_TRAIN_FIXTURE_TEST.json')['status']=='PASS')
    lock=verify_final_lock();res=C.read(C.DOC/'WOOD_RESULTS.json')
    add('predictions_and_common_D9_locked_before_scoring',C.read(C.DOC/'WOOD_SCORING_START.json')['utc']>=lock['created_at'] and res['raw_and_pose_frozen_before_reference_scoring'])
    add('detector_order_boxes_scores_exact',lock['detector_order_boxes_scores_exact'])
    add('teacher_confidence_center_preserved',lock['teacher_confidence_preserved'] and lock['teacher_center_preserved'])
    add('same_D9_deployable_contract','no oracle' in res['pose_contract'])
    add('full_denominator346_all45',all(r['twoD']['corners']==346 and r['sixD']['frames']==45 for r in res['groups']['ALL'].values()))
    add('severe_not_fabricated','SEVERE' not in res['groups'] and not res['severity_available']['SEVERE'])
    add('trusted_wood_Q1_unresolved',res['Q1_WOOD']=='UNRESOLVED')
    add('reused_DEV_not_independent',res['independent_confirmation'] is False)
    for b in res['artifact_sources']:C.verify(b)
    figure=C.read(C.DOC/'FIGURE_MANIFEST.json')
    for b in figure['files']+figure['source_artifacts']:C.verify(b)
    add('figure_results_hash_trace',True)
    # Preserve every existing original table and prior bounded-test source.
    old_tables=subprocess.check_output(['git','ls-tree','-r','--name-only',C.OLD_START,str(C.PAPER.relative_to(C.ROOT)/'generated_tables')],text=True).splitlines()
    unchanged=True
    for rel in old_tables:
        original=subprocess.check_output(['git','show',f'{C.OLD_START}:{rel}'])
        unchanged &= original==(C.ROOT/rel).read_bytes()
    add('original_Plastic_tables_retained_exact',unchanged)
    text=(C.PAPER/'manuscript.tex').read_text();alltext=text+'\n'+(C.PAPER/'material_extension.tex').read_text()
    from .material_report import load_completed,make_tables
    for name,table in make_tables(load_completed()).items():
        for suffix,expected in zip(('.md','.tex'),table['texts']):
            assert (C.DOC/(name+suffix)).read_text()==expected,(name,suffix)
    add('material_tables_recomputed_from_bound_JSON',True)
    numerical_map={}
    for arm in ('R0','WOOD_RAW_LR5','WOOD_REF_LR5'):
        r=res['groups']['ALL'][arm]
        for key,formatted in [('PCK10',f"{100*r['twoD']['PCK']['10']:.2f}"),('AUC',f"{r['sixD']['ADDsym_AUC']:.5f}")]:
            assert formatted in alltext,(arm,key,formatted)
            numerical_map[arm+'.'+key]=formatted
    C.save(C.DOC/'MANUSCRIPT_MATERIAL_NUMBERS.json',dict(source=C.bind(C.DOC/'WOOD_RESULTS.json'),
        mapping=numerical_map,material_tables=C.bind(C.DOC/'MATERIAL_NUMBER_PROVENANCE.json'),
        semantics='PCK10 fraction multiplied100; D9 AUC unitless; originalPlastic macros unchanged'))
    add('manuscript_primary_numbers_trace_to_JSON',True)
    add('historical_experiments_not_main_controls','historical material-routed' in alltext)
    add('material_routing_disclosed','externally provided' in alltext)
    add('no_universal_or_physical_accuracy_claim','not independently measured' in text and 'reused' in alltext)
    build_result=C.read(C.DOC/'BUILD_RESULT.json');C.verify(build_result['pdf'])
    add('build_no_undefined_missing_overfull',build_result['complete'] and not build_result['warnings_to_inspect'])
    pdftext=(C.RAW/'manuscript.txt').read_text()
    add('no_placeholders',not re.search(r'TODO|PLACEHOLDER|\?\?|\[xx\]',pdftext))
    add('all_30_contract_tests_pass',count>=30)
    status='PASS' if all(checks.values()) else 'FAIL'
    C.save(C.DOC/'MATERIAL_PAPER_AUDIT.json',dict(status=status,checks=checks,tests_passed=count,test_output=C.bind(C.RAW/'TEST_OUTPUT.txt'),
        overlaps=dict(accepted_vs_eval=accepted_overlap,teacher_vs_eval=teacher_overlap),
        protocol=C.bind(C.DOC/'WOOD_TRAIN_PROTOCOL.json'),prediction_lock=C.bind(C.DOC/'WOOD_PREDICTIONS_LOCK.json'),
        results=C.bind(C.DOC/'WOOD_RESULTS.json'),build=C.bind(C.DOC/'BUILD_RESULT.json'),
        manuscript=C.bind(C.PAPER/'manuscript.tex'),material_section=C.bind(C.PAPER/'material_extension.tex'),
        limitation='Code/recorded execution audits support isolation; no full per-step augmented-tensor trace or independent physical pose proof claimed.',
        independent_code_review='Three issues fixed before affected computation: recording_group metadata allowlist, teacher TF32 restoration, unwantedcommon-support feasibilitygate (Plastic had4/5point images).',
        no_more_method_development=True))
    C.save(C.DOC/'MATERIAL_PAPER_AUDIT_KO.md','# Material 최종 감사\n\n'+f'상태: {status}. 계약 테스트 {count}개 통과.\n\n'+C.table(['검사','결과'],[[k,'PASS' if v else 'FAIL'] for k,v in checks.items()])+'\n교사/학생-평가 ID·SHA·recording 중복0. 보고서는 legacy reference 범위이며 독립 물리6D·모든material일반성은 미검증이다. 원본Plastic표를 삭제하거나 새640update결과로 교체하지 않았다. 전체학습텐서 전수동일이라는 주장은 하지 않는다.\n')
    print('MATERIAL_AUDIT',status,count,[k for k,v in checks.items() if not v],flush=True)
    assert status=='PASS'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['build','audit']);a=p.parse_args()
    build() if a.phase=='build' else main()
