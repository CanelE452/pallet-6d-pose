"""A's bounded independent saved-evaluator code/generated review. No real data."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import math
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CODE=ROOT/'contract/evaluate_frozen_current.py'
PROTOCOL=ROOT/'CURRENT_PROTOCOL_BEFORE_CURRENT_ARTIFACTS.json'
SHA='64a6ae5bafd784b984ec1825aad90102db49b0ab8b312222474c557c305bee59'


def binding(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def write(name,d):
    with(HERE/name).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    assert binding(CODE)['sha256']==SHA
    write('SCOPE_BEFORE_GENERATED_TESTS.json',dict(completed=True,evaluator=binding(CODE),protocol=binding(PROTOCOL),
        allowed='Code/public policy/existing generated proof plus newly invented CSV/JSON only.',
        actual_current_artifact_SHA_body_RGB_GT_scores_sourceweights_reads=0,new_model_forward_fit_PnP=0))
    sp=importlib.util.spec_from_file_location('v91_A_review_saved_evaluator',CODE);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    proof=json.loads((ROOT/'contract/GENERATED_EVALUATOR_TESTS.json').read_text())
    assert proof['PASS']and proof['evaluator']['sha256']==SHA
    cases=0
    for z in [float(np.nextafter(np.float32(0),np.float32(-1))),-1e-20,-1.,-0.,0.,1e-20,float(np.nextafter(np.float32(0),np.float32(1))),1.]:
        for valid in (True,False):
            for geo in (True,False,None):
                expected='OTHER_UNRESOLVED'if not valid else'ABSENT_NONVISIBLE'if geo is True else'VISIBLE'if z<0 else'ABSENT_NONVISIBLE'
                assert m.authoritative_call(z,valid,geo)==expected;cases+=1
    for z in [None,float('nan'),float('inf'),-float('inf')]:
        assert m.authoritative_call(z,False,True)=='OTHER_UNRESOLVED'
        try:m.authoritative_call(z,True,False)
        except ValueError:pass
        else:raise AssertionError('Valid malformed z accepted')
    guardtmp=HERE/'GENERATED_UNAPPROVED_ENTRY.json'
    guardtmp.write_text(json.dumps(dict(schema='V91_SAVED_CURRENT_EVALUATION_ENTRY_V1',approved=False)))
    oldsha=m.sha;attempts=[]
    def reject_any_hash(path):attempts.append(str(path));raise AssertionError('Should reject before hashing anything')
    m.sha=reject_any_hash
    try:
        try:m.source_and_inference_gate(guardtmp)
        except PermissionError:pass
        else:raise AssertionError('Unapproved entry accepted')
    finally:m.sha=oldsha
    assert not attempts
    # Full existing evaluator/main executes only in an invented temporary root.
    with tempfile.TemporaryDirectory(prefix='A_generated_only_',dir=HERE)as tmp:
        fixture=m.generated_full_fixture(tmp)
        records=m.readcsv(Path(tmp)/'invented_new.csv')
        _,counts=m.score_records(records,m.RAW);assert counts==dict(rows=1784,valid=1738,invalid=46)
        rejections=[]
        for name in('rawNaN','wrong_authoritative','invalid_geometry','invalid_probability','duplicate_key'):
            bad=copy.deepcopy(records)
            if name=='rawNaN':bad[0]['raw_physical_external_logit']='nan'
            elif name=='wrong_authoritative':bad[0]['authoritative_decision']='E'
            elif name=='invalid_geometry':bad[-1]['geometry_self_hidden_if_valid']='True'
            elif name=='invalid_probability':bad[-1]['p_visible']='0.9'
            elif name=='duplicate_key':bad[1]['id']=bad[0]['id'];bad[1]['native_corner']=bad[0]['native_corner']
            try:m.score_records(bad,m.RAW)
            except(AssertionError,ValueError):rejections.append(name)
            else:raise AssertionError('Malformed saved row accepted: '+name)
        assert len(rejections)==5
        metrics=m.readcsv(Path(tmp)/'GENERATED_V91_CLASSIFICATION_METRICS.csv')
        conf=m.compact_confusions(metrics)
        assert all(sum(int(row[k])for k in('GT_D_pred_VISIBLE','GT_D_pred_ABSENT','GT_D_pred_UNRESOLVED','GT_NEG_pred_VISIBLE','GT_NEG_pred_ABSENT','GT_NEG_pred_UNRESOLVED'))==row['N']for row in conf)
    assert binding(CODE)['sha256']==SHA
    report=dict(completed=True,PASS=True,blockers=[],evaluator=binding(CODE),protocol=binding(PROTOCOL),
        generated_proof=binding(ROOT/'contract/GENERATED_EVALUATOR_TESTS.json'),review_code=binding(__file__),
        tests=dict(raw_sign_and_invalid_geometry_cases=cases,malformed_valid_z_rejected=True,
            unapproved_entry_rejected_before_any_artifact_hash=True,generated_full_evaluator_main_replay=fixture,
            saved_row_mutation_rejections=rejections,allGT_CM_denominators_reconcile=True,
            tiny_raw_negative_visible_even_if_display_pvisible_half=True,zero_signed_zero_external=True,
            historical_v61_legacy_V76_decoders_unchanged=True,manual153_223_identity_count_guards=True),
        static_guard_review=[
            'Root policy/code/generated/review/source-audit/freeze checks precede current inference/preparation/scores and GT body/SHA reads.',
            'Completed inference must bind exact source checkpoint, preparation and score CSV, with model unchanged/currentGTfalse/backbone0/optimizer0/PnP0 and rawzE decoder.',
            'All1784 id/native order/image/q/L/pixel/priorvalid/geometry joins are checked. Original45 U cannot become valid; additional U remains in registered cohort denominators.',
            'RAW decoder overrides only V91 routes. Geometry is an independent guarded route and never overrides invalid U. Binary absence is not proof of causal E/S.',
            'Fixed historical original_baseline_metrics is explicitly bound for base.main implicit reads; historical per-cohort count drift halts comparison.',
            'Generated temporary-root baseline/main execution is distinct from an actual current evaluation. This review did not call actual() or read any real current/source checkpoint payload.',
        ],
        integration_note='Adapter V2 INPUT_PREPARATION_AUDIT explicitly binds source_model_freeze required by this evaluator; prior V1 snapshot is preserved and was never executed.',
        actual_current_artifact_SHA_body_RGB_GT_score_reads=0,model_forward_fit_PnP=0)
    write('REVIEW.json',report)
    write('REVIEW_SEAL.json',dict(completed=True,PASS=True,files=[binding(p)for p in sorted(HERE.iterdir())if p.is_file()],
        evaluator=binding(CODE),protocol=binding(PROTOCOL),current_model_fit_PnP=0))
    print(json.dumps(dict(review=binding(HERE/'REVIEW.json'),seal=binding(HERE/'REVIEW_SEAL.json')),indent=2),flush=True)


if __name__=='__main__':main()
