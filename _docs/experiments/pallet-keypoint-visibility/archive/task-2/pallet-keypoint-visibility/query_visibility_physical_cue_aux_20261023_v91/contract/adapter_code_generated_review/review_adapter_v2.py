"""Bounded adapter code/generated review; no actual current/source reader run."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CODE=ROOT/'private_evaluation/infer_native_center_once.py'
TEST=ROOT/'private_evaluation/test_generated_native_center.py'


def b(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    scope=json.loads((HERE/'SCOPE_BEFORE_GENERATED_REVIEW.json').read_text())
    assert scope['code']==b(CODE)and scope['test_code']==b(TEST)
    tree=ast.parse(CODE.read_text())
    defs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    assert not any(isinstance(n,ast.Import)and any(x.name=='torch'for x in n.names)for n in tree.body)
    run_source=ast.get_source_segment(CODE.read_text(),defs['run'])
    assert run_source.index('validate_entry(path)')<run_source.index('load_current(b)')<run_source.index('torch.load(')
    assert "source_model_freeze=b['source_model_freeze']"in run_source
    current_source=ast.get_source_segment(CODE.read_text(),defs['load_current'])
    assert 'features={k:z[k]for k in INPUT_KEYS}'in current_source
    assert "box_xyxy member is not read"in current_source
    spec=importlib.util.spec_from_file_location('independent_generated_nativecentre_tests',TEST)
    test=importlib.util.module_from_spec(spec);spec.loader.exec_module(test)
    test.FIX=HERE/'INVENTED_FIXTURES_RERUN'
    test.a.GENERATED=HERE/'INDEPENDENT_GENERATED_RERUN.json'
    test.main()
    proof=json.loads(test.a.GENERATED.read_text());assert proof['PASS']and proof['code']==b(CODE)
    adapter=test.a
    # Independent enumeration of sign/valid/geometry including FP32 subnormals.
    z=np.array([-np.inf,-1,-np.nextafter(np.float32(0),np.float32(1)),-0.,0.,
        np.nextafter(np.float32(0),np.float32(1)),1,np.inf],np.float32)
    z=z[np.isfinite(z)]
    checked=0
    for oldvalid in(False,True):
        for centervalid in(False,True):
            for geometry in(None,False,True):
                base=[dict(valid=oldvalid,geometry_self_hidden=geometry,prior_reason='GENERATED_OLD_U')for _ in z]
                rows=adapter.decode_rows(base,z,np.full(len(z),.5,np.float32),np.full(len(z),centervalid,bool))
                for value,row in zip(z,rows):
                    valid=oldvalid and centervalid
                    assert row['valid']==valid
                    assert row['authoritative_decision']==('D'if value<0 else'E')if valid else row['authoritative_decision']=='UNKNOWN'
                    assert row['geometry_self_hidden']==geometry
                    assert row['geometry_self_hidden_if_valid']==(geometry if valid else None)
                    assert row['decision_visible']==(bool(value<0)if valid else None)
                    assert row['hybrid_decision_visible']==((False if geometry is True else bool(value<0))if valid else None)
                    checked+=1
    report=dict(completed=True,PASS=True,blockers=[],code=b(CODE),protocol=scope['protocol'],
        original_proof=scope['original_proof'],independent_generated_rerun=b(test.a.GENERATED),
        review_code=b(Path(__file__).resolve()),scope=b(HERE/'SCOPE_BEFORE_GENERATED_REVIEW.json'),
        static_checks=dict(no_importtime_torch_model_or_current_IO=True,sourcegate_before_currentload_before_weightload=True,
            only_three_forward_tensors=True,NPZ_box_not_deserialized=True,preparation_direct_sourcefreeze_binding=True),
        generated_checks=dict(negative_source_guards=len(proof['generated_source_guard_cases']),
            invented223x8_q_L_order_and_original45U=True,extra_metadata_bytes_pathSHA_only=True,
            wrong_sourceSHA_rejected=True,FP16_FP32_prior_dtype_preserved=True,centre_only_not_anymap=True,
            raw_signedzero_and_subnormal_decisions=True,probability_display_not_authoritative=True,
            independent_sign_valid_geometry_cases=checked,invalid_precedes_geometry=True,repeat_stub_only_no_realmodel=True),
        limitations=['Code/generated validation only; actual original input packet schema and bytes are checked later by root-approved execution.',
            'Conditional source cue evidence/native virtual8 association/target identity transfer not proven; no semantic self training claim.'],
        actual_current_artifact_body_SHA_RGB_GT_scores_reads=0,actual_source_weights_realmodel_forward_fit_PnP=0)
    out=HERE/'REVIEW.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (HERE/'REVIEW_SEAL.json').write_text(json.dumps(dict(completed=True,PASS=True,review=b(out),code=b(CODE),protocol=scope['protocol']),indent=2)+'\n')
    print(json.dumps(dict(PASS=True,review=b(out))))


if __name__=='__main__':main()
