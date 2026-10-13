"""CAL46-only cached parity and synthetic tests for the optional main port."""
from pathlib import Path
import builtins
import io
from contextlib import contextmanager
import unittest
import gzip
import json
import hashlib
import ast

import numpy as np

from scripts.research.pallet_visibility_curriculum_20261013_v1 import curriculum as M

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'geometric_outline_fusion_v1'
MAIN=Path(M.__file__).resolve()


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(line) for line in gzip.open(p,'rt')]
def cube(d):
    a,b,c=np.asarray(d)/2
    return np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])


@contextmanager
def allowed_cached_CAL_only():
    originals=builtins.open,io.open,Path.open
    allowed={str(p.resolve()) for p in [HERE/'PREDICTIONS.jsonl.gz',HERE/'PREDICTION_SEAL.json',
        HERE/'MAIN_INTEGRATION_BINDING.json',HERE/'PROTOCOL.json',HERE/'progressive_curriculum.py',
        HERE.parent/'CAL_INPUTS.json',PRIOR/'PROTOCOL.json',PRIOR/'PREDICTIONS.jsonl.gz',PRIOR/'PREDICTION_SEAL.json']}
    def wrap(fn):
        def guarded(path,*args,**kwargs):
            name=str(Path(path).resolve()) if isinstance(path,(str,Path)) else str(path)
            if any(x in name for x in ['SCORING','TARGETS','SCORED_ROWS','test_reused','FINAL']):
                raise AssertionError('Forbidden label/TEST path '+name)
            if '/_docs/' in name and name not in allowed:
                raise AssertionError('Outside isolated CAL-cache allowlist '+name)
            return fn(path,*args,**kwargs)
        return guarded
    builtins.open,io.open,Path.open=map(wrap,originals)
    try:yield
    finally:builtins.open,io.open,Path.open=originals


def main():
    output=HERE/'MAIN_INTEGRATION_VERIFICATION.json'
    if output.exists():raise FileExistsError('Preserve integration verification')
    before_sha=sha(MAIN)
    binding=read(HERE/'MAIN_INTEGRATION_BINDING.json');assert before_sha==binding['main_after_sha256']
    assert M.CurriculumConfig().progressive_active_support is False
    assert ast.dump(ast.parse(MAIN.read_text()))==ast.dump(ast.parse((HERE/'progressive_curriculum.py').read_text()))
    legacy_count=progressive_count=0;maximum_R=maximum_t=0.
    with allowed_cached_CAL_only():
        for path in [HERE.parent/'test_reused_v1/ISOLATED_TEST_INPUTS.json',HERE.parent/'OOF_pose_v1/ISOLATED_CAL_SCORING.json',
            HERE.parent.parent/'test_v1/PREDICTIONS.jsonl.gz',HERE.parent.parent/'PREDICTION_INPUTS.json']:
            try:read(path)
            except AssertionError:pass
            else:raise AssertionError('Scope access canary failed')
        suite=unittest.defaultTestLoader.loadTestsFromNames([
            'scripts.research.pallet_visibility_curriculum_20261013_v1.test_progressive_curriculum',
            'scripts.research.pallet_visibility_curriculum_20261013_v1.test_curriculum'])
        result=unittest.TextTestRunner(verbosity=1).run(suite)
        assert result.testsRun==18 and result.wasSuccessful()
        previous_seal=read(PRIOR/'PREDICTION_SEAL.json');new_seal=read(HERE/'PREDICTION_SEAL.json')
        assert previous_seal['sha256']==sha(PRIOR/'PREDICTIONS.jsonl.gz')
        assert new_seal['sha256']==sha(HERE/'PREDICTIONS.jsonl.gz')
        frames={f['id']:f for f in read(HERE.parent/'CAL_INPUTS.json')['frames']};assert len(frames)==46
        arms={a['name']:a for a in read(PRIOR/'PROTOCOL.json')['arms']}
        cases=rows(PRIOR/'PREDICTIONS.jsonl.gz')
        cases+=[r for r in rows(HERE/'PREDICTIONS.jsonl.gz') if r['arm'].endswith('PROGRESSIVE_ACTIVE_SUPPORT')]
        assert len(cases)==322
        for row in cases:
            frame=frames[row['id']];assert frame['partition']=='CAL'
            initial=frame['initial_poses']['N3_SUBPIX'];xyz=cube(initial['cf_extents'])
            if row['arm'] in arms:
                arm=arms[row['arm']]
                cfg=dict(visible_threshold=.2,staged_visible=arm['staged'],geometric_outline_priority=arm['geometric_outline_priority'])
                legacy_count+=1
            else:
                cfg=dict(visible_threshold=.2,geometric_outline_priority=True,progressive_active_support=True)
                progressive_count+=1
            result=M.solve_curriculum(np.asarray(row['observed_points'])[:8],xyz,frame['K'],
                initial['R_cf'],initial['centroid'],row['fixed_H'],row['appearance'],cfg)
            maximum_R=max(maximum_R,float(np.max(np.abs(np.asarray(result['R'])-row['solution']['R']))))
            maximum_t=max(maximum_t,float(np.max(np.abs(np.asarray(result['t'])-row['solution']['t']))))
            np.testing.assert_allclose(result['R'],row['solution']['R'],rtol=0,atol=1e-10)
            np.testing.assert_allclose(result['t'],row['solution']['t'],rtol=0,atol=1e-10)
            assert result['accepted']==row['solution']['accepted'] and result['used_ids']==row['solution']['used_ids']
            assert [(s['name'],s['accepted'],s['reason'],s['used_ids']) for s in result['stages']]==[(s['name'],s['accepted'],s['reason'],s['used_ids']) for s in row['solution']['stages']]
            if cfg.get('progressive_active_support'):
                assert result['final_guard']==row['solution']['final_guard']
                assert result['unknown_added_ids']==row['solution']['unknown_added_ids']
    assert sha(MAIN)==before_sha and legacy_count==230 and progressive_count==92
    report=dict(schema='PROGRESSIVE_MAIN_OPTIONAL_POLICY_VERIFICATION_V1',PASS=True,main_sha256=before_sha,
        main_bytes_unchanged_during_verification=True,default_progressive_flag=False,
        main_AST_exactly_matches_frozen_prototype=True,synthetic_tests_passed=18,
        original_sealed_CAL_cases_exact_policy_parity=legacy_count,new_sealed_progressive_CAL_cases_exact_policy_parity=progressive_count,
        maximum_R_element_difference=maximum_R,maximum_t_element_difference=maximum_t,
        stage_acceptance_names_reasons_used_ids_exact=True,final_guard_and_unknown_restore_exact=True,
        GT_TEST_severe_data_access_during_this_verification=0,access_allowlist_canary='PASS',
        scope='This isolated verification only. Separate historical compatibility-fixture access is disclosed in INTEGRATION_COMPATIBILITY_SCOPE.json.')
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
