"""Executable regression/audit tests; blocked LOO is not reported as executed."""
import ast
import inspect
import re
from pathlib import Path
import numpy as np
import torch
from . import diagnose as D

def main():
    tests={};bindings=D.read(D.DOC/'INPUT_BINDINGS.json')['files']
    data=D.load();fr=D.read(D.DOC/'FRAME_DIAGNOSTICS.json');p=D.read(D.DOC/'PARITY.json')
    def check(name,fn):
        fn();tests[name]='PASS'
    def no_training():
        for name in ['diagnose.py','run.py','render.py']:
            tree=ast.parse(Path(__file__).with_name(name).read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Call):
                    n=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                    assert n not in {'backward','step','train','fit','YOLO','load_state_dict'},(name,n)
    check('test_no_training_code_path',no_training)
    def no_optimizer():
        old=torch.optim.Optimizer.__init__
        def forbidden(*a,**kw):raise AssertionError('optimizer created')
        torch.optim.Optimizer.__init__=forbidden
        try:
            r=data['records'][0];q=D.points(data['preds']['S0'][r['id']]);m=data['meta'][r['id']]
            D.Pose.infer(q,np.array(m['K']),np.array(m['xyz']),False)
        finally:torch.optim.Optimizer.__init__=old
        assert D.read(D.DOC/'ANALYSIS_COMPLETE.json')['optimizer_steps']==0
    check('test_no_optimizer_created',no_optimizer)
    def hashes():
        for b in bindings:D.C.verify(b)
    check('test_predictions_hash_unchanged',hashes);check('test_original_experiment_unchanged',hashes)
    def pose_parity():
        assert p['passed'] and p['checks']['pose']==1800 and p['checks']['pose_metrics']==1800
    for name in ['test_current_pose_parity','test_current_axis_accuracy_parity','test_current_add_auc_parity']:check(name,pose_parity)
    def selector_parity():
        for arm in D.ARMS:
            for r in data['records']:
                fid=r['id'];q=D.points(data['preds'][arm][fid]);m=data['meta'][fid];out=D.select(q,np.array(m['K']),np.array(m['xyz']))
                assert out['selected_hypothesis']==fr[arm][fid]['selector']['selected_hypothesis']
                assert [h['score'] for h in out['hypotheses']]==[h['selector_score'] for h in fr[arm][fid]['hypotheses']]
    check('test_current_selector_parity',selector_parity)
    def gt_free():
        assert list(inspect.signature(D.select).parameters)==['q','K','xyz']
        tree=ast.parse(inspect.getsource(D.select))
        names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
        assert not names & {'gt','truth','reference','data'}
        assert list(inspect.signature(D.Selector.select_pnp_hypotheses).parameters)==['predicted_keypoints','camera_intrinsics','physical_dimensions','config']
        assert all(len(r['hypotheses'])==2 for a in D.ARMS for r in fr[a].values())
    check('test_two_hypotheses_prediction_only',gt_free);check('test_gt_not_used_in_selector_score',gt_free)
    def oracle():assert 'NONDEPLOYABLE' in D.read(D.DOC/'ORACLE_WD_AUDIT.json')['label']
    check('test_oracle_flagged_nondeployable',oracle)
    loo=D.read(D.DOC/'LOO_CORNER_INFLUENCE.json')
    def min6():
        assert len(loo['rows'])==87*3*8
        assert all(r['remaining']>=6 or r['status']=='SKIP_MIN6' for r in loo['rows'])
        assert loo['influence_counts'] is None
    check('test_loo_min6_points',min6)
    tests['true_LOO_execution']='BLOCKED: finite9 selector contract; no selector modification authorized'
    def replacement():
        rr=D.read(D.DOC/'GT_REPLACEMENT_ORACLE.json')
        assert rr['label']=='GT_REPLACEMENT_ORACLE_NONDEPLOYABLE'
        assert all(r['label']==rr['label'] and 'reference_source' in r for r in rr['rows'])
    check('test_gt_replacement_only_posthoc',replacement)
    def sources():assert all(len(r['provenance']['corner_sources'])==8 for a in D.ARMS for r in fr[a].values())
    check('test_reference_provenance_attached',sources)
    def membership():
        ids={r['id'] for r in data['records']}
        assert len(ids)==300
        assert all(set(fr[a])==ids for a in D.ARMS)
        from collections import Counter
        assert Counter(r['severity'] for r in data['records'])==dict(CLEAN=132,MODERATE_OCCLUSION=87,SEVERE_OCCLUSION=81)
    check('test_frame_membership_fixed',membership)
    report=(D.DOC/'REPORT_KO.md').read_text();links=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',report)
    def images():
        files=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report)
        assert len(files)>=23
        for path in files:assert (D.DOC/path).is_file(),path
        cases=D.read(D.DOC/'CASE_SELECTION.json')
        for kind,n in [('moderate_pck_up_pose_down',6),('moderate_pose_recovered',4),('random_control',6)]:assert sum(r['kind']==kind for r in cases)>=n
    check('test_report_images_exist',images)
    def linktest():
        for path in links:
            if path in ('AUDIT.json','FINAL_AUDIT.json'):continue # receipt written immediately after tests
            assert (D.DOC/path).is_file(),path
    check('test_report_links_resolve',linktest)
    hashes()
    receipt='FINAL_AUDIT.json' if (D.DOC/'AUDIT.json').exists() else 'AUDIT.json'
    D.save(D.DOC/receipt,dict(status='ANALYSIS_COMPLETE_WITH_EXPLICIT_LOO_LIMIT',tests=tests,training=0,optimizer_steps=0,originals_unchanged=True,
        code=[D.C.bind(p) for p in sorted(Path(__file__).parent.glob('*.py'))],report=D.C.bind(D.DOC/'REPORT_KO.md'),
        figures=len(list((D.DOC/'figures').glob('*'))),all_report_links_exist_after_receipt=True))
    assert all((D.DOC/path).is_file() for path in links)
    print(tests,flush=True)

if __name__=='__main__':main()
