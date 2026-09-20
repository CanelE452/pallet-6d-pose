"""Verify the exact augmentation intervention plus full selective inference."""
import subprocess
import sys
import numpy as np
from . import dino_large_gate as L
from . import audit_dino_evidence_gate as A
from . import dino_evidence_gate_tail_diagnostic as T

C=L.C;E=L.E;D=L.D


def main():
    p=L.verify();old=E.verify()
    equal=['source_train_rows','calibration_rows','test_rows','feature_names','steps','fit','calibration','candidates','features']
    for key in equal:assert p[key]==old[key],key
    run=subprocess.run([sys.executable,'-m','pytest','-q',str(C.HERE/'test_dino_large_gate.py')],capture_output=True,text=True)
    print(run.stdout,flush=True);assert run.returncode==0 and '3 passed' in run.stdout,run.stderr
    before=C.read(E.DOC/'SOURCE_CACHE.json');after=C.read(L.DOC/'SOURCE_CACHE.json');counts={}
    with L.W.scope():bank={r['row']:r for r in D.load_bank() if r['domain']=='source'}
    for a in E.ARMS:
        C.verify(before['artifacts'][a]);C.verify(after['artifacts'][a])
        with np.load(C.ROOT/before['artifacts'][a]['path']) as z:olddata={k:np.array(z[k]) for k in z.files}
        with np.load(C.ROOT/after['artifacts'][a]['path']) as z:new={k:np.array(z[k]) for k in z.files}
        for key in ['row','view','corner','after']:np.testing.assert_array_equal(new[key],olddata[key])
        unchanged=new['view']<=2
        for key in ['before','features']:np.testing.assert_array_equal(new[key][unchanged],olddata[key][unchanged])
        checked=0
        for row in np.unique(new['row']):
            r=bank[int(row)]
            for view in [3,4]:
                q=L.perturb(r,view);mask=(new['row']==row)&(new['view']==view);corners=new['corner'][mask]
                error=np.linalg.norm(q['points']-r['target'],axis=-1)/r['matrix'][0,0]
                np.testing.assert_array_equal(error[corners],new['before'][mask]);checked+=int(mask.sum())
                np.testing.assert_array_equal(q['target'],r['target']);np.testing.assert_array_equal(q['target_valid'],r['target_valid'])
        tr=np.isin(new['row'],p['source_train_rows']);positive=tr&(new['after']<=10)&(new['after']+5<new['before'])
        counts[a]=dict(samples=len(new['row']),unchanged_view012_samples=int(unchanged.sum()),
            independently_checked_large_view_samples=checked,positive=int(positive.sum()),
            positive_move_over_30pct=int((positive&(new['features'][:,8]>.3)).sum()),
            feature_changed_samples=int((new['features']!=olddata['features']).any(-1).sum()))
    # Same frozen candidates/scoring and full194 forward reproduction as parent.
    with L.scope():A.main();T.main()
    C.freeze(L.DOC/'INTERVENTION_AUDIT.json',dict(experiment_complete=True,goal_complete=False,
        identical_parent_protocol_fields=equal,extra_tests_passed=3,total_tests_passed=30,
        all_targets_and_candidate_errors_unchanged=True,source_cache_comparison=counts,
        new_annotations=0,new_tags=0,auto_promoted=False,
        evidence=[C.bound(__file__),C.bound(C.HERE/'test_dino_large_gate.py'),C.bound(L.DOC/'PROTOCOL.json'),
            C.bound(L.DOC/'COMPLETION_AUDIT.json'),C.bound(L.DOC/'REJECTED_LARGE_DIAGNOSTIC.json')]))
    print('LARGE_INTERVENTION_AUDIT_COMPLETE',counts,flush=True)


if __name__=='__main__':main()
