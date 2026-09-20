"""Source-only decomposition: fixed roles, official symmetry, unordered support.

This is a GT diagnostic, not a new decoder, gate, selection or reported method.
Keep every stored prediction and the fixed-source gates unchanged.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from . import dino_source_diversity as X
from . import dino_joint_model as J

C=X.C;D=X.D;N=X.N;V=X.V;L=X.L
SIDECAR=C.ROOT/'data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz'


def summary(values):
    a=np.asarray(values);assert np.isfinite(a).all()
    return dict(n=len(a),PCK10=float((a<=10).mean()),PCK20=float((a<=20).mean()),
        median_px=float(np.median(a)),P90_px=float(np.quantile(a,.9)),mean_px=float(a.mean()))


@torch.no_grad()
def main():
    p=X.verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    source=X.P.SourceData();side=np.load(SIDECAR);np.testing.assert_array_equal(side['record_index'],source.data.indices)
    bank=X.load_bank();sm={r['row']:r for r in bank if r['domain']=='source'};held=[sm[i] for i in p['source_held_rows']]
    models={};evidence=[C.bound(__file__),C.bound(J.__file__),C.bound(SIDECAR),C.bound(X.DOC/'PROTOCOL.json')]
    for phase in [V,L,X]:
        for arm in D.ARMS:
            fit=C.read(phase.DOC/f'FIT_{arm}.json');C.verify(fit['checkpoint']);evidence.append(C.bound(phase.DOC/f'FIT_{arm}.json'))
            ck=torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)
            m=V.V.Head().cuda();m.load_state_dict(ck['model']);models[phase.PHASE+'/'+arm]=m.eval()
    data={name:{k:[] for k in ['fixed','official_symmetry','nearest_point_oracle','one_to_one_assignment_oracle']} for name in ['R0',*models]}
    details=[]
    for offset in range(0,len(held),4):
        rows=held[offset:offset+4];b=D.tensor_batch(rows)
        predictions={name:V.V.decode(m(b['feature'],b['points'],b['valid'])).cpu().numpy() for name,m in models.items()}
        for i,r in enumerate(rows):
            mask=r['target_valid'][:8];gt=r['target'][:8][mask];gain=r['matrix'][0,0]
            perms=side['permutations'][r['row'],:int(side['order'][r['row']])]
            for name,q in [('R0',r['points']),*[(name,q[i]) for name,q in predictions.items()]]:
                fixed=np.linalg.norm(q[:8][mask]-gt,axis=-1)/gain
                canonical=J.canonical_errors(q[None],r['target'],r['target_valid'],perms,gain)[0]
                canonical=canonical[np.isfinite(canonical)]
                distance=np.linalg.norm(q[:8,None]-gt[None],axis=-1)/gain
                nearest=distance.min(0);ix,jx=linear_sum_assignment(distance);assignment=distance[ix,jx]
                assert len(fixed)==len(canonical)==len(nearest)==len(assignment)
                for key,values in [('fixed',fixed),('official_symmetry',canonical),('nearest_point_oracle',nearest),('one_to_one_assignment_oracle',assignment)]:data[name][key].extend(values.tolist())
                details.append(dict(row=r['row'],name=name,symmetry_order=int(side['order'][r['row']]),
                    fixed_mean=float(fixed.mean()),official_symmetry_mean=float(canonical.mean()),
                    nearest_point_mean=float(nearest.mean()),assignment_mean=float(assignment.mean())))
    result={name:{k:summary(v) for k,v in metrics.items()} for name,metrics in data.items()}
    for phase in [V,L,X]:
        for arm in D.ARMS:
            f=C.read(phase.DOC/f'FIT_{arm}.json')['source_after'];now=result[phase.PHASE+'/'+arm]['fixed']
            for key in ['n','PCK10','PCK20','median_px','P90_px','mean_px']:assert now[key]==f[key],(phase.PHASE,arm,key)
    C.freeze(X.DOC/'SOURCE_ROLE_DIAGNOSTIC.json',dict(status='SOURCE_GT_DIAGNOSTIC_NOT_INFERENCE',
        heldout_images=64,metrics=result,rows=details,evidence=evidence,
        predictions_changed=False,thresholds_changed=False,GT_used_for_training_in_this_script=False,
        warning='Oracle matching is explanatory only. Whole official symmetry and arbitrary assignment are different. Native raw source head outputs follow the existing source probe; no invalid-input restoration added.'))
    for name,r in result.items():print('SOURCE_ROLE',name,{k:v['PCK10'] for k,v in r.items()},flush=True)


if __name__=='__main__':main()
