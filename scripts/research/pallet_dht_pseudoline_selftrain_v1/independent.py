"""Analytic NumPy derivative check independent of Torch autograd/loss assembly."""
import numpy as np

from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.data import ObservationDataset
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.constants import EDGES
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.util import read_json
from .audit_mechanism import DOC,EXPORT,SOURCES,save


def run():
    recorded=read_json(DOC/'LINE_MECHANISM_AUDIT.json');reports=[]
    for split,path in SOURCES.items():
        preds={r['frame_id']:r for r in read_json(path)['records']};max_cos=max_derivative=max_virtual=0.
        rows={r['frame_id']:r for r in recorded['frame_records'][split]}
        for item in ObservationDataset(EXPORT/f'{split}.json',targets=True):
            p=item['base_points'].numpy().astype(float);yv=item['target_valid'].numpy();pv=item['point_valid'].numpy()
            original=item['target_points'].numpy().astype(float);D=np.linalg.norm(item['image_hw'].numpy().astype(float))
            options=[]
            for choice,perm in enumerate(item['symmetry_permutations'].numpy()):
                y=original[perm];mask=yv[perm][:8];e=np.full(8,D);valid=mask&pv[:8]
                e[valid]=np.linalg.norm(p[:8][valid]-y[:8][valid],axis=-1)
                options.append((float(e[mask].mean()) if mask.any() else D,choice,y,mask))
            base,choice,y,mask=min(options,key=lambda a:a[0]);assert choice==rows[item['frame_id']]['shared_symmetry_choice']
            pred=preds[item['frame_id']];lines=np.asarray(pred['raw_line'],float)
            weight=np.clip(1-np.asarray(pred['ambiguity']),0,1)*(np.asarray(pred['utility'])>0)
            weight*=np.isfinite(lines).all(-1)&(np.linalg.norm(lines[:,:2],axis=-1)>1e-8)
            grad_line=np.zeros_like(p);grad_gt=np.zeros_like(p)
            for e,(a,b) in enumerate(EDGES):
                if weight[e]==0:continue
                l=lines[e]/np.linalg.norm(lines[e,:2])
                for k in (a,b):
                    z=(p[k]@l[:2]+l[2])/D
                    slope=z/.01 if abs(z)<.01 else np.sign(z)
                    grad_line[k]+=weight[e]/max(weight.sum(),1e-8)*slope*l[:2]/D
            for k in range(8):
                if mask[k] and pv[k]:
                    difference=p[k]-y[k];length=np.linalg.norm(difference)
                    if length:grad_gt[k]=difference/length/max(mask.sum(),1)/D
            nl=np.linalg.norm(grad_line);ng=np.linalg.norm(grad_gt)
            cos=float(np.sum(grad_line*grad_gt)/(nl*ng)) if nl and ng else 0.
            direction=-grad_line/max(nl,1e-30);derivative=float(np.sum(grad_gt*direction))
            pp=p+.001*direction;e=np.full(8,D);valid=mask&pv[:8]
            e[valid]=np.linalg.norm(pp[:8][valid]-y[:8][valid],axis=-1)
            change=(float(e[mask].mean())-base)/D if mask.any() else 0.
            ref=rows[item['frame_id']]
            max_cos=max(max_cos,abs(cos-ref['gradient_cosine']))
            max_derivative=max(max_derivative,abs(derivative-ref['directional_derivative']))
            max_virtual=max(max_virtual,abs(change-ref['virtual_GT_loss_change']))
        assert max_cos<1e-10 and max_derivative<1e-12 and max_virtual<1e-12
        reports.append(dict(split=split,N=len(rows),max_cosine_error=max_cos,max_directional_derivative_error=max_derivative,
            max_virtual_loss_change_error=max_virtual,PASS=True))
    save('INDEPENDENT_GRADIENT_AUDIT.json',dict(PASS=True,method='analytic NumPy SmoothL1 line gradient and Euclidean GT gradient; independent shared assignment',
        optimizer_updates=0,reports=reports))
    print(reports,flush=True)


if __name__=='__main__':run()
