"""Display saved observed-outline sample contributions; no cost or pose evaluation."""
from pathlib import Path
import argparse
import math
import time

import numpy as np

from .common import read, rows, write, write_csv, sha


def render(root, public):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from PIL import Image
    started, cpu = time.monotonic(), time.process_time()
    root, public = Path(root), Path(public)
    gallery = public / 'gallery'
    out = gallery / 'outline_witness'
    out.mkdir(exist_ok=False)
    proof = read(gallery / 'GALLERY_PROOF.json')
    representatives = read(gallery / 'REPRESENTATIVE_SELECTION.json')['examples']
    ids = {r['id'] for r in representatives}
    cases = {c['id']: c for c in proof['cases'] if c['id'] in ids}
    wanted = {(c['id'], p['candidate_id']) for c in cases.values() for p in c['panels'][-2:]}
    pointers = { (r['id'], r['candidate_id']): r['dense_outline_stored_in']
                for r in rows(root / 'cost_v1/CANDIDATE_COSTS.jsonl.gz')
                if (r['id'], r['candidate_id']) in wanted }
    dense = {}
    for ident, cid in sorted(wanted):
        pointer = pointers[(ident, cid)]
        record = next(r for r in rows(root / 'cost_v1' / pointer['path'])
                      if r['id'] == ident and r['candidate_id'] == cid)
        dense[(ident, cid)] = record
    outputs = []
    for number, rep in enumerate(representatives, 1):
        ident = rep['id']; case = cases[ident]
        panels = case['panels'][-2:]
        records = [dense[(ident, p['candidate_id'])] for p in panels]
        observations = [next(o for o in r['per_observation']
                            if o.get('direction') == 'observed_to_model') for r in records]
        sample = np.asarray(observations[0]['sample_xy'])
        # Only observed samples have identical physical indices between poses.
        assert np.array_equal(sample, np.asarray(observations[1]['sample_xy']))
        values = [np.asarray(o['weighted_sample_contributions']) for o in observations]
        delta = values[1] - values[0]
        expected = observations[1]['weighted_contribution'] - observations[0]['weighted_contribution']
        assert abs(math.fsum(delta) - expected) <= 1e-9
        extent = case['viewport_original_xyxy']; x0,y0,x1,y1=extent
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
        image = Image.open(gallery / case['source_crop'])
        magnitude = max(float(np.max(np.abs(delta))), 1e-15)
        norm = TwoSlopeNorm(vmin=-magnitude,vcenter=0,vmax=magnitude)
        for ax, panel, title in zip(axes[:2], panels, ['actual selected s', 'reference diagnostic comparator']):
            ax.imshow(image, extent=(x0,x1,y1,y0), interpolation='nearest')
            q = np.asarray(panel['model_projection_xy'])
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for a,b in edges: ax.plot(q[[a,b],0],q[[a,b],1],color='#24bb8e',lw=1)
            sc = ax.scatter(sample[:,0],sample[:,1],c=delta,cmap='coolwarm',norm=norm,s=14,zorder=5)
            ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_aspect('equal')
            ax.set_title(title);ax.set_xlabel('original x [px]');ax.set_ylabel('original y [px]')
        fig.colorbar(sc,ax=list(axes[:2]),shrink=.65,label='weighted sample Delta: comparator - s')
        axes[2].plot(np.arange(len(delta)),delta,lw=.8,color='#333333')
        axes[2].axhline(0,color='#888888',lw=.5)
        axes[2].set_xlabel('original saved observed sample index')
        axes[2].set_ylabel('weighted sample Delta')
        axes[2].set_title(f'observed-to-model sum Delta = {expected:.8g}')
        fig.suptitle(ident+' | red favors selected s; blue favors comparator\n'
                     'Original saved outline samples and saved model projections; no semantic owner labels',fontsize=10)
        name=f'witness_{number:02d}.jpg';fig.savefig(out/name,dpi=110,pil_kwargs={'quality':87,'optimize':True});plt.close(fig)
        contributions=[dict(sample_index=i,x_px=float(xy[0]),y_px=float(xy[1]),
                            selected_contribution=float(values[0][i]),comparison_contribution=float(values[1][i]),
                            Delta=float(delta[i])) for i,xy in enumerate(sample)]
        write_csv(out/f'witness_{number:02d}_samples.csv.gz',contributions)
        outputs.append(dict(id=ident,group=rep['group'],image=name,observed_sample_count=len(sample),
            Delta_observed_to_model=expected,Delta_sample_sum=math.fsum(delta),
            largest_positive_samples=sorted(contributions,key=lambda r:(-r['Delta'],r['sample_index']))[:5],
            largest_negative_samples=sorted(contributions,key=lambda r:(r['Delta'],r['sample_index']))[:5],
            original_dense_pointers=[pointers[(ident,p['candidate_id'])] for p in panels],
            source_crop=case['source_crop'],source_crop_sha256=sha(gallery/case['source_crop'])))
    write(out/'WITNESS_RECEIPT.json',dict(examples=outputs,
        same_observed_sample_indices_verified=True,model_sample_indices_never_paired=True,
        semantic_owner_verified=False,new_cost_evaluations=0,new_observations=0,new_pose_calls=0,
        elapsed_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu))
    (out/'INDEX_KO.md').write_text('# 저장 외곽선 표본의 후보 선택 비용 기여\n\n'
        '각 점은 원 packet의 동일한 observed outline 표본입니다. 빨강은 실제 선택을, 파랑은 참조 대조 후보를 선호합니다. '
        '모델의 자세마다 달라지는 model-to-observed 표본 인덱스는 서로 빼지 않았습니다. '
        '표본 비용은 저장 dense 장부에서 읽었고 새 선·mask·자세는 생성하지 않았습니다. '
        '이 그림만으로 어느 픽셀이 물리적 코너/외곽선의 정답인지 판정하지 않습니다.\n\n'+
        '\n'.join(f'## {r["id"]}\n\n![{r["id"]}]({r["image"]})\n\n'
                  f'관측→모델 비용차: {r["Delta_observed_to_model"]:.9g}. '
                  f'[전체 표본](witness_{i:02d}_samples.csv.gz)\n' for i,r in enumerate(outputs,1)),encoding='utf-8')
    return outputs


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--public',required=True)
    a=p.parse_args();print(len(render(a.root,a.public)))
