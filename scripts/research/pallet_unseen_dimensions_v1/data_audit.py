from common import *
import collections,numpy as np

def main():
    render=read(OUT/'RENDER_COMPLETE.json');rows=render['rows']
    splits={s:[r for r in rows if r['split']==s] for s in ['train','val','test']}
    hashes={s:{r['sha256'] for r in rs} for s,rs in splits.items()}
    groups={s:{r['group'] for r in rs} for s,rs in splits.items()}
    checks={}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        checks[f'{a}_{b}_images_disjoint']=not hashes[a]&hashes[b]
        checks[f'{a}_{b}_groups_disjoint']=not groups[a]&groups[b]
    checks['unseen_assets_absent_train']=not {r['asset'] for r in splits['train']}&set(ASSETS[2:])
    checks['GT_projection_max_below_1e_minus8']=render['gt_projection_max_px']<1e-8
    stats={}
    for split,rr in splits.items():
        stats[split]=dict(frames=len(rr),asset_counts=dict(collections.Counter(x['asset'] for x in rr)),
            panel_actual_occlusion_fraction_mean=float(np.mean([x['object_occluded_fraction'] for x in rr if x['state']=='panel'])),
            crop_frames_with_any_GT_corner_outside=sum(not all(x['inside']) for x in rr if x['state']=='crop25'),
            crop_frames=sum(x['state']=='crop25' for x in rr))
    write(OUT/'DATA_AUDIT.json',dict(PASS=all(checks.values()),checks=checks,populations=stats,
        mesh_vertices={m['asset']:m['vertices'] for m in render['assets']},
        note='Image hashes establish exact duplicate absence, not arbitrary near-duplicate absence. Groups disjoint across splits; same sampled scenes shared across assets within split.'))
    assert all(checks.values());print('Data audit PASS')
if __name__=='__main__':main()
