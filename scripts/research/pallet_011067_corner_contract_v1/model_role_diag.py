"""Cannot load model payload until saved human decision or decisive evidence lock."""
import numpy as np
from . import common as C

def main():
    decision=C.read(C.RAW/'HUMAN_DECISION_PRIVATE.json')
    assert decision['input'] in ('human GUI selection','explicit user chat selection transcribed by assistant')
    if decision['input']!='human GUI selection':
        assert decision['confidence']=='NOT_REPORTED' and not decision['assistant_visual_judgment_used']
    assert decision['ui_lock_sha256']==C.sha(C.RAW/'HUMAN_UI_LOCK.json')
    lock=C.read(C.RAW/'HUMAN_UI_LOCK.json');assert lock['mapping_sha256']==C.sha(C.RAW/'BLIND_MAPPING_PRIVATE.json')
    geometry=C.read(C.RAW/'GEOMETRY_LOCK.json');assert C.sha(C.DOC/'GEOMETRY_ONLY_CANDIDATES.json')==geometry['geometry']['sha256']
    inputs=C.read(C.DOC/'INPUT_BINDINGS.json')
    for b in inputs['files']:assert C.sha(C.ROOT/b['path'])==b['sha256']
    # First new-stage model payload read occurs strictly after the above locks.
    base=C.ROOT/'data/pallet/results/pallet_existing_data_transfer_v1'
    target=next(r for r in C.read(base/'TARGETS.json') if r['id']==C.FRAME)
    d=C.read(base/'DIAGNOSTICS.json');preds={'TEACHER':np.asarray(target['target'])}
    for name,arm in [('T1','T1_HARD_PSEUDO'),('T2','T2_HARD_MANUAL')]:
        r=next(r for r in d[arm]['train'] if r['id']==C.FRAME)['prediction']
        preds[name]=np.asarray(r['candidates'][r['selected_index']]['keypoints_xy'])
    candidates=C.read(C.DOC/'C4_CANDIDATES.json')['candidates'];g=C.points();m=C.direct_mask()
    rows={}
    for arm,p in preds.items():
        scores=[]
        for c in candidates:
            e=np.linalg.norm(p[c['perm_new_to_stored']][m]-g[m],axis=1)
            scores.append(dict(c4=c['c4'],mean_px=float(e.mean()),median_px=float(np.median(e)),max_px=float(e.max()),n=len(e)))
        rank=sorted(scores,key=lambda s:s['mean_px'])
        rows[arm]=dict(candidates=scores,same_ID=scores[0],closest_C4=rank[0]['c4'],best_whole_C4=rank[0],
            second_best_margin_mean_px=rank[1]['mean_px']-rank[0]['mean_px'])
        c2=min((x for x in scores if x['c4'] in ('YAW_0','YAW_180')),key=lambda x:x['mean_px'])
        best_perm=next(c['perm_new_to_stored'] for c in candidates if c['c4']==rank[0]['c4'])
        inverse=np.argsort(best_perm).tolist()
        rows[arm]['native_model_role_relative_to_stored']=next(c['c4'] for c in candidates if c['perm_new_to_stored']==inverse)
        rows[arm]['best_stored_C2_diagnostic']=c2
        # Separate user-facing reference-role change from inverse prediction remapping.
        mapping=C.read(C.RAW/'BLIND_MAPPING_PRIVATE.json')['mapping']
        choice=mapping.get(decision['choice'])
        if choice:
            cp=next(c['perm_new_to_stored'] for c in candidates if c['c4']==choice)
            e=np.linalg.norm(p[m[cp]]-g[cp][m[cp]],axis=1)
            rows[arm]['user_selected_role_diagnostic']=dict(mean_px=float(e.mean()),median_px=float(np.median(e)),n=len(e),
                reference_role=choice,actual_GT_edited=False)
    old=C.read(C.ROOT/'_docs/experiments/pallet_existing_data_transfer_v1/THREE_CORNER_AUDIT.json')
    errors=np.linalg.norm(preds['T2']-g,axis=1)
    for r in old['rows']:assert abs(errors[r['corner']]-r['T2_error'])<1e-8
    C.put(C.DOC/'MODEL_ROLE_DIAGNOSTIC.json',dict(models=rows,criterion='mean error over all four direct manual points',
        permutation_direction='pred[perm_new_to_stored] compared to fixed stored direct-manual reference',
        GT_posthoc=True,deployable_performance=False,free_matching=False,residual3_reproduced=True,
        C2_diagnostic_not_official_pose_metric=True,
        human_decision_sha256=C.sha(C.RAW/'HUMAN_DECISION_PRIVATE.json'),geometry_lock_sha256=C.sha(C.RAW/'GEOMETRY_LOCK.json')))
    C.put(C.RAW/'MODEL_COORDINATES_PRIVATE.json',{k:v.tolist() for k,v in preds.items()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,(name,p) in zip(axes,preds.items()):
        ax.imshow(Image.open(C.RGB));ax.scatter(g[m,0],g[m,1],color='lime',label='Direct manual')
        ax.scatter(p[:8,0],p[:8,1],facecolors='none',edgecolors='cyan',label=name)
        for j in np.flatnonzero(m):
            ax.annotate(f'GT P{j}',g[j],xytext=(4,10),textcoords='offset points',color='lime',fontsize=8,
                        bbox=dict(facecolor='black',alpha=.6,pad=1,edgecolor='none'))
        for j in range(8):
            ax.annotate(f'P{j}',p[j],xytext=(4,-10),textcoords='offset points',color='cyan',fontsize=8,
                        annotation_clip=True,bbox=dict(facecolor='black',alpha=.6,pad=1,edgecolor='none'))
        ax.set_xlim(325,640);ax.set_ylim(375,175);ax.axis('off');ax.set_title(name+' native IDs (not remapped)')
    fig.suptitle('Green: stored manual ID | Cyan: native model ID | No GT or prediction edits',fontsize=12)
    fig.tight_layout();fig.savefig(C.FIG/'07_model_role_after_lock.jpg',dpi=150);plt.close(fig)
    print(rows)

if __name__=='__main__':main()
