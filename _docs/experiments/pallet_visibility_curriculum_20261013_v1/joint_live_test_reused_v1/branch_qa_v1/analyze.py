"""Post-seal TEST12 branch diagnostics; policy and learned model stay frozen."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path('/home/minjae/Documents/github/pallet-pose')
HERE=Path(__file__).resolve().parent
RUN=HERE.parent
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def cube(d):
    a,b,c=np.asarray(d)/2
    return np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])
def rotation_y(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def phase(pose,target,meta):
    truth=target['truth'];Q=np.asarray(pose['R_cf']).T@np.asarray(pose['R_physical'])
    physical=cube(pose['cf_extents'])@Q
    refQ=np.asarray(truth['body_R']).T@np.asarray(truth['R'])
    reference=(cube(truth['body_xyz'])@refQ)[meta['native_to_reference_corner_ids']]
    gaps=[float(np.max(np.linalg.norm(physical@rotation_y(j*2*np.pi/truth['order']).T-reference,axis=1))) for j in range(truth['order'])]
    return dict(consistent=min(gaps)<1e-6,max_native_physical_corner_gap_m=min(gaps),symmetry_order=truth['order'])
def stats(x):
    x=np.asarray(x,float)
    return dict(n=len(x),mean=float(x.mean()) if len(x) else None,median=float(np.median(x)) if len(x) else None,max=float(x.max()) if len(x) else None)


def render(frame,row,meta,gt,details,destination):
    image_path=ROOT/frame['image'];assert sha(image_path)==frame['image_sha256']
    image=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)
    q=np.asarray(row['original_points']);moved=np.asarray(row['corrected_measurements'])[:8]
    pool=row['branch_result']['common_visible_ids'];human=meta['human_states_native'];DIRECT=np.asarray([s=='DIRECT_VISIBLE' for s in human])
    valid=np.isfinite(gt).all(1)&~(gt==-1).all(1)
    content=np.row_stack([q,moved,gt[valid]])
    x0,y0=np.maximum(0,np.floor(content.min(0)-35)).astype(int);x1,y1=np.minimum(image.shape[:2][::-1],np.ceil(content.max(0)+35)).astype(int)
    fig,axes=plt.subplots(1,2,figsize=(15,7),sharex=True,sharey=True)
    for ax,candidate,label,color in zip(axes,row['branch_result']['candidates'],['Original family control','Selected alternate family'],['#ffae00','#00cafa']):
        ax.imshow(image);projection=np.asarray(candidate['projected'])
        for a,b in EDGES:ax.plot(projection[[a,b],0],projection[[a,b],1],color=color,lw=1.4,alpha=.9)
        for k in range(8):
            ax.plot(q[k,0],q[k,1],marker='x',color='white',ms=5,mew=1)
            if valid[k] and DIRECT[k]:ax.plot(gt[k,0],gt[k,1],marker='+',color='#73ff00',ms=10,mew=1.7)
            ax.annotate('',xy=moved[k],xytext=q[k],arrowprops=dict(arrowstyle='->',color='#ff6fff',lw=1.5))
            if k in pool:
                good=DIRECT[k] and np.linalg.norm(moved[k]-gt[k])<=8
                ax.plot(moved[k,0],moved[k,1],marker='o',mfc='none',mec='#73ff00' if good else '#ff4545',ms=11,mew=1.5)
                ax.text(moved[k,0]+3,moved[k,1]-3,str(k),fontsize=10,color='white',bbox=dict(facecolor='black',alpha=.6,pad=.7))
        ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_aspect('equal');ax.set_xlabel('Image x (pixels)');ax.set_ylabel('Image y (pixels)')
        ax.set_title(f'{label}: CF {np.array(candidate["dimensions"]).round(2).tolist()}\ncommon cost={candidate["cost"]:.3f}px², condition={candidate["shared_geometry"]["condition"]:.2f}',fontsize=11)
    legend=[Line2D([0],[0],marker='x',ls='',color='white',markeredgecolor='gray',label='raw detector/N3'),
        Line2D([0],[0],color='#ff6fff',label='RGB coordinate delta'),Line2D([0],[0],marker='+',ls='',color='#73b000',label='DIRECT native GT'),
        Line2D([0],[0],marker='o',mfc='none',ls='',color='#73b000',label='shared pool: DIRECT error<=8px'),
        Line2D([0],[0],marker='o',mfc='none',ls='',color='#ff4545',label='shared pool: inaccurate / occluded / unannotated')]
    fig.legend(handles=legend,loc='lower center',ncol=3,fontsize=9)
    before=details['metrics']['JOINT_SINGLE_FAMILY'];after=details['metrics']['JOINT_WD_BRANCH']
    fig.suptitle(f'{details["classification"].upper()} | {row["id"]} | {details["severity"]}\n'
        f'Physical C2 phase: {details["initial_phase"]["consistent"]} → {details["selected_phase"]["consistent"]}; '
        f'T {before["translation_cm"]:.2f}→{after["translation_cm"]:.2f}cm, R {before["rotation_deg"]:.2f}→{after["rotation_deg"]:.2f}°\n'
        f'Shared pool {pool}; {details["pool_DIRECT_count"]} DIRECT, {details["pool_nonDIRECT_states"]}; '
        f'RGB shift mean {details["pool_shift_px"]["mean"]:.2f}px | retired mixed-severity study',fontsize=11)
    fig.tight_layout(rect=(0,.12,1,.82));fig.savefig(destination,dpi=130);plt.close(fig)


def main():
    assert not (RUN/'BRANCH_DIAGNOSTICS.json').exists()
    seal=read(RUN/'PREDICTION_SEAL.json');assert seal['sha256']==sha(RUN/'PREDICTIONS.jsonl.gz')
    labels=read(RUN/'ISOLATED_TEST_SCORING.json')['rows'];inputs=read(RUN/'ISOLATED_TEST_INPUTS.json')['frames']
    assert len(labels)==len(inputs)==97 and all(f['partition']=='TEST' for f in inputs)
    frames={f['id']:f for f in inputs}
    severity={r['id']:r['severity'] for r in read(RUN.parent/'SEVERITY_SCOPE_AUDIT_V1.json')['rows'] if r['partition']=='TEST'}
    rows=[json.loads(l) for l in gzip.open(RUN/'SCORED_ROWS.jsonl.gz','rt')]
    by={(r['id'],r['method']):r for r in rows};details=[];phase_counts=Counter()
    for row in rows:
        if row['method']!='JOINT_WD_BRANCH':continue
        uid=row['id'];target=labels[uid]['target'];meta=labels[uid]['metadata']
        initial_phase=phase(by[uid,'INITIAL']['actual_pose'],target,meta);selected_phase=phase(row['actual_pose'],target,meta)
        phase_counts['initial_consistent' if initial_phase['consistent'] else 'initial_inconsistent']+=1
        if not row['branch_result']['branch_flip']:continue
        classification='recovery' if not initial_phase['consistent'] and selected_phase['consistent'] else 'false_flip' if initial_phase['consistent'] and not selected_phase['consistent'] else 'other_flip'
        branch=row['branch_result'];ids=branch['common_visible_ids'];weights=branch['common_weights']
        q=np.asarray(row['original_points']);corrected=np.asarray(row['corrected_measurements'])[:8]
        mapping=meta['native_to_reference_corner_ids'];gt=np.asarray(target['gt'])[:8][mapping]
        human=meta['human_states_native'];point=[]
        for j,k in enumerate(ids):
            gt_valid=np.isfinite(gt[k]).all() and not (gt[k]==-1).all()
            p=dict(corner=k,human_state=human[k],P_usable=row['joint_observation']['P_usable'][k],
                boundary_score=row['appearance']['boundary_score'][k],local_accuracy_score=row['appearance']['local_accuracy_score'][k],
                weight=weights[j],RGB_shift_px=float(np.linalg.norm(corrected[k]-q[k])),
                raw_GT_error_px=float(np.linalg.norm(q[k]-gt[k])) if gt_valid else None,
                corrected_GT_error_px=float(np.linalg.norm(corrected[k]-gt[k])) if gt_valid else None,
                original_fit_residual_px=branch['candidates'][0]['residuals_shared_px'][j],
                alternate_fit_residual_px=branch['candidates'][1]['residuals_shared_px'][j])
            point.append(p)
        direct=[p for p in point if p['human_state']=='DIRECT_VISIBLE'];non_direct=[p for p in point if p['human_state']!='DIRECT_VISIBLE']
        item=dict(id=uid,session=row['session'],severity=severity[uid],classification=classification,
            initial_phase=initial_phase,selected_phase=selected_phase,
            metrics={method:by[uid,method]['metric'] for method in ('INITIAL','SEGMENTED_RGB_CURRICULUM','JOINT_SINGLE_FAMILY','JOINT_WD_BRANCH')},
            common_pool=ids,common_weights=weights,pool_weight_stats=stats(weights),
            effective_weighted_support=float(np.sum(weights)**2/np.sum(np.square(weights))),
            pool_DIRECT_count=len(direct),pool_DIRECT_corrected_le8=sum(p['corrected_GT_error_px']<=8 for p in direct),
            pool_nonDIRECT_states=dict(Counter(p['human_state'] for p in non_direct)),
            pool_DIRECT_raw_error=stats([p['raw_GT_error_px'] for p in direct]),
            pool_DIRECT_corrected_error=stats([p['corrected_GT_error_px'] for p in direct]),
            pool_shift_px=stats([p['RGB_shift_px'] for p in point]),
            cost_original_px2=branch['original_cost'],cost_alternate_px2=branch['alternate_cost'],cost_gap_px2=branch['cost_improvement_px2'],
            original_fit_residual_px=stats(branch['candidates'][0]['residuals_shared_px']),
            alternate_fit_residual_px=stats(branch['candidates'][1]['residuals_shared_px']),
            original_support_geometry=branch['candidates'][0]['shared_geometry'],alternate_support_geometry=branch['candidates'][1]['shared_geometry'],
            selected_curriculum_accepted=branch['candidates'][1]['curriculum']['accepted'],
            selected_used_ids=branch['candidates'][1]['curriculum']['used_ids'],fixed_H=row['self_hidden'],pool_points=point,
            overlay='branch_qa_v1/'+classification+'_'+uid.replace(':','_')+'.png')
        render(frames[uid],row,meta,gt,item,RUN/item['overlay']);details.append(item)
    counts=Counter(d['classification'] for d in details);assert counts=={'recovery':9,'false_flip':3}
    assert all(d['severity']=='SEVERE_OCCLUSION' for d in details)
    report=dict(schema='POSTSEAL_RETIRED_MIXED_SEVERITY_BRANCH_DIAGNOSTICS_V1',
        status='retired mixed-severity research; user now requires clean/moderate-only training and primary evaluation',
        frozen_population=97,flips=len(details),classification_counts=dict(counts),initial_phase_counts=dict(phase_counts),
        flip_severity_counts=dict(Counter(d['severity'] for d in details)),
        flip_session_counts=dict(Counter(d['session'] for d in details)),
        all_flips_in_new_excluded_severe_scope=True,policy_or_model_retuning=0,FINAL_reads=0,
        input_bindings={name:sha(RUN/name) for name in ('PREDICTIONS.jsonl.gz','PREDICTION_SEAL.json','SCORED_ROWS.jsonl.gz','ISOLATED_TEST_SCORING.json','ISOLATED_TEST_INPUTS.json')},
        analysis_code_sha256=sha(__file__),phase_rule='Cuboid native corners@Q compared to annotation body corners@bodyQ under proper C2 physical symmetry; no dimension-only family label',
        assessment=['All12 flips were severe and belong to retired scope; no direction-flip gains apply to new nonsevere cohort.',
            'Each false flip pool included only4 humanDIRECT corners; two or three remained inaccurate beyond8px, plus occluded/self/unannotated support.',
            'Allfalse alternatives lowered fitted residual on the same noisy shared observations, yet physical rotation moved to the wrong90degree family: lower in-sample residual did not certify native physical correspondence.',
            'The two pallet07 false pools used uniformly weak floor-level coordinate weights; RGB deltas of roughly1-5px did not resolve ownership or W/D ambiguity.',
            'All12 flipsoccur in two held-session scene families absent from CAL; evidence indicates strong scene dependence, while severe scope now excludes thesecases.',
            'No new rejection threshold, reliability rule or model parameter was selected from these TEST cases.'],records=details)
    (RUN/'BRANCH_DIAGNOSTICS.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('BRANCH_DIAGNOSTICS',dict(counts),'all severe',len(details),flush=True)


if __name__=='__main__':main()
