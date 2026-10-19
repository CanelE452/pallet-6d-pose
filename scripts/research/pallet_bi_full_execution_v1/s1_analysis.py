"""Postseal S1 fixed-cohort attribution and rights-scoped derived gallery."""
from __future__ import annotations
import argparse,csv,hashlib,html,json,math,time
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image,ImageDraw
from . import s1_union as U,independent_check as I

SEEDS=(20261018,20261019,20261020)
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))

def analyze(d1,run,reference_dir,output):
    started=time.monotonic();d1,run,reference_dir,output=map(Path,(d1,run,reference_dir,output));output.mkdir(parents=True,exist_ok=True)
    if not U.read(run/'S1_INDEPENDENT/INDEPENDENT_VERIFICATION.json')['PASS']:raise ValueError('Independent postseal verification required')
    refs=U.read(reference_dir/'FIXED_REFERENCES.json')['references'];data={r['id']:r for r in U.rows(run/'S1_INDEPENDENT/INDEPENDENT_ROWS.jsonl.gz')}
    originals=list(U.rows(d1/'d1_real_candidates_v1/HEAD_FEATURES.jsonl.gz'));unions=list(U.rows(run/'s1_real_union/HEAD_FEATURES.jsonl.gz'));coverage=[]
    for i,(a,b)in enumerate(zip(originals,unions)):
        r=data[a['id']];ref=refs[r['partition']][a['id']];raw=next(U.rows(d1/'d1_real_candidates_v1/frames'/f'{i:04d}'/'E_RUN.jsonl.gz'))
        oldgen=any(I.metric(dict(available=True,R_physical=c['R_physical'],centroid=c['centroid']),ref)['success']for c in raw['candidates'])
        oldgood=[c['candidate_id']for c in a['candidates']if I.metric(c['actual_pose'],ref)['success']]
        eligible=[c['candidate_id']for c in a['candidates']if c['eligible']and c['candidate_id']in oldgood]
        ugood=[c['candidate_id']for c in b['candidates']if I.metric(c['actual_pose'],ref)['success']]
        ueligible=[c['candidate_id']for c in b['candidates']if c['eligible']and c['candidate_id']in ugood]
        pfgood=any(r['metrics'][m]['success']for m in U.METHODS)
        coverage.append(dict(id=a['id'],partition=r['partition'],session=r['session'],cluster_id=r['cluster_id'],old_generated_good=oldgen,
            old_unique_good=bool(oldgood),old_eligible_good=bool(eligible),union_generated_good=oldgen or pfgood,union_unique_good=bool(ugood),union_eligible_good=bool(ueligible),
            old_good_unique_ids=oldgood,old_good_eligible_ids=eligible,union_good_unique_ids=ugood,union_good_eligible_ids=ueligible,PF_good=pfgood,
            old_selected={str(s):r['metrics']['D1_H1_MLP_s'+str(s)]['success']for s in SEEDS},
            union_selected={str(s):r['metrics']['S1_UNION_s'+str(s)]['success']for s in SEEDS}))
    primary=[r for r in coverage if r['partition']!='CAL']
    fixed=dict(original31_selection_failure=[r['id']for r in primary if r['old_generated_good']and not r['old_selected']['20261018']],
        original41_candidate_absent=[r['id']for r in primary if not r['old_generated_good']],
        PF_union38_candidate_absent=[r['id']for r in primary if not r['union_generated_good']],
        PF_new_good=[r['id']for r in primary if r['PF_good']and not r['old_generated_good']],
        PF_good_old_bank_good_H1_fail=[r['id']for r in primary if r['PF_good']and r['old_generated_good']and not r['old_selected']['20261018']])
    rootsets=U.read(run/'P0_FIXED_SETS/FIXED_SETS.json')['sets']
    if any(set(rootsets[k])!=set(v)for k,v in fixed.items()):raise AssertionError('Independent fixed failure sets differ')
    byid={r['id']:r for r in coverage};cohorts={}
    for name,ids in fixed.items():
        records=[data[fid]for fid in ids];bank=[byid[fid]for fid in ids];cohorts[name]=dict(n=len(ids),ids=ids,
            old_generated_good=sum(r['old_generated_good']for r in bank),old_eligible_good=sum(r['old_eligible_good']for r in bank),
            union_generated_good=sum(r['union_generated_good']for r in bank),union_eligible_good=sum(r['union_eligible_good']for r in bank),seeds={})
        for seed in SEEDS:
            before='D1_H1_MLP_s'+str(seed);after='S1_UNION_s'+str(seed);changes=Counter();cats={k:[]for k in ('RECOVERY','DAMAGE','BOTH_SUCCESS','BOTH_FAILURE')}
            for r in records:
                old,new=r['metrics'][before]['success'],r['metrics'][after]['success'];cat='BOTH_SUCCESS'if old and new else'RECOVERY'if new else'DAMAGE'if old else'BOTH_FAILURE';cats[cat].append(r['id'])
            cohorts[name]['seeds'][str(seed)]=dict(old_selected_good=sum(r['metrics'][before]['success']for r in records),union_selected_good=sum(r['metrics'][after]['success']for r in records),
                transitions={k:len(v)for k,v in cats.items()},transition_ids=cats,continuous=I.aggregate(records,[before,after]))
    bankcounts={pop:{k:sum(r[k]for r in coverage if r['partition']in parts)for k in ('old_generated_good','old_unique_good','old_eligible_good','union_generated_good','union_unique_good','union_eligible_good')}for pop,parts in I.POPS.items()}
    pp=U.read(run/'S1_INDEPENDENT/INDEPENDENT_PAIRED.json')['PRIMARY192'];gallery={}
    for key,pair in pp.items():
        for category in ('RECOVERY','DAMAGE'):
            for fid in pair['ids'][category]:gallery.setdefault(fid,[]).append(dict(contrast=key,category=category))
    # Fixed display selection: all transitions; hash-first normal and each
    # fixed failure cohort, plus the largest translation/rotation error.
    normal=[r['id']for r in data.values()if r['partition']!='CAL'and all(r['metrics']['S1_UNION_s'+str(s)]['success']for s in SEEDS)]
    categories={'HASH_FIRST_NORMAL':normal,'HASH_FIRST_ORIGINAL_CANDIDATE_ABSENT':fixed['original41_candidate_absent'],'HASH_FIRST_ORIGINAL_SELECTION_FAILURE':fixed['original31_selection_failure']}
    for label,ids in categories.items():
        for fid in sorted(ids,key=lambda x:hashlib.sha256(x.encode()).hexdigest())[:2]:gallery.setdefault(fid,[]).append(dict(category=label))
    for axis in ('translation_cm','rotation_deg'):
        rows=[r for r in data.values()if r['partition']!='CAL'and r['metrics']['S1_UNION_s20261018'][axis]is not None]
        for r in sorted(rows,key=lambda r:(-r['metrics']['S1_UNION_s20261018'][axis],r['id']))[:2]:gallery.setdefault(r['id'],[]).append(dict(category='LARGEST_'+axis.upper()))
    with open(output/'ALL_PRIMARY_FAILURES.csv','w',newline='')as f:
        fields=['id','partition','cluster_id','seed','old_success','union_success','translation_cm','rotation_deg','old_generated_good','union_generated_good','union_eligible_good','failure_stage']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for b in primary:
            for s in SEEDS:
                m=data[b['id']]['metrics']['S1_UNION_s'+str(s)]
                if m['success']:continue
                writer.writerow(dict(id=b['id'],partition=b['partition'],cluster_id=b['cluster_id'],seed=s,old_success=b['old_selected'][str(s)],union_success=False,
                    translation_cm=m['translation_cm'],rotation_deg=m['rotation_deg'],old_generated_good=b['old_generated_good'],union_generated_good=b['union_generated_good'],union_eligible_good=b['union_eligible_good'],
                    failure_stage='CANDIDATE_ABSENT'if not b['union_generated_good']else'ADMISSION_LOSS'if not b['union_eligible_good']else'FINAL_SELECTION_LOSS'))
    U.save(output/'BANK_COVERAGE_ROWS.jsonl.gz',coverage);U.write(output/'FIXED_COHORTS.json',cohorts);U.write(output/'GALLERY_SELECTION.json',gallery)
    result=dict(complete=True,independent_fixed_set_equality=True,fixed_counts={k:len(v)for k,v in fixed.items()},bank_upper_bounds=bankcounts,
        selected_PRIMARY={str(s):sum(r['union_selected'][str(s)]for r in primary)for s in SEEDS},
        gallery_all_transition_ids=sorted({fid for p in pp.values()for cat in ('RECOVERY','DAMAGE')for fid in p['ids'][cat]}),gallery_total_cases=len(gallery),
        source_sha256=U.sha(__file__),reference_sha256=U.sha(reference_dir/'FIXED_REFERENCES.json'),wall_seconds=time.monotonic()-started)
    U.write(output/'ANALYSIS_SEAL.json',result);return result

def project(pose,frame,erun):
    if not pose.get('available'):return None,None
    fam=erun['families']['ORIGINAL_WD'];xyz=np.asarray(fam['native_xyz']);R=np.asarray(pose['R_physical'])@np.asarray(fam['physical_registration_Q']).T
    camera=(xyz-xyz.mean(0))@R.T+np.asarray(pose['centroid']);p=camera@np.asarray(frame['K']).T
    with np.errstate(divide='ignore',invalid='ignore'):uv=p[:,:2]/p[:,2:]
    return uv,camera[:,2]
def wire(ax,pose,frame,erun,color,style='-',alpha=1):
    p,z=project(pose,frame,erun)
    if p is None:return
    for a,b in EDGES:
        if np.isfinite(p[[a,b]]).all()and min(z[a],z[b])>0:ax.plot(p[[a,b],0],p[[a,b],1],color=color,ls=style,lw=1.3,alpha=alpha)
def fmt(metric):
    return 'NO POSE'if not metric['pose_available']else f"{metric['translation_cm']:.2f}cm / {metric['rotation_deg']:.2f}deg / "+('SUCCESS'if metric['success']else'FAIL')
def observation_panel(ax,frame,erun):
    spec=erun['packet']['solver_input'];revoked={b['native_corner']for b in spec['boundary_observations']};initial=np.asarray(frame['initial_points'])
    good=np.isfinite(initial).all(1);ax.scatter(initial[good,0],initial[good,1],marker='x',s=15,c='#ffb74d',label='N0 guide q')
    for obs in spec['visible']:
        k=obs['native_corner'];xy=obs['xy'];color='#00ced1'if obs['state']=='B'else'#cc8cff'
        ax.scatter(*xy,c=color,facecolors='none',s=24);ax.annotate(str(k)+':'+obs['state']+('/revoked'if k in revoked else''),xy,fontsize=6,color=color,xytext=(3,3),textcoords='offset points')
        if k not in revoked:
            cov=np.asarray(obs.get('fit_covariance_px2',obs['covariance_px2']));v,e=np.linalg.eigh(cov);angle=math.degrees(math.atan2(e[1,1],e[0,1]));ax.add_patch(Ellipse(xy,2*math.sqrt(max(0,v[1])),2*math.sqrt(max(0,v[0])),angle=angle,fill=False,ec=color,lw=.8))
    for part in spec['boundary_observations']:
        for line in part['lines']:
            points=np.asarray(line.get('provenance',{}).get('line_record',{}).get('observed_points',[]))
            if points.ndim==2 and points.shape[1]==2:ax.plot(points[:,0],points[:,1],'.-',c='#ff4f9a',lw=1,markersize=2)
    ax.set_title('Same original q / fit-covariance both arms\nB cyan; I purple; 1D support pink; owner unknown',fontsize=8)

def gallery(d1,run,reference_dir,analysis_dir,output):
    from scripts.research.pallet_boundary_action_repair_v1.display_privacy import display_image
    from scripts.research.pallet_statewise_end_to_end_recovery_v1 import solver as S
    started=time.monotonic();d1,run,reference_dir,analysis_dir,output=map(Path,(d1,run,reference_dir,analysis_dir,output));output.mkdir(parents=True,exist_ok=True)
    privacy=U.read(d1/'DISPLAY_PRIVACY_PLAN.json');rights=U.read(d1/'RIGHTS_SCOPE_BINDING.json');receipt=U.read(d1/'RIGHTS_RECEIPT.json')
    if U.sha(d1/'DISPLAY_PRIVACY_PLAN.json')!=rights['display_privacy_plan_sha256']or U.sha(d1/'RIGHTS_RECEIPT.json')!=rights['rights_receipt_sha256']:raise ValueError('Display authorization binding differs')
    selection=U.read(analysis_dir/'GALLERY_SELECTION.json');inputs=list(U.rows(U.paths(d1,'real')[0]));indices={f['id']:i for i,f in enumerate(inputs)}
    refs=U.read(reference_dir/'FIXED_REFERENCES.json')['references'];scored={r['id']:r for r in U.rows(run/'S1_INDEPENDENT/INDEPENDENT_ROWS.jsonl.gz')}
    preds={r['id']:r for r in U.rows(run/'s1_predictions/PREDICTIONS.jsonl.gz')};oldpred={r['id']:r for r in U.rows(d1/'PREDICTIONS.jsonl.gz')};features={r['id']:r for r in U.rows(run/'s1_real_union/HEAD_FEATURES.jsonl.gz')}
    config=U.read(d1.parent/'pallet-statewise-end-to-end-recovery-v1/CONFIG_FROZEN.json');cases=[];costcalls=0
    for case,fid in enumerate(sorted(selection)):
        if fid not in rights['approved_ids']:raise ValueError('Only rights-covered PRIMARY192 derived panels allowed')
        i=indices[fid];frame=inputs[i];src=d1/'d1_real_candidates_v1/frames'/f'{i:04d}';erun=next(U.rows(src/'E_RUN.jsonl.gz'));bundle=next(U.rows(src/'BOUNDARY_BUNDLE.jsonl.gz'))
        image=display_image(frame,privacy);r=scored[fid];ref=refs[r['partition']][fid];refpose=dict(available=True,R_physical=ref['R'],centroid=ref['t'])
        poses={};actions={}
        for s in SEEDS:
            for prefix,source in (('D1_H1_MLP',oldpred[fid]),('S1_UNION',preds[fid])):
                name=prefix+'_s'+str(s);p=source['predictions'][name];poses[name]=p['actual_pose'];actions[name]={k:p.get(k)for k in ('selected_candidate_id','selected_source_aliases','selected_score','fallback','prediction_status')}
        for m in U.METHODS:poses[m]=r['poses'][m]
        for family in erun['families']:
            c=next((c for c in erun['candidates']if c['candidate_id']==f'G13/{family}/selected'),None)
            if c:poses['G13_'+family]=dict(available=True,R_physical=c['R_physical'],centroid=c['centroid'])
        fig,axes=plt.subplots(3,3,figsize=(13.8,11));fig.suptitle(fid+' | '+r['partition']+' | '+', '.join(sorted({c['category']for c in selection[fid]})),fontsize=10)
        layout=[None,'D1_H1_MLP_s20261018','S1_UNION_s20261018',U.METHODS[0],'D1_H1_MLP_s20261019','S1_UNION_s20261019',U.METHODS[1],'D1_H1_MLP_s20261020','S1_UNION_s20261020']
        for ax,name in zip(axes.flat,layout):
            ax.imshow(image);ax.set_xlim(0,frame['raw_hw'][1]);ax.set_ylim(frame['raw_hw'][0],0);ax.set_xticks([]);ax.set_yticks([])
            if name is None:observation_panel(ax,frame,erun)
            else:
                wire(ax,refpose,frame,erun,'#1efc71','--',.85);wire(ax,poses[name],frame,erun,'#ff5252'if name.startswith('S1')else'#30a8ff')
                title=name.replace('_s202610',' seed ');ax.set_title(title+'\n'+fmt(r['metrics'][name]),fontsize=8)
                if name in actions:
                    alias='/'.join(str(actions[name].get('selected_candidate_id')).split('/')[:3]);ax.text(.01,.02,'chosen: '+alias,transform=ax.transAxes,color='white',fontsize=6,bbox=dict(facecolor='black',alpha=.65,pad=1))
        fig.text(.5,.012,'Green dashed = reference projection (evaluation only). Solid = saved estimate. q and covariance unchanged by S1.\nDisplay-only privacy masks; virtual cuboid role/line support is not independently verified material ownership.',ha='center',fontsize=8)
        fig.subplots_adjust(left=.015,right=.985,top=.94,bottom=.065,wspace=.05,hspace=.21);filename=f'case_{case:03d}.png';fig.savefig(output/filename,dpi=125);plt.close(fig)
        # Both physical W/D family readouts are displayed separately.
        family_names=[x for x in poses if x.startswith('G13_')];ff,aa=plt.subplots(1,len(family_names),figsize=(9.2,3.8),squeeze=False)
        for ax,name in zip(aa.flat,family_names):
            ax.imshow(image);wire(ax,refpose,frame,erun,'#1efc71','--');wire(ax,poses[name],frame,erun,'#ffbd30');ax.set_xlim(0,frame['raw_hw'][1]);ax.set_ylim(frame['raw_hw'][0],0);ax.set_xticks([]);ax.set_yticks([]);ax.set_title(name+'\n'+fmt(I.metric(poses[name],ref)),fontsize=8)
        ff.suptitle('Saved actual family readouts; no new pose solve | G10 chosen '+str(erun['selected_family']),fontsize=9);ff.tight_layout();familypng=f'case_{case:03d}_families.png';ff.savefig(output/familypng,dpi=110);plt.close(ff)
        inp=dict(frame,packet=erun['packet'],families=erun['families'],fallback=frame['initial_pose']);models=S.compile_frame(inp,config,bundle=bundle);costs={}
        for name,pose in poses.items():
            if not pose.get('available'):continue
            c=dict(candidate_id=name,R_physical=pose['R_physical'],centroid=pose['centroid']);scores=S.candidate_scores(c,models);costcalls+=1
            costs[name]={family:dict(scores=s['scores'],feasible=s['feasible'],fixed_parts=s['audits']['fixed'].get('parts'))for family,s in scores.items()}
        feat=features[fid];entry=dict(id=fid,categories=selection[fid],panel=filename,family_panel=familypng,original_RGB_sha256=frame['image_sha256'],
            privacy_rectangle_count=len(privacy['frames'][fid]['redactions']),display_only=True,metrics=r['metrics'],selected_actions=actions,
            generated_original_endpoints=len(erun['candidates']),union_unique=len(feat['candidates']),union_eligible=sum(c['eligible']for c in feat['candidates']),
            new_complete_action_admission=feat['s1_union'],family_selected_by_G10=erun['selected_family'],both_family_and_selected_pose_costs=costs,
            B_I_state_roles=erun['packet']['solver_input']['states'],physical_source_certification='UNKNOWN; automatic source/association provenance only',
            full_q=[dict(native_corner=m['native_corner'],xy=m['xy'],state=m['state'],source=m.get('source'),fit_covariance_px2=m.get('fit_covariance_px2',m['covariance_px2']),reliability_weight=m.get('fit_reliability_weight'))for m in erun['packet']['solver_input']['visible']],
            partial=[dict(native_corner=b['native_corner'],state=b.get('state'),lines=[dict(normal=l['normal_unit_xy'],variance=l['normal_variance_px2'],support_interval=l.get('support_tangent_interval_px'),source=l.get('provenance',{}).get('line_record',{}).get('measurement_source'))for l in b['lines']])for b in erun['packet']['solver_input']['boundary_observations']])
        U.write(output/f'case_{case:03d}.json',entry);cases.append({k:entry[k]for k in ('id','categories','panel','family_panel','privacy_rectangle_count')})
    cards=[]
    for j,c in enumerate(cases):
        cards.append('<article><h2>'+html.escape(c['id'])+'</h2><p>'+html.escape(', '.join(x['category']for x in c['categories']))+'</p><a href="'+c['panel']+'"><img src="'+c['panel']+'"></a><details><summary>Both family poses and complete cost/admission evidence</summary><img src="'+c['family_panel']+'"><p><a href="'+f'case_{j:03d}.json'+'">Derived numerical details</a></p></details></article>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>S1 paired failure gallery</title><style>body{font:15px system-ui;background:#f1f4f7;margin:24px;color:#162b3a}article{background:white;padding:16px;margin:20px 0;border-radius:8px}img{width:100%;height:auto}h2{font-size:18px}</style><h1>S1 complete paired recovery/damage gallery</h1><p>All three seeds; every recovery/damage ID, fixed normal/candidate-absent/selection-failure examples and largest errors. Derived research panels only; original RGB is not distributed. B/I labels are automatic roles, physical owner unknown. PRIMARY is reused development data.</p>'+''.join(cards))
    manifest=dict(complete=True,cases=cases,all_transition_ids=U.read(analysis_dir/'ANALYSIS_SEAL.json')['gallery_all_transition_ids'],
        case_count=len(cases),privacy_plan_sha256=U.sha(d1/'DISPLAY_PRIVACY_PLAN.json'),rights_scope_sha256=U.sha(d1/'RIGHTS_SCOPE_BINDING.json'),
        rights_receipt_sha256=U.sha(d1/'RIGHTS_RECEIPT.json'),raw_RGB_copied=False,raw_RGB_bulk_published=False,display_changes_affect_inference=False,
        postseal_diagnostic_individual_cost_evaluations=costcalls,new_pose_solver_calls=0,source_sha256=U.sha(__file__),
        artifact_sha256={p.name:U.sha(p)for p in output.iterdir()if p.is_file()},pixel_review='pending actual derived panel review',wall_seconds=time.monotonic()-started)
    U.write(output/'GALLERY_MANIFEST.json',manifest);return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['analyze','gallery'])
    for name in ('d1','run','reference-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--analysis-dir',type=Path);a=p.parse_args()
    if a.stage=='analyze':r=analyze(a.d1,a.run,a.reference_dir,a.output)
    else:r=gallery(a.d1,a.run,a.reference_dir,a.analysis_dir,a.output)
    print(U.canonical({k:r[k]for k in ('complete','wall_seconds')}),flush=True)
if __name__=='__main__':main()
