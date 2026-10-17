"""Independent NumPy/stdlib verification of the E0 information-contract audit.

No audit implementation imports, optimization, candidate scoring, calibration
fit, PnP, NN, or selector. Fixed reference projections verify historical CAL
arithmetic only; geometric vertex matching never authenticates semantic owner.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import time
import numpy as np

BASE=Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1')
STATE=BASE/'statewise_joint_ot_20261014_v2/nonsevere_v4_fix'
ARCHIVE=STATE/'boundary_recovery_v1/fixed_wd_archived_tangent_eval_v4_15/INPUTS.jsonl.gz'
MODEL=STATE/'calibration/final_adapter_CAL46/MODEL.json'
CALOBS=STATE/'fresh_observations_v2/CAL_OBSERVATIONS.jsonl.gz'
CALREF=BASE/'OOF_pose_v1/ISOLATED_CAL_SCORING.json'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())


def stream(p):
    with gzip.open(p,'rt')as f:
        for line in f:
            if line.strip():yield json.loads(line)


def table(p):
    with Path(p).open()as f:return list(csv.DictReader(f))


def true(v):return str(v)=='True'


def group(v):return v['state']+(':'+'fallback'if v.get('source')=='VISIBLE_BACKBONE_FALLBACK'else ':corrected')


def cuboid(extents):
    a,b,c=np.asarray(extents)/2
    return np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])


def yrot(angle):return np.array([[math.cos(angle),0,math.sin(angle)],[0,1,0],[-math.sin(angle),0,math.cos(angle)]])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--private',type=Path,required=True);parser.add_argument('--state-private',type=Path,required=True);a=parser.parse_args()
    start,cpu=time.monotonic(),time.process_time();repo=a.repo;folder=a.private/'e_contract_audit_v1';fail=[];checks=Counter();maximum=defaultdict(float);minimum=defaultdict(lambda:float('inf'))
    def check(ok,label,context=None):
        checks[label]+=1
        if not bool(ok):fail.append(dict(check=label,context=context))
    def close(x,y,label,context=None,atol=1e-9):
        x,y=np.asarray(x,float),np.asarray(y,float);gap=float(np.max(np.abs(x-y)))if x.size else 0.
        maximum[label]=max(maximum[label],gap);check(x.shape==y.shape and np.allclose(x,y,atol=atol,rtol=0),label,context)
    point=table(folder/'E_POINT_CONTRACT_ROWS.csv');pointmap={(r['id'],int(r['native_corner'])):r for r in point}
    frames=table(folder/'E_FRAME_CONTRACT_ROWS.csv');framemap={r['id']:r for r in frames}
    partial=table(folder/'E_PARTIAL_CONTRACT_ROWS.csv');partialmap={(r['id'],int(r['native_corner'])):r for r in partial}
    cp=table(folder/'CAL_PROVENANCE_POINTS.csv');cpmap={(r['id'],int(r['native_corner'])):r for r in cp};summary=read(folder/'CAL_PROVENANCE_SUMMARY.json');model=read(repo/MODEL)
    archive=repo/ARCHIVE;typed=a.state_private/'e_trace_v2/TYPED_PACKETS.jsonl.gz';trace=a.state_private/'e_trace_v2/E_TRACE.jsonl.gz'
    counts=defaultdict(Counter);sets=defaultdict(set);verified_ids=[]
    for p,old,et in zip(stream(typed),stream(archive),stream(trace),strict=True):
        fid=p['id'];verified_ids.append(fid);cohort='CAL46'if p['partition']=='CAL'else 'PRIMARY192';inp=p['solver_input'];act=p['activation'];f=framemap[fid];source=old['finish_input']['_original_source'];original={int(x['native_corner']):x for x in source['calibrated_visible_measurements']}
        check(fid==old['id']==et['id'],'stream_ID_identity',fid);check(f['partition']==p['partition'] and f['session']==p['session'],'frame_identity',fid)
        complete=set(act['complete_ids']);partids=set(act['all_partial_ids']);check(not complete&partids,'full_partial_disjoint',fid)
        visids={int(v['native_corner'])for v in inp['visible']};check(visids==complete|partids,'visible_equals_full_union_partial',fid)
        close(sorted(complete),sorted(et['consumer_authoritative_activation']['complete_ids']),'trace_complete_identity',fid)
        counts[cohort]['frames']+=1;counts[cohort]['points']+=len(visids);counts[cohort]['full_points']+=len(complete);counts[cohort]['partial_archived_points']+=len(partids)
        for state in 'SBIOTU':check(int(f[state+'_count'])==inp['states'].count(state),'frame_state_count',dict(id=fid,state=state))
        check(int(f['full_count'])==len(complete)and int(f['partial_count'])==len(partids),'frame_factor_counts',fid)
        check(not true(f['reference_is_independent_physical_GT'])and not true(f['same_RGB_point_partial_outline_independence_certified']),'no_independence_claim',fid)
        Q=np.asarray(p['registered_family']['physical_registration_Q']);native=np.asarray(inp['xyz']);oldg=source['initial_geometry'];oldphysical=np.asarray(oldg['native_xyz'])@np.asarray(oldg['physical_registration_Q']);physical=native@Q
        close(Q.T@Q,np.eye(3),'proper_Q_orthogonality',fid);close(np.linalg.det(Q),1.,'proper_Q_determinant',fid)
        close(native,p['registered_family']['native_xyz'],'registered_native_XYZ',fid)
        distance=np.linalg.norm(physical[:,None]-oldphysical[None],axis=-1);owner=distance.argmin(1);gap=max(float(distance.min(1).max()),float(distance.min(0).max()));maximum['unordered_physical_vertex_gap_m']=max(maximum['unordered_physical_vertex_gap_m'],gap)
        check(gap<1e-7 and len(set(owner))==8,'unordered_registered_bijection',fid)
        R=np.asarray(p['e0_pose']['R_cf']);RP=np.asarray(p['e0_pose']['R_physical']);close(R@Q,RP,'physical_pose_Q',fid);close(R.T@R,np.eye(3),'proper_pose_R',fid)
        scale=float(source['ot_domains']['nonself_prediction_scale_px']);ratio=float(np.clip(scale/model['reference_scale_px'],.5,2.))
        bobs={int(b['native_corner']):b for b in p['original_e_legacy_boundary_observations']}
        bobs.update({int(b['native_corner']):b for b in inp['boundary_observations']})
        for v in inp['visible']:
            k=int(v['native_corner']);key=(fid,k);r=pointmap[key];o=original[k];state=v['state'];role='FULL'if k in complete else 'PARTIAL_ARCHIVE';cov=np.asarray(v['covariance_px2']);fit=np.asarray(v['fit_covariance_px2']);g=group(v);mg=model['groups'][g]
            check(state in ('B','I')and inp['states'][k]==state and not v['measurement_is_pose_projection']and v['image_measurement'],'actual_BI_measurement_only',key)
            check(r['role']==role and r['state']==state and int(r['physical_owner_original_native_id'])==int(owner[k]),'point_state_role_owner',key)
            check(r['owner_semantic_status']=='SEMANTIC_OWNER_UNVERIFIED','semantic_owner_unverified',key)
            changed_source=o.get('source')!=v.get('source');changed_cov=not np.array_equal(np.asarray(o['covariance_px2']),cov);changed_xy=not np.array_equal(np.asarray(o['xy']),v['xy']);expected=model['covariance_scale']*cov+float(mg['floor_px2'])*ratio**2*np.eye(2)
            check(true(r['source_changed'])==changed_source and true(r['conditional_covariance_changed'])==changed_cov and true(r['xy_changed'])==changed_xy,'change_flags',key)
            check(r['original_source']==o.get('source')and r['actual_source']==v.get('source')and r['original_cal_group']==group(o)and r['current_cal_group']==g,'point_sources_groups',key)
            close(fit,o['fit_covariance_px2'],'old_fit_override_exact_copy',key,atol=0);close(v['fit_reliability_weight'],o['fit_reliability_weight'],'old_reliability_exact_copy',key,atol=0)
            actualgap=float(np.max(np.abs(fit-expected)));close(actualgap,float(r['current_adapter_covariance_max_abs_gap_px2']),'current_adapter_gap_csv',key)
            close(math.sqrt(float(np.linalg.eigvalsh(cov).max())),float(r['observed_std_max_px']),'observed_sigma_csv',key)
            close(math.sqrt(float(np.linalg.eigvalsh(fit).max())),float(r['fit_std_max_px']),'fit_sigma_csv',key)
            close(math.sqrt(float(np.linalg.eigvalsh(expected).max())),float(r['reapplied_fit_std_max_px']),'current_adapter_sigma_csv',key)
            close(mg['reliability_weight'],float(r['current_adapter_reliability']),'current_adapter_reliability_csv',key)
            close(v['fit_reliability_weight'],float(r['actual_reliability']),'actual_reliability_csv',key)
            e=float(np.linalg.eigvalsh(fit-cov).min());minimum['fit_minus_current_cov_min_eigenvalue_px2']=min(minimum['fit_minus_current_cov_min_eigenvalue_px2'],e);close(e,float(r['fit_minus_current_observed_min_eigenvalue_px2']),'PSD_difference_csv',key);check(e>=-1e-9,'fit_not_overconfident_vs_conditional_PSD',key)
            check(np.linalg.eigvalsh(fit).min()>0 and np.linalg.eigvalsh(cov).min()>0,'covariance_SPD',key)
            W=np.linalg.solve(np.linalg.cholesky(fit),np.eye(2));close(W.T@W,np.linalg.inv(fit),'own_Cholesky_inverse_identity',key,atol=1e-12)
            mismatch=actualgap>1e-9 or abs(float(v['fit_reliability_weight'])-mg['reliability_weight'])>1e-9
            check(true(r['actual_vs_current_adapter_changed'])==mismatch,'adapter_change_flag',key)
            for label,flag in [('source_changed_points',changed_source),('covariance_changed_points',changed_cov),('adapter_mismatch_points',mismatch),('adapter_mismatch_full_points',mismatch and k in complete),('same_channel_owner_changed_points',int(owner[k])!=k)]:counts[cohort][label]+=int(flag)
            if mismatch:sets[cohort].add(fid)
            if k in partids:
                b=bobs[k];pr=partialmap[key];variance=[float(line['normal_variance_px2'])for line in b['lines']]
                check(not true(pr['complete_cost_active'])and true(pr['grouped_radial_loss_not_independent_line_sum']),'partial_not_promoted_to_full',key)
                check(int(pr['line_count'])==len(variance),'partial_line_count',key)
                if variance:close([min(variance),max(variance)],[float(pr['normal_variance_min_px2']),float(pr['normal_variance_max_px2'])],'partial_variance_csv',key)
                close(b['fit_reliability_weight'],float(pr['reliability']),'partial_reliability_csv',key)
    check(len(verified_ids)==238 and len(set(verified_ids))==238,'exact238_IDs')
    for cohort,n in [('CAL46',46),('PRIMARY192',192)]:
        counts[cohort]['mismatch_frame_count']=len(sets[cohort])
        for key,value in summary['by_cohort'][cohort].items():
            if key=='fit_minus_current_observed_negative_eigenvalue_points':check(value==0,'summary_no_negative_PSD',cohort)
            else:check(counts[cohort][key]==value,'summary_cohort_reaggregation',dict(cohort=cohort,key=key,expected=value,actual=counts[cohort][key]))
    # Independent historical CAL geometry/owner convention and fixed projections.
    refs=read(repo/CALREF)['rows'];calcounts=Counter();eligiblegroups=Counter();sessions=Counter();sourcecounts=Counter();allgroups=Counter();human=Counter();states=Counter();compatibleframes=[]
    for row in stream(repo/CALOBS):
        fid=row['id'];runtime=row['runtime'];g=runtime['initial_geometry'];label=refs[fid];truth=label['target']['truth'];Q=np.asarray(g['physical_registration_Q']);xyz=np.asarray(g['native_xyz']);refR=np.asarray(truth['R']);Qref=np.asarray(truth['body_R']).T@refR;mapping=np.asarray(label['metadata']['native_to_reference_corner_ids'],int);mapped=cuboid(truth['body_xyz'])[mapping]@Qref
        hypotheses=[yrot(j*2*np.pi/int(truth.get('order',1)))for j in range(int(truth.get('order',1)))];gaps=[float(np.linalg.norm(xyz@Q@S.T-mapped,axis=1).max())for S in hypotheses];phase=int(np.argmin(gaps));compatible=gaps[phase]<1e-6
        if compatible:compatibleframes.append(fid)
        check(compatible==(fid in model['compatible_fit_ids']),'CAL_geometry_compatibility_vs_model',fid)
        scale=float(runtime['ot_domains']['nonself_prediction_scale_px']);ratio=float(np.clip(scale/model['reference_scale_px'],.5,2));sessions[row['session']]+=1
        for v in runtime['frozen_visible_measurements']:
            k=int(v['native_corner']);r=cpmap[(fid,k)];cov=np.asarray(v['covariance_px2']);key=group(v);mg=model['groups'][key];effective=model['covariance_scale']*cov+mg['floor_px2']*ratio**2*np.eye(2);hs=label['metadata']['human_states_native'][k]
            calcounts['points']+=1;calcounts['compatible']+=int(compatible);eligiblegroups[key]+=int(compatible);sourcecounts[v.get('source')]+=1;allgroups[key]+=1;human[hs]+=1;states[v['state']]+=1
            check(r['source']==v.get('source')and r['group']==key and r['human_state']==hs and true(r['compatible'])==compatible,'CAL_point_provenance',dict(id=fid,corner=k))
            close([math.sqrt(np.linalg.eigvalsh(cov).max()),math.sqrt(np.linalg.eigvalsh(effective).max()),mg['reliability_weight']],[float(r['observed_std_max_px']),float(r['effective_std_max_px']),float(r['reliability'])],'CAL_fixed_sigma_reliability',dict(id=fid,corner=k))
            if compatible:
                cam=xyz[k]@(refR@hypotheses[phase]@Q.T).T+np.asarray(truth['t']);h=cam@np.asarray(row['K']).T;res=float(np.linalg.norm(np.asarray(v['xy'])-h[:2]/h[2]));close(res,float(r['reference_residual_px']),'CAL_original_semantic_fixed_reference_residual',dict(id=fid,corner=k),atol=1e-8)
            refid=int(mapping[k])
            if hs=='DIRECT_VISIBLE'and label['target']['valid'][refid]:close(np.linalg.norm(np.asarray(v['xy'])-np.asarray(label['target']['gt'][refid])),float(r['direct_annotation_residual_px']),'CAL_direct_annotation_residual',dict(id=fid,corner=k),atol=1e-8)
    check(calcounts['points']==267 and calcounts['compatible']==254 and len(compatibleframes)==44,'CAL_exact_fit_scope')
    for key,value in model['groups'].items():check(eligiblegroups[key]==value['count'],'CAL_eligible_group_count',key)
    check(dict(sessions)==summary['original_fit_session_counts'],'CAL_session_counts')
    for observed,expected,label in [(sourcecounts,summary['original_fit_source_counts'],'CAL_source_counts'),(allgroups,summary['original_fit_group_counts'],'CAL_all_group_counts'),(human,summary['original_fit_human_counts'],'CAL_human_counts'),(states,summary['original_fit_state_counts'],'CAL_state_counts')]:check(dict(observed)==expected,label)
    # Independent distribution arithmetic, including original CAL provenance and
    # all CAL/PRIMARY source groups in the published scale tables.
    def distributions(path,source,filters):
        for r in table(path):
            selected=[x for x in source if all(x.get(src)==r[out]for src,out in filters)]
            values=np.asarray([float(x[r['field']])for x in selected if x.get(r['field'])not in ('',None)],float)
            check(len(values)==int(r['n']),'distribution_denominator',dict(table=path.name,row=r))
            if len(values):
                expected=[values.mean(),np.std(values,ddof=1)if len(values)>1 else None,np.median(values),np.quantile(values,.9),np.quantile(values,.95),values.max()]
                for key,val in zip(['mean','sample_sd','median','p90','p95','maximum'],expected):
                    if val is not None:close(val,float(r[key]),'distribution_'+key,dict(table=path.name,field=r['field']),atol=1e-8)
    distributions(folder/'CAL_PROVENANCE_DISTRIBUTIONS.csv',cp,[('session','session'),('group','group'),('source','source'),('human_state','human_state')])
    enriched=[dict(r,cohort='CAL46'if r['partition']=='CAL'else 'PRIMARY192')for r in point]
    distributions(folder/'E_SCALE_DISTRIBUTIONS.csv',enriched,[('cohort','cohort'),('session','session'),('state','state'),('role','role'),('actual_source','actual_source')])
    inputs=[typed,trace,archive,repo/MODEL,repo/CALOBS,repo/CALREF]+[folder/n for n in ['E_POINT_CONTRACT_ROWS.csv','E_FRAME_CONTRACT_ROWS.csv','E_PARTIAL_CONTRACT_ROWS.csv','CAL_PROVENANCE_POINTS.csv','CAL_PROVENANCE_SUMMARY.json','CAL_PROVENANCE_DISTRIBUTIONS.csv','E_SCALE_DISTRIBUTIONS.csv']]
    receipt=dict(schema='INDEPENDENT_E0_INFORMATION_CONTRACT_VERIFICATION_V1',passed=not fail,failures=fail,check_counts=dict(checks),maximum_numeric_errors=dict(maximum),minimum_numeric_values=dict(minimum),cohorts={k:dict(v)for k,v in counts.items()},CAL267=dict(counts=calcounts,compatible_frames=len(compatibleframes),eligible_group_counts=eligiblegroups,sessions=sessions),
        source_sha256=sha(__file__),input_sha256={str(p):sha(p)for p in inputs},independent_implementation=True,auditor_imported=False,own_Cholesky_Q_geometry_fixed_projection_and_distribution_arithmetic=True,semantic_owner_status='UNVERIFIED: geometric matching and historical reference phase are not image-semantic truth',reference_status='Existing geometry-derived CAL reference; no independent physical metrology',new_fits=0,new_candidate_costs=0,new_pose_selection=0,new_NN_forwards=0,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with(a.private/'INDEPENDENT_CONTRACT_CHECK.json').open('x')as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=not fail,failures=len(fail),checks=sum(checks.values()),wall_seconds=receipt['wall_seconds'],max_errors=dict(maximum))))
    if fail:raise SystemExit(1)


if __name__=='__main__':main()
