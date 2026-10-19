"""Reference-free whole-population F policy/input audit before pose production."""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import argparse
import numpy as np
from . import s1_union as U, qcov as Q, calibration as C


def canonical(selection):
    result=deepcopy(selection)
    for point in result['full_points']:point.pop('uncertainty_calibration',None)
    return result


def run(d1,observations,calibration,output):
    d1,observations,calibration,output=map(Path,(d1,observations,calibration,output))
    source=U.read(observations/'OBSERVATION_MANIFEST.json');packets=U.read(calibration/'PACKET_MANIFEST.json')
    if source.get('complete')is not True or packets.get('complete')is not True:
        raise ValueError('Complete own-observation and CAL packet manifests required')
    if source['rows']!=1241 or packets['rows']!=1241:raise ValueError('Full1241 required')
    if packets['observations_sha256']!=U.sha(observations/'OBSERVATION_MANIFEST.json'):
        raise ValueError('Calibration observation ancestry differs')
    if source['producer_lock_sha256']!=U.sha(observations/'PRODUCER_LOCK.json'):
        raise ValueError('Observation producer lock changed')
    if packets['models_sha256']!=U.sha(calibration/'MODELS.json'):
        raise ValueError('Own calibration model bytes changed')
    access=U.read(calibration/'CAL_REFERENCE_ACCESS.json')
    if access['numeric_reference_decodes']!={'CAL':46,'PRIMARY':0}:raise ValueError('CAL-only fitting required')
    entries={(r['domain'],r['index']):r for r in source['frames']};pentries={(r['domain'],r['index']):r for r in packets['frames']}
    counts=Counter();armcounts={a:Counter()for a in ('COMMON','BI')};records=[]
    for domain,name in [('real','REAL238'),('synthetic','SYNTHETIC1003')]:
        upstream=d1/f'd1_upstream_v1/{name}_UPSTREAM.jsonl.gz'
        frames=list(U.rows(upstream));oldroot=d1/f'd1_{domain}_observations_v1';oldseal=U.read(oldroot/'OBSERVATION_SEAL.json')
        if U.sha(upstream)!=oldseal['input_sha256']:raise ValueError('Frozen input ancestry changed')
        for index,frame in enumerate(frames):
            folder=observations/domain/f'{index:04d}';fs=U.read(folder/'FRAME_SEAL.json');entry=entries[(domain,index)]
            if entry['id']!=frame['id']or U.sha(folder/'FRAME_SEAL.json')!=entry['seal_sha256']:
                raise ValueError('Observation frame ancestry changed')
            oldfolder=oldroot/'frames'/f'{index:04d}';oldfs=U.read(oldfolder/'OBSERVATION_SEAL.json')
            if U.sha(oldfolder/'OBSERVATION_SEAL.json')!=oldseal['frame_seals_sha256'][f'{index:04d}']:
                raise ValueError('Frozen observation row ancestry changed')
            expected=dict(frame_sha256=U.payload_sha(frame),observation_sha256=U.sha(oldfolder/'OBSERVATION.jsonl.gz'),
                mask_sha256=U.sha(oldfolder/'MASK.npz'),producer_lock_sha256=U.sha(observations/'PRODUCER_LOCK.json'))
            if fs['dependencies']!=expected or oldfs['input_row_sha256']!=U.payload_sha(frame):
                raise ValueError('Actual F input/observation/mask/producer dependency differs')
            if expected['observation_sha256']!=oldfs['observation_sha256']or expected['mask_sha256']!=oldfs['mask_sha256']:
                raise ValueError('Frozen observation/mask bytes changed')
            for filename,digest in fs['artifacts'].items():
                if U.sha(folder/filename)!=digest:raise ValueError('Raw observation bytes changed')
            pp=calibration/'packets'/domain/f'{index:04d}.json.gz'
            if U.sha(pp)!=pentries[(domain,index)]['packet_sha256']:raise ValueError('Exported F packet changed')
            packet=Q.read(pp);raw=next(U.rows(folder/'SELECTIONS.jsonl.gz'));pool=next(U.rows(folder/'POOL.jsonl.gz'))
            if any(x['id']!=frame['id']for x in (packet,raw,pool)):raise ValueError('Frame identity differs')
            fixed_states=next(U.rows(oldfolder/'OBSERVATION.jsonl.gz'))['states']
            if pool['states']!=fixed_states:raise ValueError('Frozen automatic states changed')
            q0=np.asarray(frame['initial_points']);per={};prefix=domain+'/'+frame['partition']
            for corner in pool['corners']:
                if corner['state']not in ('B','I')and (corner['queried']or corner['rays']or corner['candidates']):
                    raise ValueError('S/O/T/U anchor generated a local image query')
                counts[prefix+'/queried_'+corner['state']]+=int(corner['queried'])
            for arm in ('COMMON','BI'):
                original=raw['selections'][arm];selected=packet['selections']['F_'+arm]
                if original['states']!=fixed_states or selected['states']!=fixed_states:
                    raise ValueError('Policies changed automatic visibility/state')
                points,partial,_=Q.effective(selected)
                expected_corners={k for k,state in enumerate(fixed_states)if state in ('B','I')}
                corners={c['native_corner']:c for c in original['selected_candidates']}
                if set(corners)!=expected_corners or len(corners)!=len(original['selected_candidates']):
                    raise ValueError('Every B/I native anchor requires exactly one recorded policy choice')
                if any(float(p.get('fit_reliability_weight',1.))!=1. for p in partial):
                    raise ValueError('F partial-factor Gamma is not fixed1')
                c=armcounts[arm];summary=[]
                for k,choice in sorted(corners.items()):
                    state=selected['states'][k];newdim=choice['new_cue_dimension']
                    c[prefix+'/'+state+'/new_cue_dimension'+str(newdim)]+=1
                    factor_dimension=2 if k in points else sum(len(x['lines'])for x in partial if x['native_corner']==k)
                    if factor_dimension not in (0,1,2):raise ValueError('Unexpected factor dimension')
                    c[prefix+'/'+state+'/effective_dimension'+str(factor_dimension)]+=1
                    item=dict(native_corner=k,state=state,new_cue_dimension=newdim,effective_dimension=factor_dimension)
                    if k in points:
                        row=points[k];weight=float(row.get('fit_reliability_weight',1.))
                        if weight!=1.:raise ValueError('F complete-factor Gamma is not fixed1')
                        cov=np.asarray(Q.covariance(row));C._validate_covariance(cov)
                        item.update(displacement_from_N0_px=float(np.linalg.norm(np.asarray(row['xy'])-q0[k])),
                            fit_covariance_px2=cov.tolist(),reliability_weight=weight,physical_source_ids=row.get('physical_source_ids',[]),source=row.get('source'))
                    elif factor_dimension:
                        normal=next(x for x in partial if x['native_corner']==k)
                        item.update(normal_variances_px2=[x['normal_variance_px2']for x in normal['lines']],
                            physical_source_ids=[sid for x in normal['lines']for sid in x.get('physical_source_ids',[])],
                            conditional_1D_total_error_floor_calibrated=False)
                    else:
                        item.update(reason='raw selected cue removed by effective physical-source deduplication; original evidence stays archived')
                    summary.append(item)
                per[arm]=dict(factors=summary,full2D=len(points),normal1D=sum(len(x['lines'])for x in partial),
                    selected_action_digest=U.payload_sha(original['selected_candidates']),
                    effective_selection_digest=U.payload_sha(canonical(selected)))
            same=canonical(packet['selections']['F_COMMON'])==canonical(packet['selections']['F_BI'])
            counts[prefix+'/effective_equal'if same else prefix+'/effective_different']+=1
            counts[prefix+'/rows']+=1
            records.append(dict(domain=domain,index=index,id=frame['id'],partition=frame['partition'],split=frame['split'],
                effective_solver_selection_equal=same,arms=per,physical_target_certification='VIRTUAL_NATIVE_TARGET; material correspondence unverified',
                packet_sha256=U.sha(pp),observation_seal_sha256=U.sha(folder/'FRAME_SEAL.json')))
    if len(records)!=1241 or len(entries)!=1241 or len(pentries)!=1241:raise ValueError('Manifest duplicates/extra/missing rows')
    output.mkdir(parents=True,exist_ok=True);U.save(output/'ROWS.jsonl.gz',records)
    summary=dict(complete=True,rows=1241,source_sha256=U.sha(__file__),counts=dict(counts),arm_counts={k:dict(v)for k,v in armcounts.items()},
        observation_manifest_sha256=U.sha(observations/'OBSERVATION_MANIFEST.json'),packet_manifest_sha256=U.sha(calibration/'PACKET_MANIFEST.json'),
        own_CAL_access_sha256=U.sha(calibration/'CAL_REFERENCE_ACCESS.json'),rows_sha256=U.sha(output/'ROWS.jsonl.gz'),
        complete_factor_Gamma_fixed_1=True,all_typed_partial_revocations_already_validated_by_producer=True,
        effective_selection_comparison='remove only kernel-unused uncertainty_calibration metadata; all physical source IDs and numeric inputs preserved',
        normal_1D_noise_limitation='f_observer.choose projects its declared fixed6.5px2 fallback covariance onto the selected line normal. This is not an empirical fitted-line or total-error calibration. Full2D own CAL floor is not misrepresented as 1D calibration.',
        numeric_reference_reads_in_this_process=0,new_pose_optimizer_calls=0,new_H1_fits=0,final_F_complete=False)
    U.write(output/'SUMMARY.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('d1','observations','calibration','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();r=run(a.d1,a.observations,a.calibration,a.output);print({k:r[k]for k in ('complete','rows','counts')})
