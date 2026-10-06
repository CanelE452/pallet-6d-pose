"""Registered read-only final-F/ADDsym parity, with reusable durable probes."""
from pathlib import Path
import argparse,json,time
import numpy as np
import cv2
from .cost_cache import (ROOT,DOC,OLD,BASELINE_ROOT,Data,POSE,WD,PNP,SPECS,ReadOnlyBanks,
                        BinaryJournal,replay_journal,candidate_cost,hard_target,require_ready,
                        read,write,sha,hash_value,array_hash,pnp_counter,open_arrays)


def register_cases(data,bank,bank_cache):
    cases=[]
    for split,rows,name in [('SOURCE_SELECTION',data.oracle_rows,'A_SOURCE_ORACLE.json'),('SYNTH_HELDOUT',data.eval_rows,'A_SYNTH_HELDOUT_ORACLE.json')]:
        path=OLD/'results'/name;old={r['id']:r for r in read(path)['rows'] if r['arm']=='GEO'}
        for row in rows[:2]:
            f=data.source_frame(int(row));cases.append(dict(split=split,frame=f,bank=bank.get(int(row),f),oracle=old[f['id']],reference_file=str(path),reference_sha256=sha(path)))
    path=Path(bank_cache)/'sampling_masks/REAL_DEV_generated_banks.npz'
    prior=read(OLD/'results/A_SAMPLING_MASK_RECEIPT.json');expected=next(r for r in prior['external_mask_files'] if r['path'].endswith('/REAL_DEV_generated_banks.npz'))
    assert sha(path)==expected['sha256'] and path.stat().st_size==expected['bytes']
    z=dict(np.load(path));frames=data.real_frames();lookup={f['id']:f for f in frames};reference=OLD/'results/A_REAL_DEV_ORACLE.json';old={r['id']:r for r in read(reference)['rows'] if r['arm']=='GEO'}
    assert set(z['ids'])==set(lookup)==set(old) and len(lookup)==319
    for i in range(2):
        fid=str(z['ids'][i]);f=lookup[fid];n=int(z['counts'][i]);q=np.array(z['points'][i,:n],copy=True)
        assert np.array_equal(q[0],f['q'],equal_nan=True) and np.array_equal(q[:,8],np.broadcast_to(f['q'][8],q[:,8].shape),equal_nan=True)
        cases.append(dict(split='REAL_DEV',frame=f,bank=dict(points=q),oracle=old[fid],reference_file=str(reference),reference_sha256=sha(reference),real_bank_sha256=expected['sha256']))
    return cases


def _probe(case,ordinal,cache_dir,binding):
    """Journal each actual parity F exactly once; a crash cannot reset its count."""
    directory=Path(cache_dir)/'parity'/f'case_{ordinal:02d}';header_path=directory/'binding.json';f=case['frame'];bank=case['bank'];n=len(bank['points'])
    contract=dict(binding=binding,id=f['id'],split=case['split'],bank_q_sha256=array_hash(bank['points']),K=f['K'],xyz=f['xyz'],truth=f['truth'],reference_sha256=case['reference_sha256'])
    case_binding=hash_value(contract)
    if header_path.exists():
        assert read(header_path)['case_binding']==case_binding;arrays=open_arrays(directory,1)
    else:
        directory.mkdir(parents=True,exist_ok=True)
        arrays={}
        for key,(dtype,shape,initial) in SPECS.items():
            a=np.lib.format.open_memmap(directory/(key+'.npy'),mode='w+',dtype=dtype,shape=(1,*shape));a[:]=initial;a.flush();arrays[key]=a
        arrays['action_counts'][0]=n
        write(header_path,dict(case_binding=case_binding,contract=contract))
    path=directory/'attempts.bin';replayed=replay_journal(path,arrays,[0]);assert replayed['tail_bytes']==0,'Torn parity journal; do not retry'
    receipt_path=directory/'result.json'
    if receipt_path.exists():
        old=read(receipt_path);assert old['case_binding']==case_binding and old['status']=='PASS';assert len(replayed['completed'])==n
        return dict(old,invocation_new_F_calls=0)
    journal=BinaryJournal(path);new=0;check_errors=[];begin=time.monotonic();live_pose={}
    try:
        with pnp_counter() as counter:
            for index,q in enumerate(bank['points']):
                state=int(arrays['candidate_state'][0,index])
                if state==2:continue
                assert state==0,'Incomplete attempted parity F is never repeated'
                journal.attempt(0,index);arrays['candidate_state'][0,index]=1;before=counter.copy();new+=1
                result=candidate_cost(f,q)
                # Reuse the exact same returned F pose: metric comparison causes
                # no additional F or internal PnP call.
                original=POSE.metric((f['id'],result['pose'],f['truth']))
                if result['available']:
                    for key in ('ADDsym_m','translation_cm','rotation_deg'):
                        delta=abs(result[key]-original[key]);assert delta==0,(key,delta);check_errors.append(delta)
                else:assert not original['available'] and np.isposinf(result['ADDsym_m'])
                counts={k:counter[k]-before[k] for k in PNP};journal.complete(0,index,result,counts)
                arrays['cost_ADDsym_m'][0,index]=result['ADDsym_m'];arrays['translation_cm'][0,index]=result['translation_cm'];arrays['rotation_deg'][0,index]=result['rotation_deg'];arrays['WD_hypothesis'][0,index]=result['WD'];arrays['F_available'][0,index]=result['available'];arrays['candidate_state'][0,index]=2
                if index in (0,case['oracle']['index']):live_pose[index]=result['pose']
        for a in arrays.values():a.flush()
        target=hard_target(arrays['cost_ADDsym_m'][0,:n],arrays['F_available'][0,:n]);prior=case['oracle']
        assert target==prior['index'],('First minimum final ADDsym oracle index differs',f['id'],target,prior['index'])
        raw=float(arrays['cost_ADDsym_m'][0,0]);best=float(arrays['cost_ADDsym_m'][0,target]) if target>=0 else float('inf')
        assert (not np.isfinite(raw) and prior['raw_ADDsym_m'] is None) or abs(raw-prior['raw_ADDsym_m'])<=1e-7
        assert (not np.isfinite(best) and prior['oracle_ADDsym_m'] is None) or abs(best-prior['oracle_ADDsym_m'])<=1e-7
        for index,key in ((0,'raw'),(target,'oracle')):
            if index<0 or not prior[key]['available']:continue
            for metric in ('translation_cm','rotation_deg'):
                value=float(arrays[metric][0,index]);ref=prior[key][metric];assert abs(value-ref)<=max(1e-5,abs(ref)*2e-7),('FP32 metadata parity',metric,value,ref)
        result=dict(schema='final_F_ADDsym_registered_parity_case_v1',status='PASS',case_binding=case_binding,id=f['id'],split=case['split'],source_cache_row=f.get('cache_row'),actions=n,actual_F_calls=len(replayed['attempted'])+new,invocation_new_F_calls=new,completed_F_calls=n,PnP_counts={k:replayed['PnP_counts'][k]+counter[k] for k in PNP},old_oracle_index=prior['index'],new_oracle_index=target,old_raw_ADDsym_m=prior['raw_ADDsym_m'],new_raw_ADDsym_m=raw,old_oracle_ADDsym_m=prior['oracle_ADDsym_m'],new_oracle_ADDsym_m=best,NoOp_coordinate_exact=True,center_preserved=True,ADD_metric_original_bytecode_delta_max=max(check_errors,default=0),ADDsym_parity_tolerance_m=1e-7,metadata_FP32_parity_tolerance='max(1e-5,2e-7*abs(reference))',reference_file=case['reference_file'],reference_sha256=case['reference_sha256'],original_bank_or_real_saved_bank='READ_ONLY, regenerated0',pose_snapshots={str(k):v for k,v in live_pose.items()},seconds=time.monotonic()-begin,journal_sha256=sha(path))
        write(receipt_path,result);return result
    finally:journal.close()


def parity(source_root,bank_cache,cache_dir,bindings_path,preflight_path):
    require_ready(bindings_path,preflight_path);cv2.setNumThreads(1);begin=time.monotonic()
    data=Data(source_root);bank=ReadOnlyBanks(data,bank_cache);cases=register_cases(data,bank,bank_cache)
    binding=hash_value(dict(inputs_sha256=sha(bindings_path),protocol_sha256=sha(DOC/'PROTOCOL.json'),cost_code_sha256=sha(Path(__file__).parent/'cost_cache.py'),parity_code_sha256=sha(Path(__file__)),IDs=[c['frame']['id'] for c in cases],actions=[len(c['bank']['points']) for c in cases]))
    rows=[_probe(c,i,cache_dir,binding) for i,c in enumerate(cases)]
    result=dict(schema='final_F_ADDsym_pose_target_parity_v1',status='PASS',binding=binding,cases=rows,registered_cases=6,registered_case_selection='first2 originalSOURCE_SELECTION,first2 SYNTH_HELDOUT,first2 savedREAL_DEVbank IDs; code parity only, never training inputs',actual_F_calls=sum(r['actual_F_calls'] for r in rows),invocation_new_F_calls=sum(r['invocation_new_F_calls'] for r in rows),completed_F_calls=sum(r['completed_F_calls'] for r in rows),PnP_counts={k:sum(r['PnP_counts'][k] for r in rows) for k in PNP},NoOp_F_parity='PASS on all6 cases against originalcachedRAW',oracle_reproduction='PASS exact firstargmin nativeindex on all6 completebanks',same_native_bank=True,GT_isolation='candidate_cost passes only q,K,knownxyz,source to originalPOSE.infer; truth is used only after returnedfinalF to unchangedmetric',failed_candidate_cost='positiveinf, never reselect/fallback',all_fail_frame_target=-1,GT_symmetry='unchanged approved proper rotations; originalmetric bytecode',new_bank_generation=0,new_backbone_detector_calls=0,new_optimizer_updates=0,seconds_wall=time.monotonic()-begin)
    write(DOC/'POSE_TARGET_PARITY.json',result);print('POSE_TARGET_PARITY',json.dumps(result,ensure_ascii=False),flush=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','bank-cache','cache-dir','bindings','preflight'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();parity(a.source_root,a.bank_cache,a.cache_dir,a.bindings,a.preflight)
if __name__=='__main__':main()
