"""Root-authorized metadata assembly only; no binary stat/header/body or SAM."""
from pathlib import Path
import copy, sys
sys.dont_write_bytecode=True
H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
sys.path.insert(0,str(V/'direct_cache_preparation_a'))
import build_expanded_direct_control_cache as runner
c=runner.c
SCOPE=V/'ROOT_ACTUAL_EXPANDED_SOURCE_CACHE_ASSEMBLY_AND_EXECUTION_SCOPE.json'
POLICY=V/'ROOT_WEAK_DIRECT_END_ADMISSION_POLICY_BEFORE_SOURCE_CACHE.json'
OUT_ADMISSION=H/'ROOT_ADMITTED_EXPANDED_DIRECT_SOURCE.json'
OUT_ENTRY=H/'ROOT_ACTUAL_EXPANDED_DIRECT_CONTROL_CACHE_ENTRY.json'

def main():
    scope=c.read(SCOPE);policy=c.read(POLICY)
    c.need(scope['approved_direct_control_cache'] is True and scope['source_only'] is True and policy['approved_weak_direct_end_target_admission_assembly'] is True,'Root actual source assembly scope and policy')
    c.need(c.bind(POLICY)==scope['admission_policy'],'Root policy exact binding')
    c.need(sys.executable=='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python' and c.np.__version__=='1.26.4','Exact pinned source cache Python/NumPy')
    ready=c.read(runner.H/'FINAL_READY.json')
    for key in ('code','common','protocol','checks'):c.need(ready[key]==scope[key],'Frozen root-reviewed '+key);c.checked(scope[key])
    protocol=c.read_bound(scope['protocol'])
    c.need(protocol['output_directory']==scope['output_directory'] and protocol['checkpoint']==scope['checkpoint'],'Exact new output and inherited checkpoint metadata pointer')
    old=c.read_bound(protocol['old_cache_complete']);old_admission=c.read_bound(old['admission']);old_SAM_groups=c.lines(old['SAM_groups'])
    c.need(old['completed'] is True and old['PASS'] is True and old_admission['completed'] is True and old_admission['PASS'] is True,'Completed old cache/root admission metadata')
    old_sources=old_admission['sources'];syn_source=copy.deepcopy(next(s for s in old_sources if s['domain']=='SYNTH_FINITE_END'))
    c.need(syn_source['complete']==protocol['source_controls']['SYNTH_FINITE_END'],'Unchanged synthetic source authority')
    synthetic_groups=copy.deepcopy([g for g in old_admission['groups'] if g['domain']=='SYNTH_FINITE_END'])
    c.need(len(synthetic_groups)==291 and all(g['admitted'] is True for g in synthetic_groups),'Unchanged all97 original synthetic complete triplets')
    new_binding=protocol['source_controls']['REAL_SEMANTIC_END_WEAK']
    c.need(new_binding==policy['actual_new_source_evidence'],'Replacement V193 real source only')
    new=c.read_bound(new_binding);review=c.read_bound(policy['independent_new_source_review'])
    c.need(new['completed'] is True and new['PASS'] is True and review['completed'] is True and review['PASS'] is True,'Actual replacement source and independent review closed')
    real=c.lines(new['manifest']);c.need(len(real)==66 and len({r['pair_id'] for r in real})==66,'All66 replacement attempts')
    frame_plan=c.read_bound(old_admission['original_real_frame_plan'])
    frames={f['frame_index']:f for f in frame_plan['frames']};attempts={a['pair_id']:a for a in frame_plan['attempts']}
    c.need(len(frames)==21 and len(attempts)==66 and set(attempts)=={r['pair_id'] for r in real} and all(f['split']=='SOURCE_DEVELOPMENT' for f in frames.values()),'Exact original21 source frames/all66 attempts')
    real_groups=[];supported=[]
    for r in real:
        aid=r['pair_id'];origin=attempts[aid];frame=frames[origin['frame_index']];selected=r['original_input']['native_slot']
        c.need(origin['case_id']==r['case_id'] and origin['donor_id']==r['donor_id'] and origin['native_slot']==selected,'Unchanged case/donor/native origin')
        c.need(r['training_admission'] is False and r['new_native_quality_owner_all8']==-1,'Source output admission and old label flags unchanged')
        ok=runner.real_supported(r)
        if ok:supported.append(aid)
        for var in runner.VARIANTS:
            packet=r['variants'].get('ADJACENT_SHAM' if var=='SHAM' else var)
            if not isinstance(packet,dict) or 'path' not in packet:continue
            c.safe(packet['path']);c.need(len(packet['sha256'])==64,'Inherited physical packet binding, no binary stat/hash')
            gid='REAL__'+aid+'__'+var;mm=[]
            for k in range(8):
                mm.append(dict(row_id=gid+'/native%d'%k,original_occurrence_id=frame['query_id'][k],original_global_index=frame['global_index'][k],native_slot=k,query_xy=frame['query_xy'][k],L=frame['L'][k],original_query_available=frame['native_continuous_valid'][k],original_input_flags=frame['original_query_input_metadata'][k]))
            c.need(not ok or mm[selected]['original_query_available'] is True,'Source supported selected native must remain originally available')
            real_groups.append(dict(group_id=gid,domain='REAL_SEMANTIC_END_WEAK',authority=r['endpoint_observation']['authority'],attempt_id=aid,variant=var,source_packet=packet,layout='REAL_SINGLE',selected_native_slot=selected,admitted_target=(1 if var=='COVER' else 0) if ok else -1,admitted=ok,source_status=r['status'],raw_hw=frame['raw_hw'],metadata8=mm))
    real_source=dict(domain='REAL_SEMANTIC_END_WEAK',complete=new_binding,independent_review=policy['independent_new_source_review'],census=new['manifest'],expected_attempts=66)
    admission=dict(completed=True,PASS=True,source_only=True,training_admission_direct_end=True,sources=[syn_source,real_source],groups=synthetic_groups+real_groups,original_real_frame_plan=old_admission['original_real_frame_plan'],root_actual_assembly_scope=c.bind(SCOPE),admission_policy=c.bind(POLICY),source_generators_training_admission_flags_unchanged=True,real_source_replaces_old_whole=True,no_split_filtering=True,new_native_quality_owner_targets_created=False,old_cache_complete=protocol['old_cache_complete'],assembly_code=c.bind(__file__),no_source_binary_stat_hash_header_or_body=True,fully_solved=False)
    groups,census,_=runner.source_population(admission)
    c.need(len(groups)==policy['expected_physical_groups_rows'][0] and 8*len(groups)==policy['expected_physical_groups_rows'][1] and len(census)==3270,'Root exact full physical/status denominators')
    syn_supported=len({g['attempt_id'] for g in groups if g['domain']=='SYNTH_FINITE_END' and g['admitted']});c.need(syn_supported==policy['expected_supported_triplets']['SYNTH_FINITE_END'] and len(supported)==policy['expected_supported_triplets']['REAL_SEMANTIC_END_WEAK'],'All and only root policy supported triplets')
    targets=[g['admitted_target'] if k==g['selected_native_slot'] else -1 for g in groups for k in range(8)]
    known=[targets.count(0),targets.count(1)];inputU=sum(not m['original_query_available'] for g in groups for m in g['metadata8'])
    c.need(known==policy['expected_direct_known_D_E'] and inputU==112,'Exact root-supported targets and factual inputU')
    c.need([g['metadata8'] for g in groups[:291]]==[g['metadata8'] for g in old_admission['groups'] if g['domain']=='SYNTH_FINITE_END'],'Original291 synthetic all8 metadata unchanged')
    raw_bindings=[g['source_packet'] for g in groups]+[g['outputs'] for g in old_SAM_groups if g.get('outputs')]
    bypath={}
    for b in raw_bindings:
        c.safe(b['path']);c.need(len(b['sha256'])==64,'Inherited finite asset hash pointer')
        if b['path'] in bypath:c.need(bypath[b['path']]==b['sha256'],'One inherited hash per exact path')
        bypath[b['path']]=b['sha256']
    whitelist=[dict(path=path,sha256=sh) for path,sh in sorted(bypath.items())]
    admission['counts']=dict(physical_groups=len(groups),rows=8*len(groups),status_census_rows=len(census),supported_triplets=dict(SYNTH_FINITE_END=syn_supported,REAL_SEMANTIC_END_WEAK=len(supported)),known_D_E=known,unknown_direct_targets=targets.count(-1),original_inputU=inputU)
    c.write(OUT_ADMISSION,admission)
    entry=dict(approved_direct_control_cache=True,source_only=True,code=scope['code'],common=scope['common'],protocol=scope['protocol'],checks=scope['checks'],root_scope=c.bind(SCOPE),admission_policy=c.bind(POLICY),admission=c.bind(OUT_ADMISSION),checkpoint=scope['checkpoint'],array_whitelist=whitelist,output_directory=scope['output_directory'],assembly_code=c.bind(__file__),actual_asset_header_or_body_reads_before_entry=0,automatic_retry=False)
    c.write(OUT_ENTRY,entry)
    c.write(H/'ASSEMBLY_COMPLETE.json',dict(completed=True,PASS=True,scope=c.bind(SCOPE),policy=c.bind(POLICY),code=c.bind(__file__),admission=c.bind(OUT_ADMISSION),entry=c.bind(OUT_ENTRY),counts=admission['counts'],source_packet_unique_count=len({g['source_packet']['path'] for g in groups}),old_SAM_packet_unique_count=len({g['outputs']['path'] for g in old_SAM_groups if g.get('outputs')}),finite_array_whitelist_count=len(whitelist),old_cache_untouched=True,no_source_binary_stat_hash_header_body_or_model_calls=True,fully_solved=False))
    print('ASSEMBLY_COMPLETE',admission['counts'],'whitelist',len(whitelist),flush=True)

if __name__=='__main__':main()
