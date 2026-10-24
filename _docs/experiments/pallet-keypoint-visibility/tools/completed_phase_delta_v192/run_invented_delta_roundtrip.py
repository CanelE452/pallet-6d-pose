"""Invented BASE+V187+V188+V189+V190+V191 -> V192 roundtrip; no actual phase/repo inputs."""
from pathlib import Path
import copy,hashlib,importlib.util,json,tempfile
H=Path(__file__).absolute().parent
spec=importlib.util.spec_from_file_location('invented_v192_delta_builder',H/'build_completed_phase_delta.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def b(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def w(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False)+'\n')
def raw(relative,sha='b'*64):return dict(status='PUBLISHED_RAW',published_relative=relative,published_sha256=sha,source_sha256=sha,size_bytes=1,gzipchunks=None)

def pure_checks():
    checks={};bb={'path':'/invented/PUBLICATION_MANIFEST.json','sha256':'a'*64};base={'entries':[raw('archive/BASE.json')]};records=[];refs=[]
    for i,(tag,phase) in enumerate(m.pure.ANCESTOR_PHASES):
        dr='publication/deltas/'+tag+'_20261024_v1/DELTA_MANIFEST.json';cr='MANIFEST_CHAIN_AFTER_'+tag+'.json';ir='EXPERIMENT_INDEX_THROUGH_'+tag+'_KO.md'
        d={'path':'/invented/'+dr,'sha256':str(i+1)*64};c={'path':'/invented/'+cr,'sha256':str(i+3)*64};ix={'path':'/invented/'+ir,'sha256':str(i+5)*64}
        manifest=dict(format='immutable-completed-phase-delta-v1',phase=phase,previous_base_manifest=bb,entries=[raw(ir,ix['sha256']),dict(status='PUBLISHED_GZIP_CHUNKS',published_relative='archive/'+tag+'.jsonl',gzip_descriptor_path='archive/'+tag+'.jsonl.parts.json',gzipchunks={'parts':[{'path':'archive/'+tag+'.jsonl.parts/p.gz'}]})])
        if records:manifest.update(previous_delta_manifest=records[-1]['delta_manifest'],previous_chain=records[-1]['chain'])
        refs.append(dict(path=dr,sha256=d['sha256']));chain=dict(format='immutable-publication-manifest-chain-v1',base=dict(path='PUBLICATION_MANIFEST.json',sha256=bb['sha256']),deltas=copy.deepcopy(refs),index=dict(path=ir,sha256=ix['sha256']))
        if records:chain['previous_chain']=dict(path='MANIFEST_CHAIN_AFTER_'+records[-1]['tag']+'.json',sha256=records[-1]['chain']['sha256'])
        records.append(dict(tag=tag,delta_manifest=d,chain=c,index=ix,manifest=manifest,chain_data=chain))
    entries=m.pure.validate_parent_closure(bb,base,records);checks['full_BASE_through_V191_positive_closure']=bool(entries)
    negatives={
      'missing_earlier_V187':lambda r:r.pop(0),
      'missing_latest_V191':lambda r:r.pop(),
      'reversed_order':lambda r:r.reverse(),
      'ancestor_base_SHA_drift':lambda r:r[-1]['manifest']['previous_base_manifest'].update(sha256='f'*64),
      'earlier_delta_SHA_drift_in_latest_chain':lambda r:r[-1]['chain_data']['deltas'][0].update(sha256='f'*64),
      'previous_chain_SHA_drift':lambda r:r[-1]['chain_data']['previous_chain'].update(sha256='f'*64),
      'latest_index_SHA_drift':lambda r:r[-1]['chain_data']['index'].update(sha256='f'*64),
      'explicit_previous_index_drift':lambda r:r[-1]['manifest'].update(previous_index={'path':'/wrong','sha256':'f'*64})}
    for name,mutate in negatives.items():
        r=copy.deepcopy(records);mutate(r)
        try:m.pure.validate_parent_closure(bb,base,r);checks[name+'_rejected']=False
        except (AssertionError,KeyError):checks[name+'_rejected']=True
    for name in ['archive/BASE.json','archive/V187.jsonl','archive/V187.jsonl.parts/p.gz','archive/V188.jsonl.parts.json','MANIFEST_CHAIN_AFTER_V187.json','publication/deltas/V188_20261024_v1/DELTA_MANIFEST.json']:
        try:m.pure.assert_new_paths(entries,[raw(name)]);checks['old_collision_'+name]=False
        except AssertionError:checks['old_collision_'+name]=True
    repository='/invented/repo';closing=[]
    for tag in m.pure.CLOSING_TAGS:
        plan=dict(authorized_main_push=True,repository=repository,branch='main',remote='https://github.com/CanelE452/pallet-6d-pose.git',batches=[dict(files=[dict(path=m.REPO_PREFIX+'closing/'+tag+'.json',sha256='a'*64,size_bytes=3)])])
        closing.append(dict(tag=tag,plan_data=plan))
    cp=m.pure.closing_protected_entries(closing,repository);checks['all_six_closing_plans_protected']=len(cp)==len(m.pure.CLOSING_TAGS)
    with_readme=copy.deepcopy(closing)
    with_readme[0]['plan_data']['batches'][0]['files'].append(dict(path='readme.md',sha256='c'*64,size_bytes=19))
    exceptions=[];protected=m.pure.closing_protected_entries(with_readme,repository,exceptions)
    checks['BASE_historical_README_accepted_reported_notprotected']=protected==cp and len(exceptions)==1 and exceptions[0]['tag']=='BASE' and exceptions[0]['path']=='readme.md' and exceptions[0]['sha256']=='c'*64 and exceptions[0]['size_bytes']==19
    for name,record_index,path in [('nonBASE_README',1,'readme.md'),('BASE_otheroutside',0,'OTHER.md'),('nonBASE_otheroutside',2,'OTHER.md')]:
        negative=copy.deepcopy(closing);negative[record_index]['plan_data']['batches'][0]['files'].append(dict(path=path,sha256='c'*64,size_bytes=19))
        try:m.pure.closing_protected_entries(negative,repository);checks[name+'_rejected']=False
        except AssertionError:checks[name+'_rejected']=True
    for name,field,value in [('README_badSHA','sha256','x'*64),('README_badsize','size_bytes',-1)]:
        negative=copy.deepcopy(with_readme);negative[0]['plan_data']['batches'][0]['files'][-1][field]=value
        try:m.pure.closing_protected_entries(negative,repository);checks[name+'_rejected']=False
        except AssertionError:checks[name+'_rejected']=True
    for label,negative in [('missing_closing',closing[:-1]),('reversed_closing',list(reversed(closing)))]:
        try:m.pure.closing_protected_entries(negative,repository);checks[label+'_rejected']=False
        except AssertionError:checks[label+'_rejected']=True
    try:m.pure.assert_new_paths(entries+cp,[raw('closing/V189.json')]);checks['old_closing_overwrite_rejected']=False
    except AssertionError:checks['old_closing_overwrite_rejected']=True
    assert all(checks.values()),checks
    return checks

def run():
    checks=pure_checks()
    with tempfile.TemporaryDirectory(prefix='invented_delta_',dir=H) as tmp:
        t=Path(tmp);source=t/'source';phase=source/m.V.name;phase.mkdir(parents=True);repo=t/'repo';root=repo/m.REPO_PREFIX;root.mkdir(parents=True)
        old_index=b'base original index\n';(root/'EXPERIMENT_INDEX_KO.md').write_bytes(old_index);old_readme=b'original global readme\n';(repo/'readme.md').write_bytes(old_readme)
        base={'entries':[dict(raw('EXPERIMENT_INDEX_KO.md',hashlib.sha256(old_index).hexdigest()),size_bytes=len(old_index))]};w(root/'PUBLICATION_MANIFEST.json',base);base_binding=b(root/'PUBLICATION_MANIFEST.json')
        ancestors=[];refs=[];index_bytes=old_index
        for tag,phase_name in m.pure.ANCESTOR_PHASES:
            dr='publication/deltas/'+tag+'_20261024_v1/DELTA_MANIFEST.json';cr='MANIFEST_CHAIN_AFTER_'+tag+'.json';ir='EXPERIMENT_INDEX_THROUGH_'+tag+'_KO.md'
            index_bytes=(tag+' previous index\n').encode()+index_bytes;(root/ir).write_bytes(index_bytes)
            delta=dict(format='immutable-completed-phase-delta-v1',phase=phase_name,previous_base_manifest=base_binding,entries=[dict(raw(ir,b(root/ir)['sha256']),size_bytes=len(index_bytes))])
            if ancestors:delta.update(previous_delta_manifest=ancestors[-1]['delta_manifest'],previous_chain=ancestors[-1]['chain'])
            w(root/dr,delta);db=b(root/dr);refs.append(dict(path=dr,sha256=db['sha256']))
            chain=dict(format='immutable-publication-manifest-chain-v1',base=dict(path='PUBLICATION_MANIFEST.json',sha256=base_binding['sha256']),deltas=copy.deepcopy(refs),index=dict(path=ir,sha256=b(root/ir)['sha256']))
            if ancestors:chain['previous_chain']=dict(path='MANIFEST_CHAIN_AFTER_'+ancestors[-1]['tag']+'.json',sha256=ancestors[-1]['chain']['sha256'])
            w(root/cr,chain);ancestors.append(dict(tag=tag,delta_manifest=db,chain=b(root/cr),index=b(root/ir)))
        parent=ancestors[-1];prior_index=index_bytes;sealed_old=[base_binding]+[item[key] for item in ancestors for key in ('delta_manifest','chain','index')]
        closing=[]
        for tag in m.pure.CLOSING_TAGS:
            rel='publication/prior_closing/'+tag+'.json';w(root/rel,{'invented_closing':tag})
            prior_files=[dict(path=m.REPO_PREFIX+rel,sha256=b(root/rel)['sha256'],size_bytes=(root/rel).stat().st_size)]
            if tag=='BASE':prior_files.append(dict(path='readme.md',sha256=hashlib.sha256(old_readme).hexdigest(),size_bytes=len(old_readme)))
            plan=source/('closing_'+tag+'.json');w(plan,dict(authorized_main_push=True,repository=str(repo),branch='main',remote='https://github.com/CanelE452/pallet-6d-pose.git',batches=[dict(files=prior_files)]))
            closing.append(dict(tag=tag,binding=b(plan)));sealed_old.extend([b(plan),b(root/rel)])
        (phase/'PORTABLE_REPORT_KO.md').write_text('# Invented only\n\n[JSON](RESULT.json.parts.json)\n');(phase/'RESULT.json').write_text(json.dumps({'fixture':'x'*(1024**2+17)}));(phase/'RAW_INPUTS.npz').write_bytes(b'Excluded raw payload never decoded')
        rows=[]
        for name in ['PORTABLE_REPORT_KO.md','RESULT.json','RAW_INPUTS.npz']:
            p=phase/name;s=p.stat();exclude=name.endswith('.npz');rows.append(dict(source_root_id='P',source_absolute=str(p),source_relative_path=phase.name+'/'+name,source_sha256=None if exclude else b(p)['sha256'],size_bytes=s.st_size,published_relative=m.ARCHIVE_PHASE+name,published_sha256=None,status='CANDIDATE_EXCLUDE' if exclude else 'CANDIDATE_INCLUDE',action='EXCLUDE' if exclude else 'INCLUDE',category='invented_fixture',exclusion_reason='raw array' if exclude else None,suffix=p.suffix,mtime_ns=s.st_mtime_ns,device=s.st_dev,inode=s.st_ino,gzipchunks=None))
        admin=source/'publication_root_preparation';admin.mkdir();candidate=admin/'candidate.jsonl';candidate.write_text(''.join(json.dumps(r)+'\n' for r in rows));scope=admin/'scope.json';w(scope,{'invented_only':True});review=admin/'ROOT_PHASE_RESULT.json';w(review,{'completed':True,'PASS':True});qa_report=admin/'REPORT_QA.json';w(qa_report,{'completed':True,'PASS':True});freeze=admin/'ROOT_FINAL_PHASE_FREEZE.json';w(freeze,{'frozen':True});inv=admin/'inventory.json';w(inv,dict(completed=True,PASS=True,source_phase=str(phase),candidate_manifest=b(candidate)))
        proto=admin/'protocol.json';common=dict(code=b(H/'build_completed_phase_delta.py'),pure_helper=b(H/'prepare_completed_phase_delta.py'));w(proto,common)
        entry=admin/'build_entry.json';e=dict(approved_completed_phase_delta_build=True,completed_phase_frozen=True,**common,protocol=b(proto),phase=phase.name,root_scope=b(scope),completion_reviews=[dict(role='phase_result',binding=b(review),expected_fields={'completed':True}),dict(role='independent_report_QA',binding=b(qa_report),expected_fields={'completed':True,'PASS':True}),dict(role='final_phase_freeze',binding=b(freeze),expected_fields={'frozen':True})],repository=str(repo),base_manifest=base_binding,parent_delta_manifest=parent['delta_manifest'],parent_chain=parent['chain'],base_index=parent['index'],ancestor_publications=ancestors,immutable_closing_plans=closing,global_readme_before=b(repo/'readme.md'),inventory_complete=b(inv),candidate_manifest=b(candidate),copy_packager=b(m.PACKAGER),copy_helper=b(m.PACK_HELPER),expected_remote_main='a'*40,publisher_tool_bindings=[b(H/'build_completed_phase_delta.py')],admin_output=str(admin/'build'));w(entry,e)
        m.P=source;m.V=phase;m.BUILDER_PROTOCOL=proto;m.build(entry)
        done=admin/'build/BUILD_COMPLETE.json';d=json.loads(done.read_text());delta=json.loads(Path(d['delta_manifest']['path']).read_text());chain=json.loads((root/m.NEW_CHAIN).read_text())
        checks.update(all_parent_manifest_chain_index_and_closing_SHA_immutable=all(b(Path(x['path']))==x for x in sealed_old),original_index_bytes_unchanged=(root/'EXPERIMENT_INDEX_KO.md').read_bytes()==old_index,newindex_exact_V191_suffix=(root/m.NEW_INDEX).read_bytes().endswith(prior_index),newchain_full_V187_through_V192_order=chain['deltas'][:5]==refs and len(chain['deltas'])==6,newmanifest_explicit_previous_index=delta['previous_index']==parent['index'],global_readme_exact_suffix=(repo/'readme.md').read_bytes().endswith(old_readme),rawarray_excluded_provenance=sum(r['status']=='EXCLUDED' for r in delta['entries'])==1,gzip_descriptor_record_retained=any((r.get('published_relative') or '').endswith('.parts.json') and r['status']=='PUBLISHED_RAW' for r in delta['entries']),gzip_logical_roundtrip_one=sum(r['status']=='PUBLISHED_GZIP_CHUNKS' for r in delta['entries'])==1)
        ex=d['historical_global_readme_exceptions'];checks['README_exception_metadata_survives_build_delta_chain']=len(ex)==1 and ex==delta['historical_global_readme_exceptions']==chain['historical_global_readme_exceptions'] and ex[0]['sha256']==hashlib.sha256(old_readme).hexdigest()
        qa=admin/'QA.json';w(qa,dict(completed=True,PASS=True,manifest_stored_sha256=d['delta_manifest']['sha256']));pe=admin/'push_entry.json';w(pe,dict(approved_verified_delta_push_plan=True,**common,protocol=b(proto),build_complete=b(done),delta_manifest=d['delta_manifest'],independent_delta_reviews=[dict(binding=b(qa),expected_fields={'manifest_stored_sha256':d['delta_manifest']['sha256']})],output_plan=str(admin/'PUSH_PLAN.json')))
        m.finalize(pe);plan=json.loads((admin/'PUSH_PLAN.json').read_text());flat=[r for batch in plan['batches'] for r in batch['files']]
        protected={'PUBLICATION_MANIFEST.json','EXPERIMENT_INDEX_KO.md'}|{str(Path(x['path']).relative_to(root)) for x in sealed_old if root in Path(x['path']).parents}
        checks.update(push_plan_has_only_newphysical_plus_global_readme=all(r['path']=='readme.md' or r['path'] not in {m.REPO_PREFIX+x for x in protected} for r in flat),push_plan_unique_paths=len({r['path'] for r in flat})==len(flat),push_plan_all_bytes_bound=all(hashlib.sha256((repo/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in flat),push_initial_remote_X_unchanged=plan['initial_remote_main']=='a'*40)
    assert all(checks.values()),checks
    return dict(completed=True,PASS=True,checks=checks,count=len(checks),invented_temporary_sources_only=True,actual_phase_artifact_enumeration_hash_copy=0,actual_repository_or_Git_access=0)
if __name__=='__main__':
    result=run()
    with (H/'INVENTED_COPY_BUILD_PLAN_ROUNDTRIP.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
