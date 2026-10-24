"""Invented temporary 2-source+1excluded publication, real copier and builder.
No actual V188 results/metadata inventory or repository/Git access. Only allowed
publication source code is read. All fixture data is created here then deleted.
"""
from pathlib import Path
import hashlib, importlib.util, json, tempfile

H=Path(__file__).absolute().parent
spec=importlib.util.spec_from_file_location('invented_v188_delta_builder',H/'build_completed_phase_delta.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def b(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def w(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False)+'\n')
def run():
    with tempfile.TemporaryDirectory(prefix='invented_delta_',dir=H) as tmp:
        t=Path(tmp);source=t/'source';phase=source/m.V.name;phase.mkdir(parents=True);repo=t/'repo';root=repo/m.REPO_PREFIX;root.mkdir(parents=True)
        old_index=b'old index bytes\n';(root/'EXPERIMENT_INDEX_KO.md').write_bytes(old_index)
        old_readme=b'old global README bytes\n';(repo/'readme.md').write_bytes(old_readme)
        base={'entries':[{'status':'PUBLISHED_RAW','published_relative':'EXPERIMENT_INDEX_KO.md','published_sha256':hashlib.sha256(old_index).hexdigest(),'source_sha256':hashlib.sha256(old_index).hexdigest(),'size_bytes':len(old_index),'gzipchunks':None}]};w(root/'PUBLICATION_MANIFEST.json',base);base_binding=b(root/'PUBLICATION_MANIFEST.json')
        prior_index=b'V187 immutable index\n'+old_index;(root/m.PARENT_INDEX).write_bytes(prior_index)
        prior_delta={'format':'immutable-completed-phase-delta-v1','phase':'query_visibility_source_selected_part_correspondence_20261024_v187','previous_base_manifest':base_binding,'entries':[{'status':'PUBLISHED_RAW','published_relative':m.PARENT_INDEX,'published_sha256':b(root/m.PARENT_INDEX)['sha256'],'source_sha256':b(root/m.PARENT_INDEX)['sha256'],'size_bytes':len(prior_index),'gzipchunks':None}]}
        w(root/m.PARENT_DELTA,prior_delta);prior_delta_binding=b(root/m.PARENT_DELTA)
        prior_chain={'format':'immutable-publication-manifest-chain-v1','base':{'path':'PUBLICATION_MANIFEST.json','sha256':base_binding['sha256']},'deltas':[{'path':m.PARENT_DELTA,'sha256':prior_delta_binding['sha256']}],'index':{'path':m.PARENT_INDEX,'sha256':b(root/m.PARENT_INDEX)['sha256']}}
        w(root/m.PARENT_CHAIN,prior_chain);prior_chain_binding=b(root/m.PARENT_CHAIN)
        (phase/'PORTABLE_REPORT_KO.md').write_text('# Invented only\n\n[JSON](RESULT.json.parts.json)\n')
        (phase/'RESULT.json').write_text(json.dumps({'fixture':'x'*(1024**2+17)}))
        (phase/'RAW_INPUTS.npz').write_bytes(b'Excluded raw payload, never decoded')
        rows=[]
        for name in ['PORTABLE_REPORT_KO.md','RESULT.json','RAW_INPUTS.npz']:
            p=phase/name;s=p.stat();exclude=name.endswith('.npz')
            rows.append({'source_root_id':'P','source_absolute':str(p),'source_relative_path':phase.name+'/'+name,'source_sha256':None if exclude else b(p)['sha256'],'size_bytes':s.st_size,'published_relative':m.ARCHIVE_PHASE+name,'published_sha256':None,'status':'CANDIDATE_EXCLUDE' if exclude else 'CANDIDATE_INCLUDE','action':'EXCLUDE' if exclude else 'INCLUDE','category':'invented_fixture','exclusion_reason':'raw array' if exclude else None,'suffix':p.suffix,'mtime_ns':s.st_mtime_ns,'device':s.st_dev,'inode':s.st_ino,'gzipchunks':None})
        admin=source/'publication_root_preparation';admin.mkdir();candidate=admin/'candidate.jsonl';candidate.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        scope=admin/'scope.json';w(scope,{'invented_only':True})
        review=admin/'phase_done.json';w(review,{'completed':True,'PASS':True})
        inv=admin/'inventory.json';w(inv,{'completed':True,'PASS':True,'source_phase':str(phase),'candidate_manifest':b(candidate)})
        proto=admin/'protocol.json';common={'code':b(H/'build_completed_phase_delta.py'),'pure_helper':b(H/'prepare_completed_phase_delta.py')};w(proto,common)
        entry=admin/'build_entry.json';e={'approved_completed_phase_delta_build':True,'completed_phase_frozen':True,**common,'protocol':b(proto),'phase':phase.name,'root_scope':b(scope),'completion_reviews':[{'binding':b(review),'expected_fields':{'completed':True,'PASS':True}}],'repository':str(repo),'base_manifest':base_binding,'parent_delta_manifest':prior_delta_binding,'parent_chain':prior_chain_binding,'base_index':b(root/m.PARENT_INDEX),'global_readme_before':b(repo/'readme.md'),'inventory_complete':b(inv),'candidate_manifest':b(candidate),'copy_packager':b(m.PACKAGER),'copy_helper':b(m.PACK_HELPER),'expected_remote_main':'a'*40,'publisher_tool_bindings':[b(H/'build_completed_phase_delta.py')],'admin_output':str(admin/'build')};w(entry,e)
        # Patch only this disposable imported module's roots to invented sources.
        m.P=source;m.V=phase;m.BUILDER_PROTOCOL=proto
        m.build(entry)
        done=admin/'build/BUILD_COMPLETE.json';d=json.loads(done.read_text());delta=json.loads(Path(d['delta_manifest']['path']).read_text())
        checks={
            'old_manifest_SHA_unchanged':b(root/'PUBLICATION_MANIFEST.json')==base_binding,
            'old_index_bytes_unchanged':(root/'EXPERIMENT_INDEX_KO.md').read_bytes()==old_index,
            'new_index_V187_exact_suffix':(root/m.NEW_INDEX).read_bytes().endswith(prior_index),
            'parent_V187_delta_SHA_unchanged':b(root/m.PARENT_DELTA)==prior_delta_binding,
            'parent_V187_chain_SHA_unchanged':b(root/m.PARENT_CHAIN)==prior_chain_binding,
            'parent_V187_index_bytes_unchanged':(root/m.PARENT_INDEX).read_bytes()==prior_index,
            'newchain_preserves_V187_first_delta':json.loads((root/m.NEW_CHAIN).read_text())['deltas'][0]==prior_chain['deltas'][0] and len(json.loads((root/m.NEW_CHAIN).read_text())['deltas'])==2,
            'global_readme_old_suffix':(repo/'readme.md').read_bytes().endswith(old_readme),
            'one_excluded_source_retained':sum(r['status']=='EXCLUDED' for r in delta['entries'])==1,
            'gzip_physical_descriptor_retained':any(r.get('published_relative','').endswith('.parts.json') and r['status']=='PUBLISHED_RAW' for r in delta['entries'] if r.get('published_relative')),
            'source_gzip_roundtrip_declared':sum(r['status']=='PUBLISHED_GZIP_CHUNKS' for r in delta['entries'])==1,
        }
        qa=admin/'independent_fixture_QA.json';w(qa,{'completed':True,'PASS':True,'manifest_stored_sha256':d['delta_manifest']['sha256']})
        pe=admin/'push_entry.json';w(pe,{'approved_verified_delta_push_plan':True,**common,'protocol':b(proto),'build_complete':b(done),'delta_manifest':d['delta_manifest'],'independent_delta_reviews':[{'binding':b(qa),'expected_fields':{'manifest_stored_sha256':d['delta_manifest']['sha256']}}],'output_plan':str(admin/'PUSH_PLAN.json')})
        m.finalize(pe);plan=json.loads((admin/'PUSH_PLAN.json').read_text());flat=[r for batch in plan['batches'] for r in batch['files']]
        protected={'PUBLICATION_MANIFEST.json','EXPERIMENT_INDEX_KO.md',m.PARENT_DELTA,m.PARENT_CHAIN,m.PARENT_INDEX}
        checks.update(push_plan_has_global_readme=sum(r['path']=='readme.md' for r in flat)==1,push_plan_unique_paths=len({r['path'] for r in flat})==len(flat),push_plan_all_bytes_bound=all(hashlib.sha256((repo/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in flat),push_plan_no_old_base_or_V187_paths=all(r['path'] not in {m.REPO_PREFIX+x for x in protected} for r in flat))
        return {'completed':True,'PASS':all(checks.values()),'checks':checks,'invented_temporary_sources_only':True,'actual_phase_artifact_enumeration_hash_copy':0,'actual_repository_or_Git_access':0}

if __name__=='__main__':
    result=run();out=H/'INVENTED_COPY_BUILD_PLAN_ROUNDTRIP.json'
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
