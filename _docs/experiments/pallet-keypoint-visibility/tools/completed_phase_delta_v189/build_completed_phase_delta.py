"""Root-entry-only V189 append publisher. No Git commands are executed here.
Build: phase-only sealed inventory -> existing copier -> immutable delta/index.
Finalize: independent reviews -> exact physical delta-only push plan.
Preparation uses invented metadata/bytes only; no phase enumeration or hashes.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, stat, sys
import prepare_completed_phase_delta as pure

H=Path(__file__).absolute().parent;V=pure.PHASE;P=pure.P
REPO_PREFIX=pure.PREFIX
DELTA='publication/deltas/V189_20261024_v1'
NEW_INDEX='EXPERIMENT_INDEX_THROUGH_V189_KO.md'
NEW_CHAIN='MANIFEST_CHAIN_AFTER_V189.json'
PARENT_DELTA='publication/deltas/V188_20261024_v1/DELTA_MANIFEST.json'
PARENT_CHAIN='MANIFEST_CHAIN_AFTER_V188.json'
PARENT_INDEX='EXPERIMENT_INDEX_THROUGH_V188_KO.md'
ARCHIVE_PHASE='archive/task-2/pallet-keypoint-visibility/'+V.name+'/'
PACKAGER=P/'publication_preparation_c/package_publication_snapshot.py'
PACK_HELPER=P/'publication_preparation_c/refine_inventory_and_hash.py'
BUILDER_PROTOCOL=H/'BUILDER_PROTOCOL_BEFORE_ANY_DELTA_ARTIFACT_READ.json'

def require(v,msg):
    if not bool(v):raise AssertionError(msg)
def opened(path):
    p=pure.safe_parts(path);require(p.is_absolute(),'Explicit absolute path')
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        final=os.open(p.parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    require(stat.S_ISREG(os.fstat(final).st_mode),'Regular file only');return os.fdopen(final,'rb')
def read_bytes(path):
    with opened(path) as f:return f.read()
def sha(path):
    h=hashlib.sha256()
    with opened(path) as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def bind(path):return {'path':str(path),'sha256':sha(path)}
def checked(binding):
    require(len(binding['sha256'])==64 and sha(binding['path'])==binding['sha256'],'SHA drift '+binding['path']);return Path(binding['path'])
def checked_json(binding):return json.loads(read_bytes(checked(binding)))
def write_new(path,data):
    path=pure.safe_parts(path);current=Path(path.anchor)
    for part in path.parts[1:-1]:
        current=current/part
        try:s=current.lstat()
        except FileNotFoundError:current.mkdir();s=current.lstat()
        require(stat.S_ISDIR(s.st_mode),'No destination symlink')
    with path.open('xb') as f:f.write(data)
def json_bytes(obj):return (json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode()
def raw_record(root,relative,category='publication_delta_derived'):
    path=root/relative;data=read_bytes(path)
    return {'source_root_id':'DELTA_DERIVED','source_absolute':str(path),'source_relative_path':relative,'source_sha256':pure.sha_bytes(data),'size_bytes':len(data),'published_relative':relative,'published_sha256':pure.sha_bytes(data),'status':'PUBLISHED_RAW','category':category,'exclusion_reason':None,'gzipchunks':None}
def check_reviews(requirements,completed_PASS=False):
    require(requirements,'Explicit completion/verification receipts required')
    for item in requirements:
        report=checked_json(item['binding'])
        if completed_PASS:require(report['completed'] is True and report['PASS'] is True,'Independent delta review PASS required')
        for key,value in item['expected_fields'].items():require(report[key]==value,'Review field '+key)

def parent_publication(e,root):
    """Read only explicitly bound parent metadata, never old archive bodies."""
    require(e['base_manifest']['path']==str(root/'PUBLICATION_MANIFEST.json'),'Initial base exact path')
    requested=e['ancestor_publications'];require(len(requested)==2,'Both V187 and V188 ancestor metadata mandatory')
    records=[]
    for record,(tag,phase) in zip(requested,pure.ANCESTOR_PHASES):
        require(record['tag']==tag,'Ordered ancestor entry tags')
        dr='publication/deltas/'+tag+'_20261024_v1/DELTA_MANIFEST.json';cr='MANIFEST_CHAIN_AFTER_'+tag+'.json';ir='EXPERIMENT_INDEX_THROUGH_'+tag+'_KO.md'
        for name,relative in [('delta_manifest',dr),('chain',cr),('index',ir)]:
            require(record[name]['path']==str(root/relative),'Exact bound ancestor path '+name)
        records.append(dict(record,manifest=checked_json(record['delta_manifest']),chain_data=checked_json(record['chain'])))
        checked(record['index'])
    latest=requested[-1]
    require(e['parent_delta_manifest']==latest['delta_manifest'] and e['parent_chain']==latest['chain'] and e['base_index']==latest['index'],'Latest parent aliases exact V188 bindings')
    base=checked_json(e['base_manifest']);entries=pure.validate_parent_closure(e['base_manifest'],base,records)
    return base,records[-1]['manifest'],records[-1]['chain_data'],entries


def recheck_parent_bindings(e):
    checked(e['base_manifest'])
    for ancestor in e['ancestor_publications']:
        for name in ('delta_manifest','chain','index'):checked(ancestor[name])


def build(entry_path):
    e=json.loads(read_bytes(entry_path));require(e['approved_completed_phase_delta_build'] is True,'Root build entry')
    require(e['code']==bind(__file__) and e['pure_helper']==bind(H/'prepare_completed_phase_delta.py') and e['protocol']['path']==str(BUILDER_PROTOCOL),'Exact frozen builder/helper/protocol')
    p=checked_json(e['protocol']);require(p['code']==e['code'] and p['pure_helper']==e['pure_helper'],'Protocol code/helper closure')
    require(e['completed_phase_frozen'] is True and e['phase']==V.name,'Only frozen V189')
    checked_json(e['root_scope']);check_reviews(e['completion_reviews'])
    require(e['copy_packager']['path']==str(PACKAGER) and e['copy_helper']['path']==str(PACK_HELPER),'Existing copier exact paths')
    checked(e['copy_packager']);checked(e['copy_helper'])
    repo=pure.safe_parts(e['repository']);root=repo/REPO_PREFIX
    require(repo.is_absolute(),'Absolute repository')
    base,previous,previous_chain,old_entries=parent_publication(e,root);base_hash=e['base_manifest']['sha256']
    old_index=read_bytes(checked(e['base_index']))
    require(e['global_readme_before']['path']==str(repo/'readme.md'),'Only global readme is mutable')
    old_readme=read_bytes(checked(e['global_readme_before']))
    inv=checked_json(e['inventory_complete']);require(inv['completed'] is True and inv['PASS'] is True,'Completed phase-only inventory')
    require(inv['source_phase']==str(V) and inv['candidate_manifest']==e['candidate_manifest'],'Exact inventory closure')
    candidate=checked(e['candidate_manifest']);rows=[json.loads(s) for s in read_bytes(candidate).splitlines()]
    require(rows and all(r['source_root_id']=='P' and r['source_relative_path'].startswith(V.name+'/') and r['source_absolute']==str(P/r['source_relative_path']) and r['published_relative'].startswith(ARCHIVE_PHASE) for r in rows),'No outside-phase copy')
    selected=[r for r in rows if r['action']!='EXCLUDE']
    require(all(r['action'] in {'INCLUDE','NUMERIC_REVIEW','SAVED_OUTPUT_REVIEW','SMALL_OWN_HEAD_REVIEW'} for r in selected),'Known selection categories')
    pure.assert_new_paths(old_entries,[dict(r,status='EXCLUDED' if r['action']=='EXCLUDE' else 'PUBLISHED_RAW') for r in rows])
    portable=ARCHIVE_PHASE+'PORTABLE_REPORT_KO.md'
    require(portable in {r['published_relative'] for r in selected},'Root final relative portable report included before inventory')
    admin=pure.safe_parts(e['admin_output']);require(admin.is_absolute() and P/'publication_root_preparation' in admin.parents and V not in admin.parents,'Fresh separate admin directory')
    admin.mkdir(parents=True,exist_ok=False)
    copy_plan={'format':'pallet-publication-plan-v1','approved_source_copy':True,'git_mutations_or_push':False,'target_root':str(root),'source_roots':{'P':str(P)},'archive_layout':'archive/task-2/<original root basename>/<relative path>','candidate_manifest':e['candidate_manifest'],'gzip_text_threshold_bytes':1024**2,'explicit_numerical_and_own_state_whitelist':['P:'+r['source_relative_path'] for r in selected if r['action']!='INCLUDE'],'allow_existing_identical':False,'output_manifest_relative':DELTA+'/SOURCE_ARCHIVE_MANIFEST.jsonl','output_complete_relative':DELTA+'/COPY_COMPLETE.json'}
    cp=admin/'ROOT_APPROVED_COPY_PLAN.json';write_new(cp,json_bytes(copy_plan))
    # Import only the already-reviewed publication copier, never experiment/model code.
    sys.path.insert(0,str(PACKAGER.parent))
    spec=importlib.util.spec_from_file_location('frozen_v189_publication_copier',PACKAGER);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.run(cp)
    entries=[json.loads(s) for s in read_bytes(root/copy_plan['output_manifest_relative']).splitlines()]
    md=read_bytes(root/portable);require(b'/home/' not in md and b'file://' not in md,'Portable report local absolute link forbidden')
    original_logical=len([r for r in entries if r['status']!='EXCLUDED'])
    # Existing copier creates descriptor files. Add those physical files as raw entries.
    extras={copy_plan['output_manifest_relative'],copy_plan['output_complete_relative']}
    extras.update(r['gzip_descriptor_path'] for r in entries if r.get('gzip_descriptor_path'))
    tools=e.get('publisher_tool_bindings',[]);require(bool(tools),'Versioned publication helper code/protocol/checks must be preserved')
    for tool in tools:
        source=checked(tool);require(source.parent==H and source.suffix in {'.py','.json','.md'},'Only explicit frozen publisher preparation files')
        relative='tools/completed_phase_delta_v189/'+source.name
        write_new(root/relative,read_bytes(source))
        row=raw_record(root,relative,'versioned_publication_tool')
        row.update(source_root_id='PUBLICATION_ADMIN_CODE',source_absolute=str(source),source_relative_path='publication_root_preparation/V189_DELTA_PREPARATION_B/'+source.name)
        entries.append(row)
    links=('[이번 delta manifest]('+DELTA+'/DELTA_MANIFEST.json) · [manifest chain]('+NEW_CHAIN+')\n\n').encode()
    index=links+pure.versioned_index_bytes(old_index,'V189 sampled-domain completed source records',portable)
    require(index.endswith(old_index),'Old index exact bytes suffix retained')
    write_new(root/NEW_INDEX,index);extras.add(NEW_INDEX)
    # Preserve prior global README before the permitted new global entry is written.
    prior_readme=DELTA+'/GLOBAL_README_BEFORE_V189.md';write_new(root/prior_readme,old_readme);extras.add(prior_readme)
    for source,name in [(entry_path,'ROOT_ACTUAL_DELTA_BUILD_ENTRY.json'),(cp,'ROOT_APPROVED_COPY_PLAN.json')]:
        rel=DELTA+'/'+name;write_new(root/rel,read_bytes(source));extras.add(rel)
    entries.extend(raw_record(root,p) for p in sorted(extras))
    pure.assert_new_paths(old_entries,entries)
    manifest_relative=DELTA+'/DELTA_MANIFEST.json'
    manifest={'format':'immutable-completed-phase-delta-v1','phase':V.name,'previous_base_manifest':e['base_manifest'],'previous_delta_manifest':e['parent_delta_manifest'],'previous_chain':e['parent_chain'],'previous_index':e['base_index'],'ancestor_publications':e['ancestor_publications'],'entries':entries,'source_phase_original_records':original_logical,'scope':'Publication of root-frozen V189 completed records and preserved UNKNOWN/failures; no learning, transfer, adoption or fullgoal success is inferred by publisher.','model_adopted':False,'full_goal_solved':False}
    write_new(root/manifest_relative,json_bytes(manifest));delta_binding=bind(root/manifest_relative)
    chain={'format':'immutable-publication-manifest-chain-v1','base':{'path':'PUBLICATION_MANIFEST.json','sha256':base_hash},'deltas':previous_chain['deltas']+[{'path':manifest_relative,'sha256':delta_binding['sha256']}],'previous_chain':{'path':PARENT_CHAIN,'sha256':e['parent_chain']['sha256']},'index':{'path':NEW_INDEX,'sha256':sha(root/NEW_INDEX)},'restore':'Use existing tools/restore_original_artifacts.py with the relevant base or delta manifest; original logical paths/gzip descriptors retained.','chain_receipt_note':'Chain itself and later independent QA are Git-plan-bound outside the manifest they verify.'}
    write_new(root/NEW_CHAIN,json_bytes(chain))
    new_readme=('# V189 sampled-domain source records update\n\n[최신 실험 인덱스]('+REPO_PREFIX+NEW_INDEX+') · [V189 원문·그림·표·실패기록]('+REPO_PREFIX+portable+')\n\n이 추가 기록은 source 진단이며 모델 채택이나 전체 가림 목표 해결을 뜻하지 않습니다.\n\n---\n\n').encode()+old_readme
    require(new_readme.endswith(old_readme),'Global README suffix byteexact')
    require(sha(repo/'readme.md')==e['global_readme_before']['sha256'],'Global README drift before update')
    tmp=repo/'readme.V189_publication_pending.tmp';write_new(tmp,new_readme);os.replace(tmp,repo/'readme.md')
    recheck_parent_bindings(e)
    physical={}
    for r in entries:
        if r['status']=='EXCLUDED':continue
        if r.get('gzipchunks') is None:physical[r['published_relative']]={'path':REPO_PREFIX+r['published_relative'],'sha256':r['published_sha256'],'size_bytes':r['size_bytes']}
        else:
            for part in r['gzipchunks']['parts']:physical[part['path']]={'path':REPO_PREFIX+part['path'],'sha256':part['sha256'],'size_bytes':part['stored_size_bytes']}
    for rel in (manifest_relative,NEW_CHAIN):physical[rel]={'path':REPO_PREFIX+rel,'sha256':sha(root/rel),'size_bytes':len(read_bytes(root/rel))}
    physical['readme.md']={'path':'readme.md','sha256':pure.sha_bytes(new_readme),'size_bytes':len(new_readme)}
    complete={'completed':True,'PASS':True,'staging_only_not_pushed_or_independently_verified':True,'root_entry':bind(entry_path),'repository':str(repo),'expected_remote_main':e['expected_remote_main'],'base_manifest':e['base_manifest'],'parent_delta_manifest':e['parent_delta_manifest'],'parent_chain':e['parent_chain'],'base_index':e['base_index'],'previous_index':e['base_index'],'ancestor_publications':e['ancestor_publications'],'publisher_tools':tools,'delta_manifest':delta_binding,'new_chain':bind(root/NEW_CHAIN),'new_index':bind(root/NEW_INDEX),'physical_files':list(physical.values()),'old_manifest_and_all_manifest_covered_paths_modified':False,'only_existing_mutation':'Global repository readme.md, previous exact bytes preserved separately and as new suffix.','source_originals_modified':False,'Git_commands':0}
    write_new(admin/'BUILD_COMPLETE.json',json_bytes(complete));print(json.dumps({'completed':True,'staging_only':True,'delta_manifest':delta_binding,'physical_files':len(physical),'build_complete':bind(admin/'BUILD_COMPLETE.json')}),flush=True)

def finalize(entry_path):
    e=json.loads(read_bytes(entry_path));require(e['approved_verified_delta_push_plan'] is True,'Root QA/push-plan entry')
    require(e['code']==bind(__file__) and e['pure_helper']==bind(H/'prepare_completed_phase_delta.py') and e['protocol']['path']==str(BUILDER_PROTOCOL),'Exact frozen builder/helper/protocol')
    p=checked_json(e['protocol']);require(p['code']==e['code'] and p['pure_helper']==e['pure_helper'],'Protocol code/helper closure')
    build=checked_json(e['build_complete']);require(build['completed'] and build['PASS'],'Completed staged delta')
    check_reviews(e['independent_delta_reviews'],completed_PASS=True)
    require(e['delta_manifest']==build['delta_manifest'],'Exact independently checked delta')
    checked(e['delta_manifest'])
    recheck_parent_bindings(build)
    for key in ('parent_delta_manifest','parent_chain','previous_index'):checked(build[key])
    repo=Path(build['repository']);root=repo/REPO_PREFIX;files={r['path']:r for r in build['physical_files']}
    # QA outputs live outside the manifest they verify; exact Git plan binds them.
    for i,item in enumerate(e['independent_delta_reviews']):
        rel=DELTA+'/verification/REVIEW_%02d.json'%i;source=checked(item['binding']);write_new(root/rel,read_bytes(source))
        files[REPO_PREFIX+rel]={'path':REPO_PREFIX+rel,'sha256':sha(root/rel),'size_bytes':len(read_bytes(root/rel))}
    rel=DELTA+'/BUILD_COMPLETE.json';write_new(root/rel,read_bytes(checked(e['build_complete'])))
    files[REPO_PREFIX+rel]={'path':REPO_PREFIX+rel,'sha256':sha(root/rel),'size_bytes':len(read_bytes(root/rel))}
    for r in files.values():require(sha(repo/r['path'])==r['sha256'],'Staged delta bytes drift')
    batches=pure.bounded_batches(list(files.values()))
    plan={'authorized_main_push':True,'repository':str(repo),'remote':'https://github.com/CanelE452/pallet-6d-pose.git','branch':'main','initial_remote_main':build['expected_remote_main'],'manifest_sha256':e['delta_manifest']['sha256'],'batch_max_bytes':480*1024**2,'batches':[{'files':b,'message':'Publish V189 sampled-domain completed source records and preserved failures (%d/%d)'%(i+1,len(batches))} for i,b in enumerate(batches)],'scope':'New immutable V189 delta only plus global README; no old manifest-covered path changes.'}
    out=Path(e['output_plan']);require(P/'publication_root_preparation' in out.parents,'Root admin sealed plan only');write_new(out,json_bytes(plan))
    print(json.dumps({'completed':True,'push_plan':bind(out),'batches':len(batches),'Git_commands':0}),flush=True)

def prepare():
    checks=[]
    old=b'old index\n';result=pure.versioned_index_bytes(old,'V189','archive/new/PORTABLE_REPORT_KO.md');checks.append(('new_index_old_bytes_suffix',result.endswith(old)))
    oldreadme=b'unchanged prior global readme';checks.append(('global_readme_old_suffix',(b'new header\n'+oldreadme).endswith(oldreadme)))
    checks.append(('delta_and_chain_no_self_hash_cycle',DELTA+'/DELTA_MANIFEST.json'!=NEW_CHAIN))
    try:pure.assert_new_paths([{'status':'PUBLISHED_RAW','published_relative':'EXPERIMENT_INDEX_KO.md','gzipchunks':None}],[{'status':'PUBLISHED_RAW','published_relative':'EXPERIMENT_INDEX_KO.md','gzipchunks':None}]);ok=False
    except AssertionError:ok=True
    checks.append(('old_index_overwrite_rejected',ok))
    check={'completed':True,'PASS':all(v for _,v in checks),'checks':[{'name':k,'PASS':v} for k,v in checks],'invented_only':True,'phase_artifact_enumeration_hash_copy_or_repo_Git_access':0}
    pure.write(H/'GENERATED_BUILDER_CHECKS.json',check)
    protocol={'completed':True,'prepared_only':True,'code':bind(__file__),'pure_helper':bind(H/'prepare_completed_phase_delta.py'),'actual_modes':['Root --build-entry: verify frozen completed phase/inventory/base/index/readme, invoke existing sealed copier, create source-only immutable delta+versioned index+chain; update only global readme preserving old bytes. No Git.','Root --push-plan-entry: require independent completed/PASS delta reviews, rehash exact physical stage, add verification/BUILD receipts outside manifest, seal delta-only480MiB push plan. No Git.'],'immutable_base':'Initial PUBLICATION_MANIFEST.json plus BOTH V187 and V188 DELTA_MANIFEST/chain/index and every old covered source/derived path never written. Exact parentchain closure is required; collision rejection includes logical files, gzip chunks/descriptors and cycle-excluded parent manifests/chain.','source_scope':V.name+' only; candidate prefix and source absolute identities checked before copy. All original phase code/results/failures/figures/tables included except declared exclusions.','portable':'Root creates relative PORTABLE_REPORT_KO.md before inventory; builder checks it exists in selected candidate and contains no /home/ or file://. Independent link QA is still required.','new_names':{'delta':DELTA+'/DELTA_MANIFEST.json','index':NEW_INDEX,'chain':NEW_CHAIN},'required_parent_paths':{'base':'PUBLICATION_MANIFEST.json','V187_delta':'publication/deltas/V187_20261024_v1/DELTA_MANIFEST.json','V187_chain':'MANIFEST_CHAIN_AFTER_V187.json','V187_index':'EXPERIMENT_INDEX_THROUGH_V187_KO.md','V188_delta':PARENT_DELTA,'V188_chain':PARENT_CHAIN,'V188_index':PARENT_INDEX},'cycle_avoidance':'Delta manifest excludes its own bytes, chain and later QA; strict Git physical-file plan binds those separately.','initial_make_push_plan_reuse':False,'existing_strict_pushrunner':'Root executes publication_root_preparation/push_publication_batches.py only after separate final QA/plan; no force/selection/model activity.','failures':'Exclusive new files and preserved staging; no automatic retry.','new_model_training_forward_threshold_labels_or_adoption':False,'actual_preparation_phase_or_repo_artifact_reads':0}
    pure.write(BUILDER_PROTOCOL,protocol)
    common={'code':protocol['code'],'pure_helper':protocol['pure_helper'],'protocol':bind(BUILDER_PROTOCOL)}
    template={'approved_completed_phase_delta_build':False,'completed_phase_frozen':False,**common,'phase':V.name,'root_scope':{'path':None,'sha256':None},'completion_reviews':[],'repository':'/home/minjae/Documents/pallet-visibility-publication-20261024','base_manifest':{'path':None,'sha256':None},'parent_delta_manifest':{'path':None,'sha256':None},'parent_chain':{'path':None,'sha256':None},'base_index':{'path':None,'sha256':None},'ancestor_publications':[{'tag':tag,'delta_manifest':{'path':None,'sha256':None},'chain':{'path':None,'sha256':None},'index':{'path':None,'sha256':None}} for tag,_ in pure.ANCESTOR_PHASES],'global_readme_before':{'path':None,'sha256':None},'inventory_complete':{'path':None,'sha256':None},'candidate_manifest':{'path':None,'sha256':None},'copy_packager':{'path':str(PACKAGER),'sha256':None},'copy_helper':{'path':str(PACK_HELPER),'sha256':None},'expected_remote_main':'bd9c09a2e104d7bd674e540e01adf1d556a5577a','publisher_tool_bindings':[],'admin_output':str(P/'publication_root_preparation/V189_DELTA_BUILD_v1')}
    pure.write(H/'ROOT_DELTA_BUILD_ENTRY_TEMPLATE_APPROVED_FALSE.json',template)
    pure.write(H/'ROOT_DELTA_PUSH_PLAN_ENTRY_TEMPLATE_APPROVED_FALSE.json',{'approved_verified_delta_push_plan':False,**common,'build_complete':{'path':None,'sha256':None},'delta_manifest':{'path':None,'sha256':None},'independent_delta_reviews':[],'output_plan':str(P/'publication_root_preparation/ROOT_V189_DELTA_PUSH_PLAN.json')})
    print(json.dumps(check),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--build-entry',type=Path);ap.add_argument('--push-plan-entry',type=Path);a=ap.parse_args()
    if a.prepare:prepare()
    elif a.build_entry:build(a.build_entry)
    elif a.push_plan_entry:finalize(a.push_plan_entry)
    else:ap.error('--prepare, root --build-entry or root --push-plan-entry required')
