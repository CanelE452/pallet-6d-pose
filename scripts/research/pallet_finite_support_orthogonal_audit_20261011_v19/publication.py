"""Prepare V19 public bytes in a separate checkout; never stage, commit or push.

Root executes freeze/run after the actual audits finish. Only completed output
folders and actual own-checker input declarations form the dependency closure.
Historical receipt/protection JSON is copied as bytes, not recursively resolved.
Large gzip files are sliced into 40 MiB compressed-byte parts without decoding.
An ordered-part streaming hash proves reconstruction; a stage raw hardlink is
explicitly a reused original, not a newly restored file or a portable-reader PASS.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import traceback

FAMILY='pallet_finite_support_orthogonal_audit_20261011_v19'
PARENT='bb1c3a97c46703cd960220afe969e11da698a381'
CODE='scripts/research/'+FAMILY
DOC='_docs/experiments/'+FAMILY
LATEST='_docs/experiments/pallet_observation_refiner_status_20261011/LATEST_KO.md'
PART_BYTES=40*1024*1024
ENCODING='ordered byte slices of original gzip; concatenate without recompression'
FILES=('PUBLICATION_PROTOCOL.json','PUBLICATION_STARTED.json','PUBLICATION_CHECKS.json','PUBLICATION_PATHS.nul')
GENERATED=('PUBLIC_RELOCATION_MANIFEST.json','ARCHIVE_MANIFEST.json','PUBLICATION_MANIFEST.json','.gitignore')
ALLOWED={'.py','.md','.json','.jsonl','.gz','.csv','.png','.txt','.pt'}
HEAD_SHAS={
 'GEOMETRY_ONLY':'d188dcc68bd8795c88232d5bf1b85259d684695723b96809017abd47d6ac009b',
 'IMAGE_NO_ROLE':'9c52a2e2036ee8f65ba2e191bdcbebe93ac3ecde60a6aa31e71a4ffbadc0f835',
 'IMAGE_ROLE':'882a6eddc964258abfbc1ad3baeee380ab9fac42b3b90c7f64166bd98d777d5d'}
TRAINING=('TRAINING_COMPLETION.json','AUTHORIZED_TRAINING_START.json','FORMAL_UPDATE_ROWS.jsonl','TRAIN_LOGS.jsonl')
LIMITS=dict(new_training_model_RGB_GT_metric_pose_timing_calls=0,
 historical_nested_declarations_opened_as_live_dependencies=False,
 full_historical_V16_scalar_PASS_claim=False,physical_truth_or_accuracy_goal_certified=False,
 Git_index_stage_commit_push_or_source_writes=False,portable_reader_PASS_fabricated=False,
 gzip_decompression_or_recompression_calls=0)

def require(value,message):
    if not value:raise ValueError(message)

def safe(path):
    path=Path(path).absolute()
    require('..' not in path.parts and not any(p.is_symlink() for p in (path,*path.parents)),'unsafe/symlink '+str(path))
    return path

def relative(value):
    path=Path(value)
    require(isinstance(value,str) and not path.is_absolute() and path.parts and '..' not in path.parts and '\n' not in value and '\0' not in value,'unsafe relative path')
    return path.as_posix()

def public_path(value):
    value=relative(value)
    require(value.startswith(('scripts/research/','_docs/experiments/')),'outside publication trees: '+value)
    return value

def digest(path):
    path=safe(path);require(path.is_file(),'missing regular file '+str(path));h=hashlib.sha256();count=0
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block);count+=len(block)
    require(count==path.stat().st_size,'file changed while hashing '+str(path))
    return dict(bytes=count,sha256=h.hexdigest())

def binding(path):return dict(path=str(safe(path)),**digest(path))
def same(a,b):return all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def read(path):return json.loads(safe(path).read_bytes())
def encoded(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()

def write_new(path,value,raw=False):
    path=safe(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(value if raw else encoded(value));stream.flush();os.fsync(stream.fileno())

def git(repo,*args,data=None):
    return subprocess.run(['git','--no-optional-locks','-C',str(repo),*args],input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout

def git_state(repo):
    # Neither hash-object without -w nor these reads changes an index or ref.
    repo=safe(repo);index=Path(git(repo,'rev-parse','--path-format=absolute','--git-path','index').decode().strip())
    return dict(HEAD=git(repo,'rev-parse','HEAD').decode().strip(),index=digest(index),
        staged=hashlib.sha256(git(repo,'diff','--cached','--no-ext-diff','--binary')).hexdigest(),
        tracked=hashlib.sha256(git(repo,'diff','--no-ext-diff','--binary')).hexdigest(),
        status=hashlib.sha256(git(repo,'status','--porcelain=v1','-z','-uall')).hexdigest())

def committed(repo):
    result={}
    for item in git(repo,'ls-tree','-r','-z','HEAD').split(b'\0'):
        if item:
            meta,name=item.split(b'\t',1);mode,kind,oid=meta.decode().split();result[name.decode()]=(mode,kind,oid)
    return result

def allow_file(path):
    require(path.suffix in ALLOWED or path.name=='.gitignore' or re.fullmatch(r'.+\.part[0-9]{3}',path.name),'unsupported artifact '+str(path))

def evidence(args):
    folders={}
    for item in args.evidence:
        name,sep,value=item.partition('=');require(sep and name not in folders,'unique NAME=absolute folder')
        folder=safe(value);require(folder.is_dir() and folder.is_relative_to(args.private) and folder!=args.private,'explicit completed private subfolder')
        folders[name]=folder
    required={'cpu','failed_geometry','geometry','validation','evaluation','statistics','runtime','runtime_checks','figures','correspondence'}
    require(set(folders)==required,'exact evidence folders required: '+','.join(sorted(required)))
    require(len(set(folders.values()))==len(folders) and all(not p.is_relative_to(q) for k,p in folders.items() for j,q in folders.items() if k!=j),'disjoint evidence folders')
    require(all(not args.receipts.is_relative_to(p) and not p.is_relative_to(args.receipts) for p in folders.values()),'own publication receipts disjoint from frozen evidence')
    return folders

def authority(folders):
    specs={
      'cpu':('CPU_CHECKS.json','observed_frame_actual_CPU_checks_v19'),
      'geometry':('GEOMETRY_RECEIPT.json','observed_frame_repair_geometry_receipt_v19'),
      'validation':('VALIDATION_CHECKS.json','observed_frame_hybrid_scalar_checks_v19'),
      'statistics':('STATISTICS_CHECKS.json','independent_observed_frame_statistics_checks_v19'),
      'runtime':('RUNTIME_RECEIPT.json','fresh_observed_frame_primary_runtime_receipt_v19'),
      'runtime_checks':('RUNTIME_CHECKS.json','independent_observed_frame_runtime_checks_v19'),
      'figures':('FIGURE_BINDINGS.json','observed_frame_results_figure_bindings_v19'),
      'correspondence':('CORRESPONDENCE_CHECKS.json','saved_observed_frame_correspondence_checks_v19')}
    proofs={};values={}
    for key,(name,schema) in specs.items():
        path=folders[key]/name;value=read(path)
        require(value.get('schema')==schema and value.get('complete') is True,'actual completed '+key)
        if key in ('cpu','validation','statistics','runtime_checks','correspondence'):require(value.get('passed') is True,'actual independent PASS '+key)
        require(value.get('exception') is None and value.get('error') is None,'actual no exception '+key)
        proofs[key]=binding(path);values[key]=value
    require(values['cpu']['tests_run']==6 and not values['cpu']['failures'] and not values['cpu']['errors'] and not values['cpu']['cleanup_errors'] and values['cpu']['source_unchanged'] is True,'CPU6 actual PASS')
    require(values['geometry']['actual_execution']==dict(frames_started=245,geometry_rows=490,ledger_rows=245,frames_complete=245) and not values['geometry']['cleanup_errors'] and values['geometry']['environment_restored'],'full geometry cleanup')
    v=values['validation'];require(v['failure_count']==0 and v['actual_counts']==dict(v['actual_counts'],geometry_rows=490,ledger_rows=245,frames=245) and v['old_full_V16_validation_passed'] is False,'full V19 scalar PASS, old FAIL remains')
    c=values['statistics']['actual_counts']
    require(all(c[k]==n for k,n in (('saved_error_rows_streamed',1960),('saved_error_rows_selected',1715),('moment_distributions_checked',675),('moment_scalars_checked',5400),('paired_groups_checked',54),('metric_CI_slots_checked',162))),'full recorded statistics audit')
    r=values['runtime'];require(r['status']=='COMPLETE' and r['serialized_rows']==150 and r['statistics_official'] is True and not r['cleanup_errors'],'fresh150 runtime')
    t=values['runtime_checks'];require(t['rows']==150 and t['measured']==130 and t['summary_slots']==56 and t['failure_count']==0,'150/130/56 independent clock audit')
    fig=values['figures'];require(fig['actual_counts']['PNG_files']==9 and fig['actual_counts']['case_panels']==18,'completed fixed9PNG/18panels')
    for key,names in {'validation':('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json'),'statistics':('STATISTICS_CHECK_PROTOCOL.json','STATISTICS_CHECK_STARTED.json'),'runtime_checks':('RUNTIME_CHECK_PROTOCOL.json','RUNTIME_CHECK_STARTED.json')}.items():
        p,s=(read(folders[key]/n) for n in names);check=values[key]
        require(p['inputs']==s['inputs']==check['inputs'],'actual own input closure '+key)
        for item in (s['own_protocol'],check['own_protocol']):require(same(binding(folders[key]/names[0]),item),'actual own protocol '+key)
    return proofs,values

class Inventory:
    def __init__(self,args):
        self.args=args;self.files={};self.origins={};self.external=[];self.archive_cache={};self.archives={}

    def location(self,path):
        path=safe(path)
        if path.is_relative_to(self.args.source_repo):return 'repo',public_path(path.relative_to(self.args.source_repo).as_posix())
        require(path.is_relative_to(self.args.private),'unmapped absolute dependency '+str(path))
        rel=relative(path.relative_to(self.args.private).as_posix())
        if len(Path(rel).parts)==1 and path.suffix=='.py':return 'repo',CODE+'/'+rel
        return 'artifact',rel

    def add(self,path,expected=None,location=None):
        path=safe(path)
        if location is not None:
            allow_file(Path(location[1]))
            require(path.suffix in ALLOWED or path.name.endswith('.md.ready'),'unsupported explicitly mapped source '+str(path))
        else:allow_file(path)
        actual=digest(path)
        if expected is not None:require(same(actual,expected),'current required byte mismatch '+str(path))
        root,rel=location or self.location(path);rel=relative(rel)
        target=public_path(rel if root=='repo' else DOC+'/'+rel)
        entry=dict(source=str(path),target=target,**actual)
        if target in self.files:require(same(entry,self.files[target]),'two source versions map to one public target '+target)
        else:self.files[target]=entry
        record=dict(original_path=str(path),root=root,relative_path=rel,**actual)
        if str(path) in self.origins:require(record==self.origins[str(path)],'origin maps to two physical locations')
        else:self.origins[str(path)]=record

    def declared(self,key,value):
        require(isinstance(value,dict) and all(k in value for k in ('path','bytes','sha256')),'own input declaration '+key)
        if value.get('origin')=='external_readonly_dependency':
            self.external.append(dict(alias=key,declaration=value,current_external_assets_not_verified=True));return
        origin=value.get('origin');path=Path(value['path'])
        if origin=='public_repository':
            require(not path.is_absolute(),'relative explicit public origin required');path=self.args.source_repo/public_path(path.as_posix())
        else:require(origin is None and path.is_absolute(),'explicit public/private input required '+key)
        self.add(path,value)

    def legacy_archive(self,entry):
        path=Path(entry['source']);require(path.is_relative_to(self.args.source_repo),'legacy archive outside source repo')
        for folder in path.parents:
            if not folder.is_relative_to(self.args.source_repo):break
            manifest=folder/'ARCHIVE_MANIFEST.json'
            if not manifest.is_file():continue
            if str(manifest) not in self.archive_cache:self.archive_cache[str(manifest)]=read(manifest)
            declared_manifest=self.archive_cache[str(manifest)]
            for record in declared_manifest['files']:
                original=record['original']
                if safe(folder/relative(original['path']))!=path:continue
                require(record.get('storage','ordered_parts' if record.get('parts') else 'direct_original')=='ordered_parts' and same(original,entry),'large parent original needs exact committed ordered parts')
                width=record.get('part_size_bytes',record.get('part_bytes',declared_manifest.get('part_size_bytes',declared_manifest.get('part_bytes'))))
                require(width in (PART_BYTES,50*1024*1024),'explicit historical part-size declaration')
                parts=[]
                for index,part in enumerate(record['parts']):
                    name=relative(part['path']);require(name.endswith('.part%03d'%index) and 0<part['bytes']<=width,'ordered declared-width parent parts')
                    source=safe(folder/name);self.add(source,part);parts.append(dict(path=public_path(source.relative_to(self.args.source_repo).as_posix()),bytes=part['bytes'],sha256=part['sha256']))
                require(parts and sum(p['bytes'] for p in parts)==entry['bytes'],'parent archive byte total')
                self.add(manifest);return parts,width
        raise ValueError('no authoritative ordered archive for '+str(path))

def checkpoints(inventory,args):
    source=args.source_repo/'scripts/research/pallet_three_head_observation_20261010_v7/pipeline.py'
    tree=ast.parse(source.read_bytes());values=[n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CHECKPOINT_SHAS' for t in n.targets)]
    require(len(values)==1 and ast.literal_eval(values[0])==HEAD_SHAS,'immutable V7 checkpoint declarations')
    inventory.add(source);completion=read(args.checkpoint_root/'TRAINING_COMPLETION.json')
    require(completion['complete'] is True and completion['formal_updates']==completion['total_updates']==9000 and completion['input_hashes_unchanged'] is True,'existing corrected training completion')
    records={r['arm']:r for r in completion['checkpoints']};require(set(records)==set(HEAD_SHAS) and len(completion['checkpoints'])==3,'same three original checkpoints')
    for arm,sha in HEAD_SHAS.items():
        path=safe(args.checkpoint_root/(arm+'.pt'));expected=records[arm]['checkpoint'];require(expected['sha256']==sha,'corrected checkpoint identity '+arm)
        inventory.add(path,expected,('artifact','checkpoints/'+path.name))
    for name in TRAINING:inventory.add(args.checkpoint_root/name,location=('artifact','checkpoints/'+name))

def collect(args):
    folders=evidence(args);proofs,values=authority(folders);inventory=Inventory(args)
    for path in sorted(args.private.glob('*.py')):inventory.add(path)
    for key,folder in folders.items():
        for path in sorted(folder.rglob('*')):
            require(not path.is_symlink(),'symlink in evidence '+str(path))
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':inventory.add(path)
    # These are the actual successful standalone checker closures, not the
    # much broader dictionaries nested inside old protection/failure receipts.
    for key in ('validation','statistics','runtime_checks'):
        for alias,value in values[key]['inputs'].items():inventory.declared(key+':'+alias,value)
    inventory.add(args.private/'GEOMETRY_READY_PROOF.json')
    supplements={}
    for item in args.supplement:
        name,sep,value=item.partition('=');require(sep and name and name not in supplements,'unique supplementary NAME=absolute path')
        path=safe(value);require(path.is_relative_to(args.private) or path.is_relative_to(args.source_repo),'supplement outside explicit immutable research roots')
        require(not path.is_relative_to(args.receipts) and not args.receipts.is_relative_to(path),'supplement overlaps own publication receipts')
        if path.is_dir():
            require(path!=args.private and path!=args.source_repo,'explicit completed subfolder, not whole research root')
            files=[]
            for q in sorted(path.rglob('*')):
                require(not q.is_symlink(),'symlink in explicit supplement')
                if q.is_file() and '__pycache__' not in q.parts and q.suffix!='.pyc':inventory.add(q);files.append(str(q))
            require(files,'empty supplementary folder');supplements[name]=dict(path=str(path),files=files)
        else:
            inventory.add(path);supplements[name]=dict(path=str(path),files=[str(path)])
    checkpoints(inventory,args)
    for item in args.document:
        name,sep,path=item.partition('=');require(sep and Path(name).name==name and Path(name).suffix=='.md','explicit basename.md=absolute document')
        inventory.add(path,location=('artifact',name))
    require(any(p['target']==DOC+'/RESULT_KO.md' for p in inventory.files.values()),'final report source must map to RESULT_KO.md (ready source remains preserved)')
    inventory.add(args.latest_source,location=('repo',LATEST))
    old=committed(args.destination)
    # Old committed direct blobs below the historical 50 MiB cap retain
    # their established storage. Only new V19 gzip has the new 40 MiB split.
    large=[p for p in inventory.files.values() if p['bytes']>PART_BYTES and not
        (Path(p['source']).is_relative_to(args.source_repo) and p['target'] in old and p['bytes']<=50*1024*1024)]
    for entry in large:
        require(Path(entry['source']).suffix=='.gz','large Git artifact must be gzip with ordered parts')
        if Path(entry['source']).is_relative_to(args.source_repo):parts,width=inventory.legacy_archive(entry)
        else:
            width=PART_BYTES
            parts=[dict(path=DOC+'/archives/'+relative(entry['target'][len(DOC)+1:])+'.part%03d'%i,bytes=min(width,entry['bytes']-i*width)) for i in range((entry['bytes']+width-1)//width)]
        inventory.archives[entry['target']]=dict(original=dict(path=entry['target'],bytes=entry['bytes'],sha256=entry['sha256']),parts=parts,part_bytes=width,source=entry['source'],legacy=Path(entry['source']).is_relative_to(args.source_repo))
    entries=[]
    for target,entry in sorted(inventory.files.items()):
        if target in inventory.archives:continue
        oid=git(args.destination,'hash-object','--stdin',data=Path(entry['source']).read_bytes()).decode().strip()
        previous=old.get(target)
        if previous and target!=LATEST:require(previous==('100644','blob',oid),'published byte conflict '+target)
        require(not previous or previous[:2]==('100644','blob'),'existing public file mode/type')
        physical=safe(args.destination/target)
        if physical.exists() and target!=LATEST:require(same(digest(physical),entry),'existing stage bytes differ '+target)
        previous_binding=None
        if target==LATEST:
            old_bytes=git(args.destination,'show','HEAD:'+LATEST)
            previous_binding=dict(bytes=len(old_bytes),sha256=hashlib.sha256(old_bytes).hexdigest())
            require(physical.is_file() and same(digest(physical),previous_binding),'LATEST working bytes must match committed parent')
        entries.append(dict(entry,git_blob=oid,existing_committed=previous is not None,previous_git_blob=previous[2] if previous else None,previous_binding=previous_binding))
    require(LATEST in old,'LATEST replacement requires existing committed parent path')
    return dict(schema='frozen_observed_frame_publication_plan_v19',family=FAMILY,proofs=proofs,
        files=entries,archives=[inventory.archives[k] for k in sorted(inventory.archives)],
        relocation_records=[inventory.origins[k] for k in sorted(inventory.origins)],external_declarations=inventory.external,
        completed_evidence_folders={k:str(v) for k,v in sorted(folders.items())},part_bytes=PART_BYTES,
        explicit_supplements=supplements,supplementary_json_content_treated_as_copied_bytes_not_recursive_dependencies=True,
        old_full_V16_scalar_FAIL_remains_FAIL=True,ROOT_visual_review_claimed=False,limits=LIMITS)

def configuration(args):
    return {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='stage'}

def protected(args):return {str(p):git_state(p) for p in dict.fromkeys((args.source_repo,args.original_repo))}

def guard(args,stage):
    for name in ('source_repo','original_repo','private','destination','receipts','checkpoint_root','latest_source'):setattr(args,name,safe(getattr(args,name)))
    require(args.private.is_dir() and args.source_repo.is_dir() and args.original_repo.is_dir(),'existing frozen roots')
    require(args.destination.is_dir() and all(not args.destination.is_relative_to(root) and not root.is_relative_to(args.destination) for root in (args.source_repo,args.original_repo,args.private)),'publication checkout disjoint from both protected repos and private inputs')
    require(args.receipts.is_relative_to(args.private) and args.receipts!=args.private,'own private receipt subfolder')
    require(not args.receipts.exists() or args.receipts.is_dir(),'receipt folder type')
    parent=git(args.destination,'rev-parse',args.parent+'^{commit}').decode().strip()
    require(parent==PARENT and git(args.destination,'rev-parse','HEAD').decode().strip()==parent,'exact published B parent required')
    require(not git(args.destination,'status','--porcelain=v1','-z','-uall'),'publication checkout must initially be clean')
    for name in FILES if stage=='freeze' else FILES[1:]:require(not (args.receipts/name).exists(),'preserve '+name)
    for name in GENERATED:require(not (args.destination/DOC/name).exists(),'new public generated output must be absent '+name)
    for name in FILES[:3]:require(not(args.destination/DOC/'publication'/name).exists(),'new public packaging receipt must be absent '+name)
    return parent

def link_verified(source,target,expected):
    source=safe(source);target=safe(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():require(same(digest(target),expected),'existing target differs '+str(target));return 'existing_byte_identical'
    try:os.link(source,target);kind='immutable_source_hardlink'
    except OSError as exc:
        if exc.errno!=18:raise
        with source.open('rb') as src,target.open('xb') as out:shutil.copyfileobj(src,out,1048576);out.flush();os.fsync(out.fileno())
        kind='exclusive_copy_cross_device'
    require(same(digest(target),expected),'copied/hardlinked byte mismatch '+str(target));return kind

def archive_run(args,item):
    original=item['original'];parts=[]
    if item['legacy']:
        for part in item['parts']:
            source=args.source_repo/part['path'];target=args.destination/part['path']
            # Existing sparse/skip-worktree blobs are materialized only after
            # their source bytes were compared with the exact committed blob.
            link_verified(source,target,part);parts.append(part)
    else:
        with safe(item['source']).open('rb') as source:
            for expected in item['parts']:
                target=safe(args.destination/expected['path']);target.parent.mkdir(parents=True,exist_ok=True)
                h=hashlib.sha256();remaining=expected['bytes']
                with target.open('xb') as output:
                    while remaining:
                        block=source.read(min(1048576,remaining));require(block,'short compressed original');output.write(block);h.update(block);remaining-=len(block)
                    output.flush();os.fsync(output.fileno())
                part=dict(expected,sha256=h.hexdigest());require(same(digest(target),part),'written part differs');parts.append(part)
            require(not source.read(1),'compressed original grew')
    h=hashlib.sha256();size=0
    for part in parts:
        require(same(digest(args.destination/part['path']),part),'ordered public part changed')
        with (args.destination/part['path']).open('rb') as stream:
            for block in iter(lambda:stream.read(1048576),b''):h.update(block);size+=len(block)
    require(same(dict(bytes=size,sha256=h.hexdigest()),original),'ordered streaming reconstruction differs')
    kind=link_verified(item['source'],args.destination/original['path'],original)
    return dict(original=original,storage='ordered_parts',parts=parts,part_bytes=item['part_bytes'],legacy_original_not_resliced=item['legacy'],
        ordered_part_streaming_reconstruction_hash_verified=True,reconstructed_duplicate_file_retained=False,
        stage_original_storage=kind,stage_original_is_reused_source_not_restored_from_parts=True)

def freeze(args):
    parent=guard(args,'freeze');before=protected(args);destination=git_state(args.destination);plan=collect(args)
    require(protected(args)==before and git_state(args.destination)==destination,'read-only preparation changed Git state')
    args.receipts.mkdir(parents=True,exist_ok=True)
    write_new(args.receipts/FILES[0],dict(schema='observed_frame_publication_protocol_v19',configuration=configuration(args),
        parent=parent,helper=binding(__file__),protected=before,destination_git=destination,plan=plan,
        frozen_before_copy_or_archive_creation=True,limits=LIMITS))

def run(args):
    parent=guard(args,'run');own=read(args.receipts/FILES[0]);before=protected(args);dest_before=git_state(args.destination)
    require(own['schema']=='observed_frame_publication_protocol_v19' and own['configuration']==configuration(args)
        and own['helper']==binding(__file__) and own['parent']==parent and own['protected']==before and own['destination_git']==dest_before
        and own['plan']==collect(args),'own frozen source/bytes/roots/Git plan differs')
    protocol=binding(args.receipts/FILES[0]);write_new(args.receipts/FILES[1],dict(protocol=protocol,inputs=own['plan']['proofs'],helper=own['helper'],limits=LIMITS))
    actions=[];archives=[];error=None;generated=[]
    try:
        for entry in own['plan']['files']:
            target=safe(args.destination/entry['target'])
            if entry['target']==LATEST:
                require(target.is_file() and git(args.destination,'hash-object',str(target)).decode().strip()==entry['previous_git_blob'],'LATEST parent bytes changed')
                pending=target.with_name('LATEST_V19_PENDING.md');require(not pending.exists(),'preserve LATEST prefix')
                with safe(entry['source']).open('rb') as src,pending.open('xb') as out:shutil.copyfileobj(src,out,1048576);out.flush();os.fsync(out.fileno())
                require(same(digest(pending),entry),'new LATEST differs');os.replace(pending,target);kind='explicit_LATEST_only_replacement'
            else:kind=link_verified(entry['source'],target,entry)
            actions.append(dict(path=entry['target'],bytes=entry['bytes'],sha256=entry['sha256'],storage=kind,existing_committed=entry['existing_committed'],previous_binding=entry['previous_binding']))
        for item in own['plan']['archives']:archives.append(archive_run(args,item))
        receipt_records=[]
        for name in FILES[:2]:
            source=args.receipts/name;target=args.destination/DOC/'publication'/name;actual=binding(source)
            link_verified(source,target,actual);generated.append(dict(path=DOC+'/publication/'+name,bytes=actual['bytes'],sha256=actual['sha256']))
            receipt_records.append(dict(original_path=str(source),root='artifact',relative_path='publication/'+name,bytes=actual['bytes'],sha256=actual['sha256']))
        relocation=dict(schema='observed_frame_public_relocation_manifest_v19',original_artifact_root=str(args.private),records=own['plan']['relocation_records']+receipt_records)
        archive_manifest=dict(schema='observed_frame_compressed_original_archive_manifest_v19',encoding=ENCODING,
            part_bytes=PART_BYTES,original_files_preserved=True,original_file_count=len(archives),files=archives,
            all_parts_paths_relative_to_repo=True,gzip_decompression_or_recompression_calls=0)
        ignored=[a['original']['path'][len(DOC)+1:] for a in archives if a['original']['path'].startswith(DOC+'/')]
        public_manifest=dict(schema='observed_frame_publication_manifest_v19',complete=True,parent=parent,
            own_protocol=protocol,files=actions,ordered_archives=archives,relocation_manifest_path=DOC+'/'+GENERATED[0],
            completed_evidence=own['plan']['proofs'],external_declarations=own['plan']['external_declarations'],
            own_publication_receipt_folder=DOC+'/publication',own_checks_copied_after_this_manifest_without_self_hash=True,
            full_historical_asset_dependency_reproduction_claimed=False,portable_copied_reader_execution_pending=True,
            current_required_original_bytes_hashed=True,ROOT_visual_review_claimed=False,limits=LIMITS)
        for name,value in zip(GENERATED,(relocation,archive_manifest,public_manifest,('\n'.join('/'+p for p in ignored)+'\n').encode())):
            path=args.destination/DOC/name;write_new(path,value,raw=name=='.gitignore');generated.append(dict(path=DOC+'/'+name,**digest(path)))
        # The secondary index/ref stays unchanged. Only LATEST is a permitted
        # tracked edit; every other changed path in this prepare step is new.
        require(git(args.destination,'rev-parse','HEAD').decode().strip()==parent,'secondary HEAD changed')
        after=git_state(args.destination);require(after['index']==dest_before['index'] and after['staged']==dest_before['staged'],'secondary index/staged bytes changed')
        require(not git(args.destination,'diff','--no-ext-diff','--binary','HEAD','--','.',':(exclude)'+LATEST),'protected existing secondary tracked bytes changed')
        require(protected(args)==before,'original/source Git state changed')
        for entry in own['plan']['files']:require(same(digest(entry['source']),entry),'source byte changed during publication')
        for item in own['plan']['archives']:require(same(digest(item['source']),item['original']),'original archive byte changed')
        publish={a['path'] for a in actions if not a['existing_committed'] or a['path']==LATEST}
        publish.update(p['path'] for a in archives for p in a['parts'] if p['path'] not in committed(args.destination))
        publish.update(e['path'] for e in generated)
        publish.add(DOC+'/publication/'+FILES[2])
        write_new(args.receipts/FILES[3],b''.join((p+'\0').encode() for p in sorted(publish)),raw=True)
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    checks=dict(schema='observed_frame_publication_checks_v19',complete=error is None,passed=error is None,
        protocol=protocol,helper=binding(__file__),parent=parent,configuration=configuration(args),error=error,
        actual_direct_files=actions,actual_ordered_archives=archives,generated_files=generated,
        source_protection_after=protected(args),original_and_source_Git_state_unchanged=protected(args)==before,
        limits=LIMITS,portable_copied_readers_executed=False,publication_commit_or_push_performed=False,
        partial_destination_prefix_preserved_on_failure=True,self_receipt_copy_not_counted_as_an_input_or_hashed_in_itself=True)
    write_new(args.receipts/FILES[2],checks)
    if error is None:link_verified(args.receipts/FILES[2],args.destination/DOC/'publication'/FILES[2],binding(args.receipts/FILES[2]))
    print(json.dumps({k:checks[k] for k in ('complete','passed','error','original_and_source_Git_state_unchanged')}),flush=True)
    if error:raise SystemExit(1)

def restore(args):
    """Independent future reader: concatenate public parts; never open /tmp origins."""
    repo=safe(args.repo);manifest=safe(args.manifest);out=safe(args.output)
    require(manifest.is_relative_to(repo) and out.is_relative_to(repo),'restore manifest/receipts inside copied repo')
    value=read(manifest);require(value['schema']=='observed_frame_compressed_original_archive_manifest_v19' and value['encoding']==ENCODING and value['part_bytes']==PART_BYTES,'exact compressed-byte archive manifest')
    names=('PUBLIC_RESTORE_PROTOCOL.json','PUBLIC_RESTORE_STARTED.json','PUBLIC_RESTORE_CHECKS.json');inputs={'manifest':binding(manifest),'helper':binding(__file__)};initial={}
    for item in value['files']:
        original=item['original'];name=public_path(original['path']);target=safe(repo/name);require(name not in initial,'duplicate restore original')
        require(item['storage']=='ordered_parts' and item['parts'],'nonempty ordered archive')
        width=item['part_bytes'];require(width in (PART_BYTES,50*1024*1024) and (item['legacy_original_not_resliced'] or width==PART_BYTES),'new40MiB/exact historical declared width')
        require(sum(p['bytes'] for p in item['parts'])==original['bytes'],'compressed original byte total')
        for index,part in enumerate(item['parts']):
            path=public_path(part['path']);require(path.endswith('.part%03d'%index) and 0<part['bytes']<=width,'bounded ordered parts')
            actual=binding(repo/path);require(same(actual,part),'public part bytes differ');inputs['part:'+path]=actual
        initial[name]=dict(present=target.exists(),binding=digest(target) if target.exists() else None)
        require(not target.with_name(target.name+'.RESTORE_PENDING').exists(),'preserve restore prefix')
        if target.exists():require(same(initial[name]['binding'],original),'existing original differs')
    out.mkdir(parents=True,exist_ok=True)
    for name in names if args.stage=='restore-freeze' else names[1:]:require(not(out/name).exists(),'preserve '+name)
    if args.stage=='restore-freeze':write_new(out/names[0],dict(schema='observed_frame_public_restore_protocol_v19',inputs=inputs,initial=initial,limits=LIMITS));return
    own=read(out/names[0]);require(own['inputs']==inputs and own['initial']==initial and own['limits']==LIMITS,'own frozen archive/target state changed')
    proof=binding(out/names[0]);write_new(out/names[1],dict(protocol=proof,inputs=inputs));results=[];error=None
    try:
        for item in value['files']:
            original=item['original'];target=safe(repo/original['path']);exists=initial[original['path']]['present']
            if not exists:
                target.parent.mkdir(parents=True,exist_ok=True);pending=target.with_name(target.name+'.RESTORE_PENDING')
                with pending.open('xb') as stream:
                    for part in item['parts']:
                        with safe(repo/part['path']).open('rb') as source:shutil.copyfileobj(source,stream,1048576)
                    stream.flush();os.fsync(stream.fileno())
                require(same(digest(pending),original),'restored original mismatch');os.link(pending,target);pending.unlink()
            require(same(digest(target),original),'final restored original mismatch')
            results.append(dict(original=original,actually_created_from_public_ordered_parts=not exists,existing_original_reused=exists))
        require(all(binding(v['path'])==v for v in inputs.values()),'archive/code inputs changed during restore')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    write_new(out/names[2],dict(schema='observed_frame_public_restore_checks_v19',complete=error is None,passed=error is None,protocol=proof,inputs=inputs,files=results,error=error,limits=LIMITS))
    if error:raise SystemExit(1)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run','restore-freeze','restore-run'))
    p.add_argument('--repo',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--source-repo',type=Path,default=Path('/tmp/pallet-kp-publication-20261010-v6'))
    p.add_argument('--original-repo',type=Path,default=Path('/home/minjae/Documents/github/pallet-pose'))
    p.add_argument('--private',type=Path,default=Path('/tmp/pallet-finite-support-orthogonal-private-20261011-v19'))
    p.add_argument('--destination',type=Path);p.add_argument('--parent',default=PARENT)
    p.add_argument('--receipts',type=Path);p.add_argument('--evidence',action='append',default=[],metavar='NAME=ABSOLUTE_FOLDER')
    p.add_argument('--document',action='append',default=[],metavar='PUBLIC_NAME.md=ABSOLUTE_FILE')
    p.add_argument('--supplement',action='append',default=[],metavar='NAME=ABSOLUTE_FILE_OR_COMPLETED_FOLDER')
    p.add_argument('--latest-source',type=Path)
    p.add_argument('--checkpoint-root',type=Path,default=Path('/dev/shm/pallet-kp-supervision-repair-private-20261010/learned_fits'))
    args=p.parse_args()
    if args.stage.startswith('restore-'):
        require(all(getattr(args,n) is not None for n in ('repo','manifest','output')),'restore requires repo/manifest/output');restore(args)
    else:
        require(all(getattr(args,n) is not None for n in ('destination','receipts','latest_source')),'publication requires destination/receipts/latest-source')
        (freeze if args.stage=='freeze' else run)(args)

if __name__=='__main__':main()
