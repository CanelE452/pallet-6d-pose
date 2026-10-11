"""Publish explicitly selected private packages as bytes; ROOT alone executes.

No experiment is run and no producer JSON is rewritten. Completed reader input
protocols must be explicitly selected; historical nested declarations are not
live dependencies. Large gzip originals retain their bytes and are represented
in Git by ordered 40 MiB parts. Existing publication manifests remain unchanged.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import traceback

PART_BYTES=40*1024*1024
ENCODING='ordered byte slices of original gzip; concatenate without recompression'
ALLOWED={'.py','.md','.json','.jsonl','.gz','.csv','.png','.txt','.pt'}
PARENT_HELPER_SHA='ff28a731475cc114db41d21a60a56af19a23e95593633b789436ab57b43191ea'
OLD_FAMILY='pallet_finite_support_orthogonal_audit_20261011_v19'
FILES=('PUBLICATION_NEXT_PROTOCOL.json','PUBLICATION_NEXT_STARTED.json','PUBLICATION_NEXT_CHECKS.json','PUBLICATION_NEXT_PATHS.nul')
LIMITS=dict(new_experiment_training_model_RGB_GT_pose_metric_timing_calls=0,
    producer_JSON_rewritten=False,historical_nested_declarations_opened_as_live_dependencies=False,
    original_and_source_repository_writes=False,Git_add_commit_push_performed=False,
    portable_reader_PASS_or_physical_accuracy_claimed=False,gzip_decompression_or_recompression_calls=0,
    original_large_files_preserved=True,stage_raw_original_is_reused_source_not_restored=True)

# BEGIN EXACT V19 HELPERS
# The following definitions are copied, with whole canonical AST equality
# checked against the immutable helper before any inventory or copy action.
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

def committed(repo):
    result={}
    for item in git(repo,'ls-tree','-r','-z','HEAD').split(b'\0'):
        if item:
            meta,name=item.split(b'\t',1);mode,kind,oid=meta.decode().split();result[name.decode()]=(mode,kind,oid)
    return result

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
# END EXACT V19 HELPERS

def source_proof(args):
    require(binding(args.semantics)['sha256']==PARENT_HELPER_SHA,'exact unchanged V19 helper source')
    parent=ast.parse(args.semantics.read_bytes());current=ast.parse(Path(__file__).read_bytes())
    names=('require','safe','relative','public_path','digest','binding','same','read','encoded','write_new','git','committed','link_verified','archive_run')
    p={n.name:n for n in parent.body if isinstance(n,ast.FunctionDef)};c={n.name:n for n in current.body if isinstance(n,ast.FunctionDef)}
    require(all(ast.dump(p[n],include_attributes=False)==ast.dump(c[n],include_attributes=False) for n in names),'whole V19 byte/copy/archive helper AST changed')
    return dict(schema='unchanged_publication_byte_and_archive_semantics_v21',original_helper=binding(args.semantics),
        copied_functions=list(names),whole_function_AST_equal=True,canonical_Git_state_uses_logical_index_entries_not_physical_index_timestamps=True)

def git_state(repo):
    return dict(HEAD=git(repo,'rev-parse','HEAD').decode().strip(),
        index_entries_sha256=hashlib.sha256(git(repo,'ls-files','--stage','-z')).hexdigest(),
        staged_diff_sha256=hashlib.sha256(git(repo,'diff','--cached','--no-ext-diff','--binary')).hexdigest(),
        tracked_diff_sha256=hashlib.sha256(git(repo,'diff','--no-ext-diff','--binary')).hexdigest(),
        status_sha256=hashlib.sha256(git(repo,'status','--porcelain=v1','-z','-uall')).hexdigest())

def parse_named(values):
    result={}
    for item in values:
        name,sep,value=item.partition('=');require(sep and name and name not in result,'unique NAME=absolute path')
        result[name]=safe(value);require(result[name].is_absolute(),'absolute path required')
    return result

def family_name(value):
    require(re.fullmatch(r'pallet_[a-zA-Z0-9_]+',value) is not None,'owned research family basename')
    return value

def owned_target(family,name,code=False):
    return public_path(('scripts/research/' if code else '_docs/experiments/')+family_name(family)+'/'+relative(name))

def allowed(path):
    require(path.suffix in ALLOWED- {'.pt'} or path.name.endswith('.md.ready') or path.name=='.gitignore'
        or re.fullmatch(r'.+\.part[0-9]{3}',path.name),'unsupported selected file or model weight '+str(path))

class Inventory:
    def __init__(self,args):
        self.args=args;self.files={};self.records={};self.externals=[];self.readers={};self.byte_cache={}
        self.packages=parse_named(args.package)
        require(self.packages and all(p.is_dir() for p in self.packages.values()),'explicit existing packages required')
        for family in self.packages:family_name(family)
        roots=list(self.packages.values());require(len(set(roots))==len(roots) and all(not p.is_relative_to(q) for i,p in enumerate(roots) for j,q in enumerate(roots) if i!=j),'disjoint packages')
        self.families=set(self.packages)|{OLD_FAMILY};self.history=read(args.history_manifest)
        require(self.history['schema']=='observed_frame_public_relocation_manifest_v19','unchanged V19 explicit relocation base')
        hp=args.history_publication/'PUBLICATION_PROTOCOL.json';hc=args.history_publication/'PUBLICATION_CHECKS.json'
        p,c=read(hp),read(hc)
        require(p['schema']=='observed_frame_publication_protocol_v19' and c['schema']=='observed_frame_publication_checks_v19'
            and c['complete'] is c['passed'] is True and c['error'] is None and same(c['protocol'],binding(hp)),'actual completed prepared V19 publication authority')
        require(all(p['protected'][root][key]==c['source_protection_after'][root][key] for root in p['protected']
            for key in ('HEAD','staged','tracked','status')),'historical semantic source protection')
        self.prepared={}
        for item in c['actual_direct_files']+c['generated_files']:
            target=public_path(item['path']);require(target not in self.prepared or same(item,self.prepared[target]),'one prepared byte version')
            self.prepared[target]=dict(bytes=item['bytes'],sha256=item['sha256'])
        self.prepared_archives={}
        for item in c['actual_ordered_archives']:
            target=public_path(item['original']['path']);self.prepared_archives[target]=item
            for part in item['parts']:self.prepared[public_path(part['path'])]=dict(bytes=part['bytes'],sha256=part['sha256'])
        self.history_authority=dict(protocol=binding(hp),checks=binding(hc),
            original_and_source_Git_state_unchanged=c['original_and_source_Git_state_unchanged'],
            historical_physical_index_flag_preserved=True,semantic_HEAD_staged_tracked_status_unchanged=True)
        self.old_records={}
        for record in self.history['records']:
            require(set(record)=={'original_path','root','relative_path','bytes','sha256'},'exact base relocation fields')
            require(record['root'] in ('repo','artifact'),'explicit base relocation root')
            target=public_path(record['relative_path'] if record['root']=='repo' else owned_target(OLD_FAMILY,record['relative_path']))
            self.record(record['original_path'],target,record);self.old_records[record['original_path']]=target
        for name in ('PUBLICATION_PROTOCOL.json','PUBLICATION_CHECKS.json'):
            target=owned_target(OLD_FAMILY,'publication/'+name);actual=binding(args.history_publication/name)
            self.prepared[target]=dict(bytes=actual['bytes'],sha256=actual['sha256'])

    def record(self,source,target,actual):
        item=dict(original_path=str(safe(source)),root='repo',relative_path=public_path(target),bytes=actual['bytes'],sha256=actual['sha256'])
        require(type(item['bytes']) is int and item['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',item['sha256']) is not None,'exact byte declaration')
        previous=self.records.get(item['original_path'])
        require(previous is None or previous==item,'conflicting explicit original relocation '+item['original_path'])
        self.records[item['original_path']]=item

    def location(self,path):
        path=safe(path)
        for repo in (self.args.source_repo,self.args.original_repo):
            if path.is_relative_to(repo):return public_path(path.relative_to(repo).as_posix())
        for family,root in self.packages.items():
            if path.is_relative_to(root):
                rel=relative(path.relative_to(root).as_posix())
                return owned_target(family,rel,code=len(Path(rel).parts)==1 and path.suffix=='.py')
        require(str(path) in self.old_records,'unmapped private origin; supply explicit --file: '+str(path))
        return self.old_records[str(path)]

    def add(self,path,expected=None,target=None):
        path=safe(path);allowed(path)
        if path not in self.byte_cache:self.byte_cache[path]=digest(path)
        actual=self.byte_cache[path]
        require(expected is None or same(actual,expected),'required source byte mismatch '+str(path))
        target=public_path(target or self.location(path));allowed(Path(target))
        item=dict(source=str(path),target=target,**actual)
        require(target not in self.files or same(item,self.files[target]),'two versions map to one public target '+target)
        self.files.setdefault(target,item)
        # A document alias is copied as an additional identical file. The first
        # explicit mapping is canonical for this origin and cannot be changed.
        if str(path) not in self.records:self.record(path,target,actual)
        else:require(same(self.records[str(path)],actual),'existing relocation byte mismatch '+str(path))
        return item

    def folder(self,family,folder):
        require(family in self.packages and folder.is_dir() and folder.is_relative_to(self.packages[family]) and folder!=self.packages[family],'owned explicit evidence subfolder')
        require(not folder.is_relative_to(self.args.receipts) and not self.args.receipts.is_relative_to(folder),'evidence overlaps own receipts')
        count=0
        for path in sorted(folder.rglob('*')):
            require(not path.is_symlink(),'symlink in evidence')
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':self.add(path);count+=1
        require(count>0,'empty evidence folder')

    def reader(self,family,alias,path):
        key=family+':'+alias;require(key not in self.readers,'unique explicit reader protocol')
        value=read(path);require(isinstance(value.get('inputs'),dict),'reader own input declarations')
        self.add(path);self.readers[key]=dict(protocol=binding(path),schema=value.get('schema'),inputs=value['inputs'])
        for name,item in value['inputs'].items():
            require(isinstance(item,dict) and all(k in item for k in ('path','bytes','sha256')),'exact own reader binding '+name)
            if item.get('origin')=='external_readonly_dependency':
                self.externals.append(dict(reader=key,alias=name,declaration=item,current_external_asset_not_opened=True));continue
            origin=item.get('origin');source=Path(item['path'])
            if origin=='public_repository':require(not source.is_absolute(),'relative explicit public path');source=self.args.source_repo/public_path(source.as_posix())
            else:require(origin is None and source.is_absolute(),'explicit current public/private reader binding '+name)
            self.add(source,item)

def selector(item,inventory):
    key,sep,path=item.partition('=');family,colon,name=key.partition(':')
    require(sep and colon and name and family in inventory.families,'FAMILY:NAME=absolute path')
    return family,name,safe(path)

def parent_check(args,entry,old):
    previous=old.get(entry['target']);physical=safe(args.destination/entry['target'])
    if previous:
        require(previous[:2]==('100644','blob'),'parent target mode/type')
        raw=git(args.destination,'show','HEAD:'+entry['target'])
        require(same(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),entry),'immutable parent byte conflict '+entry['target'])
    if physical.exists():require(same(digest(physical),entry),'existing destination byte conflict '+entry['target'])
    return dict(entry,existing_committed=previous is not None)

def legacy_archive(args,inventory,entry):
    source=Path(entry['source'])
    for folder in source.parents:
        manifest=folder/'ARCHIVE_MANIFEST.json'
        if not manifest.is_file():continue
        require(folder.is_relative_to(args.source_repo) or folder.is_relative_to(args.original_repo),'legacy archive must be public source metadata')
        value=read(manifest)
        for record in value['files']:
            original=record['original'];rel=relative(original['path'])
            original_path=(args.source_repo/rel) if value.get('all_parts_paths_relative_to_repo') else folder/rel
            if safe(original_path)!=source:continue
            require(same(original,entry) and record.get('storage','ordered_parts' if record.get('parts') else 'direct_original')=='ordered_parts','exact parent compressed archive')
            width=record.get('part_bytes',record.get('part_size_bytes',value.get('part_bytes',value.get('part_size_bytes'))))
            require(width in (PART_BYTES,50*1024*1024),'explicit historical archive width')
            inventory.add(manifest);parts=[]
            for index,part in enumerate(record['parts']):
                path=relative(part['path']);require(path.endswith('.part%03d'%index) and 0<part['bytes']<=width,'ordered historical parts')
                physical=args.source_repo/path if value.get('all_parts_paths_relative_to_repo') else folder/path
                target=inventory.location(physical)
                # archive_run's unchanged legacy dispatch uses source_repo.
                item=inventory.add(args.source_repo/target,part);parts.append(dict(path=item['target'],bytes=part['bytes'],sha256=part['sha256']))
            require(parts and sum(p['bytes'] for p in parts)==entry['bytes'],'ordered historical byte total')
            return dict(original=dict(path=entry['target'],bytes=entry['bytes'],sha256=entry['sha256']),parts=parts,part_bytes=width,source=entry['source'],legacy=True)
    raise ValueError('large public dependency needs authoritative ordered parts: '+str(source))

def prior_private_archive(args,inventory,entry):
    """Old V19 private bytes already have frozen prepared parts; never reslice."""
    name=owned_target(OLD_FAMILY,'ARCHIVE_MANIFEST.json');path=safe(args.destination/name)
    value=read(path);require(value['schema']=='observed_frame_compressed_original_archive_manifest_v19'
        and value['encoding']==ENCODING and value['all_parts_paths_relative_to_repo'] is True,'exact published V19 archive authority')
    inventory.add(path,target=name)
    matches=[item for item in value['files'] if item['original']['path']==entry['target']]
    require(len(matches)==1,'one exact published old compressed original')
    record=matches[0];require(record==inventory.prepared_archives[entry['target']]
        and same(record['original'],entry) and record['storage']=='ordered_parts','old original identity and actual prepared receipt join')
    width=record['part_bytes'];require(width in (PART_BYTES,50*1024*1024),'old declared compressed part width')
    parts=[]
    for index,part in enumerate(record['parts']):
        target=public_path(part['path']);require(target.endswith('.part%03d'%index) and 0<part['bytes']<=width,'ordered parent parts')
        inventory.add(args.destination/target,part,target=target);parts.append(part)
    require(parts and sum(p['bytes'] for p in parts)==entry['bytes'],'old compressed byte total')
    return dict(original=dict(path=entry['target'],bytes=entry['bytes'],sha256=entry['sha256']),parts=parts,
        part_bytes=width,source=entry['source'],legacy=True,legacy_parts_source_repo=str(args.destination))

def collect(args):
    proof=source_proof(args);i=Inventory(args);i.add(__file__);i.add(args.semantics)
    history_target=owned_target(OLD_FAMILY,'PUBLIC_RELOCATION_MANIFEST.json')
    i.add(args.history_manifest,i.prepared[history_target],target=history_target)
    for name in ('PUBLICATION_PROTOCOL.json','PUBLICATION_CHECKS.json'):
        i.add(args.history_publication/name,target=owned_target(OLD_FAMILY,'publication/'+name))
    explicitly_owned_late_targets=set()
    for item in args.document:
        family,name,path=selector(item,i);require(Path(name).name==name and Path(name).suffix=='.md','document basename.md')
        target=owned_target(family,name);i.add(path,target=target);explicitly_owned_late_targets.add(target)
    for item in args.file:
        family,name,path=selector(item,i);name=relative(name)
        require(Path(name).name not in ('ARCHIVE_MANIFEST.json','PUBLIC_RELOCATION_MANIFEST.json'),'immutable base manifest must not be replaced')
        target=owned_target(family,name,code=path.suffix=='.py' and len(Path(name).parts)==1)
        i.add(path,target=target);explicitly_owned_late_targets.add(target)
    for family,root in i.packages.items():
        for path in sorted(root.iterdir()):
            require(not path.is_symlink(),'symlink in package root')
            if path.is_file() and path.suffix in ('.py','.json','.md'):i.add(path)
    for item in args.evidence:
        family,name,path=selector(item,i);i.folder(family,path)
    for item in args.reader_protocol:
        family,name,path=selector(item,i);i.reader(family,name,path)
    old=committed(args.destination);archives={}
    for target,entry in list(sorted(i.files.items())):
        if entry['bytes']<=PART_BYTES or target in old and entry['bytes']<=50*1024*1024:continue
        require(Path(entry['source']).suffix=='.gz','large selected artifact must be original gzip')
        if entry['source'] in i.old_records and target.startswith('_docs/experiments/'+OLD_FAMILY+'/'):
            archives[target]=prior_private_archive(args,i,entry)
        elif Path(entry['source']).is_relative_to(args.source_repo) or Path(entry['source']).is_relative_to(args.original_repo):
            archives[target]=legacy_archive(args,i,entry)
        else:
            family=next(f for f in i.families if target.startswith('_docs/experiments/'+f+'/'));doc='_docs/experiments/'+family+'/'
            width=PART_BYTES;parts=[dict(path=doc+'archives/'+relative(target[len(doc):])+'.part%03d'%n,bytes=min(width,entry['bytes']-n*width)) for n in range((entry['bytes']+width-1)//width)]
            archives[target]=dict(original=dict(path=target,bytes=entry['bytes'],sha256=entry['sha256']),parts=parts,part_bytes=width,source=entry['source'],legacy=False)
    files=[parent_check(args,e,old) for t,e in sorted(i.files.items()) if t not in archives]
    for entry in files:
        owned=any(entry['target'].startswith(prefix+family+'/') for family in i.packages for prefix in ('scripts/research/','_docs/experiments/'))
        prepared=i.prepared.get(entry['target'])
        if prepared is not None:require(same(entry,prepared) and same(digest(args.destination/entry['target']),prepared),'prepared V19 stage byte differs '+entry['target'])
        require(entry['existing_committed'] or owned or entry['target'] in explicitly_owned_late_targets or prepared is not None,'unowned dependency lacks exact parent or prepared V19 authority: '+entry['target'])
    for item in archives.values():
        physical=safe(args.destination/item['original']['path'])
        if physical.exists():require(same(digest(physical),item['original']),'existing stage compressed original differs')
        if item['legacy']:
            for part in item['parts']:parent_check(args,dict(source=str(Path(item.get('legacy_parts_source_repo',args.source_repo))/part['path']),target=part['path'],**{k:part[k] for k in ('bytes','sha256')}),old)
        else:require(all(not safe(args.destination/p['path']).exists() and p['path'] not in old for p in item['parts']),'new archive part prefix must be absent')
    record_locations={}
    for record in i.records.values():
        key=record['relative_path'];value=(record['bytes'],record['sha256'])
        require(key not in record_locations or record_locations[key]==value,'merged relocation location conflict');record_locations[key]=value
    touched=set(i.packages)|{OLD_FAMILY};generated=[]
    for family in sorted(touched):
        doc='_docs/experiments/'+family+'/'
        generated.extend([doc+'PUBLIC_RELOCATION_MANIFEST_NEXT.json',doc+('ARCHIVE_MANIFEST.json' if family in i.packages else 'ARCHIVE_MANIFEST_NEXT.json')])
        if family in i.packages:generated.append(doc+'.gitignore')
    generated+=['_docs/experiments/'+args.receipt_family+'/publication_next/'+n for n in FILES[:3]]
    require(all(not safe(args.destination/p).exists() and p not in old and p not in i.files for p in generated),'new generated targets must be absent')
    return dict(schema='frozen_explicit_private_publication_plan_v21',source_proof=proof,
        packages={f:str(p) for f,p in sorted(i.packages.items())},files=files,archives=list(archives.values()),
        relocation_records=[i.records[k] for k in sorted(i.records)],base_relocation_manifest=binding(args.history_manifest),
        original_artifact_roots={**{f:str(p) for f,p in i.packages.items()},OLD_FAMILY:i.history['original_artifact_root']},
        reader_protocols=i.readers,external_declarations=i.externals,generated_targets=generated,
        prepared_V19_publication_authority=i.history_authority,
        live_reader_closure_scope='only explicitly selected own protocol.inputs; no nested historical JSON traversal',limits=LIMITS)

def configuration(args):return {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='stage'}
def protected(args):return {str(p):git_state(p) for p in dict.fromkeys((args.source_repo,args.original_repo))}

def guard(args):
    for name in ('destination','source_repo','original_repo','receipts','semantics','history_manifest','history_publication'):setattr(args,name,safe(getattr(args,name)))
    require(all(p.is_dir() for p in (args.destination,args.source_repo,args.original_repo)),'existing repository roots')
    family_name(args.receipt_family);packages=parse_named(args.package)
    require(args.receipt_family in packages and args.receipts.is_relative_to(packages[args.receipt_family]) and args.receipts!=packages[args.receipt_family],'owned publication receipt subfolder')
    roots=[args.source_repo,args.original_repo,*packages.values()]
    require(all(not args.destination.is_relative_to(p) and not p.is_relative_to(args.destination) for p in roots),'destination disjoint from every source')
    require(all(not p.is_relative_to(repo) and not repo.is_relative_to(p) for p in packages.values() for repo in (args.source_repo,args.original_repo)),'private packages disjoint from protected repositories')
    require(all(not args.receipts.is_relative_to(p) and not p.is_relative_to(args.receipts) for p in (args.source_repo,args.original_repo)),'receipts outside protected repositories')
    parent=git(args.destination,'rev-parse',args.parent+'^{commit}').decode().strip()
    require(re.fullmatch('[0-9a-f]{40}',args.parent) is not None and parent==args.parent and git(args.destination,'rev-parse','HEAD').decode().strip()==parent,'exact current stage parent')
    # The existing V19 preparation is uncommitted. Its exact canonical state is
    # frozen below; this collector preserves all tracked/staged/index content.
    for name in FILES if args.stage=='freeze' else FILES[1:]:require(not(args.receipts/name).exists(),'preserve own publication prefix '+name)
    return parent

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'))
    for name in ('destination','source-repo','original-repo','parent','receipts','receipt-family'):p.add_argument('--'+name,required=True,type=str if name in ('parent','receipt-family') else Path)
    p.add_argument('--semantics',type=Path,default=Path('/tmp/pallet-finite-support-orthogonal-private-20261011-v19/publication.py'))
    p.add_argument('--history-manifest',type=Path,required=True)
    p.add_argument('--history-publication',type=Path,required=True)
    for name in ('package','evidence','document','file','reader-protocol'):p.add_argument('--'+name,action='append',default=[])
    a=p.parse_args();parent=guard(a);before=protected(a);destination=git_state(a.destination);plan=collect(a)
    require(protected(a)==before and git_state(a.destination)==destination,'read-only plan changed Git state')
    a.receipts.mkdir(parents=True,exist_ok=True);proto=a.receipts/FILES[0]
    if a.stage=='freeze':
        write_new(proto,dict(schema='explicit_private_publication_protocol_v21',configuration=configuration(a),parent=parent,
            helper=binding(__file__),protected=before,destination_git=destination,plan=plan,frozen_before_copy_or_split=True,limits=LIMITS));return
    own=read(proto);require(own['schema']=='explicit_private_publication_protocol_v21' and own['configuration']==configuration(a)
        and own['parent']==parent and own['helper']==binding(__file__) and own['protected']==before
        and own['destination_git']==destination and own['plan']==plan,'exact frozen publication source/plan/Git state')
    protocol=binding(proto);write_new(a.receipts/FILES[1],dict(protocol=protocol,helper=own['helper'],selected_packages=plan['packages'],limits=LIMITS))
    actions=[];archives=[];generated=[];error=None
    try:
        for entry in plan['files']:
            kind=link_verified(entry['source'],a.destination/entry['target'],entry);actions.append(dict(entry,storage=kind))
        for item in plan['archives']:
            dispatch=argparse.Namespace(**vars(a));dispatch.source_repo=Path(item.get('legacy_parts_source_repo',a.source_repo))
            archives.append(archive_run(dispatch,item))
        records=list(plan['relocation_records']);receipt_doc='_docs/experiments/'+a.receipt_family+'/publication_next/'
        for name in FILES[:2]:
            source=a.receipts/name;actual=binding(source);target=receipt_doc+name
            link_verified(source,a.destination/target,actual);generated.append(dict(path=target,**digest(source)))
            records.append(dict(original_path=str(source),root='repo',relative_path=target,bytes=actual['bytes'],sha256=actual['sha256']))
        for family,root in sorted(plan['original_artifact_roots'].items()):
            doc='_docs/experiments/'+family+'/'
            relocation=dict(schema='observed_frame_public_relocation_manifest_v19',original_artifact_root=root,records=records,
                nested_historical_declarations_are_not_live_dependencies=True,base_manifest_preserved=plan['base_relocation_manifest'])
            selected=[item for item in archives if item['original']['path'].startswith(doc)]
            manifest=dict(schema='observed_frame_compressed_original_archive_manifest_v19',encoding=ENCODING,part_bytes=PART_BYTES,
                original_files_preserved=True,original_file_count=len(selected),files=selected,all_parts_paths_relative_to_repo=True,gzip_decompression_or_recompression_calls=0)
            archive_name='ARCHIVE_MANIFEST.json' if family in plan['packages'] else 'ARCHIVE_MANIFEST_NEXT.json'
            for name,value in (('PUBLIC_RELOCATION_MANIFEST_NEXT.json',relocation),(archive_name,manifest)):
                target=doc+name;write_new(a.destination/target,value);generated.append(dict(path=target,**digest(a.destination/target)))
            if family in plan['packages']:
                ignored=['/'+item['original']['path'][len(doc):] for item in selected]
                target=doc+'.gitignore';write_new(a.destination/target,('\n'.join(ignored)+'\n').encode(),raw=True)
                generated.append(dict(path=target,**digest(a.destination/target)))
        require(git(a.destination,'rev-parse','HEAD').decode().strip()==parent,'destination HEAD changed')
        after=git_state(a.destination);require(after['index_entries_sha256']==destination['index_entries_sha256'] and after['staged_diff_sha256']==destination['staged_diff_sha256'],'destination logical index/staged contents changed')
        require(after['tracked_diff_sha256']==destination['tracked_diff_sha256'],'existing tracked destination bytes changed')
        require(protected(a)==before,'source/original canonical Git state changed')
        for entry in plan['files']:require(same(digest(entry['source']),entry),'source changed during copy')
        for item in plan['archives']:require(same(digest(item['source']),item['original']),'original compressed source changed')
        publish={entry['target'] for entry in plan['files'] if not entry['existing_committed']}
        publish.update(part['path'] for item in archives for part in item['parts'] if part['path'] not in committed(a.destination))
        publish.update(item['path'] for item in generated);publish.add(receipt_doc+FILES[2])
        write_new(a.receipts/FILES[3],b''.join((path+'\0').encode() for path in sorted(publish)),raw=True)
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    checks=dict(schema='explicit_private_publication_checks_v21',complete=error is None,passed=error is None,error=error,
        protocol=protocol,helper=binding(__file__),parent=parent,selected_packages=plan['packages'],actual_direct_files=actions,
        actual_ordered_archives=archives,generated_files=generated,protected_after=protected(a),original_and_source_Git_state_unchanged=protected(a)==before,
        limits=LIMITS,partial_destination_prefix_preserved_on_failure=True,publication_commit_or_push_performed=False,
        own_CHECKS_not_self_hashed_or_declared_as_input=True,portable_copied_reader_execution_pending=True)
    write_new(a.receipts/FILES[2],checks)
    if error is None:link_verified(a.receipts/FILES[2],a.destination/receipt_doc/FILES[2],binding(a.receipts/FILES[2]))
    print(json.dumps(dict(complete=checks['complete'],passed=checks['passed'],error=error)),flush=True)
    if error:raise SystemExit(1)

if __name__=='__main__':main()
