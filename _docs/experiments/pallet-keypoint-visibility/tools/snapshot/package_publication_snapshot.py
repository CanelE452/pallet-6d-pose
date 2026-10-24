"""Root-operated portable snapshot copier. Preparation does not copy any source.
Original bytes or losslessly decompressed chunks validate against original SHA.
Never follows symlinks; V112 names are refused before metadata/read operations.
"""
from pathlib import Path
import argparse,hashlib,json,os,stat,tempfile,zlib
import refine_inventory_and_hash as safe

FORMAT='pallet-publication-plan-v1'
CHUNK=32*1024**2

def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(4*1024**2):h.update(b)
    return h.hexdigest()

def reject_path(relative):
    p=Path(relative)
    if p.is_absolute() or not p.parts or any(s in {'.','..'} or safe.forbidden(s) or safe.secret(s) for s in p.parts):raise ValueError('unsafe or forbidden publication relative path')
    return p

def safe_destination(root,relative):
    p=reject_path(relative);current=root
    for part in p.parts[:-1]:
        current=current/part
        try:st=current.lstat()
        except FileNotFoundError:current.mkdir();st=current.lstat()
        if not stat.S_ISDIR(st.st_mode):raise ValueError('destination parent is not an ordinary directory')
    destination=root/p
    try:destination.lstat()
    except FileNotFoundError:return destination
    raise FileExistsError(str(destination))

def gzip_bytes(data):
    c=zlib.compressobj(6,zlib.DEFLATED,31)
    return c.compress(data)+c.flush()

def copy_raw(row,dest_root):
    destination=safe_destination(dest_root,row['published_relative']);h=hashlib.sha256();n=0
    with safe.safe_open(row) as src,destination.open('xb') as dst:
        while block:=src.read(4*1024**2):dst.write(block);h.update(block);n+=len(block)
    if n!=row['size_bytes'] or h.hexdigest()!=row['source_sha256']:raise ValueError('source-copy hash/size drift')
    return {'status':'PUBLISHED_RAW','published_sha256':h.hexdigest(),'stored_size_bytes':n,'gzipchunks':None}

def copy_chunks(row,dest_root):
    h=hashlib.sha256();n=0;parts=[]
    with safe.safe_open(row) as src:
        while block:=src.read(CHUNK):
            h.update(block);n+=len(block);packed=gzip_bytes(block)
            relative=row['published_relative']+'.parts/part%05d.gz'%len(parts)
            dst=safe_destination(dest_root,relative)
            with dst.open('xb') as f:f.write(packed)
            parts.append({'path':relative,'sha256':hashlib.sha256(packed).hexdigest(),'compression':'gzip','stored_size_bytes':len(packed),'decoded_size_bytes':len(block)})
    if n!=row['size_bytes'] or h.hexdigest()!=row['source_sha256']:raise ValueError('gzip source hash/size drift')
    descriptor={'format':'concatenated-bytes-v1','parts':parts,'content_sha256':h.hexdigest(),'content_size_bytes':n}
    descriptor_path=row['published_relative']+'.parts.json'
    descriptor_file=safe_destination(dest_root,descriptor_path)
    with descriptor_file.open('x') as f:json.dump(descriptor,f,ensure_ascii=False,indent=2);f.write('\n')
    return {'status':'PUBLISHED_GZIP_CHUNKS','published_sha256':None,'stored_size_bytes':sum(p['stored_size_bytes'] for p in parts),'gzipchunks':descriptor,'gzip_descriptor_path':descriptor_path,'gzip_descriptor_sha256':sha_file(descriptor_file),'gzip_descriptor_size_bytes':descriptor_file.stat().st_size}

def validate_plan(plan):
    if plan.get('format')!=FORMAT or plan.get('approved_source_copy') is not True:raise ValueError('root-approved copy plan required')
    if plan.get('git_mutations_or_push') is not False:raise ValueError('packager never handles git')
    if plan.get('archive_layout')!='archive/task-2/<original root basename>/<relative path>':raise ValueError('layout drift')
    return plan

def run(plan_path):
    plan=validate_plan(json.loads(plan_path.read_text()));manifest=Path(plan['candidate_manifest']['path'])
    if sha_file(manifest)!=plan['candidate_manifest']['sha256']:raise ValueError('candidate manifest drift')
    declared_roots=plan.get('source_roots',{k:str(v) for k,v in safe.ROOTS.items()})
    for value in declared_roots.values():
        pp=Path(value)
        if not pp.is_absolute() or any(safe.forbidden(s) or safe.secret(s) or s=='..' for s in pp.parts):raise ValueError('unsafe declared source root')
    safe.ROOTS={k:Path(v) for k,v in declared_roots.items()}
    root=Path(plan['target_root'])
    if not root.is_absolute():raise ValueError('target root must be absolute')
    # No directory resolve/readlink. Refuse all existing symlink components.
    current=Path(root.anchor)
    for part in root.parts[1:]:
        if safe.forbidden(part):raise ValueError('forbidden destination')
        current=current/part
        try:st=current.lstat()
        except FileNotFoundError:current.mkdir();st=current.lstat()
        if not stat.S_ISDIR(st.st_mode):raise ValueError('non-directory destination component')
    if any(root==v or root in v.parents or v in root.parents for v in safe.ROOTS.values()):raise ValueError('destination must be separate from source roots')
    all_rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    whitelist=set(plan.get('explicit_numerical_and_own_state_whitelist',[]))
    allowed={'INCLUDE'};threshold=int(plan['gzip_text_threshold_bytes']);result=[];content_copies={};index=0
    manifest_relative=plan.get('output_manifest_relative','PUBLICATION_MANIFEST.jsonl')
    complete_relative=plan.get('output_complete_relative','PUBLICATION_COPY_COMPLETE.json')
    finalmanifest=safe_destination(root,manifest_relative)
    with finalmanifest.open('x') as f:
        for row in all_rows:
            if any(safe.forbidden(part) for part in Path(row['source_relative_path']).parts):raise ValueError('forbidden row, no metadata access')
            record={k:v for k,v in row.items() if k not in {'mtime_ns','device','inode','action','status','array_headers_only'}}
            key=row['source_root_id']+':'+row['source_relative_path']
            include=row['action'] in allowed or (key in whitelist and row['action'] in {'SAVED_OUTPUT_REVIEW','NUMERIC_REVIEW','SMALL_OWN_HEAD_REVIEW'})
            if not include:
                record.update(status='EXCLUDED',published_relative=None,published_sha256=None,gzipchunks=None,exclusion_reason=row.get('exclusion_reason') or 'not selected by root explicit publication whitelist')
            else:
                if not row.get('source_sha256'):raise ValueError('selected file lacks source SHA')
                ext=row['suffix'];compress=ext in safe.TEXT and row['size_bytes']>=threshold
                if not compress and row['size_bytes']>95*1024**2:raise ValueError('raw file exceeds safe Git size; root must select lossless compression')
                existing=root/reject_path(row['published_relative'])
                existing_descriptor=root/reject_path(row['published_relative']+'.parts.json')
                # New phase/delta plans may reuse previously archived byteidentical files.
                if plan.get('allow_existing_identical') and not compress:
                    try:st=existing.lstat()
                    except FileNotFoundError:st=None
                    if st is not None:
                        if not stat.S_ISREG(st.st_mode) or sha_file(existing)!=row['source_sha256']:raise ValueError('existing archive differs; preserve history with a new logical path')
                        record.update(status='PUBLISHED_RAW',published_sha256=row['source_sha256'],stored_size_bytes=st.st_size,gzipchunks=None,copy_mode='REUSED_EXISTING_IDENTICAL')
                    else:record.update(copy_raw(row,root))
                elif plan.get('allow_existing_identical') and compress:
                    try:st=existing_descriptor.lstat()
                    except FileNotFoundError:st=None
                    if st is not None:
                        if not stat.S_ISREG(st.st_mode):raise ValueError('nonregular existing gzip descriptor')
                        desc=json.loads(existing_descriptor.read_text())
                        if desc.get('format')!='concatenated-bytes-v1' or desc.get('content_sha256')!=row['source_sha256'] or desc.get('content_size_bytes')!=row['size_bytes']:raise ValueError('existing logical gzip content differs')
                        for part in desc['parts']:
                            pp=root/reject_path(part['path']);ss=pp.lstat()
                            if not stat.S_ISREG(ss.st_mode) or sha_file(pp)!=part['sha256']:raise ValueError('existing gzip part differs')
                        record.update(status='PUBLISHED_GZIP_CHUNKS',published_sha256=None,stored_size_bytes=sum(p['stored_size_bytes'] for p in desc['parts']),gzipchunks=desc,gzip_descriptor_path=row['published_relative']+'.parts.json',gzip_descriptor_sha256=sha_file(existing_descriptor),gzip_descriptor_size_bytes=st.st_size,copy_mode='REUSED_EXISTING_IDENTICAL')
                    else:record.update(copy_chunks(row,root))
                else:record.update(copy_chunks(row,root) if compress else copy_raw(row,root))
                record['exclusion_reason']=None
                index+=1
                if index%1000==0:print(json.dumps({'published_files':index}),flush=True)
            f.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();result.append(record)
    summary={'completed':True,'PASS':True,'source_originals_modified':0,'git_operations':0,'source_manifest':plan['candidate_manifest'],'plan':{'path':str(plan_path),'sha256':sha_file(plan_path)},'manifest':{'path':manifest_relative,'sha256':sha_file(finalmanifest)},'published_logical_files':sum(r['status'].startswith('PUBLISHED_') for r in result),'published_logical_bytes':sum(r['size_bytes'] for r in result if r['status'].startswith('PUBLISHED_')),'published_stored_bytes':sum(r.get('stored_size_bytes',0) for r in result),'excluded_files':sum(r['status']=='EXCLUDED' for r in result),'V112_access':'name refusal before source metadata/read; no forbidden source records copied','symlink_targets_followed':0,'portable_derivatives':'separate root/A step; original bytes never rewritten'}
    with safe_destination(root,complete_relative).open('x') as f:json.dump(summary,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(summary),flush=True)

def prepare():
    checks=[]
    original=b'original research records\n'*100
    packed=gzip_bytes(original)
    checks.append({'name':'gzip_roundtrip_original_SHA','PASS':zlib.decompress(packed,31)==original})
    checks.append({'name':'gzip_deterministic_byte_same','PASS':packed==gzip_bytes(original)})
    for path in ('../escape','archive/task-2/v112_secret/file.json','/absolute','archive/.env'):
        try:reject_path(path);passed=False
        except ValueError:passed=True
        checks.append({'name':'reject_'+path,'PASS':passed})
    try:validate_plan({'format':FORMAT,'approved_source_copy':False});passed=False
    except ValueError:passed=True
    checks.append({'name':'unapproved_copy_rejected','PASS':passed})
    with tempfile.TemporaryDirectory(prefix='invented_only_',dir=safe.HERE) as t:
        root=Path(t);(root/'a').mkdir();(root/'a/file').write_bytes(b'x')
        try:safe_destination(root,'a/file');passed=False
        except FileExistsError:passed=True
        checks.append({'name':'no_overwrite','PASS':passed})
        (root/'link').symlink_to(root/'a',target_is_directory=True)
        try:safe_destination(root,'link/new');passed=False
        except ValueError:passed=True
        checks.append({'name':'destination_symlink_not_followed','PASS':passed})
    out=safe.HERE/'PACKAGER_INVENTED_CHECKS.json';safe.write_json(out,{'completed':True,'PASS':all(c['PASS'] for c in checks),'checks':checks,'source_inputs_read':0,'repository_access':False})
    template={'format':FORMAT,'approved_source_copy':False,'git_mutations_or_push':False,'target_root':'/home/minjae/Documents/pallet-visibility-publication-20261024/_docs/experiments/pallet-keypoint-visibility','source_roots':{k:str(v) for k,v in safe.ROOTS.items()},'archive_layout':'archive/task-2/<original root basename>/<relative path>','candidate_manifest':{'path':str(safe.HERE/'v5_final_candidates/PUBLICATION_FILE_MANIFEST_CANDIDATE.jsonl'),'sha256':None},'gzip_text_threshold_bytes':1024**2,'explicit_numerical_and_own_state_whitelist':[],'allow_existing_identical':False,'output_manifest_relative':'PUBLICATION_MANIFEST.jsonl','output_complete_relative':'PUBLICATION_COPY_COMPLETE.json'}
    safe.write_json(safe.HERE/'ROOT_COPY_PLAN_TEMPLATE_APPROVED_FALSE.json',template)
    print(json.dumps({'completed':True,'PASS':all(c['PASS'] for c in checks),'invented_checks':len(checks),'source_copy':0,'repository_access':False}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--approved-plan',type=Path);args=ap.parse_args()
    if args.prepare:prepare()
    elif args.approved_plan:run(args.approved_plan)
    else:ap.error('--prepare or --approved-plan required')
