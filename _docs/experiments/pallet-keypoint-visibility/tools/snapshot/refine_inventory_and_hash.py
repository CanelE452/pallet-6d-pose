"""Publication v2: no-follow metadata inventory and authorized candidate SHA census.
Never copies source, accesses Git, decodes arrays, or descends a V112-named entry.
"""
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse, hashlib, json, os, stat, time, zlib

HERE=Path(__file__).absolute().parent
P=HERE.parent
ROOTS={'P':P,'V60':P.parent/'external_ray_proxy_head_20261022_v60','V61':P.parent/'target_presence_classifier_20261022_v61','V62':P.parent/'presence_previous_models_audit_20261022_v62'}
TEXT=set('.py .sh .bash .ps1 .js .mjs .cjs .ts .tsx .jsx .css .scss .toml .yaml .yml .cfg .ini .sql .ipynb .md .html .htm .rst .txt .tex .pdf .json .jsonl .ndjson .csv .tsv .xml .log .patch .diff .in .dockerfile .graphql .frag .vert .cu .c .cpp .h .hpp .lock .body'.split())
VISUAL=set('.png .jpg .jpeg .svg .webp .gif'.split())
WEIGHTS=set('.pt .pth .ckpt .onnx .safetensors .pkl .pickle .joblib .h5 .hdf5 .pb .tflite .bin'.split())
ENV={'.git','.venv','venv','env','envs','node_modules','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','vendor','site-packages','sam2_repo'}
RAW_SEGMENTS={'rgb','depth','mask','masks','mask_visib','canonical_rgb','train_real','train_pbr','test_real','rgb_fixed','edited_fullrgb','alpha_gt_only','after_visible_masks_gt_only','pasted_masks_gt_only','provided_metadata'}
RAW_PREFIXES=[('query_visibility_controlled_pairs_20261023_v77','pair_generation','images'),('query_visibility_controlled_real_20261022_v66','controlled','images'),('query_visibility_continuous_native_source_20261024_v136','actual_corpus_v1','bases'),('query_visibility_registered_native_diversity_20261023_v129','actual_corpus_v1','bases'),('query_visibility_public_pallet_loco_20261023_v73','data','intake','annotations')]
NPZ_LIMIT=8*1024**2;OWN_LIMIT=16*1024**2;CHUNK=32*1024**2

def forbidden(name):return 'v112' in name.casefold()
def secret(name):return name.casefold() in {'.env','.env.local','.env.production','credentials.json','credentials.toml','secrets.json','secrets.toml','id_rsa','id_ed25519'} or name.casefold().endswith(('.pem','.key'))
def write_json(path,obj):
    with path.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
def write_rows(path,rows):
    with path.open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
def classify(rel,size):
    p=Path(rel);parts=tuple(s.casefold() for s in p.parts);name=p.name.casefold();ext=p.suffix.casefold()
    raw=any(s in RAW_SEGMENTS for s in parts[:-1]) or any(parts[:len(x)]==x for x in RAW_PREFIXES) or ('full_source_v1' in parts and 'original' in parts)
    if raw:return 'EXCLUDE','raw_dataset_training_asset','original bank/training RGB/GT mask/annotation; direct report image refs may override individually'
    if ext in {'.whl','.so','.nbc','.nbi','.db3','.mcap','.bag'}:return 'EXCLUDE','environment_or_raw_recording','dependency binary/JIT/raw sensor recording'
    if ext in WEIGHTS:
        if size<OWN_LIMIT and ext in {'.pt','.pth','.ckpt'} and not any(s in name for s in ['sam_vit','sam2','dinov2','yolo','backbone','resnet']):return 'SMALL_OWN_HEAD_REVIEW','small_trained_state_candidate','<16MiB possible project head/optimizer state; explicit root whitelist required'
        return 'EXCLUDE','foundation_weight_or_opaque_state','pretrained/large weights or opaque serialized payload'
    if ext in TEXT or name in {'license','makefile','dockerfile','notice','copying','requirements'}:return 'INCLUDE','research_text_code_table','preserve byteexact regardless of size; >100MB lossless gzip32MiB chunks'
    if ext in VISUAL:return 'INCLUDE','research_visual','review/card/contact/plot/display artifact outside raw banks'
    if ext=='.npz':
        deny=('feature','input','cache','rgb','crop','pixel','payload','sam_output','scene_','group_','first_surface','raw_diagnostic','sam96')
        if any(s in name for s in deny) or size>NPZ_LIMIT:return 'EXCLUDE','array_input_feature_spatial_cache','raw/spatial/input/feature arrays or >8MiB; no header/body decoding'
        if any(s in name for s in ('output','predict','logit','score','result','confusion','metric','state','parity','check')):return 'SAVED_OUTPUT_REVIEW','saved_small_numerical_output_candidate','<=8MiB saved classifier/numeric result; explicit whitelist/schema inspection required'
        return 'NUMERIC_REVIEW','small_numeric_npz_candidate','<=8MiB possible numeric targets/checks; explicit root whitelist required'
    if ext in {'.npy','.np','.dat','.memmap','.mp4','.avi','.mov','.ply','.obj','.stl','.mtl','.glb','.gltf','.zip','.tar','.gz','.bz2','.xz','.7z','.rar','.tgz','.zst'}:return 'EXCLUDE','raw_array_geometry_archive_or_media','large input/cache/dataset/geometry/archive; retain metadata pointers'
    return 'REVIEW','unclassified','no automatic publication'

def scan():
    rows=[];pruned=[];errors=[]
    for root_id,root in ROOTS.items():
        # Roots are explicit authorized paths, not discovered or symlink-resolved.
        st=os.stat(root,follow_symlinks=False)
        if not stat.S_ISDIR(st.st_mode):raise RuntimeError('authorized root is not a directory')
        stack=[root]
        while stack:
            d=stack.pop()
            try:entries=sorted(os.scandir(d),key=lambda e:e.name)
            except OSError as e:errors.append({'root_id':root_id,'relative':str(d.relative_to(root)),'error':type(e).__name__});continue
            for ent in entries:
                rel=str(Path(ent.path).relative_to(root));name=ent.name
                if forbidden(name):pruned.append({'root_id':root_id,'relative':rel,'reason':'V112_NAME_PRUNED_BEFORE_STAT_OR_DESCENT','bytes':None});continue
                if secret(name):pruned.append({'root_id':root_id,'relative':rel,'reason':'SECRET_NAME_PRUNED_BEFORE_STAT_OR_READ','bytes':None});continue
                if name.startswith('publication_preparation_') or name=='publication_root_preparation':pruned.append({'root_id':root_id,'relative':rel,'reason':'PUBLICATION_ADMIN_NOT_RECURSED','bytes':None});continue
                st=ent.stat(follow_symlinks=False)
                if stat.S_ISLNK(st.st_mode):pruned.append({'root_id':root_id,'relative':rel,'reason':'SYMLINK_NOT_READ_RESOLVED_OR_FOLLOWED','link_metadata_size':st.st_size,'target_size':None});continue
                if stat.S_ISDIR(st.st_mode):
                    if name.casefold() in ENV or name.startswith('.'):pruned.append({'root_id':root_id,'relative':rel,'reason':'ENV_THIRDPARTY_OR_HIDDEN_TREE_NOT_RECURSED','bytes':None});continue
                    stack.append(Path(ent.path));continue
                if not stat.S_ISREG(st.st_mode):pruned.append({'root_id':root_id,'relative':rel,'reason':'NONREGULAR_FILE_NOT_READ','bytes':None});continue
                action,category,reason=classify(rel,st.st_size)
                archive='archive/task-2/'+root.name+'/'+rel
                rows.append(dict(source_root_id=root_id,source_absolute=str(root/rel),source_relative_path=rel,source_sha256=None,size_bytes=st.st_size,published_relative=archive,published_sha256=None,status='CANDIDATE_'+action,action=action,category=category,exclusion_reason=reason if action=='EXCLUDE' else None,reason=reason,gzipchunks=None,suffix=Path(rel).suffix.casefold(),mtime_ns=st.st_mtime_ns,device=st.st_dev,inode=st.st_ino))
    return sorted(rows,key=lambda r:(r['source_root_id'],r['source_relative_path'])),pruned,errors

def safe_open(r):
    root=ROOTS[r['source_root_id']];parts=Path(r['source_relative_path']).parts
    if any(forbidden(s) or secret(s) or s in {'.','..'} for s in parts):raise RuntimeError('forbidden path component')
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        final=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    st=os.fstat(final)
    if not stat.S_ISREG(st.st_mode) or (st.st_size,st.st_mtime_ns,st.st_dev,st.st_ino)!=(r['size_bytes'],r['mtime_ns'],r['device'],r['inode']):os.close(final);raise RuntimeError('source metadata drift before read')
    return os.fdopen(final,'rb')

def hash_one(r):
    h=hashlib.sha256()
    with safe_open(r) as f:
        while block:=f.read(4*1024**2):h.update(block)
        s=os.fstat(f.fileno())
        if (s.st_size,s.st_mtime_ns)!=(r['size_bytes'],r['mtime_ns']):raise RuntimeError('source drift during hash')
    return h.hexdigest()

def compress_stats(r):
    parts=[]
    with safe_open(r) as f:
        index=0
        while data:=f.read(CHUNK):
            # Independent gzip members are deterministic; no data is written here.
            comp=zlib.compressobj(6,zlib.DEFLATED,31);packed=comp.compress(data)+comp.flush()
            parts.append({'index':index,'decoded_size_bytes':len(data),'stored_size_bytes':len(packed),'sha256':hashlib.sha256(packed).hexdigest(),'compression':'gzip'});index+=1
    return {'source_relative_path':r['source_relative_path'],'source_root_id':r['source_root_id'],'source_sha256':r['source_sha256'],'size_bytes':r['size_bytes'],'compressed_size_bytes':sum(p['stored_size_bytes'] for p in parts),'chunk_decoded_max_bytes':CHUNK,'parts':parts,'statistics_only_no_payload_written':True}

def summarize(rows):
    actions=defaultdict(lambda:{'files':0,'bytes':0});cats=defaultdict(lambda:{'files':0,'bytes':0});exts=defaultdict(lambda:{'files':0,'bytes':0});prefixes=defaultdict(lambda:{'files':0,'bytes':0});byhash=defaultdict(list)
    for r in rows:
        for d,k in ((actions,r['action']),(cats,r['category'])):d[k]['files']+=1;d[k]['bytes']+=r['size_bytes']
        if r['source_sha256']:
            exts[r['suffix']]['files']+=1;exts[r['suffix']]['bytes']+=r['size_bytes'];prefix=r['source_root_id']+'/'+ '/'.join(Path(r['source_relative_path']).parts[:3]);prefixes[prefix]['files']+=1;prefixes[prefix]['bytes']+=r['size_bytes'];byhash[r['source_sha256']].append(r)
    unique=sum(v[0]['size_bytes'] for v in byhash.values());logical=sum(r['size_bytes'] for r in rows if r['source_sha256'])
    return {'actions':dict(actions),'categories':dict(cats),'hashed_extensions':dict(exts),'hashed_top30_prefixes':sorted([dict(prefix=k,**v) for k,v in prefixes.items()],key=lambda x:-x['bytes'])[:30],'hashed_logical_bytes':logical,'hashed_unique_content_bytes':unique,'duplicate_bytes_savings':logical-unique,'unique_sha256_count':len(byhash),'hashed_files':sum(len(v) for v in byhash.values())},byhash

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='v2_refined');ap.add_argument('--hash',action='store_true');args=ap.parse_args();out=HERE/args.out;out.mkdir(exist_ok=False);start=time.monotonic()
    rows,pruned,errors=scan()
    overrides_path=P/'publication_preparation_a/DIRECT_REPORT_IMAGE_REFERENCES.jsonl'
    overrides={r['source_relative_path']:r for r in map(json.loads,overrides_path.read_text().splitlines())}
    for r in rows:
        if r['source_root_id']=='P' and r['source_relative_path'] in overrides:
            if r['suffix'] not in VISUAL:raise RuntimeError('report image override is not an image path')
            r.update(action='INCLUDE',status='CANDIDATE_INCLUDE',category='direct_report_image_override',reason='individual direct HTML/MD image reference; not entire raw pool',exclusion_reason=None,referenced_by=overrides[r['source_relative_path']]['referenced_by'])
    write_rows(out/'METADATA_INVENTORY_BEFORE_HASH.jsonl',rows);write_rows(out/'DIRECTORY_PRUNE_RECORDS.jsonl',pruned)
    if args.hash:
        candidates=[r for r in rows if r['action'] in {'INCLUDE','SAVED_OUTPUT_REVIEW','NUMERIC_REVIEW','SMALL_OWN_HEAD_REVIEW'}]
        with ThreadPoolExecutor(max_workers=4) as pool:
            fs={pool.submit(hash_one,r):r for r in candidates}
            for i,fut in enumerate(as_completed(fs),1):
                r=fs[fut]
                try:r['source_sha256']=fut.result()
                except Exception as e:errors.append({'source':r['source_absolute'],'error':type(e).__name__+': '+str(e)});r['status']='HASH_FAILED'
                if i%5000==0:print(json.dumps({'hashed':i,'of':len(candidates)}),flush=True)
    summary,byhash=summarize(rows)
    large=[r for r in rows if r['action']=='INCLUDE' and r['suffix'] in TEXT and r['size_bytes']>100_000_000]
    compression=[compress_stats(r) for r in large if r['source_sha256']]
    write_rows(out/'PUBLICATION_FILE_MANIFEST_CANDIDATE.jsonl',rows)
    write_rows(out/'DUPLICATE_CONTENT_GROUPS.jsonl',[{'source_sha256':k,'size_bytes':v[0]['size_bytes'],'paths':[r['published_relative'] for r in v]} for k,v in sorted(byhash.items()) if len(v)>1])
    for action,name in [('SMALL_OWN_HEAD_REVIEW','SMALL_OWN_HEAD_CANDIDATES.jsonl'),('SAVED_OUTPUT_REVIEW','SAVED_NUMERICAL_OUTPUT_CANDIDATES.jsonl'),('NUMERIC_REVIEW','OTHER_SMALL_NUMERIC_CANDIDATES.jsonl')]:write_rows(out/name,[r for r in rows if r['action']==action])
    write_json(out/'LARGE_TEXT_LOSSLESS_COMPRESSION_STATS.json',compression)
    summary.update(completed=not errors,source_roots={k:str(v) for k,v in ROOTS.items()},files_enumerated=len(rows),pruned_reason_counts=dict(Counter(r['reason'] for r in pruned)),pruned_tree_sizes='UNKNOWN deliberately unmeasured',source_files_modified=0,source_array_headers_or_data_decoded=0,source_or_repository_copy=0,repository_access=False,hashing_authorized=args.hash,actual_publish=False,errors=errors,elapsed_seconds=time.monotonic()-start,archive_layout='archive/task-2/<original root basename>/<relative path>',uncompressed_text_over100MB=len(large),compression_statistics=compression)
    write_json(out/'REFINED_INVENTORY_AND_DEDUP_SUMMARY.json',summary)
    print(json.dumps({k:summary[k] for k in ['completed','files_enumerated','actions','hashed_logical_bytes','hashed_unique_content_bytes','duplicate_bytes_savings','hashed_files','uncompressed_text_over100MB','elapsed_seconds']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
