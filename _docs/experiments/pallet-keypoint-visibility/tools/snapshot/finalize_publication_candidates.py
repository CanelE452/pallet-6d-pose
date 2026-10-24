"""Refine v2 by path+NPY headers only; quantify lossless text compression.
Source data/array payloads and pickle weights are never deserialized.
"""
from pathlib import Path
from collections import defaultdict,Counter
import ast,hashlib,json,struct,time,zipfile,zlib
import refine_inventory_and_hash as base

OUT=base.HERE/'v5_final_candidates'
RAW_NPZ_DIR_TOKENS=('feature','cache','p3','input_pack','source_crops','rgb96_chunk','native_scenes','donor_cutout','native_mosaics','sam_source','frozen_real268_sam','frozen_sam_source','sam_predictions','geometry_inventory')

def array_headers(row):
    headers=[]
    with base.safe_open(row) as source:
        with zipfile.ZipFile(source) as z:
            for member in z.infolist():
                if not member.filename.endswith('.npy'):raise ValueError('non-NPY zip member')
                with z.open(member) as f:
                    if f.read(6)!=b'\x93NUMPY':raise ValueError('bad npy magic')
                    version=f.read(2);n=struct.unpack('<H' if version[0]==1 else '<I',f.read(2 if version[0]==1 else 4))[0]
                    if n>65536:raise ValueError('oversize npy header')
                    obj=ast.literal_eval(f.read(n).decode('latin1' if version[0]<3 else 'utf8'))
                shape=obj['shape'];dtype=obj['descr'];key=member.filename[:-4]
                if not isinstance(shape,tuple) or not isinstance(dtype,str):raise ValueError('unsupported structured dtype/header')
                if 'O' in dtype:raise ValueError('object pickle dtype')
                kl=key.casefold()
                # Metadata flags such as feature_constructed are not feature arrays.
                if kl in {'image','rgb','rgb96','rgb96u8','full_rgb','features','feature','embedding','embeddings','p3','weights','texture','depth_map','alpha','mask96','full_mask','hidden','hidden64','hidden_features'}:raise ValueError('raw feature/image header key')
                result_array=any(s in kl for s in ('logit','probabil','prediction','score','attention','response'))
                if (len(shape)>=3 or (len(shape)==2 and min(shape)>64)) and not result_array:raise ValueError('spatial/dense-feature shape without explicit saved prediction/logit meaning')
                headers.append({'key':key,'shape':shape,'dtype':dtype,'uncompressed_bytes':member.file_size})
    return headers

def compression_stats(row):
    parts=[]
    with base.safe_open(row) as f:
        index=0
        while data:=f.read(base.CHUNK):
            comp=zlib.compressobj(6,zlib.DEFLATED,31);packed=comp.compress(data)+comp.flush()
            parts.append({'index':index,'decoded_size_bytes':len(data),'stored_size_bytes':len(packed),'sha256':hashlib.sha256(packed).hexdigest(),'compression':'gzip'});index+=1
    return {'source_sha256':row['source_sha256'],'size_bytes':row['size_bytes'],'compressed_size_bytes':sum(p['stored_size_bytes'] for p in parts),'parts':parts}

def main():
    OUT.mkdir(exist_ok=False);start=time.monotonic()
    rows=[json.loads(s) for s in (base.HERE/'v2_refined/PUBLICATION_FILE_MANIFEST_CANDIDATE.jsonl').read_text().splitlines()]
    headers=[]
    for row in rows:
        if row['suffix']!='.npz':continue
        parts=[x.casefold() for x in Path(row['source_relative_path']).parts[:-1]];name=Path(row['source_relative_path']).name.casefold()
        # Experiment names can contain "frozen_features" while their runs hold outputs.
        # Only descendant role directories classify raw inputs, never the experiment name.
        rawdir=any(any(token in part for token in RAW_NPZ_DIR_TOKENS) for part in parts[1:])
        explicit_result=any(s in name for s in ('output','predictions','logits','scores','confusion','readout')) and not any(s in name for s in ('feature','sam_output','scene_','group_'))
        if rawdir:
            row.update(action='EXCLUDE',status='CANDIDATE_EXCLUDE',category='raw_input_feature_spatial_cache',exclusion_reason='NPZ raw cache/feature/P3/input/donor/spatial directory; no array payload read');continue
        if row['action']=='EXCLUDE' and not (explicit_result and row['size_bytes']<100*1024**2):continue
        if row['action'] not in {'NUMERIC_REVIEW','SAVED_OUTPUT_REVIEW','EXCLUDE'}:continue
        try:
            hh=array_headers(row)
            # Output-tagged numeric tables only; GT/sampler scalar arrays are a separate opt-in list.
            action='SAVED_OUTPUT_REVIEW' if explicit_result or row['action']=='SAVED_OUTPUT_REVIEW' else 'NUMERIC_REVIEW'
            row.update(action=action,status='CANDIDATE_'+action,category='header_verified_saved_numerical_output' if action=='SAVED_OUTPUT_REVIEW' else 'header_verified_small_numeric_record',exclusion_reason=None,array_headers_only=True)
            if not row['source_sha256']:row['source_sha256']=base.hash_one(row)
            headers.append({'source_root_id':row['source_root_id'],'source_relative_path':row['source_relative_path'],'source_sha256':row['source_sha256'],'headers':hh,'data_decoded':False})
        except Exception as e:
            row.update(action='EXCLUDE',status='CANDIDATE_EXCLUDE',category='spatial_feature_object_or_unverified_npz',exclusion_reason=type(e).__name__+': '+str(e))
    # Compression is an optional size proposal. No compressed source copy is written.
    seen=set();stats=[]
    previous=base.HERE/'v3_final_candidates/LOSSLESS_TEXT_COMPRESSION_STATS.jsonl'
    oldstats={s['source_sha256']:s for s in map(json.loads,previous.read_text().splitlines())} if previous.exists() else {}
    for row in rows:
        if row['action']=='INCLUDE' and row['suffix'] in base.TEXT and row['size_bytes']>=1024**2 and row['source_sha256'] not in seen:
            seen.add(row['source_sha256']);s=oldstats.get(row['source_sha256']) or compression_stats(row);s.update(source_root_id=row['source_root_id'],source_relative_path=row['source_relative_path']);stats.append(s)
    selected=[r for r in rows if r['action'] in {'INCLUDE','SAVED_OUTPUT_REVIEW','SMALL_OWN_HEAD_REVIEW','NUMERIC_REVIEW'}]
    byhash=defaultdict(list)
    for r in selected:byhash[r['source_sha256']].append(r)
    categories=defaultdict(lambda:{'files':0,'logical_bytes':0,'unique_bytes':0});seen=set()
    for r in selected:
        c=categories[r['action']];c['files']+=1;c['logical_bytes']+=r['size_bytes']
        if r['source_sha256'] not in seen:c['unique_bytes']+=r['size_bytes'];seen.add(r['source_sha256'])
    unique_raw=sum(v[0]['size_bytes'] for v in byhash.values());cs={s['source_sha256']:s for s in stats};compressed_proposal=sum(cs[k]['compressed_size_bytes'] if k in cs else v[0]['size_bytes'] for k,v in byhash.items())
    base.write_rows(OUT/'PUBLICATION_FILE_MANIFEST_CANDIDATE.jsonl',rows)
    base.write_rows(OUT/'NPZ_HEADER_SCHEMA_RECORDS.jsonl',headers)
    base.write_rows(OUT/'LOSSLESS_TEXT_COMPRESSION_STATS.jsonl',stats)
    for action,name in [('SMALL_OWN_HEAD_REVIEW','SMALL_OWN_HEAD_EXPLICIT_CANDIDATES.jsonl'),('SAVED_OUTPUT_REVIEW','SAVED_NUMERICAL_OUTPUT_EXPLICIT_CANDIDATES.jsonl'),('NUMERIC_REVIEW','OTHER_SMALL_NUMERIC_EXPLICIT_CANDIDATES.jsonl')]:base.write_rows(OUT/name,[r for r in rows if r['action']==action])
    base.write_rows(OUT/'DUPLICATE_CONTENT_CROSSWALK.jsonl',[{'source_sha256':k,'size_bytes':v[0]['size_bytes'],'published_paths':[r['published_relative'] for r in v]} for k,v in sorted(byhash.items()) if len(v)>1])
    summary={'completed':True,'source_roots':{k:str(v) for k,v in base.ROOTS.items()},'candidate_categories':dict(categories),'all_candidates_logical_bytes':sum(r['size_bytes'] for r in selected),'all_candidates_unique_raw_bytes':unique_raw,'proposed_all_text_ge1MiB_lossless_gzip_unique_bytes':compressed_proposal,'compression_sources':len(stats),'compression_bytes_saved':unique_raw-compressed_proposal,'npz_headers_read':len(headers),'array_payloads_decoded':0,'weight_pickle_loads':0,'repo_or_source_copy':0,'source_modified':0,'actual_publish':False,'elapsed_seconds':time.monotonic()-start,'policy':'All code/tables/docs retained; all review visuals/direct-report overrides; explicit numerical output and <16MiB own-state whitelist candidates; rawbanks/features/cache excluded. Optional compression of all >=1MiB text proposed, not selected.','excluded_action_counts':dict(Counter(r['category'] for r in rows if r['action']=='EXCLUDE'))}
    base.write_json(OUT/'FINAL_CANDIDATE_SIZE_AND_COMPRESSION_SUMMARY.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
