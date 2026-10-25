"""Root가 승인한 selected TRAIN110 GT JSON의 asset 계보 문자열만 조사한다."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import stat

H=Path(__file__).absolute().parent
SCOPE=H.parent/'V234_ROOT_LABEL_DIAG/SOURCE_GT110_ANCESTRY_READ_SCOPE.json'
SCOPE_SHA='6eaab3f39071dc7fe96682af8164af0a33d3a673ba31f689fa8b974b03d1a1bf'
OUT=H/'actual_read_v1'
HINT=re.compile(r'background|(?:^|_)bg(?:_|$)|occlud|donor|render|asset|catalog|texture|provenance|lineage|source_(?:dataset|path|file|scene|frame|rgb)|full_rgb|original_rgb|raw_rgb|config',re.I)
IDENTITY=re.compile(r'^(?:base_index|scene_index|variant|anchor_.*|source_image_original_global_index|source_frame_id|source_instance_id|frame_id|frame_index)$',re.I)
YCB=re.compile(r'ycb|train_real',re.I)
SCENE_PATH=re.compile(r'(?:train_real|(?:ycb[^/\\]*))[/\\]+0*(6|7)(?:[/\\]|$)',re.I)


def safe(p):
    p=Path(p)
    if not p.is_absolute() or '..'in p.parts or 'v112'in str(p).casefold() or '__pycache__'in p.parts or p.suffix=='.pyc':raise ValueError('Forbidden before FS')
    for q in list(p.parents)[::-1]+[p]:
        try:mode=os.lstat(q).st_mode
        except FileNotFoundError:continue
        if stat.S_ISLNK(mode):raise ValueError('Symlink')
    return p


def raw(p):
    with os.fdopen(os.open(safe(p),os.O_RDONLY|os.O_NOFOLLOW),'rb')as f:return f.read()


def sha(b):return hashlib.sha256(b).hexdigest()


def bind(p):
    b=raw(p);return dict(path=str(p),sha256=sha(b),size_bytes=len(b))


def write(name,value):
    p=OUT/name
    with safe(p).open('x',encoding='utf-8')as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')


def csv_write(name,rows,fields):
    with safe(OUT/name).open('x',newline='',encoding='utf-8')as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    scope_raw=raw(SCOPE)
    if sha(scope_raw)!=SCOPE_SHA:raise ValueError('Scope SHA')
    scope=json.loads(scope_raw);whitelist=scope['whitelist']
    if len(whitelist)!=110 or len({b['path']for b in whitelist})!=110 or scope['new_pointer_access_requires_separate_scope']is not True:raise ValueError('Closed110 source scope')
    # Scope/protocolを先に固定し、pointer抽出結果をこのread関数へ渡さない。
    protocol=raw(H/'PROTOCOL_BEFORE_SELECTED110_GT_JSON_READS.json')
    if json.loads(protocol)['scope']['sha256']!=SCOPE_SHA:raise ValueError('Pre-read protocol')
    safe(OUT).mkdir(exist_ok=False)
    schema=Counter();schema_parents=defaultdict(set);parents=[];variants=[];fields=[];pointers=[];cross=[];reads=[]
    for binding in whitelist:
        p=Path(binding['path'])
        if p.name!='SCENES_GT_ONLY.json':raise ValueError('Only declared GT JSON')
        # Whitelist membership above precedes every source filesystem operation.
        b=raw(p)
        if sha(b)!=binding['sha256']:raise ValueError('Selected source SHA mismatch')
        doc=json.loads(b)
        base=int(re.fullmatch(r'V129_BASE_(\d+)',p.parent.name).group(1))
        reads.append(dict(path=str(p),sha256=sha(b),size_bytes=len(b),base_index=base))
        start=len(fields);cross_start=len(cross);pointer_start=len(pointers);presence=defaultdict(set)
        rows=doc.get('rows')if isinstance(doc,dict)else None
        def walk(value,where='',normalized='',row_index=None,variant=None,lineage_context=False):
            kind='null'if value is None else type(value).__name__
            schema[(normalized or '/',kind)]+=1;schema_parents[(normalized or '/',kind)].add(base)
            key=where.rsplit('/',1)[-1]
            if re.search(r'background|(?:^|_)bg(?:_|$)',key,re.I):presence['background'].add(where)
            if re.search(r'occlud|donor',key,re.I):presence['donor'].add(where)
            if re.search(r'render|asset|catalog|texture|config',key,re.I):presence['renderer'].add(where)
            relevant=lineage_context or bool(HINT.search(key))
            if isinstance(value,dict):
                explicit_ycb=any(isinstance(v,str)and YCB.search(v)and re.search(r'dataset|origin|source|donor|asset',str(k),re.I)for k,v in value.items())
                if relevant and explicit_ycb:
                    for k,v in value.items():
                        if re.fullmatch(r'(?:source_|donor_)?scene(?:_id)?',str(k),re.I)and isinstance(v,(str,int))and str(v).isdigit()and int(v)in(6,7):
                            cross.append(dict(base_index=base,row_index=row_index,variant=variant,JSON_pointer=where+'/'+str(k),scene_id=f'{int(v):06d}',evidence_kind='EXPLICIT_YCB_DATASET_AND_SCENE_SIBLINGS',literal_value=v))
                for k,v in value.items():walk(v,where+'/'+str(k),normalized+'/'+str(k),row_index,variant,relevant)
            elif isinstance(value,list):
                for i,v in enumerate(value):
                    ri=i if where=='/rows'else row_index
                    va=v.get('variant')if where=='/rows'and isinstance(v,dict)else variant
                    walk(v,where+'/'+str(i),normalized+'/*',ri,va,relevant)
            else:
                identity=bool(IDENTITY.fullmatch(key))
                if (relevant or identity)and isinstance(value,(str,int,float,bool)):
                    # Geometry arrays/numeric pose values are not exported; only identity scalars.
                    if isinstance(value,str)or identity or re.search(r'(?:(?:source_|donor_)?scene(?:_id)?|frame(?:_id)?|object_id|seed|donor_id|asset_id)$',key,re.I):
                        fields.append(dict(base_index=base,row_index=row_index,variant=variant,JSON_pointer=where,value=value,category='IDENTITY'if identity and not relevant else 'ANCESTRY_FIELD'))
                if isinstance(value,str):
                    pathlike=value.startswith(('/','./','../'))or bool(re.search(r'\.(?:py|json|jsonl|png|jpe?g|npz|npy|ply|obj|pt|pth|toml|ya?ml)(?:$|[?#])',value,re.I))
                    if pathlike:
                        pointers.append(dict(base_index=base,row_index=row_index,variant=variant,JSON_pointer=where,literal=value,inert_only=True,filesystem_followed=False))
                    if YCB.search(value):
                        fields.append(dict(base_index=base,row_index=row_index,variant=variant,JSON_pointer=where,value=value,category='EXPLICIT_YCB_STRING'))
                        for scene in SCENE_PATH.findall(value):cross.append(dict(base_index=base,row_index=row_index,variant=variant,JSON_pointer=where,scene_id=f'{int(scene):06d}',evidence_kind='LITERAL_YCB_OR_TRAIN_REAL_ASSET_PATH',literal_value=value))
        walk(doc)
        own=fields[start:];bg=[x for x in own if re.search(r'background|(?:/|_)bg(?:_|/|$)',x['JSON_pointer'],re.I)]
        donor=[x for x in own if re.search(r'occlud|donor',x['JSON_pointer'],re.I)]
        renderer=[x for x in own if re.search(r'render|asset|catalog|texture|config',x['JSON_pointer'],re.I)]
        ycb=[x for x in own if x['category']=='EXPLICIT_YCB_STRING']
        scenes=cross[cross_start:]
        parents.append(dict(base_index=base,GT_path=str(p),GT_sha256=sha(b),GT_bytes=len(b),scene_rows=len(rows)if isinstance(rows,list)else 0,
                            variants='|'.join(str(r.get('variant'))for r in rows)if isinstance(rows,list)else '',background_fields=len(presence['background']),occluder_donor_fields=len(presence['donor']),renderer_asset_fields=len(presence['renderer']),explicit_YCB_strings=len(ycb),literal_scene000006=sum(x['scene_id']=='000006'for x in scenes),literal_scene000007=sum(x['scene_id']=='000007'for x in scenes),inert_pointer_strings=len(pointers)-pointer_start,asset_ancestry_status='LITERAL_YCB_DEV_REFERENCE_PRESENT'if scenes else 'NO_LITERAL_YCB_DEV_REFERENCE_IN_THIS_JSON;_ANCESTRY_UNRESOLVED'))
        if isinstance(rows,list):
            for i,row in enumerate(rows):
                if not isinstance(row,dict):continue
                identity={k:v for k,v in row.items()if IDENTITY.fullmatch(k)and isinstance(v,(str,int,float,bool))}
                variants.append(dict(base_index=base,row_index=i,variant=row.get('variant'),scene_index=row.get('scene_index'),identity_fields=identity,document_ancestry_fields=[x for x in own if x['row_index']is None and x['category']!='IDENTITY'],own_row_ancestry_fields=[x for x in own if x['row_index']==i and x['category']!='IDENTITY']))
    csv_write('ALL110_PARENT_ANCESTRY.csv',parents,list(parents[0]))
    csv_write('ALL_SCHEMA_FIELDS.csv',[dict(JSON_pointer=p,JSON_type=t,occurrences=n,parents=len(schema_parents[(p,t)]))for(p,t),n in sorted(schema.items())],['JSON_pointer','JSON_type','occurrences','parents'])
    write('ANCESTRY_FIELDS_AND_IDENTITIES.json',fields)
    write('ALL_VARIANT_IDENTITY_AND_ANCESTRY.json',variants)
    write('INERT_POINTER_STRINGS.json',pointers)
    write('EXPLICIT_YCB_DEV_ASSET_REFERENCES.json',cross)
    summary=dict(completed=True,PASS=True,selected_TRAIN_GT_JSON_files=110,source_bytes=sum(r['size_bytes']for r in reads),source_SHA_verified_before_parse=True,variant_rows=len(variants),schema_field_type_pairs=len(schema),parents_with_background_field=sum(r['background_fields']>0 for r in parents),parents_with_occluder_donor_field=sum(r['occluder_donor_fields']>0 for r in parents),parents_with_renderer_asset_field=sum(r['renderer_asset_fields']>0 for r in parents),parents_with_explicit_YCB_string=sum(r['explicit_YCB_strings']>0 for r in parents),parents_with_literal_YCB_scene000006=sum(r['literal_scene000006']>0 for r in parents),parents_with_literal_YCB_scene000007=sum(r['literal_scene000007']>0 for r in parents),explicit_YCB_DEV_asset_reference_count=len(cross),inert_pointer_strings=len(pointers),cross_family_overlap_status='LITERAL_REFERENCES_RECORDED;_NOT_PIXEL_OR_FULL_ANCESTRY_CERTIFICATION'if cross else'UNKNOWN;_NO_LITERAL_YCB_DEV_REFERENCE_FOUND_IN_THE_SELECTED110_JSONS',absence_of_field_is_not_proof_of_asset_absence=True,new_pointer_FS=0,RGB_depth_mesh_NPZ_weight_model_or_protected_reads=0,geometry_recomputed=False,model_data_split_or_threshold_selection=False)
    write('SUMMARY.json',summary)
    names=['ALL110_PARENT_ANCESTRY.csv','ALL_SCHEMA_FIELDS.csv','ANCESTRY_FIELDS_AND_IDENTITIES.json','ALL_VARIANT_IDENTITY_AND_ANCESTRY.json','INERT_POINTER_STRINGS.json','EXPLICIT_YCB_DEV_ASSET_REFERENCES.json','SUMMARY.json']
    write('COMPLETE.json',dict(completed=True,PASS=True,scope=dict(path=str(SCOPE),sha256=SCOPE_SHA),protocol=dict(path=str(H/'PROTOCOL_BEFORE_SELECTED110_GT_JSON_READS.json'),sha256=sha(protocol)),code=bind(Path(__file__).absolute()),source_reads=reads,outputs=[bind(OUT/n)for n in names],new_pointer_FS=0,no_geometry_model_data_selection=True))
    print(json.dumps(summary));print(json.dumps(bind(OUT/'COMPLETE.json')))


if __name__=='__main__':main()
