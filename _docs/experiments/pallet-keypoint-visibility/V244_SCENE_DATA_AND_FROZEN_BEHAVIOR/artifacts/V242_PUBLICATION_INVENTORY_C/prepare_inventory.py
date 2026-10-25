"""명시된 준비/QA 폴더의 publication 후보만 닫음. 복사/Git/배열/영상 read 없음."""
import csv,hashlib,json,os,stat
from pathlib import Path

H=Path(__file__).absolute().parent;R=H.parent
DIRECTORIES=('V242_ROOT_SCENE_CF','V242_SCENE_CF_DRIVER_B','V242_SCENE_MATH_REVIEW_A',
 'V242_SCENE_DRIVER_CODE_REVIEW_A','V242_SCENE_DATA_QA_C','V242_SCENE_DATA_QA_C/actual_review_v1',
 'V242_SCENE_GALLERY_A','V242_CF_FEATURE_PRODUCER_B','V242_CF_FEATURE_CODE_REVIEW_C',
 'V242_CF_FEATURE_PACKET_QA_C','V242_QUERY_REFERENCE_DIAGNOSTIC_C',
 'V243_SCENE_CF_TRAINER_A','V243_TRAINER_CODE_REVIEW_B')
ROOT_CODES=('assemble_v242_actual_qa_entry.py','assemble_v242_gallery_entry.py')
SMALL_NUMERIC={'PARENT_INDICES.npy','PAIR_TARGETS.npy','CONSTRUCTED.npy','PROOF_PASS.npy','GROUP_GEOMETRY.npz','PAIR_ALPHA_PACKBITS.npy'}
TEXT={'.py','.json','.jsonl','.md','.txt','.log','.csv','.html','.svg'}
CAP=2*1024*1024

def need(ok,message):
    if not ok:raise ValueError(message)

def lexical(p):
    p=Path(p);need(p.is_absolute() and '..' not in p.parts and not any(t in str(p).casefold() for t in ('v112','__pycache__','.pyc')),'lexical rejection before FS');return p

def parents(p):
    p=lexical(p)
    for q in reversed(p.parents):need(stat.S_ISDIR(os.lstat(q).st_mode),'parent directory/symlink guard')
    return p

def read(p):
    p=parents(p);need(stat.S_ISREG(os.lstat(p).st_mode),'regular metadata/code leaf')
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def bound(p,b=None):
    b=read(p) if b is None else b
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b))

def write(name,value):
    p=H/name
    with os.fdopen(os.open(parents(p),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return bound(p)

def bindings(o):
    if isinstance(o,dict):
        if {'path','sha256','size_bytes'}<=set(o):yield {k:o[k] for k in ('path','sha256','size_bytes')}
        for v in o.values():yield from bindings(v)
    elif isinstance(o,list):
        for v in o:yield from bindings(v)

def run():
    candidates={};exclusions=[];snapshots=[];docs={};large_text=[];png_names=[]
    def add(b,reason,mode='small_CODE_or_metadata_opaque_SHA_verified',provenance=None):
        p=lexical(b['path']);need(p.is_relative_to(R),'candidate stays within declared R')
        item={**b,'R_relative_path':str(p.relative_to(R)),'proposed_package_relative_path':'artifacts/'+str(p.relative_to(R)),
              'category':reason,'verification_this_inventory':mode}
        if provenance:item['saved_binding_provenance']=provenance
        if str(p) in candidates:need(candidates[str(p)]['sha256']==b['sha256'] and candidates[str(p)]['size_bytes']==b['size_bytes'],'same leaf binding consistency')
        else:candidates[str(p)]=item
    for name in DIRECTORIES:
        directory=parents(R/name);need(stat.S_ISDIR(os.lstat(directory).st_mode),'explicit directory')
        names=os.listdir(directory);allowed=[n for n in names if not any(t in n.casefold() for t in ('v112','__pycache__','.pyc'))]
        snapshots.append({'directory':str(directory),'names':sorted(allowed),'lexically_rejected_names_not_statted':len(names)-len(allowed)})
        for n in sorted(allowed):
            p=directory/n
            if p.suffix=='.png':png_names.append(str(p));continue # image bytes/stat forbidden in this inventory
            if p.suffix not in TEXT:continue # no stat of numeric caches or nested dirs
            p=parents(p);st=os.lstat(p);need(stat.S_ISREG(st.st_mode),'text leaf regular')
            if st.st_size>CAP:large_text.append(str(p));continue
            raw=read(p);add(bound(p,raw),'code_preparation_receipt_failure_or_diagnostic')
            if p.suffix=='.json':docs[str(p)]=json.loads(raw)
    for name in ROOT_CODES:
        p=R/name;raw=read(p);need(len(raw)<CAP,'small Root helper code');add(bound(p,raw),'Root_admin_entry_assembly_CODE')
    dcpath=R/'V242_ROOT_SCENE_CF/scene_pairs_v1/COMPLETE.json';dcraw=read(dcpath);dc=json.loads(dcraw)
    need(hashlib.sha256(dcraw).hexdigest()=='12f973806eccdbeae25afea8f6e23fb43da738f2d412d7dbdbb82ea8d087d8bc' and dc['completed'] and dc['PASS'],'terminal data COMPLETE pin')
    add(bound(dcpath,dcraw),'actual_dataset_COMPLETE');docs[str(dcpath)]=dc
    for name,b in dc['outputs'].items():
        if name in SMALL_NUMERIC:add({k:b[k] for k in ('path','sha256','size_bytes')},'small_numeric_output_candidate_PENDING_Root_copy_admission','saved_COMPLETE_binding_only_NO_leaf_stat_hash_header_or_body',str(dcpath))
        elif name=='PAIR_METADATA.jsonl':add({k:b[k] for k in ('path','sha256','size_bytes')},'all5406_PAIR_METADATA_compress_candidate','saved_COMPLETE_binding_only_NO_leaf_stat_or_body',str(dcpath))
        else:exclusions.append({**{k:b[k] for k in ('path','sha256','size_bytes')},'reason':'large native-scene/model-input or validity/layer cache; retain COMPLETE binding, omit payload','verification_this_inventory':'saved_COMPLETE_binding_only_NO_leaf_stat_hash_header_or_body'})
    gp=R/'V242_SCENE_GALLERY_A/gallery_v2/COMPLETE.json';gb=read(gp);gc=json.loads(gb)
    need(hashlib.sha256(gb).hexdigest()=='747f4bb5f58cf2b1d17d1cecbc6ef6e74d46f397c3bf5f7a1c34fd68f8b8512a' and gc['completed'] and gc['PASS'],'actual gallery COMPLETE pin')
    add(bound(gp,gb),'actual_fixed4_gallery_COMPLETE');docs[str(gp)]=gc
    for b in gc['outputs']:
        add(b,'actual_fixed4_gallery_native_display_or_metadata','saved_gallery_COMPLETE_binding_only_NO_image_or_leaf_body_read',str(gp))
    # Big diagnostic JSON is already pinned by the separate saved diagnostic COMPLETE.
    for p in large_text:
        found=[(origin,b) for origin,d in docs.items() for b in bindings(d) if b['path']==p]
        need(bool(found),'large metadata needs saved binding, not unapproved body hash')
        origin,b=found[0];add(b,'saved_independent_numeric_QA_row_observations','saved_receipt_binding_only_NO_new_body_read',origin)
    # Preserve declared toy PNG/failure illustrations only through their existing receipt bindings.
    allbind=[(origin,b) for origin,d in docs.items() for b in bindings(d)]
    unmatched_png=[]
    for p in png_names:
        found=[(origin,b) for origin,b in allbind if b['path']==p]
        if found:
            origin,b=found[0];add(b,'invented_toy_or_failure_illustration','saved_toy_receipt_binding_only_NO_image_stat_or_body_read',origin)
        else:unmatched_png.append({'path':p,'reason':'invented top-level PNG has no complete path/SHA/size binding found in read small receipts; inert until Root explicit opaque inclusion'})
    dataqa=docs[str(R/'V242_SCENE_DATA_QA_C/actual_review_v1/REVIEW.json')]
    a=docs[str(R/'V243_SCENE_CF_TRAINER_A/FINAL_READY_V2.json')]
    need(dataqa['completed'] and dataqa['PASS'] and a['actual_executed'] is False and a['actual_training_updates']==0,'actual data QA versus unexecuted trainer status')
    count=dc['counts'];need(count['attempted']==5406 and count['proof_pass']==5367 and count['unconstructed']==39,'all attempted failures retained')
    rows=sorted(candidates.values(),key=lambda b:b['R_relative_path'])
    selected=[x for x in rows if any(t in Path(x['path']).name.upper() for t in ('FAIL','FIRST','PRESERV','BEFORE','REVIEW_V1'))]
    status={'base_main':'8c2ae5c47a831339f626e5d502ef9092a9a6e474','V242_data_actual_completed':True,'all_original_parents':5406,'proof_PASS':5367,'constructed_UNKNOWN':0,'unconstructed_retained':39,'V242_independent_all5406_numeric_QA_PASS':True,'fixed4_gallery_actual_completed':True,'gallery_cases':gc['cases'],'V242_new_CF_feature_extraction':'CODE_READY_UNEXECUTED_at_inventory_instruction','V242_CF_feature_numeric_QA':'CODE_READY_pending_separate_actual_entry','V243_training':'CODE_READY_UNEXECUTED','new_CF_features_or_head_fit_executed':0,'new_current_or_held_payload':0,'new_PnP':0,'current_admission':False,'model_adopted':False,'goal':'active_unsolved','V241_previous39publishedfiles':'retain unchanged; no clone/Git access or parent-source mutation in C inventory','V244':'not in this snapshot; Root will add separate diagnostics later'}
    manifest={'completed':True,'PASS':True,'inventory_only':True,'not_publication_execution_authority':True,'status':status,'candidate_count':len(rows),'candidate_raw_bytes':sum(x['size_bytes'] for x in rows),'candidates':rows,'preserved_failure_fix_and_prior_version_artifacts':selected,'unresolved_inert_toy_PNG_candidates':unmatched_png,'no_numeric_array_RGB_or_model_leaf_reads':True,'no_copy_Git_network_or_source_mutation':True}
    m=write('CANDIDATE_FILES.json',manifest)
    x=write('EXCLUSIONS.json',{'completed':True,'PASS':True,'excluded_numeric_cache_count':len(exclusions),'excluded_numeric_cache_bytes':sum(b['size_bytes'] for b in exclusions),'bindings':exclusions,'additional_policies':['No external source image/donor/rawGT/old weights pointer follow','No feature/model payload exists in this CODE-only snapshot; no inferred future output inclusion','Small6 numeric data leaves above remain candidates only until Root explicit copy admission','Prior V241 files/history remain unchanged','Rejected v112/__pycache__/.pyc entries are never statted/read/copied'], 'huge_cache_FS_access':0})
    s=write('DIRECTORY_SNAPSHOT.json',{'directories':snapshots,'recursion':False,'numeric_array_or_image_leaf_stat':False})
    note='''# V242/V243 게시 후보와 검산 순서

이 목록은 게시 실행 권한이 아닌 CODE/metadata inventory입니다. 원본이나 clone을 수정하지 않았습니다.

V242 데이터 5,406개는 전부 시도됐고 5,367개 proof PASS, 39개 미생성으로 보존됐습니다. 독립 all5406 pixel QA와 fixed4 gallery는 실제 완료됐습니다. gallery의 full/native·crop·card PNG는 COMPLETE에 선언된 원 byte binding으로만 후보에 넣었으며 이 inventory에서 영상·배열을 읽지 않았습니다. q/r diagnostic은 합성 footprint 관측이지 실사 literal-q GT 또는 detector error 측정이 아닙니다.

CF feature producer·packet QA와 V243 trainer는 CODE 준비 상태입니다. 실제 CF 특징 생성/새 학습/성능 개선/현재셋 검증으로 과장하지 않습니다. current admission과 model adoption은 false입니다. SOURCE 데이터의 prospective feature/trial admission과 실사 모델 admission을 구분합니다.

Root builder는 CANDIDATE_FILES.json의 exact leaf만 사용하고 source-map에 원 SHA/size를 보존합니다. PAIR_METADATA JSONL을 gzip으로 실으면 압축 byte SHA/size와 해제 원문 SHA/27,613,654B를 함께 기록합니다. small numeric 6개는 명시 Root copy admission 뒤에만 포함합니다. 나머지 큰 5개 cache는 EXCLUSIONS의 원 binding만 남기고 파일 자체 stat/read/hash/copy를 하지 않습니다. 미래 feature/model caches도 자동 포함하지 않습니다.

게시 검산은 manifest exact leaf SHA/size, source copy byte equality, 압축본 decoded SHA/size, README/HTML 내부 상대링크 closure 및 제3자고지를 확인합니다. 외부 literal bindings는 문서로만 보존하며 따라가지 않습니다. V1 실패/준비 수리/가상 harness 실패 및 원 code를 모두 함께 보존합니다. 기존 V241 39파일과 prior history prefix는 변경하지 않습니다. Root의 V244 결과 추가 뒤 최종 manifest를 별도로 닫아야 합니다.

Git/복사/네트워크·모델/데이터/architecture 선택은 이 작업에서 하지 않았습니다. main base는 Root가 준 8c2ae5c47이며 C는 clone을 열지 않았습니다.
'''
    p=H/'QA_PLAN_KO.md'
    with os.fdopen(os.open(parents(p),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:f.write(note)
    complete=write('COMPLETE.json',{'completed':True,'PASS':True,'inventory_only':True,'code':bound(Path(__file__)),'candidate_manifest':m,'exclusions':x,'directory_snapshot':s,'QA_plan':bound(p),'status':status,'candidate_count':len(rows),'candidate_raw_bytes':sum(x['size_bytes'] for x in rows),'failure_fix_prior_version_candidates':len(selected),'unresolved_toy_png_count':len(unmatched_png),'actual_publication_or_new_numeric_payload_reads':0,'frozen_after_complete':True})
    print(json.dumps({'COMPLETE':complete,'candidate_count':len(rows),'candidate_raw_bytes':manifest['candidate_raw_bytes'],'failure_fix_prior_version_candidates':len(selected),'unresolved_toy_PNGs':unmatched_png,'excluded_numeric_cache_count':len(exclusions),'excluded_numeric_cache_bytes':sum(b['size_bytes'] for b in exclusions)},ensure_ascii=False),flush=True)

if __name__=='__main__':run()
