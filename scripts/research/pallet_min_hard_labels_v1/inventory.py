"""Inventory only human/capture metadata; never open predictions or pseudo labels."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from scripts.research.pallet_single_model_preserve_v1 import common as P

NAME='pallet_min_hard_labels_v1';DOC=P.ROOT/'_docs/experiments'/NAME;RAW=P.ROOT/'data/pallet/results'/NAME;OUT=P.ROOT/'outputs'/NAME
def save(p,x):
    assert any(p.resolve().is_relative_to(r) for r in (DOC,RAW,OUT));p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(x if isinstance(x,str) else json.dumps(P.clean(x),ensure_ascii=False,indent=2)+'\n')
def key(s):return hashlib.sha256(('min-hard-v1:'+s).encode()).hexdigest()
def main():
    assert P.read(P.DOC/'SUPERVISION_GAP_DIAGNOSTIC.json')['decision']=='MIN_HARD_LABELING_JUSTIFIED';assert not DOC.exists()
    groups=P.read(P.ROOT/'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json');aliases={s['session_key']:g['recording_id'] for g in groups['groups'] if not g['is_collection'] for s in g['sessions']}
    split=P.read(P.ROOT/'_docs/experiments/pallet_existing_data_transfer_v1/SPLIT_LOCK.json');targets=P.read(P.ROOT/'data/pallet/results/pallet_existing_data_transfer_v1/TARGETS.json');exclude=[r for r in split['heldout']]+[r for r in targets if r['role'] in ('C0','H')]
    # Also exclude every anchor identity, not merely scored HARD36 coordinates.
    final=P.read(P.ROOT/'data/pallet/results/pallet_verified_anchor_v1/metadata_conflict_qa/VERIFIED_LABELS_FINAL_PRIVATE.json');anchor_sha={f['image_sha256'] for f in final['frames']};excluded_sha={r['image']['sha256'] for r in exclude}|anchor_sha
    heldrec=set(split['heldout_recordings']);oldsplit=P.read(P.ROOT/'data/pallet/eval_results/split_lock/split_assignment.json');reserved_sessions=set()
    for pop,section in oldsplit.items():
        if isinstance(section,dict):reserved_sessions.update(section.get('final_test',{}).get('sessions',[]))
    reserved_rec={rec for session,rec in aliases.items() if Path(session).name in reserved_sessions}
    # Final-labelled sessions outside the current released DEV/training split remain sealed.
    current_recs=set(split['heldout_recordings'])|set(split['train_recordings'])
    reserved_rec|={g['recording_id'] for g in groups['groups'] if not g['is_collection'] and g['recording_id'] not in current_recs and any('/final/' in s['session_key'] for s in g['sessions'])}
    tags={};bindings=[]
    # Existing fully locked human 278-frame direct severity answers.
    mp=P.ROOT/'data/pallet/results/pallet_eval_occlusion_severity_v1/PRIVATE_FRAME_MAPPING.json';ans=P.ROOT/'data/pallet/results/pallet_eval_occlusion_severity_v1/direct_review/LOCKED_REVIEW_RESPONSES.json'
    manifest=P.read(P.ROOT/'data/pallet/results/pallet_eval_occlusion_severity_v1/direct_review/OCCLUSION_SEVERITY_MANIFEST.json')
    # The combined319 split already contains the verified matching human labels.
    for r in split['train']+split['heldout']:tags[r['image']['sha256']]=dict(severity=r['severity'],source='HUMAN_COMBINED319',id=r['id'])
    # Different namespace/schema for the original278; direct response map is bound.
    responses=P.read(ans)
    response_map=responses.get('responses',{})
    for r in P.read(mp)['rows']:
        s=response_map.get(r['review_id']);s=s.get('severity') if isinstance(s,dict) else s
        if s in ('CLEAN','MODERATE_OCCLUSION','SEVERE_OCCLUSION'):tags.setdefault(r['image']['sha256'],dict(severity=s,source='HUMAN_DIRECT278',id=r['frame_id']))
    bindings.extend([P.bind(mp),P.bind(ans),P.bind(P.ROOT/'_docs/experiments/pallet_existing_data_transfer_v1/SPLIT_LOCK.json'),P.bind(P.ROOT/'data/pallet/eval_results/split_lock/split_assignment.json')])
    records={};source_count={};csvpath=P.ROOT/'data/evaluation/pallet_eval_v1/manifests/frames.csv';csvrows=list(csv.DictReader(csvpath.open()));bindings.append(P.bind(csvpath))
    unknown_conditions=Counter()
    for r in csvrows:
        if r['is_positive'].lower()!='true' or r['object_type']!='plastic':continue
        image=P.ROOT/'data/evaluation/pallet_eval_v1'/r['image_path'];sha=r['image_sha256'];session=str(image.parent.parent.relative_to(P.ROOT));rec=aliases.get(session)
        tag=tags.get(sha);occ=r['occlusion'].lower();trunc=r['truncation'].lower();unknown_conditions[(occ,trunc)]+=1
        if tag is None:
            sev='SEVERE_OCCLUSION' if occ in ('heavy','severe') or trunc in ('major','severe') else 'MODERATE_OCCLUSION' if occ in ('partial','moderate','mild') else None
            tag=dict(severity=sev,source='EXISTING_CAPTURE_CONDITION_TAG' if sev else 'NO_HARD_METADATA',id=r['frame_id'])
        records.setdefault(sha,dict(id=tag['id'],image=dict(path=str(image.relative_to(P.ROOT)),sha256=sha),recording=rec,session=session,**{k:tag[k] for k in ('severity','source')}))
    source_count['positive_plastic_frame_manifest']=len(records)
    poolpath=P.ROOT/'_docs/experiments/pallet_type_selftrain_v1/POOL.json';pool=P.read(poolpath);bindings.append(P.bind(poolpath));rawplastic=0
    for r in pool['records']:
        if r['kind']!='PLASTIC':continue
        rawplastic+=1;sha=r['image']['sha256'];tag=tags.get(sha,dict(severity=None,source='NO_HARD_METADATA',id=r['id']))
        records.setdefault(sha,dict(id=tag['id'],image=r['image'],recording=r['recording_id'],session=r['session'],severity=tag['severity'],source=tag['source']))
    source_count['unlabeled_plastic_pool']=rawplastic
    # Do not inspect RGB to manufacture a model-independent hard proxy when only day/night metadata exists.
    rejected=Counter();eligible=[];relaxed=[]
    for sha,r in records.items():
        if sha in excluded_sha:rejected['EVAL_ANCHOR_CLEAN10_H10_SHA']+=1;continue
        if r['recording'] is None:rejected['UNKNOWN_RECORDING']+=1;continue
        if r['recording'] in reserved_rec:rejected['RESERVED_FINAL_RECORDING']+=1;continue
        if r['severity'] not in ('MODERATE_OCCLUSION','SEVERE_OCCLUSION'):rejected['NO_MODEL_INDEPENDENT_HARD_TAG']+=1;continue
        relaxed.append(r)
        if r['recording'] in heldrec:rejected['HELDOUT_RECORDING_GUARD']+=1;continue
        eligible.append(r)
    # MAD2 grayscale64x48, same established rule, against exact excluded frames and selected candidates.
    forbidden_thumbs=[]
    for r in exclude:
        im=cv2.imread(str(P.ROOT/r['image']['path']));assert im is not None;forbidden_thumbs.append(cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(64,48)).astype(np.float32))
    forbidden=np.array(forbidden_thumbs);clean=[];near=[]
    for r in sorted(eligible,key=lambda r:key(r['id'])):
        P.verify(r['image']);im=cv2.imread(str(P.ROOT/r['image']['path']));thumb=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(64,48)).astype(np.float32);mad=float(np.abs(forbidden-thumb).mean((1,2)).min())
        if mad<=2:near.append(dict(id=r['id'],MAD=mad));continue
        clean.append(dict(r,min_MAD_to_exclusions=mad))
    byrec=Counter(r['recording'] for r in clean);bysev=Counter(r['severity'] for r in clean);enough=len(clean)>=8 and len(byrec)>=3
    save(RAW/'CANDIDATES_PRIVATE.json',dict(records=clean,near_duplicates=near,excluded_sha=sorted(excluded_sha),reserved_recordings=sorted(reserved_rec),heldout_recordings=sorted(heldrec),sha_exclusion_count=len(excluded_sha)))
    status='CANDIDATES_AVAILABLE_SELECTION_PENDING' if enough else 'HARD_CANDIDATE_METADATA_INSUFFICIENT'
    audit=dict(created_at=P.now(),status=status,gate='MIN_HARD_LABELING_JUSTIFIED',sources=source_count,unique_inventory=len(records),rejected=dict(rejected),hard_eligible_before_MAD=len(eligible),near_duplicate_rejected=len(near),eligible_after_MAD=len(clean),
        recordings=dict(byrec),severity=dict(bysev),frame_only_relaxed_recordings=dict(Counter(r['recording'] for r in relaxed)),
        no_model_prediction_reads=True,no_confidence_or_error_selection=True,model_independent_proxy='No further reliable hard proxy found; day/night alone is not occlusion severity.',
        extra_conservative_rule='Other frames from HELDOUT128 recordings excluded to retain recording-disjoint status in a future clean+hard training control. Also report frame-only relaxed recording counts.',
        bindings=bindings,near_duplicate_rule='mean grayscale absolute difference64x48 <=2 intensity units ==2/255; against excluded frame thumbnails; no candidate selected yet',
        annotations_requested=0,initial_images=0,reserve_images=0,user_action_required=False,GUI_created=False,labels_locked=False)
    save(DOC/'INVENTORY_AUDIT.json',audit)
    report=['# 최소 hard 어노테이션 — 후보 메타데이터 감사','',f'**{status}**','',
        'supervision-gap gate는 통과했지만, 이것이 곧 적격 후보8장이 있다는 뜻은 아니다. 예측/오차/confidence를 읽지 않고 기존 human severity와 capture condition만 확인했다.','',
        f"일반 플라스틱 unique inventory {len(records)}장. 제외 내역 `{dict(rejected)}`. Hard metadata를 가진 후보 {len(eligible)}장에서 MAD≤2/255 근접 중복 {len(near)}장을 제외했다. 최종 {len(clean)}장 /{len(byrec)} recording.",'',
        f"recording별: `{dict(byrec)}`. 난도별: `{dict(bysev)}`. HELDOUT recording 추가 보존을 풀고 프레임 기준만 적용해도 recording 분포는 `{dict(Counter(r['recording'] for r in relaxed))}`다.",'',
        '기존 split에서 이미 train/DEV로 명시된 recording과 별개로, 유지된 final_test 세션 및 current split 밖의 FINAL 세션 recording을 제외했다. green/wood는 이번 일반 플라스틱 대상에 자동 편입하지 않았다. 학습/eval GT 변경은 없다.','',
        '필요한 8장·3 recording 조건을 충족하지 못하면 후보를 임의 채우지 않는다. 낮은 confidence나 모델 실패로 고르거나 낮/밤만으로 hard라고 만들지 않았다. 따라서 아직 annotation GUI/라벨 입력/학습을 요청하지 않는다. 다음은 최소한 새 후보 recording의 model-independent hard tag가 있어야 한다.','',
        '[원천·집계 감사](INVENTORY_AUDIT.json) · [보존 실험 보고서](../pallet_single_model_preserve_v1/REPORT_KO.md)','']
    save(DOC/'REPORT_KO.md','\n'.join(report));print('HARD_INVENTORY',status,len(clean),dict(byrec),dict(rejected),flush=True)
if __name__=='__main__':main()
