"""Exhaust the full8,031 adaptation pool, not only the balanced training sample."""
import csv
from collections import Counter
from .inventory import P,DOC,RAW,save

def main():
    initial=P.read(DOC/'INVENTORY_AUDIT.json');priv=P.read(RAW/'CANDIDATES_PRIVATE.json');excluded=set(priv['excluded_sha']);sealed=set(priv['reserved_recordings']);held=set(priv['heldout_recordings'])
    gd=P.read(P.ROOT/'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json');aliases={s['session_key']:g['recording_id'] for g in gd['groups'] if not g['is_collection'] for s in g['sessions']}
    split=P.read(P.ROOT/'_docs/experiments/pallet_existing_data_transfer_v1/SPLIT_LOCK.json');tags={r['image']['sha256']:r['severity'] for r in split['train']+split['heldout']};framepath=P.ROOT/'data/evaluation/pallet_eval_v1/manifests/frames.csv'
    for r in csv.DictReader(framepath.open()):
        if r['object_type']!='plastic':continue
        occ=r['occlusion'].lower();trunc=r['truncation'].lower();sev='SEVERE_OCCLUSION' if occ in ('heavy','severe') or trunc in ('major','severe') else 'MODERATE_OCCLUSION' if occ in ('partial','moderate','mild') else None
        if sev:tags.setdefault(r['image_sha256'],sev)
    stats=Counter();recs=Counter();missing_recs=Counter();paths=[];seen=set();extra=[]
    poollockpath=P.ROOT/'data/evaluation/pallet_eval_v1/adaptation/ADAPTATION_POOL_LOCK.json';poollock=P.read(poollockpath);paths.append(P.bind(poollockpath));scanned_sessions={}
    for condition,subdir in [('daytime','outside'),('nighttime','night')]:
        for name in poollock['condition_audit'][condition]['sessions']:
            folder=P.ROOT/'data/pallet/raw_data'/subdir/name/'rgb';images=sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg'));scanned_sessions[name]=len(images)
            for image in images:
                sha=P.sha(image)
                if sha in seen:continue
                seen.add(sha);session=str(image.parent.parent.relative_to(P.ROOT));rec=aliases.get(session);recs[str(rec)]+=1;sev=tags.get(sha)
                if sha in excluded:stats['EXCLUDED_IMAGE_SHA']+=1
                elif rec is None:stats['UNKNOWN_RECORDING']+=1
                elif rec in sealed:stats['RESERVED_FINAL_RECORDING']+=1
                elif rec in held:stats['HELDOUT_RECORDING']+=1
                elif sev not in ('MODERATE_OCCLUSION','SEVERE_OCCLUSION'):stats['NO_MODEL_INDEPENDENT_HARD_TAG']+=1;missing_recs[rec]+=1
                else:extra.append(dict(image=str(image.relative_to(P.ROOT)),sha256=sha,recording=rec,severity=sev))
    assert len(seen)==8031
    save(RAW/'FULL_POOL_EXTRA_CANDIDATES.json',extra)
    save(DOC/'EXPANDED_POOL_AUDIT.json',dict(created_at=P.now(),source_frames=len(seen),scanned_sessions=scanned_sessions,recordings=dict(recs),reasons=dict(stats),missing_hard_tag_recordings=dict(missing_recs),
        additional_hard_tagged_candidates=len(extra),bindings=paths,model_outputs_opened=0,RGB_error_or_confidence_used=False,
        status=initial['status'] if not extra else 'ADDITIONAL_METADATA_CANDIDATES_REQUIRE_SELECTION_AUDIT'))
    assert not extra,'New eligible candidates found; perform full exclusion/MAD/selection, do not call inventory insufficient.'
    path=DOC/'REPORT_KO.md';path.write_text(path.read_text()+f'\n## 전체 adaptation pool 추가 확인\n\nBalanced sample만 확인하지 않고 daytime+nighttime 전체 **{len(seen)}장**도 확인했다. `{dict(stats)}`. 기존 model-independent hard 태그로 새로 추가할 수 있는 후보는0장이다. 원본에 가림이 없다는 뜻이 아니라, 현재 메타데이터로 hard를 정할 근거가 없다는 뜻이다. 모델 실패나 confidence로 대체하지 않았다.\n\n[전체 pool 감사](EXPANDED_POOL_AUDIT.json)\n')
    print('FULL_POOL_AUDIT',len(seen),dict(stats),'extra',len(extra),flush=True)
if __name__=='__main__':main()
