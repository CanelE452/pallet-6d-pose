"""Link old blind review and add missing U29 cases using the SAME UI/template."""
import json
import random
from pathlib import Path
from PIL import Image
from . import common as C

def main():
    old=C.B.L.PREV;mapping=C.read(old/'PRIVATE_REVIEW_MAPPING.json')['rows'];bykey={(r['frame_id'],r['corner_id']):r for r in mapping}
    subset=C.read(C.DOC/'SUBSET_LOCK.json')['groups'];records={r['id']:r for r in C.read(C.DOC/'INPUT_LOCK.json')['eval_records']}
    candidates=list(old.glob('*REVIEW*RESPONSES*.json'))+list(Path('/home/minjae/Downloads').glob('*REVIEW*RESPONSES*.json'))
    locked=[]
    for p in candidates:
        obj=C.read(p)
        if obj.get('locked') and obj.get('review_status')=='HUMAN_RESPONSES_LOCKED':locked.append((p,obj))
    responses={}
    for p,obj in locked:
        for key,value in obj['responses'].items():
            if key in responses:assert responses[key]==value,'Conflicting locked review answers'
            responses[key]=value
    overlap=[];missing=[]
    for r in subset['U29']:
        key=(r['id'],r['canonical'])
        if key in bykey:
            rec=bykey[key];answer=responses.get(rec['review_id'])
            overlap.append(dict(id=r['id'],canonical=r['canonical'],review_id=rec['review_id'],response=answer,status='REVIEW_PENDING' if answer is None else 'LOCKED_RESPONSE_AVAILABLE'))
        else:missing.append(r)
    rng=random.Random(2026092202);rng.shuffle(missing);dest=C.OUT/'blinded_review_addendum';dest.mkdir(parents=True,exist_ok=False);assets=dest/'assets';assets.mkdir()
    public=[];private=[];seen={}
    for r in missing:
        fid=r['id']
        if fid not in seen:
            token=f'IMG_{rng.getrandbits(80):020x}';path=assets/(token+'.jpg');record=records[fid];C.verify(record['image'])
            with Image.open(C.ROOT/record['image']['path']) as im:im.convert('RGB').save(path,quality=92,subsampling=0)
            seen[fid]=(token,path)
        token,path=seen[fid];review_id=f'CASE_{rng.getrandbits(80):020x}'
        public.append(dict(review_id=review_id,sample_id=token,image=str(path.relative_to(dest)),corner_index=r['canonical']))
        private.append(dict(id=fid,canonical=r['canonical'],review_id=review_id,source=records[fid]['image']))
    template=(old/'review_template.html').read_text();html=template.replace('__PUBLIC_CASES__',json.dumps(public,ensure_ascii=False))
    # Separate browser storage for this addendum; no new UI or answer semantics.
    html=html.replace("key='pallet_blind_review_v1_20260921'", "key='pallet_crop_completion_review_v2'")
    assert "key='pallet_crop_completion_review_v2'" in html
    for bad in ('reference_crop','keypoints_xy','canonical_errors','PRIVATE_REVIEW_MAPPING'):assert bad not in html
    for fid in records:assert fid not in html
    C.save(dest/'BLINDED_REVIEW.html',html)
    C.save(C.RAW/'REVIEW_PRIVATE_LINKS.json',dict(existing=overlap,new=private,private_do_not_publish=True))
    C.save(C.DOC/'REVIEW_STATUS.json',dict(status='REVIEW_PENDING' if len(responses)==0 else 'PARTIAL_LOCKED_REVIEW_AVAILABLE',
        old29=len(subset['U29']),existing_cases=len(overlap),additional_cases=len(missing),additional_images=len(seen),locked_response_files=[C.bind(p) for p,_ in locked],
        template_reused=C.bind(old/'review_template.html'),no_auto_answers=True,no_coordinate_substitution=True,
        public_viewer='outputs/pallet_posefix_crop_completion_v2/blinded_review_addendum/BLINDED_REVIEW.html',
        prior_exposure='UNKNOWN; reviewer must disclose; previously shown result images not independent blinded evidence'))
    print('REVIEW_LINKED',len(overlap),'existing',len(missing),'new',len(locked),'locked files',flush=True)

if __name__=='__main__':main()
