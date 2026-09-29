"""Metadata and RGB-only review preparation. No prediction/error-based cleaning."""
import argparse
from collections import Counter
import csv
import io
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from . import common as C

def prepare():
    destination=C.RAW/'CLEAN_MEMBERSHIP_PRIVATE.json'
    if destination.exists():
        for b in C.read(destination)['bindings']:C.verify(b)
        print('POOL_ALREADY_PREPARED');return
    base=C.ROOT/'data/pallet/results/pallet_type_selftrain_v1'
    protpath=C.ROOT/'_docs/experiments/pallet_type_selftrain_v1/selftrain_recovery_v1/pose_only/PROTOCOL.json'
    prot=C.read(protpath);trainpath=C.ROOT/prot['datasets']['REF']['train_list']['path']
    used=Counter(Path(p).stem for p in trainpath.read_text().splitlines() if Path(p).name.startswith('PLASTIC__'))
    # Explicit whitelist extraction: never pass prediction/GT/error arrays to review.
    accepted=[{k:r[k] for k in ('id','image','session','recording_id','object_type')}
              for r in C.read(base/'PSEUDO_ACCEPTED.json') if r['kind']=='PLASTIC']
    decisions=[{k:r[k] for k in ('id','accepted','reason')} for r in C.read(base/'PSEUDO_DECISIONS.json') if r['kind']=='PLASTIC']
    tags_path=C.ROOT/'data/pallet/results/pallet_min_hard_ab_v1/ROUND_1_TAGS_PRIVATE.json'
    tag_by_sha={v['image_sha256']:v['tag'] for v in C.read(tags_path).values()}
    adapt=C.ROOT/'data/evaluation/pallet_eval_v1/adaptation/MAIN_UNLABELED_BALANCED.csv'
    bysha={r['image_sha256']:r for r in csv.DictReader(adapt.open())}
    assert len(accepted)==249 and len(used)==217 and len(decisions)==len(bysha)==1000
    rows=[]
    for i,r in enumerate(sorted(accepted,key=lambda r:(r['session'],r['image']['path']))):
        C.verify(r['image']);assert r['image']['sha256'] in bysha
        rows.append(dict(review_id=f'C{i+1:03d}',train_id=r['id'],image=r['image'],
            recording=r['recording_id'],session=Path(r['session']).name,
            used=r['id'] in used,occurrences_per_epoch=used[r['id']],
            previous_tag=tag_by_sha.get(r['image']['sha256']),
            EXTERNAL_CLEAN='UNREVIEWED',TRUNCATION='UNREVIEWED',HARD_VIEW='UNREVIEWED',
            SELF_OCCLUSION='NOT_ASSESSED',evidence='NONE'))
    bindings=[C.bind(p) for p in (base/'PSEUDO_ACCEPTED.json',base/'PSEUDO_DECISIONS.json',protpath,trainpath,
        tags_path,adapt,C.ROOT/'data/evaluation/pallet_eval_v1/adaptation/ADAPTATION_POOL_LOCK.json',
        C.ROOT/'scripts/self_training_yolo/build_adaptation_pool.py',
        C.ROOT/'_docs/experiments/pallet_min_hard_ab_v1/DIFFICULTY_TAG_LOCK.json',
        C.ROOT/'_docs/experiments/pallet_min_hard_ab_v1/HARD_SELECTION_PUBLIC.json')]
    C.save(destination,dict(rows=rows,bindings=bindings,predictions_or_GT_used_for_clean=False),True)
    public=dict(candidate=1000,accepted=249,used=217,used_real_slots=sum(used.values()),
        accepted_sessions=dict(Counter(r['session'] for r in rows)),
        used_sessions=dict(Counter(r['session'] for r in rows if r['used'])),
        previous_tag_overlap=dict(Counter(r['previous_tag'] or 'UNTAGGED' for r in rows)),
        USED217_CLEAN='UNRESOLVED',ACCEPTED249_CLEAN='UNRESOLVED',
        source=C.bind(destination),source_bindings=bindings,
        exclusion='No prediction or correction magnitude used. Previously unused32 will not enter primary training.')
    C.save(C.DOC/'CLEAN_MEMBERSHIP_PUBLIC.json',public)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    sheets=[]
    for group,selected in [('accepted249',rows),('used217',[r for r in rows if r['used']])]:
        for page in range((len(selected)+15)//16):
            subset=selected[page*16:(page+1)*16]
            canvas=Image.new('RGB',(1600,1328),'#171b22');draw=ImageDraw.Draw(canvas)
            for j,row in enumerate(subset):
                with Image.open(C.ROOT/row['image']['path']) as im:
                    im=im.convert('RGB');im.thumbnail((400,300))
                    x=(j%4)*400;y=(j//4)*332
                    canvas.paste(im,(x+(400-im.width)//2,y+(300-im.height)//2))
                    draw.text((x+5,y+304),f"{row['review_id']}  {row['session']}  {'USED' if row['used'] else 'not used'}",font=font,fill='white')
            path=C.OUT/'rgb_only'/f'{group}_{page+1:02d}.jpg';path.parent.mkdir(parents=True,exist_ok=True)
            canvas.save(path,quality=92);sheets.append(C.bind(path))
    review=io.StringIO();writer=csv.DictWriter(review,fieldnames=['review_id','EXTERNAL_CLEAN','TRUNCATION','HARD_VIEW','SELF_OCCLUSION','evidence']);writer.writeheader()
    writer.writerows({k:r[k] for k in writer.fieldnames} for r in rows)
    C.save(C.OUT/'RGB_ONLY_REVIEW.csv',review.getvalue(),True)
    C.save(C.RAW/'CONTACT_SHEETS_LOCK.json',dict(sheets=sheets,private=True,no_prediction_GT_error_overlay=True),True)
    C.save(C.DOC/'CLEAN_POOL_AUDIT.md',f'''# Current217 clean 감사 — 검토 전

1000 후보 →249 승인 →217 실제 TRAIN 고유영상 /512 real 슬롯을 복원했다. 기존123장 난도tag와 승인249장의 정확한 image SHA 교집합은 {sum(r['previous_tag'] is not None for r in rows)}장이다. 세션meta는 촬영세션/주야만 제공하며 외부 가림 여부를 증명하지 않는다.

따라서 현재 USED217/ACCEPTED249를 clean이라고 선언할 수 없다. RGB-only contact sheet와 review CSV를 private outputs에 준비했다. prediction·GT·error·confidence·보정량은 화면이나 판정에 사용하지 않는다. accepted249는 감사용, 새primary TRAIN은 used217 안에서만 선택한다.

EXTERNAL_CLEAN은 외부물체가 팔레트를 가리지 않는다는 뜻이다. 자기 가림·잘림·어려운 시점은 별도로 기록하며 모든 코너 가시성과 같지 않다. 확신 없는 RGB는 UNRESOLVED로 두고 fit을 시작하지 않는다.
''')
    print('RGB_ONLY_SHEETS_READY',len(sheets),public['used_sessions'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare'])
    prepare()
