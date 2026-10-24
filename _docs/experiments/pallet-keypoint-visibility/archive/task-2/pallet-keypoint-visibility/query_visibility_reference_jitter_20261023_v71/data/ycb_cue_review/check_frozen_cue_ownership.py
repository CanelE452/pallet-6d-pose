"""Ownership sanity AFTER RGB-only cue geometry freeze; never move a cue."""
from pathlib import Path
import hashlib,json,math
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
PROPOSALS=HERE/'FROZEN_AI_CUE_PROPOSALS_GT_ONLY.json'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def text(d,xy,s,size=20,color='#172536'):d.text(xy,s,font=ImageFont.truetype(FONT,size),fill=color)
def boundary(m):
    m=np.asarray(m,bool);p=np.pad(m,1);inside=np.ones_like(m)
    for y in range(3):
        for x in range(3):inside&=p[y:y+m.shape[0],x:x+m.shape[1]]
    return m&~inside


def main():
    out=HERE/'FROZEN_CUE_OWNERSHIP_SANITY';out.mkdir(exist_ok=False)
    seal=json.loads((HERE/'CUE_GEOMETRY_SEAL.json').read_text());assert checked(seal['proposal'])==PROPOSALS
    frozen=ref(PROPOSALS);p=json.loads(PROPOSALS.read_text());assert p['geometry_frozen_before_ownership_sanity']
    line={r['id']:r for r in map(json.loads,(HERE/'SOURCE_LINEAGE_GT_ONLY.jsonl').read_text().splitlines())}
    results=[];labels=[];rasters=[]
    for cue in p['proposals']:
        row=line[cue['target_id']];r=row['original'];c=row['controlled'];xy=cue['reference_xy'];bb=cue['uncertainty_bbox_xyxy']
        im=np.asarray(Image.open(checked(r['image'])).convert('RGB'));vis=np.asarray(Image.open(checked(r['visible'])))>0;full=np.asarray(Image.open(checked(r['full_proxy'])))>0
        x,y=np.floor(np.asarray(xy)+.5).astype(int);ys=np.arange(math.ceil(bb[1]),math.ceil(bb[3]));xs=np.arange(math.ceil(bb[0]),math.ceil(bb[2]));region=vis[np.ix_(ys,xs)]
        owned=bool(vis[y,x]);any_owned=bool(region.any());owned_count=int(region.sum());N=region.size
        ownership='COMPATIBLE_REFERENCE_PIXEL_VISIBLE_TARGET'if owned else'U_REFERENCE_PIXEL_OWNERSHIP_CONFLICT_OR_BOUNDARY_AMBIGUITY'
        # Mask agreement is a sanity condition, not the physical-cue D evidence.
        base='D_CONDITIONAL_AI_PHYSICAL_CUE'if owned else'U'
        controlled_state='U';coverage=None;after=None;paste=None
        if c is not None:
            after=np.asarray(Image.open(checked(c['image'])).convert('RGB'))
            with np.load(checked(c['labels']),allow_pickle=False)as z:paste=z['paste_mask'];newvis=z['visible_positive']
            pr=paste[np.ix_(ys,xs)];coverage=int(pr.sum())
            if base!='U':controlled_state='E_CONDITIONAL_OPAQUE_INTERVENTION'if pr.all()else'D_CONDITIONAL_UNTOUCHED_CUE'if not pr.any()else'U_PARTIAL_CUE_UNCERTAINTY_COVER'
        corr=cue['GT_only_existing_query_correspondence'];availability=corr['uncertainty_crop_availability'];q=corr['query_xy'];qx,qy=np.floor(np.asarray(q)+.5).astype(int)
        eligible=availability=='FULL'
        original_training=base if eligible else'U_CROP_'+availability
        after_training=controlled_state if eligible else'U_CROP_'+availability
        report=dict(cue_refID=cue['cue_refID'],target_id=cue['target_id'],split=cue['split'],reference_xy=xy,uncertainty_bbox_xyxy=bb,cue_description_ko=cue['cue_description_ko'],RGB_visible_cue_proposal_preserved=cue['reference_visibility_proposal'],native8_semantic_ID='UNKNOWN',reference_pixel_visible_mask=owned,reference_pixel_full_proxy=bool(full[y,x]),uncertainty_bbox_native_pixels=N,uncertainty_visible_pixels=owned_count,uncertainty_visible_fraction=owned_count/N,uncertainty_has_target_visible_support=any_owned,ownership_sanity_status=ownership,original_cue_teacher_candidate=base,existing_controlled_paste_covered_uncertainty_pixels=coverage,controlled_cue_teacher_candidate=controlled_state,observed_crop_availability=availability,unchanged_query_row=corr['native_query_row_lineage'],unchanged_query_xy=q,unchanged_L=corr['side_L_unchanged'],query_pixel_original_target_visible=bool(vis[qy,qx]),query_pixel_after_target_visible=None if after is None else bool(newvis[qy,qx]),original_training_teacher_candidate=original_training,controlled_training_teacher_candidate=after_training,geometry_moved_after_mask_sanity=False,source_RGB=r['image'],source_visible_mask=r['visible'],source_full_proxy=r['full_proxy'],controlled_RGB=None if c is None else c['image'],GT_controlled_labels=None if c is None else c['labels'],teacher_limit='AI observational physical-cue proposal + BOP raster ownership/paste sanity only; not human/native8 semantic gold; edge uncertainty may straddle foreground/background.',root_review_and_identifiability_admission_pending=True)
        results.append(report);labels.append(dict(**report,semantic_training_admitted=False,GT_only=True,reference_or_state_or_offset_is_not_model_input=True))
        rasters.append((im,vis,full,after,paste))
    figures=[]
    for page in range(3):
        sheet=Image.new('RGB',(1540,1510),'#f5f7fa');d=ImageDraw.Draw(sheet)
        text(d,(20,12),'고정 RGB cue → mask 소유권 sanity / 좌표 재배치 없음',28)
        text(d,(20,55),'raw64 원본 / visible·full 경계 / 기존 controlled · 회색/노랑/분홍은 GT ref·uncertainty·원 q 표시, 모델 입력 아님',19)
        for j in range(4):
            k=page*4+j;a,vis,full,after,paste=rasters[k];c=p['proposals'][k];r=results[k];x,y=r['reference_xy'];bb=r['uncertainty_bbox_xyxy'];q=c['GT_only_existing_query_correspondence']['query_xy'];L=c['GT_only_existing_query_correspondence']['side_L_unchanged'];oy=145+j*330
            text(d,(20,oy-40),f'{k+1:02d} {r["split"]} {r["target_id"]}   r={r["reference_xy"]} / q{r["unchanged_query_row"]}, L={L:.1f}',21)
            for col,array in enumerate([a,a.copy(),after if after is not None else a]):
                if col==1:
                    array=array.copy();array[boundary(full)]=[50,160,255];array[boundary(vis)]=[0,255,110]
                if col==2 and paste is not None:
                    array=array.copy();array[boundary(paste)]=[255,160,0]
                raw=Image.fromarray(array).crop((x-32,y-32,x+32,y+32)).resize((256,256),Image.Resampling.NEAREST)
                if col:
                    dd=ImageDraw.Draw(raw);rect=tuple((v-(x-32 if n%2==0 else y-32))*4 for n,v in enumerate(bb));dd.rectangle(rect,outline='#ffdd00',width=2);dd.line((120,128,136,128),fill='#dddddd',width=2);dd.line((128,120,128,136),fill='#dddddd',width=2)
                    qx=(q[0]-(x-32))*4;qy=(q[1]-(y-32))*4;dd.line((qx-6,qy,qx+6,qy),fill='#ff00bb',width=2);dd.line((qx,qy-6,qx,qy+6),fill='#ff00bb',width=2)
                    dd.rectangle(((q[0]-L/2-(x-32))*4,(q[1]-L/2-(y-32))*4,(q[0]+L/2-(x-32))*4,(q[1]+L/2-(y-32))*4),outline='#00d9ee',width=1)
                sheet.paste(raw,(20+col*280,oy))
            tx=875;text(d,(tx,oy+3),r['cue_description_ko'],19)
            text(d,(tx,oy+42),f'original visible pixel={r["reference_pixel_visible_mask"]}',20)
            text(d,(tx,oy+78),f'bbox target visible {r["uncertainty_visible_pixels"]}/{r["uncertainty_bbox_native_pixels"]}',20)
            text(d,(tx,oy+114),f'crop availability: {r["observed_crop_availability"]}',22)
            text(d,(tx,oy+150),f'paste covers bbox: {r["existing_controlled_paste_covered_uncertainty_pixels"]}/{r["uncertainty_bbox_native_pixels"]}',20)
            text(d,(tx,oy+188),'original teacher: '+r['original_training_teacher_candidate'],17)
            text(d,(tx,oy+222),'after teacher: '+r['controlled_training_teacher_candidate'],17)
        text(d,(20,1470),'mask0/경계불일치는 원인을 E/O/S로 증명하지 않습니다. cue proposal은 그대로 보존하고 충돌·PARTIAL·MISS는 학습 후보 U입니다.',20)
        f=out/f'FROZEN_CUE_NATIVE64_SANITY_{page+1}.png';sheet.save(f);figures.append(ref(f))
    report=dict(schema='V71_FROZEN_YCB_AI_CUE_OWNERSHIP_SANITY_V1',completed=True,source_only=True,code=ref(__file__),frozen_proposals=frozen,geometry_seal=ref(HERE/'CUE_GEOMETRY_SEAL.json'),reference_geometry_unchanged=ref(PROPOSALS)==frozen,all12_preserved=True,counts=dict(reference_cue_teacher_candidates=dict(Counter(r['original_cue_teacher_candidate']for r in results)),crop_availability=dict(Counter(r['observed_crop_availability']for r in results)),original_training_teacher_candidates=dict(Counter(r['original_training_teacher_candidate']for r in results)),controlled_training_teacher_candidates=dict(Counter(r['controlled_training_teacher_candidate']for r in results))),cases=results,figures=figures,semantic_training_admitted=False,human_or_native8_gold=False,prior_mask_exposure=True,RGB_only_localization_before_this_sanity=True,new_reference_or_query_recenter_or_L_change=False,new_jitter_newcomposite_model_forward_training_current_PnP=0)
    f=out/'SANITY_REPORT.json';f.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (out/'GT_ONLY_TEACHER_CANDIDATES.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in labels))
    (out/'SEAL.json').write_text(json.dumps(dict(completed=True,report=ref(f),GT_only_candidates=ref(out/'GT_ONLY_TEACHER_CANDIDATES.jsonl'),figures=figures,reference_geometry_unchanged=ref(PROPOSALS)==frozen),indent=2)+'\n')
    print(json.dumps(dict(report=ref(f),counts=report['counts']),ensure_ascii=False))


if __name__=='__main__':main()
