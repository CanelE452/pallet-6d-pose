"""Saved source proposals/GT evaluation and native display only. No model."""
from pathlib import Path
from collections import Counter,defaultdict
import csv
import hashlib
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
PUBLIC=HERE.parent
ROOT=PUBLIC.parent
RUN=PUBLIC/'DECODED_RGB_V2_RUN'
VIS=PUBLIC/'VISUAL_REVIEW'
V73=ROOT.parent/'query_visibility_public_pallet_loco_20261023_v73/data/intake'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def load(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def check(r):assert sha(r['path'])==r['sha256'],r['path']
def font(n,b=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if b else '')+'.ttf',n)


def xyxy(a):
    x,y,w,h=a['bbox'];return np.array([x,y,x+w,y+h],np.float64)


def iou(a,b):
    lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:]);inter=float(np.maximum(hi-lo,0).prod())
    aa=float(np.maximum(a[2:]-a[:2],0).prod());bb=float(np.maximum(b[2:]-b[:2],0).prod())
    return inter/(aa+bb-inter) if aa+bb-inter>0 else 0.


def draw_panel(rgb,bounds,pallets,rep,selected,points,size=(730,440)):
    """Native-coordinate crop; outside native canvas shown blank, not reflection."""
    x0,y0,x1,y1=map(int,bounds);assert x1>x0 and y1>y0
    h,w=rgb.shape[:2];canvas=Image.new('RGB',(x1-x0,y1-y0),(205,205,213))
    ax,ay,bx,by=max(x0,0),max(y0,0),min(x1,w),min(y1,h)
    if bx>ax and by>ay:canvas.paste(Image.fromarray(rgb[ay:by,ax:bx]),(ax-x0,ay-y0))
    scale=min(size[0]/canvas.width,size[1]/canvas.height)
    resized=canvas.resize((max(1,round(canvas.width*scale)),max(1,round(canvas.height*scale))),Image.Resampling.BILINEAR)
    out=Image.new('RGB',size,(241,242,246));offset=((size[0]-resized.width)//2,(size[1]-resized.height)//2);out.paste(resized,offset)
    d=ImageDraw.Draw(out)
    def point(p):return ((float(p[0])-x0)*scale+offset[0],(float(p[1])-y0)*scale+offset[1])
    for ann in pallets:
        b=xyxy(ann);u,v=point(b[:2]);s,t=point(b[2:]);d.rectangle((u,v,s,t),outline=(50,190,205),width=1)
    b=xyxy(rep);u,v=point(b[:2]);s,t=point(b[2:]);d.rectangle((u,v,s,t),outline=(250,0,230),width=3)
    if selected is not None:
        u,v=point(selected[:2]);s,t=point(selected[2:]);d.rectangle((u,v,s,t),outline=(255,220,0),width=3)
        colors=[(255,80,50),(255,160,0),(245,235,0),(150,255,0),(0,255,140),(0,235,255),(0,110,255),(155,80,255),(255,100,220)]
        for k,p in enumerate(points):
            if not np.isfinite(p).all():continue
            u,v=point(p)
            if 0<=u<size[0] and 0<=v<size[1]:
                d.ellipse((u-4,v-4,u+4,v+4),fill=colors[k],outline=(0,0,0));d.text((u+4,v-18),str(k),font=font(16,True),fill=colors[k],stroke_width=1,stroke_fill=(0,0,0))
    return out


def main():
    assert not (HERE/'OUTPUT_AUDIT_RESULT.json').exists()
    protocol=load(HERE/'BBOX_AUDIT_PROTOCOL.json');assert protocol['IoU_match_threshold']==.5
    for b in protocol['bindings']:check(b)
    result=load(RUN/'INFERENCE_RESULT.json');assert result['completed'] and result['frames']==20 and result['prediction_first_repeat_bit_exact'] and result['trained_parameters_and_registered_buffers_unchanged_during_all_forward_calls']
    for b in result['outputs']:check(b)
    geometrylock=load(ROOT/'public_reference_review/RGB_REFERENCE_GEOMETRY_LOCK.json')
    assert geometrylock['model_output_read']==0 and geometrylock['frozen_before_ownership_sanity'] is True
    for b in geometrylock['bindings']:check(b)
    # The wrapper already subtracts pad from boxes/points. Never subtract again.
    wrapper=Path('/tmp/pallet-bi-full-execution-20261019-v1/scripts/research/pallet_line_pose_v1/features.py')
    text=wrapper.read_text();assert 'box_xyxy=(boxes[index]-pad)' in text and 'keypoints_xy=(points[index]-pad)' in text
    write(HERE/'COORDINATE_AND_IOU_INTERPRETATION.json',dict(completed=True,source_wrapper=bind(wrapper),raw_export_coordinate_space='Original native RGB coordinates; pad already subtracted upstream.',additional_coordinate_transform='NONE',raw9_order='0..8 unchanged; no semantic remapping',primary_IoU='Continuous xyxy, no+1 pixel and no clipping of predicted box. GT comparison only; threshold>=.5.',outside_native_points='Retain and flag; native visual margins show gray outside image, never pretend reflected producer padding is real scene.',protocol=bind(HERE/'BBOX_AUDIT_PROTOCOL.json')))
    proposals=rows(RUN/'FROZEN_PROPOSALS.jsonl');inputs=rows(RUN/'INPUTS_RGB_ONLY.jsonl')
    repdoc=load(ROOT/'public_reference_review/REPRESENTATIVE_INSTANCE_LOCK.json');reps=repdoc['records'];assert repdoc['frozen_before_model_outputs']
    coco=load(V73/'annotations/loco-all-v1.json');gt=defaultdict(list)
    for ann in coco['annotations']:
        if ann['category_id']==7:gt[ann['image_id']].append(ann)
    assert len(inputs)==len(proposals)==len(reps)==20
    VIS.mkdir(parents=True,exist_ok=False);(VIS/'INDIVIDUAL').mkdir()
    output=[];individual=[];summary=Counter()
    for i,(inp,pr,rep) in enumerate(zip(inputs,proposals,reps)):
        assert pr['input_id']==inp['id'] and pr['image_sha256']==inp['image_sha256']==rep['image_sha256'] and rep['selection_index']==i
        check(inp['decoded_RGB_input']);rgb=np.load(inp['decoded_RGB_input']['path'],allow_pickle=False)
        assert rgb.dtype==np.uint8 and list(rgb.shape[:2])==inp['raw_hw'] and hashlib.sha256(rgb.tobytes()).hexdigest()==inp['RGB_pixels_sha256']
        anns=sorted(gt[rep['image_id']],key=lambda a:a['id']);officialrep=sorted(anns,key=lambda a:(-a['bbox'][2]*a['bbox'][3],a['id']))[0]
        assert officialrep==rep['representative_annotation']
        cs=pr['candidates'];selected_index=pr['selected_index'];assert len(cs)==0 if selected_index is None else selected_index==int(np.argmax([c['score'] for c in cs]))
        for j,c in enumerate(cs):
            assert c['candidate_index']==j and np.asarray(c['keypoints_xy']).shape==(9,2) and np.asarray(c['keypoints_conf']).shape==(9,)
            assert np.isfinite(c['score']) and np.isfinite(c['box_xyxy']).all() and np.isfinite(c['keypoints_xy']).all() and np.isfinite(c['keypoints_conf']).all()
        c=None if selected_index is None else cs[selected_index];box=None if c is None else np.asarray(c['box_xyxy'],np.float64);points=None if c is None else np.asarray(c['keypoints_xy'],np.float64)
        best=bestann=repiou=None;valid_count=None;out_fraction=None
        if c is None:status='NO_DETECTION';focus=rep['expanded_bbox_native_xyxy'];focus_source='Separate pre-locked GT representative display only; no selected proposal exists.'
        else:
            overlaps=[iou(box,xyxy(a)) for a in anns];pick=int(np.argmax(overlaps));best=overlaps[pick];bestann=anns[pick]['id'];repiou=iou(box,xyxy(officialrep))
            status='MATCH_LOCKED_REPRESENTATIVE' if repiou>=.5 else ('MATCH_OTHER_OFFICIAL_PALLET' if best>=.5 else 'POOR_OR_UNMATCHED_BOX_AT_0_5')
            h,w=rgb.shape[:2];point_valid=(points[:,0]>=0)&(points[:,0]<w)&(points[:,1]>=0)&(points[:,1]<h)
            rounded=np.floor(points+.5);point_valid&=(rounded[:,0]>=0)&(rounded[:,0]<w)&(rounded[:,1]>=0)&(rounded[:,1]<h);valid_count=int(point_valid.sum())
            inter=np.maximum(np.minimum(box[2:],[w,h])-np.maximum(box[:2],[0,0]),0).prod();area=np.maximum(box[2:]-box[:2],0).prod();out_fraction=float(1-inter/area) if area>0 else 1.
            lo=np.minimum(box[:2],points.min(0));hi=np.maximum(box[2:],points.max(0));margin=max(16.,float((hi-lo).max())*.15)
            focus=np.r_[np.floor(lo-margin),np.ceil(hi+margin)].astype(int).tolist();focus_source='Selected box/raw9 bounds+fixed15percent/max16px display margin; no GT recentring.'
        summary[status]+=1
        h,w=rgb.shape[:2];full=draw_panel(rgb,[-100,-100,w+100,h+100],anns,officialrep,box,points)
        detail=draw_panel(rgb,focus,anns,officialrep,box,points)
        fig=Image.new('RGB',(1520,610),(248,249,252));d=ImageDraw.Draw(fig)
        d.text((20,12),f"{i:02d} subset{inp['source_subset']} {inp['id']} | {status}",font=font(23,True),fill=(20,30,45))
        d.text((20,47),'Cyan: all official pallet boxes | Magenta: pre-locked representative | Yellow+0..8: unchanged selected raw proposal',font=font(18),fill=(35,45,65))
        fig.paste(full,(20,80));fig.paste(detail,(770,80))
        d.text((20,532),'Full native scene + gray outside-native margin (not reflected image).',font=font(17),fill=(35,45,65))
        if c is None:txt=f'No detection; official pallet instances={len(anns)}. Representative focus is display-only.'
        else:txt=f"selected={selected_index}, score={c['score']:.5f}; maxGT IoU={best:.3f}; representative IoU={repiou:.3f}; native-valid raw9={valid_count}/9"
        d.text((20,563),txt,font=font(20,True),fill=(25,45,65))
        fp=VIS/'INDIVIDUAL'/f'{i:02d}_subset{inp["source_subset"]}_NATIVE_RAW9.png';fig.save(fp);individual.append(bind(fp))
        output.append(dict(selection_index=i,input_id=inp['id'],source_subset=inp['source_subset'],official_image_id=rep['image_id'],raw_hw=inp['raw_hw'],candidate_count=len(cs),selected_index=selected_index,
            selected_confidence=None if c is None else c['score'],selected_raw_native_box_xyxy=None if c is None else c['box_xyxy'],selected_raw9_xy=None if c is None else c['keypoints_xy'],selected_raw9_conf=None if c is None else c['keypoints_conf'],raw9_indices=None if c is None else list(range(9)),
            max_IoU_any_official_pallet=best,max_IoU_annotation_id=bestann,representative_annotation_id=officialrep['id'],IoU_locked_representative=repiou,official_pallet_count=len(anns),
            diagnostic_status=status,selected_box_outside_native_area_fraction=out_fraction,raw9_native_valid_count=valid_count,
            all_selected_point_indices_and_float_values_preserved=True,candidate_selection_GT_influenced=False,native8_semantics='UNKNOWN_NOT_INFERRED_FROM_RAW9',semantic_D_E_S_from_bbox='UNAVAILABLE',
            visualization_native_bounds=focus,focus_rule=focus_source,decoded_RGB=inp['decoded_RGB_input'],native_overlay=bind(fp)))
    contacts=[]
    for subset in range(1,6):
        ids=[i for i,r in enumerate(output) if r['source_subset']==subset];assert len(ids)==4
        canvas=Image.new('RGB',(1520,1220),(248,249,252))
        for j,i in enumerate(ids):
            pic=Image.open(individual[i]['path']).convert('RGB').resize((760,610),Image.Resampling.LANCZOS);canvas.paste(pic,((j%2)*760,(j//2)*610))
        path=VIS/f'SUBSET_{subset}_ALL4_NATIVE_RAW9.png';canvas.save(path);contacts.append(bind(path))
    (HERE/'PER_FRAME_BBOX_DIAGNOSTIC.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in output))
    csvrows=[{k:r[k] for k in ['selection_index','input_id','source_subset','candidate_count','selected_index','selected_confidence','max_IoU_any_official_pallet','max_IoU_annotation_id','representative_annotation_id','IoU_locked_representative','official_pallet_count','diagnostic_status','selected_box_outside_native_area_fraction','raw9_native_valid_count']} for r in output]
    with open(HERE/'PER_FRAME_BBOX_DIAGNOSTIC.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    audit=dict(schema='V74_FIXED20_SAVED_INITIALIZER_BBOX_OUTPUT_AUDIT_V1',completed=True,PASS=True,protocol=bind(HERE/'BBOX_AUDIT_PROTOCOL.json'),code=bind(__file__),inference=bind(RUN/'INFERENCE_RESULT.json'),
        frames=20,candidates_preserved=sum(len(x['candidates']) for x in proposals),selected_first_argmax_unchanged=True,raw9_order_and_values_preserved=True,native_coordinates_already_unpadded=True,
        no_detection=sum(r['selected_index'] is None for r in output),detected_frames=sum(r['selected_index'] is not None for r in output),IoU_threshold=.5,IoU_prediction_clipping=False,
        statuses=dict(summary),maxGT_IoU_at_least_0_5=sum(r['max_IoU_any_official_pallet'] is not None and r['max_IoU_any_official_pallet']>=.5 for r in output),representative_IoU_at_least_0_5=sum(r['IoU_locked_representative'] is not None and r['IoU_locked_representative']>=.5 for r in output),
        representative_comparison_scope='Different match relative to a separately chosen fixed representative, not proof producer chose the wrong object for an unconstrained all-pallet task.',
        no_semantic_D_E_S_label_from_bbox=True,no_query_recenter_reorder_GT_candidate_choice=True,all_candidate_original_file=bind(RUN/'FROZEN_PROPOSALS.jsonl'),
        outputs=[bind(HERE/'PER_FRAME_BBOX_DIAGNOSTIC.jsonl'),bind(HERE/'PER_FRAME_BBOX_DIAGNOSTIC.csv')],all20_individual_overlays=individual,subset_contacts=contacts,visual_QA='PENDING_ADDITIVE_ACTUAL_REVIEW',new_model_forward_training_current_PnP=0,oldfiles_modified=False)
    write(HERE/'OUTPUT_AUDIT_RESULT.json',audit);print(json.dumps(dict(result=bind(HERE/'OUTPUT_AUDIT_RESULT.json'),statuses=dict(summary),maxIoUmatched=audit['maxGT_IoU_at_least_0_5'],repMatched=audit['representative_IoU_at_least_0_5'])))


if __name__=='__main__':main()
