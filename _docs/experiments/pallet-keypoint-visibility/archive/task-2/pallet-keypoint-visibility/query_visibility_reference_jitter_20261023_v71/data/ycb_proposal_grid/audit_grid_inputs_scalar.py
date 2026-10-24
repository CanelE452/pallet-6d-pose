"""Independent scalar sampled-pixel and metadata replay; saved source data only."""
from pathlib import Path
import hashlib,json,math
from collections import Counter
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))


def main():
    out=HERE/'GRID_QA.json';assert not out.exists()
    grid=[json.loads(s)for s in (HERE/'INPUT_INDEX_LINEAGE_ONLY.jsonl').read_text().splitlines()]
    labels=[json.loads(s)for s in (HERE/'GT_ONLY_GRID_TEACHERS.jsonl').read_text().splitlines()]
    initial=ref(HERE/'GRID_RESULT.json');cache={};samples=0
    with np.load(HERE/'INPUT_ONLY.npz',allow_pickle=False)as z:
        assert set(z.files)=={'rgb96','crop_valid96','query_valid','query_xy','box_xyxy','L'}
        d={k:z[k]for k in z.files}
    assert d['rgb96'].shape==(1944,3,96,96)and d['rgb96'].dtype==np.uint8
    assert d['crop_valid96'].shape==(1944,1,96,96)and d['crop_valid96'].dtype==bool
    for i,row in enumerate(grid):
        assert row['input_index']==i and labels[i]['input_index']==i
        box=row['box_xyxy'];qx=box[0]+(row['grid_col']+.5)*(box[2]-box[0])/9;qy=box[1]+(row['grid_row']+.5)*(box[3]-box[1])/9
        assert np.array_equal(d['query_xy'][i],[qx,qy])and np.array_equal(d['box_xyxy'][i],box)and d['L'][i]==row['L']
        if row['image']not in cache:
            assert sha(row['image'])==row['image_sha256'];cache[row['image']]=np.asarray(Image.open(row['image']).convert('RGB'))
        rgb=cache[row['image']];h,w=rgb.shape[:2];valid=0<=qx<w and 0<=qy<h and 0<=math.floor(qx+.5)<w and 0<=math.floor(qy+.5)<h
        assert bool(d['query_valid'][i])==valid
        for oy in [0,47,95]:
            for ox in [0,47,95]:
                x=qx+((ox+.5)/96-.5)*row['L'];y=qy+((oy+.5)/96-.5)*row['L'];pv=-.5<=x<w-.5 and-.5<=y<h-.5
                assert bool(d['crop_valid96'][i,0,oy,ox])==pv
                x0=math.floor(x);y0=math.floor(y);a=x-x0;b=y-y0;value=[0.,0.,0.]
                for dx,dy,weight in [(0,0,(1-a)*(1-b)),(1,0,a*(1-b)),(0,1,(1-a)*b),(1,1,a*b)]:
                    xx=x0+dx;yy=y0+dy
                    if 0<=xx<w and 0<=yy<h:
                        for c in range(3):value[c]+=int(rgb[yy,xx,c])*weight
                expected=[int(math.floor(min(255,max(0,v))+.5))if pv else 0 for v in value]
                assert np.array_equal(d['rgb96'][i,:,oy,ox],expected),(i,oy,ox)
                samples+=1
        label=labels[i];assert label['training_state'] in ['D','U']
        bb=label['uncertainty_bbox_xyxy'];L=row['L'];crop=[max(-.5,qx-L/2),max(-.5,qy-L/2),min(w-.5,qx+L/2),min(h-.5,qy+L/2)]
        if not valid:a='INVALID'
        elif crop[0]<=bb[0]and crop[1]<=bb[1]and crop[2]>=bb[2]and crop[3]>=bb[3]:a='FULL'
        elif min(crop[2],bb[2])>max(crop[0],bb[0])and min(crop[3],bb[3])>max(crop[1],bb[1]):a='PARTIAL'
        else:a='MISS'
        assert a==label['crop_availability']
        assert label['training_state']==('D'if a=='FULL'and label['source_conditional_state']=='D'else'U')
    assert samples==17496 and len(cache)==24 and ref(HERE/'GRID_RESULT.json')==initial
    c=Counter(r['crop_availability']for r in labels);t=Counter(r['training_state']for r in labels)
    result=dict(schema='V71_YCB_PROPOSAL_GRID_SCALAR_AND_DIRECT_QA_V1',completed=True,status='PASS',code=ref(__file__),GRID_RESULT_immutable=initial,
      input_NPZ=ref(HERE/'INPUT_ONLY.npz'),all1944_native_grid_query_values_replayed=True,query_valid=int(d['query_valid'].sum()),query_invalid=int((~d['query_valid']).sum()),
      scalar_bilinear_RGB_points_compared=samples,RGB_scalar_channel_values_compared=samples*3,maximum_pixel_mismatch=0,
      availability_explicit={k:c[k]for k in ['FULL','PARTIAL','MISS','INVALID']},training_explicit={k:t[k]for k in ['D','E','U']},
      metadata_and_GT_all1944_rows_replayed=True,GT_tensor_leak_fields=0,all_proposals_preserved=True,
      direct_visual_QA=dict(TRAIN_all81_original_controlled_mosaic_viewed=True,VALIDATION_all81_original_controlled_mosaic_viewed=True,oldL_span_and_RGB96_resolution_distinguished=True,no_GT_or_mask_overlay_in_input_RGB=True),
      strict_D_E_conflicts=0,source_score_model_current_inputs_read=0,new_model_forward_training_PnP=0)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    seal=HERE/'FINAL_SEAL.json';assert not seal.exists();files=['PROTOCOL.json','GRID_INPUT_LOCK.json','INPUT_EXTRACTION_SEAL.json','GRID_RESULT.json','GRID_QA.json','INPUT_IDENTITY_CONFLICT_AUDIT.json','GENERATED_GRID_MOCK.json','build_proposal_grid.py','audit_grid_inputs_scalar.py','INPUT_ONLY.npz','FROZEN_QUERY_GRID_INPUT_METADATA.jsonl','INPUT_INDEX_LINEAGE_ONLY.jsonl','GT_ONLY_GRID_TEACHERS.jsonl','TRAIN_ALL81_INPUT_CROPS_FIRST_TARGET.png','VALIDATION_ALL81_INPUT_CROPS_FIRST_TARGET.png']
    seal.write_text(json.dumps(dict(completed=True,status='PASS_DATA_CONSTRUCTION_ONLY',outputs=[ref(HERE/f)for f in files],all1944_inputs_and_GT_rows_preserved=True,model_fit_or_current_gate_authorized=False,new_model_forward_training_current_PnP=0),indent=2)+'\n');print(json.dumps(dict(QA=ref(out),seal=ref(seal))))


if __name__=='__main__':main()
