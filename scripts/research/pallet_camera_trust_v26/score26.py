"""Paired camera effects and actual observed object fraction, with fixed populations."""
from context26 import *
import cv2
METHODS=('stock','base','line_vector','gate_paired','vector_local_edge','vector_line_edge')

def metrics(error,valid,det,mask):
    keep=mask&valid&det[:,None];values=error[keep]
    requested=mask&valid;den=requested.sum();normalized=np.where(det[:,None],np.minimum(np.nan_to_num(error,nan=800)/800,1),1)
    return dict(requested_points=int(den),detected_points=int(keep.sum()),mean_px=float(values.mean()) if len(values) else None,p90_px=float(np.quantile(values,.9)) if len(values) else None,full_normalized=float(normalized[requested].mean()) if den else None)

def main():
    assert read(RUN/'TARGET_AUDIT.json')['status']=='PASS';m=read(RUN/'cache/manifest.json');ts=read(RUN/'TARGETS.json')['rows'];gt=np.asarray([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts]);det=np.array([r['detected'] for r in m['rows']]);perimage=[];observed=[];pointrows=[];sources={}
    for r in m['rows']:
        raw=np.load(ROOT/r['depth_path']);v=(raw['object']>0)&(np.abs(raw['scene']-raw['object'])<.003);x,y=r['origin'];h,w=r['image_hw'];visible=int(v[y:y+h,x:x+w].sum());object_pixels=int((raw['object']>0).sum());observed.append(dict(id=r['id'],visible_fraction=visible/max(object_pixels,1),visible_pixels=visible,object_pixels=object_pixels));assert cv2.imread(str(ROOT/r['image'])).shape[:2]==tuple(r['original_hw'])
    for arm in METHODS:
        for seed in ([1] if arm=='stock' else SEEDS):
            name='stock' if arm=='stock' else f'{arm}_seed{seed}';path=RUN/'predictions'/f'{name}.npz';p=np.load(path);sources[str(path.relative_to(ROOT))]=sha(path);assert p['ids'].tolist()==[r['id'] for r in m['rows']];assert np.array_equal(p['detected'],det);error=np.linalg.norm(p['points'].astype(float)-gt,axis=-1)
            for i,r in enumerate(m['rows']):
                b=p['base'][i].astype(float);delta=p['points'][i]-b;res=gt[i]-b;side=r['condition'];inward=np.array([1,0]) if side=='crop_left20' else np.array([-1,0]) if side=='crop_right20' else None
                for typ,mask in [('all',valid[i]),('front',np.arange(8)<4),('outside',np.array(ts[i]['outside']))]:
                    if not mask.any():continue
                    perimage.append(dict(id=r['id'],source_group=r['source_group'],asset=r['asset'],yaw_deg=r['yaw_deg'],height=r['camera_height_m'],condition=r['condition'],arm=arm,seed=seed,subset=typ,detected=bool(det[i]),visible_fraction=observed[i]['visible_fraction'],mean_px=float(error[i,mask].mean()) if det[i] else None,full_normalized=float(np.minimum(error[i,mask]/800,1).mean()) if det[i] else 1,points=int(mask.sum())))
                if det[i]:
                    for j in range(8):pointrows.append(dict(id=r['id'],source_group=r['source_group'],yaw_deg=r['yaw_deg'],height=r['camera_height_m'],condition=r['condition'],arm=arm,seed=seed,point=j,base_px=float(np.linalg.norm(res[j])),after_px=float(error[i,j]),outside=bool(ts[i]['outside'][j]),need_inward=bool(res[j]@inward>0) if inward is not None else None,propose_inward=bool(delta[j]@inward>0) if inward is not None else None,wrong_direction=bool(res[j]@delta[j]<=0),move_px=float(np.linalg.norm(delta[j])),visible_fraction=observed[i]['visible_fraction']))
    table=[]
    for c in ('clean','crop_left20','crop_right20','physical_panel'):
        for arm in METHODS:
            for seed in ([1] if arm=='stock' else SEEDS):
                for yaw in (-30,-10,0,10,30):
                    for height in (.12,.4,.9):
                        for typ in ('all','front','outside'):
                            rows=[r for r in perimage if (r['condition'],r['arm'],r['seed'],r['yaw_deg'],r['height'],r['subset'])==(c,arm,seed,yaw,height,typ)]
                            if not rows:continue
                            good=[r for r in rows if r['detected']];ps=[r for r in pointrows if (r['condition'],r['arm'],r['seed'],r['yaw_deg'],r['height'])==(c,arm,seed,yaw,height) and (typ=='all' or typ=='front' and r['point']<4 or typ=='outside' and r['outside'])]
                            table.append(dict(condition=c,arm=arm,seed=seed,yaw_deg=yaw,height=height,subset=typ,requested=len(rows),detected=len(good),detection_rate=len(good)/len(rows),mean_px=float(np.mean([r['after_px'] for r in ps])) if ps else None,frame_mean_px=float(np.mean([r['mean_px'] for r in good])) if good else None,full_normalized=float(np.mean([r['full_normalized'] for r in rows])),visible_fraction=float(np.mean([r['visible_fraction'] for r in rows])),need_inward_fraction=float(np.mean([r['need_inward'] for r in ps])) if ps and ps[0]['need_inward'] is not None else None,wrong_moving_fraction=float(np.mean([r['wrong_direction'] for r in ps if r['move_px']>.1])) if any(r['move_px']>.1 for r in ps) else None))
    paired=[]
    # Within each fixed asset/appearance/height, compare frontal and +/-30deg views.
    for arm in METHODS:
        for seed in ([1] if arm=='stock' else SEEDS):
            for c in ('clean','crop_left20','crop_right20','physical_panel'):
                pop=[r for r in perimage if r['arm']==arm and r['seed']==seed and r['condition']==c and r['subset']=='front'];groups={}
                for r in pop:groups.setdefault((r['source_group'],r['height']),{})[r['yaw_deg']]=r
                diffs=[];normdiff=[]
                for key,rows in groups.items():
                    if not all(y in rows for y in (0,-30,30)):continue
                    normdiff.append(rows[0]['full_normalized']-(rows[-30]['full_normalized']+rows[30]['full_normalized'])/2)
                    if all(rows[y]['detected'] for y in (0,-30,30)):diffs.append(rows[0]['mean_px']-(rows[-30]['mean_px']+rows[30]['mean_px'])/2)
                paired.append(dict(arm=arm,seed=seed,condition=c,matched_detected_triplets=len(diffs),requested_triplets=len(groups),frontal_minus_oblique_px=float(np.mean(diffs)) if diffs else None,frontal_minus_oblique_normalized=float(np.mean(normdiff)),scope='Same asset/color/height/distance, actual camera orbit. Not a general real-camera performance guarantee.'))
    with (RUN/'CAMERA_POINTS.jsonl').open('w') as f:
        for r in pointrows:f.write(json.dumps(r)+'\n')
    write(RUN/'CAMERA_SUMMARY.json',dict(complete=True,rows=table,paired_camera_effects=paired,actual_visibility=observed,prediction_bindings=sources,full_population=360,rendered_images=180,source_groups=6,assets=2,scope='No new learning; frozen outputs on controlled renders. Actual mask measures observed object pixels relative to isolated same-camera object, not virtual cuboid edge visibility. Point visibility labels are not inferred from empty bounding corners. Six asset/color contexts are not six independent assets.'))
    write(RUN/'CAMERA_PER_IMAGE.json',dict(rows=perimage));record('실제 카메라 통제 결과 집계 완료:360개 분모·검출실패 포함 정규화 오차, 앞면4점/전체8점/잘림밖 점을 분리. 동일메시/색/높이의 정면 대±30도 사선 짝비교 및 깊이기반 실제 남은 물체비율 기록.')
    for c in ('clean','crop_left20','crop_right20','physical_panel'):
        subset=[r for r in table if r['condition']==c and r['arm']=='base' and r['seed']==1 and r['subset']=='front'];print('CAMERA26',c,[(r['yaw_deg'],r['height'],r['detected'],None if r['mean_px'] is None else round(r['mean_px'],2)) for r in subset],flush=True)

if __name__=='__main__':main()
