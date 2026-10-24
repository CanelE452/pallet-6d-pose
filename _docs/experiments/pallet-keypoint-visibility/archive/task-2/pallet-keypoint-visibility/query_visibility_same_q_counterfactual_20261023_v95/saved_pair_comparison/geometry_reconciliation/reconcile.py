"""Independent continuous-box/raster-centre/footprint reconciliation; masks only."""
from pathlib import Path
import copy
import csv
import hashlib
import json
import math
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent;COMP=HERE.parent;V=COMP.parent;I=V/'source_pair_intake';V2=I/'image_only_v2_exclusive_alpha'
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x)for x in Path(p).read_text().splitlines()]
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')


def geometry(mask,q,L):
    yy,xx=np.nonzero(mask);q=np.asarray(q,np.float64);h,w=mask.shape
    nn=np.floor(q+.5).astype(int)
    near=bool(0<=nn[0]<w and 0<=nn[1]<h and mask[nn[1],nn[0]])
    centres=float(np.min(np.hypot(xx-q[0],yy-q[1])))
    footprint=float(np.min(np.hypot(np.maximum(abs(xx-q[0])-.5,0),np.maximum(abs(yy-q[1])-.5,0))))
    def bilinear(x,y):
        ix,iy=math.floor(x),math.floor(y);dx,dy=x-ix,y-iy;value=0.
        for ax,ay,wt in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
            if 0<=ix+ax<w and 0<=iy+ay<h:value+=float(mask[iy+ay,ix+ax])*wt
        return value
    central=[bilinear(float(q[0]+dx*L/192),float(q[1]+dy*L/192))for dy in [-1,1]for dx in [-1,1]]
    return dict(nearest_native_pixel_xy=nn.tolist(),nearest_native_pixel_inside=near,q_inside_halfopen_unit_pixel_footprint_union=near,
                distance_to_native_pixel_CENTRES_px=centres,distance_to_native_pixel_CENTRES_over_L=centres/L,
                distance_to_CLOSED_unit_pixel_FOOTPRINTS_px=footprint,distance_to_CLOSED_unit_pixel_FOOTPRINTS_over_L=footprint/L,
                q_bilinear_mask_alpha=bilinear(float(q[0]),float(q[1])),RGB96_central4_bilinear_alphas=central,
                RGB96_central4_any_positive=bool(max(central)>0),RGB96_central4_mean_alpha=float(np.mean(central)))


def main():
    oldpath=COMP/'ALL15_PAIR_COMPARISON.json';oldbinding=bind(oldpath);old=read(oldpath);new=copy.deepcopy(old)
    feasibility=lines(V2/'FEASIBILITY960_GT_ONLY.jsonl')
    donors={x['id']:x for x in lines(V2/'actual/SELECTED_DONOR_RGBA_CERTIFICATES.jsonl')}
    brows={int(x['base_row_index']):x for x in csv.DictReader((V/'source_diagnosis/QUERY_TO_AUX_SUPPORT_METADATA.csv').open())}
    results=[]
    for record in new['records']:
        i=record['base_row_index'];f=feasibility[i];q=np.asarray(record['original_q'],np.float64);L=record['original_L']
        saved=np.load(f['protected_union_file']['path'],allow_pickle=False)['protected_union']
        reconstructed=np.zeros_like(saved);distances=[];inside=[]
        for region in f['protected_regions_GT_only']:
            x0,y0,x1,y1=map(float,region['bbox'])
            left,top=np.floor(np.array([x0,y0])+.5).astype(int);right,bottom=np.floor(np.array([x1,y1])+.5).astype(int)
            reconstructed[top:bottom+1,left:right+1]=True
            distances.append(math.hypot(max(x0-q[0],0,q[0]-x1),max(y0-q[1],0,q[1]-y1)))
            inside.append(bool(x0<=q[0]<=x1 and y0<=q[1]<=y1))
        assert np.array_equal(saved,reconstructed)
        assert min(distances)==record['q_min_distance_to_union_px']and any(inside)==record['q_inside_any_protected_range']
        g=geometry(saved,q,L);b=brows[i]
        for own,key in [('distance_to_native_pixel_CENTRES_px','distance_native_cell_centres_px'),('distance_to_CLOSED_unit_pixel_FOOTPRINTS_px','distance_native_cell_footprints_px'),('q_bilinear_mask_alpha','protected_q_bilinear_alpha'),('RGB96_central4_mean_alpha','protected_RGB96_central4_mean_alpha')]:assert math.isclose(g[own],float(b[key]),abs_tol=1e-12),(i,own)
        assert g['nearest_native_pixel_inside']==(b['protected_nearest_q_inside']=='True')
        assert g['RGB96_central4_any_positive']==(float(b['protected_RGB96_central4_max_alpha'])>0)
        donor=donors[f['assigned_donor_id']]
        with np.load(donor['compact_RGBA_cutout']['path'],allow_pickle=False)as z:base_alpha=z['alpha']
        rh,rw=f['donor_transform']['resized_hw'];cut=np.asarray(Image.fromarray(base_alpha).resize((rw,rh),Image.Resampling.NEAREST)).astype(bool)
        covers={}
        for variant,pos,mustcover in [('COVER',f['cover_xy'],True),('SHAM',f['sham_xy'],False)]:
            alpha=np.zeros_like(saved);x,y=pos;alpha[y:y+rh,x:x+rw]=cut
            assert bool(alpha[saved].all())if mustcover else not bool(alpha[saved].any())
            covers[variant]=geometry(alpha,q,L)
            prefix=variant.lower();assert covers[variant]['nearest_native_pixel_inside']==(b[prefix+'_nearest_q_inside']=='True')
            assert covers[variant]['RGB96_central4_any_positive']==(float(b[prefix+'_RGB96_central4_max_alpha'])>0)
        legacy={k:record.pop(k)for k in ['q_inside_any_protected_range','q_inside_all_protected_ranges','q_distance_to_each_range_px','q_min_distance_to_union_px','q_max_distance_to_range_px','q_min_distance_to_union_over_L']}
        record['original_continuous_CLOSED_uncertainty_box_geometry']=dict(q_inside_any_box=any(inside),q_inside_all_boxes=all(inside),
                     distances_to_each_box_px=distances,min_distance_to_box_union_px=min(distances),max_distance_to_box_px=max(distances),min_distance_over_L=min(distances)/L)
        record['actual_protected_raster_geometry']=g;record['actual_pasted_alpha_geometry']=covers
        record['legacy_ambiguous_field_values_continuous_boxes_only']=legacy
        assert record['models']==old['records'][len(results)]['models']
        results.append(dict(base_row_index=i,original_q=q.tolist(),original_ranges=[x['bbox']for x in f['protected_regions_GT_only']],
                            original_continuous_box_inside=any(inside),protected=g,actual_alpha=covers,all_native_mask_pixels_reconstructed_exact=True))
    summary=dict(accepted_pairs=15,inside_original_continuous_closed_uncertainty_boxes=sum(x['original_continuous_box_inside']for x in results),
                 nearest_q_pixel_inside_actual_protected_union=sum(x['protected']['nearest_native_pixel_inside']for x in results),
                 q_inside_actual_halfopen_pixel_footprints=sum(x['protected']['q_inside_halfopen_unit_pixel_footprint_union']for x in results),
                 protected_central4_bilinear_any=sum(x['protected']['RGB96_central4_any_positive']for x in results),
                 cover_nearest_q_inside=sum(x['actual_alpha']['COVER']['nearest_native_pixel_inside']for x in results),
                 cover_central4_untouched=sum(not x['actual_alpha']['COVER']['RGB96_central4_any_positive']for x in results),
                 sham_nearest_q_inside=sum(x['actual_alpha']['SHAM']['nearest_native_pixel_inside']for x in results),
                 sham_central4_untouched=sum(not x['actual_alpha']['SHAM']['RGB96_central4_any_positive']for x in results),
                 nearest_inside_rows=[x['base_row_index']for x in results if x['protected']['nearest_native_pixel_inside']])
    assert summary['nearest_inside_rows']==[260,284,625]and summary['protected_central4_bilinear_any']==3
    definitions={'original_continuous_boxes':'Original native CLOSED uncertainty ranges supplied by sourceAIannotations. Distance is to their continuous union. This was the previous comparison distance.',
                 'native_pixel_CENTRES':'Finite set of integer(x,y) centres whose saved protected mask isTrue. A positive distance for fractionalq does NOT imply q is outside a protected pixel footprint.',
                 'native_pixel_FOOTPRINTS':'Union of selected unit cells [x-.5,x+.5)×[y-.5,y+.5). Membership equals protected[floor(qy+.5),floor(qx+.5)]. Distance is reported to CLOSED cell footprints (infimum; zero can also occur at an excluded half-open edge).',
                 'RGB96_central4':'Four bilinear mask samples at q+(±L/192,±L/192); any positive alpha is sampling overlap, not a new semantic label.',
                 'pasted_alpha':'Larger actual cover/sham alpha is distinct from protected cue range; a protected-range miss does not prove cover misses q.'}
    new['geometry_definition_version']='V2_EXPLICIT_CONTINUOUS_BOX_VS_RASTER_CENTRE_VS_PIXEL_FOOTPRINT'
    new['geometry_definition_summary']=summary;new['geometry_definitions']=definitions
    new['geometry']='Use explicit geometry namespaces. The old unqualified all15-outside wording referred only to original continuous closed boxes and must not be used for actual protected pixel footprints: 3/15 nearest q pixels are protected.'
    new['preserved_prior_comparison']=oldbinding
    target=COMP/'ALL15_PAIR_COMPARISON_V2_GEOMETRY_EXPLICIT.json';write(target,new)
    add=dict(completed=True,PASS=True,prior_preserved=oldbinding,V2_comparison=bind(target),summary=summary,definitions=definitions,records=results,
        conclusion='No numeric error was found in the previous point-to-original-continuous-box calculation, but its field naming and unqualified outside wording were too broad. Correct raster/footprint statement: 3/15 protected nearestq cells (rows260,284,625), 12/15 outside; 3/15 central4 sampling overlaps. Row421 is outside protected footprints but actualcover reachesq. All model logits/predictions/correctness and q/labels/ranges remain unchanged.',
        B_metadata_reconciled=bind(V/'source_diagnosis/QUERY_TO_AUX_SUPPORT_METADATA.json'),new_RGB_feature_model_private_fit_PnP=0)
    write(HERE/'GEOMETRY_DEFINITION_RECONCILIATION.json',add)
    assert bind(oldpath)==oldbinding
    write(COMP/'GEOMETRY_V2_ADDENDUM_SEAL.json',dict(completed=True,PASS=True,original_comparison=oldbinding,
        original_seal=bind(COMP/'SEAL.json'),V2_comparison=bind(target),reconciliation=bind(HERE/'GEOMETRY_DEFINITION_RECONCILIATION.json'),code=bind(__file__),no_label_q_model_change=True))
    print(json.dumps(dict(V2_comparison=bind(target),reconciliation=bind(HERE/'GEOMETRY_DEFINITION_RECONCILIATION.json'),summary=summary),indent=2))


if __name__=='__main__':main()
