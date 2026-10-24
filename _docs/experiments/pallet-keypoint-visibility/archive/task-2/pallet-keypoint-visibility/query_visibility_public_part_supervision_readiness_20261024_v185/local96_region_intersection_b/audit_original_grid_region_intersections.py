"""Metadata-only coarse region / original grid overlap. Actual requires root entry."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import math
import platform
import traceback
import numpy as np

H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
V164=P/'query_visibility_native_part_counterfactual_20261024_v164'
INPUT=V164/'generator_preparation/FIXED22_FRAME_INPUTS_AND_66_ATTEMPTS_BEFORE_PIXELS.json'
REGIONS=P/'query_visibility_train_native_part_witness_20261024_v163/reviewer_a/part_region_pilot/ALL22_COARSE_PART_REGION_OBSERVATIONS.jsonl'
PROOFS=V164/'source_part_pixels_v1/ALL66_PART_PIXEL_INPUT_PROOFS.jsonl'
RULES=H/'RULES_BEFORE_ACTUAL_REGION_INTERSECTIONS.json'
PROTOCOL=H/'PROTOCOL_BEFORE_ACTUAL_REGION_INTERSECTIONS.json'
CHECKS=H/'GENERATED_MEANINGFUL_CHECKS.json';PREP=H/'PREPARATION_COMPLETE.json'
OUT=H/'actual_intersections_v1'


def require(x,msg):
    if not x:raise AssertionError(msg)
def sha(path):
    p=Path(path);require(p.suffix in ('.json','.jsonl','.py') and '_v112' not in str(p).lower(),'Only exact metadata/source paths')
    return hashlib.sha256(p.read_bytes()).hexdigest()
def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def check(b):require(bind(b['path'])==b,'Immutable metadata SHA');return Path(b['path'])
def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def grid(q,L):
    q=np.asarray(q,np.float64);L=np.float64(L)
    require(q.shape==(2,) and np.isfinite(q).all() and np.isfinite(L) and L>0,'Original finite F64 q/L')
    offset=(np.arange(96,dtype=np.float64)+np.float64(.5))/np.float64(96)-np.float64(.5)
    x,y=np.meshgrid(q[0]+offset*L,q[1]+offset*L)
    return np.stack((x,y),axis=-1)


def point_weights(points,hw,bbox):
    """Pure fourtap positive raw-pixel overlap; no image or semantic data used."""
    p=np.asarray(points,np.float64);require(p.ndim==2 and p.shape[1]==2 and np.isfinite(p).all(),'Points F64 Nx2')
    h,w=map(int,hw);require(h>0 and w>0,'Original image HW')
    b=np.asarray(bbox);require(b.shape==(4,) and np.issubdtype(b.dtype,np.integer) and b[0]<b[2] and b[1]<b[3],'Integer half-open bbox')
    x,y=p.T;ix=np.floor(x).astype(np.int64);iy=np.floor(y).astype(np.int64);dx=x-ix;dy=y-iy
    valid=(x>=-.5)&(x<w-.5)&(y>=-.5)&(y<h-.5)
    mass=np.zeros(len(p),np.float64);unique=set();unique_valid=set()
    for ax,ay,weight in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
        xx,yy=ix+ax,iy+ay
        take=(weight>0)&(xx>=0)&(xx<w)&(yy>=0)&(yy<h)&(xx>=b[0])&(xx<b[2])&(yy>=b[1])&(yy<b[3])
        mass[take]+=weight[take]
        unique.update(zip(xx[take].tolist(),yy[take].tolist()))
        unique_valid.update(zip(xx[take&valid].tolist(),yy[take&valid].tolist()))
    # Roundoff tolerance only; preserve raw sums without clipping/renormalizing.
    require((mass>=0).all() and (mass<=1+4*np.finfo(np.float64).eps).all(),'Contributing onehot bbox mass bounded to F64 roundoff')
    return mass,valid,unique,unique_valid


def measure(points,hw,bbox):
    mass,valid,unique,unique_valid=point_weights(points,hw,bbox)
    positive=mass>0;n=len(points);nv=int(valid.sum())
    return dict(samples=n,original_domain_valid_samples=nv,original_domain_invalid_samples=n-nv,
        raw_positive_bbox_overlap_samples=int(positive.sum()),valid_positive_bbox_overlap_samples=int((positive&valid).sum()),
        raw_positive_bbox_overlap_fraction=float(positive.sum()/n),valid_positive_bbox_overlap_fraction_over_valid=float((positive&valid).sum()/nv) if nv else None,
        raw_bbox_contributing_weight_sum=float(np.sum(mass,dtype=np.float64)),raw_bbox_contributing_weight_mean_over_all=float(np.sum(mass,dtype=np.float64)/n),
        valid_bbox_contributing_weight_sum=float(np.sum(mass[valid],dtype=np.float64)),
        valid_bbox_contributing_weight_mean_over_valid=float(np.sum(mass[valid],dtype=np.float64)/nv) if nv else None,
        max_bbox_contributing_weight=float(mass.max()) if n else None,
        unique_positive_raw_bbox_pixels=len(unique),unique_positive_raw_bbox_pixels_at_valid_samples=len(unique_valid),
        any_raw_positive_overlap=bool(positive.any()),any_valid_positive_overlap=bool((positive&valid).any()))


def region_geometry(region,frame):
    if region['region_status']!='SUPPORTED' or region['original_fullRGB_bbox_xyxy'] is None:
        return dict(geometry_status='UNKNOWN_REGION_NO_BBOX',geometry=None)
    k=region['native_slot'];q=frame['query_xy'][k];L=frame['L'][k]
    if q is None or L is None or not np.isfinite(np.asarray(q,np.float64)).all() or not math.isfinite(float(L)) or L<=0:
        return dict(geometry_status='UNKNOWN_ORIGINAL_Q_L',geometry=None)
    sample=grid(q,L);bbox=region['original_fullRGB_bbox_xyxy'];hw=frame['raw_hw']
    return dict(geometry_status='COMPUTED_COARSE_BOX_ONLY',geometry=dict(
        local96=measure(sample.reshape(-1,2),hw,bbox),central32=measure(sample[32:64,32:64].reshape(-1,2),hw,bbox),
        center4=measure(sample[47:49,47:49].reshape(-1,2),hw,bbox)))


def inventory():
    inp=read(INPUT);regions=lines(REGIONS);proofs=lines(PROOFS)
    require(inp['regions_count']==len(regions)==22 and inp['attempts_count']==len(inp['attempts'])==len(proofs)==66,'All22/66 input metadata')
    regionmap={r['case_id']:r for r in regions};frames={f['frame_index']:f for f in inp['frames']};proofmap={r['pair_id']:r for r in proofs}
    require(len(regionmap)==22 and len(proofmap)==66 and len(frames)==21,'Original unique cases/attempts/frames')
    for a in inp['attempts']:
        r=regionmap[a['case_id']];p=proofmap[a['pair_id']];f=frames[a['frame_index']];k=a['native_slot']
        require(a['region']==p['region']==r and a['case_id']==p['case_id'] and a['native_slot']==p['native_slot']==r['native_slot'],'Exact original case/region binding')
        require(all(name in p for name in ('PASS','constructed','reason','variants')),'Every success/failure/UNKNOWN proof has explicit original status')
        require(r['split']==f['split']=='SOURCE_DEVELOPMENT' and r['frame_index']==f['frame_index'],'TRAIN only, no new selection')
        require(r['query_id']==f['query_id'][k] and r['query_xy']==f['query_xy'][k] and r['L']==f['L'][k] and r['global_index']==f['global_index'][k],'Original selected q/L/native/query immutable')
        require('integer half-open' in r['bbox_convention'] and r['new_native_target']==r['new_quality_target']==-1 and r['training_admitted'] is False,'Original weak bbox/authority unchanged')
        if r['original_fullRGB_bbox_xyxy'] is not None:
            d=r['diagnostic_bbox_xyxy'];o=r['diagnostic_raw_xyxy_offset'];require(r['original_fullRGB_bbox_xyxy']==[d[0]+o[0],d[1]+o[1],d[2]+o[0],d[3]+o[1]],'Original diagnostic->raw translation only')
    return inp,regions,frames,proofmap


def invented():
    checks=[]
    p=np.array([[.25,.5]],np.float64);mass,valid,u,uv=point_weights(p,[4,4],[0,0,1,1])
    require(mass.tolist()==[.375] and valid.tolist()==[True] and u=={(0,0)} and uv==u,'Exact fractional fourtap bbox weight')
    checks.append('Fractional_F64_four_positive_taps_exact_expected_weight')
    mass,valid,u,uv=point_weights(np.array([[1.,1.]],np.float64),[4,4],[2,1,3,2]);require(mass.tolist()==[0.] and not u,'Zero-weight adjacent tap excluded')
    checks.append('Zero_weight_boundary_neighbors_do_not_create_bbox_overlap')
    mass,_,u,_=point_weights(np.array([[1.,1.]],np.float64),[4,4],[1,1,2,2]);require(mass.tolist()==[1.] and u=={(1,1)},'Integer exact cell')
    checks.append('Positive_integer_pixel_and_halfopen_bbox_boundary')
    sample_points=np.array([[1.123456789,1.987654321],[2.314159265,2.718281828]],np.float64)
    mass,_,_,_=point_weights(sample_points,[8,8],[0,0,8,8]);require(np.allclose(mass,1,rtol=0,atol=4*np.finfo(np.float64).eps),'Fractional wholebox sum without renormalization')
    checks.append('Fractional_all_four_taps_sum_one_with_FP64_roundoff_only_no_clipping')
    mass,valid,u,uv=point_weights(np.array([[-.6,.5],[-10.,.5]],np.float64),[4,4],[0,0,1,1])
    require(np.array_equal(mass,[.2,0.]) and valid.tolist()==[False,False] and u=={(0,0)} and not uv,'OOB raw partialtap vsdomain validity separate')
    checks.append('Partial_and_fully_OOB_keep_raw_footprint_and_original_sampling_domain_separate')
    sample=grid([47.5,47.5],96);m=measure(sample.reshape(-1,2),[96,96],[0,0,96,96])
    require(m['samples']==9216 and m['raw_positive_bbox_overlap_samples']==9216 and m['original_domain_valid_samples']==9216 and m['raw_bbox_contributing_weight_sum']==9216,'Full96 positive fixture')
    require(sample.dtype==np.float64 and np.array_equal(sample[47,47]+sample[48,48],[95.,95.]),'Exact pixelcenter q47.5')
    checks.append('Actual96_F64_pixelcenter_grid_and_full_positive_weight_sum')
    require(measure(sample[32:64,32:64].reshape(-1,2),[96,96],[0,0,96,96])['samples']==1024 and measure(sample[47:49,47:49].reshape(-1,2),[96,96],[0,0,96,96])['samples']==4,'Exact saved subgrids')
    checks.append('Central32_and_center4_are_original_indices_no_reference_recrop')
    require(not measure(sample.reshape(-1,2),[96,96],[100,100,110,110])['any_raw_positive_overlap'],'Zero-overlap disjoint bbox')
    checks.append('Disjoint_bbox_zero_overlap_not_imputed_part_visibility_label')
    require(region_geometry(dict(region_status='UNKNOWN',original_fullRGB_bbox_xyxy=None),{})==dict(geometry_status='UNKNOWN_REGION_NO_BBOX',geometry=None),'Unknown bbox staysnull')
    checks.append('Unknown_region_preserved_null_not_fake_zero_geometry')
    f=dict(query_xy=[[47.5,47.5]],L=[96.],raw_hw=[96,96],native_continuous_valid=[False])
    r=dict(region_status='SUPPORTED',original_fullRGB_bbox_xyxy=[0,0,96,96],native_slot=0)
    require(region_geometry(r,f)['geometry']['local96']['samples']==9216 and f['native_continuous_valid']==[False],'Original inputU not filtered/corrected')
    checks.append('Original_inputU_is_separate_metadata_and_never_changes_q_or_drops_row')
    return dict(completed=True,PASS=True,invented_only=True,count=len(checks),checks=checks,actual_region_intersections_computed=0,pixel_NPZ_model_label_body_reads=0)


def prepare():
    require(platform.python_version()=='3.10.20' and np.__version__=='1.26.4','Pinned CPU F64 runtime')
    inp,regions,frames,proofmap=inventory()  # Identity/count joins only, no grid or actual overlap evaluation.
    rules=read(RULES);require(rules['actual_intersections_computed_before_rule'] is False,'Prospective rules first')
    write(CHECKS,invented())
    p=dict(completed=True,prepared_only=True,code=bind(__file__),rules=bind(RULES),checks=bind(CHECKS),
        inputs=dict(fixed22_and66=bind(INPUT),regions22=bind(REGIONS),saved66proofs=bind(PROOFS)),
        population=dict(regions22=len(regions),attempts66=len(inp['attempts']),frames21=len(frames)),output=str(OUT),
        comparison_only='Exact original metadata/proof comparison; no remeasured pixels or labels, no selection/admission.',actual_grid_intersections_computed=0,automatic_retry=False)
    write(PROTOCOL,p)
    template=H/'ROOT_ACTUAL_REGION_INTERSECTION_ENTRY_TEMPLATE_APPROVED_FALSE.json'
    write(template,dict(approved_metadata_only_region_intersections=False,source_TRAIN_only=True,no_pixels_models_labels_or_selection=True,
        code=bind(__file__),protocol=bind(PROTOCOL),output=str(OUT)))
    write(PREP,dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),entry_template=bind(template),actual_grid_intersections_computed=0))
    print(json.dumps(dict(FINAL_READY=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),preparation=bind(PREP))))


def run(entry):
    e=read(entry);require(e.get('approved_metadata_only_region_intersections') is True and e.get('source_TRAIN_only') is True
        and e.get('no_pixels_models_labels_or_selection') is True,'Separate root actual metadata entry')
    require(e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL) and e['output']==str(OUT),'Frozen code/protocol/output')
    p=read(PROTOCOL);check(p['rules']);require(read(check(p['checks']))['PASS'] and not OUT.exists(),'Checks and single new output')
    for b in p['inputs'].values():check(b)
    require(platform.python_version()=='3.10.20' and np.__version__=='1.26.4','Pinned F64 runtime')
    OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry),code=bind(__file__),protocol=bind(PROTOCOL),actual_pixel_NPZ_model_reads=False))
    try:
        inp,regions,frames,proofmap=inventory();geometry={r['case_id']:region_geometry(r,frames[r['frame_index']]) for r in regions};rows=[]
        for a in inp['attempts']:
            r=a['region'];k=a['native_slot'];f=frames[a['frame_index']];proof=proofmap[a['pair_id']];detail=proof.get('proof_detail',{})
            rows.append(dict(pair_id=a['pair_id'],case_id=a['case_id'],frame_index=a['frame_index'],native_slot=k,query_id=r['query_id'],
                original_global_index=r['global_index'],query_xy=r['query_xy'],L=r['L'],raw_hw=f['raw_hw'],split=r['split'],
                original_query_available=f['native_continuous_valid'][k],original_crop_constructed=f['feature_constructed'][k],region_status=r['region_status'],
                original_fullRGB_bbox_xyxy=r['original_fullRGB_bbox_xyxy'],bbox_semantics=r['region_semantics'],**geometry[a['case_id']],
                original_pixel_proof=dict(PASS=proof['PASS'],constructed=proof['constructed'],reason=proof['reason'],variants=proof['variants'],
                    selected_view_order=detail.get('selected_preprocess_view_order'),cover_selected_view_changed=detail.get('cover_selected_view_changed'),
                    sham_selected_view_changed=detail.get('sham_selected_view_changed')),
                new_support_owner_native_quality_target_created=False,training_admission=False))
        require(len(rows)==66 and len({r['case_id'] for r in rows})==22,'Keep every original attempt and unknown')
        rp=OUT/'ALL66_ORIGINAL_GRID_COARSE_REGION_INTERSECTIONS.jsonl'
        with rp.open('x') as handle:
            for row in rows:handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        summary=dict(attempts66=len(rows),regions22=len(regions),geometry_status=dict(Counter(r['geometry_status'] for r in rows)),
            original_region_status=dict(Counter(r['region_status'] for r in rows)),original_proof_PASS=dict(Counter(str(r['original_pixel_proof']['PASS']) for r in rows)),
            unique_case_status=dict(Counter(g['geometry_status'] for g in geometry.values())),grids={})
        for name in ('local96','central32','center4'):
            known=[r for r in rows if r['geometry'] is not None]
            summary['grids'][name]=dict(computed_attempts=len(known),unknown_attempts=len(rows)-len(known),
                positive_raw_bbox_overlap_attempts=sum(r['geometry'][name]['any_raw_positive_overlap'] for r in known),
                positive_valid_bbox_overlap_attempts=sum(r['geometry'][name]['any_valid_positive_overlap'] for r in known),
                zero_raw_bbox_overlap_pair_ids=[r['pair_id'] for r in known if not r['geometry'][name]['any_raw_positive_overlap']])
        sp=OUT/'SUMMARY.json';write(sp,summary)
        for b in p['inputs'].values():check(b)
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry),inputs=p['inputs'],
            outputs=dict(all66=bind(rp),summary=bind(sp)),all_original_attempts_unknown_and_failures_retained=True,source_TRAIN_only=True,
            image_NPZ_model_forward_fit_label_or_admission_changes=0,interpretation=read(RULES)['interpretation'],selected_model=None,training_admission=False,automatic_retry=False))
        print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'))))
    except BaseException:
        write(OUT/'FAILED.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();m=ap.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--entry',type=Path)
    args=ap.parse_args();prepare() if args.prepare else run(args.entry)
