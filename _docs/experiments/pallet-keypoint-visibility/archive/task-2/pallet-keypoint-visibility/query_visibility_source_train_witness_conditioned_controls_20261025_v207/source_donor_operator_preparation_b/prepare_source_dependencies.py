"""Bounded metadata/code readiness. Never stat/hash/decode binary assets."""
from pathlib import Path
import csv,hashlib,io,json,math,os,stat
H=Path(__file__).parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_PROSPECTIVE_SOURCE_DONOR_AND_OPERATOR_CODE_PREPARATION_SCOPE.json'
V204=P/'query_visibility_source_train_intervention_query_alignment_20261025_v204/public_source_coverage_b'
V194=P/'query_visibility_source_group_split_direct_end_training_20261024_v194/direct_cache_preparation_a'

def rb(path):
    p=Path(path)
    assert p.is_absolute() and P in p.parents and '..'not in p.parts and 'v112'not in str(p).casefold()
    assert p.suffix in {'.py','.json','.jsonl','.csv'} # Before ANY filesystem access.
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for x in p.parts[1:-1]:
            n=os.open(x,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    assert stat.S_ISREG(os.fstat(n).st_mode)
    with os.fdopen(n,'rb')as f:return f.read()

def bind(p):return dict(path=str(p),sha256=hashlib.sha256(rb(p)).hexdigest())
def get(b):
    z=rb(b['path']);assert hashlib.sha256(z).hexdigest()==b['sha256'];return json.loads(z)
def put(n,d):
    with (H/n).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def inert(b):
    # Asset names and recorded hashes ONLY; deliberately no filesystem operation.
    p=Path(b['path']);assert p.is_absolute()and '..'not in p.parts and 'v112'not in str(p).casefold()and len(b['sha256'])==64
    return b

def witness_taps(q,L,raw_hw,roi,available=True):
    """Scalar grid diagnostic, not a replacement actual RGB sampler."""
    if not available or q is None or L is None:return {'status':'ORIGINAL_INPUT_U','taps':[],'valid_outputs':0,'invalid_outputs':0}
    assert all(math.isfinite(x)for x in q)and math.isfinite(L)and L>0
    if roi is None:return {'status':'UNKNOWN_WITNESS','taps':[],'valid_outputs':0,'invalid_outputs':0}
    assert len(roi)==4 and all(type(x)is int for x in roi)
    x0,y0,x1,y1=roi;assert 0<=x0<=x1<=95 and 0<=y0<=y1<=95
    h,w=raw_hw;taps=set();valid=invalid=0
    for j in range(y0,y1+1):
        for i in range(x0,x1+1):
            # Exact V90 expression order, F64 scalar arithmetic.
            x=q[0]+((float(i)+.5)/96-.5)*L;y=q[1]+((float(j)+.5)/96-.5)*L
            if not(-.5<=x<w-.5 and -.5<=y<h-.5):invalid+=1;continue
            valid+=1;ix=math.floor(x);iy=math.floor(y);dx=x-ix;dy=y-iy
            for ax,ay,weight in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
                xx,yy=ix+ax,iy+ay
                if weight>0 and 0<=xx<w and 0<=yy<h:taps.add((xx,yy))
    return dict(status='SUPPORTED_GRID_DIAGNOSTIC'if valid else'NO_VALID_WITNESS_OUTPUT',taps=sorted(map(list,taps),key=lambda z:(z[1],z[0])),valid_outputs=valid,invalid_outputs=invalid)

def placed(alpha,origin,valid):
    p={(x+origin[0],y+origin[1])for x,y in alpha}
    return p if p<=valid else None

def tests():
    checks=[]
    def t(n,v):assert v,n;checks.append(n)
    t('Identity_one_positive_tap',witness_taps([47.5,47.5],96,[96,96],[0,0,0,0])['taps']==[[0,0]])
    t('Fractional_four_positive_taps',witness_taps([48.,48.],96,[100,100],[0,0,0,0])['taps']==[[0,0],[1,0],[0,1],[1,1]])
    t('OriginalU_not_E',witness_taps(None,None,[96,96],[0,0,1,1],False)['status']=='ORIGINAL_INPUT_U')
    t('UNKNOWN_no_center_imputation',witness_taps([47.5,47.5],96,[96,96],None)['taps']==[])
    r=witness_taps([-50.,-50.],96,[96,96],[0,0,1,1]);t('Invalid_outputs_zero_RGB_contributors',r['valid_outputs']==0 and r['invalid_outputs']==4 and not r['taps'])
    alpha={(x,y)for x in range(3)for y in range(3)};valid={(x,y)for x in range(8)for y in range(8)};witness={(1,1),(2,2)}
    a=placed(alpha,(1,1),valid);b=placed(alpha,(5,5),valid)
    t('Cover_sham_same_mass_disjoint_support',witness<=a and not witness&b and len(a)==len(b)==9)
    t('Clipping_padding_not_legal',placed(alpha,(6,6),valid)is None)
    t('Coarse_full_witness_no_legal_sham_no_shrink',not any(not placed(alpha,(x,y),valid)&valid for y in range(6)for x in range(6)))
    return dict(completed=True,PASS=True,invented_only=True,count=len(checks),checks=checks,source_images_arrays_or_ROIs_read=0)

def main():
    assert bind(SCOPE)['sha256']=='26225fa9314f842e13c7400043ecf300731580cd625f8f1eaf7c906b624297b5';get(bind(SCOPE))
    ip=V204/'YCB_TRAIN_DONOR_INVENTORY.csv';inventory=list(csv.DictReader(io.StringIO(rb(ip).decode())));assert len(inventory)==59
    dp_path=V194/'PROTOCOL_BEFORE_EXPANDED_SOURCE_CONTROL_CACHE_BODIES.json';dp=json.loads(rb(dp_path))
    rc=get(dp['source_controls']['REAL_SEMANTIC_END_WEAK']);rp=get(rc['protocol']);dc=get(rp['donor_proof_complete'])
    assert dc['completed']and dc['PASS']and dc['attempted']==dc['passed']==59 and dc['records']==rp['donor_records']
    z=rb(rp['donor_records']['path']);assert hashlib.sha256(z).hexdigest()==rp['donor_records']['sha256']
    rr=[json.loads(x)for x in z.splitlines()if x.strip()];train=[r for r in rr if r['split']=='TRAIN']
    assert len(train)==len(rr)==59 and all(r['PASS']for r in train)
    by={r['donor_id']:r for r in train};assert set(by)=={r['donor_id']for r in inventory}
    donors=[]
    for item in inventory:
        r=by[item['donor_id']];assert json.loads(item['source_RGB_binding'])==r['original_RGB'];idx=int(item['instance_id']);mi=r['complete_mask_inventory'];chosen=[x for x in mi['instances']if x['instance_id']==idx]
        assert len(chosen)==1 and chosen[0]['object_id']==int(item['object_id'])
        donors.append(dict(donor_id=r['donor_id'],role='YCB_V_TRAIN_FOREIGN_DONOR_NOT_PALLET_TARGET',scene_id=int(item['scene_id']),frame_id=int(item['frame_id']),instance_id=idx,object_id=int(item['object_id']),compact_RGBA_cutout=inert(r['compact_RGBA_cutout']),native_alpha_proof_reference=inert(r['native_alpha_proof']),original_RGB_reference=inert(r['original_RGB']),selected_visible_mask_reference=inert(chosen[0]['mask']),other_visible_masks_reference=[inert(x['mask'])for x in mi['instances']if x['instance_id']!=idx],scene_GT_reference_inert=mi['scene_gt'],existing_array_proof=r['array_proof'],runtime_contract=r['runtime_cutout_contract'],fresh_asset_stat_hash_header_decode=False))
    with (H/'SOURCE_TRAIN59_DONOR_BINDINGS.jsonl').open('x')as f:
        for r in donors:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    sc=get(dp['source_controls']['SYNTH_FINITE_END']);sp=get(sc['protocol'])
    codes=dict(V194_cache=dp['cache_code'],V194_common=dp['source_bindings']['common_code'],V193_compositor=rp['code'],V193_protocol=rc['protocol'],V161_pixel_helper=rp['original_pixel_helper'],V161_proof_code=dc['code'],V161_proof_protocol=dc['protocol'],V90_original_sampler=rp['original_RGB_sampler'],SYNTH_producer=sp['code'],SYNTH_candidate=sp['candidate'],SYNTH_registered_geometry=sp['sources']['source_constructive_code'],SYNTH_broad_support_operator=sp['sources']['source_end_operator'])
    for b in codes.values():assert bind(Path(b['path']))==b
    synth=dict(code_pointers={k:v for k,v in codes.items()if k.startswith('SYNTH')},configuration_metadata_pointer=sp['sources']['source_configs'],original_input_plan_pointer=sp['sources']['source_TRAIN_input_plan'],fields=['cfg.permCF_to_canonical[native_slot]','native_anchor3D[canonical]','native_primary_part_id','native_witness_points[canonical][0]','camera_K/R/t','native8_CF_xy','projected_reference_xy'],point_vs_region='V127 classify_hits: point D at expected part/reference, E if closer owner999, S if closer target front incl samepart, else U. region_teacher adds activeface/edge/inframe conditions and is distinct from anchorpoint visibility.',projected_reference='region_teacher record project[0] is native_witness_points[canonical][0] projection. Diagnostic/teacher reference only, never replace initializer q/L.',no_promotion='V189 inner0.2 finite_end and V180 visible/foreign support keep native target -1. Never copy them or cropcenter owner to nativepoint truth.',future_role_boundary='V189 original TRAIN128 predates V194 component split. Filter immutable V194 TRAIN before any future buffer read; no componentHOLDOUT/score access.',arrays_GT_or_projection_executed=False)
    put('SOURCE_SYNTH_NATIVE_CODE_POINTERS.json',synth)
    operator=dict(existing_helper=rp['original_pixel_helper'],functions=['erode1','transformed_cutout','opaque_anchor','paste','materialize','verify_before'],alpha='Compact bool core=erode1(selected_visible & ~union(other_visible)), nativebbox scale1.0. Controlled paste opacity, not measured optical opacity/nativegold.',transform='RGB Pillow BILINEAR; alpha NEAREST; floor(dim*scale+.5); false-outside3x3 erode1; retain holes.',anchor='Padded false-outside EDT first row-major argmax.',paste='Hard-alpha RGB copy over target fullRGB, outside alpha exact. Helper permits Y clipping: future equal-mass proof must reject any illegal/clipped pair footprint.',sampler='V90 F64 u=(arange+.5)/96-.5 then q+uL; fourtap bilinear; originalvalid domain; uint8 floor(clip(acc)+.5). No nearest RGB substitution.',future_guards=['Freeze new fixedcase-to-existing59 donor assignment and one transform/placement policy before source PNG/cutout bytes; no model or label outcome search.','Recommended existing fullRGB-first paste then original sampler. Lift raw96 weak feature outputs through original positive valid fourtap contributors to check fullimage opaque obligations.','COVER both retained weak witness regions/enclosing obligation; same transformed RGB/alpha and positive pasted mass in SHAM; legal placement, witness alpha0 and actual witness RGB unchanged SHAM, actual changed COVER. Preserve q/L/native/U/valid and all original source bytes outside respective alpha.','Large coarse union or padding may make legal samefootprint SHAM impossible. Preserve infeasible/UNKNOWN without shrinking or moving q/ROI, filling alpha holes, changing factual D or forcing native E.','Separate root pixel entry, actual invariants and fixedcase visual evidence required; no fit/E/nativegold admission here.'],status='DEPENDENCIES_READY_ONLY_NO_ACTUAL_FEASIBILITY_OR_PLACEMENT_SELECTED',actual_compositor_or_labels_or_fit=False)
    put('OPERATOR_READINESS.json',operator);put('INVENTED_GEOMETRY_CHECKS.json',tests())
    readiness=dict(completed=True,PASS=True,scope=bind(SCOPE),code=bind(Path(__file__)),metadata_chain=[bind(V194/'build_expanded_direct_control_cache.py'),bind(dp_path),dp['source_controls']['REAL_SEMANTIC_END_WEAK'],rc['protocol'],rp['donor_proof_complete'],rp['donor_records']],source_TRAIN_inventory=bind(ip),donors=bind(H/'SOURCE_TRAIN59_DONOR_BINDINGS.jsonl'),donor_count=59,unique_RGBs=len({r['original_RGB_reference']['sha256']for r in donors}),source_scenes=len({r['scene_id']for r in donors}),object_IDs=sorted({r['object_id']for r in donors}),code_bindings=codes,source_synth=bind(H/'SOURCE_SYNTH_NATIVE_CODE_POINTERS.json'),operator=bind(H/'OPERATOR_READINESS.json'),checks=bind(H/'INVENTED_GEOMETRY_CHECKS.json'),source_asset_bindings_inert=True,new_asset_stat_hash_header_decode_GT_ROI_or_model_reads=0,new_E_fit_or_admission=False,goal_solved=False)
    put('DEPENDENCY_READINESS.json',readiness)
    with (H/'KO.md').open('x')as f:f.write('V207 기존 YCB-V TRAIN 공여자59 재사용 경로를 확인했다. V194 protocol→V193 compositor protocol→V161 donor proof→V95 compact RGBA이다. MR6D는 pallet target, YCB-V는 foreign donor다. 원RGB/selected·other mask와compact rgb/alpha의 기록된경로·SHA만 보존했고 binary/GT stat·hash·header·decode0이다. 새 dataset 또는 완벽한 physicalgold 인증을 요구하지 않았다.\n\n기존RGB bilinear/alpha nearest+erode/hardpaste 및V90원F64 sampler를 재사용할수있다. 후속실행은동일donor/변환/alpha mass, 양쪽ROI COVER·SHAM alpha0와실제RGB불변, valid/qL/native/U를별도입증해야한다. coarse union/padding으로legal SHAM이없으면구성불가를보존하고q·ROI축소이동이나factualD→E변환을하지않는다.\n\nSOURCE SYNTH의 native→canonical/anchor/witness0/투영참조·point D/E999/S/U와regionteacher 코드를포인터로남겼다. finite-end/support/owner를native gold로복사하지않으며, 향후sourcebuffer는V194 TRAINrole부터적용해야한다. 실제공여자·placement선정/개입/label/fit은0이고가상기하8검사만준비했다.\n')
    put('FINAL_READY.json',dict(completed=True,PASS=True,prepared_code_metadata_only=True,code=bind(Path(__file__)),readiness=bind(H/'DEPENDENCY_READINESS.json'),donors=bind(H/'SOURCE_TRAIN59_DONOR_BINDINGS.jsonl'),operator=bind(H/'OPERATOR_READINESS.json'),source_synth=bind(H/'SOURCE_SYNTH_NATIVE_CODE_POINTERS.json'),checks=bind(H/'INVENTED_GEOMETRY_CHECKS.json'),KO=bind(H/'KO.md'),actual_source_asset_stat_hash_header_decode_or_new_ROI_reads=0,actual_intervention_E_fit_or_admission=False))
    print(json.dumps(dict(completed=True,PASS=True,ready=bind(H/'FINAL_READY.json'),donors=59,geometry_checks=8)))

if __name__=='__main__':main()
