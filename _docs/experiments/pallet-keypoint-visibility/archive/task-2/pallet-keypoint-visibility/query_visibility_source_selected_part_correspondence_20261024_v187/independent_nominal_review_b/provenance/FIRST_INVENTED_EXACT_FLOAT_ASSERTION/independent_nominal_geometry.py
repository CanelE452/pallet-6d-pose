"""Independent scalar geometry oracle; no A/V74 code or AST is imported.

Fixed signs/faces are literal V74 contract constants. Vertical normals are
analytic axis normals; rear counterparts flip the chosen front-face axis.
Camera inversion is a separate scalar pivoted elimination, not R transpose.
"""
import copy
import math

SIGNS=((-1,-1,1),(1,-1,1),(1,-1,-1),(-1,-1,-1),(-1,1,1),(1,1,1),(1,1,-1),(-1,1,-1))
FACES=((0,1,2,3),(4,5,6,7),(0,4,7,3),(1,5,6,2))
FIXED=tuple((sx*400.,sy*600.,sz*72.) for sx,sy,sz in SIGNS)
NAMES=('front_top_left','front_top_right','front_bottom_right','front_bottom_left','rear_top_left','rear_top_right','rear_bottom_right','rear_bottom_left')

def need(ok,message):
    if not ok:raise ValueError(message)

def matrix3(value):
    if isinstance(value,list) and len(value)==9 and all(type(v) in (int,float) for v in value):
        rows=[value[0:3],value[3:6],value[6:9]]
    elif isinstance(value,list) and len(value)==3 and all(isinstance(r,list) and len(r)==3 for r in value):rows=value
    else:raise ValueError('Invalid3x3 provided notation')
    need(all(type(x) in (int,float) and math.isfinite(x) for r in rows for x in r),'Finite matrix')
    return [[float(x) for x in r] for r in rows]

def solve3(A,b):
    aug=[list(row)+[float(v)] for row,v in zip(A,b)]
    for col in range(3):
        pivot=max(range(col,3),key=lambda row:abs(aug[row][col]))
        need(aug[pivot][col]!=0,'Singular provided R')
        aug[col],aug[pivot]=aug[pivot],aug[col]
        factor=aug[col][col];aug[col]=[x/factor for x in aug[col]]
        for row in range(3):
            if row==col:continue
            factor=aug[row][col]
            aug[row]=[x-factor*y for x,y in zip(aug[row],aug[col])]
    return [aug[row][3] for row in range(3)]

def dot(a,b):return sum(x*y for x,y in zip(a,b))

def order_from_camera(projected,camera):
    length=math.sqrt(dot(camera,camera));need(length>=1e-12,'Camera at model center')
    # Analytic normals for the literal axis-aligned nominal box, not cross-products.
    scores=[-camera[1]/length,camera[1]/length,-camera[0]/length,camera[0]/length]
    face=max(range(4),key=lambda i:scores[i])  # listed first face wins exact ties
    top=[i for i in FACES[face] if FIXED[i][2]>0]
    top.sort(key=lambda i:projected[i][0])
    lookup={p:i for i,p in enumerate(FIXED)}
    bottom=lambda i:lookup[(FIXED[i][0],FIXED[i][1],-FIXED[i][2])]
    front=[top[0],top[1],bottom(top[1]),bottom(top[0])]
    axis=1 if face<2 else 0
    rear=[]
    for i in front:
        p=list(FIXED[i]);p[axis]=-p[axis];rear.append(lookup[tuple(p)])
    return front+rear,scores,face

def project_instance(inst,hw):
    common={k:inst[k] for k in ('frame_index','scene','frame_key','gt_instance_index','obj_id')}
    provided={k:copy.deepcopy(inst[k]) for k in ('provided_cam_K','provided_cam_R_m2c','provided_cam_t_m2c_mm')}
    result=dict(common,provided_values=provided,raw_hw=list(hw),status='UNKNOWN_INVALID_PROVIDED_GEOMETRY',
        dynamic_to_fixed_permutation=None,normal_face_scores=None,exact_tie_original_writer_policy_uncertified=None,references=[])
    try:
        K=matrix3(provided['provided_cam_K']);R=matrix3(provided['provided_cam_R_m2c']);t=provided['provided_cam_t_m2c_mm']
        need(isinstance(t,list) and len(t)==3 and all(type(x) in (int,float) and math.isfinite(x) for x in t),'Finite provided t3')
        need(K[2]==[0.,0.,1.],'Original pinhole K')
        camera_points=[[dot(row,p)+tr for row,tr in zip(R,t)] for p in FIXED]
        homogeneous=[[dot(row,p) for row in K] for p in camera_points]
        need(all(math.isfinite(x) for p in homogeneous for x in p) and all(p[2]!=0 for p in homogeneous),'Nonzero finite projection denominator')
        xy=[[p[0]/p[2],p[1]/p[2]] for p in homogeneous]
        camera=solve3(R,[-x for x in t]);permutation,scores,face=order_from_camera(xy,camera)
        need(sorted(permutation)==list(range(8)),'Native order is a bijection')
        sorted_scores=sorted(scores);margin=sorted_scores[-1]-sorted_scores[-2]
        separation=xy[permutation[1]][0]-xy[permutation[0]][0]
        uncertain=margin<=1e-10 or separation<=0
        orthogonal=max(abs(sum(R[k][i]*R[k][j] for k in range(3))-(1. if i==j else 0.)) for i in range(3) for j in range(3))
        determinant=(R[0][0]*(R[1][1]*R[2][2]-R[1][2]*R[2][1])-R[0][1]*(R[1][0]*R[2][2]-R[1][2]*R[2][0])+R[0][2]*(R[1][0]*R[2][1]-R[1][1]*R[2][0]))
        result.update(status='NOMINAL_COMPUTED_ROLE_UNCERTAIN' if uncertain else 'NOMINAL_COMPUTED',
            camera_in_model_exact_linear_inverse=camera,dynamic_to_fixed_permutation=permutation,normal_face_scores=scores,
            front_face_index=face,front_runner_margin=margin,front_top_imageX_separation_px=separation,
            exact_tie_original_writer_policy_uncertified=bool(margin<=1e-10),R_orthogonality_max_abs=orthogonal,R_determinant=determinant,geometry_error=None)
        for native,canonical in enumerate(permutation):
            x,y=xy[canonical];z=camera_points[canonical][2];h,w=hw
            result['references'].append(dict(common,native_slot=native,native_name=NAMES[native],canonical_fixed_index=canonical,
                nominal_model_XYZ_mm=list(FIXED[canonical]),nominal_camera_XYZ_mm=camera_points[canonical],nominal_reference_xy=[x,y],
                positive_camera_Z=bool(z>0),continuous_reference_inframe=bool(z>0 and 0<=x<w and 0<=y<h),nominal_role_status=result['status'],
                semantic_state='UNKNOWN',native_visibility_target=-1,point_quality_target=-1,training_admission=False,physical_coordinate_gold=False))
    except (ValueError,ZeroDivisionError,OverflowError):
        result['geometry_error']='INDEPENDENT_INVALID_PROVIDED_GEOMETRY'
        result['references']=[dict(common,native_slot=native,native_name=NAMES[native],canonical_fixed_index=None,nominal_model_XYZ_mm=None,
            nominal_camera_XYZ_mm=None,nominal_reference_xy=None,positive_camera_Z=None,continuous_reference_inframe=None,
            nominal_role_status=result['status'],semantic_state='UNKNOWN',native_visibility_target=-1,point_quality_target=-1,training_admission=False,physical_coordinate_gold=False) for native in range(8)]
    return result

def footprint_or_unknown(reference,q,L,hw):
    unknown=dict(valid_nominal_discrepancy=False,reason='MISSING_NONFINITE_SENTINEL_Q_OR_INVALID_L',delta_xy_px=None,nominal_distance_px=None,
        delta_xy_over_L=None,nominal_distance_over_L=None,reference_crop_index_xy=None,inside_outer_crop_extent=None,
        inside_sample_center_rectangle=None,reference_in_original_sampler_valid_domain=None)
    def finite2(a):return isinstance(a,(list,tuple)) and len(a)==2 and all(type(x) in (int,float) and math.isfinite(x) for x in a)
    if not finite2(q) or list(q)==[-1.,-1.] or type(L) not in (int,float) or not math.isfinite(L) or L<=0:return unknown
    if not finite2(reference):return dict(unknown,reason='NONFINITE_OR_UNAVAILABLE_NOMINAL_REFERENCE')
    delta=[reference[i]-q[i] for i in range(2)];distance=math.hypot(delta[0],delta[1])
    crop=[96.*d/L+47.5 for d in delta];h,w=hw
    return dict(valid_nominal_discrepancy=True,reason='NOMINAL_DIAGNOSTIC_NOT_PHYSICAL_COORDINATE_GOLD',delta_xy_px=delta,
        nominal_distance_px=distance,delta_xy_over_L=[d/L for d in delta],nominal_distance_over_L=distance/L,
        reference_crop_index_xy=crop,inside_outer_crop_extent=all(-.5<=x<95.5 for x in crop),
        inside_sample_center_rectangle=all(0.<=x<=95. for x in crop),
        reference_in_original_sampler_valid_domain=bool(-.5<=reference[0]<w-.5 and -.5<=reference[1]<h-.5))

def selected_query_diagnostic(row,association,nominal,hw):
    state='UNKNOWN_NO_UNIQUE_OBJ1_WEAK_ASSOCIATION';ref=None
    if association['status']=='UNIQUE_POSITIVE_MAX_BBOX_IOU_WEAK_IDENTITY' and association['selected_obj_id']==1:
        need(nominal is not None and nominal['gt_instance_index']==association['selected_gt_instance_index'],'Exact original GT-list selected lookup')
        ref=nominal['references'][row['native_slot']]
        state='WEAK_INSTANCE_NOMINAL_REFERENCE' if nominal['status']=='NOMINAL_COMPUTED' else 'UNKNOWN_NOMINAL_ROLE_OR_GEOMETRY'
        if ref['positive_camera_Z'] is not True:state='UNKNOWN_NONPOSITIVE_OR_INVALID_REFERENCE_Z'
    return dict(row,nominal_correspondence=dict(status=state,weak_association_status=association['status'],
        selected_gt_instance_index=association['selected_gt_instance_index'],selected_obj_id=association['selected_obj_id'],
        selected_instance_identity_certified=False,nominal_reference=ref,
        nominal_discrepancy_and_crop_diagnostic=footprint_or_unknown(None if ref is None else ref['nominal_reference_xy'],row['original_query_xy'],row['original_L'],hw),
        original_input_U_preserved=not row['original_query_available'],semantic_state='UNKNOWN',native_visibility_target=-1,point_quality_target=-1,
        training_admission=False,physical_coordinate_gold=False))

def invented_checks():
    checks=[]
    def inst(t=(0.,2000.,2000.),R=None):return dict(frame_index=7,scene='000000',frame_key='9',gt_instance_index=3,obj_id=1,
        provided_cam_K=[800.,0.,640.,0.,800.,360.,0.,0.,1.],provided_cam_R_m2c=R or [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],provided_cam_t_m2c_mm=list(t))
    a=project_instance(inst(),[720,1280]);need(a['dynamic_to_fixed_permutation']==list(range(8)),'Known negativeY order')
    checks.append('Literal_Zup_signs_and_negativeY_dynamic_order')
    b=project_instance(inst((2000.,0.,2000.)),[720,1280]);need(b['dynamic_to_fixed_permutation']==[0,4,7,3,1,5,6,2],'Rear topology preserved')
    checks.append('NegativeX_front_and_rear_flip_axis_not_independent_imageX_sort')
    for ref in a['references']:
        x,y,z=ref['nominal_camera_XYZ_mm'];need(ref['nominal_reference_xy']==[800*x/z+640,800*y/z+360],'Pinhole direct scalar oracle')
    checks.append('Independent_scalar_pinhole_oracle_all8')
    tie=project_instance(inst((2000.,2000.,2000.)),[720,1280]);need(tie['status']=='NOMINAL_COMPUTED_ROLE_UNCERTAIN','Exact side tie uncertain')
    checks.append('Exact_side_tie_first_list_order_retained_role_uncertainty')
    R=[[1.,1e-7,0.],[0.,1.,0.],[0.,0.,1.]];pert=project_instance(inst(R=R),[720,1280]);camera=pert['camera_in_model_exact_linear_inverse']
    need(max(abs(dot(row,camera)+v) for row,v in zip(R,(0.,2000.,2000.)))<1e-10 and camera!=[0.,-2000.,-2000.],'Supplied nonorthogonal solve without transpose')
    checks.append('Scalar_pivot_solve_not_transpose_and_no_R_repair')
    bad=project_instance(inst(R=[[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]),[720,1280]);need(len(bad['references'])==8 and all(r['nominal_reference_xy'] is None for r in bad['references']),'Invalid geometry retains8U')
    checks.append('Invalid_geometry_all8_UNKNOWN_retained')
    behind=project_instance(inst((0.,2000.,-2000.)),[720,1280]);need(all(not r['positive_camera_Z'] and r['semantic_state']=='UNKNOWN' for r in behind['references']),'Behind is not invisible gold')
    checks.append('Negative_cameraZ_preserved_as_numeric_reference_no_visibility_claim')
    f=footprint_or_unknown([12.,20.],[12.,20.],96.,[720,1280]);need(f['reference_crop_index_xy']==[47.5,47.5] and f['nominal_distance_px']==0.,'Center47.5')
    checks.append('Original_q_maps_between_central4_samples')
    need(footprint_or_unknown([-48.,0.],[0.,0.],96.,[720,1280])['inside_outer_crop_extent'] and not footprint_or_unknown([48.,0.],[0.,0.],96.,[720,1280])['inside_outer_crop_extent'],'Halfopen extent')
    checks.append('Outer_halfopen_extent_lower_in_upper_out')
    need(footprint_or_unknown([-47.5,47.5],[0.,0.],96.,[720,1280])['inside_sample_center_rectangle'],'Inclusive center rectangle')
    need(footprint_or_unknown([-.25,0.],[0.,0.],96.,[720,1280])['reference_in_original_sampler_valid_domain'],'Halfpixel source domain')
    checks.append('Sample_centers_and_halfpixel_source_domain_separate')
    for q,L in ((None,1.),([None,None],2.),([-1.,-1.],2.),([0.,0.],None),([0.,0.],0.)):
        need(not footprint_or_unknown([1.,1.],q,L,[720,1280])['valid_nominal_discrepancy'],'Missing qL safe')
    checks.append('Missing_null_sentinel_qL_retains_UNKNOWN')
    row=dict(native_slot=0,original_query_xy=[10.,20.],original_L=96.,original_query_available=False,original_global_index=56,original_input_flags={'continuous_valid':False})
    assoc=dict(status='UNIQUE_POSITIVE_MAX_BBOX_IOU_WEAK_IDENTITY',selected_obj_id=1,selected_gt_instance_index=3)
    joined=selected_query_diagnostic(row,assoc,a,[720,1280]);need(all(joined[k]==v for k,v in row.items()) and joined['nominal_correspondence']['original_input_U_preserved'],'Original U remains')
    shifted=selected_query_diagnostic(dict(row,original_query_xy=[1000.,500.]),assoc,a,[720,1280]);need(shifted['nominal_correspondence']['nominal_reference']==joined['nominal_correspondence']['nominal_reference'],'No qbest permutation')
    checks.append('Original_U_qL_flags_and_q_independent_order')
    for change in (dict(selected_obj_id=7),dict(status='UNKNOWN_EXACT_MAX_IOU_TIE',selected_obj_id=None,selected_gt_instance_index=None)):
        other=selected_query_diagnostic(row,dict(assoc,**change),None,[720,1280]);need(other['nominal_correspondence']['nominal_reference'] is None,'No fallback')
    checks.append('Other_object_or_tie_no_obj1_fallback_reference')
    need(all(r['semantic_state']=='UNKNOWN' and r['native_visibility_target']==r['point_quality_target']==-1 and not r['training_admission'] for r in a['references']),'No label promotion')
    checks.append('All_reference_authority_unknown_no_admission')
    return dict(completed=True,PASS=True,invented_only=True,count=len(checks),checks=checks,actual_source_or_reference_body_reads=0)
