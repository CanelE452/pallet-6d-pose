"""Source annotation/proxy provenance only; no RGB/mask/feature/model inference."""
from pathlib import Path
from collections import Counter
import csv,gzip,hashlib,json,sys,time
import numpy as np

BASE=Path(__file__).resolve().parents[2]
RUNS=BASE.parent;HERE=Path(__file__).resolve().parent
PROXY=RUNS/'virtual_cuboid_ray_oracle_20261022_v57/proxy_source_excerpt.py'
sys.path.insert(0,str(PROXY.parent))
from proxy_source_excerpt import self_visible_corner_mask,external_corner_gate_metrics

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def readgz(p,lines=False):
    with gzip.open(p,'rt')as f:return[json.loads(x)for x in f]if lines else json.load(f)
def validxy(xy,h,w,rounded=False):
    x,y=xy;v=0<=x<w and 0<=y<h
    if rounded:
        px,py=np.floor(np.asarray(xy)+.5).astype(int);v=v and 0<=px<w and 0<=py<h
    return bool(v)

def main():
    out=HERE/'SYNTH_REFERENCE_VISIBILITY_INTAKE.json';assert not out.exists();start=time.monotonic()
    sam=BASE/'query_visibility_public_real_20261022_v63/sam_source/INPUTS.jsonl'
    prep=RUNS/'external_ray_proxy_head_20261022_v60/PREPARED_INPUTS_SYNTH.json.gz'
    gtpath=Path('/home/minjae/Documents/pallet-research/pallet-statewise-end-to-end-recovery-v1/synthetic_v1/SYNTH_GT_LABELS.jsonl.gz')
    inputs=[x for x in map(json.loads,sam.read_text().splitlines())if x['source']=='SYNTH'];assert len(inputs)==1003
    prepared={x['id']:x for x in readgz(prep)};gt={x['id']:x for x in readgz(gtpath,True)}
    assert set(x['id']for x in inputs)==set(prepared)==set(gt)
    counts={s:Counter()for s in ['train','validation']};field=Counter();schemas=Counter();perms=Counter();assets=Counter();records=[];frames=[];diffs=[]
    findings=[];max_projection_diff=0.;max_reference_diff=0.;all_corner_orders=True
    for m in inputs:
        fid=m['id'];u=prepared[fid];g=gt[fid];split=m['split'];c=counts[split];c['frames']+=1
        image=Path(m['image']);annotation=image.parent.parent/'labels'/image.name.replace('_rgb.png','_label.json')
        raw=annotation.read_bytes();ann=json.loads(raw);o=ann['objects'][0];lab=o['v2_labels'];r=np.asarray(o['projected_cuboid'],np.float64);q=np.asarray(m['query'],np.float64)
        assert r.shape==q.shape==(8,2)and np.isfinite(r).all()and np.isfinite(q).all()
        assert np.array_equal(q,np.asarray(u['initial_points'][:8]))and u['split']==g['split']==split
        assert np.array_equal(r,np.asarray(g['projected_GT_native_xy']))
        assert hashlib.sha256(raw).hexdigest()==g['renderer_annotation_sha256']
        h,w=m['raw_hw'];assert u['raw_hw']==g['raw_hw']==[h,w]and ann['camera_data']['height']==h and ann['camera_data']['width']==w
        assert o['keypoint_convention']=='camera_dynamic_0123_v4'
        f=np.asarray(lab['occlusion_fraction'],float);assert f.shape==(9,)and np.isfinite(f).all()and np.all((f>=0)&(f<=1))
        world=np.asarray(o['cuboid']);campos=np.asarray(ann['camera_data']['location_worldframe'])
        camera=np.asarray(g['native_GT_xyz'])@np.asarray(g['R_fixed']).T+np.asarray(g['t'])
        sv=np.asarray(self_visible_corner_mask(world,campos));sv_cam=np.asarray(self_visible_corner_mask(camera,[0,0,0]))
        assert np.array_equal(sv,sv_cam),'World/camera convex-self proxy mismatch'
        p=camera@np.asarray(u['K']).T;uv=p[:,:2]/p[:,2:]
        max_projection_diff=max(max_projection_diff,float(np.max(np.abs(uv-r))))
        assert np.max(np.abs(uv-r))<.01
        legacy=(camera[:,2]>1e-9)&(r[:,0]>=0)&(r[:,0]<=w)&(r[:,1]>=0)&(r[:,1]<=h)
        agg=external_corner_gate_metrics(legacy,f[:8],self_visible=sv)
        aggregate_match=all(int(agg[a])==int(lab[b])for a,b in [('V_inframe','V_actual'),('V_vis','V_vis_actual'),('ext_occ_corners','ext_occ_corners_actual')])
        if not aggregate_match:findings.append(dict(id=fid,kind='source_aggregate_proxy_mismatch'))
        c['aggregate_replay_matching_frames']+=aggregate_match;c['world_camera_self_masks_equal_frames']+=1
        kp=o.get('efront_kp12')or{};kp2=np.asarray(kp.get('kp12_2d')or[]);kp3=np.asarray(kp.get('kp12_3d')or[]);kpin=kp.get('kp12_in_frame')or[]
        field['projected_cuboid8']+=r.shape==(8,2);field['occlusion_fraction9']+=f.shape==(9,)
        field['efront_kp12_present']+=bool(kp);field['efront_kp12_absent_or_null']+=not bool(kp)
        field['kp12_visible_explicit_null']+=bool(kp)and 'kp12_visible'in kp and kp['kp12_visible']is None
        field['kp12_visible_nonnull']+=kp.get('kp12_visible')is not None
        field['kp12_coords12x2']+=kp2.shape==(12,2);field['kp12_coords12x3']+=kp3.shape==(12,3)
        field['kp12_valid_true']+=kp.get('kp12_valid')is True
        field['target_depth_or_instance_map_top_field']+=any(x in o for x in ['depth','depth_path','target_depth','instance_id_map','keypoint_visibility','self_visible','corner_visible'])
        field['occlusion_decomposition_available_true']+=lab.get('occlusion_decomposition_available')is True
        schemas[str(tuple(sorted(o.keys())))]+=1;perms[str(o['perm_v4'])]+=1;assets[o['source_asset']]+=1
        c['coordinate_anchors8']+=8;c['kp12_coordinate_anchors']+=len(kp2);c['kp12_declared_inframe']+=sum(bool(x)for x in kpin)
        c['kp12_visible_state_known']+=sum(v is not None for v in(kp.get('kp12_visible')or[]))
        frame=dict(id=fid,split=split,image=str(image),image_sha256=m['image_sha256'],annotation=str(annotation),annotation_sha256=g['renderer_annotation_sha256'],
          raw_hw=[h,w],source_asset=o['source_asset'],keypoint_convention=o['keypoint_convention'],perm_v4_dynamic_to_original=o['perm_v4'],
          source_initial_query_provenance=u['upstream_provenance']['source'],aggregate_proxy_replay_match=aggregate_match)
        frames.append(frame)
        for k in range(8):
            rf=validxy(r[k],h,w);rp=validxy(r[k],h,w,True);qv=validxy(q[k],h,w,True)
            front=bool(camera[k,2]>1e-9);inside=rf and front;eligible=inside and bool(sv[k])
            proxy='F_REFERENCE_PROXY'if not inside else'S_CONVEX_PROXY'if not sv[k]else'E_RAY_GE_HALF_PROXY'if f[k]>=.5 else'D_UNBLOCKED_RAY_CONVEX_PROXY'
            delta=q[k]-r[k];distance=float(np.linalg.norm(delta));diffs.append(distance)
            c['reference_continuous_inframe_positiveZ']+=inside;c['reference_nearestpixel_valid_positiveZ']+=rp and front;c['initial_query_nearestpixel_valid']+=qv
            c[proxy]+=1;c['reference_query_different']+=bool(np.any(delta!=0));c['q_invalid_but_reference_inside']+=not qv and inside
            c['reference_inside_but_q_outside']+=inside and not qv
            if qv:c['Q_VALID_'+proxy]+=1
            records.append(dict(id=fid,split=split,native_corner=k,reference_x=float(r[k,0]),reference_y=float(r[k,1]),initial_query_x=float(q[k,0]),initial_query_y=float(q[k,1]),
                query_minus_reference_x=float(delta[0]),query_minus_reference_y=float(delta[1]),source_coordinate_displacement_px=distance,
                reference_continuous_inframe=rf,reference_nearestpixel_valid=rp,reference_camera_Z_positive=front,initial_query_nearestpixel_valid=qv,
                convex_self_visible_proxy=bool(sv[k]),reference_external_9ray_fraction=float(f[k]),diagnostic_conditional_proxy_state=proxy,
                certified_semantic_D_E_S=None,teacher_tier='LEGACY_GEOMETRY_RAY_PROXY_ONLY_NOT_ADMITTED_AS_SEMANTIC_GT',annotation=str(annotation),annotation_sha256=g['renderer_annotation_sha256']))
    assert not findings and len(records)==8024
    assert counts['train']['frames']==768 and counts['validation']['frames']==235
    assert counts['train']['initial_query_nearestpixel_valid']==5673 and counts['validation']['initial_query_nearestpixel_valid']==1742
    inventory=HERE/'SYNTH_REFERENCE_ANCHOR_INVENTORY.csv'
    with inventory.open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    manifest=HERE/'SYNTH_REFERENCE_FRAME_MANIFEST.jsonl'
    with manifest.open('x')as f:
        for row in frames:f.write(json.dumps(row,separators=(',',':'))+'\n')
    table=[
      dict(source_fact='projected_cuboid8 + camera_dynamic_0123_v4 + annotationSHA',available='All1003frames/8024point identities',sufficient_for='Fixed source reference coordinate r and native index; keep r unchanged while perturbing q.',insufficient_for='No independent D/E/S state follows from coordinate existence.',evidence_tier='RENDERER_REFERENCE_GEOMETRY'),
      dict(source_fact='Stored initial_points in V60 -> V63 query',available='All1003frames;7415nearestpixel-valid q',sufficient_for='Input-only initial proposal and measured source q-r correspondence. It can be reused as q0 before a new preregistered jitter recipe.',insufficient_for='These are frozen upstream N3/SubPix proposals, not a newly generated random-jitter sample and not semantic visibility labels.',evidence_tier='INITIAL_PROPOSAL_LINEAGE'),
      dict(source_fact='occlusion_fraction9',available='All1003frames;8corners+centroid',sufficient_for='Legacy external multi-ray blockage teacher at fixed reference identity; threshold.5 is the historical rule, not newly tuned.',insufficient_for='No per-ray hit IDs/depths/center-ray flag retained;6px-style neighborhood can be blocked while the exact reference is not. Virtual corner ray is not automatically visible pallet point evidence.',evidence_tier='CONDITIONAL_EXTERNAL_RAY_PROXY'),
      dict(source_fact='cuboid/world camera and self_visible_corner_mask excerpt',available='All1003frames;world/camera boolean parity and3storedaggregate counts replay1003/1003',sufficient_for='Reconstruct convex-cuboid front-facing semantic-index proxy at fixed r.',insufficient_for='Convex-cuboid self is not independently annotated pallet semantic S; current Windows excerpt is transcribed and exact generation snapshot binding remains incomplete.',evidence_tier='CONDITIONAL_CONVEX_SELF_PROXY'),
      dict(source_fact='V_actual/V_vis_actual/ext_occ_corners_actual/front_face_visibility',available='All sampled annotations',sufficient_for='Aggregate provenance/replay check of the above legacy convention.',insufficient_for='Aggregate counts cannot supply an independent per-corner visibility label; front-face/opening aggregate cannot be expanded into8D/E/S states.',evidence_tier='AGGREGATE_DIAGNOSTIC'),
      dict(source_fact='M0 target-only and M4 final-visible masks',available='Dataset mask_amodal/mask_visible;V63-67pixelGT derives from them',sufficient_for='Target ownership at actual q, crop/window support, observed target-positive anchor proxy under its explicit mask assumptions.',insufficient_for='M4(r)=0 does not establish semantic E/S. M4(r)=1 does not prove that a back reference corner itself is visible. Random mask-positive anchors are not native semantic corners.',evidence_tier='MASK_PIXEL_SUPPORT_ONLY'),
      dict(source_fact='efront_kp12_2d/3d/in_frame',available='Present/absent/coordinate counts below;kp12_visible supplies no nonnull visibility labels',sufficient_for='Additional coordinate/identity candidates for front-face/opening definitions if separately reviewed.',insufficient_for='No corresponding visibility flags; cannot transplant8corner occlusion fractions to12different points.',evidence_tier='ADDITIONAL_COORDINATES_ONLY'),
      dict(source_fact='Target/scene depth + per-reference owner/hit truth',available='No corresponding exported per-point fields or datasetdepth/instance-map directory found in the scopedsource root',sufficient_for='If later obtained/verified, optional stronger conditional teacher evidence.',insufficient_for='Not available in this intake. Diagnostic depth_offset fields and cameraZ are not scene-depth ownership tests. Such geometry is not a mandatory prerequisite for the operational2D user goal.',evidence_tier='UNAVAILABLE_OPTIONAL_TEACHER'),
      dict(source_fact='Human/official semantic D/E/S annotations for these synthetic native8',available='None established in the scoped source evidence',sufficient_for='An independently reviewed operational2D convention could admit source semantic labels without PnP or physical-meshDIRECT.',insufficient_for='No gold semantic anchor count can currently be claimed from existing masks/ray proxies alone.',evidence_tier='NOT_ESTABLISHED')]
    source_bindings=[ref(p)for p in [sam,prep,gtpath,PROXY,
      RUNS/'external_ray_proxy_head_20261022_v60/prepare_inputs.py',
      BASE/'query_visibility_public_real_20261022_v63/build_synthetic_cache.py',
      BASE/'query_visibility_controlled_real_20261022_v66/model/merge_partial_caches.py',
      BASE/'query_visibility_controlled_real_20261022_v66/provenance/MERGE_INPUT_LOCK.json',
      BASE/'HANDOFF_KO.md',BASE/'PATH_INDEX.csv',
      RUNS/'synthetic_mask_support_contract_20261022_v56/RECEIVED_WINDOWS_EVIDENCE.json']]
    result=dict(schema='SYNTH_REFERENCE_VISIBILITY_SOURCE_INTAKE_V1',completed=True,
      status='SOURCE_FACTS_AND_CONDITIONAL_PROXY_INVENTORY_ONLY; semantic label admission HOLD; no new labels/cache/model fit admitted',
      source_bindings=source_bindings,source_split_counts={s:dict(v)for s,v in counts.items()},
      annotation_field_coverage=dict(field),keypoint_conventions={'camera_dynamic_0123_v4':1003},permutation_histogram=dict(perms),asset_histogram=dict(assets),
      reference_order=dict(indices='0front-top-left,1front-top-right,2front-bottom-right,3front-bottom-left;4..7correspondingback;8centroidexcluded',
        storage='projected_cuboid/cuboid/occlusion_fraction are already camera-dynamic order. perm_v4 maps dynamic index to original writer native index; do not apply twice.',
        correspondence='All1003projected_cuboid arrays equal saved source projected_GT_native_xy, all8q equal V60initial_points. Native index correspondence is retained; native-physical/operational landmark semantics still need review.'),
      exact_reference_query_relationship=dict(q_lineage='SEALED_ACTUAL_N3_SUBPIX_F; preserved initial native estimates. No new synthetic jitter has been applied by this intake.',
        r_lineage='Renderer annotation projected_cuboid with perframe annotationSHA; source GT side table retains same coordinates.',
        projection_max_abs_px=max_projection_diff,all1003annotation_SHA_bindings_match=True,all1003q_prepared_parity=True,
        source_coordinate_displacement_summary=dict(N=len(diffs),median=float(np.median(diffs)),p95=float(np.percentile(diffs,95)),max=float(np.max(diffs)),
          interpretation='Annotation/input coordinate relationship only; no visibility/model performance or new solver evaluation.'),
        new_contract='Keep r,semanticID and its independently admitted reference state fixed; perturb q only, keep GT r/state outside model inputs, report all initial/crop-invalid and ambiguous correspondence coverage.'),
      evidence_sufficiency_table=table,
      anchor_capacity=dict(
        coordinate_native8=dict(TRAIN=6144,VAL=1880),
        q_valid_existing_pixel_queries=dict(TRAIN=5673,VAL=1742),
        conditional_legacy_reference_proxy_counts='See source_split_counts D_UNBLOCKED_RAY_CONVEX_PROXY/E_RAY_GE_HALF_PROXY/S_CONVEX_PROXY/F_REFERENCE_PROXY. These are not semantic D/E/S gold.',
        independently_certified_native_semantic_D_E_S=dict(TRAIN=0,VAL=0),
        additional_kp12='Coordinates and in-frame flags only; no visible state. Counts are not extra certified semantic labels.',
        jitter_multiplier='No jitter generated. A futureKjitter recipe producesKrecords per admitted reference anchor without increasing independent reference-label evidence.'),
      sourceGT_v66_trace=dict(SYNTH='V66 merge code takes V65 dense/global source6joint mask-support labels and converts each validjoint to allowed=1<<joint. No source reference-semantic visibility flag is added.',
        labels='Exact nativequery M4/M0 and fixedradius2mask support remain the existing task; this audit does not reinterpret any savednegative as semanticpositive.'),
      recommendation=dict(
        immediate='Reuse all existing reference coordinates and source-only q0 lineage as metadata. Legacy convex-self/ray classes may be retained as explicitly conditional teachers after a new versioned label protocol; do not name them independent semantic gold.',
        sufficient_next_facts='For operational semantic labels, review source semantic point conventions and obtain independent per-reference state evidence, e.g. source RGB annotation/explicitly verified teacher. Controlled opaque cover of an independently certified Dreference can create E at fixedr with matched q jitter.',
        no_extra_physical_requirement='Do not impose physical meshDIRECT support or PnP success as a fullgoal prerequisite. Geometry can strengthen a teacher tier when present; an audited2D semantic convention remains a valid route.',
        preserve_unknown='Where only supportmask0, aggregate visibility, or unverified virtual-point interpretation exists, semanticD/E/S stays UNKNOWN. Source/self distinction must be honestly reported.',
        R2B='The3R2BAI-proposed localwoodcorners remain conditional review candidates, humanconfirmed0/GT0/native8identityunknown; they cannot substantiate completed semantictraining or the fullvisibility goal.'),
      outputs=dict(anchor_inventory=ref(inventory),frame_manifest=ref(manifest)),
      read_scope=dict(source_annotation_JSON_count=1003,source_GT_side_table=True,source_query_metadata=True,
        RGB_mask_depth_feature_or_checkpoint_arrays_decoded=0,current_RGB_GT_metrics_or_outputs_opened=0,
        incidental_exception='Earlier provenance search opened historical v57INPUT_OUTPUT_PACKETS.json which also contained historical SYNTH solver/pose metrics. Root was notified immediately. Those values are unused for labels/counts/model choice, no further performancepacket was opened.',
        metadata_code_note='Source scripts containing historical solver code were read for annotation/proxy definitions only; none were executed. Only the pure proxy_source_excerpt functions were called.'),
      new_render_training_forward_current_inference_PnP_or_pose_optimization=0,existing_source_or_current_artifacts_mutated=False,
      code=ref(Path(__file__)),seconds=time.monotonic()-start)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('SYNTH_REFERENCE_INTAKE_READY',sha(out),'seconds',result['seconds']);print(json.dumps(result['source_split_counts'],indent=2));print('fieldcoverage',dict(field))

if __name__=='__main__':main()
