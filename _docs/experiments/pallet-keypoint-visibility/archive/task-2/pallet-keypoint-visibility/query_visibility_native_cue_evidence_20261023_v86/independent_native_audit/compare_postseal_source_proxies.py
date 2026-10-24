"""Post-raw-seal source metadata/GT-support comparison, no RGB/model run."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,hashlib,csv,importlib.util,numpy as np
R=Path(__file__).resolve().parent;V=R.parent;P=V.parent
HFILE=P/'query_visibility_loco_native_context_20261023_v74/source_contract/native_context_contract_v2.py'
GTP=P/'query_visibility_reference_local_support_20261023_v76/gt_corpus_build/GT_ONLY_TEACHER_CORPUS.npz'
def bind(p):
 p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def rd(p):return json.loads(Path(p).read_text())
def save(p,o):
 with Path(p).open('x')as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
seal=rd(R/'RAW_OBSERVATION_SEAL_BEFORE_PROXY.json');assert seal['PASS']and seal['no_proxy_annotation_read_before_this_seal']
assert bind(seal['raw_observations']['path'])==seal['raw_observations'];raw=rd(seal['raw_observations']['path'])
rawmap={(x['id'],x['native_k']):x for x in raw['raw_observations']}
scope=rd(V/'SOURCE_NATIVE8_VISUAL_SCOPE_BEFORE_RGB.json');targets=rd(scope['targets']['path'])['targets']
spec=importlib.util.spec_from_file_location('pure_convex_source_geometry',HFILE);H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
keys=['id','source','split','label8','rule8','native_GTref8','perm_dynamic_to_fixed8','convex_self_proxy8_separate','reference_camera_Z_positive8','center_M08','center_M48','rho_px','expected_support_pixels_A8','visible_over_expected_ratio8']
with np.load(GTP,allow_pickle=False)as z:
 ids=z['id'];idx={str(ident):i for i,ident in enumerate(ids)};rows=[idx[t['id']]for t in targets];gt={k:z[k][rows]for k in keys}
records=[];framefacts=[]
for i,t in enumerate(targets):
 assert bind(t['annotation'])['sha256']==t['annotation_sha256']
 a=rd(t['annotation']);o=a['objects'][0];r=np.asarray(t['fixed_existing_renderer_r8'],float)
 assert np.array_equal(r,np.asarray(o['projected_cuboid']))and np.array_equal(r,gt['native_GTref8'][i])
 assert o['perm_v4']==t['perm_v4_dynamic_to_original']==gt['perm_dynamic_to_fixed8'][i].tolist()
 assert str(gt['split'][i])=='train'and str(gt['source'][i])=='SYNTHETIC'
 self_replay=H.convex_self_visible_proxy(o['cuboid'],a['camera_data']['location_worldframe']).astype(np.int8)
 assert np.array_equal(self_replay,gt['convex_self_proxy8_separate'][i])
 fraction=np.asarray(o['v2_labels']['occlusion_fraction'][:8],float);assert fraction.shape==(8,)
 proxy,inside=H.source_reference_proxy(r,gt['reference_camera_Z_positive8'][i],t['raw_hw'],self_replay,fraction)
 framefacts.append({'id':t['id'],'annotation':bind(t['annotation']),'first8_stored_fraction':fraction.tolist(),
  'stored_fraction_extra_centroid_not_native8':o['v2_labels']['occlusion_fraction'][8:],
  'pure_convex_self_visible_replayed':self_replay.tolist(),'weak_native_proxy_states_D_E_S_F_U':proxy.tolist(),
  'V76_support_labels_C0_V1_Uminus1':gt['label8'][i].tolist(),'V76_rules':gt['rule8'][i].tolist(),
  'stored_explicit_occluder_placed':o['v2_labels'].get('occluder_placed'),'kp12_visible':o.get('efront_kp12',{}).get('kp12_visible'),
  'source_RGB_annotation_r_permutation_GTpack_exact':True})
 for k in range(8):
  obs=rawmap[(t['id'],k)];assert bool(inside[k])==obs['in_native_frame']
  ratio=float(gt['visible_over_expected_ratio8'][i,k]);ratio=ratio if np.isfinite(ratio)else None
  records.append({'id':t['id'],'native_k':k,'native_name_from_frozen_convention':H.NATIVE_NAMES[k],
   'raw_category_before_proxy':obs['raw_evidence_category'],'stored_first8_occlusion_fraction':float(fraction[k]),
   'pure_convex_may_visible':int(self_replay[k]),'weak_native_proxy':H.PROXY_NAMES[int(proxy[k])],
   'V76_support_label_C0_V1_Uminus1':int(gt['label8'][i,k]),'V76_teacher_rule':str(gt['rule8'][i,k]),
   'V76_center_M0':int(gt['center_M08'][i,k]),'V76_center_M4':int(gt['center_M48'][i,k]),
   'V76_expected_support_pixels':int(gt['expected_support_pixels_A8'][i,k]),'V76_visible_expected_ratio':ratio,
   'V76_rho_px':float(gt['rho_px'][i]),'in_native_frame':bool(inside[k]),
   'proxy_is_independent_semantic_gold':False,'labels_or_coords_changed':False})
assert len(records)==24
counts={'all_slots':24,'raw_categories':dict(Counter(x['raw_category_before_proxy']for x in records)),
 'weak_native_proxy_counts':dict(Counter(x['weak_native_proxy']for x in records)),
 'V76_support_labels':dict(Counter(str(x['V76_support_label_C0_V1_Uminus1'])for x in records)),
 'V76_support_rules':dict(Counter(x['V76_teacher_rule']for x in records)),
 'inframe_V76V_center_M0zero_N':sum(x['in_native_frame']and x['V76_support_label_C0_V1_Uminus1']==1 and x['V76_center_M0']==0 for x in records),
 'raw_observed_structural_V76_center_M0zero_N':sum(x['raw_category_before_proxy']=='OBSERVED_STRUCTURAL_CUE_CONDITIONAL_AI'and x['V76_center_M0']==0 for x in records),
 'convex_self_hidden_but_V76_supportV_N':sum(x['pure_convex_may_visible']==0 and x['V76_support_label_C0_V1_Uminus1']==1 for x in records),
 'V76supportV_raw_unresolved_or_ambiguous_N':sum(x['V76_support_label_C0_V1_Uminus1']==1 and x['raw_category_before_proxy']in ['NO_IDENTIFIABLE_CUE_OR_AMBIGUOUS_U','REFERENCE_IDENTITY_UNRESOLVED_U']for x in records),
 'weak_D_proxy_raw_unresolved_or_ambiguous_N':sum(x['weak_native_proxy']==H.PROXY_NAMES[0]and x['raw_category_before_proxy']in ['NO_IDENTIFIABLE_CUE_OR_AMBIGUOUS_U','REFERENCE_IDENTITY_UNRESOLVED_U']for x in records),
 'legacy_fraction_ge_half_first8':sum(x['stored_first8_occlusion_fraction']>=.5 for x in records),
 'legacy_fraction_nonzero_first8':sum(x['stored_first8_occlusion_fraction']>0 for x in records),
 'raw_direct_foreign_blocker_at_inframe_fixed_reference_N':0}
assert counts['inframe_V76V_center_M0zero_N']==12 and counts['convex_self_hidden_but_V76_supportV_N']==4
with (R/'NATIVE24_POSTSEAL_PROXY_COMPARISON.csv').open('x')as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
report={'schema':'V86_NATIVE24_POST_RAW_SEAL_PROXY_COMPARISON','completed':True,'PASS':True,'created_UTC':datetime.now(timezone.utc).isoformat(),
 'raw_observation_seal':bind(R/'RAW_OBSERVATION_SEAL_BEFORE_PROXY.json'),'all3_annotation_r_perm_GT_support_identity_exact':True,
 'pure_convex_self_replay_all24_exact_V76':True,'summary':counts,'frame_facts':framefacts,'per_slot':records,
 'proxy_targets_KO':{'legacy_fraction':'기존fixedreference 주변 다중ray blockage proxy의 저장첫8개. .5는historical gate이며 새선택이아니다. per-rayhit ID/center-ray/깊이는이저장값에없어RGBcue semanticvisibility와동일시할수없음. 9번째centroid는native8에서제외.',
 'convex_self':'가상convexcuboid 인접면 중camera-facing면이있는지의puregeometry. 실제slatted/moldedpallet mesh나인간Sgold가아님.',
 'V76_local_support':'r주변M0expectedtarget지원에서 M4visible 지원비율/3x3 또는footprint teacher. 실제고정nativecorner가 보인다는 label이아니며 자기면도target visible-support로판정될수있음.'},
 'evidence_conclusions_KO':[
  '원native24 중 관찰structuralcue10, raw ambiguous/identityU10, OOB4. 기존weaknativeproxy는D16/S4/F4이고 E0, V76support는V20/U4/C0이다. 이차이를gold오분류율로계산하지않는다.',
  'V76supportV20 중4개는convexselfhidden이며, raw에서도nativepartidentityU였다. f16972 k6은perforatedtopdeck pattern 위에lowerrearvirtualr가투영되는 직접사례다. targetmaterial이보이는것과backreference corner가보이는것을같이채점하면안된다.',
  'V76supportV20 중12개는storedexactcenterM0=M4=0이다. 이중rawstructuralcue10개중8개도center0이다. 물리partline이근처에보이더라도virtualextreme/rasterpixelmembership은다를수있으며 physicalDIRECT로 admission을제한하면원목표를잘못축소한다.',
  'rayfraction첫8은최대.3333/.5초과0이고 rawfixedrefs에서도foreignblocker는관찰되지않았다. 그러나 zero/subhalf fraction이특정nativepartcueD의독립증거인것은아니다. weakD16중6개는raw에서partidentity/외곽관계가미해결이었다.',
  'f36687 뒤상부r4/r5는보이는slat끝보다바닥쪽에놓인다는raw관찰이다. 근처partcue range가4px진단과겹칠수있어도 정확한nativecorner동일성이나D를확정할수없다.',
  'f29051 중앙cardboard는native8fixedrefs를직접덮지않는다. 저장9번째centroidfraction1은native8외이며cornerE로복제하면안된다.',
  '이관찰은더강한모델필요성이나한가지실패원인을입증하지않는다. 다음source감독은r/nativeID와cue-evidence/partidentity uncertainty, qcropavailability, exactpixelownership을별개로저장해야한다. 기존label/primary를재해석하거나현재평가에맞춰radius를고르지않는다.'
 ],
 'no_new_label_admission_no_old_corpus_reinterpretation':True,'raw_report_or_r_q_order_changed':False,'M0_M4_depth_pixel_arrays_read':0,
 'modelweights_features_forward_fit_current_PnP':0,
 'bindings':[bind(GTP),bind(HFILE),bind(P/'query_visibility_visible_preservation_20261023_v67/source_audit/SYNTH_REFERENCE_VISIBILITY_INTAKE.json'),bind(Path(__file__))]}
o=R/'POSTSEAL_PROXY_COMPARISON_KO.json';save(o,report)
files=[R/'RAW_REVIEW_POLICY_BEFORE_RGB.json',R/'RAW_INSPECTION_RENDER_RECEIPT.json',R/'RAW_NATIVE24_OBSERVATIONS.json',R/'RAW_OBSERVATION_SEAL_BEFORE_PROXY.json',R/'NATIVE24_POSTSEAL_PROXY_COMPARISON.csv',o,Path(__file__),R/'write_raw_observations.py',R/'render_raw_native_inspection.py']
s=R/'FINAL_NATIVE24_EVIDENCE_SEAL.json';save(s,{'completed':True,'PASS':True,'all24_native_slots_original_r_q_retained':True,'files':[bind(x)for x in files],'actual_RGB_inspection_before_proxy_seal':True,'new_forward_fit_current_PnP':0,'existing_artifact_changes':False})
print(json.dumps({'comparison':bind(o),'seal':bind(s),'summary':counts},ensure_ascii=False))
