"""V160 source metadata/inputnative provenance only. NPZ/weights/RGB inaccessible here."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,math
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_native_slot_owner_factorial_20261024_v160';O=Path(__file__).resolve().parent
B={};CHECKS=[];EXPECTED=[]
def guard(p):
 p=Path(p);s=str(p).lower();assert p.is_absolute() and str(p).startswith(str(P)+'/')
 assert not any(t in s for t in ('v112','current223','/private'))
 assert p.suffix in ('.json','.jsonl','.py'),'No NPZ/weights/RGB/sourcepacket bodies in metadata audit'
 return p
def bind(p,expected=None):
 p=guard(p);b=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 if expected:assert b['sha256']==expected,p
 B[str(p)]=b;return b
def checked(b):bind(b['path'],b['sha256']);return Path(b['path'])
def read(p):bind(p);return json.loads(guard(p).read_text())
def readb(b):return read(checked(b))
def lineb(b):return [json.loads(s) for s in checked(b).read_text().splitlines() if s]
def ck(name,c,n=None):CHECKS.append(dict(name=name,PASS=bool(c),rows=n));assert c,name
def xy(v):
 if v is None:return (None,None)
 return tuple(None if x is None or not math.isfinite(float(x)) else float(x) for x in v)
def side(v):return None if v is None or not math.isfinite(float(v)) else float(v)
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def seqhash(rows):return hashlib.sha256(('\n'.join(json.dumps(r,sort_keys=True,separators=(',',':'),ensure_ascii=False) for r in rows)+'\n').encode()).hexdigest()
design_path=V/'ROOT_DESIGN_BEFORE_NEW_SOURCE_FEATURE_OR_MODEL_BODIES.json'
design=readb(dict(path=str(design_path),sha256='5b49cc780ca99fae895fe7cc785fbee74ba34f26301ee8fa5e41c1f1cabbfd2b'))
p=readb(dict(path=str(P/'query_visibility_independent_auxiliary_source_control_20261024_v155/source_fit_preparation/PROTOCOL_BEFORE_SOURCE_FEATURE_OR_CHECKPOINT_BODIES.json'),sha256='a042ca1c7daa02959f08ccc45ee65920870d40090518b8ec61c685b7e50910d8'))
old=readb(p['parent_protocol']);bind(p['code']['path'],p['code']['sha256']);bind(p['helper']['path'],p['helper']['sha256']);bind(p['head']['path'],p['head']['sha256']);bind(old['helper']['path'],old['helper']['sha256'])
bridge_binding=next(b for b in design['evidence'] if b['path'].endswith('/owner_supervision_bridge/COMPLETE.json'))
bridge=readb(bridge_binding);owners=lineb(bridge['outputs'][0]);families={};source_rows={};provenance={}

for family,native,qbinding in zip(('OLD6144','NEW20480'),old['native'],p['quality_labels']):
 seal=readb(native['feature_seal']);ck(family+' featurebinding inherited exactly',seal['features']==native['features'])
 bind(seal['code']['path'],seal['code']['sha256']);facts=readb(seal['input_crop_facts'])['rows'];queries=lineb(seal['initializer_queries']);proposals=lineb(seal['initializer_proposals']);quality=lineb(qbinding)
 gt=lineb(native['GT']);N=int(seal['rows']);primary=gt[0::2];sensitivity=gt[1::2]
 ck(family+' allmetadata rowcounts',len(facts)==len(queries)==len(quality)==len(primary)==len(sensitivity)==N and len(proposals)*8==N,N)
 records=[];no_det=0;different_conventions=0
 for i,(f,q,l,g,s) in enumerate(zip(facts,queries,quality,primary,sensitivity)):
  slot=f['native_slot'];scene=f['scene_index'];proposal=proposals[scene]
  assert i==f['occurrence_index'] and scene==i//8 and slot==i%8 and 0<=slot<8
  assert f['occurrence_id']==q['query_id']==l['query_id']==g['query_id']==s['query_id']==f['scene_id']+':native'+str(slot)
  assert q['native_slot']==l['native_slot']==g['native_slot']==s['native_slot']==slot
  assert f['split']==q['split']==l['split']==g['split']==s['split']==proposal['split']
  assert g['convention']=='CF_PRIMARY_DECLARED' and s['convention']=='V4_SENSITIVITY_ONLY'
  assert xy(f['query_xy'])==xy(q['query_xy'])==xy(l['original_query_xy'])==xy(g['query_xy'])==xy(s['query_xy'])
  assert side(f['L'])==side(q['roi_side'])==side(l['original_roi_side'])==side(g['roi_side'])==side(s['roi_side'])
  assert l['original_role_teacher_state_UNCHANGED']==g['teacher_state']
  different_conventions+=g['teacher_state']!=s['teacher_state']
  selected=proposal['selected_index'];nd=selected is None;no_det+=nd
  assert selected==f['selected_index']==q['selected_index']
  if nd:
   assert xy(q['query_xy'])==(None,None) and side(q['roi_side']) is None and not f['feature_constructed'] and not f['query_available']
  else:assert xy(q['query_xy'])==tuple(proposal['candidates'][selected]['keypoints_xy'][slot])
  assert f['query_available']==(f['feature_constructed'] and f['factual_query_available'] and f['context_available'])
  assert f['factual_query_available']==(f['native_continuous_valid'] and f['native_nearest_valid'] and f['RGB_central4_available'])
  assert not f['query_available'] or f['feature_constructed']
  record=dict(family=family,global_index=i,query_id=q['query_id'],frame_index=scene,native_slot=slot,split=f['split'],query_xy=list(xy(f['query_xy'])),L=side(f['L']),
    no_detection=nd,query_available=f['query_available'],feature_constructed=f['feature_constructed'],native_continuous_valid=f['native_continuous_valid'],nearest_valid=f['native_nearest_valid'],
    local_central4_available=f['RGB_central4_available'],context_available=f['context_available'])
  records.append(record)
 ck(family+' originalinputslot/rawquery/splits/qL/quality/GT identity',True,N)
 ck(family+' familyavailability formula/noDetection noresurrection',True,N)
 source_rows[family]=records;EXPECTED.extend(records)
 families[family]=dict(rows=N,frames=len(proposals),feature_packet=native['features'],feature_seal=native['feature_seal'],inputfacts=seal['input_crop_facts'],initializer_queries=seal['initializer_queries'],initializer_proposals=seal['initializer_proposals'],role_labels=native['GT'],point_quality_labels=qbinding,
   native_slot_dtype='int8',native_slot_packet_field='native_slot',identity_packet_field='occurrence_id',split_packet_field=None,split_source='CF_PRIMARY_DECLARED existing separate metadata',
   query_available=sum(r['query_available'] for r in records),inputU=sum(not r['query_available'] for r in records),feature_constructed=sum(r['feature_constructed'] for r in records),no_detection=no_det,
   split_counts=dict(Counter(r['split'] for r in records)),native_slot_counts=dict(Counter(r['native_slot'] for r in records)),
   two_GT_conventions_different_state_rows=different_conventions,sensitivity_convention_not_native_hint=True,expected_input_identity_sha256=seqhash(records),
   availability='feature_constructed & factual_query_available & context_available; factual=originalcontinuous & nearest & RGB96central4',
   metadata_sources_verified_without_feature_body=True)
 provenance[family]=dict(producer=seal['code'],lines='OLD V132117-144 inputnative enumeration;218-229 common_arrays;364-368 save. NEW V13698-133 inputnative enumeration/common_arrays;273-277 reuse V132run_c16.',
   proof='native_slot originates from frozen initializer first8 keypoint channel enumeration; query ID/native_slot/qxy[slot] and scene/rawindex are explicitly guarded; cache copies inputrows.native_slot. NotGTcanonical_index orpermutation.')

mrseal=readb(p['MR_feature_seal']);mrinputs=readb(mrseal['inputs']);mrqueries=lineb(mrinputs['queries']);mrfacts=lineb(p['MR_inputfacts']);mrlabels=lineb(p['source_labels']);bind(mrseal['code']['path'],mrseal['code']['sha256'])
mr_initializer_seal=readb(mrinputs['initializer']);bind(mr_initializer_seal['code']['path'],mr_initializer_seal['code']['sha256'])
mr_initializer_protocol=readb(mr_initializer_seal['protocol']);bind(mr_initializer_protocol['effective_run']['path'],mr_initializer_protocol['effective_run']['sha256'])
pal_seal=readb(p['Pallox_feature_seal']);palprotocol=readb(pal_seal['protocol']);palqueries=lineb(palprotocol['original_input_bindings']['queries']);palfacts=lineb(p['Pallox_inputfacts']);pallabels=lineb(old['public_labels']);bind(pal_seal['code']['path'],pal_seal['code']['sha256'])
public_fixed=readb(p['original_native35']);ck('Pallox original publiclabel binding preserved',public_fixed['public_labels']==old['public_labels'])
for family,seal,queries,facts,labels,feature,fb in [('MR1840',mrseal,mrqueries,mrfacts,mrlabels,p['MR_features'],p['MR_feature_seal']),('PALLOX512',pal_seal,palqueries,palfacts,pallabels,p['Pallox_features'],p['Pallox_feature_seal'])]:
 N=1840 if family=='MR1840' else 512;records=[]
 ck(family+' immutable featurebinding inherited',seal['features']==feature)
 ck(family+' complete metadata rows',len(queries)==len(facts)==len(labels)==N,N)
 for i,(q,f,l) in enumerate(zip(queries,facts,labels)):
  slot=f['native_slot'];frame=f['frame_index'];frameid=f['frame_id']
  assert i==f['global_index']==l['global_index'] and frame==i//8 and slot==i%8
  assert q['native_slot']==l['native_slot']==slot and q['frame_index']==l['frame_index']==frame
  assert q['query_id']==f['query_id']==frameid+':native'+str(slot)
  assert q['frame_id']==frameid==l['id'] and q['source_id']==f['source_id']==l['source_id'] and q['split']==f['split']==l['split']
  assert xy(q['query_xy'])==xy(f['query_xy'])==xy(l['query_xy']) and side(q['L'])==side(f['L'])==side(l['L'])
  assert q['native_continuous_valid']==f['native_continuous_valid']==l['original_input_valid']
  nd=q['no_detection'];constructed=f.get('feature_constructed',True)
  if family=='MR1840':
   assert nd==l['no_detection']
   assert not nd or (not constructed and not f['native_continuous_valid'])
   assert not f['native_continuous_valid'] or constructed
  else:assert not nd and constructed
  record=dict(family=family,global_index=i,query_id=q['query_id'],frame_index=frame,native_slot=slot,split=f['split'],query_xy=list(xy(f['query_xy'])),L=side(f['L']),
   no_detection=nd,query_available=f['native_continuous_valid'],feature_constructed=constructed,native_continuous_valid=f['native_continuous_valid'],nearest_valid=f['nearest_valid'],
   local_central4_available=f['local_central4_available'],context_available=f['context_available'])
  records.append(record)
 ck(family+' frozeninput native/query/index/qL/split/availability identity',True,N)
 source_rows[family]=records;EXPECTED.extend(records)
 families[family]=dict(rows=N,frames=N//8,feature_packet=feature,feature_seal=fb,inputfacts=p['MR_inputfacts' if family=='MR1840' else 'Pallox_inputfacts'],
   initializer_queries=mrinputs['queries'] if family=='MR1840' else palprotocol['original_input_bindings']['queries'],role_labels=p['source_labels'] if family=='MR1840' else old['public_labels'],
   native_slot_dtype='int8',native_slot_packet_field='native_slot',identity_packet_field='query_id',split_packet_field='split',
   query_available=sum(r['query_available'] for r in records),inputU=sum(not r['query_available'] for r in records),feature_constructed=sum(r['feature_constructed'] for r in records),no_detection=sum(r['no_detection'] for r in records),
   split_counts=dict(Counter(r['split'] for r in records)),native_slot_counts=dict(Counter(r['native_slot'] for r in records)),expected_input_identity_sha256=seqhash(records),
   availability='query_available exactly original native_continuous_valid;nearest/context/central4/Q_inside independent,do not introduce new U rule',
   feature_constructed_packet_field='feature_constructed' if family=='MR1840' else 'absent;producer establishes all512constructed',metadata_sources_verified_without_feature_body=True)
 provenance[family]=dict(producer=seal['code'],lines='MR V15360 inputJSONnative-slot/query-ID guard;167inputfacts;187copiesqdata.native_slot. Pallox V147100-105inputenumeration/qxy guard;268-269inputfacts;287-294guardednp.tile native-slot cache.',
   proof='Inputnative channel enumeration remains originalfirst8 initializer query order. Labels andprovidedGT reference are separate. Modulo is anidentity assertion only; runtimehint mustreadcache.native_slot.')

ck('All28976source rows and535 originalinputU preserved',len(EXPECTED)==28976 and sum(not r['query_available'] for r in EXPECTED)==535)
ck('MR ownercomplete/sidecar1840',bridge['completed'] and bridge['PASS'] and len(owners)==1840)
ownerlookup={r['global_index']:r for r in owners};owner_counts=Counter();known=[]
for i,(o,f,l,q) in enumerate(zip(owners,mrfacts,mrlabels,mrqueries)):
 assert i==o['global_index']==f['global_index']==l['global_index'] and o['native_slot']==f['native_slot']==q['native_slot']
 assert o['query_id']==f['query_id']==q['query_id'] and o['split']==f['split']==l['split']
 assert xy(o['query_xy'])==xy(f['query_xy'])==xy(l['query_xy']) and side(o['L'])==side(f['L'])==side(l['L'])
 assert o['owner_aux_target']=={'TARGET_VISIBLE_SUPPORTED':0,'FOREIGN_VISIBLE_SUPPORTED':1,'BACKGROUND_SUPPORTED':2,'UNKNOWN':-1}[o['owner_state']]
 assert o['owner_target_available']==(o['owner_aux_target']>=0)
 assert o['source_only_weak_observation'] and not any(o[k] for k in ('human_gold','physical_native_gold','pixel_coordinate_gold','training_admitted','native_visibility_target_defined','point_quality_target_defined'))
 if o['owner_aux_target']>=0:
  assert o['split']=='SOURCE_DEVELOPMENT' and f['native_continuous_valid'] and f['feature_constructed'] and o['center4_valid'] and l['role_target']>=0
  known.append(i)
 if o['split']=='SOURCE_VALIDATION':assert o['owner_state']=='UNKNOWN' and o['owner_aux_target']==-1 and not o['prior_owner_observation_present']
 assert l['point_quality_label']==-1 and not l['human_gold'] and not l['exact_real_point_gold']
 owner_counts[o['owner_state']]+=1
ck('OwnerT53F52BG148U1587/253TRAINknown andheld360U onlyaux',owner_counts==Counter(TARGET_VISIBLE_SUPPORTED=53,FOREIGN_VISIBLE_SUPPORTED=52,BACKGROUND_SUPPORTED=148,UNKNOWN=1587) and len(known)==253 and sum(o['split']=='SOURCE_VALIDATION' for o in owners)==360,1840)
for frame in range(230):assert len({r['entity_target'] for r in mrlabels[frame*8:frame*8+8] if r['entity_target']>=0})<=1
ck('MR perframe entitylabels no conflictingknown classes;primarylabels untouched',True,230)
for k,target in [('TRAIN_D35','D'),('TRAIN_E8','E')]:
 for i in public_fixed[k]:assert pallabels[i]['semantic_state']==target and source_rows['PALLOX512'][i]['split']=='TRAIN_LAB' and source_rows['PALLOX512'][i]['query_available']
ck('Existing publicTRAIN35D8E pools exact; knowninputU retained',len(public_fixed['TRAIN_D35'])==35 and len(public_fixed['TRAIN_E8'])==8 and [i for i,r in enumerate(pallabels) if r['semantic_state'] in ('D','E') and not r['original_input_valid']]==[402,434])
mr_primary=[i for i,r in enumerate(mrlabels) if r['split']=='SOURCE_DEVELOPMENT' and r['original_input_valid'] and r['role_target']>=0]
ck('OriginalMR390 primary maskD341E49 unaffectedbyownerunknown',len(mr_primary)==390 and sum(mrlabels[i]['role_state']=='D' for i in mr_primary)==341 and sum(mrlabels[i]['role_state']=='E' for i in mr_primary)==49 and set(known)<=set(mr_primary))
for path,expected in [('query_visibility_boundary_query_source_20261023_v125/feature_baseline_preparation/run_frozen_boundary_source.py','c59d986a29668c04e19500ac1418062ad1912fdcdcfd5ed0e56209e8225be1e1')]:bind(P/path,expected)
packet_order=('OLD6144','NEW20480','PALLOX512','MR1840')
EXPECTED=[r for family in packet_order for r in source_rows[family]]
families={family:families[family] for family in packet_order};provenance={family:provenance[family] for family in packet_order}
provenance['MR1840']['initializer_wrapper']=mr_initializer_seal['code']
provenance['MR1840']['initializer_packet_writer']=mr_initializer_protocol['effective_run']
provenance['MR1840']['initializer_packet_writer_proof']='Bound EFFECTIVE_FULL230_R0_RUN.py lines66,78:inputquery native_slot=k andcommon sourceinput arrays explicitlywrite native_slot=np.tile(np.arange(8,dtype=np.int8),230);encoder copies thatqdatafield unchanged atencode_full230.py187. Wrapperinfer_full230.py freezes thiseffectivecount-adaptedrun inR0protocol.'
expected_path=O/'EXPECTED_SOURCE_INPUT_NATIVE_METADATA28976.jsonl'
with expected_path.open('x') as f:
 for r in EXPECTED:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
limitations=['FeatureNPZ/weight bodies were not read orhashed; listed featureSHA are inherited fromverifiedsource seals.Theactualfitentry must verifythembeforeroot-authorizeddeserialize.',
 'native_slot is known inputchannel enumeration,not GTcanonical/nativepermutation. Usecached native_slot directly foronehot;never derivehintfromlabel/modulo/nearestGT. Missing/noninteger/outofrange/conflictingnative_slot mustfailclosed.',
 'Synthetic and public availability definitions differ.Preserveeachoriginalmask andfull rowdenominator; nativehint/owner/entity/quality cannot revive inputU.',
 'Ownerlabels areweakRGB96center2x2categories,not physicalnativeD/E/coordinatequalitygold. UNKNOWN1587 is excludedonlyfromownerloss;all1840remain.',
 'All4arms mustfreeze before sourceheld comparison.Owneraccuracy alone isnot success;WOOD/SQUARE/native8fullobjective andlegacy role/quality/native35 qualification remain.',
 'V155 historicalanchorcheckpoint is lineageonly,not aV160freshinitializationinput. No source/currentcheckpoint warmstart permittedbyfrozendesign.']
contract=dict(schema='V160_SOURCE_INPUT_BINDINGS_AND_NATIVE_SLOT_PROVENANCE',completed=True,PASS=True,prepared_only=True,
 design=bind(design_path),V155_source_protocol=bind(Path(p['code']['path']).parent/'PROTOCOL_BEFORE_SOURCE_FEATURE_OR_CHECKPOINT_BODIES.json'),V148_parent_protocol=p['parent_protocol'],
 source_rows_total=28976,source_families=families,native_slot_provenance=provenance,
 original_fit_packet_order=list(packet_order),expected_metadata_order='OriginalV155allpacks order OLD6144,NEW20480,PALLOX512,MR1840; eachfamilyglobal_index ascending. Alwaysjoin by(family,global_index/query_id).',
 expected_metadata=bind(expected_path),expected_metadata_schema={k:type(v).__name__ for k,v in EXPECTED[0].items()},
 packet_runtime_schema=dict(features='float32[N,1928],finite,unchanged',global_index='integer[N],exactarangeN',native_slot='int8[N],required,0..7;readpacketfielddirectly',
  query_xy='float64[N,2],raworiginal;None/NaN normalizedonlyforidentitycomparison',L='float64[N],raworiginal',query_available='bool[N],originalfamily-specificmask',
  feature_constructed='bool[N] exceptPalloximplicitall512True',no_detection='bool[N] inOLD/NEW/MR;Palloxnoneabsentselectedindexalwaysvalid',
  split='OLD/NEW fromseparateprimaryGTmetadata;Pallox/MR cachedandseparateinputmetadata mustagree'),
 runtime_requirements_before_fit=['Verifyexpected featureSHA afterrootactualentry,thencheckeachcachednative_slot against expectedmetadataandquery IDs. No fallbackslotconstruction.',
  'Useprovidednative_slotforonehot8;usearange/moduloonlyasassertionagainstexplicitproducerinputenumeration.',
  'If unavailable,primarypredictionmustbeU(-1)regardlessnative/owner/entity/quality outputs;retainallrowsandflags.',
  'Replayoriginalsourcepools/draws;primaryMR390/public43/syntheticCF masksunchanged.Ownerfull253balancedCE hasnoextraRNGdraws.',
  'NeverloadV155anchorcheckpointforV160freshinit.Allfourbaseinitialstatesanddrawidentitiesmustmatchfrozenrootdesign/inventedchecks.'],
 owner_sidecar=bridge['outputs'][0],owner_bridge=bridge_binding,owner_auxiliary_map=bridge['owner_auxiliary_target_map'],owner_counts=dict(owner_counts),known_owner_train_indices=known,
 owner_known_train_rows=253,owner_held360_all_unknown=True,primaryMR390_mask_unchanged=True,MR_point_quality_all_minus1=True,
 native_role_hint_no_GT_permutation_or_coordinate_dependency=True,source_packet_conflicts_found=0,blockers_before_feature_body_read=[],
 source_metadata_and_code_bindings=list(B.values()),checks=CHECKS,check_count=len(CHECKS),semantic_limits=limitations,
 source_feature_NPZ_or_weight_or_RGB_or_private_or_V112_bodies_read=0,model_or_encoder_or_initializer_or_fit_calls=0,actual_fit_performed=False,
 actual_fit_authorized_by_this_audit=False,selected_model=None,goal_status='active_unsolved',code=bind(__file__))
path=O/'INPUT_BINDINGS_AND_NATIVE_SLOT_PROVENANCE.json';write(path,contract)
complete=dict(completed=True,PASS=True,prepared_only=True,contract=bind(path),code=bind(__file__),design=bind(design_path),checks=len(CHECKS),
 blockers=[],source_feature_or_weight_body_reads=0,actual_fit_performed=False,root_actual_fitentry_still_required=True,
 independent_design_scope_review=dict(source_leakage_or_scope_narrowing_blocker_found=False,
  supported_findings=['Inputnative_slot precedesGTlabels andfollowsguarded frozenquerychannel enumeration inall4families.','Owner253TRAINonly;held360auxUNKNOWN;primarynative8/role/qualitylabels untouched.','Fourarms/fixedseed/draws/updates/threshold/freeze-before-validation preventadaptivedata/armselection underdesign.'],
  required_caveats=limitations))
write(O/'PREPARATION_COMPLETE.json',complete)
print(json.dumps(dict(completed=True,contract=bind(path),preparation_complete=bind(O/'PREPARATION_COMPLETE.json'),checks=len(CHECKS),families={k:{kk:v[kk] for kk in ('rows','query_available','inputU','feature_constructed','no_detection')} for k,v in families.items()},owner_counts=dict(owner_counts)),indent=2))
