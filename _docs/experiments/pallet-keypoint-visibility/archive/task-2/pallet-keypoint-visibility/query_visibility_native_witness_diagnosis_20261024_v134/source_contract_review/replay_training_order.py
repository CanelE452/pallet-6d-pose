"""Replay source-label ontology, not model predictions or accuracy-selected maps."""
from pathlib import Path
import ast, collections, hashlib, json, time
import numpy as np

H=Path(__file__).resolve().parent;P=H.parent.parent
V74=P/'query_visibility_loco_native_context_20261023_v74/source_contract'
CONTRACT=V74/'native_context_contract_v2.py'
PACK=V74/'BUILD_V2/TYPED_SOURCE_PACK_GT_ONLY.npz'
SEAL=V74/'BUILD_V2/TYPED_SOURCE_PACK_SEAL.json'
CFG=P/'query_visibility_registered_native_diversity_20261023_v129/render_preparation/BASE_CONFIGS192_BEFORE_RGB.json'
REPO=Path('/home/minjae/Documents/github/pallet-pose')
MANIFEST=REPO/'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json'
CACHE=REPO/'data/pallet/results/pallet_line_pose_v1/cache/CACHE_MANIFEST.json'
TRAIN=REPO/'challenge/yolo_pose_one_model/spatial_concat_scratch/TRAINING_CONTRACT.json'
LEGACY=REPO/'scripts/annotate/convert_to_camera_facing_v4.py'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size)

def run():
    start=time.monotonic();seal=read(SEAL);assert seal['completed']and seal['PASS']
    assert sha(PACK)=='576d19c1f190356d4dbcfe1527564ea30047eea4cc6367dea39de28b14702931'
    assert sha(CONTRACT)=='4ddd134aa26da1d676628bd6987a161ab36f09328dfdaa89bedd213e397755ab'
    tree=ast.parse(CONTRACT.read_text());nodes=[]
    for node in tree.body:
        if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='VERTICAL_FIXED_FACES'for t in node.targets):nodes.append(node)
        elif isinstance(node,ast.FunctionDef)and node.name in('outward_normals','normal_facing_permutation'):nodes.append(node)
    namespace={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(CONTRACT),'exec'),namespace)
    normal=namespace['normal_facing_permutation']
    oldnodes=[n for n in ast.parse(LEGACY.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in('compute_perm_v4','polyarea')]
    legacy_ns={'np':np};exec(compile(ast.Module(body=oldnodes,type_ignores=[]),str(LEGACY),'exec'),legacy_ns)
    with np.load(PACK,allow_pickle=False)as f:
        world=f['cuboid_world_dynamic8'];perm=f['perm_dynamic_to_writer_fixed8'];projected=f['reference_projected8']
        cameras=f['camera_world'];sample_ids=f['sample_id'];sources=f['source'];record_index=f['record_index']
    assert world.shape==(60000,8,3)and np.array_equal(record_index,np.arange(60000))
    counts=collections.Counter();margins=[];mismatch=[]
    for i in range(60000):
        fixed=np.empty((8,3));fixed[perm[i]]=world[i]
        fixed_xy=np.empty((8,2));fixed_xy[perm[i]]=projected[i]
        replay,cosine=normal(fixed,fixed_xy,cameras[i])
        exact=np.array_equal(replay,perm[i]);counts[(str(sources[i]),'EXACT'if exact else'MISMATCH')]+=1
        if not exact:mismatch.append(int(i))
        sorted_cos=np.sort(cosine);margins.append(float(sorted_cos[-1]-sorted_cos[-2]))
        if(i+1)%20000==0:print(json.dumps(dict(source_rows_replayed=i+1,seconds=round(time.monotonic()-start,2))),flush=True)
    assert not mismatch
    manifest=read(MANIFEST);cache=read(CACHE);train=read(TRAIN)
    assert sha(MANIFEST)==cache['source_manifest_sha256']
    assert len(manifest['records'])==60000
    assert cache['backbone_sha256']=='970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'
    assert Path(manifest['dataset']).resolve()==(REPO/Path(train['dataset']['yaml']).parent).resolve()
    assert train['dataset']['train']['total']==55980 and train['dataset']['val']['total']==4020
    observed_known=0;unknown_slots=0;max_error=0.;source_label_counts=collections.Counter();max_by_source=collections.defaultdict(float)
    for i,r in enumerate(manifest['records']):
        assert r['index']==i and r['id']==str(sample_ids[i])and r['source_kind']=='synthetic'
        assert len(r['targets'])==1
        labels=np.asarray(r['targets'][0]['keypoints_normalized'],np.float64)[:8]
        assert labels.shape==(8,3)
        good=labels[:,2]>0
        hw=np.asarray(r['prepared_shape_hw']);wh=hw[::-1]
        decoded=labels[:,:2]*wh-r['reflect_pad_px']
        error=np.abs(decoded[good]-projected[i][good])
        if len(error):
            e=float(error.max());max_error=max(max_error,e);max_by_source[r['source']]=max(max_by_source[r['source']],e)
            # Exact six-decimal training text can differ by <=0.5e-6*padded dimension.
            assert (error<=(wh*.500001e-6)[None,:]).all()
        observed_known+=int(good.sum());unknown_slots+=int((~good).sum());source_label_counts[(r['source'],r['source_split'])]+=1

    configs=read(CFG)['rows'];synthetic=[];normal_exact=0;area_disagreements=[]
    for r in configs:
        xyz=np.asarray(r['native8_canonical_xyz'],np.float64)
        # Fixed canonical Xright/Ydown/Zfar -> proper world Xright/Yfar/Zup.
        zup=xyz[:,[0,2,1]]*np.array([1,1,-1])
        eye=np.asarray(r['camera']['eye_world']);eye_zup=eye[[0,2,1]]*np.array([1,1,-1])
        fixed_xy=np.empty((8,2));fixed_xy[r['permCF_to_canonical']]=np.asarray(r['native8_CF_xy'])
        p,cosine=normal(zup,fixed_xy,eye_zup)
        area=legacy_ns['compute_perm_v4'](zup,fixed_xy)[:8]
        expected=r['permCF_to_canonical'];exact=p.tolist()==expected
        normal_exact+=exact;different=area!=expected
        if different:area_disagreements.append(r['base_index'])
        margin=float(np.sort(cosine)[-1]-np.sort(cosine)[-2])
        synthetic.append(dict(base_index=r['base_index'],split=r['split'],camera_pair_deg=r['factors']['camera_pair_deg'],
            declared_CF_to_canonical=expected,empirical_training_normal_rule_to_canonical=p.tolist(),
            historical_area_rule_to_canonical=area,declared_CF_matches_training_normal_rule=exact,
            normal_maximum_margin=margin,historical_area_disagrees=different))
    assert normal_exact==192 and len(area_disagreements)==22
    out=dict(completed=True,PASS=True,source_only=True,goal_status='active',fully_solved=False,
      headline='Declared V129 CF mapping matches the empirically verified actual R0 training-label ontology on all192known configurations. Historical area-V4 is not the renderer training rule.',
      exact_original_exporter_code_recovered=False,
      certainty='Actual source-label equivalence replay; original C#/Blender writer snapshot and tie policy remain unrecovered. This certifies the effective archived training-label rule on60000records and transfers that fixed rule to nontied known synthetic geometry; not a claim of model prediction correctness.',
      checkpoint_data_chain=dict(cache_backbone=cache['backbone'],cache_backbone_sha256=cache['backbone_sha256'],
          current_frozen_initializer_checkpoint_sha256=read(P/'query_visibility_actual_initializer_source_20261024_v132/ROOT_INITIALIZER_POLICY_BEFORE_INPUTS.json')['checkpoint']['sha256'],
          training_dataset=manifest['dataset'],TRAIN=55980,VAL=4020,
          source_raw_label_coordinates_match_prepared_training_targets_with_only_sixdecimal_rounding=True,
          native_slots_with_known_training_targets=observed_known,native_slots_without_training_target=unknown_slots,
          max_prepared_target_roundtrip_error_px=max_error,max_prepared_target_roundtrip_error_by_source=dict(max_by_source),
          source_label_counts={str(k):v for k,v in source_label_counts.items()}),
      original60000_replay=dict(normal_rule_permutation_exact=60000,mismatches=[],minimum_top_cosine_margin=min(margins),
          counts={str(k):v for k,v in counts.items()},algorithm='Four vertical sides: greatest outward unit-normal cosine to camera from object center; front top ordered by imageX; worldZ vertical pairs; rear preserves depth-edge pairing, not independently Xsorted.'),
      V129_replay=dict(normal_rule_exact=192,TRAIN_exact=128,VAL_exact=64,
          smallest_normal_margin=min(x['normal_maximum_margin']for x in synthetic),rows=synthetic,
          historical_area_V4_different_bases=area_disagreements,TRAIN_historical_area_disagreement=0,VAL_historical_area_disagreement=22),
      report_interpretation='Keep CF_PRIMARY_DECLARED unchanged. Retain old V4 sensitivity for provenance but relabel it obsolete projected-area convention comparison; its22VALdifferences are not evidence the original R0 training ontology was ambiguous on these configured cameras. Do not choose a mapping by model accuracy.',
      limitations=['Original writer executable source/tie policy not recovered.','This does not certify real-human annotation alignment or model output correctness.','No original artifacts, labels, queries or previous reports were changed.'],
      actions=dict(original_source_metadata_reads=True,source_training_manifest_targets_read=True,RGB_reads=0,feature_reads=0,model_weights_loaded=0,model_calls=0,current223=0,V112=0,PnP=0,new_fit=0),
      evidence={p.name:bind(p)for p in[Path(__file__),CONTRACT,PACK,SEAL,CFG,MANIFEST,CACHE,TRAIN,LEGACY,
          REPO/'challenge/yolo_pose_one_model/spatial_concat_scratch/build_dataset.py',
          REPO/'challenge/yolo_pose_one_model/scripts/prepare_yolo_pose.py']},seconds=time.monotonic()-start)
    path=H/'ORIGINAL_INITIALIZER_TRAINING_ORDER_REPLAY.json';assert not path.exists()
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(result=bind(path),source60000_exact=True,prepared_target_slots=observed_known,V129normal192_exact=True,obsolete_area_different=len(area_disagreements),seconds=round(time.monotonic()-start,2))),flush=True)

if __name__=='__main__':run()
