from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[3]
KIT=Path(__file__).resolve().parent
PREVIOUS=ROOT/'data/pallet/results/pallet_unseen_dimensions_v1'
OUT=ROOT/'data/pallet/results/pallet_role_selection_v2'
BASE=ROOT/'challenge/yolo_pose_one_model/spatial_concat_scratch'
CKPT=BASE/'runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt'
ASSETS=['scene.usd','scene_1.usd','scene_2.usd','scene_3.usd']
ARMS=['A_direct','B_select_only','C_select_residual','D_select_residual_no_dims']
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,o):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def register():
    path=OUT/'PROTOCOL.json'
    if path.exists():return read(path)
    parent=read(PREVIOUS/'PROTOCOL.json')
    for rel,digest in read(PREVIOUS/'COMPLETION.json')['files'].items():assert sha(ROOT/rel)==digest,rel
    o=dict(schema='role_selection_residual_v2',parent_sha256=sha(PREVIOUS/'COMPLETION.json'),
        upstream_checkpoint_sha256=parent['upstream_checkpoint_sha256'],seeds=[1,2,3],arms=ARMS,
        steps=1200,batch=32,lr=.0003,weight_decay=.0001,selection='fixed final checkpoint; no threshold sweep',
        train='previous1536 images, scene.usd/scene_1.usd only; same frozen observation caches',
        old_test='previous576 images permanently exploratory DEV',
        fresh_test='576 newly randomized observations; seed651000 versus previous451000; same two seen and two unseen meshes',
        train_groups_per_seen_asset=0,val_groups_per_seen_asset=0,test_groups_per_asset=12,views_per_group=4,
        conditions=['clean','panel','crop25'],render_seed_base=651000,
        architecture='Four whole-object corner-ID hypotheses, shared expected-error selector, separate small residual branch. Select exactly one candidate; no coordinate averaging. All candidate point features and visibility follow the same permutation. Centre remains index8.',
        candidates='Four horizontal cuboid ID permutations are error hypotheses, NOT assertions of physical C4 symmetry.',
        selector='Predict pre-residual mean corner error / observed point extent diagonal; smooth L1 beta.05. Choose minimum predicted error in pixels plus fixed2px nonidentity switching cost. No GT at inference.',
        residual='Independent parameters from selector; each-coordinate16px tanh. Train on the lowest pre-correction GT-error candidate; A trains on identity. Smooth L1 beta4 /16. Tie chooses lowest candidate index. No targets relabeled by an inference-time oracle.',
        dimensions='Fixed world XYZ known size. D zeros neural dimensions; other arms use them. PnP always gets correct sizes.',
        initialization='Selector output and residual output zero; switching cost retains original order and exact YOLO output.',
        controls=['stock','previous geometry architecture frozen','observation-only deterministic canonicalization','A direct residual','B selection without correction','C selection and residual','D C without dimensions','C selector removed at inference','C residual removed at inference','neural dimensions shuffled'],
        primary='all-frame mean8corner error / raw image diagonal, capped1; common .25 confidence/.5IoU matching; failed frames penalty1',
        acceptance=dict(C_vs_stock_each_seed_relative_gain=.02,C_vs_A_each_seed_relative_gain=.02,
            paired_scene_cluster_C_minus_stock_CI_upper_below_zero=True,
            each_seen_unseen_condition_primary_no_more_than_2pct_worse_than_stock=True,
            no_new_gt20px_frames_from_stock_le10px_each_seed=True),
        supplementary='Matched errors, role-selection regret, good-frame harms, conditional cases, known-size PnP translation diagnostic. No semantic rotation claim.',
        constraints=['Frozen YOLO cannot repair missed detections.','Only two unseen meshes, shared with previous experiment; new scenes are not new real pallets.',
            'Discrete branch training uses separate supervision; no gradient through argmin, no gradient surgery.','No line/Hough branch in this causal architecture test.'])
    write(path,o);return o
