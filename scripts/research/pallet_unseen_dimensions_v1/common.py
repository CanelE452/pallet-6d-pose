"""Bounded, synthetic unseen-mesh architecture screen (not a paper FINAL)."""
from pathlib import Path
import hashlib, json
ROOT=Path(__file__).resolve().parents[3]
KIT=Path(__file__).resolve().parent
OUT=ROOT/'data/pallet/results/pallet_unseen_dimensions_v1'
BASE=ROOT/'challenge/yolo_pose_one_model/spatial_concat_scratch'
CKPT=BASE/'runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt'
ASSETS=['scene.usd','scene_1.usd','scene_2.usd','scene_3.usd']
ARMS=['A_points','B_concat','C_geometry','D_no_relations']
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def register():
    p=OUT/'PROTOCOL.json'
    if p.exists():return read(p)
    assert sha(CKPT)=='970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'
    rows=[json.loads(l) for l in (BASE/'PROBE_METADATA_60K.jsonl').open()]
    source_assets={r.get('source_asset') or 'scene.usd' for r in rows}
    assert not set(ASSETS[2:]) & source_assets
    o=dict(schema='unseen_dimensions_v1',scope='exploratory synthetic unseen-mesh screen',
        source_assets=sorted(source_assets),train_assets=ASSETS[:2],unseen_assets=ASSETS[2:],
        upstream_checkpoint_sha256=sha(CKPT),metadata_sha256=sha(BASE/'PROBE_METADATA_60K.jsonl'),
        legacy_asset_evidence='_docs/experiments/pallet_translation_loss_v1/LOSS_SYMMETRY_CONTRACT.json',
        train_groups_per_seen_asset=64,val_groups_per_seen_asset=8,test_groups_per_asset=12,
        views_per_group=4,conditions=['clean','panel','crop25'],seeds=[1,2,3],arms=ARMS,
        steps=1200,batch=32,lr=.0003,weight_decay=.0001,selection='fixed final update; no test selection',
        conditioning='known fixed object XYZ dimensions, NOT camera-facing width/depth; two possible footprint assignments retained without GT selection',
        model='frozen strong YOLO plus identity-initialized spatial keypoint residual; shared image attention; B concat; C dimension-derived pairwise 3D relations; D same dimension input without relations',
        geometry='object world Z up, X/Y footprint, centered origin; K updated through crop and letterbox; camera-dynamic corner IDs; no physical symmetry assumption for new assets',
        loss='same raw-pixel smooth L1 (beta 4) on eight corners; cap residual at 64 raw pixels via tanh; no line, symmetry or pose auxiliary loss',
        evaluation='GT-free top confidence .25; IoU .5 only for scoring; all frames retained, misses assigned raw diagonal; paired scene-cluster bootstrap; report conditional matched errors separately',
        primary='equal weight unseen assets and conditions, mean corner error divided by raw image diagonal, capped at one',
        screen_support=dict(C_vs_B_relative_gain_each_seed=.02,bootstrap_upper_delta_C_minus_B_below_zero=True,
            C_beats_A_and_stock_each_seed=True,each_asset_condition_no_more_than_2pct_worse_than_B=True),
        controls=['stock YOLO','same-renderer seen-mesh test','D relation removal','neural dimensions shuffled with PnP dimensions correct','5 percent dimension noise'],
        pose_limit='metric translation via GT-free cuboid correspondences is diagnostic; no semantic rotation claim without physical role resolution',
        limits=['New meshes to this checkpoint, not newly collected real pallets. Assets exist in historical research pools: exploratory, not untouched paper FINAL.',
                'Only two unseen mesh instances; scene bootstrap does not establish population-level unseen-type confidence.',
                'Frozen-backbone residual architecture; not full-network finetuning or proof of universal impossibility.',
                'Cuboid relation encoding is not by itself a novel pallet-specific contribution.'])
    write(p,o);return o
