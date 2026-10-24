"""Freeze TRAIN-only source sample before opening any source RGB."""
from pathlib import Path
import csv, hashlib, json, collections

BASE = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
OLD = BASE / 'query_visibility_visible_preservation_20261023_v67/source_audit'
OUT = BASE / 'query_visibility_reference_semantic_20261023_v69/source_anchor_review'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    target = OUT / 'SELECTION_AND_REVIEW_PROTOCOL.json'
    assert not target.exists(), 'Immutable selection already exists'
    frames = [json.loads(s) for s in (OLD/'SYNTH_REFERENCE_FRAME_MANIFEST.jsonl').read_text().splitlines()]
    train = [r for r in frames if r['split'].lower() == 'train']
    key = lambda r: (hashlib.sha256(r['id'].encode()).hexdigest(), r['id'])
    assets = sorted({r['source_asset'] for r in train})
    chosen = []
    for asset in assets:
        chosen.extend(sorted([r for r in train if r['source_asset'] == asset], key=key)[:2])
    ids = {r['id'] for r in chosen}
    chosen.extend(sorted([r for r in train if r['id'] not in ids], key=key)[:8-len(chosen)])
    chosen = sorted(chosen, key=lambda r: (r['source_asset'], key(r)))
    anchors = collections.defaultdict(list)
    with (OLD/'SYNTH_REFERENCE_ANCHOR_INVENTORY.csv').open() as f:
        for row in csv.DictReader(f):
            if row['id'] in {r['id'] for r in chosen}:
                assert row['split'].lower() == 'train'
                anchors[row['id']].append(row)
    for row in chosen:
        row['selection_sha256_of_frameID'] = key(row)[0]
        row['anchors8'] = sorted(anchors[row['id']], key=lambda r: int(r['native_corner']))
        assert [int(r['native_corner']) for r in row['anchors8']] == list(range(8))
    output = {
        'schema': 'SYNTH_REFERENCE_RGB_REVIEW_SELECTION_V1',
        'status': 'FROZEN_BEFORE_ANY_RGB_DECODE_IN_THIS_PILOT',
        'source_bindings': [{'path':str(OLD/name),'sha256':digest(OLD/name)} for name in [
            'SYNTH_REFERENCE_FRAME_MANIFEST.jsonl','SYNTH_REFERENCE_ANCHOR_INVENTORY.csv','SYNTH_REFERENCE_VISIBILITY_INTAKE.json']],
        'selection_rule': 'TRAIN only. Per available TRAIN source asset take 2 smallest SHA256(UTF8 id), then fill to8 from remaining TRAIN with global smallest same SHA. Final display ordered by asset then SHA. No visibility, image content, proxy state, crop validity or model score used.',
        'train_asset_population': dict(collections.Counter(r['source_asset'] for r in train)),
        'four_asset_balance_possible': False,
        'fourth_asset_note': 'Only3 source assets have TRAIN records. No VAL image is used to fill the missing4th asset.',
        'selected_frame_count':len(chosen), 'selected_anchor_count':8*len(chosen),
        'selected_asset_counts':dict(collections.Counter(r['source_asset'] for r in chosen)),
        'reference_mapping': 'projected_cuboid already camera_dynamic_0123_v4:0 frontTL,1 frontTR,2 frontBR,3 frontBL;4..7 corresponding back; index8 centroid excluded. perm_v4 maps dynamic to original and must not be applied again.',
        'query_mapping': 'Saved initial q equals V60 initial_points, SEALED_ACTUAL_N3_SUBPIX_F; no new jitter or coordinates refined.',
        'visualization_recipe': {'reference_marker':'cyan plus and r-index','query_marker':'magenta ring and q-index','line':'yellow segment between r and q','reference_crop':'native96x96 centered at floor(r+0.5); out-of-image raster padded dark; no point moved','zoom':'nearest2x for crop, original RGB remains raster background','per_frame':'all8 labels and all8 r-centered crops regardless of visibility/proxy'},
        'review_tags': ['D_candidate','E_candidate','S_proxy_only','ambiguous','virtual-no-cue'],
        'review_evidence_tier': 'AI_OBSERVATIONAL_CONDITIONAL_TEACHER_CANDIDATE_ONLY_NOT_HUMAN_VERIFIED_NOT_GOLD_NOT_TRAINING_ADMITTED',
        'tag_scope': 'D_candidate means an observed physical target corner cue near projected r, not GT-mask certification. E_candidate requires visible distinct external cover evidence but remains conditional. S_proxy_only reports existing convex self proxy with no directly observable landmark. virtual-no-cue means projected cuboid extremum lacks an identifiable physical corner cue; not a semantic absence label. Out-of-frame/occluder ambiguity remains ambiguous unless noted as proxy-only.',
        'read_scope': {'RGB_before_this_lock':0,'source_RGB_after_lock':8,'source_VAL_RGB':0,'current_data':0,'mask_based_D_certification':False,'model_or_new_forward':False,'render_train_PnP':False},
        'frames':chosen,
    }
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'path':str(target),'sha256':digest(target),'selected':[(r['id'],r['source_asset']) for r in chosen]},ensure_ascii=False))

if __name__ == '__main__':
    main()
