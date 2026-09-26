"""Opaque hash provenance, then metadata/RGB-only inventory and locked temporal queue.

No detector, teacher, label-coordinate, pose or scoring module is imported.
Historical metadata is used only for image/recording exclusion, never hard ranking.
"""
from collections import Counter
from pathlib import Path
import json
import time
import cv2
import numpy as np
from . import common as C
from .policy import temporal_queue

SPLIT = '_docs/experiments/pallet_existing_data_transfer_v1/SPLIT_LOCK.json'
GROUPS = 'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json'
POOL = 'data/evaluation/pallet_eval_v1/adaptation/ADAPTATION_POOL_LOCK.json'
OLD_EXCLUSION = 'data/pallet/results/pallet_min_hard_labels_v1/CANDIDATES_PRIVATE.json'
ANCHOR = 'data/pallet/results/pallet_verified_anchor_v1/ANCHOR_SELECTION.json'
OLD_POOL = '_docs/experiments/pallet_type_selftrain_v1/POOL.json'
META = [SPLIT, GROUPS, POOL, OLD_EXCLUSION, ANCHOR, OLD_POOL,
        'data/pallet/eval_results/split_lock/split_assignment.json']
IMAGE_EXT = {'.png', '.jpg', '.jpeg'}

def preflight():
    if (C.DOC / 'INPUT_BINDINGS.json').exists():
        for b in C.read(C.DOC / 'INPUT_BINDINGS.json')['files']:
            C.verify(b)
        return
    assert C.git('branch', '--show-current') == 'main'
    C.save(C.RAW / 'WORKTREE_START.json', dict(head=C.git('rev-parse','HEAD'), branch='main',
           status=C.git('status','--short','--branch'), created_at=C.now()), immutable=True)
    upstream = C.read(C.ROOT / '_docs/experiments/pallet_single_model_preserve_v1/INPUT_BINDINGS.json')
    # Binary hashing is not deserialization/scoring; no model output values are consumed.
    files = {b['path']: b for b in upstream['files']}
    for ns in ('pallet_single_model_preserve_v1', 'pallet_min_hard_labels_v1'):
        for p in (C.ROOT / '_docs/experiments' / ns).rglob('*'):
            if p.is_file():
                files[str(p.relative_to(C.ROOT))] = C.bind(p)
    for name in META:
        files[name] = C.bind(C.ROOT / name)
    for b in files.values():
        C.verify(b)
    args = C.read(C.ROOT / '_docs/experiments/pallet_clean19_structured_easyhard_v1/PREFLIGHT.json')['args']
    fit = C.read(C.ROOT / '_docs/experiments/pallet_clean19_structured_easyhard_v1/FIT_PLASTIC_S1.json')
    C.save(C.DOC / 'INPUT_BINDINGS.json', dict(head=C.git('rev-parse','HEAD'), branch='main',
           files=list(files.values()), created_at=C.now(), model_output_values_consumed=0,
           provenance_hashing='opaque bytes only; no model/checkpoint/reference deserialization'), immutable=True)
    C.save(C.DOC / 'PROTOCOL_LOCK.json', dict(created_at=C.now(), baseline=fit['checkpoint'],
           selector='frozen GEO_LINEAR', args=args, seed=42, updates=320, epochs=5,
           real_per_epoch=512, clean_per_epoch=448, hard_per_epoch=64, synthetic_per_epoch=512,
           replace='lowest64 sha256(min-hard-slot-v1:<epoch>:<slot>) of original real512 each epoch',
           manual_vs_pseudo='same RGB/bbox/augmentation/occurrence/support; only DIRECT_VISIBLE xy source differs',
           pseudo_finite_coverage_min=.9, no_new_baseline_fit=True, last_only=True,
           hard_loss='xy on manual DIRECT_VISIBLE only; hard box/cls/dfl/visibility and hidden/P8 ignored',
           queue='max48 temporal bins/recording; sha256(hard-tag-v1:<frame_id>) min per bin; rounds bin%3+1',
           near_duplicate='OpenCV RGB->GRAY then default INTER_LINEAR resize64x48, MAD <=2 intensity units',
           input_policy='adaptation-only; all HELDOUT recording IDs excluded; reserved raw recordings excluded',
           extra_unused_guard='Entire old balanced PLASTIC1000 pool excluded by SHA, including untrained members, to avoid reusing historical selftraining frames without reading predictions.',
           decision_rules={
               'DEPLOYABLE':'hard CURRENT improves in >=1 stratum, other hard/CLEAN nondecreasing; improving stratum PCK10 or oracle also improves',
               'LOCALIZATION_SELECTOR_LIMIT':'hard PCK10/oracle signal not transmitted to CURRENT',
               'HARD_GAIN_CLEAN_TRADEOFF':'hard signal but CLEAN CURRENT decreases',
               'NO_CONSISTENT_GAIN':'no consistent hard signal or opposing severity changes; stop further labels'},
           no_new_labels_training_before_human_lock=True, independent_test=False), immutable=True)
    C.save(C.DOC / 'PREFLIGHT_AUDIT.md', '# Model-blind hard A/B preflight\n\n'
           f"HEAD `{C.git('rev-parse','HEAD')}` / main. Upstream {len(files)} files hash-bound. "
           'Baseline S1 and GEO_LINEAR unchanged. Model predictions and coordinate GT are not used for this inventory. '
           'Only image identities/recording metadata are extracted from historical manifests for exclusions. '
           'No inference or training occurs before human tagging and label lock.\n')

def thumbnail(p):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError('Unreadable image: ' + str(p))
    return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (64,48)).astype(np.int16)

def coarse(a):
    return a.reshape(4,12,4,16).mean((1,3)).reshape(16)

class MADIndex:
    """Exact L1 <=2 search with Jensen lower bound pruning, no approximation."""
    def __init__(self, size):
        self.full = np.empty((max(1,size),48,64), np.int16)
        self.low = np.empty((max(1,size),16), np.float64)
        self.n = 0
        self.ids = []
    def add(self, a, identity):
        self.full[self.n] = a; self.low[self.n] = coarse(a)
        self.ids.append(identity); self.n += 1
    def match(self, a):
        lower = np.abs(self.low[:self.n] - coarse(a)).mean(1)
        candidates = np.flatnonzero(lower <= 2 + 1e-9)
        for start in range(0, len(candidates), 128):
            idx = candidates[start:start+128]
            distances = np.abs(self.full[idx] - a).mean((1,2))
            hits = np.flatnonzero(distances <= 2)
            if len(hits):
                k = int(hits[0]); return self.ids[int(idx[k])], float(distances[k])
        return None

def inventory():
    if (C.DOC / 'CANDIDATE_POOL_AUDIT.json').exists():
        a = C.read(C.DOC / 'CANDIDATE_POOL_AUDIT.json'); C.verify(a['private_inventory']); return
    start = time.monotonic()
    split = C.read(C.ROOT / SPLIT)
    groups = C.read(C.ROOT / GROUPS)['groups']
    aliases = {s['session_key']:g['recording_id'] for g in groups if not g['is_collection'] for s in g['sessions']}
    old = C.read(C.ROOT / OLD_EXCLUSION)
    pool = C.read(C.ROOT / POOL)
    # Identity-only metadata. Never load TARGETS, predictions, annotation JSON or PnP.
    excluded = {r['image']['sha256']:r['image'] for r in split['train'] + split['heldout']}
    anchor = {r['image']['sha256']:r['image'] for r in C.read(C.ROOT / ANCHOR)['frames']}
    excluded.update(anchor)
    assert set(old['excluded_sha']) <= set(excluded), 'Unresolved old exclusion identity'
    heldrec = set(split['heldout_recordings'])
    reserved = set(old['reserved_recordings'])
    old_pool = {r['image']['sha256']:r['image'] for r in C.read(C.ROOT / OLD_POOL)['records'] if r['kind']=='PLASTIC'}
    train = {}
    train_list = C.ROOT / 'data/pallet/results/pallet_clean19_structured_easyhard_v1/dataset/PLASTIC/train.txt'
    for name in set(train_list.read_text().splitlines()):
        b = C.bind(Path(name)); train[b['sha256']] = b
    # Verify every forbidden image identity before RGB near-duplicate comparison.
    forbidden = dict(excluded)
    reserved_paths = set()
    for g in groups:
        if g['recording_id'] not in reserved:
            continue
        found = []
        for s in g['sessions']:
            folder = C.ROOT / s['session_key'] / 'rgb'
            if folder.exists():
                found.extend(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)
        assert found, f'No RGB for reserved {g["recording_id"]}'
        reserved_paths.update(found)
    for i, p in enumerate(sorted(reserved_paths)):
        b = C.bind(p); forbidden.setdefault(b['sha256'], b)
        if i % 1000 == 0:
            print('RESERVED_HASH', i, len(reserved_paths), flush=True)
    forbidden_index = MADIndex(len(forbidden))
    for sha, b in forbidden.items():
        C.verify(b); forbidden_index.add(thumbnail(C.ROOT / b['path']), sha)
    source = []; source_counts = Counter(); rejects = Counter(); details = []; seen = set()
    for condition, parent in [('daytime','outside'),('nighttime','night')]:
        for session in pool['condition_audit'][condition]['sessions']:
            folder = C.ROOT / 'data/pallet/raw_data' / parent / session / 'rgb'
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() not in IMAGE_EXT:
                    continue
                rec = aliases.get(str(p.parent.parent.relative_to(C.ROOT)))
                source_counts[str(rec)] += 1
                b = C.bind(p); sha = b['sha256']
                reason = ('DUPLICATE_SHA' if sha in seen else
                          'HELDOUT_RECORDING' if rec in heldrec else
                          'RESERVED_RECORDING' if rec in reserved else
                          'EVAL_ANCHOR_CLEAN10_H10_SHA' if sha in excluded else
                          'CURRENT_TRAINING_SHA' if sha in train else
                          'HISTORICAL_BALANCED_POOL_SHA' if sha in old_pool else
                          'RESERVED_IMAGE_SHA' if sha in forbidden else
                          'UNKNOWN_RECORDING' if rec is None else None)
                seen.add(sha)
                if reason:
                    rejects[reason] += 1; details.append(dict(image=b,reason=reason)); continue
                assert p.stem.isdecimal(), 'Temporal ordering needs explicit metadata: ' + str(p)
                source.append(dict(frame_id=f'{session}:{p.stem}', image=b, recording=rec, order=int(p.stem)))
    assert sum(source_counts.values()) == sum(s['images_found'] for s in pool['condition_audit'].values())
    retained_index = MADIndex(len(source)); retained = []
    for i, r in enumerate(sorted(source, key=lambda r:(r['image']['sha256'],r['frame_id']))):
        try:
            t = thumbnail(C.ROOT / r['image']['path'])
        except ValueError:
            rejects['UNREADABLE'] += 1; details.append(dict(frame_id=r['frame_id'],reason='UNREADABLE')); continue
        hit = forbidden_index.match(t)
        reason = 'NEAR_DUP_EVAL_RESERVED_OR_PROTECTED' if hit else None
        if not hit:
            hit = retained_index.match(t)
            reason = 'NEAR_DUP_WITHIN_POOL' if hit else None
        if hit:
            rejects[reason] += 1; details.append(dict(frame_id=r['frame_id'],reason=reason,match=hit)); continue
        retained.append(r); retained_index.add(t,r['frame_id'])
        if i % 500 == 0:
            print('POOL_MAD', i, len(source), 'kept',len(retained),flush=True)
    C.save(C.RAW / 'EXCLUSION_IDENTITIES_PRIVATE.json', dict(protected_images=list(forbidden.values()),
           current_training_images=list(train.values()), historical_pool_images=list(old_pool.values()),
           heldout_recordings=sorted(heldrec),reserved_recordings=sorted(reserved),
           anchor_sha=sorted(anchor), exclusion_details=details), immutable=True)
    C.save(C.RAW / 'CANDIDATE_POOL_PRIVATE.json', dict(rows=retained), immutable=True)
    counts = Counter(r['recording'] for r in retained)
    C.save(C.DOC / 'CANDIDATE_POOL_AUDIT.json', dict(created_at=C.now(),status='INVENTORY_COMPLETE',
           source_frames=sum(source_counts.values()),source_recordings=dict(source_counts),
           eligible_frames=len(retained),eligible_recordings=dict(counts),exclusions=dict(rejects),
           SHA_overlap_after_exclusion=0,near_duplicate_overlap_after_exclusion=0,
           protected_thumbnails=len(forbidden),current_training_unique=len(train),old_pool_unique=len(old_pool),
           model_outputs_opened=0,coordinate_GT_opened=0,raw_RGB_only=True,
           internal_near_duplicate_order='greedy ascending image SHA256; retained pairs all MAD>2',
           protected_scope='full reserved recording RGB + HELDOUT128 + anchor + current train split metadata',
           extra_guard='Exclude all old PLASTIC balanced pool1000 by SHA; conservative superset of historical training membership, not a claim all1000 were trained.',
           elapsed_seconds=time.monotonic()-start,metadata_sources=[C.bind(C.ROOT/p) for p in META],
           training_list=C.bind(train_list),private_inventory=C.bind(C.RAW/'CANDIDATE_POOL_PRIVATE.json')),
           immutable=True)

def make_queue():
    if (C.DOC / 'DIFFICULTY_QUEUE_LOCK.json').exists():
        C.queue()
        if not (C.RAW/'STATE.json').exists():
            C.set_state('WAITING_FOR_HUMAN_DIFFICULTY_TAGS',round=1,rounds_completed=0,
                        command='python -m scripts.research.pallet_min_hard_ab_v1.tag_difficulty',
                        training='NOT_RUN',annotation='NOT_RUN')
        return
    rows = C.read(C.RAW / 'CANDIDATE_POOL_PRIVATE.json')['rows']
    result = temporal_queue(rows)
    C.save(C.RAW / 'DIFFICULTY_QUEUE_PRIVATE.json', dict(rows=result), immutable=True)
    counts = {str(i):dict(Counter(r['recording'] for r in result if r['round']==i)) for i in (1,2,3)}
    C.save(C.DOC / 'DIFFICULTY_QUEUE_LOCK.json', dict(created_at=C.now(),
           private_queue=C.bind(C.RAW/'DIFFICULTY_QUEUE_PRIVATE.json'),round_counts=counts,
           total=len(result),policy=C.bind(C.DOC/'PROTOCOL_LOCK.json'),model_outputs_opened=0,
           tags_exist_at_lock=False), immutable=True)
    C.set_state('WAITING_FOR_HUMAN_DIFFICULTY_TAGS',round=1,rounds_completed=0,
                command='python -m scripts.research.pallet_min_hard_ab_v1.tag_difficulty',
                training='NOT_RUN',annotation='NOT_RUN')

def main():
    with C.exclusive('prepare'):
        preflight(); inventory(); make_queue()
    print('WAITING_FOR_HUMAN_DIFFICULTY_TAGS',C.read(C.DOC/'DIFFICULTY_QUEUE_LOCK.json')['round_counts']['1'])

if __name__ == '__main__':
    main()
