"""Read-only split and interpretation plan. No RGB/mask/model/score reads."""
from pathlib import Path
from collections import Counter
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()]

inputs=rows(ROOT/'pair_generation/INPUTS_RGB_ONLY.jsonl')
lineage=rows(ROOT/'pair_generation/PAIR_LINEAGE_GT_ONLY.jsonl')
assert len(inputs)==len(lineage)==4799
material={p:set()for p in ('train','val')}
donors={p:set()for p in ('train','val')}
donor_images={p:set()for p in ('train','val')}
donor_objects={p:set()for p in ('train','val')}
target_ids={p:set()for p in ('train','val')}
target_images={p:set()for p in ('train','val')}
scene={p:set()for p in ('train','val')}
count=Counter()
original_links=rows(ROOT.parent/'query_visibility_reference_local_support_20261023_v76/source_inventory/UNCHANGED_MODEL_INPUT_LINKS.jsonl')
public_links=rows(ROOT.parent/'query_visibility_reference_local_support_20261023_v76/public_query_emulation/INPUTS_RGB_ONLY.jsonl')
for a,l in zip(inputs,lineage):
    assert a['id']==l['after_id']and a['frame_index']==l['after_frame_index']
    p=a['split'];d=l['donor'];assert d['split']==p
    count[(a['source'],p)]+=1
    original=(original_links[l['original_row_index']] if l['original_row_index']<1003 else public_links[l['original_row_index']-1003])
    # The two immutable input schemas have different raw-image field names.
    image_hash=(original.get('image_sha256') or original.get('raw_RGB_sha256') or original.get('raw_rgb_sha256'))
    if image_hash is None:
        for k,v in original.items():
            if 'sha' in k.lower()and('rgb' in k.lower()or 'image' in k.lower()):
                image_hash=v;break
    assert isinstance(image_hash,str)and len(image_hash)==64, list(original)
    target_ids[p].add(l['original_id']);target_images[p].add(image_hash)
    donors[p].add(d['id']);donor_images[p].add(d['image_sha256']);donor_objects[p].add(d['object_id'])
    material[p].update((image_hash,d['image_sha256']))
    if a['source']=='YCB_PUBLIC_REAL':scene[p].add(int(l['original_id'].split(':')[1]))

out=dict(
    completed=True,scope='Metadata-only before reviewing any V77 outcomes.',
    input_bindings=[bind(ROOT/'EXPERIMENT_DECISIONS_BEFORE_PAIR_BUILD.json'),bind(ROOT/'pair_generation/PAIR_BUILD_PROTOCOL_BEFORE_COMPOSITION.json'),bind(ROOT/'pair_generation/SEAL.json'),bind(ROOT/'pair_generation/INPUTS_RGB_ONLY.jsonl'),bind(ROOT/'pair_generation/PAIR_LINEAGE_GT_ONLY.jsonl')],
    after_rows=4799,by_source_split=[dict(source=s,split=p,rows=n)for(s,p),n in sorted(count.items())],
    unique_target_ids={p:len(v)for p,v in target_ids.items()},unique_recipient_image_byte_SHAs={p:len(v)for p,v in target_images.items()},
    unique_donor_instance_ids={p:len(v)for p,v in donors.items()},unique_donor_image_byte_SHAs={p:len(v)for p,v in donor_images.items()},
    public_target_capture_scenes={p:sorted(v)for p,v in scene.items()},
    cross_split_intersections=dict(target_ids=len(target_ids['train']&target_ids['val']),recipient_image_byte_SHAs=len(target_images['train']&target_images['val']),donor_instance_ids=len(donors['train']&donors['val']),donor_image_byte_SHAs=len(donor_images['train']&donor_images['val']),all_recipient_and_donor_image_byte_SHAs=len(material['train']&material['val'])),
    donor_object_ids={p:sorted(v)for p,v in donor_objects.items()},donor_object_ids_shared_across_splits=sorted(donor_objects['train']&donor_objects['val']),
    interpretation_fixed_before_scores=[
       'Natural validation is also the best-checkpoint selection set. It is not a new untouched test.',
       'Pair validation is not used by checkpoint selection, but uses the same composition generator as training.',
       'Same q/L/box before-after pairs test response to changed RGB at fixed geometry; they do not test jitter sensitivity or native semantic reference association.',
       'C-anchor plus V-nonanchor decisions in the SAME after RGB can challenge a constant global any-occluder shortcut. They do not isolate which input branch caused success.',
       'Natural validation preservation is required alongside after-pair accuracy; all-covered output can win some pair-negative metrics while losing visible support.',
       'Seams/opacity/EDT scaling and composition artifacts remain available at train and validation. Pair success alone cannot rule out those shortcuts.',
       'Disjoint donor images/captures are distinct from disjoint object categories or physical object identities. The same YCB object ID may occur in both splits.',
       'V77 keeps architecture, initialization, batch96 and 2000-update optimizer budget fixed against V76, but changes training data and mixture allocation; the effect is the combined data/sampling intervention.',
       'Public q is source emulation derived from refs plus a frozen SYNTH TRAIN error bank, not deployed initializer output. Teacher labels describe conditional reference-local support, not human semantic D/E/S.',
       'Synthetic validation includes 84 images previously used by V75 pretrain checkpoint selection. No V75 TRAIN overlap was found by the prior immutable inventory.',
    ],
    requested_result_views=['natural TRAIN/VAL by source and class versus frozen V76','pair TRAIN/VAL both-correct and pBefore>pAfter','same-after anchorC plus nonanchorV preservation','same-after known nonanchor confusion','source/class counts and unresolved/missing admission','train versus VAL gaps on identical metric definitions'],
    forbidden_conclusions=['natural semantic transfer solved','causal E/O/self classification learned','query jitter tolerance measured','realistic natural-occlusion or object-category generalization established'],
    RGB_mask_model_score_current_reads=0,new_forward_fit_PnP=0,code=bind(Path(__file__)),
)
p=ROOT/'source_interpretation_review/METADATA_AND_REVIEW_PLAN_BEFORE_RESULTS.json'
with open(p,'x')as f:f.write(json.dumps(out,indent=2)+'\n')
print(json.dumps(bind(p)))
