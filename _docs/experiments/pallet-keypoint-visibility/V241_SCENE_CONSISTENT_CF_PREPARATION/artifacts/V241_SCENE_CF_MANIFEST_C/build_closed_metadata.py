"""Root 고정 all ORIGINAL FIT D5406 + old donor64/plan 메타데이터 결합만."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
from collections import Counter

R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H=Path(__file__).absolute().parent
PACKAGE=R/'V236_PUBLICATION_PACKAGE'
READS=[]

def need(ok,message):
    if not ok:raise ValueError(message)

def lexical(value):
    s=str(value);p=Path(s)
    need(p.is_absolute() and '..' not in p.parts and not any(x in s.casefold()
        for x in ('v112','__pycache__','.pyc')), 'forbidden lexical path before FS')
    return p

def read(path,sha=None):
    p=lexical(path)
    for q in list(p.parents)[::-1]+[p]:
        need(not stat.S_ISLNK(os.lstat(q).st_mode),'metadata symlink')
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:b=f.read()
    digest=hashlib.sha256(b).hexdigest();need(sha is None or digest==sha,'metadata SHA')
    READS.append(dict(path=str(p),sha256=digest,size_bytes=len(b)));return b

def identity(b):
    # Asset path is a string only; never stat/hash/read a payload here.
    return str(lexical(b['path'])),b['sha256'],b.get('format'),b.get('member')

def out(name,obj):
    with (H/name).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def lines(name,rows):
    with (H/name).open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')

def binding(path):
    b=path.read_bytes();return dict(path=str(path),sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b))

def run():
    direct=json.loads(read(R/'V237_SOURCE_INPUT_MANIFEST_C/DIRECT_GROUPED_INPUT_MANIFEST.json',
        '68827e79feb8115294f147675130091792b6c3be26b28640a07bdcd402cd7f36'))
    groups=[json.loads(x) for x in read(Path(direct['groups']['path']),direct['groups']['sha256']).splitlines()]
    rows=[json.loads(x) for x in read(Path(direct['all_rows']['path']),direct['all_rows']['sha256']).splitlines()]
    need(len(groups)==1767 and len(rows)==15040,'original full metadata counts')
    bygroup={g['group_index']:g for g in groups};bysample={r['sample_index']:r for r in rows if r['sample_index'] is not None}
    need(len(bysample)==7617 and sum(r['target']==-1 for r in rows)==7423,'known/masked retained')
    parents=[r for r in rows if r['sample_index'] is not None and r['fit'] and r['target']==0]
    need(Counter(r['family_code'] for r in parents)=={0:4565,1:841},'Root fixed allFIT D policy')
    parents.sort(key=lambda r:r['sample_index']);parent_ids={r['sample_index'] for r in parents}
    dev=[r for r in rows if not r['fit']]
    dev_groups={r['group_index'] for r in dev};parent_groups={r['group_index'] for r in parents}
    need(not parent_groups&dev_groups,'original FIT parent vs DEV group overlap')
    dev_images={identity(bygroup[r['group_index']]['image']) for r in dev}
    parent_images={identity(bygroup[r['group_index']]['image']) for r in parents}
    need(not parent_images&dev_images,'original parent image vs DEV overlap')
    manifest=json.loads(read(PACKAGE/'MANIFEST.json','1209aa19c7ec860289f6134e1f42629dce871d2560c2f1045151f6ccb43f58c2'))
    declared={x['path']:x for x in manifest['files']}
    def package_meta(name):
        b=read(PACKAGE/name,declared[name]['sha256']);need(len(b)==declared[name]['bytes'],'package metadata size');return b
    source_map=json.loads(package_meta('PUBLICATION_SOURCE_MAP.json'))
    donor=json.loads(package_meta('artifacts/V235_DONOR_MANIFEST_B/DONOR_WHITELIST.json'))
    oldcomplete=json.loads(package_meta('artifacts/V235_ROOT_PUBLIC_COUNTERFACTUAL/pairs_v1/COMPLETE.json'))
    name='artifacts/V235_ROOT_PUBLIC_COUNTERFACTUAL/pairs_v1/PAIR_METADATA.jsonl.gz'
    encoded=package_meta(name);decoded=gzip.decompress(encoded)
    sm=next(x for x in source_map['files'] if x['path']==name)
    need((hashlib.sha256(decoded).hexdigest(),len(decoded))==(sm['source_sha256'],sm['source_bytes']), 'lossless original pair metadata provenance')
    need(sm['source_sha256']=='feaa7dee841c68aebfc6c1fb1fc2fb35e56a49d538e8191c6c74a02d9d6c468f','old metadata pin')
    oldrows=[json.loads(x) for x in decoded.splitlines()];old={r['parent_sample_index']:r for r in oldrows}
    need(len(oldrows)==len(old)==5406 and set(old)==parent_ids,'all5406 old proposals, not5367success selection')
    donors=donor['donors'];need(len(donors)==64 and len({d['donor_id'] for d in donors})==64,'fixed donor64')
    donorbyid={d['donor_id']:d for d in donors};donor_group=set();donor_images=set()
    for d in donors:
        r=bysample[d['canonical_D_sample_index']];g=bygroup[r['group_index']]
        need(r['sample_index'] in parent_ids and r['row_id']==d['canonical_D_row_id'] and
            r['family_code']==1 and d['original_role']=='TRAIN' and d['internal_source_role']=='INTERNAL_SOURCE_FIT'
            and d['original_associated_r_state']=='D' and d['native_mapping']=='UNKNOWN','donor original FIT D authority')
        need(d['RGB']['path']==g['image']['path'] and d['RGB']['sha256']==g['image']['sha256'],'donor RGB literal group closure')
        donor_group.add(r['group_index']);donor_images.add(identity(g['image']))
        identity(d['visible_mask'])
    need(not donor_group&dev_groups and not donor_images&dev_images,'no SOURCEDEV donors')
    whitelist=donor['exact_future_PNG_whitelist']
    need(len(whitelist)==125 and len({(x['path'],x['sha256']) for x in whitelist})==125,'exact125 donor PNG leaves')
    for x in whitelist:identity(x)
    need(len({(d['RGB']['path'],d['RGB']['sha256']) for d in donors})==61 and
        len({(d['visible_mask']['path'],d['visible_mask']['sha256']) for d in donors})==64,'donor RGB61/mask64')
    result=[];ref_null=0;old_success=Counter();different_obj=Counter()
    for p in parents:
        oldr=old[p['sample_index']];g=bygroup[p['group_index']];s=p['active_slot'];did=oldr['donor_id'];d=donorbyid[did]
        need(oldr['row_id']==p['row_id'] and oldr['family']==p['family_code'] and oldr['original_role']=='FIT'
            and oldr['base_original_state']=='D' and oldr['original_query_xy']==p['q'] and oldr['original_L']==p['L'], 'old/current parent q/L/role authority')
        need(p['q']==g['q8'][s] and p['L']==g['L8'][s] and p['original_query_available']==g['original_available8'][s], 'original q8 address identity')
        need(oldr['donor_RGB']==d['RGB'] and oldr['donor_mask']==d['visible_mask'],'same historical donor choice')
        baseobj=oldr['base_object_id_sampling_only'];donorobj=oldr['donor_object_id_sampling_only']
        need(baseobj is None or baseobj!=donorobj,'same V235 different-object rule')
        different_obj['SYN_object_unknown' if baseobj is None else 'YCB_different_object']+=1
        ref=oldr['teacher_reference_xy'];ref_null+=ref is None;old_success[bool(oldr['success'])]+=1
        result.append(dict(parent_sample_index=p['sample_index'],original_input=p,
            group_index=p['group_index'],group_id=p['group_id'],active_slot=s,
            original_q8_authority='PARENT_GROUPS_INPUT_ONLY.jsonl:group_index; preserve all8 including S/U and original-unavailable finite slots',
            image=g['image'],teacher_reference_xy=ref,teacher_reference_is_supervision_only=True,
            teacher_authority='SYN intended-native witness0 MAIN' if p['family_code']==0 else 'YCB associated-r AUX/nativeUNKNOWN',
            teacher_reference_provenance=dict(packaged_pair_metadata=name,raw_sha256=sm['source_sha256'],
                raw_original_diagnostic_binding=oldcomplete['source_bindings']['diagnostics'] if p['family_code']==0 else oldcomplete['source_bindings']['YCB_reference_diagnostic']),
            expected_BEFORE_RGB96_CHW_sha256=oldr['original_pixel_sha256'],expected_BEFORE_valid96_bool_sha256=oldr['original_valid_sha256'],
            same_V235_donor_id=did,same_V235_donor_RGB=d['RGB'],same_V235_donor_visible_mask=d['visible_mask'],
            same_V235_base_object_id_sampling_only=baseobj,same_V235_donor_object_id_sampling_only=donorobj,
            same_V235_donor_plan=oldr['donor_plan'],old_V235_pair_index=oldr['pair_index'],
            old_V235_success_diagnostic_only=oldr['success'],old_V235_failure_reason_diagnostic_only=oldr['failure_reason'],
            new_full_scene_COVER_SHAM_states='UNKNOWN until newly sampled full-scene teacher proof',
            old_pixel_teacher_proof_inherited=False,native_physical_gold_claim=False))
    lines('PARENTS_ALL5406.jsonl',result)
    lines('PARENT_GROUPS_INPUT_ONLY.jsonl',[bygroup[i] for i in sorted(parent_groups)])
    images=[]
    for key in sorted(parent_images):
        uses=[i for i in sorted(parent_groups) if identity(bygroup[i]['image'])==key]
        images.append(dict(image=bygroup[uses[0]]['image'],parent_group_indices=uses))
    out('PARENT_IMAGE_WHITELIST.json',dict(completed=True,PASS=True,metadata_only=True,images=images,payload_access_authorized=False))
    out('DONOR64_CLOSED_METADATA.json',dict(completed=True,PASS=True,original_manifest_sha256='5259215ea7a53c7df206a25f5e57c01589afef02b98fa4d7aef49e6145838a86',
        donors=donors,exact_future_PNG_whitelist=whitelist,same_donor_rule_and_choice=True,new_donor_selection=False,payload_access_authorized=False))
    census=dict(all_original_rows=15040,known=7617,masked_target_minus1=7423,
        parents=5406,SYN_parents=4565,YCB_parents=841,parent_groups=len(parent_groups),
        parent_groups_by_family=dict(Counter(bygroup[i]['family_code'] for i in parent_groups)),
        parent_image_assets=len(parent_images),parent_images_by_family=dict(Counter(bygroup[next(i for i in parent_groups if identity(bygroup[i]['image'])==k)]['family_code'] for k in parent_images)),
        original_DEV_rows=len(dev),original_DEV_groups=len(dev_groups),parent_DEV_group_overlap=0,parent_DEV_image_overlap=0,
        donor_instances=64,donor_RGB_assets=61,donor_visible_masks=64,donor_DEV_group_overlap=0,donor_DEV_image_overlap=0,
        parent_donor_group_overlap=len(parent_groups&donor_group),parent_donor_image_overlap=len(parent_images&donor_images),
        historical_V235_successful_parents=old_success[True],historical_V235_failed_parents=old_success[False],
        historical_success_is_not_new_selection_or_new_teacher_proof=True,teacher_reference_null_parents=ref_null,
        different_object_rule=dict(different_obj),raw_RGB_NPZ_NPY_GT_or_model_FS_access=0)
    out('CENSUS_AND_OVERLAP.json',dict(completed=True,PASS=True,**census))
    files=['PARENTS_ALL5406.jsonl','PARENT_GROUPS_INPUT_ONLY.jsonl','PARENT_IMAGE_WHITELIST.json','DONOR64_CLOSED_METADATA.json','CENSUS_AND_OVERLAP.json']
    out('MANIFEST.json',dict(completed=True,PASS=True,metadata_only=True,Root_fixed_policy='ALL ORIGINAL FIT knownD parents; no DEV/error/metric selection',
        inputs=READS,outputs=[binding(H/x) for x in files],census=census,
        original_direct_manifest=direct['schema'],raw_payload_access_authorized=False,
        before_proof_required='New BEFORE5406 active RGB96/valid hashes must match expected inherited originals. Actual fullscene proof not yet executed.',
        actual_vs_anchor='Use group.image actual RGB only. Preserve original q8; never replace with GT/anchor image or derive q8 from clipped condition32. Direct records do not expose separate actual-vs-anchor indices.',
        source_role='SOURCE FIT parents and donors only. Input-only all8 nodes retained; teacher/r/object identifiers are bookkeeping/supervision, no forward features.',
        SOURCE_DEV_current_or_held_parent_or_donor_selected=False,new_CF_or_head_fit=False,model_data_architecture_selection=False))
    print(json.dumps(dict(completed=True,PASS=True,census=census,manifest=binding(H/'MANIFEST.json')),ensure_ascii=False))

if __name__=='__main__':run()
