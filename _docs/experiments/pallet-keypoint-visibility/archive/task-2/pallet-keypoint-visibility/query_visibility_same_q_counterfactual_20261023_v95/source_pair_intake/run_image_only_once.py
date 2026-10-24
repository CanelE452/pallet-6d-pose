"""Fixed public-source image-only intake. Never imports torch/model/feature code."""
from pathlib import Path
import collections
import hashlib
import json
import time
import numpy as np
from PIL import Image

import protected_union_adapter as adapter
from normalize_and_lock import bind, read, lines, write, jsonl, HERE, P, V94

OUT = HERE / 'image_only_v1'


def pixel_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def check(d):
    assert hashlib.sha256(Path(d['path']).read_bytes()).hexdigest() == d['sha256'], d['path']


def main():
    protocol = read(HERE / 'INTAKE_PROTOCOL_BEFORE_PIXELS.json')
    assert protocol['root_approved_actual_source_image_only'] is True
    for d in protocol['code'].values():
        check(d)
    check(protocol['metadata_lock'])
    OUT.mkdir(exist_ok=False)
    write(OUT / 'STARTED.json', dict(protocol=bind(HERE / 'INTAKE_PROTOCOL_BEFORE_PIXELS.json'),
                                    model_P3_weights_fit_current_PnP=0))
    (OUT / 'donor_cutouts').mkdir(); (OUT / 'edited_fullRGB').mkdir(); (OUT / 'pair_geometry').mkdir()
    rows = lines(HERE / 'ALL960_NORMALIZED_INPUT_ONLY.jsonl')
    decisions = lines(HERE / 'CONSTRUCTION960_GT_ONLY.jsonl')
    source = lines(HERE / 'SOURCE960_LINEAGE_ONLY.jsonl')
    assigned = lines(HERE / 'ONE_DONOR_ASSIGNMENT960_LOCK_BEFORE_PIXELS.jsonl')
    donor_pool = lines(V94 / 'paired_source_preparation/YCB4200_DONOR_METADATA_CANDIDATES_NOT_ADMITTED.jsonl')
    original_loco = read(P / 'query_visibility_same_q_counterfactual_20261023_v95/loco_before_input_pack/INPUT_PACK_SEAL.json')
    compact = P / 'query_visibility_same_q_counterfactual_20261023_v95/loco_before_input_pack/BEFORE_INPUTS160.npz'
    compact_binding = original_loco['inputs']
    assert compact_binding['path'] == str(compact) and original_loco['PASS']
    # Descriptor only; no P3/RGB96 tensor body is opened by this intake.
    pm = read(V94 / 'pallox_actual_query_input_pack/PACK_MANIFEST.json')
    rgb_cache, target_certificates = {}, []
    start = time.monotonic()
    for line in source:
        key = line['fullRGB_path']
        if key in rgb_cache:
            continue
        check(line['fullRGB_file_binding'])
        if line['fullRGB_format'] == 'CANONICAL_RGB_UINT8_HWC_NPY':
            rgb = np.load(key, allow_pickle=False)
        else:
            rgb = np.asarray(Image.open(key).convert('RGB')).copy()
        row = rows[line['base_row_index']]
        assert rgb.dtype == np.uint8 and rgb.shape == (*row['raw_hw'], 3)
        digest = pixel_sha(rgb)
        if line['before_decoded_sha256'] is not None:
            assert digest == line['before_decoded_sha256']
        rgb_cache[key] = rgb
        target_certificates.append(dict(source=line['source'], path=key, original_image_sha256=row['image_sha256'],
                                        decoded_RGB_sha256=digest, shape=list(rgb.shape), rawfile_binding=line['fullRGB_file_binding']))
    assert len(rgb_cache) == 120
    rgb_digest = {x['path']: x['decoded_RGB_sha256'] for x in target_certificates}
    driver_lineage = []
    for row, line in zip(rows, source):
        i = row['slot_index']; out = dict(line, **{k:row[k] for k in ['slot_index','input_id','native_slot_logical_only','query_xy','L','image_sha256','split','raw_hw']})
        out['decoded_RGB_sha256'] = rgb_digest[line['fullRGB_path']]
        out['before_decoded_sha256'] = out['decoded_RGB_sha256']
        if line['source'] == 'LOCO':
            out['before_input_refs'] = {k:dict(file=compact_binding, key=k, index=i) for k in ['p3_roi','rgb96','crop_valid96']}
        else:
            j = line['original_row_index']; chunk = next(c for c in pm['chunks'] if c['start'] <= j < c['end'])
            out['before_input_refs'] = {k:dict(file=chunk['file'], key=k, index=j-chunk['start']) for k in ['p3_roi','rgb96','crop_valid96']}
        driver_lineage.append(out)
    jsonl(HERE / 'SOURCE960_DRIVER_LINEAGE.jsonl', driver_lineage)
    jsonl(OUT / 'SOURCE120_RGB_SHA_CERTIFICATES.jsonl', target_certificates)
    print(json.dumps(dict(stage='TARGET_RGB_AND_DRIVER_LINEAGE_READY', frames=120, rows=960, seconds=time.monotonic()-start)), flush=True)
    mask_cache, donor_rgb_cache, donor_certificates, donor_data = {}, {}, [], {}
    def mask_file(d):
        path = d['visible_alpha_source']['path']
        if path not in mask_cache:
            check(d['visible_alpha_source'])
            m = np.asarray(Image.open(path)).copy()
            assert m.dtype == np.uint8 and m.shape == tuple(d['raw_hw']) and set(np.unique(m)).issubset({0,255})
            mask_cache[path] = m > 0
        return mask_cache[path]
    selected = [a['assigned_donor'] for a,d in zip(assigned,decisions) if d['before_aux_state'] == 'D']
    for d in selected:
        if d is None or d['id'] in donor_data:
            continue
        if d['image']['path'] not in donor_rgb_cache:
            check(d['image']); donor_rgb_cache[d['image']['path']] = np.asarray(Image.open(d['image']['path']).convert('RGB')).copy()
        rgb = donor_rgb_cache[d['image']['path']]; vis = mask_file(d)
        assert rgb.dtype == np.uint8 and rgb.shape == (*d['raw_hw'], 3)
        padded = np.pad(vis,1)
        eroded = np.logical_and.reduce([padded[y:y+vis.shape[0],x:x+vis.shape[1]] for y in range(3) for x in range(3)])
        others = [x for x in donor_pool if x['scene_id']==d['scene_id'] and x['frame_id']==d['frame_id'] and x['instance_id'] != d['instance_id']]
        overlap = np.zeros_like(vis)
        for other in others:
            overlap |= mask_file(other)
        core_overlap = int((eroded & overlap).sum())
        expected_mask = f"{d['frame_id']:06d}_{d['instance_id']:06d}.png"
        ownership_ok = Path(d['visible_alpha_source']['path']).name == expected_mask and core_overlap == 0
        count_ok = int(vis.sum()) == int(d['metadata_vis_pixels'])
        admitted = bool(ownership_ok and count_ok and eroded.any())
        meta = dict(d, foreign_opaque_cutout_admitted=admitted)
        cert = dict(id=d['id'], split=d['split'], scene_id=d['scene_id'], frame_id=d['frame_id'], instance_id=d['instance_id'], object_id=d['object_id'],
                    image=d['image'], mask=d['visible_alpha_source'], image_decoded_RGB_sha256=pixel_sha(rgb),
                    visible_pixels=int(vis.sum()), metadata_visible_pixels=d['metadata_vis_pixels'], eroded_core_pixels=int(eroded.sum()),
                    selected_other_instance_visible_overlap_in_eroded_core=core_overlap, other_instance_masks=len(others),
                    ownership_filename_and_instance_metadata_match=ownership_ok, visible_count_match=count_ok,
                    foreign_opaque_cutout_admitted=admitted,
                    opacity_basis='Controlled hard-paste alpha=1 on erode1 visible mask; source material optical opacity not independently measured. BOP instance mask teacher is conditional, not human gold.',
                    source_mask_threshold='uint8 strict0/255, >0 foreground; erode1 zero outside image', replacement_allowed=False)
        if eroded.any():
            yy,xx=np.nonzero(eroded); longest=max(int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1))
            cut = adapter.v3.v1.prepare_cutout(rgb,vis,longest)
            safe = d['id'].replace(':','_')
            path = OUT / 'donor_cutouts' / (safe+'.npz')
            np.savez_compressed(path,rgb=cut['rgb'],alpha=cut['alpha'])
            cert['compact_RGBA_cutout'] = bind(path); cert['native_bbox_xyxy'] = cut['donor_native_bbox_xyxy']
        donor_certificates.append(cert); donor_data[d['id']] = (rgb,vis,meta)
    jsonl(OUT / 'SELECTED_DONOR_RGBA_CERTIFICATES.jsonl',donor_certificates)
    feasibility, occurrences = [], []
    crop,_ = adapter.v3.v1.frozen_rgb96_operator(P / 'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py','802f2e17f1ee98ae09d8468b8ab749653a50054a85f24ef0af16385a7c262b66')
    for row, decision, line, assignment in zip(rows,decisions,driver_lineage,assigned):
        i = row['slot_index']; target = rgb_cache[line['fullRGB_path']]; before_sha = line['decoded_RGB_sha256']
        donor = assignment['assigned_donor']
        drgb,dvis,dmeta = (None,None,None) if donor is None or donor['id'] not in donor_data else donor_data[donor['id']]
        edits,cut,protected,facts = adapter.edit_geometry(target,row,decision,drgb,dvis,dmeta)
        facts.update(base_row_index=i,source=line['source'],split=row['split'],query_key=decision['query_key'],
                     original_q=row['query_xy'],original_L=row['L'],original_natural_label_GT_only=decision['original_natural_label_GT_only'],
                     assigned_donor_id=None if donor is None else donor['id'],native_expected_part_state='UNKNOWN',human_gold=False,training_admitted=False,
                     protected_regions_GT_only=decision['protected_regions_GT_only'])
        if protected is not None:
            geom = OUT / 'pair_geometry' / f'protected_{i:04d}.npz'
            np.savez_compressed(geom,protected_union=protected);facts['protected_union_file']=bind(geom)
        feasibility.append(facts)
        for vi,variant in enumerate(adapter.v3.VARIANTS):
            changed = bool(vi and edits is not None and not np.array_equal(edits[vi-1],target))
            full_sha = before_sha; edited = None
            if changed:
                rgb = edits[vi-1];full_sha = pixel_sha(rgb)
                path = OUT / 'edited_fullRGB' / f'row_{i:04d}_{variant}.png';Image.fromarray(rgb).save(path)
                assert np.array_equal(np.asarray(Image.open(path).convert('RGB')),rgb)
                edited=bind(path)
            natural = decision['original_natural_label_GT_only']
            label = natural if vi==0 else (1 if vi==1 else 0) if facts['pair_ready'] else -1
            occurrences.append(dict(occurrence_index=i*3+vi,base_row_index=i,slot_index=i,variant=variant,
                occurrence_id=f"{row['input_id']}::slot{row['native_slot_logical_only']}::{variant}",
                full_rgb_changed=changed,edited_fullRGB_file=edited,edited_fullRGB_path_or_null=None if edited is None else edited['path'],
                edited_fullRGB_format=None if edited is None else'PNG',full_rgb_decoded_sha256=full_sha,before_decoded_sha256=before_sha,
                base_ref=dict(source_driver_lineage=str(HERE/'SOURCE960_DRIVER_LINEAGE.jsonl'),row_index=i),
                candidate_aux_hidden_label_GT_only=label,known_candidate_GT_only=label>=0,original_natural_label_GT_only=natural,
                pair_ready=facts['pair_ready'],pair_reason=facts['reason'],native_expected_part_state='UNKNOWN',human_gold=False,training_admitted=False))
        if decision['before_aux_state']=='D' and sum(x['before_aux_state']=='D' for x in decisions[:i+1])%25==0:
            print(json.dumps(dict(stage='FIXED_D_FEASIBILITY',done=sum(x['before_aux_state']=='D' for x in decisions[:i+1]),pair_ready=sum(f['pair_ready']for f in feasibility),seconds=time.monotonic()-start)),flush=True)
    jsonl(HERE/'FEASIBILITY960_GT_ONLY.jsonl',feasibility);jsonl(HERE/'OCCURRENCES2880_LINEAGE.jsonl',occurrences)
    assert len(feasibility)==960 and len(occurrences)==2880
    eligible=[f for f,d in zip(feasibility,decisions)if d['before_aux_state']=='D']; assert len(eligible)==143
    summary=dict(completed=True,PASS=True,source_only=True,base_rows=960,occurrences=2880,eligible_D=143,
                 before_D_E_U={'D':143,'E':20,'U':797},selected_unique_donors=len(donor_certificates),
                 donor_admitted=sum(x['foreign_opaque_cutout_admitted']for x in donor_certificates),
                 feasible_pairs=sum(f['pair_ready']for f in eligible),
                 eligible_reasons=dict(collections.Counter(f['reason']for f in eligible)),
                 by_source_split={f'{s}|{sp}':dict(D_attempts=sum(f['source']==s and f['split']==sp for f in eligible),
                        pair_ready=sum(f['source']==s and f['split']==sp and f['pair_ready']for f in eligible))
                        for s,sp in sorted({(f['source'],f['split'])for f in eligible})},
                 full_RGB_changed_occurrences=sum(x['full_rgb_changed']for x in occurrences),
                 all_failures_and_original_q_L_retained=True,protected_boxes_shrunk=False,donor_replacement=False,
                 training_admission=False,native_expected_part_state='UNKNOWN',human_gold=False,
                 actual_target_RGB_frames=120,model_P3_weights_fit_current_PnP=0,seconds=time.monotonic()-start)
    write(HERE/'IMAGE_ONLY_FEASIBILITY_SUMMARY_KO.json',summary)
    outputs=[HERE/x for x in ['SOURCE960_DRIVER_LINEAGE.jsonl','FEASIBILITY960_GT_ONLY.jsonl','OCCURRENCES2880_LINEAGE.jsonl','IMAGE_ONLY_FEASIBILITY_SUMMARY_KO.json']]
    outputs += sorted(p for p in OUT.rglob('*')if p.is_file())
    write(HERE/'IMAGE_ONLY_INTAKE_SEAL.json',dict(completed=True,PASS=True,protocol=bind(HERE/'INTAKE_PROTOCOL_BEFORE_PIXELS.json'),
            metadata_lock=bind(HERE/'METADATA_INPUT_LOCK.json'),files=[bind(p)for p in outputs],summary=bind(HERE/'IMAGE_ONLY_FEASIBILITY_SUMMARY_KO.json'),
            source_only=True,source_models_P3_fit_current_PnP=0,training_admission=False))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
