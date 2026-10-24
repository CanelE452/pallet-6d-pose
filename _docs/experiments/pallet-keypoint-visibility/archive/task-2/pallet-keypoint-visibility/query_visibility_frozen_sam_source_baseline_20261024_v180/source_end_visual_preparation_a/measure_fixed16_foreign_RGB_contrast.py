"""Fixed16 raw same-pixel contrast only. No model, threshold or target changes."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,sys
sys.dont_write_bytecode=True
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent
OPDIR=V/'source_end_supervision_preparation_a';sys.path.insert(0,str(OPDIR))
import source_end_support as op
O=V/'source_end_spatial_visual128_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def checked(b):assert bind(b['path'])==b;return Path(b['path'])


def stats(difference):
    a=np.asarray(difference,np.int16).reshape(-1,3);n=len(a)
    maximum=a.max(1) if n else np.empty(0,np.int16)
    equality=int((maximum==0).sum())
    pixel_hist=np.bincount(maximum,minlength=256)
    channel_hist=np.bincount(a.ravel(),minlength=256)
    return dict(pixel_N=n,exact_RGB_equal_pixels=equality,exact_RGB_equal_fraction=equality/n if n else None,
        max_absolute_channel_difference=int(a.max()) if n else None,
        mean_absolute_channel_difference=float(a.sum(dtype=np.int64))/(3*n) if n else None,
        mean_per_pixel_max_absolute_channel_difference=float(maximum.sum(dtype=np.int64))/n if n else None,
        per_pixel_max_abs_difference_histogram_0_to255=pixel_hist.tolist(),
        per_channel_abs_difference_histogram_0_to255=channel_hist.tolist(),
        histogram_units='Raw uint8 intensity steps; full exact integer bins0..255, no chosen visibility/learning threshold')


def run():
    complete=read(V/'source_end_supervision_train4096_v1/COMPLETE.json');assert complete['completed'] and complete['PASS']
    assert complete['operator']==bind(OPDIR/'source_end_support.py')
    scope=read(V/'ROOT_SCOPE_BEFORE_SAM_CODE_WEIGHTS_OR_SOURCE_RGB.json')
    selection=lines(checked(scope['input_manifest']));assert len(selection)==128
    plans={r['original_global_index']:r for r in lines(checked(complete['plan']))}
    geometry={(g['base_index'],g['native_slot']):g for g in lines(checked(complete['geometry']))}
    scenes=defaultdict(list)
    for r in selection:scenes[r['scene_index']].append(plans[r['original_global_index']])
    assert len(scenes)==16 and set(scenes)=={0,1,2,3,4,5,6,7,64,65,66,67,68,69,70,71}
    cache={};bindings={}
    def load(binding):
        path=str(checked(binding));bindings[path]=binding
        if path not in cache:
            with np.load(path,allow_pickle=False) as z:cache[path]={k:z[k].copy() for k in ('RGB_uint8','part_id','camera_depth','global_triangle_id','ambiguity','camera_K','camera_R','camera_t','native8_CF_xy','permCF_to_canonical')}
        return cache[path]
    scene_rows=[];all_differences=[];native_rows=[]
    for si,rows in sorted(scenes.items()):
        rows.sort(key=lambda r:r['native_slot']);assert [r['native_slot'] for r in rows]==list(range(8))
        before=load(rows[0]['BEFORE_buffer']);scene=load(rows[0]['scene_buffer'])
        assert before['RGB_uint8'].dtype==scene['RGB_uint8'].dtype==np.uint8 and before['RGB_uint8'].shape==scene['RGB_uint8'].shape==(512,512,3)
        difference=np.abs(scene['RGB_uint8'].astype(np.int16)-before['RGB_uint8'].astype(np.int16))
        union=np.zeros((512,512),bool);membership=np.zeros((512,512),np.uint8)
        for row in rows:
            g=geometry[row['base_index'],row['native_slot']]
            for b in (before,scene):op.validate_buffer_metadata(b,g['camera'],g['native8_CF_xy'],g['permCF_to_canonical'])
            patch=op.expected_patch_depth(g['quads_world'],g['camera'],g['raw_hw'])
            classes=op.classify_end_pixels(before,scene,patch,g['primary_part_id'])
            mask=classes==2
            # Assert the defining conjunction directly at every included source pixel.
            assert (before['part_id'][mask]==g['primary_part_id']).all() and (scene['part_id'][mask]==999).all()
            assert np.isfinite(before['camera_depth'][mask]).all() and np.isfinite(scene['camera_depth'][mask]).all()
            assert (scene['camera_depth'][mask]>0).all() and (scene['camera_depth'][mask]<before['camera_depth'][mask]).all()
            assert not before['ambiguity'][mask].any() and not scene['ambiguity'][mask].any()
            native_rows.append(dict(scene_index=si,original_global_index=row['original_global_index'],native_slot=row['native_slot'],variant=row['variant'],
                scope='Full raw projected fixed-end class2 support for this native, not a q-centred/repositioned crop; native patches may overlap. Aggregate scenes use union to avoid duplication.',
                **stats(difference[mask])))
            union|=mask;membership+=mask.astype(np.uint8)
        d=difference[union];all_differences.append(d)
        scene_rows.append(dict(scene_index=si,base_index=rows[0]['base_index'],variant=rows[0]['variant'],factors=rows[0]['factors'],native_slots8=list(range(8)),
            confirmed_foreign_raw_pixel_mask_sha256=hashlib.sha256(union.tobytes()).hexdigest(),
            native_support_membership_count_sum=int(membership.sum()),overlapping_native_support_pixels=int((membership>1).sum()),
            before_buffer=rows[0]['BEFORE_buffer'],scene_buffer=rows[0]['scene_buffer'],
            **stats(d)))
    combined=np.concatenate(all_differences,axis=0)
    for binding in bindings.values():checked(binding)
    total=stats(combined)
    assert sum(r['pixel_N'] for r in scene_rows)==total['pixel_N'] and sum(r['exact_RGB_equal_pixels'] for r in scene_rows)==total['exact_RGB_equal_pixels']
    assert np.array_equal(np.asarray([r['per_pixel_max_abs_difference_histogram_0_to255'] for r in scene_rows]).sum(0),total['per_pixel_max_abs_difference_histogram_0_to255'])
    out=O/'FIXED16_CONFIRMED_END_FOREIGN_RAW_RGB_CONTRAST.json'
    result=dict(completed=True,PASS=True,code=bind(__file__),source_complete=bind(V/'source_end_supervision_train4096_v1/COMPLETE.json'),operator=complete['operator'],definition=complete['definition'],fixed_selection=scope['input_manifest'],render_complete=bind(O/'COMPLETE.json'),
        scope='Exactly16 frozen TRAIN scenes128slots. Raw original512 source pixels for frozen END_FOREIGN_FRONT class2. Scene totals use union over8native fixed-end regions. BEFORE and SCENE compared at identical raw pixel and camera, never q-centre relocated or recropped.',
        palette_or_rendered_RGB_used_for_measurement=False,integer_uint8_exact_difference=True,no_threshold_selected=True,
        total=total,scene_rows=scene_rows,native_rows=native_rows,
        no_foreign_scene_indices=[r['scene_index'] for r in scene_rows if r['pixel_N']==0],
        source_buffers_all_hash_unchanged=True,source_body_loads=len(bindings),
        limitations=['Exact equal RGB can occur despite constructive source owner/depth change. It establishes local colour equality, not equality of the whole image/crop or an impossibility theorem for context-based inference.',
          'A nonzero intensity difference is a contrast measurement, not a certified human-observable cue or learned segmentation/occlusion success.',
          'Per-native masks may overlap; only per-scene union contributes to total. This is fixed smoke coverage, not representative full TRAIN/real transfer evidence.',
          'No SAM scores/quality, model outputs, validation/held/private bodies, new labels or q/L/rule changes were read or produced.'],
        new_model_fit_SAM_forward_threshold_qL_label_or_training_admission_changes=0,selected_model=None,fully_solved=False)
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(PASS=True,output=bind(out),total=total,scene_counts=[dict(scene=r['scene_index'],N=r['pixel_N'],equal=r['exact_RGB_equal_pixels'],maxdiff=r['max_absolute_channel_difference'],meandiff=r['mean_absolute_channel_difference']) for r in scene_rows])),flush=True)


if __name__=='__main__':run()
