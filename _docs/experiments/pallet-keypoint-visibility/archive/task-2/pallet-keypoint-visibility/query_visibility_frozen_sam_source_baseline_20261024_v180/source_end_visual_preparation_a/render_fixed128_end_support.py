"""Authorized fixed16/128 TRAIN source end-support spatial QA. No model IO."""
from pathlib import Path
from collections import defaultdict
import ast,hashlib,json,sys,time
sys.dont_write_bytecode=True
import numpy as np
from PIL import Image,ImageDraw,ImageFont

H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
OPDIR=V/'source_end_supervision_preparation_a';sys.path.insert(0,str(OPDIR))
import source_end_support as op
OUT=V/'source_end_spatial_visual128_v1'
COMPLETE=V/'source_end_supervision_train4096_v1/COMPLETE.json'
SCOPE=V/'ROOT_SCOPE_BEFORE_SAM_CODE_WEIGHTS_OR_SOURCE_RGB.json'
PALETTE=np.array([[130,136,145],[38,200,94],[239,62,68],[49,108,233],[170,77,212],[247,139,41],[252,218,48]],np.float64)
NATCOL=['#00ffff','#ff99bb','#bbff55','#ffa040','#aa88ff','#ffff55','#77bbff','#ff66ff']
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def check(b):assert bind(b['path'])==b;return Path(b['path'])
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def font(n,b=False):return ImageFont.truetype(BOLD if b else FONT,n)
def image(a,scale=1):return Image.fromarray(a).resize((a.shape[1]*scale,a.shape[0]*scale),Image.Resampling.NEAREST)
def label(v):return {-1:'UNKNOWN',0:'NO',1:'YES'}[int(v)]


def run():
    assert not OUT.exists(),'Preserve previous render; no silent retry.'
    done=read(COMPLETE);assert done['completed'] and done['PASS']
    check(done['operator']);assert done['operator']==bind(OPDIR/'source_end_support.py')
    scope=read(SCOPE);smoke=lines(check(scope['input_manifest']))
    assert len(smoke)==128 and len({r['scene_index'] for r in smoke})==16 and all(r['split']=='TRAIN' for r in smoke)
    ids=[r['original_global_index'] for r in smoke];assert len(set(ids))==128
    plan={r['original_global_index']:r for r in lines(check(done['plan']))}
    geometry={(g['base_index'],g['native_slot']):g for g in lines(check(done['geometry']))}
    savedrows={r['original_global_index']:r for r in lines(check(done['outputs']['rows']))}
    with np.load(check(done['outputs']['sidecar']),allow_pickle=False) as z:saved={k:z[k].copy() for k in z.files}
    assert len(saved['original_global_index'])==4096 and saved['source_end_class_order'].tolist()==list(op.CLASSES)
    by_index={int(i):j for j,i in enumerate(saved['original_global_index'])}
    crop_protocol=read(P/'query_visibility_actual_initializer_source_20261024_v132/frozen_visibility_preparation/PROTOCOL_BEFORE_VISIBILITY_FORWARD.json')
    crop_binding=crop_protocol['c16']['local_crop_code'];cp=check(crop_binding)
    fn=next(n for n in ast.parse(cp.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='crop')
    ns=dict(np=np);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(cp)+'::crop_ONLY','exec'),ns);crop=ns['crop']
    OUT.mkdir();write(OUT/'RENDER_RULES_BEFORE_PIXELS.json',dict(code=bind(__file__),source_complete=bind(COMPLETE),operator=done['operator'],definition=done['definition'],input_manifest=scope['input_manifest'],exact_selection_ids=ids,RGB_crop_code=crop_binding,
        class_order=list(op.CLASSES),class_colors=PALETTE.astype(int).tolist(),blend='Display only:0.52raw+0.48classcolor',
        q_display='Clean RAW96 untouched. Cyan cross/2x2box only on overlay/color copies. q is saved96 index47.5,47.5; displayed at centre boundary96 of192px. Fullscene display position=(nativepixelxy+.5)*scale.',
        GT_for_supervision_only_not_model_input=True,new_mask_rule_fit_selection_or_labels=False))
    chunks={};buffers={};patches={};grouped=defaultdict(list);records=[];start=time.monotonic()
    def buffer(binding):
        key=binding['path']
        if key not in buffers:
            check(binding)
            with np.load(key,allow_pickle=False) as z:
                buffers[key]={k:z[k].copy() for k in ('RGB_uint8','part_id','camera_depth','global_triangle_id','ambiguity','camera_K','camera_R','camera_t','native8_CF_xy','native8_canonical_xy','permCF_to_canonical','state8_CF')}
        return buffers[key]
    for m in smoke:
        idx=m['original_global_index'];r=plan[idx];g=geometry[r['base_index'],r['native_slot']]
        assert r['query_xy']==m['original_query_xy'] and r['L']==m['original_L'] and r['original_input_flags']==m['original_input_flags']
        b=buffer(r['BEFORE_buffer']);s=buffer(r['scene_buffer']);rgb=s['RGB_uint8']
        assert rgb.dtype==np.uint8 and rgb.shape==(512,512,3) and ash(rgb)==m['decoded_RGB_sha256_inherited']
        for pack in (b,s):op.validate_buffer_metadata(pack,g['camera'],g['native8_CF_xy'],g['permCF_to_canonical'])
        assert s['state8_CF'][r['native_slot']]==r['original_semantic_state']
        cb=m['saved_crop_chunk'];key=cb['path']
        if key not in chunks:
            check(cb)
            with np.load(key,allow_pickle=False) as z:chunks[key]={k:z[k].copy() for k in ('rgb96','valid96','crop_constructed','global_index')}
        ch=chunks[key];j=m['saved_crop_row'];raw=ch['rgb96'][j];valid=ch['valid96'][j]
        assert raw.dtype==np.uint8 and raw.shape==(96,96,3) and valid.dtype==bool and valid.shape==(96,96)
        assert int(ch['global_index'][j])==idx and bool(ch['crop_constructed'][j])==r['original_input_flags']['crop_constructed']
        assert ash(raw)==m['saved_rgb96_sha256_inherited'] and ash(valid)==m['saved_valid96_sha256_inherited']
        rebuilt,vm=crop(rgb,np.asarray(r['query_xy'],np.float64),r['L'],96)
        assert np.array_equal(rebuilt.transpose(1,2,0),raw) and np.array_equal(vm[0],valid)
        keyg=g['geometry_key']
        if keyg not in patches:patches[keyg]=dict(op.expected_patch_depth(g['quads_world'],g['camera'],g['raw_hw']),geometry_key=keyg)
        patch=patches[keyg]
        result=op.process_one(r,g,b,s,patch);classes=op.classify_end_pixels(b,s,patch,g['primary_part_id'])
        weights=op.sample_classes(classes,op.original_crop_positions(r['query_xy'],r['L'])).reshape(96,96,7)
        row_index=by_index[idx];mass=weights.reshape(-1,7).mean(0)
        assert np.array_equal(mass,saved['jointmass7'][row_index])
        for field in ('visible_support_target','foreign_front_support_target'):
            assert result[field]==int(saved[field][row_index])==savedrows[idx][field]
        assert result['full96_average_mass']==savedrows[idx]['full96_average_mass']
        colored=np.floor(np.clip(weights@PALETTE,0,255)+.5).astype(np.uint8)
        overlay=np.floor(.52*raw.astype(np.float64)+.48*colored+.5).astype(np.uint8)
        folder=OUT/f'G{idx:04d}';folder.mkdir()
        output_arrays=dict(RAW96=raw,VALID96=(valid.astype(np.uint8)*255),END_CLASS_COLOR96=colored,OVERLAY96=overlay)
        bindings={}
        for name,array in output_arrays.items():
            path=folder/(name+'.png');Image.fromarray(array).save(path)
            with Image.open(path) as im:assert np.array_equal(np.asarray(im),array)
            bindings[name]=bind(path)
        record=dict(original_global_index=idx,scene_index=r['scene_index'],base_index=r['base_index'],native_slot=r['native_slot'],variant=r['variant'],factors=r['factors'],query_xy=r['query_xy'],L=r['L'],original_input_flags=r['original_input_flags'],original_semantic_state=r['original_semantic_state'],
            source_buffer=r['scene_buffer'],BEFORE_buffer=r['BEFORE_buffer'],crop_chunk=cb,crop_row=j,raw96_array_sha256=ash(raw),valid96_array_sha256=ash(valid),sample_weights96x96x7_array_sha256=ash(weights),full96_average_mass=mass.tolist(),visible_support_target=result['visible_support_target'],foreign_front_support_target=result['foreign_front_support_target'],reasons=result['reasons'],projected_region_out_of_frame=patch['projected_region_out_of_frame'],
            raw96_valid96_exact_hash_and_replay=True,saved_sidecar_and_JSONL_mass_target_exact=True,images=bindings)
        records.append(record);grouped[r['scene_index']].append((record,raw,valid,colored,overlay,patch,r,g))
    cards=[]
    for scene,rr in sorted(grouped.items()):
        rr.sort(key=lambda z:z[0]['native_slot']);assert [x[0]['native_slot'] for x in rr]==list(range(8))
        first=rr[0][0];rgb=buffers[first['source_buffer']['path']]['RGB_uint8']
        canvas=Image.new('RGB',(1920,2010),'#f7f9fb');d=ImageDraw.Draw(canvas)
        d.text((24,16),f"SCENE {scene:04d} | BASE {first['base_index']:03d} | {first['variant']} | all8 original slots",font=font(28,True),fill='#13263b')
        d.text((24,58),'Fixed small end support in unchanged q/L crop | GT diagnostic only; NOT model input/native D/E rewrite',font=font(20),fill='#304456')
        clean=image(rgb).resize((384,384),Image.Resampling.NEAREST);canvas.paste(clean,(24,110));canvas.paste(clean,(440,110))
        d.text((24,86),'Original RGB512 (clean)',font=font(17,True),fill='#152d43');d.text((440,86),'Original q8 + registered end-patch quads',font=font(17,True),fill='#152d43')
        for row,*_,patch,r,g in rr:
            col=NATCOL[r['native_slot']]
            for face in patch.get('projected_quad_xy',[]):
                xy=[(440+(p[0]+.5)*.75,110+(p[1]+.5)*.75) for p in face];d.line(xy+[xy[0]],fill=col,width=2)
            q=r['query_xy'];x,y=440+(q[0]+.5)*.75,110+(q[1]+.5)*.75
            d.line((x-5,y,x+5,y),fill=col,width=2);d.line((x,y-5,x,y+5),fill=col,width=2);d.text((x+5,y-16),str(r['native_slot']),font=font(15,True),fill=col)
        for k,(name,color) in enumerate(zip(op.CLASSES,PALETTE.astype(int))):
            y=105+k*31;d.rectangle((860,y,882,y+20),fill=tuple(color));d.text((895,y-2),name,font=font(19),fill='#182f44')
        for k,txt in enumerate([f"{first['factors']['material']} / {first['factors']['aspect']}",f"{first['factors']['geometry_family']} | Original inputU kept",'Targets are separate/non-exclusive: visible support + foreign front', 'Cyan centre2 box/cross on copies only; raw96 is untouched.', 'Self/ambiguity/registration unknown kept; gray is OUTSIDE END.', 'Color mixtures show exact bilinear mass, not categorical ID interpolation.']):
            d.text((860,345+k*25),txt,font=font(17),fill='#243e54')
        for slot,(row,raw,valid,colored,overlay,patch,r,g) in enumerate(rr):
            x=24+(slot%2)*948;y=535+(slot//2)*350
            d.rounded_rectangle((x,y,x+924,y+334),radius=8,fill='white',outline='#c8d3dc',width=2)
            title=f"N{slot} | G{row['original_global_index']:04d} | old native {row['original_semantic_state']} | end visible {label(row['visible_support_target'])} / foreign {label(row['foreign_front_support_target'])}"
            d.text((x+10,y+8),title,font=font(19,True),fill='#102b43')
            images=[(raw,10,'Original RGB96'),(overlay,216,'RGB + class mass'),(colored,422,'End-class96 mass'),(colored[32:64,32:64],628,'End-class center32')]
            for array,dx,caption in images:
                pic=Image.fromarray(array).resize((192,192),Image.Resampling.NEAREST);canvas.paste(pic,(x+dx,y+66))
                d.text((x+dx,y+42),caption,font=font(16,True),fill='#1e3a50')
                if dx in (216,422):
                    cx,cy=x+dx+96,y+66+96;d.rectangle((cx-2,cy-2,cx+2,cy+2),outline='#00ffff',width=1);d.line((cx-6,cy,cx+6,cy),fill='#00ffff');d.line((cx,cy-6,cx,cy+6),fill='#00ffff')
            canvas.paste(Image.fromarray(valid.astype(np.uint8)*255).resize((88,88),Image.Resampling.NEAREST),(x+830,y+66));d.text((x+830,y+42),'Valid96',font=font(16,True),fill='#1e3a50')
            d.text((x+830,y+160),f"C4:{int(valid[47:49,47:49].sum())}/4",font=font(15),fill='#17374a')
            mass=row['full96_average_mass'];unknown=sum(mass[3:6]);d.text((x+10,y+267),f"q=({r['query_xy'][0]:.3f},{r['query_xy'][1]:.3f}) L={r['L']:.3f} | visible={mass[1]:.6f} foreign={mass[2]:.6f} unknown={unknown:.6f}",font=font(16),fill='#1e3b50')
            short=';'.join(row['reasons']).replace('REGISTERED_PATCH_GEOMETRY_OR_OUT_OF_FRAME_UNKNOWN','PATCH_OOF/GEOMETRY_UNKNOWN').replace('SELF_AMBIGUOUS_OR_REGISTRATION_MASS_RETAINED_UNKNOWN','SELF/AMBIGUOUS/REGISTRATION_UNKNOWN').replace('NO_SAMPLED_END_SUPPORT_IN_ORIGINAL_CROP_NOT_EXTERNAL_E','NO_END_IN_CROP; NOT_EXTERNAL_E')
            d.text((x+10,y+293),short,font=font(15),fill='#65472b')
        d.text((24,1955),'All128 slots /16 frozen TRAIN scenes retained. No SAM/model scores shown. No q correction, new native labels, training admission or full-goal claim.',font=font(19),fill='#2b4154')
        path=OUT/f'SCENE_{scene:04d}_CARD.png';canvas.save(path);card=dict(scene_index=scene,case_ids=[z[0]['original_global_index'] for z in rr],card=bind(path));cards.append(card)
        print(json.dumps(dict(card=card,done=len(cards),total16=16)),flush=True)
    manifest=OUT/'ALL128_SPATIAL_RECORDS.json';write(manifest,dict(completed=True,rows=records,cards=cards))
    html=['<!doctype html><meta charset="utf-8"><title>Fixed128 source end support</title><style>body{font:18px sans-serif;max-width:1920px;margin:20px auto}img{max-width:100%;height:auto}</style><h1>Fixed16 scenes /128 slots: source-end support spatial QA</h1><p>GT diagnostic only, no model input or new native D/E labels. Rendered copies use exact original96 sample positions.</p>']
    for c in cards:html.append(f'<h2>Scene {c["scene_index"]}</h2><a href="{c["card"]["path"]}"><img src="{c["card"]["path"]}"></a>')
    (OUT/'index.html').write_text('\n'.join(html))
    for r in smoke:
        check(r['full_RGB_packet']);check(r['saved_crop_chunk'])
    check(done['outputs']['sidecar']);check(done['outputs']['rows'])
    write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),scope=bind(SCOPE),source_complete=bind(COMPLETE),operator=done['operator'],definition=done['definition'],fixed_input_manifest=scope['input_manifest'],RGB_crop_code=crop_binding,
        original_selection128_exact=True,rows128=len(records),cards16=len(cards),all128_RGB96_valid96_hash_and_original_qL_replay_exact=True,all128_full96_endmass_and_targets_exact_to_saved_NPZ_JSONL=True,
        source_end_class_order=list(op.CLASSES),class_colors=PALETTE.astype(int).tolist(),manifest=bind(manifest),cards=cards,index=bind(OUT/'index.html'),actual_tool_image_QA='PENDING_SEPARATE_RECEIPT',HTML_browser_rendered=False,
        original_source_files_unchanged=True,source_GT_supervision_diagnostic_only=True,new_mask_rule_model_forward_fit_qL_label_or_admission_changes=0,selected_model=None,fully_solved=False,seconds=time.monotonic()-start))
    print(json.dumps(dict(PASS=True,complete=bind(OUT/'COMPLETE.json'))),flush=True)


if __name__=='__main__':run()
