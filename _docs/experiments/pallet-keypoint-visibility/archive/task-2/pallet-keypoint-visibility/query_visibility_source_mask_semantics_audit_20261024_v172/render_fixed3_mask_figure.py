"""One fixed3 source forensic figure from existing RGB/mask buffers; no renderer/model."""
from pathlib import Path
import argparse,ast,hashlib,json,subprocess,traceback
import numpy as np
from PIL import Image,ImageDraw,ImageFont

V=Path(__file__).resolve().parent;P=V.parent
COMPLETE=V/'saved_mask_semantics_measurement_v1/COMPLETE.json';COMPLETE_SHA='12d2d8375de1e573f021536176be81ed4aae1f0c10ece179deb2b5782dbd2b78'
CODE=V/'measurement_preparation/measure_saved_source_semantics.py';CODE_SHA='8e673bb930e9fa49017c85c4355141e0c0846526d49c0297a82db59309418e42'
CROP=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py';CROP_SHA='6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
OUT=V/'fixed3_forensic_figure_v1';PROTOCOL=V/'FIXED3_FIGURE_PROTOCOL_BEFORE_PIXEL_REREAD.json';TESTS=V/'FIXED3_FIGURE_INVENTED_CHECKS.json'
CLASSES=('INCIDENT_TARGET','OTHER_TARGET','FOREIGN','BG','AMBIGUOUS','INVALID_SUPPORT')
PALETTE=np.array([[59,162,114],[84,112,198],[238,102,102],[217,217,217],[154,96,180],[25,25,25]],np.float64)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(b):
    p=Path(b['path']).resolve();assert p.is_relative_to(P)
    assert not any(s in str(p).lower() for s in ('current223','_v169','_v112','/private_evaluation/'))
    assert sha(p)==b['sha256'];return p
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def pure(path,names,ns):
    nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::PURE_DISPLAY','exec'),ns);return ns
def color(w):return np.floor(np.clip(w@PALETTE,0,255)+.5).astype(np.uint8)
def prepare():
    assert sha(COMPLETE)==COMPLETE_SHA and sha(CODE)==CODE_SHA and sha(CROP)==CROP_SHA
    w=np.zeros((2,2,6),np.float64);w[:,:,0]=1;w[0,0]=[0,0,.25,.75,0,0]
    mixed=color(w);assert np.array_equal(mixed[1,1],PALETTE[0].astype(np.uint8))
    assert np.array_equal(mixed[0,0],np.floor(.25*PALETTE[2]+.75*PALETTE[3]+.5).astype(np.uint8))
    clean=np.arange(96*96*3,dtype=np.uint16).reshape(96,96,3).astype(np.uint8);original=ash(clean);display=Image.fromarray(clean.copy()).resize((320,320),Image.Resampling.NEAREST)
    ImageDraw.Draw(display).line((150,160,170,160),fill='black',width=2);assert ash(clean)==original
    write(TESTS,dict(completed=True,PASS=True,invented_only=True,onehot_palette_endpoints_exact=True,onehot_bilinear_color_mixture_matches_analytic=True,display_marker_copy_does_not_mutate_raw96=True,source_pixel_bodies_reread_during_preparation=0,code=bind(__file__)))
    complete=read(COMPLETE)
    write(PROTOCOL,dict(completed=True,code=bind(__file__),complete=bind(COMPLETE),selection=complete['selection'],readiness=complete['readiness'],rows=complete['rows'],pure_measurement_code=bind(CODE),pure_crop_code=bind(CROP),tests=bind(TESTS),fixed_witness_indices=[5353,5586,5609],palette={k:PALETTE[i].astype(int).tolist() for i,k in enumerate(CLASSES)},
        RGB96='Frozen original q/L crop; decoded sceneRGB SHA and raw96 SHA must equal inherited V171 actual source/crop facts. No marker on cleanRGB96.',
        owner96='Exact original q/L96 grid; six categorical onehot bilinear weights including INVALID; mixture of fixed RGB palette colors, never interpolate owner IDs. Mean central4/32 must match completed measurement JSON.',
        display='Owner plot cross marks q center on a display copy only. Three extreme witnesses, no new selection; same saved SCENE/BEFORE only. No reference-centered crop, new scene render/model/fit/label/coordinate/threshold/admission.',automatic_retry=False))
    print(json.dumps(dict(prepared=True,protocol=bind(PROTOCOL))),flush=True)
def run():
    p=read(PROTOCOL);assert p['code']==bind(__file__) and sha(COMPLETE)==COMPLETE_SHA and sha(CODE)==CODE_SHA and sha(CROP)==CROP_SHA
    assert read(checked(p['tests']))['PASS'];c=read(checked(p['complete']));assert c['PASS'] and c['row_count']==24
    selection=read(checked(p['selection']));ready=read(checked(p['readiness']));allrows=lines(checked(p['rows']));selected={r['original_global_index']:r for r in allrows if r['is_original_selected_witness']};assert set(selected)=={5353,5586,5609}
    ns=pure(CODE,{'categories','bilinear_weights'},{'np':np,'CLASSES':CLASSES});crop=pure(CROP,{'crop'},{'np':np})['crop']
    assert not OUT.exists();OUT.mkdir();source_bindings=[];proofs=[];panels=[]
    try:
        for witness,scene in zip(selection['unique_witnesses'],ready['scenes']):
            i=witness['global_index'];assert i==scene['global_index'];row=selected[i];incident=row['incident_part_ids'];q=np.asarray(witness['query_xy'],np.float64);L=witness['L'];u=(np.arange(96,dtype=np.float64)+.5)/96-.5;xx,yy=np.meshgrid(q[0]+u*L,q[1]+u*L);points=np.column_stack((xx.ravel(),yy.ravel()));maps={};rgb96=None;raw_before=None
            for label,key in [('SCENE','selected_combined_scene_buffer'),('BEFORE_TARGET_ONLY','saved_target_only_all_parts_buffer')]:
                b=scene[key]['inherited_binding'];source_bindings.append(b)
                with np.load(checked(b),allow_pickle=False) as z:buf={k:z[k].copy() for k in ('RGB_uint8','part_id','ambiguity')}
                assert buf['RGB_uint8'].shape==(512,512,3) and buf['RGB_uint8'].dtype==np.uint8
                cat=ns['categories'](buf['part_id'],buf['ambiguity'],incident);weights,valid=ns['bilinear_weights'](cat,points,True);weights=weights.reshape(96,96,6)
                for lo,hi,name in [(47,49,'original_q_L_central4'),(32,64,'original_q_L_central32')]:
                    actual=weights[lo:hi,lo:hi].reshape(-1,6).mean(0);declared=row['measurements'][label][name]['owner5_plus_INVALID']
                    assert np.allclose(actual,[declared[k] for k in CLASSES],rtol=0,atol=2e-15)
                maps[label]=color(weights)
                if label=='SCENE':
                    assert ash(buf['RGB_uint8'])==witness['decoded_RGB_sha256_inherited'];cr,vr=crop(buf['RGB_uint8'],q,L,96);rgb96=cr.transpose(1,2,0).copy()
                    assert ash(rgb96)==witness['rgb96_array_sha256_inherited'] and ash(vr[0])==witness['valid96_array_sha256_inherited'];raw_before=ash(rgb96)
                proofs.append(dict(global_index=i,variant_label=label,source_buffer=b,onehot_owner96_array_SHA=ash(weights),owner96_display_palette_SHA=ash(maps[label]),central4_and_central32_match_completed_measurement=True))
            panels.append((witness,row,rgb96,maps,raw_before))
        fontpath=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK KR'],text=True).strip();fonts={n:ImageFont.truetype(fontpath,n) for n in (19,20,22,24,28,36)}
        image=Image.new('RGB',(2020,1510),'#f7f8fb');d=ImageDraw.Draw(image)
        def text(x,y,s,n=22,fill='#18283a'):d.text((x,y),s,font=fonts[n],fill=fill)
        text(28,20,'V172 | Source semantic role and original-query raster ownership',36)
        text(28,76,'Fixed3 extreme witnesses / same saved scene + BEFORE / original q,L / no correction or new training',24)
        for j,(name,rgb) in enumerate(zip(CLASSES,PALETTE)):
            x=28+(j%3)*640;y=122+(j//3)*37;d.rectangle((x,y,x+23,y+23),fill=tuple(rgb.astype(int)));text(x+32,y-3,name+(' (declared CF)' if j==0 else ''),20)
        for x,title in [(205,'SCENE: clean RGB96'),(570,'SCENE: owner96 weights'),(935,'BEFORE: owner96 weights')]:text(x,210,title,24)
        text(1310,210,'Original semantic / original q diagnostics',24)
        for n,(witness,row,rgb96,maps,original_hash) in enumerate(panels):
            top=260+n*370;i=witness['global_index'];semantic=witness['source_GT_metadata']['teacher_state'];text(28,top+4,str(i),28);text(28,top+48,'native '+str(witness['native_slot']),22);text(28,top+85,'semantic '+semantic,22)
            image.paste(Image.fromarray(rgb96).resize((320,320),Image.Resampling.NEAREST),(205,top))
            for x,label in [(570,'SCENE'),(935,'BEFORE_TARGET_ONLY')]:
                display=Image.fromarray(maps[label].copy()).resize((320,320),Image.Resampling.NEAREST);md=ImageDraw.Draw(display);md.line((150,160,170,160),fill='black',width=2);md.line((160,150,160,170),fill='black',width=2);image.paste(display,(x,top))
            qdiag=row['measurements']['SCENE']['original_q_point'];bd=row['measurements']['BEFORE_TARGET_ONLY']['original_q_point'];weights=qdiag['bilinear']['owner5_plus_INVALID']
            text(1310,top+4,f"Semantic {semantic} is the registered region role",24)
            text(1310,top+50,'q nearest: '+qdiag['nearest']['category']+'  id='+str(qdiag['nearest']['part_id']),22)
            text(1310,top+92,'q exact bilinear weights:',22);line=0
            for name,val in weights.items():
                if val>0:text(1330,top+128+line*34,f'{name}: {100*val:.4f}%',22);line+=1
            beforeweights=bd['bilinear']['owner5_plus_INVALID'];compact=' / '.join(k+f' {100*z:.2f}%' for k,z in beforeweights.items() if z>0)
            text(1310,top+214,'BEFORE q: '+compact,20)
            text(1310,top+251,f"q-reference error: {witness['source_GT_metadata']['error_px']:.6f}px",20)
            text(1310,top+285,'Owner-map black cross: display-only q center',19)
            assert ash(rgb96)==original_hash
        text(28,1380,'INCIDENT is relative to the declared CF association; initializer physical native identity is NOT certified.',22)
        text(28,1417,'CF/V4 correspondence_boundary is an ordering disagreement, not a pixel boundary. All24 original native rows remain in the numeric inventory.',20)
        text(28,1453,'No representative-sample, physical visibility gold, label-error or solution claim. No reference-centered crop / model / fit / relabel / admission.',20)
        png=OUT/'FIXED3_SOURCE_MASK_FORENSIC.png';image.save(png)
        for b in source_bindings:checked(b)
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,phase='V172',code=bind(__file__),protocol=bind(PROTOCOL),tests=p['tests'],measurement_complete=p['complete'],all24_inventory=p['rows'],selection=p['selection'],figure=bind(png),proofs=proofs,fixed_selected_rows=3,all24_inventory_unchanged=True,clean_RGB96_SHA_parity_all3=True,decoded_RGB_parity_all3=True,onehot_weight_mean_parity_all6_maps=True,source_buffers_SHA_unchanged=True,legend_class_order=list(CLASSES),new_source_scene_render=0,reference_centered_crop=0,new_model_fit_label_qL_threshold_or_training_admission=0,current_V169_private_V112_feature_weight_access=0,fully_solved=False,actual_visual_inspection='Pending separate tools.view_image receipt',automatic_retry=False))
        print(json.dumps(dict(figure=bind(png),complete=bind(OUT/'COMPLETE.json'))),flush=True)
    except BaseException:
        write(OUT/'FAILED_EXECUTION_PRESERVED.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if a.prepare and not a.run:prepare()
    elif a.run and not a.prepare:run()
    else:raise SystemExit('Use --prepare then authorized one --run')
