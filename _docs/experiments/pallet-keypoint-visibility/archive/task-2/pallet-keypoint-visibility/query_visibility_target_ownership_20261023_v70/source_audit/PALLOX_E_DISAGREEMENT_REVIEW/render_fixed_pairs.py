"""Two fixed public source RGB pairs and saved conditional-certificate replay."""
from pathlib import Path
import hashlib,json,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
OUT=Path(__file__).resolve().parent;V70=OUT.parents[1];BASE=V70.parent
V66=BASE/'query_visibility_controlled_real_20261022_v66';PUB=V66/'public_pallet'
F='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=lambda n:ImageFont.truetype(F,n)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quant(a):
    a=np.asarray(a).reshape(-1);return {'N':int(a.size),'min':float(a.min()),'p05':float(np.percentile(a,5)),'median':float(np.median(a)),'p95':float(np.percentile(a,95)),'max':float(a.max())}
def marker(im,q,origin=(0,0),scale=1):
    d=ImageDraw.Draw(im);x=(q[0]-origin[0]+.5)*scale;y=(q[1]-origin[1]+.5)*scale
    for line in [(x-11,y,x-4,y),(x+4,y,x+11,y),(x,y-11,x,y-4),(x,y+4,x,y+11)]:d.line(line,fill=(0,255,255),width=2)
    d.ellipse((x-3,y-3,x+3,y+3),outline=(255,0,220),width=1)
def main():
    dest=OUT/'SAVED_SOURCE_CERTIFICATE_AND_CROPS.json';assert not dest.exists()
    lock=OUT/'SELECTION_LOCK.json';assert sha(lock)=='d2fb818f37602a0d2bfc22fd088902698f3425bdc546d8985291e2a18459d853'
    a=json.loads(lock.read_text())
    for b in a['bindings']:assert sha(b['path'])==b['sha256']
    results=[];panels=[]
    for rank,r in enumerate(a['selected']):
        meta=r['source_pair_evidence_metadata'];files=meta['files'];q=r['selected_label']['query'];L=r['selected_input_metadata']['side'];x,y=[math.floor(v+.5) for v in q];cx,cy=x,y;b=(cx-48,cy-48,cx+48,cy+48)
        images={};depth={};bindings={}
        for view in ['A','B']:
            for kind in ['image','depth','mask_teacher','pose','K']:assert sha(files[view][kind]['path'])==files[view][kind]['sha256']
            images[view]=Image.open(files[view]['image']['path']).convert('RGB')
            depth[view]=np.asarray(Image.open(files[view]['depth']['path'])).astype(np.float64)
            assert list(images[view].size[::-1])==[480,640] and depth[view].shape==(480,640)
            bindings[view]=files[view]
        assert files['A']['pose']['sha256']==files['B']['pose']['sha256'] and files['A']['K']['sha256']==files['B']['K']['sha256']
        evidence=PUB/meta['evidence']['path'];assert sha(evidence)==meta['evidence']['sha256']
        keys=['visible_A','visible_B','external_A','external_B','teacher_core_A','teacher_core_B','mesh_agree_A','mesh_agree_B','source_mesh_depth_m']
        with np.load(evidence,allow_pickle=False) as z:ev={k:z[k] for k in keys}
        assert ev['external_A'][y,x] and ev['visible_B'][y,x]
        win=(slice(y-2,y+3),slice(x-2,x+3));DA=depth['A'];DB=depth['B'];M=ev['source_mesh_depth_m'].astype(np.float64)*1000
        tauB=np.maximum(20,.01*DB);meshmargin=np.maximum(50,2*tauB)
        qstats={'native_xy':[x,y],'query_xy':q,'fixed5x5_window_xyxy':[x-2,y-2,x+3,y+3],
            'existing_external_A_true_in25':int(ev['external_A'][win].sum()),'existing_visible_B_true_in25':int(ev['visible_B'][win].sum()),
            'existing_teacher_core_B_true_in25':int(ev['teacher_core_B'][win].sum()),'existing_mesh_agree_B_true_in25':int(ev['mesh_agree_B'][win].sum()),
            'exact_saved_certificate':{k:bool(ev[k][y,x]) for k in keys if k!='source_mesh_depth_m'},
            'exact_depth_mm':{'A':float(DA[y,x]),'B':float(DB[y,x]),'supplied_mesh':float(M[y,x]),'B_minus_A':float(DB[y,x]-DA[y,x]),'mesh_minus_A':float(M[y,x]-DA[y,x]),'tau_target_B':float(tauB[y,x]),'fixed_mesh_foreground_margin':float(meshmargin[y,x])},
            'fixed25_depth_B_minus_A_mm':quant((DB-DA)[win]),'fixed25_mesh_minus_depth_A_mm':quant((M-DA)[win]),'fixed25_abs_mesh_minus_depth_B_mm':quant(np.abs(M-DB)[win]),
            'conditional_formula_replay_all25':bool(((DB-DA>=50)&(M-DA>=meshmargin)&ev['visible_B']&~ev['visible_A']&(DA>0))[win].all()),
            'formula_scope':'Existing fixed50mm external certificate, tauB=max20mm,1%DB, mesh foreground>=max50mm,2tauB. Replay diagnostics only; labels/protocol unchanged.'}
        panel=Image.new('RGB',(1600,1150),(24,26,29));d=ImageDraw.Draw(panel)
        d.text((12,8),f"{rank}: {r['sample_id']} | same q across paired views | source conditional E vs V",font=font(22),fill='white')
        d.text((12,42),f"q=({q[0]:.3f},{q[1]:.3f}), original L={L:.3f}; cyan open cross=q, thin cyan rectangle=original q/L region",font=font(17),fill='white')
        for vi,view in enumerate(['A','B']):
            image=images[view].copy();md=ImageDraw.Draw(image);md.rectangle((q[0]-L/2,q[1]-L/2,q[0]+L/2,q[1]+L/2),outline=(0,200,220),width=1);marker(image,q,scale=1)
            xx=12+vi*790;panel.paste(image,(xx,110));d.text((xx,80),f"{view} frame{meta[view+'_frame']}  "+('selected covered view / conditional E' if view=='A' else 'paired reference-visible view / conditional V'),font=font(18),fill=(255,190,110) if view=='A' else(160,240,160))
        for vi,view in enumerate(['A','B']):
            raw=images[view].crop(b).resize((384,384),Image.Resampling.NEAREST);marked=raw.copy();marker(marked,q,(b[0],b[1]),4)
            xx=12+vi*790;panel.paste(marked,(xx,638));panel.paste(raw,(xx+392,638))
            d.text((xx,608),f"{view} same native96 crop x4: marked / RAW unchanged",font=font(18),fill='white')
        ytxt=1037
        for j,(name,short) in enumerate([('MOBILE_SAM_RAW','MobileSAM'),('V65_BASELINE','V65'),('V66_CONTROLLED_PLUS_PALLOX','V66'),('V70_OWNERSHIP_LOO_PAIR_V1','V70')]):
            d.text((12+(j%2)*790,ytxt+(j//2)*30),f"{short} pV: A={float(r['selected_saved_scores'][name]['p_visible']):.6f} / B={float(r['paired_saved_scores'][name]['p_visible']):.6f}",font=font(18),fill='white')
        d.text((12,1110),'Supplied same pose/K and depth teacher support pairing; native RGB-depth pixel registration is assumed, not certified. No new PnP/fit.',font=font(16),fill=(255,210,130))
        path=OUT/f'FIXED_PAIR_{rank}_{meta["pair_id"].replace(":","__")}.png';panel.save(path);panels.append(panel)
        results.append({'sample_id':r['sample_id'],'source_pair_id':meta['pair_id'],'source_family':meta['session'],'q':q,'original_L':L,'diagnostic_native96_crop_bounds':list(b),
            'selected_label':r['selected_label'],'paired_label':r['paired_reference_label'],'saved_scores_selected_A':r['selected_saved_scores'],'saved_scores_paired_B':r['paired_saved_scores'],
            'query_saved_teacher_certificate_diagnostics':qstats,'unchanged_pairwide_quantiles':{k:meta[k] for k in ['common_teacher_depth_abs_residual_mm','mesh_minus_depth_B_teacher_core_mm','supplied_pose_reprojection_px','coverage_descriptive']},
            'teacher_limits':{'tier':meta['tier'],'pixel_RGB_depth_alignment_certified':meta['pixel_RGB_depth_alignment_certified'],'native_depth_is_assumed_RGB_grid':meta['native_depth_is_assumed_RGB_grid'],'same_pose_and_K':meta['supplied_pose_and_K_identical'],'pose_numerical_identity_does_not_prove_physical_registration':True,'human_verified':False,'corner_semantic_truth':None},
            'source_bindings':bindings,'evidence_npz':{'path':str(evidence),'sha256':sha(evidence),'decoded_keys':keys},'figure':{'path':str(path),'sha256':sha(path)}})
    contact=Image.new('RGB',(1600,2300),(18,20,23))
    for j,panel in enumerate(panels):contact.paste(panel,(0,j*1150))
    cp=OUT/'FIXED_TWO_PUBLIC_PALLOX_E_DISAGREEMENTS.png';contact.save(cp)
    result={'schema':'FIXED_PUBLIC_PALLOX_PAIR_VISUAL_EVIDENCE_V1','selection_lock':{'path':str(lock),'sha256':sha(lock)},'case_count':2,'unique_RGB_images_decoded':4,'native_depth_images_decoded_for_existing_certificate_replay':4,
        'source_evidence_scope':'Existing saved conditional teacher and original public source only. No new inference, model/features/weights/current data, label correction, relocation, threshold/radius fitting, PnP or pose optimization.',
        'all_sample_IDs_queries_originalL_labels_and_saved_scores_preserved':True,'contact_sheet':{'path':str(cp),'sha256':sha(cp)},'cases':results,'code_sha256':sha(Path(__file__))}
    write=lambda path,obj:path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    write(dest,result);print(json.dumps({'path':str(dest),'sha256':sha(dest),'figures':[r['figure']['path'] for r in results],'query_certificates':[r['query_saved_teacher_certificate_diagnostics'] for r in results]},ensure_ascii=False))
if __name__=='__main__':main()
