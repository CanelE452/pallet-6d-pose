"""Inert V116 renderer. Complete corpus+root entry+selection lock precede pixels."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import traceback
import numpy as np
from PIL import Image
import visual_math as vm

R=Path(__file__).resolve().parent;V=R.parent;P=V.parent
PYTHON=Path('/home/minjae/anaconda3/envs/pallet-yolo26/bin/python')
CORPUS=V/'corpus_build_v1/CORPUS_OUTPUT_SEAL.json'
POLICY=V/'ROOT_SOURCE_POLICY_BEFORE_DONORS_AND_PIXELS.json'
OUT=V/'fixed_source_visual_review_v1'
STATIC=dict(code=Path(__file__).resolve(),helper=R/'visual_math.py',
    generated=R/'GENERATED_VISUAL_RENDER_TESTS.json',protocol=R/'VISUAL_RENDER_PROTOCOL.json',
    root_policy=POLICY,corpus_seal=CORPUS)


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s]
def pixels(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def check(b):require(sha(b['path'])==b['sha256'],'Binding changed: '+b['path']);return Path(b['path'])
def write(p,o):
    with p.open('x')as f:json.dump(o,f,indent=2,allow_nan=False);f.write('\n')


def entry_structure(e):
    require(set(e)=={'schema','approved','allow_public_visual_payloads','source_only','current223','V112','fit','PnP','expected_attempts','expected_occurrences','output_directory','bindings'},'Exact entry fields')
    require(e['schema']=='V116_ROOT_FIXED_SOURCE_VISUAL_ENTRY_V1'and e['approved']is True and e['allow_public_visual_payloads']is True,'Separate approved root visual entry required')
    require(e['source_only']is True and e['current223']is False and e['V112']is False and e['fit']is False and e['PnP']==0,'Source-only visual scope')
    require(e['expected_attempts']==448 and e['expected_occurrences']==924,'Full fixed cohort')
    require(Path(e['output_directory']).resolve()==OUT,'Fixed new visual output')
    require(Path(sys.executable).absolute()==PYTHON.absolute(),'Pinned display Python')
    require(set(e['bindings'])==set(STATIC),'Exact binding whitelist')
    for k,p in STATIC.items():
        b=e['bindings'][k];require(set(b)=={'path','sha256'}and Path(b['path']).resolve()==p,'Strict binding path '+k)


def guarded_metadata(entry):
    e=read(entry);entry_structure(e)
    require(not OUT.exists(),'Preserve outputs; no overwrite or automatic retry')
    for b in e['bindings'].values():check(b)
    protocol=read(STATIC['protocol']);proof=read(STATIC['generated'])
    require(protocol['completed']and protocol['root_policy']==e['bindings']['root_policy'],'Exact root policy')
    require(protocol['code']==e['bindings']['code']and protocol['helper']==e['bindings']['helper'],'Protocol code')
    require(proof['PASS']and proof['bindings']['renderer']==e['bindings']['code']and proof['bindings']['helper']==e['bindings']['helper'],'Generated proof code')
    seal=read(CORPUS)
    require(seal['completed']is True and seal['PASS']is True and seal['source_only']is True,'Completed source corpus before pixels')
    require(seal['current223']is False and seal['V112']is False and seal['model_forward_or_fit']==0,'No private/model corpus')
    # Deliberately do not iterate corpus files[]: it contains pixel payloads.
    for k in ['occurrences','GT','feasibility','donor_certificates','summary']:check(seal[k])
    occ=rows(seal['occurrences']['path']);gt=rows(seal['GT']['path']);attempts=rows(seal['feasibility']['path']);donors=rows(seal['donor_certificates']['path'])
    require(len(occ)==len(gt)==924 and len(attempts)==448 and len(donors)==16,'Complete 924/448/16 metadata')
    require([r['occurrence_index']for r in occ]==list(range(924)),'Original occurrence order')
    for j,(o,g)in enumerate(zip(occ,gt)):
        require(all(o[k]==g[k]for k in ['occurrence_index','occurrence_id','base_index','original_base_row_index','donor_rank','variant']),'Input/GT identity join')
        require(o['base_index']==j//33,'33 records per original base')
    for j,a in enumerate(attempts):
        require(a['attempt_index']==j and a['base_index']==j//16 and a['donor_rank']==j%16,'Original attempt/base/rank order')
    require([d['donor_rank']for d in donors]==list(range(16))and len({d['donor_id']for d in donors})==16,'Locked16 donors')
    return e,seal,occ,gt,attempts,donors


def run(entry):
    e,seal,occ,gt,attempts,donors=guarded_metadata(entry)
    chosen=vm.select_attempts(attempts)
    OUT.mkdir();(OUT/'cards').mkdir();(OUT/'raw96').mkdir();(OUT/'donors').mkdir()
    write(OUT/'RUN_STARTED.json',dict(entry=bind(entry),corpus_seal=e['bindings']['corpus_seal'],source_only=True,model_calls=0))
    lock=dict(completed=True,selection=chosen,selected_attempt_metadata=[attempts[j]for j in chosen['selected_attempt_indices']],
        all448_attempts_retained_in_source=seal['feasibility'],selection_before_any_RGB_or_NPZ_read=True,
        source_only=True,training_admission=False,native_mapping='UNKNOWN',human_gold=False)
    write(OUT/'SELECTION_LOCK_BEFORE_RGB.json',lock)
    # No source pixel or array can be reached above this line.
    start=time.monotonic();cards=[];case_receipts=[];cache={};donor_cache={}
    def rgb(b,fmt,expected,hw=None):
        key=(b['path'],b['sha256'],fmt)
        if key not in cache:
            p=check(b)
            if fmt=='NPY':a=np.load(p,allow_pickle=False)
            elif fmt=='PNG':
                with Image.open(p)as im:a=np.array(im.convert('RGB'),copy=True)
            else:raise ValueError('Explicit source RGB format required')
            require(a.dtype==np.uint8 and a.ndim==3 and a.shape[2]==3,'Native RGB uint8 HWC')
            require(pixels(a)==expected,'Exact decoded source/composite SHA')
            if hw is not None:require(a.shape[:2]==tuple(hw),'Native HW unchanged')
            cache[key]=a
        return cache[key]
    try:
        for ai in chosen['selected_attempt_indices']:
            a=attempts[ai];bi=a['base_index'];rank=a['donor_rank'];d=donors[rank]
            require(d['donor_id']==a['donor_id'],'Fixed donor ID')
            indices=[33*bi,33*bi+1+2*rank,33*bi+2+2*rank];variants=[occ[j]for j in indices]
            require([o['variant']for o in variants]==['BEFORE','OPAQUE_COVER','CUE_PRESERVING_SHAM'],'Three fixed variants')
            require(all(o['query_xy']==variants[0]['query_xy']and o['L']==variants[0]['L']and o['raw_hw']==variants[0]['raw_hw']for o in variants),'Unchanged q/L/HW')
            require(all(gt[j]['pair_ready']==a['pair_ready']for j in indices[1:]),'Construction status unchanged')
            protected_path=check(a['protected_union_file'])
            with np.load(protected_path,allow_pickle=False)as z:protected=z[a['protected_union_key']]
            require(protected.dtype==bool and protected.shape==tuple(variants[0]['raw_hw']),'Native protected union')
            support_hash=pixels(protected);q=np.asarray(variants[0]['query_xy'],np.float64);L=float(variants[0]['L'])
            rawctx=[];markedctx=[];rawcrops=[];markedcrops=[];crop_arrays=[];valid_arrays=[];variant_receipts=[]
            for vi,(o,j)in enumerate(zip(variants,indices)):
                image=rgb(o['fullRGB'],o['format'],o['decoded_RGB_sha256'],o['raw_hw']);beforehash=pixels(image)
                local,valid=vm.crop(image,q,L,96);raw=local.transpose(1,2,0)
                local_support=vm.protected_crop(protected,q,L)
                rawctx.append(Image.fromarray(image));markedctx.append(vm.marked(image,protected,q,[q[0]-L/2,q[1]-L/2,q[0]+L/2,q[1]+L/2]))
                rawcrops.append(Image.fromarray(raw));markedcrops.append(vm.marked(raw,local_support,[47.5,47.5]))
                cp=OUT/'raw96'/('ATTEMPT_%03d_%s.png'%(ai,o['variant']));Image.fromarray(raw).save(cp)
                crop_arrays.append(local);valid_arrays.append(valid)
                require(beforehash==pixels(image)and support_hash==pixels(protected),'Display never mutates source/support')
                variant_receipts.append(dict(occurrence_index=j,variant=o['variant'],fullRGB=o['fullRGB'],
                    decoded_RGB_sha256=o['decoded_RGB_sha256'],raw_hw=o['raw_hw'],query_xy=o['query_xy'],L=o['L'],
                    original_record=o,construction_GT_record=gt[j],RGB96_pixels_sha256=pixels(local),
                    valid96_fraction=float(valid.mean()),raw96_png=bind(cp)))
            if rank not in donor_cache:
                drgb=rgb(d['image'],'PNG',d['decoded_RGB_sha256']);dh=pixels(drgb)
                with np.load(check(d['native_ownership_proof']),allow_pickle=False)as z:
                    require(set(z.files)==set(d['native_ownership_keys']),'Donor ownership schema')
                    alpha=z['exclusive_alpha'];core=z['erode1_exclusive_alpha']
                require(alpha.dtype==bool and core.dtype==bool and alpha.shape==core.shape==drgb.shape[:2],'Native donor alpha')
                panel,extent=vm.donor_display(drgb,alpha)
                dp=OUT/'donors'/('DONOR_%02d_CANONICAL_EXCLUSIVE_ALPHA.png'%rank);panel.save(dp)
                require(pixels(drgb)==dh,'Donor RGB unchanged')
                donor_cache[rank]=(panel,dict(donor_id=d['donor_id'],donor_rank=rank,image=d['image'],
                    decoded_RGB_sha256=d['decoded_RGB_sha256'],ownership_proof=d['native_ownership_proof'],
                    exclusive_alpha_pixels=int(alpha.sum()),erode1_core_pixels=int(core.sum()),
                    display_only_native_alpha_extent=extent,canonical_display=bind(dp)))
            donor_panel,donor_receipt=donor_cache[rank]
            combined=Image.new('RGB',(400,200),'white');combined.paste(donor_panel,(0,0))
            transformed=None
            if a['cutout_RGBA_file']is not None:
                with np.load(check(a['cutout_RGBA_file']),allow_pickle=False)as z:
                    require(z.files==a['cutout_RGBA_keys'],'Transformed cutout schema');cr=z['rgb'];ca=z['alpha']
                require(cr.shape[:2]==ca.shape and cr.shape[2]==3 and np.isfinite(cr).all()and np.isfinite(ca).all(),'Finite transformed RGBA')
                require(cr.min()>=0 and cr.max()<=255 and ca.min()>=0 and ca.max()<=1,'RGBA display value ranges')
                cu=np.floor(np.clip(cr,0,255)+.5).astype(np.uint8);yy,xx=np.indices(ca.shape)
                bg=np.repeat(np.where(((xx//8+yy//8)%2)[...,None]==0,220.,245.),3,axis=2)
                shown=np.floor(cu*ca[...,None]+bg*(1-ca[...,None])+.5).astype(np.uint8)
                combined.paste(vm.fit(Image.fromarray(shown),(200,200)),(200,0));transformed=a['cutout_RGBA_file']
            group='ACCEPTED'if a['pair_ready']else'FAILED_RETAINED_U'
            before=variants[0];title='Attempt %03d | %s | base %02d (original %s) | donor rank%02d'%(ai,group,bi,a['original_base_row_index'],rank)
            subtitle='Same original q=%s L=%.6f | source=%s | %s | reason=%s'%(before['query_xy'],L,before.get('source','UNKNOWN'),before.get('raw_split',before.get('split','UNKNOWN')),a['reason'])
            notes='Object16 fixed donor %s. Left: original native exclusive-alpha display; right: same transformed RGBA if available. Alpha/checkerboard is display-only. Failed afters remain U. Protected union includes all matched cue regions + central stencil.'%d['donor_id']
            card=vm.card(title,subtitle,rawctx,markedctx,rawcrops,markedcrops,combined,notes,
                'Rows: RAW full / marked full / RAW96 / marked96. AUX diagnostic only; native UNKNOWN; no accuracy/gold.')
            path=OUT/'cards'/('ATTEMPT_%03d_%s.png'%(ai,group));card.save(path);cards.append(card)
            npz=OUT/'raw96'/('ATTEMPT_%03d_RGB96_VALID.npz'%ai)
            np.savez_compressed(npz,rgb96=np.stack(crop_arrays),crop_valid96=np.stack(valid_arrays))
            case_receipts.append(dict(attempt_index=ai,group=group,feasibility_record=a,
                original_q=before['query_xy'],original_L=L,variants=variant_receipts,
                protected_union=a['protected_union_file'],donor=donor_receipt,transformed_RGBA=transformed,
                case_card=bind(path),raw96_valid=bind(npz),semantic_labels_changed=False))
        contact=OUT/'CONTACT_ALL_FIXED_CASES.png';vm.contact(cards,chosen['missing_groups']).save(contact)
        receipt=dict(completed=True,PASS=True,entry=bind(entry),corpus_seal=e['bindings']['corpus_seal'],
            selection_lock=bind(OUT/'SELECTION_LOCK_BEFORE_RGB.json'),selected=chosen,cases=case_receipts,
            case_count=len(case_receipts),contact=bind(contact),source_RGB_files_decoded=len(cache),
            seconds=time.monotonic()-start,source_only=True,model_calls=0,fit=0,current223=False,V112=False,
            training_admission=False,human_gold=False,native_mapping='UNKNOWN',source_pixel_mutations=0,PnP=0,
            interpretation='Conditional recorded-cue cover/sham construction examples, not independent performance or semantic/native pixel gold. Inspect boundary mixing, own holes, wood foreground versus wood target, and protected/centre edits.')
        write(OUT/'RENDERING_RECEIPT.json',receipt)
        write(OUT/'RENDERING_SEAL.json',dict(completed=True,PASS=True,receipt=bind(OUT/'RENDERING_RECEIPT.json'),
            entry=bind(entry),corpus_seal=e['bindings']['corpus_seal'],files=[bind(p)for p in sorted(OUT.rglob('*'))if p.is_file()],
            model_calls=0,fit=0,current223=False,V112=False,training_admission=False))
        print(json.dumps(dict(completed=True,cases=len(case_receipts),selected=chosen,receipt=bind(OUT/'RENDERING_RECEIPT.json'),seal=bind(OUT/'RENDERING_SEAL.json'))))
    except BaseException:
        write(OUT/'FAILED.json',dict(completed=False,exception=traceback.format_exc(),no_automatic_retry=True,model_calls=0,fit=0,current223=False,V112=False));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--entry',required=True);run(ap.parse_args().entry)
