#!/usr/bin/env python3
"""Independent saved V235 pair QA. No compositing/producer imports/raw donor reads."""
import argparse,csv,hashlib,io,json,math,traceback
from collections import Counter
from pathlib import Path
import numpy as np
H=Path(__file__).parent

def ident(b):return (b['path'],b['sha256'])
def nearest_taps(q,L,r,halfup=False):
    nearest=[math.floor(float(v)+.5) if halfup else round(float(v)) for v in r]
    source=[[nearest[0]+dx,nearest[1]+dy] for dy in (-1,0,1) for dx in (-1,0,1)]
    mapped=[];taps=[]
    for sx,sy in source:
        u=(float(sx)-float(q[0]))/float(L)*96.+47.5;v=(float(sy)-float(q[1]))/float(L)*96.+47.5
        mapped.append([u,v]);x=math.floor(u);y=math.floor(v)
        taps.append([[x,y],[x+1,y],[x,y+1],[x+1,y+1]])
    return nearest,source,mapped,np.array(taps,np.int64)

def layer_consistency(c,s,ca,sa,valid,cp,sp,hw):
    height,width=hw;cx,cy=cp;sx,sy=sp
    assert height>0 and width>0 and 0<=cx<=96-width and 0<=sx<=96-width and 0<=cy<=96-height and 0<=sy<=96-height
    cm=ca[cy:cy+height,cx:cx+width];sm=sa[sy:sy+height,sx:sx+width]
    cv=valid[cy:cy+height,cx:cx+width];sv=valid[sy:sy+height,sx:sx+width]
    assert int(cm.sum())==int(ca.sum()) and int(sm.sum())==int(sa.sum())
    possible=cm|sm
    assert np.array_equal(cm,possible&cv) and np.array_equal(sm,possible&sv),'same donor-local alpha inconsistent with original-valid clipping'
    shared=cm&sm
    assert np.array_equal(c[:,cy:cy+height,cx:cx+width][:,shared],s[:,sy:sy+height,sx:sx+width][:,shared]),'same opaque donor-layer RGB mismatch'
    return int(shared.sum()),int((~cv&~sv).sum())

def self_test():
    # Half-tie disagreement is detected, never silently treated as equivalent.
    assert nearest_taps([10.,10.],96.,[10.5,11.5],False)[0]==[10,12]
    assert nearest_taps([10.,10.],96.,[10.5,11.5],True)[0]==[11,12]
    a=np.zeros((96,96),bool);a[20:24,20:24]=True;valid=np.ones_like(a);valid[20:22,20:22]=False
    ca=a&valid;sa=np.zeros_like(a);sa[40:44,40:44]=True
    c=np.zeros((3,96,96),np.uint8);s=c.copy() # Zero-change layers must be accepted.
    shared,unknown=layer_consistency(c,s,ca,sa,valid,[20,20],[40,40],[4,4]);assert shared==12
    bad=s.copy();bad[:,42,42]=255
    try:layer_consistency(c,bad,ca,sa,valid,[20,20],[40,40],[4,4])
    except AssertionError:pass
    else:raise AssertionError('corrupt same-layer RGB accepted')
    return {'completed':True,'PASS':True,'invented_only':True,'half_tie_detection':True,'valid_clipping_and_zero_RGB_change':True,'corrupt_layer_rejected':True}

def run(entry_path):
    # Only own explicit entry is opened before approval/closed-whitelist validation.
    ep=Path(entry_path);assert ep.parent==H
    e=json.loads(ep.read_text());assert e['approved_actual_saved_CF_QA'] is True
    assert e['raw_donor_PNGs_allowed'] is False and e['GT_prefix_model_current_held_allowed'] is False
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==e['verifier']['sha256']
    white={b['path']:b for b in e['whitelist']};assert len(white)==len(e['whitelist'])
    access=[];numeric_indices=set();out=Path(e['output_directory']);assert out.parent==H;out.mkdir(exist_ok=False)
    def safe(p):
        p=Path(p);low=str(p).lower()
        assert p.is_absolute() and '..' not in p.parts
        for word in ('v112','__pycache__','current223','publicheld','oldval','source_holdout','v177','v178','v182'):assert word not in low,word
        assert str(p) in white
        for q in [p]+list(p.parents):assert not q.is_symlink(),str(q)
        return p
    def checked(b):
        p=safe(b['path']);assert ident(b)==ident(white[str(p)])
        h=hashlib.sha256();n=0
        with p.open('rb') as f:
            for v in iter(lambda:f.read(4*1024*1024),b''):h.update(v);n+=len(v)
        assert h.hexdigest()==b['sha256'],str(p)
        if 'size_bytes' in b:assert n==b['size_bytes']
        access.append({'path':str(p),'sha256':b['sha256'],'size_bytes':n,'whole_file_access':'opaque SHA stream or permitted saved metadata'})
        return p
    def js(b):return json.loads(checked(b).read_text())
    try:
        complete=js(e['complete']);rootentry=js(e['root_build_entry'])
        assert complete['completed'] is True and complete['PASS'] is True
        cb=complete['entry']; eb=e['root_build_entry']
        ep=Path(cb['path'])
        if not ep.is_absolute():
            assert ep.parts==(Path(eb['path']).name,)  # exact recorded CLI basename only
            ep=Path(eb['path']).parent/ep
        assert str(ep)==eb['path'] and cb['sha256']==eb['sha256']
        if 'size_bytes' in cb:assert cb['size_bytes']==eb['size_bytes']
        assert rootentry['approved_exact_FIT_donor_pixels_and_pair_build'] is True
        assert complete['new_fit']==complete['current_or_held_payload']==0
        assert complete['source_bindings']==rootentry['source_bindings']
        donor_manifest=js(complete['donor_manifest']);donors=donor_manifest['donors'];donorby={d['donor_id']:d for d in donors}
        assert len(donors)==len(donorby)==64
        source=complete['source_bindings'];source_paths={k:checked(b) for k,b in source.items()}
        with np.load(source_paths['samples'],allow_pickle=False) as z:
            data={k:z[k] for k in z.files}
        assert len(data['target'])==7617
        diag={d['sample_index']:d for d in (json.loads(x) for x in source_paths['diagnostics'].read_text().splitlines())}
        assert len(diag)==7617
        with source_paths['YCB_reference_diagnostic'].open() as f:ycb={r['row_id']:r for r in csv.DictReader(f)}
        assert len(ycb)==1521
        output=complete['outputs'];assert set(output)=={'parent_indices.npy','success.npy','PAIR_RGB96.npy','PAIR_ALPHA_PACKBITS.npy','PAIR_METADATA.jsonl','SUMMARY.json'}
        paths={k:checked(b) for k,b in output.items()}
        parents=np.load(paths['parent_indices.npy'],allow_pickle=False);success=np.load(paths['success.npy'],allow_pickle=False)
        expected=np.flatnonzero(data['fit']&data['available']&(data['target']==0)).astype(np.int64)
        assert parents.dtype==np.int64 and parents.shape==(5406,) and np.array_equal(parents,expected)
        assert success.dtype==bool and success.shape==(5406,)
        rgb=np.load(paths['PAIR_RGB96.npy'],mmap_mode='r',allow_pickle=False);pack=np.load(paths['PAIR_ALPHA_PACKBITS.npy'],mmap_mode='r',allow_pickle=False)
        assert rgb.dtype==np.uint8 and rgb.shape==(5406,2,3,96,96)
        assert pack.dtype==np.uint8 and pack.shape==(5406,2,1152)
        raw=np.load(source_paths['rgb96'],mmap_mode='r',allow_pickle=False);valids=np.load(source_paths['valid96'],mmap_mode='r',allow_pickle=False)
        assert raw.dtype==np.uint8 and raw.shape==(7617,3,96,96)
        assert valids.dtype==bool and valids.shape==(7617,1,96,96)
        summary=json.loads(paths['SUMMARY.json'].read_text());counts=Counter();rows=[];rounding=Counter();zero=Counter();clipped=Counter();inside_shared=0
        metadata_lines=paths['PAIR_METADATA.jsonl'].read_text().splitlines();assert len(metadata_lines)==5406
        eligible=set(map(int,parents))
        for j,line in enumerate(metadata_lines):
            m=json.loads(line);i=int(parents[j]);assert i in eligible and bool(data['fit'][i]) and bool(data['available'][i]) and int(data['target'][i])==0
            # The only numerical accesses to original pixel/valid rows are through these checked FIT-D indices.
            base=np.array(raw[i],copy=True);valid=np.array(valids[i,0],copy=True);numeric_indices.add(i)
            d=diag[i];f=int(data['family'][i]);assert f in (0,1)
            assert m['pair_index']==j and m['parent_sample_index']==i and m['family']==f
            assert m['base_original_state']=='D' and m['original_role']=='FIT' and d['fit'] is True and d['available'] is True and d['target']==0
            assert str(data['row_id'][i])==d['row_id']==m['row_id']
            assert hashlib.sha256(base.tobytes()).hexdigest()==m['original_pixel_sha256']==d['pixel_sha256']
            assert hashlib.sha256(valid.tobytes()).hexdigest()==m['original_valid_sha256']==d['valid_sha256']
            assert m['original_query_xy']==d['query_xy'] and m['original_L']==d['L'] and m['q_L_crop_and_validity_unchanged'] is True
            assert m['source_generated_not_natural'] is True and m['native_gold_claim'] is False
            q=d['query_xy'];L=d['L'];yr=ycb[d['row_id']] if f==1 else None
            ref=[float(yr['r_x_GT_diagnostic_only']),float(yr['r_y_GT_diagnostic_only'])] if yr else d['reference_xy_GT_diagnostic_only']
            obj=int(yr['object_id']) if yr else None;assert m['teacher_reference_xy']==ref and m['base_object_id_sampling_only']==obj
            donor=donorby[m['donor_id']];assert donor['internal_source_role']=='INTERNAL_SOURCE_FIT' and donor['original_associated_r_state']=='D' and donor['scene_id'] in ('000000','000001','000002','000003','000004')
            assert ident(m['donor_RGB'])==ident(donor['RGB']) and ident(m['donor_mask'])==ident(donor['visible_mask'])
            assert m['donor_object_id_sampling_only']==donor['object_id'] and (obj is None or obj!=donor['object_id'])
            pool=[v for v in donors if obj is None or v['object_id']!=obj]
            pick=int.from_bytes(hashlib.sha256(('V235:pick:'+d['row_id']).encode()).digest(),'big')%len(pool);assert pool[pick]['donor_id']==donor['donor_id']
            nearest,points,mapped,taps=nearest_taps(q,L,ref,False);hn,hp,hm,ht=nearest_taps(q,L,ref,True)
            rounding['parents']+=1;rounding['nearest_rint_vs_halfup_different']+=int(nearest!=hn);rounding['required_taps_different']+=int(not np.array_equal(taps,ht))
            checks=m.get('teacherchecks')
            if checks:
                assert checks['reference_nearest_xy']==nearest and checks['reference_original_3x3_xy']==points
                assert np.allclose(checks['mapped_RAW96_xy'],mapped,rtol=0,atol=1e-12) and np.array_equal(checks['required_taps_xy'],taps)
            inside=bool(((taps>=0)&(taps<96)).all());hi=bool(((ht>=0)&(ht<96)).all())
            assert m['success']==bool(success[j])
            if not success[j]:
                assert m['status']=='CONSTRUCTION_FAILED' and m['masked'] is True and m['proposed_states'] is None
                assert not np.asarray(rgb[j]).any() and not np.asarray(pack[j]).any()
                reason=m['failure_reason'];counts[f'family{f}_failed:'+reason]+=1
                if reason=='REFERENCE_3X3_FOUR_TAP_SUPPORT_OUTSIDE_RAW96':assert not inside
                if reason=='REFERENCE_REQUIRED_TAP_ORIGINAL_INVALID':assert inside and not valid[taps[:,:,1],taps[:,:,0]].all()
                rows.append({'pair_index':j,'parent_sample_index':i,'family':f,'success':False,'failure_reason':reason});continue
            assert inside and hi and valid[taps[:,:,1],taps[:,:,0]].all() and valid[ht[:,:,1],ht[:,:,0]].all()
            assert m['status']=='CONSTRUCTED' and m['masked'] is False and m['proposed_states']=={'COVER':'E','SHAM':'D'}
            alpha=np.unpackbits(np.asarray(pack[j]),axis=1,bitorder='big').reshape(2,96,96).astype(bool)
            image=np.asarray(rgb[j]);c,s=image;ca,sa=alpha
            assert not ca[~valid].any() and not sa[~valid].any()
            assert ca[taps[:,:,1],taps[:,:,0]].all() and not sa[taps[:,:,1],taps[:,:,0]].any()
            # Original SOURCE half-up is independently checked as well as actual compositor rint.
            halfup_ok=bool(ca[ht[:,:,1],ht[:,:,0]].all() and not sa[ht[:,:,1],ht[:,:,0]].any());rounding['successful_pairs_halfup_teacher_pass']+=int(halfup_ok);assert halfup_ok
            assert np.array_equal(c[:,~ca],base[:,~ca]) and np.array_equal(s[:,~sa],base[:,~sa])
            assert [hashlib.sha256(image[k].tobytes()).hexdigest() for k in (0,1)]==m['pair_RGB_sha256_CHW']
            assert [hashlib.sha256(alpha[k].tobytes()).hexdigest() for k in (0,1)]==m['pair_alpha_sha256_bool']
            plan=m['donor_plan'];height,width=plan['resized_hw'];bbox=plan['foreground_bbox_xyxy_exclusive'];bh=bbox[3]-bbox[1];bw=bbox[2]-bbox[0]
            seed=int.from_bytes(hashlib.sha256(('V235:'+d['row_id']).encode()).digest(),'big')^235;assert plan['seed_integer']==seed
            rng=np.random.Generator(np.random.PCG64(seed));side=(32,40,48,56)[int(rng.integers(0,4))]
            assert plan['chosen_max_side']==side and [height,width]==[max(1,math.floor(bh*side/max(bh,bw)+.5)),max(1,math.floor(bw*side/max(bh,bw)+.5))]
            positions={}
            for role in ('COVER','SHAM'):
                xs=rng.integers(0,97-width,size=256,dtype=np.int64);ys=rng.integers(0,97-height,size=256,dtype=np.int64);a=np.stack((xs,ys),axis=1);positions[role]=a
                assert np.array_equal(a,plan['candidate_positions_xy'][role])
                assert hashlib.sha256(a.astype('<i8').tobytes()).hexdigest()==plan['candidate_positions_sha256'][role]
            rolechecks=checks['roles'];cp=rolechecks['COVER']['position_xy'];sp=rolechecks['SHAM']['position_xy']
            assert list(positions['COVER'][rolechecks['COVER']['candidate_index']])==cp and list(positions['SHAM'][rolechecks['SHAM']['candidate_index']])==sp
            shared,unseen=layer_consistency(c,s,ca,sa,valid,cp,sp,[height,width]);inside_shared+=shared
            changed=[int(np.any(im!=base,axis=0).sum()) for im in (c,s)]
            for k,role in enumerate(('COVER','SHAM')):
                z=rolechecks[role];assert z['changed_RGB_pixels']==changed[k] and z['alpha_visible_pixels']==int(alpha[k].sum())
                assert z['exact_base_RGB_same']==(changed[k]==0) and z['unchanged_outside_alpha'] is True and z['all_changes_only_original_valid'] is True
                zero[role]+=int(changed[k]==0)
            assert int(sa.sum())>=64
            clipped['pair_alpha_area_differs']+=int(ca.sum()!=sa.sum());counts[f'family{f}_success']+=1
            rows.append({'pair_index':j,'parent_sample_index':i,'family':f,'success':True,'COVER_alpha':int(ca.sum()),'SHAM_alpha':int(sa.sum()),'COVER_changed':changed[0],'SHAM_changed':changed[1],'same_layer_shared_pixels':shared,'both_positions_invalid_local_pixels':unseen,'halfup_teacher_pass':halfup_ok})
            if (j+1)%1000==0:print(json.dumps({'checked':j+1,'total':5406}),flush=True)
        assert len(numeric_indices)==5406 and numeric_indices==eligible
        assert int(success.sum())==5367 and counts['family0_success']==4537 and counts['family1_success']==830
        assert summary['successful']==int(success.sum()) and summary['pairs_attempted']==5406 and summary['original_rows_preserved']==7617
        assert dict(counts)==summary['counts'] and summary['failed_pair_not_E'] is True and summary['original_q_L_native_condition_valid_unchanged'] is True
        # Re-stream original source hashes, without numeric access to any additional rows.
        for key in ('samples','diagnostics','YCB_reference_diagnostic','rgb96','valid96'):checked(source[key])
        columns=sorted(set().union(*(r.keys() for r in rows)))
        with (out/'ALL5406_PAIR_QA.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
        result={'completed':True,'PASS':True,'complete':e['complete'],'root_build_entry':e['root_build_entry'],'verifier':e['verifier'],'counts':dict(counts),'parents_checked':5406,'success_pairs_checked':5367,'failed_original_D_preserved':39,'source_original_rows_retained':7617,'original_pixel_numeric_rows_accessed':5406,'original_DEV_pixel_numeric_rows_accessed':0,'rounding_rint_vs_original_halfup':dict(rounding),'exact_zero_changed_RGB_pairs_by_role':dict(zero),'valid_clipping_alpha_area_difference':dict(clipped),'common_visible_donor_local_pixels_compared':inside_shared,'all_source_hashes_unchanged_after_QA':True,'raw_donor_GT_depth_model_prefix_current_held_access':0,'new_compositing_or_model_forward_or_fit':0,'access':access,'limits':['Same donor-local RGB/alpha is verified where observable under both placements and original validity; resized raw donor is not rerendered or decoded. Regions invalid in both placements cannot be independently checked.','Failure zeros/masked/original D preservation and recorded reference-support failure reasons are checked; candidate-budget exhaustiveness is not independently rerendered from donor pixels.','Original q/L equality is explicit; native/condition and original validity are inherited through exact parent indices and unchanged source pack SHA, not new per-variant feature sidecars.','Associated-r AUX and SYN witness0 MAIN remain separate weak/source authorities, not new native human gold or current-real evidence.']}
        (out/'REVIEW.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'completed':True,'PASS':True,'review':str(out/'REVIEW.json'),'counts':dict(counts),'rounding':dict(rounding),'zero_changed':dict(zero),'alpha_area_differs':dict(clipped)}))
    except BaseException:
        (out/'FAILED_REVIEW.json').write_text(json.dumps({'completed':True,'PASS':False,'traceback':traceback.format_exc(),'automatic_retry':False,'original_pixel_numeric_indices_accessed':sorted(numeric_indices)},indent=2)+'\n')
        raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:
        d=self_test();(H/'SEALED_ADAPTER_INVENTED_CHECKS.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
    else:run(a.entry)
