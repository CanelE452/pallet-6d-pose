"""Independent SOURCE TRAIN saved augmentation audit, never a producer/model import."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,time
import numpy as np
from independent_math import raw96,owner4,project_patch,end_classes,mass96,target
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent;A=V/'augmentation_preparation_a'
V180=P/'query_visibility_frozen_sam_source_baseline_20261024_v180';V181=P/'query_visibility_local_support_raw_head_20261024_v181'
SCOPE=V/'ROOT_SCOPE_BEFORE_NEW_AUGMENTATION_OR_FIT.json';VAR=['BEFORE','COVER','NEAR_SHAM','FAR_SHAM'];N=65536
def need(x,msg):
    if not bool(x):raise AssertionError(msg)
def sha(p):
    p=Path(p).resolve();need(p.is_relative_to(P),'Source project only')
    need(not any(s in str(p).lower() for s in ['_v112','_v169','_v177','_v178','current223','synth_val','validation2048','/private_']),'TRAIN whitelist')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def binding(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):need(sha(b['path'])==b['sha256'],'SHA '+b['path']);return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def npz(b,keys=None):
    with np.load(check(b),allow_pickle=False) as z:return {k:z[k] for k in (z.files if keys is None else keys)}
def metadata():
    scope=read(SCOPE);states=scope['training_synthetic']['states15'];scenes=rows(A/'SCENE512_ORIGINALS_INPUT_ONLY.jsonl');rr=rows(A/'AUG65536_ROW_REFS_INPUT_ONLY.jsonl')
    old=rows(V180/'ALL4096_TRAIN_INPUTS_METADATA_ONLY.jsonl');need(len(scenes)==512 and len(rr)==N and len(old)==4096,'All original and augmentation counts')
    for s in scenes:need(s['split']=='TRAIN' and s['original_input_metadata8']==old[s['scene_index']*8:s['scene_index']*8+8],'Exact originalscene metadata8')
    for i,r in enumerate(rr):
        if i<61440:orig,st=divmod(i,15);anchor=orig;family='SELF_SCENE15'
        else:
            j=i-61440;base,rem=divmod(j,32);vi,k=divmod(rem,8);orig=base*32+vi*8+k;anchor=base*32+k;st=0;family='BEFORE_ANCHOR4'
        im=old[orig];an=old[anchor];state=states[st];q=None if an['original_query_xy'] is None else [float(an['original_query_xy'][0])+float(state['dx']),float(an['original_query_xy'][1])+float(state['dy'])];L=None if an['original_L'] is None else float(an['original_L'])*float(state['scale'])
        need(r['original_global_index']==r['diagnostic_row_index']==i and r['family']==family,'new indices/family')
        need(r['source_image_original_global_index']==orig and r['anchor_original_global_index']==anchor,'dual production origin')
        need(r['scene_index']==im['scene_index'] and r['anchor_scene_index']==an['scene_index'] and r['base_index']==im['base_index'] and r['variant']==im['variant'] and r['native_slot']==im['native_slot'],'source dimensions')
        need(r['source_image_original_query_id']==im['original_occurrence_id'] and r['anchor_original_query_id']==an['original_occurrence_id'],'original query IDs')
        need(r['query_xy']==q and r['L']==L and r['original_query_available']==an['original_input_flags']['query_available'],'all FP64 qL and factual availability')
        need(r['state_index']==st and r['state_name']==state['name'] and [r['dx'],r['dy'],r['L_scale']]==[state['dx'],state['dy'],state['scale']],'root15states ordering')
        need(r['original_flags_ref']==[an['scene_index'],im['native_slot']] and r['image_flags_ref']==[im['scene_index'],im['native_slot']],'all originalflags refs')
        expected_sam='SELF_SCENE_%04d'%im['scene_index'] if family=='SELF_SCENE15' else 'BEFORE_ANCHOR_BASE%03d/%s'%(im['base_index'],im['variant']);need(r['sam_group_id']==expected_sam and r['split']=='TRAIN','SAM prompt family/TRAIN')
    need(len({r['row_id'] for r in rr})==N and sum(not r['original_query_available'] for r in rr)==1264,'all IDs/U1264')
    first={}
    for base in range(128):
        f=old[base*32]['factors'];first.setdefault(tuple(f[k] for k in ['geometry_family','material','aspect']),base)
    need(len(first)==8,'8fullfactorcells')
    return scenes,rr,old,sorted(first.values())
def invented():
    tests={};rgb=np.full((8,8,3),125,np.uint8);im,valid=raw96(rgb,[4.,4.],1.);tests['independent_raw_uniform_interior_exact']=np.all(im==125) and valid.all()
    pkt=dict(part_id=np.full((8,8),999,np.int32),ambiguity=np.zeros((8,8),bool));own=owner4(pkt,[4.,4.],1.);tests['ownerF_separate_from_end_foreign']=own.tolist()==[0.,1.,0.,0.,0.]
    pkt['ambiguity'][:]=True;tests['ownerambiguity_priority']=owner4(pkt,[4.,4.],1.).tolist()==[0.,0.,0.,1.,0.]
    tests['owner_invalid_not_BG']=owner4(pkt,[-2.,4.],.1).tolist()==[0.,0.,0.,0.,1.]
    tests['tiny_visible_existence_and_unknown_nonexclusive']=target(np.array([1-1e-17,1e-17,0,0,0,0,0]),True,True,False)==[1,0] and target(np.array([0,.5,0,.5,0,0,0]),True,True,False)==[1,-1]
    tests['inputU_oof_noend_rules']=target(np.array([1.,0,0,0,0,0,0]),True,True,False)==[0,-1] and target(np.array([0.,1,0,0,0,0,0]),False,True,False)==[-1,-1] and target(np.array([0.,1,0,0,0,0,0]),True,True,True)==[-1,-1]
    cam=dict(K=[[10.,0.,4.],[0.,10.,4.],[0.,0.,1.]],R=np.eye(3).tolist(),t=[0.,0.,0.]);quad=[[-.1,-.1,2.],[.1,-.1,2.],[.1,.1,2.],[-.1,.1,2.]];dep,good,oof=project_patch(dict(camera=cam,quads_world=[quad]*3));tests['independent_ray_plane_patch_depth']=good and not oof and dep[4,4]==2 and np.isfinite(dep).sum()==1
    before=dict(part_id=np.zeros((512,512),np.int32),camera_depth=np.full((512,512),2.,np.float64),global_triangle_id=np.zeros((512,512),np.int32),ambiguity=np.zeros((512,512),bool));scene={k:v.copy() for k,v in before.items()}
    tests['independent_exact_visible_conjunction']=end_classes(before,scene,dep,0)[4,4]==1
    scene['part_id'][4,4]=999;scene['camera_depth'][4,4]=1.;tests['independent_registered_foreign_strictfront']=end_classes(before,scene,dep,0)[4,4]==2
    before['part_id'][4,4]=5;before['camera_depth'][4,4]=1.;tests['independent_self_stays_UNKNOWN']=end_classes(before,scene,dep,0)[4,4]==3
    before['ambiguity'][4,4]=True;tests['independent_ambiguity_priority']=end_classes(before,scene,dep,0)[4,4]==4
    need(all(tests.values()),'Invented independent checks');return tests
def prepare():
    scenes,rr,old,bases=metadata();tests=invented();write(H/'GENERATED_INVENTED_CHECKS.json',dict(completed=True,PASS=True,invented_only=True,tests={k:bool(v) for k,v in tests.items()},checks=len(tests),actual_source_pixel_target_header_hash_reads=0))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=binding(__file__),independent_math=binding(H/'independent_math.py'),scope=binding(SCOPE),producer_preparation=binding(A/'PREPARATION_COMPLETE.json'),fixed_first_bases=bases,prospective_independent_raw_and_end_rows=4096,prospective_owner_rows=N,all_metadata_rows_checked=N,checks=binding(H/'GENERATED_INVENTED_CHECKS.json'),actual_new_source_output_or_target_body_reads=0,newmodel_forward_fit=0))
    print(json.dumps(dict(PASS=True,prepared_only=True,code=binding(__file__),fixed_bases=bases)),flush=True)
def run(complete):
    start=time.monotonic();c=read(complete);need(c['completed'] and c['PASS'] and c['rows']==N and c['source_TRAIN_only'] and c['original_nominal4096_raw_valid_byteparity'] and c['original_aug_inputU']==1264,'Actual completed all65536 barrier')
    for k in ['scope','code','protocol','entry','inputs','targets','metadata','scene_metadata','row_input_metadata','measures']:check(c[k])
    scenes,planned,old,bases=metadata();actual=rows(c['metadata']['path']);need(len(actual)==N,'metadata rows65536');X=npz(c['inputs']);Y=npz(c['targets']);M=npz(c['measures'])
    need(set(X)=={'rgb96','valid96','row_id','original_global_index','native_slot','query_xy','L','original_query_available'} and set(Y)=={'support_target','owner_soft','owner_target','owner_ambiguous_mass','owner_invalid_mass'},'NoGTininput/exact8+5')
    need(set(M)=={'end_mass7','projected_region_out_of_frame'} and not c['SAM_arrays_present'],'Separate diagnostic mass and no SAMsubstitution')
    need(X['rgb96'].shape==(N,96,96,3) and X['rgb96'].dtype==np.uint8 and X['valid96'].shape==(N,96,96) and X['valid96'].dtype==bool,'raw input schema')
    for key,dt in [('original_global_index',np.int64),('native_slot',np.int8),('query_xy',np.float64),('L',np.float64),('original_query_available',bool)]:need(X[key].dtype==dt,'input dtype '+key)
    need(X['row_id'].dtype.kind=='U' and Y['owner_target'].dtype==np.int8 and Y['owner_ambiguous_mass'].dtype==Y['owner_invalid_mass'].dtype==np.float64 and M['projected_region_out_of_frame'].dtype==bool,'target/metadata dtype')
    need(Y['support_target'].dtype==np.int8 and Y['support_target'].shape==(N,2) and Y['owner_soft'].dtype==np.float64 and Y['owner_soft'].shape==(N,3) and M['end_mass7'].shape==(N,7) and M['end_mass7'].dtype==np.float64,'all target schemas')
    missing=0;known_targets=[Counter(),Counter()];maxsum=0.;hashrows=0
    for i,(p,r) in enumerate(zip(planned,actual)):
        need(all(r[k]==v for k,v in p.items()),'all compactmetadata retained')
        need(X['row_id'][i]==r['row_id'] and X['original_global_index'][i]==i and X['native_slot'][i]==r['native_slot'] and X['original_query_available'][i]==r['original_query_available'],'all NPZ IDs/native/availability')
        q=np.asarray([np.nan,np.nan] if r['query_xy'] is None else r['query_xy'],np.float64);L=np.float64(np.nan if r['L'] is None else r['L']);need(exact(X['query_xy'][i],q) and exact(np.asarray(X['L'][i]),np.asarray(L)),'all qL numeric bytes')
        need(r['target_recomputed'] is True and r['actual_rgb96_sha256']==ah(X['rgb96'][i]) and r['actual_valid96_sha256']==ah(X['valid96'][i]),'all serialized raw facthashes');hashrows+=1
        mass=M['end_mass7'][i];own=np.r_[Y['owner_soft'][i],Y['owner_ambiguous_mass'][i],Y['owner_invalid_mass'][i]];is_missing=r['query_xy'] is None or r['L'] is None or not np.isfinite(q).all() or not np.isfinite(L) or L<=0 or np.array_equal(q,[-1.,-1.])
        if is_missing:
            missing+=1;need(np.isnan(mass).all() and np.isnan(own).all() and Y['support_target'][i].tolist()==[-1,-1],'missing remainsNaN/UNKNOWN')
        else:
            need(np.isfinite(mass).all() and np.isfinite(own).all() and (mass>=0).all() and (own>=0).all(),'finite nonnegative rawclassmass');err=max(abs(float(mass.sum())-1),abs(float(own.sum())-1));maxsum=max(maxsum,err);need(err<=2e-13,'known/unknown/invalid mass conservation')
            need(target(mass,bool(r['original_query_available']),True,bool(M['projected_region_out_of_frame'][i]))==Y['support_target'][i].tolist(),'all two targets independently decoded from mass/U/OOF')
        for ch in range(2):known_targets[ch][int(Y['support_target'][i,ch])]+=1
    need((Y['owner_target']==-1).all() and (~X['original_query_available']).sum()==1264 and (Y['support_target'][~X['original_query_available']]==-1).all(),'all hardowner/sourceU preserved')
    geom=rows(V180/'source_end_supervision_preparation_a/TRAIN1024_FIXED_END_PATCH_GEOMETRY_METADATA.jsonl');gm={(g['base_index'],g['native_slot']):g for g in geom};source={};crops={};cached_crop={};rep_raw=rep_end=owner_rows=nominal=0;max_owner=max_end=0.
    for base in range(128):
        before=None;patches={k:project_patch(gm[base,k]) for k in range(8)} if base in bases else {}
        for vi,var in enumerate(VAR):
            s=scenes[base*4+vi];sb=s['source_RGB_binding'];path=check(sb);source[str(path)]=sb;keys=['part_id','ambiguity']+(['RGB_uint8','camera_depth','global_triangle_id','camera_K','camera_R','camera_t','native8_CF_xy','permCF_to_canonical'] if base in bases else [])
            with np.load(path,allow_pickle=False) as z:packet={k:z[k].copy() for k in keys}
            if vi==0:before=packet
            for k in range(8):
                oi=base*32+vi*8+k;original=old[oi];ids=list(range(oi*15,oi*15+15))+[61440+oi]
                cl=None
                if base in bases:
                    dep,good,oof=patches[k];cl=end_classes(before,packet,dep,gm[base,k]['primary_part_id'])
                    for key in ['K','R','t']:need(np.array_equal(packet['camera_'+key],np.asarray(gm[base,k]['camera'][key],np.float64)),'representative camera metadata')
                    need(np.array_equal(packet['native8_CF_xy'],np.asarray(gm[base,k]['native8_CF_xy'],np.float64)) and np.array_equal(packet['permCF_to_canonical'],np.asarray(gm[base,k]['permCF_to_canonical'],np.int32)),'representative declarednative metadata')
                cb=original['saved_crop_chunk'];cp=Path(cb['path']);crops[str(cp)]=cb
                if str(cp) not in cached_crop:
                    check(cb)
                    with np.load(cp,allow_pickle=False) as z:cached_crop[str(cp)]={key:z[key].copy() for key in ['rgb96','valid96','global_index','crop_constructed']}
                j0=oi*15;ci=original['saved_crop_row'];z=cached_crop[str(cp)];need(int(z['global_index'][ci])==oi and bool(z['crop_constructed'][ci])==actual[j0]['crop_constructed'],'ownnominal originalconstruction')
                need(exact(X['rgb96'][j0],z['rgb96'][ci]) and exact(X['valid96'][j0],z['valid96'][ci]),'all4096ownnominal raw/valid bytes');need(ah(X['rgb96'][j0])==original['saved_rgb96_sha256_inherited'] and ah(X['valid96'][j0])==original['saved_valid96_sha256_inherited'],'all4096nominal facts');nominal+=1
                for j in ids:
                    r=actual[j];own=owner4(packet,r['query_xy'],r['L']);sv=np.r_[Y['owner_soft'][j],Y['owner_ambiguous_mass'][j],Y['owner_invalid_mass'][j]];need(np.allclose(own,sv,atol=2e-15,rtol=0,equal_nan=True),'all scalar central4 sourceowner '+str(j));owner_rows+=1
                    if np.isfinite(own).all():max_owner=max(max_owner,float(np.max(np.abs(own-sv))))
                    if base in bases:
                        rgb,valid=raw96(packet['RGB_uint8'],r['query_xy'],r['L']);need(exact(rgb,X['rgb96'][j]) and exact(valid,X['valid96'][j]),'representative independent raw RGB/valid '+str(j));rep_raw+=1
                        if r['query_xy'] is not None and r['L'] is not None:
                            mass=mass96(cl,r['query_xy'],r['L']);diff=float(np.max(np.abs(mass-M['end_mass7'][j])));max_end=max(max_end,diff);need(diff<=1e-12,'independent endmass source ray '+str(j));need(target(mass,r['original_query_available'],good,oof)==Y['support_target'][j].tolist(),'independent endlabels '+str(j))
                        rep_end+=1
        if (base+1)%16==0:print(json.dumps(dict(bases=base+1,owner_rows=owner_rows,nominal_raw=nominal,representative_raw_end=rep_raw)),flush=True)
    need(owner_rows==N and nominal==4096 and rep_raw==rep_end==4096,'actual coverage exact')
    oldend=read(V180/'source_end_supervision_train4096_v1/COMPLETE.json');end=npz(oldend['outputs']['sidecar'],['visible_support_target','foreign_front_support_target','jointmass7','original_global_index','native_slot','query_xy','L','query_available']);idx=np.arange(4096)*15
    need(np.array_equal(end['original_global_index'],np.arange(4096)) and np.array_equal(end['native_slot'],X['native_slot'][idx]) and np.array_equal(end['query_xy'],X['query_xy'][idx],equal_nan=True) and np.array_equal(end['L'],X['L'][idx],equal_nan=True),'oldTRAINsupport identityjoin')
    need(np.array_equal(np.c_[end['visible_support_target'],end['foreign_front_support_target']],Y['support_target'][idx]),'all4096 ownnominal oldend2labels exact');need(np.array_equal(end['jointmass7'],M['end_mass7'][idx],equal_nan=True),'all4096 ownnominal oldend7mass exact')
    oldcache=read(V181/'source_raw96_cache_v1/COMPLETE.json');oldown=npz(oldcache['domains']['SYNTH_TRAIN4096']['targets']);nominal_owner={}
    for key in ['owner_soft','owner_ambiguous_mass','owner_invalid_mass']:
        aa=Y[key][idx];bb=oldown[key];need(np.allclose(aa,bb,atol=2e-13,rtol=0,equal_nan=True),'oldTRAINcentral4 owner '+key);mask=np.isfinite(aa);nominal_owner[key]=dict(byteequal=exact(aa,bb),max_abs_difference=float(np.max(np.abs(aa[mask]-bb[mask]))) if mask.any() else 0.)
    for b in list(source.values())+list(crops.values()):check(b)
    for k in ['inputs','targets','metadata','scene_metadata','row_input_metadata','measures']:check(c[k])
    result=dict(completed=True,PASS=True,code=binding(__file__),independent_math=binding(H/'independent_math.py'),actual_complete=binding(complete),coverage=dict(all_metadata_qL_state_origin_flags_rows=N,all_raw_serialized_facthashes=N,all_target_unknown_mass_conservation_rows=N,all_original_source_buffer_SHA_before_after=len(source),all_original_crop_chunk_SHA_before_after=len(crops),all_ownnominal_raw_valid_fullbytes=nominal,all_ownnominal_old_end_target_and_mass=4096,all_ownnominal_oldcentral4_owner=4096,all_independent_scalar_owner_rows=owner_rows,independent_source_RGB_and_end_rays_rows=rep_raw,representative_SELF_states_rows=3840,representative_BEFORE_anchor_rows=256,fixed_first_bases=bases,inputU=1264,missing_mass_rows=missing),numerical=dict(independent_owner5_max_abs_difference=max_owner,independent_representative_end7_max_abs_difference=max_end,all_mass_sum_max_abs_error=maxsum,ownnominal_oldowner_comparison=nominal_owner),support_target_counts={'visible':dict(known_targets[0]),'foreign':dict(known_targets[1])},all_source_and_output_SHA_unchanged=True,new_model_SAM_forward_fit_or_selection=0,VAL_private_held_or_V112_reads=0,V182_diagnostic_payload_reads=0,limitations=['Full65536 end targets checked via savedmass/U/OOF rule, all4096ownnominal compared originalTRAINtarget; source ray/classification independently recomputed only fixedoutcome-blind8base4096augmentationrows.','Full65536 central4 owner independently calculated from source part/ambiguity4taps; this literalpointowner is distinct from intended-end/external support.','New crop-only augmentation and BEFORE-anchor contrast, not original productionreplay, physicalnativegold or real transfer evidence.'],selected_model=None,fully_solved=False,seconds=time.monotonic()-start)
    write(H/'REVIEW.json',result);print(json.dumps(dict(PASS=True,review=binding(H/'REVIEW.json'))),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--complete',type=Path);a=ap.parse_args();prepare() if a.prepare else run(a.complete)
