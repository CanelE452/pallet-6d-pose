"""V228: immutable associated-r teacher replay + separate literal-q observations.

실제 payload는 exact true root entry 이후에만 읽는다. 새 ray/모델/학습/RGB 없음.
"""
import argparse
import ast
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import traceback

import cv2
import numpy as np

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R = P / 'publication_root_preparation'
V = P / 'query_visibility_ycb_real_surface_source_20261023_v109'
PROTOCOL = R / 'V228_ROOT_QUERY_OBSERVATION/PROTOCOL_BEFORE_TRAIN_GT_OR_PIXEL_ACCESS.json'
PROTOCOL_SHA = '9f3d265782b1b3fc295892e5137eb58d0f7526c08b585cfac941688ed8bfa19f'
SEAL = V / 'corpus_build_v1/CORPUS_SEAL.json'
SEAL_SHA = '934a9a3aa88854cf0632fc7b3f4c89b5f0630cc435cfc1f64e1c4eea220903db'
PRODUCER = V / 'producer/build_surface_source.py'
PRODUCER_SHA = 'a431eaa92c06b5a9a7e2d14ac7a8d6fde18c685edf988c7d099ce02411b65f38'
RAW_ROOT = P / 'query_visibility_public_real_20261022_v63/ycb/data/train_real'
OUT = R / 'V228_ROOT_QUERY_OBSERVATION/query_observations_v1'
COUNT_KEYS = ('in_bounds','positive_depth','unique_own','unique_other_only','overlap','unassigned','own_full')
Q_STATES = ('QUERY_CONTINUOUS_OOB','QUERY_NEAREST_OOB','UNIQUE_OWN','UNIQUE_FOREIGN','OVERLAP','UNASSIGNED')


def safe(path):
    p = Path(os.path.abspath(os.fspath(path)))
    text = str(p).casefold()
    for forbidden in ('v112','current223','publicheld','oldval','v177','v178','source_holdout','v182','common'):
        assert forbidden not in text, f'Forbidden lexical path: {forbidden}'
    for q in reversed((p,*p.parents)):
        assert not q.is_symlink(), f'Symlink: {q}'
    return p


def read(path):
    p = safe(path)
    fd = os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:return f.read()


def digest(b):return hashlib.sha256(b).hexdigest()


def identity(b):return str(Path(os.path.abspath(b['path']))),b['sha256']


def binding(path):
    b=read(path)
    return {'path':str(safe(path)),'sha256':digest(b),'size_bytes':len(b)}


def checked(b):
    raw=read(b['path']);assert digest(raw)==b['sha256'],b['path']
    if 'size_bytes' in b:assert len(raw)==b['size_bytes'],b['path']
    return raw


def write(path,obj):
    with safe(path).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def clean_json(x):
    if isinstance(x,np.ndarray):return clean_json(x.tolist())
    if isinstance(x,np.generic):return clean_json(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    if isinstance(x,dict):return {k:clean_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean_json(v) for v in x]
    return x


def seal_metadata():
    raw=read(SEAL);assert digest(raw)==SEAL_SHA
    s=json.loads(raw);assert s['completed'] and s['all8000_preserved']
    assert identity(s['code'])==(str(PRODUCER),PRODUCER_SHA)
    byname={Path(b['path']).name:b for b in s['outputs']}
    return s,byname


def entry_template():
    s,o=seal_metadata()
    return {'approved_query_observation_build':False,'source_TRAIN_only':True,
        'protocol':{'path':str(PROTOCOL),'sha256':PROTOCOL_SHA},'code':binding(__file__),
        'corpus_seal':{'path':str(SEAL),'sha256':SEAL_SHA},'producer':s['code'],
        'inputs1000':s['inputs'],'reference1000':s['reference_GT_only'],
        'original_teacher_npz':o['RAW_SURFACE_TEACHER8000_GT_ONLY.npz'],
        'raw_file_bindings':o['SOURCE_DEPTH_MASK_BINDINGS.json'],
        'independent_scope_QA':{'path':None,'sha256':None},'payload_whitelist':None,
        'output_directory':str(OUT)}


def sample_window(depth,owners,own_full,own_index,xy):
    """floor(x+.5); dy outer/dx inner. No reproject, snap, resize or fallback."""
    H,W=depth.shape
    q=np.asarray(xy,dtype=np.float64);assert q.shape==(2,) and np.isfinite(q).all()
    centre=np.floor(q+.5).astype(np.int64)
    inside=np.zeros(9,bool);raw=np.zeros(9,np.uint16);full=np.zeros(9,bool);ownvis=np.zeros(9,bool)
    counts=np.zeros(9,np.uint8);bits=np.zeros(9,np.uint64)
    for j,(dy,dx) in enumerate((dy,dx) for dy in (-1,0,1) for dx in (-1,0,1)):
        x,y=(centre+[dx,dy]).tolist();ok=0<=x<W and 0<=y<H;inside[j]=ok
        if ok:
            raw[j]=depth[y,x];full[j]=own_full[y,x];ownvis[j]=owners[own_index,y,x]
            ids=np.flatnonzero(owners[:,y,x]);counts[j]=len(ids);bits[j]=sum(1<<int(i) for i in ids)
    positive=inside&(raw>0);unique_own=inside&(counts==1)&ownvis
    unique_foreign=inside&(counts==1)&~ownvis
    evidence=np.asarray([inside.sum(),positive.sum(),unique_own.sum(),unique_foreign.sum(),((counts>1)&inside).sum(),((counts==0)&inside).sum(),full.sum()],np.uint8)
    continuous=bool(0<=q[0]<W and 0<=q[1]<H)
    if not continuous:state='QUERY_CONTINUOUS_OOB'
    elif not inside[4]:state='QUERY_NEAREST_OOB'
    elif counts[4]>1:state='OVERLAP'
    elif counts[4]==0:state='UNASSIGNED'
    elif ownvis[4]:state='UNIQUE_OWN'
    else:state='UNIQUE_FOREIGN'
    same_foreign=bool(inside.all() and (counts==1).all() and not ownvis.any() and len(set(map(int,bits)))==1)
    return {'continuous_valid':continuous,'nearest_xy':centre,'nearest_valid':bool(inside[4]),'state':state,
        'inbounds9':inside,'rawdepth9':raw,'ownfull9':full,'ownvis9':ownvis,'ownercount9':counts,'ownerbits9':bits,
        'counts':evidence,'unique_own_and_positive_count':int((unique_own&positive).sum()),'same_foreign9':same_foreign}


def old_classifier():
    raw=read(PRODUCER);assert digest(raw)==PRODUCER_SHA
    tree=ast.parse(raw);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='classify']
    assert len(nodes)==1
    ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(PRODUCER),'exec'),ns)
    return ns['classify']


def self_test():
    depth=np.arange(25,dtype=np.uint16).reshape(5,5)+1
    own=np.ones((5,5),bool);owners=np.stack([own,np.zeros_like(own)])
    w=sample_window(depth,owners,own,0,[2,2])
    assert w['rawdepth9'].tolist()==[7,8,9,12,13,14,17,18,19] and w['state']=='UNIQUE_OWN'
    assert sample_window(depth,owners,own,0,[1.5,1.5])['nearest_xy'].tolist()==[2,2]
    assert sample_window(depth,owners,own,0,[-.01,2])['state']=='QUERY_CONTINUOUS_OOB'
    assert sample_window(depth,owners,own,0,[4.9,2])['state']=='QUERY_NEAREST_OOB'
    owners[0]=False;owners[1]=True
    f=sample_window(depth,owners,own,0,[2,2]);assert f['state']=='UNIQUE_FOREIGN' and f['same_foreign9'] and (f['ownerbits9']==2).all()
    owners[0,2,2]=True
    assert sample_window(depth,owners,own,0,[2,2])['state']=='OVERLAP'
    owners[:]=False
    assert sample_window(depth,owners,own,0,[2,2])['state']=='UNASSIGNED'
    owners[0]=True;depth[:]=0
    q=sample_window(depth,owners,own,0,[2,2]);assert q['state']=='UNIQUE_OWN' and q['counts'][2]==9 and q['counts'][1]==0 and q['unique_own_and_positive_count']==0
    print(json.dumps({'completed':True,'PASS':True,'synthetic_arrays_only':True,'payload_reads':0,'optimizer_steps':0,'new_rays_or_models':0,'checks':['window_order','floor_half_up','continuous_vs_nearest_OOB','same_foreign_bits','overlap_and_unassigned','mask_only_vs_positive_depth_intersection']}))


def run(entry_path):
    eraw=read(entry_path);e=json.loads(eraw)
    assert e['approved_query_observation_build'] is True and e['source_TRAIN_only'] is True
    assert identity(e['protocol'])==(str(PROTOCOL),PROTOCOL_SHA)
    proto=json.loads(checked(e['protocol']));assert proto['phase']=='V228' and proto['new_training_admission'] is False
    assert identity(e['code'])==(str(safe(__file__)),binding(__file__)['sha256']);checked(e['code'])
    assert identity(e['corpus_seal'])==(str(SEAL),SEAL_SHA);checked(e['corpus_seal'])
    s,o=seal_metadata()
    expected={'inputs1000':s['inputs'],'reference1000':s['reference_GT_only'],'original_teacher_npz':o['RAW_SURFACE_TEACHER8000_GT_ONLY.npz'],'raw_file_bindings':o['SOURCE_DEPTH_MASK_BINDINGS.json'],'producer':s['code']}
    assert all(identity(e[k])==identity(b) for k,b in expected.items())
    qa=json.loads(checked(e['independent_scope_QA']));assert qa['completed'] is True and qa['PASS'] is True
    checked(e['producer'])
    assert safe(e['output_directory'])==OUT and not OUT.exists(),'Fresh fixed output required'
    OUT.mkdir()
    write(OUT/'STARTED.json',{'entry':{'path':str(safe(entry_path)),'sha256':digest(eraw)},'code':e['code'],'protocol':e['protocol'],'new_training_admission':False,'automatic_retry':False})
    try:
        source=json.loads(checked(e['raw_file_bindings']))
        files=source['files'];filemap={identity(b)[0]:b for b in files};assert len(filemap)==len(files)
        white=e['payload_whitelist'];assert isinstance(white,list)
        assert len({identity(b) for b in white})==len(white)==len(files)
        assert {identity(b) for b in white}=={identity(b) for b in files},'Exact root/source raw-whitelist identity'
        # All opaque bytes are verified before any raw depth/mask decode. JSON values are not parsed here.
        for b in white:
            p=safe(b['path']);assert p.is_relative_to(RAW_ROOT) and '/rgb/' not in str(p).casefold()
            checked(b)
        inputs=[json.loads(line) for line in checked(e['inputs1000']).splitlines() if line]
        refs=[json.loads(line) for line in checked(e['reference1000']).splitlines() if line]
        assert len(inputs)==len(refs)==1000
        with np.load(io.BytesIO(checked(e['original_teacher_npz'])),allow_pickle=False) as z:
            old={k:z[k] for k in z.files}
        needed={'raw_surface_teacher_state','reason','geometry_state','geometry_stable','centre_integer_ray_first_cameraZ_mm','neighbour_depth_raw_uint16','neighbour_inbounds','neighbour_own_full','neighbour_own_visible','neighbour_visible_owner_count','neighbour_owner_bits_uint64','evidence_counts','reference_xy_GT_only','query_xy_input_copy','roi_side_input_copy','query_reference_error_px','query_continuous_valid','query_nearest_valid','reference_continuous_valid','reference_nearest_valid','depth_scale_to_mm','frame_index','instance_row_index','logical_aux_slot'}
        assert needed.issubset(old) and old['raw_surface_teacher_state'].shape==(1000,8)
        assert old['neighbour_depth_raw_uint16'].shape==(1000,8,9) and old['neighbour_depth_raw_uint16'].dtype==np.uint16
        assert old['neighbour_visible_owner_count'].dtype==np.uint8 and old['neighbour_owner_bits_uint64'].dtype==np.uint64
        assert np.array_equal(old['instance_row_index'],np.arange(1000))
        assert np.array_equal(old['logical_aux_slot'],np.broadcast_to(np.arange(8),(1000,8)))
        frames=source['frames'];assert len(frames)==700 and source['all_visible_owner_sets_from_complete_scene_GT'] is True
        byframe={(f['scene_id'],f['frame_id']):f for f in frames};assert len(byframe)==700
        assert {f['frame_index'] for f in frames}==set(range(700))
        classify=old_classifier()
        def decode(path):
            p=str(safe(path));assert p in filemap
            a=cv2.imdecode(np.frombuffer(checked(filemap[p]),dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            assert a is not None,p
            return a
        qarrays={'state':np.full((1000,8),'',dtype='U24'),'inbounds9':np.zeros((1000,8,9),bool),'rawdepth9':np.zeros((1000,8,9),np.uint16),'depth_mm9':np.zeros((1000,8,9),np.float64),'ownfull9':np.zeros((1000,8,9),bool),'ownvis9':np.zeros((1000,8,9),bool),'ownercount9':np.zeros((1000,8,9),np.uint8),'ownerbits9':np.zeros((1000,8,9),np.uint64),'counts':np.zeros((1000,8,7),np.uint8),'unique_own_and_positive_count':np.zeros((1000,8),np.uint8),'same_foreign9':np.zeros((1000,8),bool),'continuous_valid':np.zeros((1000,8),bool),'nearest_valid':np.zeros((1000,8),bool),'nearest_xy':np.zeros((1000,8,2),np.int64)}
        cached_key=None;depth=None;owners=None;frame=None;table=collections.Counter();jitter_tables=collections.defaultdict(collections.Counter);label_counts=collections.Counter();used_frames=set();raw_decodes=0
        with safe(OUT/'QUERY_OBSERVATIONS8000.jsonl').open('x',encoding='utf-8') as stream:
            for i,(inp,g) in enumerate(zip(inputs,refs)):
                assert inp['input_id']==g['input_id']==f'Y109_SOURCE_{i:06d}' and inp['instance_row_index']==g['instance_row_index']==i
                inst=g['source_instance'];assert inp['split']=='TRAIN' and inst['split']=='train' and inp['native_mapping']=='UNKNOWN'
                key=(inst['scene_id'],inst['frame_id']);frame=byframe[key]
                assert old['frame_index'][i]==frame['frame_index'] and identity(frame['depth'])==identity(filemap[str(safe(inst['depth']))])
                assert inst['instance_id']>=0 and frame['instance_object_ids'][inst['instance_id']]==inst['object_id']
                assert frame['complete_scene_GT_owner_count']==len(frame['visible_masks'])==len(frame['instance_object_ids'])<=64
                shape=tuple(inp['raw_hw']);assert len(shape)==2 and min(shape)>0
                scale=float(inst['depth_scale']);assert np.isfinite(scale) and scale>0 and old['depth_scale_to_mm'][i]==scale
                q=np.asarray(inp['initial_points'],np.float64);r=np.asarray(g['reference_xy'],np.float64)
                assert q.shape==r.shape==(8,2) and np.isfinite(q).all() and np.isfinite(r).all()
                assert np.array_equal(q,old['query_xy_input_copy'][i]) and np.array_equal(r,old['reference_xy_GT_only'][i])
                assert old['roi_side_input_copy'][i]==inp['roi_side']
                distances=np.linalg.norm(q-r,axis=1);assert (distances<=4).all() and np.array_equal(distances,old['query_reference_error_px'][i])
                if key!=cached_key:
                    depth=decode(inst['depth']);assert depth.dtype==np.uint16 and depth.shape==shape
                    owner_list=[]
                    for mask_binding in frame['visible_masks']:
                        assert identity(mask_binding)==identity(filemap[str(safe(mask_binding['path']))])
                        a=decode(mask_binding['path']);assert a.shape==shape and set(np.unique(a)).issubset({0,255});owner_list.append(a>0)
                    owners=np.stack(owner_list);cached_key=key;raw_decodes+=1+len(owner_list)
                assert depth.shape==shape and owners.shape[1:]==shape
                full=decode(inst['mask_full']);raw_decodes+=1
                assert full.shape==shape and set(np.unique(full)).issubset({0,255});full=full>0
                own=int(inst['instance_id']);used_frames.add(frame['frame_index'])
                for k in range(8):
                    a=sample_window(depth,owners,full,own,r[k]);b=sample_window(depth,owners,full,own,q[k])
                    pairs={'rawdepth9':'neighbour_depth_raw_uint16','inbounds9':'neighbour_inbounds','ownfull9':'neighbour_own_full','ownvis9':'neighbour_own_visible','ownercount9':'neighbour_visible_owner_count','ownerbits9':'neighbour_owner_bits_uint64','counts':'evidence_counts'}
                    assert all(np.array_equal(a[new],old[saved][i,k]) for new,saved in pairs.items()),('r evidence mismatch',i,k)
                    assert a['continuous_valid']==bool(old['reference_continuous_valid'][i,k]) and a['nearest_valid']==bool(old['reference_nearest_valid'][i,k])
                    assert b['continuous_valid']==bool(old['query_continuous_valid'][i,k]) and b['nearest_valid']==bool(old['query_nearest_valid'][i,k])
                    state,reason=classify(str(old['geometry_state'][i,k]),b['continuous_valid'],a['continuous_valid'],a['inbounds9'],a['rawdepth9'],a['ownercount9'],a['ownvis9'],a['ownerbits9'],a['ownfull9'],own,scale,float(g['reference_camera_xyz_mm'][k][2]),float(old['centre_integer_ray_first_cameraZ_mm'][i,k,0]))
                    assert state==str(old['raw_surface_teacher_state'][i,k]) and reason==str(old['reason'][i,k]),('old state/reason mismatch',i,k,state,reason)
                    b['depth_mm9']=b['rawdepth9'].astype(np.float64)*scale
                    for name,array in qarrays.items():array[i,k]=b[name]
                    original={name:clean_json(value[i,k]) for name,value in old.items() if value.shape[:2]==(1000,8)}
                    original.update({'depth_scale_to_mm':scale,'frame_index':int(old['frame_index'][i]),'instance_row_index':i})
                    row={'slot_index':i*8+k,'input_id':inp['input_id'],'instance_row_index':i,'frame_index':int(frame['frame_index']),'scene_id':inst['scene_id'],'source_frame_id':inst['frame_id'],'source_instance_id':own,'object_id':inst['object_id'],'logical_aux_slot':k,'aux_surface_id':g['aux_surface_ids'][k],'native_mapping':'UNKNOWN','q':q[k].tolist(),'r_GT_teacher_only':r[k].tolist(),'L_input':inp['roi_side'],'constructed_jitter_distance_px':float(distances[k]),'original_teacher_immutable':original,'q_observation_not_semantic_target':clean_json(b),'q_centre_observation':{'ownership':b['state'],'raw_depth':int(b['rawdepth9'][4]),'depth_mm':float(b['depth_mm9'][4]),'positive_depth':bool(b['inbounds9'][4] and b['rawdepth9'][4]>0),'own_full':bool(b['ownfull9'][4]),'own_visible':bool(b['ownvis9'][4]),'owner_count':int(b['ownercount9'][4]),'owner_bits':int(b['ownerbits9'][4])},'new_native_visibility_target':None,'training_admission':False,'GT_teacher_only_no_deployment_input_change':True}
                    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    table[(state,b['state'])]+=1;label_counts[state]+=1
                    bin_index=int(np.searchsorted([1,2,3,4],distances[k],side='left'))
                    assert bin_index<4
                    jitter_tables[f'bin_{bin_index}'][(state,b['state'])]+=1
                    if distances[k]<=3:jitter_tables['distance_le3'][(state,b['state'])]+=1
                    if (i*8+k+1)%2000==0:print(json.dumps({'rows':i*8+k+1,'total':8000}),flush=True)
        assert len(used_frames)==700 and sum(table.values())==8000
        payload={f'original_{k}':v for k,v in old.items()}
        payload.update({f'q_observation_{k}':v for k,v in qarrays.items()})
        fd=os.open(safe(OUT/'QUERY_OBSERVATIONS8000.npz'),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
        with os.fdopen(fd,'wb') as f:np.savez_compressed(f,**payload)
        def matrix(counter):return {state:{qstate:int(counter[(state,qstate)]) for qstate in Q_STATES} for state in ('D','E','S','U')}
        rows=[{'original_associated_r_state':a,'literal_q_observation':b,'rows':int(table[(a,b)])} for a in ('D','E','S','U') for b in Q_STATES]
        with safe(OUT/'OLD_STATE_BY_Q_OWNER.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        summary={'completed':True,'PASS':True,'instances':1000,'slots':8000,'frames':700,'old_reference_raw_evidence_exact_all8000':True,'old_states_and_reasons_exact_all8000':True,'original_state_counts':dict(label_counts),'old_associated_r_state_x_literal_q_observation':matrix(table),'jitter_bins_definition':['[0,1]','(1,2]','(2,3]','(3,4]'],'jitter_bin_tables':{k:{'rows':int(sum(c.values())),'table':matrix(c)} for k,c in jitter_tables.items()},'q_nearest_OOB':int((~qarrays['nearest_valid']).sum()),'q_continuous_OOB':int((~qarrays['continuous_valid']).sum()),'q_mask_only_unique_own_count':int((qarrays['state']=='UNIQUE_OWN').sum()),'q_unique_own_and_positive_window_intersection_counts_separate':True,'opaque_raw_whitelist_files':len(white),'all_raw_SHA_verified_before_decode':True,'depth_and_mask_decode_calls':raw_decodes,'raw_scene_GT_or_camera_JSON_value_parse':0,'RGB_decode':0,'new_ray_CAD_model_forward_fit':0,'native_mapping':'UNKNOWN','new_training_admission':False,'observation_table_is_not_semantic_classifier_confusion':True,'limits':['Original labels remain associated-r visibility, not literal-q owner labels.','Constructed <=4px jitter is not actual initializer error.','Mask/depth absence and SELF are not converted to E.','q observations may differ while original associated-r target remains valid; no corruption or mislabel inference.','GT/reference/mask/depth remain teacher-side, not deployment channels.']}
        write(OUT/'SUMMARY.json',summary)
        outputs={n:binding(OUT/n) for n in ('QUERY_OBSERVATIONS8000.jsonl','QUERY_OBSERVATIONS8000.npz','OLD_STATE_BY_Q_OWNER.csv','SUMMARY.json')}
        write(OUT/'COMPLETE.json',{'completed':True,'PASS':True,'entry':{'path':str(safe(entry_path)),'sha256':digest(eraw)},'code':e['code'],'protocol':e['protocol'],'corpus_seal':e['corpus_seal'],'producer':e['producer'],'independent_scope_QA':e['independent_scope_QA'],'inputs':{k:e[k] for k in expected if k!='producer'},'outputs':outputs,'all8000_preserved':True,'old_labels_and_input_q_L_unchanged':True,'new_training_admission':False,'new_model_forward_optimizer_or_PnP':0})
        print(json.dumps({'completed':True,'PASS':True,'COMPLETE':binding(OUT/'COMPLETE.json')}),flush=True)
    except BaseException:
        write(OUT/'FAILED_EXECUTION.json',{'completed':False,'PASS':False,'traceback':traceback.format_exc(),'automatic_retry':False})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--entry');g.add_argument('--entry-template',action='store_true');g.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test()
    elif args.entry_template:print(json.dumps(entry_template(),ensure_ascii=False,indent=2))
    else:run(args.entry)
