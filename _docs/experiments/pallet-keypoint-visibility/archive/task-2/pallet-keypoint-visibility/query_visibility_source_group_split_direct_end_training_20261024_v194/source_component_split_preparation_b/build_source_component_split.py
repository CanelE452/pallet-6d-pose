"""Root-entry metadata-only source components. No array/model library or asset I/O."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,importlib.util,json,math,os,stat,traceback

H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_PROSPECTIVE_SOURCE_COMPONENT_SPLIT_BEFORE_METADATA_CALCULATIONS.json'
SCOPE_SHA='8bb43e745b66cd49b5189f7be87258267d3d18eaf37742633385b5b4f97df72c'
PROTOCOL=H/'PROTOCOL_BEFORE_SOURCE_COMPONENT_METADATA_BODIES.json'
CHECKS=H/'GENERATED_INVENTED_COMPONENT_AND_MASK_CHECKS.json'
OUT=V/'source_component_split_v1'
SALT='pallet_visibility_source_groups_v194_20261024'
DOMAINS=('SYNTH_AUG65536','PUBLIC_TRAIN1480','CF3816','DIRECT_CONTROL')
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
VAR_SYN=('BEFORE','COVER','SHAM');VAR_REAL=('BEFORE','COVER','SHAM')

def need(x,m):
    if not bool(x):raise AssertionError(m)

def safe(p):
    p=Path(p);need(p.is_absolute() and '..' not in p.parts and all('v112'not in x.casefold()for x in p.parts),'Lexical safe path before access')
    need(P in p.parents,'Exact source project path');return p

def opened(p):
    p=safe(p);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for x in p.parts[1:-1]:n=os.open(x,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        last=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(last).st_mode),'Ordinary text file');return os.fdopen(last,'rb')

def body(p):
    with opened(p)as f:return f.read()
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(body(p)).hexdigest())
def read(p):return json.loads(body(p))
def checked(b):need(bind(b['path'])==b,'Bound metadata '+str(b['path']));return b['path']
def lines(b):return [json.loads(x)for x in body(checked(b)).splitlines()if x.strip()]
def write(p,d):
    safe(p)
    with Path(p).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def jlines(p,rr):
    safe(p)
    with Path(p).open('x')as f:
        for r in rr:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')

def node(*parts):return json.dumps(list(parts),ensure_ascii=False,separators=(',',':'),allow_nan=False)
def role(component_id):return 'HOLDOUT' if int(hashlib.sha256((SALT+'\n'+component_id).encode()).hexdigest()[:16],16)%100<20 else 'TRAIN'

class Graph:
    def __init__(self):self.parent={}
    def root(self,x):
        self.parent.setdefault(x,x)
        while self.parent[x]!=x:self.parent[x]=self.parent[self.parent[x]];x=self.parent[x]
        return x
    def add(self,nodes):
        nodes=sorted(set(nodes));need(bool(nodes),'At least one certified origin')
        roots=[self.root(x)for x in nodes];root=min(roots)
        for r in roots:self.parent[r]=root
    def close(self):
        groups=defaultdict(list)
        for n in sorted(self.parent):groups[self.root(n)].append(n)
        ids={};records=[]
        for ns in groups.values():
            ns=sorted(ns);cid=hashlib.sha256('\n'.join(ns).encode()).hexdigest();record=dict(component_id=cid,role=role(cid),origin_nodes=ns,origin_node_count=len(ns));records.append(record)
            for n in ns:ids[n]=cid
        return ids,sorted(records,key=lambda r:r['component_id'])

def fingerprint_nodes(r,rgb,valid,constructed):
    if r['original_query_available'] is True and constructed is True and rgb is not None and valid is not None:
        need(all(isinstance(h,str)and len(h)==64 for h in (rgb,valid)),'Inherited exact RGB/valid fingerprint')
        return [node('AVAILABLE_CONSTRUCTED_RGB96_VALID96',rgb,valid)]
    return []

def mr_nodes(source_id,query_id,native,q,L):
    return [node('MR_FRAME','MR6D',source_id),node('MR_ORIGINAL_QUERY_ID','MR6D',query_id),node('MR_ORIGINAL_QUERY_F64','MR6D',source_id,native,q,L)]

def donor_projection(assignment):
    d=assignment['donor'];return dict(donor_id=assignment['donor_id'],scene_id=d['scene_id'],frame_id=d['frame_id'],instance_id=d['instance_id'],object_id=d['object_id'],source_RGB_binding=d['image'])

def donor_nodes(d):
    need(isinstance(d['source_RGB_binding']['sha256'],str)and len(d['source_RGB_binding']['sha256'])==64,'Inherited donor sourceRGB identity')
    return [node('YCB_DONOR_CUTOUT_ID',d['donor_id']),node('YCB_SOURCE_FRAME','TRAIN_REAL',d['scene_id'],d['frame_id']),node('YCB_SOURCE_RGB_FILE_SHA256',d['source_RGB_binding']['sha256'])]

def stamp(row,domain,index,nodes,graph,**extra):
    need(row['native_slot']in range(8)and type(row['original_query_available'])is bool,'Original native/U')
    graph.add(nodes)
    return dict(domain=domain,cache_index=index,**{k:row[k]for k in META},origin_nodes=sorted(set(nodes)),**extra)

def build(data,known,expected_counts=(65536,512,1480,3816,2792,1024,66,159)):
    aug,scenes,pub,cf,old_direct,syn,real,assign=(data[k]for k in ('aug_rows','synth_scene_metadata','public_rows','cf_rows','direct_rows','synthetic_attempts','real193_attempts','assignments159'))
    need([len(x)for x in (aug,scenes,pub,cf,old_direct,syn,real,assign)]==list(expected_counts),'Frozen complete source populations')
    need(all(r['split']=='TRAIN'for r in aug+scenes)and all(r['split']=='SOURCE_DEVELOPMENT'for r in pub+cf),'TRAIN role only')
    donors={r['pair_id']:donor_projection(r)for r in assign};need(len(donors)==expected_counts[7],'Unique fixed CF pairs')
    donor_ids={}
    for d in donors.values():
        need(d['donor_id']not in donor_ids or donor_ids[d['donor_id']]==d,'Repeated donor identity exact source projection');donor_ids[d['donor_id']]=d
    assignment_anchor={r['pair_id']:r['anchor_query_id']for r in assign}
    pubid={r['row_id']:r for r in pub};pubframe=defaultdict(dict)
    for r in pub:pubframe[r['original']['source_id']][r['native_slot']]=r
    need(len(pubid)==expected_counts[2] and all(set(v)==set(range(8))for v in pubframe.values()),'Every original PUBLIC frame/native8')
    graph=Graph();allrows={d:[]for d in DOMAINS};status=[];base_factors={r['base_index']:r['factors']for r in scenes};base_by_scene={r['scene_index']:r['base_index']for r in scenes}
    for i,r in enumerate(aug):
        need(base_by_scene[r['scene_index']]==r['base_index']and base_by_scene[r['anchor_scene_index']]==r['base_index'],'Source and anchor scene remain original base')
        ns=[node('SYNTH_BASE','V129',r['base_index']),node('SYNTH_ORIGINAL_QUERY_ID','V129',r['source_image_original_query_id']),node('SYNTH_ORIGINAL_QUERY_ID','V129',r['anchor_original_query_id'])]
        ns+=fingerprint_nodes(r,r['actual_rgb96_sha256'],r['actual_valid96_sha256'],r['crop_constructed'])
        allrows[DOMAINS[0]].append(stamp(r,DOMAINS[0],i,ns,graph,source_image_original_global_index=r['source_image_original_global_index'],anchor_original_global_index=r['anchor_original_global_index'],base_index=r['base_index'],scene_index=r['scene_index'],family=r['family'],factors=base_factors[r['base_index']]))
    for i,r in enumerate(pub):
        o=r['original'];ns=mr_nodes(o['source_id'],r['row_id'],r['native_slot'],r['query_xy'],r['L'])+fingerprint_nodes(r,r['actual_raw_RGB96_sha256'],r['actual_raw_valid96_sha256'],o['feature_constructed'])
        allrows[DOMAINS[1]].append(stamp(r,DOMAINS[1],i,ns,graph,source_id=o['source_id'],original_frame_index=o['frame_index'],owner_known_original=r['original_global_index']in known['public']))
    for i,r in enumerate(cf):
        o=r['original'];need(o['pair_id']in donors and o['original_query_id']in pubid,'Every CF exact donor/source assignment');d=donors[o['pair_id']];original=pubid[o['original_query_id']]
        need(all(r[k]==original[k]for k in ('original_global_index','native_slot','query_xy','L','original_query_available'))and o['source_id']==original['original']['source_id']and o['original_frame_index']==original['original']['frame_index'],'CF original PUBLIC q/L/native/U/frame exact')
        if r['native_slot']==o['selected_anchor_native_slot']:need(o['original_query_id']==assignment_anchor[o['pair_id']],'Exact selected anchor query ID')
        ns=mr_nodes(o['source_id'],o['original_query_id'],r['native_slot'],r['query_xy'],r['L'])+donor_nodes(d)+fingerprint_nodes(r,r['actual_raw_RGB96_sha256'],r['actual_raw_valid96_sha256'],o['feature_constructed'])
        allrows[DOMAINS[2]].append(stamp(r,DOMAINS[2],i,ns,graph,source_id=o['source_id'],original_frame_index=o['original_frame_index'],pair_id=o['pair_id'],variant=o['variant'],donor=d,owner_known_original=r['domain_index']in known['CF']))
    synmap={r['attempt_id']:r for r in syn};need(len(synmap)==expected_counts[5] and {(r['base_index'],r['native_slot'])for r in syn}=={(b,k)for b in range(expected_counts[5]//8)for k in range(8)},'All synthetic base/native attempts')
    oldsyn=defaultdict(list)
    for r in old_direct:
        if r['domain']=='SYNTH_FINITE_END':oldsyn[(r['attempt_id'],r['variant'])].append(r)
    for r in syn:
        aid=r['attempt_id'];supported=r['status']=='CONTROLLED_DIRECT_END_WEAK_PAIR_SUPPORTED';ns=[node('SYNTH_BASE','V129',r['base_index'])]
        for vi,var in enumerate(VAR_SYN):
            physical=r.get('outputs')if supported else None;graph.add(ns)
            group='SYNTH__'+aid+'__'+var
            status.append(dict(domain='SYNTH_FINITE_END',attempt_id=aid,variant=var,group_id=group,source_status=r['status'],source_packet=physical,physical=physical is not None,selected_native_slot=r['native_slot'],stored_supported=supported,origin_nodes=ns,cache_row_indices=None))
            if physical is None:continue
            mm=sorted(oldsyn[(aid,var)],key=lambda z:z['native_slot']);need([m['native_slot']for m in mm]==list(range(8))and all(m['source_packet']==physical for m in mm),'Immutable synthetic direct all8 packet metadata')
            for m in mm:
                nn=ns+[node('SYNTH_ORIGINAL_QUERY_ID','V129',m['original_occurrence_id'])]+fingerprint_nodes(m,m['raw_RGB96_sha256'],m['raw_valid96_sha256'],m['original_input_flags']['crop_constructed'])
                idx=len(allrows['DIRECT_CONTROL']);allrows['DIRECT_CONTROL'].append(stamp(m,'DIRECT_CONTROL',idx,nn,graph,control_domain='SYNTH_FINITE_END',control_join_key=['SYNTH_FINITE_END',aid,var,m['native_slot']],group_id=group,attempt_id=aid,variant=var,source_packet=physical,layout='SYNTH_TRIPLET',variant_index=vi,selected_native_slot=r['native_slot'],stored_weak_direct_target=(1 if var=='COVER'else 0)if m['native_slot']==r['native_slot']else -1,authority='CONSTRUCTIVE_SOURCE_FIXED_NEAR_ANCHOR_END',factors=r['factors'],training_admission=False))
    for r in real:
        aid=r['pair_id'];o=r['original_input'];original_id=o['original_row_id'];need(original_id in pubid,'All real-control anchor sourceTRAIN');p=pubid[original_id];source=p['original']['source_id'];need(r['donor_id']in donor_ids,'Real donor exact existing CF catalog');d=donor_ids[r['donor_id']]
        need(o['native_slot']==p['native_slot']and o['query_xy']==p['query_xy']and o['L']==p['L'],'Selected original real q/L/native exact')
        supported=r['PASS']is True and r['constructed']is True;ns=mr_nodes(source,original_id,p['native_slot'],p['query_xy'],p['L'])+donor_nodes(d)
        for vi,var in enumerate(VAR_REAL):
            source_variant='ADJACENT_SHAM'if var=='SHAM'else var;packet=r['variants'].get(source_variant);physical=packet if isinstance(packet,dict)and isinstance(packet.get('path'),str)and isinstance(packet.get('sha256'),str)else None;graph.add(ns);group='REAL__'+aid+'__'+var
            status.append(dict(domain='REAL_SEMANTIC_END_WEAK',attempt_id=aid,variant=var,group_id=group,source_variant=source_variant,source_status=r['status'],source_packet=physical,physical=physical is not None,selected_native_slot=o['native_slot'],stored_supported=supported,origin_nodes=ns,cache_row_indices=None))
            if physical is None:continue
            for k in range(8):
                m=pubframe[source][k];qid=m['row_id'];record=dict(m,original_occurrence_id=qid,row_id=group+'/native'+str(k))
                sha96=r.get('pixel_detail',{}).get('all8_RGB96_array_sha256',{}).get(source_variant)
                rgb=m['actual_raw_RGB96_sha256']if var=='BEFORE'else sha96[k]if sha96 is not None else None
                need(rgb is not None,'Constructed real RGB96 fingerprint declared in proof');valid=m['actual_raw_valid96_sha256']
                nn=mr_nodes(source,qid,k,m['query_xy'],m['L'])+donor_nodes(d)+fingerprint_nodes(m,rgb,valid,m['original']['feature_constructed'])
                idx=len(allrows['DIRECT_CONTROL']);allrows['DIRECT_CONTROL'].append(stamp(record,'DIRECT_CONTROL',idx,nn,graph,control_domain='REAL_SEMANTIC_END_WEAK',control_join_key=['REAL_SEMANTIC_END_WEAK',aid,var,k],group_id=group,attempt_id=aid,variant=var,source_variant=source_variant,source_packet=physical,layout='REAL_SINGLE',variant_index=vi,selected_native_slot=o['native_slot'],stored_weak_direct_target=(1 if var=='COVER'else 0)if supported and k==o['native_slot']else -1,authority='SOURCE_RGB_SEMANTIC_END_WEAK',source_id=source,original_frame_index=m['original']['frame_index'],donor=d,training_admission=False))
    ids,components=graph.close()
    for rr in list(allrows.values())+[status]:
        for r in rr:
            cids={ids[n]for n in r['origin_nodes']};need(len(cids)==1,'All connected row origins same component');r['component_id']=next(iter(cids));r['role']=role(r['component_id'])
    bygroup=defaultdict(list)
    for r in allrows['DIRECT_CONTROL']:bygroup[r['group_id']].append(r['cache_index'])
    for s in status:s['cache_row_indices']=bygroup.get(s['group_id']);need((s['cache_row_indices']is not None)==s['physical'],'Null missing, never fake rows')
    need(len(status)==3*(expected_counts[5]+expected_counts[6]),'Every original attempt times3 retained')
    return allrows,status,components

def invented():
    tests={};a=Graph();a.add([node('SYNTH_BASE','V129',1),node('QUERY','a')]);a.add([node('QUERY','a'),node('DONOR','d')]);ids,rec=a.close();tests['transitive_origin_and_donor_component']=len(rec)==1
    b=Graph();b.add([node('DONOR','d'),node('QUERY','a')]);b.add([node('QUERY','a'),node('SYNTH_BASE','V129',1)]);tests['canonical_component_order_invariant']=a.close()==b.close()
    tests['component_id_exact_no_trailing_newline']=rec[0]['component_id']==hashlib.sha256('\n'.join(sorted(rec[0]['origin_nodes'])).encode()).hexdigest()
    tests['fixed20_assignment_exact']=rec[0]['role']==('HOLDOUT'if int(hashlib.sha256((SALT+'\n'+rec[0]['component_id']).encode()).hexdigest()[:16],16)%100<20 else'TRAIN')
    q=Graph();q.add([node('SYNTH_BASE','V129',1)]);q.add([node('MR_FRAME','MR6D','1')]);tests['numeric_namespace_does_not_alias']=len(q.close()[1])==2
    placeholder=dict(original_query_available=False);tests['U_placeholder_fingerprints_do_not_connect']=not fingerprint_nodes(placeholder,'a'*64,'b'*64,True)
    good=dict(original_query_available=True);tests['valid_constructed_fingerprint_explicit']=len(fingerprint_nodes(good,'a'*64,'b'*64,True))==1 and not fingerprint_nodes(good,'a'*64,'b'*64,False)
    spec=importlib.util.spec_from_file_location('invented_split_mask_adapter',H/'trainer_component_mask_adapter.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);np=m.np
    rows=[dict(row_id='r'+str(i),original_global_index=i,native_slot=i,query_xy=[1.,2.]if i else None,L=2. if i else None,original_query_available=bool(i))for i in range(3)]
    split=[dict(r,cache_index=i,role='HOLDOUT'if i==1 else'TRAIN')for i,r in enumerate(rows)];mask=m.fit_mask(rows,split);tests['allrows_U_null_META_retained']=mask.tolist()==[True,False,True]and rows[0]['original_query_available']is False
    full=dict(support=[[np.array([0,1]),np.array([2,3])]]*2,conditional=[np.array([.1,.2,.3,.4]),np.array([.25]*4),np.array([0.,.5,.5,0.])],public=[np.array([0,1]),np.array([0,1]),np.array([0,1])],CF=[np.array([0,1]),np.array([0,1])]);calls=[]
    caches={d:dict(x=dict(original_query_available=np.ones(n,bool)))for d,n in [('SYNTH_AUG65536',4),('PUBLIC_TRAIN1480',2),('CF3816',2)]};masks={d:np.array([True,False,True,True])if d=='SYNTH_AUG65536'else np.array([True,False])for d in caches}
    def old(caches):calls.append(True);return full
    result=m.masked_old_pools(caches,masks,old);tests['full_old_guards_called_before_masks']=calls==[True]
    tests['masked_soft_classconditional_no_row_renorm']=result['feasible']and np.allclose(result['pools']['conditional'][0],[.125,0,.375,.5],rtol=0,atol=1e-15)and result['pools']['conditional'][2].tolist()==[0.,0.,1.,0.]
    empty=m.masked_direct_pools(np.array([True,True,False]),np.array([0,1,-1],np.int8),np.array([True,False,True]));tests['empty_class_no_rebalance']=not empty['feasible']and empty['counts']==[1,0]
    def h(text):return hashlib.sha256(text.encode()).hexdigest()
    def binding(name):return dict(path='/invented/'+name,sha256=h(name))
    def meta(k,prefix='MR'):
        return dict(row_id=prefix+str(k),original_global_index=k,native_slot=k,query_xy=[10.+k,20.+k],L=96.,original_query_available=k!=7)
    aa=[];pp=[];cc=[];oo=[]
    for k in range(8):
        mm=meta(k,'SYN');aa.append(dict(mm,split='TRAIN',base_index=0,scene_index=0,anchor_scene_index=0,family='SELF_SCENE15',source_image_original_global_index=k,anchor_original_global_index=k,source_image_original_query_id='SYN'+str(k),anchor_original_query_id='SYN'+str(k),actual_rgb96_sha256=h('SYN'+str(k)),actual_valid96_sha256=h('valid'+str(k)),crop_constructed=True))
        mm=meta(k);pp.append(dict(mm,split='SOURCE_DEVELOPMENT',actual_raw_RGB96_sha256=h('MR'+str(k)),actual_raw_valid96_sha256=h('valid'+str(k)),original=dict(source_id='dev/1:7',frame_index=0,feature_constructed=True)))
    donor=dict(donor_id='YCB:toy',scene_id=1,frame_id=7,instance_id=0,object_id=9,source_RGB_binding=binding('YCB.png'))
    for vi,var in enumerate(VAR_SYN):
        for k in range(8):
            mm=meta(k);cc.append(dict(mm,domain_index=vi*8+k,split='SOURCE_DEVELOPMENT',actual_raw_RGB96_sha256=h('CF'+var+str(k)),actual_raw_valid96_sha256=h('valid'+str(k)),original=dict(pair_id='PAIR',source_id='dev/1:7',original_query_id='MR'+str(k),original_frame_index=0,variant=var,selected_anchor_native_slot=1,feature_constructed=True)))
            mm=meta(k,'SYN');oo.append(dict(mm,domain='SYNTH_FINITE_END',attempt_id='S0',variant=var,original_occurrence_id='SYN'+str(k),source_packet=binding('S0.npz'),original_input_flags=dict(crop_constructed=True),raw_RGB96_sha256=h('SYN'+str(k))if var=='BEFORE'else h('DS'+var+str(k)),raw_valid96_sha256=h('valid'+str(k))))
    synthetic=[dict(attempt_id='S'+str(k),base_index=0,native_slot=k,status='CONTROLLED_DIRECT_END_WEAK_PAIR_SUPPORTED'if k==0 else'UNKNOWN_RETAINED',outputs=binding('S0.npz')if k==0 else None,factors=dict(material='WOOD'))for k in range(8)]
    real=dict(pair_id='R0',donor_id='YCB:toy',original_input=dict(original_row_id='MR1',native_slot=1,query_xy=[11.,21.],L=96.),PASS=True,constructed=True,status='SOURCE_REAL_SAMPLED_DOMAIN_CONTROLLED_END_WEAK_PAIR_SUPPORTED',variants={v:binding(v+'.npz')for v in ('BEFORE','COVER','ADJACENT_SHAM')},pixel_detail=dict(all8_RGB96_array_sha256={v:[h('REAL'+v+str(k))for k in range(8)]for v in ('BEFORE','COVER','ADJACENT_SHAM')}))
    assignment=dict(pair_id='PAIR',anchor_query_id='MR1',donor_id='YCB:toy',donor=dict(scene_id=1,frame_id=7,instance_id=0,object_id=9,image=binding('YCB.png')))
    toy=dict(aug_rows=aa,synth_scene_metadata=[dict(base_index=0,scene_index=0,split='TRAIN',factors=dict(material='WOOD'))],public_rows=pp,cf_rows=cc,direct_rows=oo,synthetic_attempts=synthetic,real193_attempts=[real],assignments159=[assignment])
    rr,ss,comp=build(toy,dict(public={1},CF={9,17}),expected_counts=(8,1,8,24,24,8,1,1))
    tests['scaled_full_adapter_preserves_all8_and_missing_null']=len(rr['DIRECT_CONTROL'])==48 and len(ss)==27 and sum(r['cache_row_indices']is None for r in ss)==21 and sum(not r['original_query_available']for r in rr['DIRECT_CONTROL'])==6
    tests['real_logical_SHAM_source_ADJACENT_SHAM']=all(r['source_variant']=='ADJACENT_SHAM'and r['row_id'].startswith('REAL__R0__SHAM/')for r in rr['DIRECT_CONTROL']if r.get('control_domain')=='REAL_SEMANTIC_END_WEAK'and r['variant']=='SHAM')
    mr_roles={r['component_id']for d in ('PUBLIC_TRAIN1480','CF3816','DIRECT_CONTROL')for r in rr[d]if d!='DIRECT_CONTROL'or r['control_domain']=='REAL_SEMANTIC_END_WEAK'}
    tests['shared_public_CF_real_donor_frame_cannot_split']=len(mr_roles)==1 and len(comp)==2
    tests['repeated_auxiliary_rows_do_not_become_direct_labels']=sum(r['stored_weak_direct_target']>=0 for r in rr['DIRECT_CONTROL'])==6
    broken=json.loads(json.dumps(toy));broken['cf_rows'][0]['L']=95.
    try:build(broken,dict(public={1},CF={9,17}),expected_counts=(8,1,8,24,24,8,1,1));ok=False
    except AssertionError:ok=True
    tests['CF_original_qL_drift_rejected']=ok
    broken=json.loads(json.dumps(toy));broken['assignments159'][0]['anchor_query_id']='MR2'
    try:build(broken,dict(public={1},CF={9,17}),expected_counts=(8,1,8,24,24,8,1,1));ok=False
    except AssertionError:ok=True
    tests['assignment_anchor_drift_rejected']=ok
    need(all(tests.values()),'Invented components/masks');return dict(completed=True,PASS=True,invented_only=True,tests=tests,source_bulk_metadata_or_asset_read=0)

def prepare():
    need(bind(SCOPE)['sha256']==SCOPE_SHA,'Frozen root component rule');scope=read(SCOPE);lineage=read(checked(scope['source_lineage_evidence']));new=read(checked(scope['new_source_generation']))
    need(new['completed']and new['PASS']and new['rows66']==66,'Completed V193 source metadata schema')
    inputs={k:lineage['input_bindings'][k]for k in ('aug_rows','synth_scene_metadata','public_rows','cf_rows','direct_rows','synthetic_attempts')};inputs.update(real193_attempts=new['manifest'],assignments159={k:scope['extra_source_metadata_authorized'][k]for k in ('path','sha256')},raw_cache_protocol=dict(path=str(P/'query_visibility_local_support_raw_head_20261024_v181/raw_cache_preparation/PROTOCOL_BEFORE_RAW_SAM_AND_TARGET_BODIES.json'),sha256='b4e6b4e540b123d0204790a120d000c88c245a7d7867dffe34980b5d2cce9c91'))
    write(CHECKS,invented());protocol=dict(completed=True,prepared_only=True,code=bind(__file__),mask_adapter=bind(H/'trainer_component_mask_adapter.py'),checks=bind(CHECKS),root_scope=bind(SCOPE),lineage=scope['source_lineage_evidence'],real193_complete=scope['new_source_generation'],inputs=inputs,output=str(OUT),canonical_nodes='Compact ensure_ascii=False JSON arrays; nodes sorted lexicographically; componentSHA=SHA256(newline-joined nodes UTF8), no trailing newline. Namespaces SYNTH_BASE/V129/base, SYNTH_ORIGINAL_QUERY_ID/V129/id, MR_FRAME/MR6D/source_id, MR_ORIGINAL_QUERY_ID/MR6D/id, MR_ORIGINAL_QUERY_F64/MR6D/source_id/native/q/L, YCB_DONOR_CUTOUT_ID/id, YCB_SOURCE_FRAME/TRAIN_REAL/scene/frame, YCB_SOURCE_RGB_FILE_SHA256/hash, AVAILABLE_CONSTRUCTED_RGB96_VALID96/rgbhash/validhash.',split_rule=scope['split_rule_before_new_metadata_body_calculations'],direct_plan='V189 synthetic and replacing V193 real only; full3270variant census, null missing cache indices, all8 physical rows. Logical variant is BEFORE/COVER/SHAM in both domains; real source_variant maps SHAM to ADJACENT_SHAM only for lookup. group IDs SYNTH__attempt__variant and REAL__pair__variant; row_id=group/nativeN. Existingweak direct target is a metadata candidate only, no new admission.',trainer_integration='Full caches/target authority guards remain first; fit_mask is separate from original query_available. Call masked_old_pools and masked_direct_pools; preserve old192 draw_step/five_losses and separate direct16+16, RNG recipe and paired equality; old past drawhash is not expected after pool filtering. Refuse empty class, never choose replacement seed/rows. Both final states frozen before heldout report.',fit_feasibility='Metadata census reports giant/empty holdout and direct/native/ownerknown coverage. Old support/softclass numerical train-mass feasibility remains UNASSESSED until separately allowed target-array checks; never inferred from counts.',actual_bulk_source_metadata_array_header_pixel_model_fit_or_eval_read=0,automatic_retry=False)
    write(PROTOCOL,protocol);template=dict(approved_source_component_metadata_census=False,code=protocol['code'],protocol=bind(PROTOCOL),root_scope=protocol['root_scope'],inputs=inputs,output=str(OUT),automatic_retry=False);write(H/'ROOT_ACTUAL_COMPONENT_CENSUS_ENTRY_TEMPLATE_APPROVED_FALSE.json',template);write(H/'FINAL_READY.json',dict(completed=True,PASS=True,prepared_only=True,FINAL_READY_FROZEN=True,code=protocol['code'],mask_adapter=protocol['mask_adapter'],protocol=bind(PROTOCOL),checks=protocol['checks'],false_entry=bind(H/'ROOT_ACTUAL_COMPONENT_CENSUS_ENTRY_TEMPLATE_APPROVED_FALSE.json'),bulk_metadata_census_split_selection_array_header_fit_or_eval_read=0));print(json.dumps(dict(completed=True,prepared_only=True,protocol=bind(PROTOCOL),code=protocol['code'])))

def run(entry):
    e=read(entry);need(e['approved_source_component_metadata_census']is True and e['automatic_retry']is False,'Separate root actual metadata entry');p=read(checked(e['protocol']));need(e['code']==p['code']==bind(__file__)and e['root_scope']==p['root_scope']==bind(SCOPE)and e['inputs']==p['inputs']and e['output']==p['output']==str(OUT),'Frozen exact finite source whitelist');need(read(checked(p['checks']))['PASS'],'Invented checks');checked(p['mask_adapter']);scope=read(checked(e['root_scope']));need(p['split_rule']==scope['split_rule_before_new_metadata_body_calculations']and p['split_rule']['salt']==SALT,'No changed split salt/cutoff');OUT.mkdir(exist_ok=False)
    try:
        data={k:lines(v)for k,v in p['inputs'].items()if k!='raw_cache_protocol'};proto=read(checked(p['inputs']['raw_cache_protocol']));known={'public':set(proto['public_known_original_indices']),'CF':set(proto['CF_known_occurrence_indices'])};rr,status,components=build(data,known)
        for domain in DOMAINS:jlines(OUT/(domain+'_SPLIT_ROWS.jsonl'),rr[domain])
        jlines(OUT/'ALL3270_DIRECT_VARIANT_STATUS_AND_ROLES.jsonl',status);jlines(OUT/'ALL_COMPONENTS_AND_CANONICAL_ORIGIN_NODES.jsonl',components)
        overview={d:dict(rows=len(rows),roles=dict(Counter(r['role']for r in rows)),inputU=sum(not r['original_query_available']for r in rows),native_by_role={role_:dict(Counter(str(r['native_slot'])for r in rows if r['role']==role_))for role_ in ('TRAIN','HOLDOUT')})for d,rows in rr.items()}
        direct=[]
        for r in rr['DIRECT_CONTROL']:
            if r['stored_weak_direct_target']>=0:direct.append(r)
        summary=dict(completed=True,PASS=True,metadata_only=True,domains=overview,component_count=len(components),component_roles=dict(Counter(r['role']for r in components)),largest_component_origin_nodes=max([r['origin_node_count']for r in components],default=0),direct_physical_rows=len(rr['DIRECT_CONTROL']),direct_variant_census=len(status),missing_variants=sum(not r['physical']for r in status),existing_weak_direct_known_by_role={role_:{str(label):sum(r['role']==role_ and r['stored_weak_direct_target']==label for r in direct)for label in (0,1)}for role_ in ('TRAIN','HOLDOUT')},existing_weak_direct_native_by_role={role_:{str(k):sum(r['role']==role_ and r['native_slot']==k for r in direct)for k in range(8)}for role_ in ('TRAIN','HOLDOUT')},existing_weak_direct_authority_native_by_role={role_:dict(Counter(r['authority']+'/native'+str(r['native_slot'])for r in direct if r['role']==role_))for role_ in ('TRAIN','HOLDOUT')},old5_train_class_mass_feasibility='UNASSESSED_WITHOUT_SEPARATELY_ALLOWED_TARGET_ARRAYS',source_holdout_for_new_fromscratch_only=True,no_retry_rebalance_or_new_admission=True,old_exposed_C54_C55_independent_validation=False,new_fit_or_eval=0)
        summary['declared_factor_rows_by_domain_role']={d:{role_:{factor:dict(Counter(str(r.get('factors',{}).get(factor,'NOT_DECLARED'))for r in rows if r['role']==role_))for factor in ('material','aspect','geometry_family','foreign_family')}for role_ in ('TRAIN','HOLDOUT')}for d,rows in rr.items()};summary['owner_known_rows_by_role']={d:{role_:sum(r.get('owner_known_original')is True and r['role']==role_ for r in rr[d])for role_ in ('TRAIN','HOLDOUT')}for d in ('PUBLIC_TRAIN1480','CF3816')};write(OUT/'SUMMARY.json',summary);outputs={x.name:bind(x)for x in [OUT/(d+'_SPLIT_ROWS.jsonl')for d in DOMAINS]+[OUT/'ALL3270_DIRECT_VARIANT_STATUS_AND_ROLES.jsonl',OUT/'ALL_COMPONENTS_AND_CANONICAL_ORIGIN_NODES.jsonl',OUT/'SUMMARY.json']};write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=p['code'],protocol=bind(PROTOCOL),entry=bind(entry),inputs=p['inputs'],outputs=outputs,training_admission=False,actual_array_header_pixel_model_fit_eval_reads=0,automatic_retry=False));print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'),summary=summary)))
    except BaseException:
        write(OUT/'FAILED_EXECUTION_PRESERVED.json',dict(completed=False,PASS=False,automatic_retry=False,traceback=traceback.format_exc()));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);args=ap.parse_args();prepare()if args.prepare else run(args.entry)
