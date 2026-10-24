"""Root-entry-only saved input observability table. No GT/image/model input."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,sys,time,traceback
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_SAVED_INPUT_RELIABILITY_POLICY_BEFORE_FEATURES.json'
POLICY_SHA='cd94b9d260d0d2111dc9ad0ced34dd437577a0c48c799f8654db5ba6973cccfd'
MATH=H/'reliability_math.py';SCHEMA=H/'FEATURE_SCHEMA.json';GENERATED=H/'GENERATED_INPUT_MATH_CHECKS.json';PROTOCOL=H/'PROTOCOL_BEFORE_INPUT_BUILD.json'
CERT=V/'independent_design_review/STATIC_REGISTERED_CUBOID_CF_EDGE_CERTIFICATE.json'
CERT_SHA='86116533ee23ac81adce96048284024a6b0328046a6e60fd542abcd0db3a4ee0'
G42=P/'query_visibility_loco_native_context_20261023_v74/source_contract/native_context_contract_v2.py'
G42_SHA='4ddd134aa26da1d676628bd6987a161ab36f09328dfdaa89bedd213e397755ab'
CROP=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
TARGET=P/'query_visibility_target_context_rgb_20261023_v107/target_context_math.py'
OUT=V/'input_only_reliability_v1';N=26624

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def pure(path,names):
    nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names]
    assert {n.name for n in nodes}==set(names);ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return ns
def lines(p):return[json.loads(x)for x in Path(p).read_text().splitlines()if x]
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def sources(policy):
    out={}
    for corpus,cfg in policy['source_corpora'].items():
        init=read(check(cfg['initializer_seal']));crop=read(check(cfg['input_crop_seal']))
        assert init['completed']and init['PASS']and init['GT_read']is False
        assert crop['completed']and crop['PASS']and crop['GT_read']is False and crop['rows']==cfg['slots']
        assert crop['initializer_seal']==cfg['initializer_seal']
        result=read(check(init['result']));assert result['completed']and result['PASS']and result['frames']==cfg['scenes']and result['slots']==cfg['slots']
        assert result['proposals']==crop['initializer_proposals']and result['queries']==crop['initializer_queries']
        for b in(result['proposals'],result['queries'],crop['input_crop_facts']):check(b)
        out[corpus]=dict(config=cfg,initializer_result=init['result'],proposals=result['proposals'],queries=result['queries'],crop_facts=crop['input_crop_facts'])
    return out

def whitelist(p):
    rows=[]
    for corpus,src in p['sources'].items():
        cfg=src['config'];proposals=lines(src['proposals']['path']);queries=lines(src['queries']['path']);facts=read(src['crop_facts']['path'])
        assert facts['completed']and facts['GT_read']is False and len(proposals)==cfg['scenes']and len(queries)==len(facts['rows'])==cfg['slots']
        prefix='V129'if corpus=='OLD_NATIVE'else'V136'
        for si,proposal in enumerate(proposals):
            assert proposal['scene_index']==si and proposal['base_index']==si//4 and proposal['id']==f'{prefix}_SCENE_{si:04d}'
            assert proposal['split']==('TRAIN'if si//4<cfg['TRAIN_bases']else'VALIDATION')and proposal['raw_hw']==[512,512]
            cs=proposal['candidates'];selected=None if not cs else int(np.argmax([c['score']for c in cs]));assert selected==proposal['selected_index']
            if selected is None:q8=conf=bbox=score=None
            else:
                c=cs[selected];q9=np.asarray(c['keypoints_xy'],np.float64);conf9=np.asarray(c['keypoints_conf'],np.float64)
                assert q9.shape==(9,2)and conf9.shape==(9,)and np.isfinite(q9).all()and np.isfinite(conf9).all()
                q8=q9[:8].tolist();conf=conf9[:8].tolist();bbox=c['box_xyxy'];score=c['score']
            for k in range(8):
                i=si*8+k;qr=queries[i];f=facts['rows'][i];qid=proposal['id']+':native'+str(k)
                assert qr['query_id']==f['occurrence_id']==qid and qr['native_slot']==f['native_slot']==k and f['occurrence_index']==i
                for key in('scene_index','base_index','split'):
                    assert qr[key]==f[key]==proposal[key]
                q=None if q8 is None else q8[k];assert q==qr['query_xy']==f['query_xy']and f['initial_points']==q8 and f['L']==qr['roi_side']
                assert f['actual_original_initializer']is True and f['source_emulation']is False
                fields=('context_box_xyxy','finite_non_sentinel_query','native_continuous_valid','native_nearest_valid','context_geometry_valid',
                 'crop_constructed','feature_constructed','query_available','factual_query_available','context_available','Q_inside','RGB_central4_available')
                row=dict(global_index=len(rows),corpus=corpus,source_occurrence_index=i,occurrence_id=qid,scene_index=si,base_index=si//4,native_slot=k,split=proposal['split'],
                 query_xy=q,q8_raw=q8,native_conf8=conf,selected_detector_confidence=score,selected_bbox=bbox,L=f['L'],raw_hw=proposal['raw_hw'],selected_index=selected,
                 valid96_chunk=f['rgb96_chunk'],valid96_row=f['rgb96_row'],valid96_sha256=f['valid96_sha256'],**{key:f[key]for key in fields})
                rows.append(row)
    assert len(rows)==N and len({r['occurrence_id']for r in rows})==N
    return rows

class Masks:
    def __init__(self):self.key=None;self.pack=None;self.files=set()
    def get(self,row):
        b=row['valid96_chunk'];key=(b['path'],b['sha256'])
        if key!=self.key:
            # Deliberately do NOT hash the whole NPZ (it contains RGB bytes).
            # Inherit sealed whole-file binding; verify actual valid96 row hashes.
            with np.load(b['path'],allow_pickle=False)as f:self.pack={k:f[k].copy()for k in('valid96','global_index','crop_constructed')}
            assert self.pack['valid96'].dtype==self.pack['crop_constructed'].dtype==bool
            assert self.pack['valid96'].shape[1:]==(96,96);self.key=key;self.files.add(key)
        j=row['valid96_row'];assert int(self.pack['global_index'][j])==row['source_occurrence_index']and bool(self.pack['crop_constructed'][j])==row['crop_constructed']
        mask=self.pack['valid96'][j];assert ash(mask)==row['valid96_sha256'];return mask

def generate_tests():
    m=module(MATH,'v138_invented_math');target=pure(TARGET,('query_extent','valid_box','target_context_grid','sample_target_context'))
    crop=pure(CROP,('crop',))['crop'];g=pure(G42,('input_context_features',))['input_context_features']
    sign=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)
    xyz=sign*[.6,.12,.5];project=np.array([[400.,20.,30.,1000.],[10.,500.,15.,900.],[.2,.1,1.,5.]])
    hp=np.c_[xyz,np.ones(8)]@project.T;q=hp[:,:2]/hp[:,2,None]
    ideal=[m.edge_signals(q,k)for k in range(8)];assert all(z['global_rms']<1e-12 and z['heldout_rms']<1e-12 for z in ideal)
    bad=q.copy();bad[0]+=[12.,-7.];assert m.edge_signals(bad,0)['heldout_rms']>1e-3 and m.edge_signals(bad,0)['global_rms']>1e-5
    small=bad*.0001;assert np.allclose(m.edge_signals(small,0)['heldout_group_residual'],m.edge_signals(bad,0)['heldout_group_residual'],rtol=1e-10,atol=1e-12)
    collapsed=q.copy();collapsed[0]=collapsed[1]
    other=[m.line_through(collapsed[a],collapsed[b])for a,b in m.EDGE_GROUPS[0][1:]]
    assert m.null_of_lines(other)[0]is not None and not m.edge_signals(collapsed,0)['heldout_group_valid'][0]and np.isnan(m.edge_signals(collapsed,0)['heldout_rms'])
    affine=xyz@np.array([[100.,20.],[15.,110.],[35.,-40.]])+[220.,240.]
    assert all(m.edge_signals(affine,k)['heldout_rms']<1e-12 for k in range(8)) # VP infinity legitimate
    for broken in(np.tile([20.,30.],(8,1)),np.c_[np.arange(8),2*np.arange(8)],None):
        z=m.edge_signals(broken,0);assert np.isnan(z['global_rms'])and np.isnan(z['heldout_rms'])and not z['global_group_valid'].any()
    translated=q+[800.,-600.];assert m.edge_signals(translated,0)['heldout_rms']<1e-12
    warp=np.array([[1.1,.03,100.],[.02,.9,-30.],[.0002,-.0001,1.]])
    wh=np.c_[q,np.ones(8)]@warp.T;warped=wh[:,:2]/wh[:,2,None];assert m.edge_signals(warped,0)['heldout_rms']<1e-12 and np.linalg.norm(warped-q,axis=1).min()>3
    dummy=np.zeros((8,10,3),np.uint8)
    for pos in([-.5,0.],[-.25,2.],[0.,0.],[9.49,7.49],[9.5,7.5],[11.,-2.]):
        _,valid=crop(dummy,np.asarray(pos),16.,96);assert np.array_equal(valid[0],m.local_valid96(pos,16.,[8,10]))
    for box in([-2.,-2.,12.,10.],[0.,0.,9.,7.],[9.,7.,10.,8.],[-1.,0.,0.,7.]):
        _,valid=target['sample_target_context'](dummy,box);assert np.array_equal(valid,m.context_valid224(box,[8,10],target['target_context_grid']))
    box=target['query_extent'](q);gg=g(q,0,box,[512,512]);assert gg.shape==(42,)and gg.dtype==np.float32
    risk=m.risk_values(np.nan,np.nan,np.nan,np.nan,np.nan,False,False,False,np.nan,m.edge_signals(None,0))
    assert risk.shape==(10,)and np.array_equal(np.isfinite(risk),[0,0,0,0,0,1,1,0,0,0])and np.array_equal(risk[5:7],[1,1])
    cert=read(check(dict(path=str(CERT),sha256=CERT_SHA)));assert cert['PASS']and tuple(tuple(map(tuple,cert['CF_edge_families'][name]))for name in m.EDGE_GROUP_NAMES)==m.EDGE_GROUPS
    write(GENERATED,dict(completed=True,PASS=True,code=bind(__file__),math=bind(MATH),invented_only=True,static_contract_certificate=bind(CERT),
     projected_cuboid_zero_and_single_corner_perturbation_positive=True,valid_infinite_VP=True,rank1_zeroedge_undefined=True,own_collapsed_edge_undefined_even_when_other3_support_rank2=True,subpixel_uniform_scale_invariance=True,
     common_translation_and_projective_warp_can_hide_absolute_error=True,local96_mask_exact_existing_crop=True,context224_mask_exact_existing_sampler=True,
     noquery_binary_risks1_other_missing_explicit=True,source_q_GT_RGB_feature_prediction_model_body_reads=0))

def prepare():
    assert sha(POLICY)==POLICY_SHA and sha(CERT)==CERT_SHA and sha(G42)==G42_SHA
    m=module(MATH,'v138_schema_math');proof=read(GENERATED);assert proof['PASS']and proof['code']==bind(__file__)and proof['math']==bind(MATH)
    schema=dict(completed=True,N=N,risk_names=list(m.RISK_NAMES),risk_direction='Higher=worse, fixed before GT join; no sign/model/threshold selection.',
     descriptor_names=list(m.DESCRIPTOR_NAMES),descriptor_count=58,metadata_not_numeric_features=['global_index','corpus','source_occurrence_index','occurrence_id','scene_index','base_index','native_slot','split'],
     table_keys=dict(risks='float64[N,10]; undefined NaN',risk_valid='bool[N,10]',risk_names='unicode[10]',descriptors='float64[N,58]',descriptor_valid='bool[N,58]',descriptor_names='unicode[58]',
      G42='float32[N,42] exactV74; missing0',G42_constructed='bool[N]',q8_image_normalized='float64[N,16] rawq8/[W,H]',q8_image_normalized_valid='bool[N,16]',query_xy='float64[N,2], null→NaN',L='float64[N], original',
      global_group_residual='float64[N,3]',global_group_condition='float64[N,3]',global_group_valid='bool[N,3]',heldout_group_residual='float64[N,3]',heldout_group_condition='float64[N,3]',heldout_group_valid='bool[N,3]'),
     G42_placeholder='All42descriptor_valid false when unconstructed; zeros are not measured geometry.',q8_sentinel='Original[-1,-1] retained; numeric descriptor validity false for sentinel.',
     edge_groups={name:[list(e)for e in group]for name,group in zip(m.EDGE_GROUP_NAMES,m.EDGE_GROUPS)},edge_contract=bind(CERT),
     algebra='Normalize2D q by exact maxspan after span>EPS. Homogeneous-line SVD only(3/4×3); legitimate VP infinity; rank/zero EPS1e-10. No fitted camera/pose/query and no query correction.',
     limits=['Low concurrence does not certify absolute location or identity; common projective warp can preserve it.','Heldout residual can implicate qk or another endpoint, not uniquely qk.','Input-unavailable partly defines known failure including modelU; report definitional association.'],
     crop_mask_access='Read valid96/global_index/crop_constructed keys only; inherit wholeNPZ sealed binding without hashing RGB bytes; verify perrowvalid96SHA.',context_valid='Pure existing224 sampler positive-bilinear-neighbor geometry; verified with invented zero images.',forbidden_numeric_inputs=['corpus/fold/IDs','factors/variant/material/occluder','GT/reference/error/teacher/camera/perimagepermutation','C29/v61scores'])
    write(SCHEMA,schema)
    policy=read(POLICY);src=sources(policy)
    protocol=dict(completed=True,prepared_only=True,code=bind(__file__),math=bind(MATH),schema=bind(SCHEMA),generated=bind(GENERATED),policy=bind(POLICY),static_contract=bind(CERT),
     G42_code=bind(G42),crop_code=bind(CROP),target_code=bind(TARGET),sources=src,runtime=dict(executable='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python',python='3.10',NumPy='1.26.4'),output=str(OUT),N=N,GT_RGB_visualfeatures_prediction_scores_model_read=False)
    write(PROTOCOL,protocol)
    write(H/'ROOT_INPUT_ENTRY_TEMPLATE.json',dict(approved_input_only_reliability=False,GT_read=False,RGB_body_read=False,visual_feature_body_read=False,prediction_score_read=False,
     model_forward=False,fit=False,current223=False,V112=False,PnP=False,q_L_correction=False,automatic_retry=False,valid96_key_only_read=True,expected_all_rows=N,
     **{k:protocol[k]for k in('code','math','schema','generated','policy','static_contract')},protocol=bind(PROTOCOL)))
    print(json.dumps(dict(prepared_only=True,code=bind(__file__),math=bind(MATH),schema=bind(SCHEMA),protocol=bind(PROTOCOL),template=bind(H/'ROOT_INPUT_ENTRY_TEMPLATE.json'))),flush=True)

def run(entry):
    e=read(entry);assert e['approved_input_only_reliability']is True and e['valid96_key_only_read']is True and e['expected_all_rows']==N
    assert all(e[k]is False for k in('GT_read','RGB_body_read','visual_feature_body_read','prediction_score_read','model_forward','fit','current223','V112','PnP','q_L_correction','automatic_retry'))
    for key,path in(('code',Path(__file__)),('math',MATH),('schema',SCHEMA),('generated',GENERATED),('policy',POLICY),('static_contract',CERT),('protocol',PROTOCOL)):assert e[key]==bind(path)
    assert sha(POLICY)==POLICY_SHA and sha(CERT)==CERT_SHA and sha(G42)==G42_SHA and not OUT.exists()
    p=read(PROTOCOL);assert all(p[k]==e[k]for k in('code','math','schema','generated','policy','static_contract'))
    assert np.__version__==p['runtime']['NumPy']and sys.version.startswith(p['runtime']['python'])and Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve()
    for key in('G42_code','crop_code','target_code'):check(p[key])
    assert read(GENERATED)['PASS'];assert sources(read(POLICY))==p['sources']
    rows=whitelist(p);OUT.mkdir();lock=OUT/'INPUT_WHITELIST_BEFORE_NUMERICAL_FEATURES.json';write(lock,dict(completed=True,entry=bind(entry),protocol=bind(PROTOCOL),sources=p['sources'],rows=rows,GT_read=False,RGB_body_read=False))
    m=module(MATH,'v138_actual_input_math');target=pure(TARGET,('query_extent','valid_box','target_context_grid'));g42=pure(G42,('input_context_features',))['input_context_features']
    risks=np.full((N,10),np.nan);G=np.zeros((N,42),np.float32);gc=np.zeros(N,bool);qn=np.full((N,16),np.nan);qvalid=np.zeros((N,16),bool)
    fractions=np.full((N,3),np.nan);aux={key:np.zeros((N,3),bool)if key.endswith('_valid')else np.full((N,3),np.nan)for key in('global_group_residual','global_group_condition','global_group_valid','heldout_group_residual','heldout_group_condition','heldout_group_valid')}
    masks=Masks();contextcache={};start=time.monotonic()
    for i,r in enumerate(rows):
        q8=None if r['q8_raw']is None else np.asarray(r['q8_raw'],np.float64);box=r['context_box_xyxy'];q=r['query_xy'];hw=r['raw_hw'];k=r['native_slot']
        assert box==(target['query_extent'](q8)if q8 is not None else None)
        if q8 is not None:
            assert box==target['query_extent'](q8);qn[i]=(q8/np.array(hw[::-1])).ravel();qvalid[i]=np.repeat([m.finite_native(x)for x in q8],2)
        mask=masks.get(r)
        if r['crop_constructed']:
            assert m.finite_native(q)and r['L']is not None and np.array_equal(mask,m.local_valid96(q,r['L'],hw))
            fractions[i,:2]=[mask.mean(),mask[32:64,32:64].mean()]
        else:assert not mask.any()
        frame=(r['corpus'],r['scene_index'])
        if frame not in contextcache:contextcache[frame]=float(m.context_valid224(box,hw,target['target_context_grid']).mean())if target['valid_box'](box)else np.nan
        fractions[i,2]=contextcache[frame]
        if m.finite_native(q)and target['valid_box'](box):G[i]=g42(q8,k,box,hw);gc[i]=True
        geometry=m.edge_signals(q8,k)
        for key in aux:aux[key][i]=geometry[key]
        score=np.nan if r['selected_detector_confidence']is None else r['selected_detector_confidence'];conf=np.nan if r['native_conf8']is None else r['native_conf8'][k]
        risks[i]=m.risk_values(score,conf,*fractions[i],r['query_available'],r['native_continuous_valid'],r['native_nearest_valid'],m.outside_bbox(q,r['selected_bbox']),geometry)
        if(i+1)%4096==0:print(json.dumps(dict(input_rows=i+1,all_rows=N,seconds=time.monotonic()-start)),flush=True)
    descriptors=np.c_[G.astype(np.float64),qn];dv=np.c_[np.repeat(gc[:,None],42,axis=1),qvalid]
    assert risks.shape==(N,10)and descriptors.shape==(N,58)and not G[~gc].any()
    pack=dict(global_index=np.arange(N,dtype=np.int32),corpus=np.array([r['corpus']for r in rows]),source_occurrence_index=np.array([r['source_occurrence_index']for r in rows],np.int32),
     occurrence_id=np.array([r['occurrence_id']for r in rows]),scene_index=np.array([r['scene_index']for r in rows],np.int32),base_index=np.array([r['base_index']for r in rows],np.int32),native_slot=np.array([r['native_slot']for r in rows],np.int8),split=np.array([r['split']for r in rows]),
     query_xy=np.array([r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]for r in rows],np.float64),L=np.array([np.nan if r['L']is None else r['L']for r in rows]),
     risks=risks,risk_valid=np.isfinite(risks),risk_names=np.array(m.RISK_NAMES),descriptors=descriptors,descriptor_valid=dv,descriptor_names=np.array(m.DESCRIPTOR_NAMES),
     G42=G,G42_constructed=gc,q8_image_normalized=qn,q8_image_normalized_valid=qvalid,crop_valid_fractions_local_central32_context=fractions,**aux)
    for key in('finite_non_sentinel_query','native_continuous_valid','native_nearest_valid','context_geometry_valid','crop_constructed','feature_constructed','query_available','factual_query_available','context_available','Q_inside','RGB_central4_available'):
        pack[key]=np.array([r[key]for r in rows],bool)
    pack['q8_raw']=np.array([r['q8_raw']if r['q8_raw']is not None else np.full((8,2),np.nan)for r in rows],np.float64)
    pack['selected_detector_confidence']=np.array([np.nan if r['selected_detector_confidence']is None else r['selected_detector_confidence']for r in rows],np.float64)
    pack['native_keypoint_confidence']=np.array([np.nan if r['native_conf8']is None else r['native_conf8'][r['native_slot']]for r in rows],np.float64)
    table=OUT/'INPUT_ONLY_RELIABILITY26624.npz';np.savez_compressed(table,**pack)
    summary=dict(completed=True,PASS=True,rows=N,risk_defined_counts=np.isfinite(risks).sum(0).tolist(),descriptor_defined_counts=dv.sum(0).tolist(),G42_constructed=int(gc.sum()),valid96_chunks_read=len(masks.files),mask_RGB_key_reads=0,seconds=time.monotonic()-start)
    write(OUT/'INPUT_BUILD_RESULT.json',summary)
    seal=dict(completed=True,PASS=True,rows=N,table=bind(table),schema=bind(SCHEMA),risk_names=list(m.RISK_NAMES),descriptor_names=list(m.DESCRIPTOR_NAMES),counts=dict(all_rows=N,OLD_NATIVE=6144,CONTINUOUS_NATIVE=20480),
     whitelist=bind(lock),result=bind(OUT/'INPUT_BUILD_RESULT.json'),code=bind(__file__),math=bind(MATH),protocol=bind(PROTOCOL),policy=bind(POLICY),entry=bind(entry),
     GT_read=False,C29score_read=False,v61score_read=False,RGB_body_read=False,visual_feature_body_read=False,valid96_key_only_read=True,model_optimizer_fit_current_V112_PnP_q_L_correction=0,
     all_original_rows_retained=True,no_source_fold_or_factor_numeric_feature=True,source_input_bindings=p['sources'])
    write(OUT/'INPUT_SEAL.json',seal);print(json.dumps(dict(completed=True,PASS=True,seal=bind(OUT/'INPUT_SEAL.json'),summary=summary)),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--generated-checks',action='store_true');ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.generated_checks and not a.prepare and a.entry is None:generate_tests()
    elif a.prepare and not a.generated_checks and a.entry is None:prepare()
    elif a.entry and not a.prepare and not a.generated_checks:run(a.entry)
    else:raise SystemExit('Use --generated-checks, --prepare, or root --entry')
