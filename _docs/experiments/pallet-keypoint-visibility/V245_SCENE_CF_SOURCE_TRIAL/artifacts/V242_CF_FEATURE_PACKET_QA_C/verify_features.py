"""별도 Root entry 후 CF 저장 특징만 독립 검산. encoder/classifier 실행 없음."""
import argparse,hashlib,json,traceback
from pathlib import Path
import numpy as np
from packet_io import need,identity,safe,raw,binding,verify,load_json,load_rows,map_npy,load_npz,write_json

H=Path(__file__).absolute().parent
ROLES=('COVER','ADJACENT_SHAM')
PRODUCER_SHA='c025749399f86fe6c0f09fed6ef1c2510bb44a45fc61a196cecb0446864f4335'
DATA_CODE_SHA='83a84fb22e177b0ea012b04f5c622e0ac71f9c5bdd612b9b63e4f1bae76538ca'
SAMPLES_SHA='837bbe7ff406e27c6ed2981493a83c2adab830cd5b2de498e4a48de456456e99'
STATE_SHA='bb4c7a80aa873bf3d34c3c1d44a28cee331503d7ccf52aba85f2d23edb65c3c3'
SAMPLE_KEYS=('full_view','central_view','fractions','central4','condition','target','family','fit','available','row_id')
GEO_KEYS=('group_index','q8','L8','original_available8','node_available','node_fraction','context_fraction','box_xyxy','box_supported','context_supported')
LAYOUT={r+'_'+k+'.npy':('float32',sh) for r in ROLES for k,sh in
    [('BASE1186',(5406,1186)),('NODE_CLS',(5406,8,384)),('CONTEXT_CLS',(5406,384))]}
LAYOUT.update({'RELATIVE_XY8.npy':('float32',(5406,8,2)),'PARENT_INDICES.npy':('int64',(5406,)),'CONSTRUCTED.npy':('bool',(5406,))})
DATA_LAYOUT={'PARENT_INDICES.npy':('int64',(5406,)),'CONSTRUCTED.npy':('bool',(5406,)),
    'PROOF_PASS.npy':('bool',(5406,)),'GROUP_NODE_VALID96.npy':('bool',(1198,8,96,96)),
    'GROUP_CONTEXT_VALID224.npy':('bool',(1198,224,224))}

def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def relative(g,cache,slot):
    q=g['q8'][cache];avail=g['original_available8'][cache];a=q[np.arange(len(slot)),slot];L=g['L8'][cache,slot]
    need(np.isfinite(a).all() and np.isfinite(L).all() and (L>0).all() and np.isfinite(q[avail]).all(),'original available q/L finite')
    x=np.zeros(q.shape,np.float64);np.divide(q-a[:,None],L[:,None,None],out=x,where=avail[:,:,None])
    x=np.clip(x,-8,8).astype(np.float32);x[~avail]=0;x[np.arange(len(slot)),slot]=0
    return x

def norm_error(x):
    if not len(x):return 0.0
    v=np.asarray(x,np.float64);err=float(np.max(np.abs(np.sqrt((v*v).sum(-1))-1)))
    need(err<=1e-5,'supported unit-L2 CLS roundoff <=1e-5; byte sharing remains exact')
    return err

def geometry_check(g,v,cv,d,indices,rows,constructed,proof):
    n=len(indices);m=len(g['group_index']);cache=np.empty(n,np.int64);slot=np.empty(n,np.int64)
    need(n==5406 and m==1198 and len(rows)==n and len(set(indices.tolist()))==n,'all5406 parents/all1198 groups')
    need(g['group_index'].dtype==np.int64 and g['group_index'].shape==(m,) and len(set(g['group_index'].tolist()))==m,'sparse original group indices')
    for k,sh,dt in [('q8',(m,8,2),np.float64),('L8',(m,8),np.float64),('box_xyxy',(m,4),np.float64),
            ('original_available8',(m,8),bool),('node_available',(m,8),bool),('node_fraction',(m,8),np.float32),
            ('context_fraction',(m,),np.float32),('box_supported',(m,),bool),('context_supported',(m,),bool)]:
        need(g[k].shape==sh and g[k].dtype==dt,'geometry header '+k)
    full=v.mean((2,3)).astype(np.float32);centre=v[:,:,32:64,32:64].mean((2,3)).astype(np.float32)
    central4=v[:,:,47:49,47:49].all((2,3));support=g['original_available8']&central4&v.any((2,3))
    need(same(g['node_fraction'],full) and same(g['context_fraction'],cv.mean((1,2)).astype(np.float32)) and same(g['node_available'],support),'independent input-only validity/fractions')
    need(len(d['target'])==7617 and d['target'].dtype==np.int8 and d['family'].dtype==np.int8 and np.isin(d['target'],[0,1]).all() and np.isin(d['family'],[0,1]).all(),'original known7617 family/target')
    for k in ('fit','available','central4'):need(d[k].dtype==bool and d[k].shape==(7617,),'original sample bool')
    need(d['row_id'].dtype.kind=='U' and len(set(d['row_id'].tolist()))==7617,'original sample IDs')
    need(same(d['full_view'],np.arange(7617,dtype=np.int64)*2) and same(d['central_view'],np.arange(7617,dtype=np.int64)*2+1),'original view order')
    need(d['condition'].dtype==np.float32 and d['condition'].shape==(7617,32) and np.isfinite(d['condition']).all() and d['fractions'].dtype==np.float32 and d['fractions'].shape==(7617,2),'original condition/fraction header')
    expected=np.flatnonzero(d['fit']&(d['target']==0)&d['available']).astype(np.int64)
    need(same(indices,expected),'all original FIT available D, not previous success subset')
    for i,r in enumerate(rows):
        p=int(indices[i]);ci=int(r['group_cache_index']);si=int(r['active_slot']);o=r['original_input']
        need(r['pair_index']==i and r['parent_sample_index']==p and 0<=ci<m and 0<=si<8 and r['group_index']==int(g['group_index'][ci]),'pair/parent/group/slot order')
        need(bool(r['constructed'])==bool(constructed[i]) and bool(r['proof_pass'])==bool(proof[i]) and (not proof[i] or constructed[i]),'constructed/proof UNKNOWN distinct')
        need(o['row_id']==str(d['row_id'][p]) and o['family_code']==int(d['family'][p]) and o['fit'] is True and o['target']==0,'original FIT D identity')
        need(o['q']==g['q8'][ci,si].tolist() and o['L']==g['L8'][ci,si] and bool(o['original_query_available'])==bool(g['original_available8'][ci,si]) and bool(o['crop_available'])==bool(support[ci,si]),'original q/L/availability exact')
        need(support[ci,si] and d['available'][p] and central4[ci,si]==d['central4'][p],'active support/central4')
        need(same(np.array([full[ci,si],centre[ci,si]],np.float32),d['fractions'][p]),'original full/central fractions bytes')
        cache[i],slot[i]=ci,si
    syn=d['family'][indices]==0;c=d['condition'][indices]
    need(int(syn.sum())==4565 and int((~syn).sum())==841 and not np.count_nonzero(c[~syn]),'fixed SOURCE families/YCB native UNKNOWN')
    need(np.array_equal(slot[syn],c[syn,:8].argmax(1)) and np.isin(c[syn,:8],[0,1]).all() and (c[syn,:8].sum(1)==1).all(),'original SYN native onehot')
    av=g['original_available8'][cache].copy();av[np.arange(n),slot]=False
    need(same(c[syn,8:16],av[syn].astype(np.float32)) and same(relative(g,cache,slot)[syn],c[syn,16:].reshape(-1,8,2)),'original SYN others-available/relative condition')
    return cache,slot

def check_role(base,node,context,g,cache,slot,constructed,d,indices):
    result=dict(constructed_rows=int(constructed.sum()),proof_UNKNOWN_is_not_a_feature_gate=True,
        supported_nodes_processed=int(g['node_available'][cache[constructed]].sum()),
        context_rows_processed=int((g['context_fraction'][cache[constructed]]>0).sum()),
        active_first384_byte_exact=True,original_condition32_byte_exact=True,known_fractions_byte_exact=True,
        UNCONSTRUCTED_feature_nonzeros=0)
    errors=dict(node_CLS=0.0,context_CLS=0.0,active_centre_CLS=0.0);digest=hashlib.sha256()
    for first in range(0,len(indices),64):
        sl=slice(first,first+64);b=base[sl];n=node[sl];c=context[sl];use=constructed[sl];ci=cache[sl];si=slot[sl];p=indices[sl]
        need(np.isfinite(b).all() and np.isfinite(n).all() and np.isfinite(c).all(),'all features finite')
        need(not(np.count_nonzero(b[~use])+np.count_nonzero(n[~use])+np.count_nonzero(c[~use])),'UNCONSTRUCTED feature zero')
        need(same(b[use,:384],n[use,si[use]]),'active first384 equals NODE bytes, tolerance0')
        need(same(b[use,1154:],d['condition'][p[use]]) and same(b[use,1152:1154],d['fractions'][p[use]]),'condition32/fractions exact bytes')
        mask=g['node_available'][ci]&use[:,None];ctx=(g['context_fraction'][ci]>0)&use
        need(not np.count_nonzero(n[~mask]) and not np.count_nonzero(c[~ctx]),'unsupported node/context zero')
        errors['node_CLS']=max(errors['node_CLS'],norm_error(n[mask]))
        errors['context_CLS']=max(errors['context_CLS'],norm_error(c[ctx]))
        errors['active_centre_CLS']=max(errors['active_centre_CLS'],norm_error(b[use,768:1152]))
        digest.update(b[use,:384].tobytes())
    return result,errors,digest.hexdigest()

def run(entry_path):
    e=json.loads(raw(entry_path))
    need(e.get('approved_independent_SOURCE_CF_feature_packet_QA') is True and e.get('source_only') is True and e.get('automatic_retry') is False,'separate exact Root numeric QA entry required')
    code,meta,payload=e['code_whitelist'],e['metadata_whitelist'],e['payload_whitelist']
    need(identity(e['code'])==identity(binding(__file__)),'exact checker pin');verify(e['code'],code);verify(e['io_code'],code);verify(e['protocol'],code)
    need(identity(e['io_code'])==identity(binding(H/'packet_io.py')),'exact I/O pin')
    scope=load_json(e['root_scope'],meta);load_json(e['root_protocol'],meta)
    need(scope.get('approved_independent_SOURCE_CF_feature_packet_QA') is True,'Root packet-only scope')
    c=load_json(e['feature_complete'],meta);dc=load_json(e['data_complete'],meta);qa=load_json(e['data_QA'],meta)
    need(all(x.get('completed') is True and x.get('PASS') is True for x in (c,dc,qa)),'terminal complete and independent data QA PASS')
    need(c['schema']=='V242_CANONICAL_CF_FEATURES_V1' and c['code']['sha256']==PRODUCER_SHA and dc['schema']=='V242_SCENE_CONSISTENT_CF_DATA_V1' and dc['code']['sha256']==DATA_CODE_SHA,'frozen producer schema/code')
    need(identity(c['data_complete'])==identity(e['data_complete'])==identity(qa['dataset_complete']) and identity(c['data_QA'])==identity(e['data_QA']) and identity(c['root_protocol'])==identity(e['root_protocol']),'feature/data QA/protocol exact closure')
    need(c['rows']==5406 and dc['counts']['attempted']==5406 and dc['all_original5406_attempted'] is True,'all5406 retained')
    need(c['base_state_sha256_before']==c['base_state_sha256_after']==STATE_SHA and c['official_base']['sha256']=='f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb' and c['asset_identity']['vendor_revision']=='7764ea0f912e53c92e82eb78a2a1631e92725fc8','frozen official state metadata; weights pointer inert')
    need(c['new_forward_GT_references_owners_pose_ID'] is False and c['GT_teacher_targets_used_as_features'] is False and c['classifier_optimizer_or_training']==0 and c['old_cached_prefix_or_CONTROL_payload']==0,'no teacher/IDs/classifier forward metadata')
    need(c['new_SOURCE_DEV_RGB_or_CF_feature_inputs']==0 and c['SOURCE_DEV_nonpixel_metadata_in_original_samples_npz_loaded'] is True and c['SOURCE_DEV_metadata_used_for_fit_or_selection'] is False and c['current_or_held_payloads']==0,'precise DEV nonpixel disclosure; no new DEV RGB or current payload')
    need(set(e['features'])==set(c['features'])==set(LAYOUT) and c['features']==c['outputs'] and all(identity(e['features'][k])==identity(c['features'][k]) for k in LAYOUT),'exact9 features')
    names=set(DATA_LAYOUT)|{'GROUP_GEOMETRY.npz'};need(set(e['data_arrays'])==names,'exact6 nonpixel dataset arrays')
    need(all(identity(e['data_arrays'][k])==identity(dc['outputs'][k]) for k in names) and identity(e['pair_metadata'])==identity(dc['outputs']['PAIR_METADATA.jsonl']),'data arrays/sidecar exact closure')
    for key,name in [('parent_indices','PARENT_INDICES.npy'),('constructed','CONSTRUCTED.npy'),('node_valid','GROUP_NODE_VALID96.npy'),('context_valid','GROUP_CONTEXT_VALID224.npy'),('geometry','GROUP_GEOMETRY.npz')]:
        need(identity(c['data_inputs'][key])==identity(e['data_arrays'][name]),'feature producer same dataset bookkeeping input')
    need(identity(c['data_inputs']['pair_metadata'])==identity(e['pair_metadata']),'same pair metadata, no external pointer follow')
    need(identity(e['samples_npz'])==identity(c['samples_npz']) and e['samples_npz']['sha256']==SAMPLES_SHA,'original samples exact pin')
    need({identity(x) for x in payload}=={identity(x) for x in [*e['features'].values(),*e['data_arrays'].values(),e['samples_npz']]},'only exact16 nonpixel packet payloads')
    need(identity(e['sharing_QA'])==identity(c['canonical_sharing_QA_binding']),'sharing receipt binding')
    for b in [*e['features'].values(),*e['data_arrays'].values(),e['samples_npz']]:verify(b,payload)
    out=safe(e['output_directory']);need(out.parent==H and not out.exists(),'fresh own QA output');out.mkdir()
    try:
        f={k:map_npy(e['features'][k],payload,*LAYOUT[k]) for k in LAYOUT}
        a={k:map_npy(e['data_arrays'][k],payload,*DATA_LAYOUT[k]) for k in DATA_LAYOUT}
        need(same(f['PARENT_INDICES.npy'],a['PARENT_INDICES.npy']) and same(f['CONSTRUCTED.npy'],a['CONSTRUCTED.npy']),'feature/data parent+constructed bytes')
        g=load_npz(e['data_arrays']['GROUP_GEOMETRY.npz'],payload,GEO_KEYS);d=load_npz(e['samples_npz'],payload,SAMPLE_KEYS)
        rows=load_rows(e['pair_metadata'],meta);indices=a['PARENT_INDICES.npy'];cons=a['CONSTRUCTED.npy'];proof=a['PROOF_PASS.npy']
        cache,slot=geometry_check(g,a['GROUP_NODE_VALID96.npy'],a['GROUP_CONTEXT_VALID224.npy'],d,indices,rows,cons,proof)
        need(same(f['RELATIVE_XY8.npy'],relative(g,cache,slot)),'all5406 F64 input-only relative math, including unconstructed')
        sharing=load_json(e['sharing_QA'],meta);need(sharing['completed'] is True and sharing['PASS'] is True and sharing['same_forward_construction'] is True and sharing['feature_overwrite_or_tolerance_relaxation'] is False,'canonical construction metadata')
        evidence={}
        for r in ROLES:
            counts,errors,digest=check_role(f[r+'_BASE1186.npy'],f[r+'_NODE_CLS.npy'],f[r+'_CONTEXT_CLS.npy'],g,cache,slot,cons,d,indices)
            need(counts==sharing['roles'][r]==c['canonical_sharing_QA'][r],'independent role counts/sharing receipt')
            evidence[r]=dict(sharing=counts,CLS_max_abs_norm_error=errors,active_shared_bytes_sha256=digest)
        counts=dict(parents=5406,groups=1198,SYN=4565,YCB=841,constructed=int(cons.sum()),proof_PASS=int(proof.sum()),constructed_UNKNOWN=int((cons&~proof).sum()),unconstructed=int((~cons).sum()),SOURCE_DEV_parents=0)
        need(c['constructed']==counts['constructed'] and c['unconstructed']==counts['unconstructed'] and dc['counts']['constructed']==counts['constructed'] and dc['counts']['proof_pass']==counts['proof_PASS'],'dataset/feature construction counts')
        result=dict(completed=True,PASS=True,feature_complete=e['feature_complete'],dataset_complete=e['data_complete'],data_QA=e['data_QA'],entry=binding(entry_path),code=binding(__file__),io_code=e['io_code'],protocol=e['protocol'],counts=counts,roles=evidence,headers={k:dict(dtype=LAYOUT[k][0],shape=list(LAYOUT[k][1]),binding=e['features'][k]) for k in LAYOUT},all_original_condition_fractions_central4_q_L_relative_exact=True,active_NODE_byte_tolerance=0,features_overwritten=0,CLS_unit_norm_abs_tolerance=1e-5,official_state_SHA_before_after=STATE_SHA,original_SOURCE_DEV_nonpixel_samples_metadata_loaded=True,SOURCE_DEV_metadata_used_for_fit_or_model_selection=False,current_admission=False,selected_arm=None,raw_RGB_model_weights_encoder_classifier_or_current_held_payload_reads=0,limits='저장 특징과 원 nonpixel 입력 metadata 검산. 원 RGB/encoder forward/teacher 물리 gold를 재인증하지 않음. constructed UNKNOWN을 supervision mask와 별도로 검산함. 부모 전수는 원 FIT D mask와 비교하며 전체 원 FIT/DEV group closure는 exact binding된 독립 data QA를 인용함.')
        receipt=write_json(out/'REVIEW.json',result);print(json.dumps(receipt),flush=True)
    except BaseException:
        write_json(out/'FAILED_EXECUTION.json',dict(completed=False,PASS=False,automatic_retry=False,traceback=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)
