"""Independent saved-array/context-plan audit. No learned-model/producer imports."""
from pathlib import Path
import hashlib
import json
import numpy as np

O=Path(__file__).resolve().parent
V=O.parent
EXPECTED='fd80ea1decd024b0cdc69ea6e825694354ddfe83081cf180f2957e897e384f46'


def read(p):return json.loads(Path(p).read_text())
def rows(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for v in iter(lambda:f.read(1<<20),b''):h.update(v)
    return h.hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def verify(b):assert sha(b['path'])==b['sha256'],b['path']


def main():
    assert not(O/'REVIEW.json').exists(),'No overwrite'
    rp=V/'consistent_context_features/FEATURE_COMPLETE.json'
    assert sha(rp)==EXPECTED
    r=read(rp);assert r['completed']and r['PASS']
    for k in('output','code','protocol','entry','design','new_query_lineage','input_unit_lineage','fresh_context_facts','C13_input_metadata_scores'):verify(r[k])
    for b in r['metadata_inputs'].values():verify(b)
    for b in r['source_feature_packets']:verify(b)
    u=rows(r['input_unit_lineage']['path']);oc=rows(r['new_query_lineage']['path']);facts=rows(r['fresh_context_facts']['path'])
    assert len(u)==15004 and len(oc)==46108 and len(facts)==623
    with np.load(r['output']['path'],allow_pickle=False)as f:new={k:np.array(f[k],copy=True)for k in f.files}
    expected_keys={'features','occurrence_unit','query_available','factual_query_available','context_available','global_occurrence_index','unit_Q_inside'}
    assert set(new)==expected_keys
    x=new['features'];assert x.dtype==np.float32 and x.shape==(15004,1928)and np.isfinite(x).all()
    source={}
    for b in r['source_feature_packets']:
        with np.load(b['path'],allow_pickle=False)as f:source[b['path']]=np.array(f['features'],copy=True)
    old=np.stack([source[z['source_feature_ref']['file']['path']][z['source_feature_ref']['index']]for z in u])
    del source
    assert old.shape==x.shape and old.dtype==np.float32
    assert x[:,768:1920].tobytes()==old[:,768:1920].tobytes()
    assert x[:,1925:1927].tobytes()==old[:,1925:1927].tobytes()
    equal=np.array([z['box_exact_equal']for z in u],bool)
    assert equal.sum()==13560 and x[equal].tobytes()==old[equal].tobytes()
    aliases=[]
    for i,z in enumerate(u):
        assert z['source_feature_unit_index']==i
        c=z['canonical_context_feature_unit_index']
        if i!=c:
            assert x[i].tobytes()==x[c].tobytes()
            aliases.append((i,c))
    assert len(aliases)==28
    assert all(x[i,:768].tobytes()!=old[i,:768].tobytes()for i,c in aliases)
    fresh=[i for i,z in enumerate(u)if z['new_context_needed']]
    assert len(fresh)==1400 and all(x[i,:768].tobytes()!=old[i,:768].tobytes()for i in fresh)
    un=np.array([z['context_unavailable']for z in u],bool)
    assert un.sum()==72 and not x[un,:768].any()and not x[un,1920:1925].any()and not x[un,1927].any()
    fractions=np.array([z['proposed_context_valid224_fraction_geometry']for z in u],np.float32)
    assert np.array_equal(x[:,1924],fractions)
    expected_inside=[]
    for z in u:
        q=z['query_xy'];b=z['proposed_q8_box_xyxy']
        ok=not z['context_unavailable']and q is not None and b is not None
        if ok:
            q=np.asarray(q,np.float64);b=np.asarray(b,np.float64)
            ok=bool(np.isfinite(q).all()and not np.array_equal(q,[-1,-1])and(q>=b[:2]).all()and(q<=b[2:]).all())
        expected_inside.append(ok)
    inside=np.array(expected_inside,bool)
    assert new['unit_Q_inside'].dtype==bool and np.array_equal(new['unit_Q_inside'],inside)
    assert np.array_equal(x[:,1927],inside.astype(np.float32))and not x[~inside,384:768].any()
    mapping=np.array([z['source_feature_unit_index']for z in oc],np.int32)
    assert new['occurrence_unit'].dtype==np.int32 and np.array_equal(mapping,new['occurrence_unit'])
    assert set(mapping.tolist())==set(range(15004))
    assert np.array_equal(new['global_occurrence_index'],np.arange(46108))
    with np.load(r['C13_input_metadata_scores']['path'],allow_pickle=False)as f:
        oldmeta={k:np.array(f[k],copy=True)for k in('input_unit_index','global_occurrence_index','query_available','factual_query_available','context_available')}
    assert np.array_equal(mapping,oldmeta['input_unit_index'])
    for k in('global_occurrence_index','query_available','factual_query_available','context_available'):assert np.array_equal(new[k],oldmeta[k]),k
    for k in('query_available','factual_query_available','context_available'):assert new[k].shape==(46108,)and new[k].dtype==bool
    assert np.array_equal(new['query_available'],new['factual_query_available']&new['context_available'])
    assert np.array_equal((~un)[mapping],new['context_available'])
    for n,z in enumerate(oc):
        assert z['global_occurrence_index']==n
        unit=u[int(mapping[n])]
        for k in('input_id','logical_slot','query_xy','L','raw_split','split'):assert z[k]==unit[k]
    # Independent float64 cosine replay. Bound declared from384-term FP32 error,
    # not from observed maxima or any model performance result.
    eps=np.finfo(np.float32).eps;gamma=(385*eps)/(1-385*eps)
    bound=8*gamma+4*eps
    errors=[]
    for j,(a,b)in enumerate(((0,3),(0,2),(1,3),(1,4))):
        aa=x[:,384*a:384*(a+1)].astype(np.float64);bb=x[:,384*b:384*(b+1)].astype(np.float64)
        want=np.sum(aa*bb,axis=1)/(np.maximum(np.linalg.norm(aa,axis=1),1e-12)*np.maximum(np.linalg.norm(bb,axis=1),1e-12))
        error=float(np.max(np.abs(want-x[:,1920+j])))
        assert error<=bound,(j,error,bound)
        errors.append(error)
    covered=[];groupids=set()
    for f in facts:
        assert f['context_group_id']not in groupids;groupids.add(f['context_group_id'])
        assert f['units']and f['representative_unit']==f['units'][0]
        for i in f['units']:
            z=u[i];assert z['new_context_needed']and z['context_group_id']==f['context_group_id']
            assert z['decoded_RGB_sha256']==f['native_RGB_sha256']and z['proposed_q8_box_xyxy']==f['native_box']
            assert np.float32(f['target_valid_fraction'])==fractions[i]
            covered.append(i)
    assert sorted(covered)==fresh and len(groupids)==623
    assert r['encoder_calls']==79 and r['primary_encoder_calls']==78 and r['repeat_calls']==1
    assert r['model_state_unchanged']and r['first_batch_repeat_exact']and r['frozen_encoder_state_before']==r['frozen_encoder_state_after']
    report=dict(completed=True,PASS=True,blockers=[],feature_pack=r['output'],feature_receipt=bind(rp),
        own_code=bind(__file__),scope='Saved PUBLIC source feature vectors and input-only metadata only; no new model/encoder/weights/sourcepixels/teacher/private/V112/PnP.',
        checks=dict(units=15004,occurrences=46108,unchanged_box_vectors_byte_exact=13560,
            fresh_canonical_TQ_changed=1400,UPDATED_before_aliases_byte_exact=28,
            fresh_native_context_facts=623,all_F_C_P_and_localfractions_bitexact=True,
            original_mapping_factual_context_final_availability_exact_C13=True,
            Q_inside_replayed=int(inside.sum()),context_unavailable_units=int(un.sum()),
            context_unavailable_occurrences=int((~new['context_available']).sum()),
            final_unavailable_occurrences=int((~new['query_available']).sum()),
            factual_unavailable_occurrences=int((~new['factual_query_available']).sum()),
            four_cosines_max_abs_error=errors,cosine_theoretical_FP32_error_bound=bound),
        source_features_only_changed_permitted_TQ_fields=True,
        producer_only_certificates=dict(frozen_encoder_state_unchanged=True,firstrepeat_exact=True,primary_calls=78,repeat_calls=1),
        limits=['Fresh T/Q embeddings are not independently re-encoded. NativeRGB/context pixel correctness remains bound producer/replay certificate.',
            'All46108 q/L/IDs/folds are verified against sealed input metadata, with no teacher relabel or filtering.',
            'No semantic/nativegold, accuracy, model selection or current-admission claim.'],
        source_payloads_read=['Savedfeatures3sourcepackets','CompletedC14featurepacket','C13sourceNPZinputmetadata_members_only'],
        model_encoder_head_forward_fit_current_V112_PnP=0)
    reportpath=O/'REVIEW.json';reportpath.write_text(json.dumps(report,indent=2)+'\n')
    (O/'SEAL.json').write_text(json.dumps(dict(completed=True,PASS=True,report=bind(reportpath),feature_pack=r['output'],feature_receipt=bind(rp),own_code=bind(__file__)),indent=2)+'\n')
    print(json.dumps(dict(review=bind(reportpath),seal=bind(O/'SEAL.json'),checks=report['checks']),indent=2))


if __name__=='__main__':main()
