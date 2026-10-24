"""Read only RGB/query manifest text, query CSV and seal; never image/NPZ."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math

ROOT=Path(__file__).resolve().parents[1]
def ref(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def main():
    here=ROOT/'source_inputs';out=ROOT/'source_audit/INDEPENDENT_SOURCE_INPUT_METADATA_AUDIT.json';assert not out.exists()
    mp=here/'INPUTS_RGB_ONLY.jsonl';cp=here/'QUERY_ID_MAP.csv';sp=here/'SEAL.json'
    seal=json.loads(sp.read_text());rows=[json.loads(x)for x in mp.read_text().splitlines()]
    with cp.open()as f:queries=list(csv.DictReader(f))
    assert len(rows)==seal['targets']==1091 and len(queries)==seal['queries']==8590
    for p in [mp,cp]:assert ref(p)['sha256']==next(x['sha256']for x in seal['outputs']if x['path']==str(p))
    top={'id','global_id','joined_target_index','source','source_metadata_label','group_id','split','image','image_sha256','raw_hw','box','side','queries'}
    inner={'sample_id','joined_query_index','native_corner','query','pixel','native_input_valid'}
    assert all(set(r)==top for r in rows)
    assert all(set(q)==inner for r in rows for q in r['queries'])
    assert set(r['source']for r in rows)=={'SYNTH','ORIGINAL_YCB','CONTROLLED_YCB','PALLOX'}
    assert sorted(r['joined_target_index']for r in rows)==list(range(1091))
    assert len({r['global_id']for r in rows})==1091
    lookup={};source_summary=[];native_valid=0
    raw_alias={'SYNTH':'SYNTH','ORIGINAL_YCB':'YCB','CONTROLLED_YCB':'ABSENT_SOURCE_KEY_IN_CONTROLLED_MANIFEST','PALLOX':'PALLOX'}
    for r in rows:
        assert raw_alias[r['source']]==r['source_metadata_label']and r['id']==r['global_id']and r['split']=='VAL'
        h,w=r['raw_hw'];assert h>0 and w>0 and len(r['box'])==4 and r['side']>0
        assert all(math.isfinite(v)for v in r['box']+[r['side']])
        assert len(r['image_sha256'])==64 and all(c in '0123456789abcdef'for c in r['image_sha256'])
        assert len({q['native_corner']for q in r['queries']})==len(r['queries'])
        for q in r['queries']:
            k=q['native_corner'];assert 0<=k<8 and int(q['sample_id'].rsplit(':',1)[1])==k
            ix=q['joined_query_index'];assert ix not in lookup
            x,y=q['query'];assert math.isfinite(x)and math.isfinite(y)
            px,py=math.floor(x+.5),math.floor(y+.5);assert q['pixel']==[px,py]
            valid=0<=x<w and 0<=y<h and 0<=px<w and 0<=py<h
            assert valid==q['native_input_valid'];native_valid+=valid
            lookup[ix]=(r,q)
    assert sorted(lookup)==list(range(8590))and len({q['sample_id']for r,q in lookup.values()})==8590
    for q in queries:
        ix=int(q['joined_query_index']);r,a=lookup[ix]
        assert int(q['joined_target_index'])==r['joined_target_index']
        for k in ['global_id','source','group_id','image','image_sha256']:assert q[k]==r[k]
        assert q['sample_id']==a['sample_id']and int(q['native_corner'])==a['native_corner']
        assert [float(q['query_x']),float(q['query_y'])]==a['query']
        assert [int(q['pixel_x']),int(q['pixel_y'])]==a['pixel']
        assert (q['native_input_valid']=='True')==a['native_input_valid']and float(q['side'])==r['side']
    assert native_valid==seal['native_valid']==8584
    for s in ['SYNTH','ORIGINAL_YCB','CONTROLLED_YCB','PALLOX']:
        rr=[r for r in rows if r['source']==s];qq=[q for r in rr for q in r['queries']]
        summary=dict(source=s,targets=len(rr),queries=len(qq),unique_RGB_SHA=len({r['image_sha256']for r in rr}),native_input_valid=sum(q['native_input_valid']for q in qq),
            target_query_count_histogram={str(k):v for k,v in sorted(Counter(len(r['queries'])for r in rr).items())},
            native_HW_counts={str(k):v for k,v in sorted(Counter(tuple(r['raw_hw'])for r in rr).items())})
        expected=next(x for x in seal['source_counts']if x['source']==s)
        for k in ['targets','queries','unique_RGB_SHA','native_input_valid','target_query_count_histogram']:assert summary[k]==expected[k]
        source_summary.append(summary)
    synth=next(x for x in source_summary if x['source']=='SYNTH');assert len(synth['native_HW_counts'])==4
    result=dict(PASS=True,schema='SOURCE_INPUT_METADATA_ONLY_INDEPENDENT_AUDIT_V1',
        input_bindings=[ref(p)for p in [mp,cp,sp]],code=ref(Path(__file__)),targets=1091,query_rows=8590,native_input_valid=8584,
        canonical_joined_target_index_coverage='0..1090 exactly once',canonical_joined_query_index_coverage='0..8589 exactly once',
        per_target_native_corner='Unique0..7 subset. SYNTH4..8queries are retained source-valid subset, not silently expanded. Other sources8each.',
        source_summary=source_summary,source_metadata_aliases=raw_alias,GT_fields_in_model_input=0,
        auditor_assumption_correction='The draft incorrectly assumed canonical source==raw metadata source label. Corrected to preserve the explicit lineage aliases above; no input/code/model artifact was changed.',
        explicit_input_key_whitelist=dict(target=sorted(top),query=sorted(inner)),
        forbidden_data_not_read=['RGBbytes','current RGB/GT/scores','models/checkpoints','P3/x/xvalid','NPZ/feature/label arrays'],
        limitation='This metadata audit verifies text identity/coordinates/validity and bound text hashes. It does not independently replay image decoding or feature values; root owns separate x-valid audit.',
        new_forward_training_or_PnP=0)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('SOURCE_METADATA_PASS',ref(out),source_summary)

if __name__=='__main__':main()
