"""Register external/outside evaluation strata for an existing source cache.

Only source labels/metadata are read. No change to cached training labels.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readcsv(p):
    with Path(p).open(newline='')as f:return list(csv.DictReader(f))
def yes(x):return str(x).lower()in('true','1')


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--ycb-queries',action='append',default=[])
    p.add_argument('--synthetic-states',default=str(HERE/'SYNTH_SOURCE_QUERY_STATES.csv'));p.add_argument('--output',required=True)
    args=p.parse_args();out=HERE/args.output;assert out.resolve().is_relative_to(HERE)
    source={(r['id'],int(r['native_corner'])):r for r in readcsv(args.synthetic_states)}
    for path in args.ycb_queries:
        for r in readcsv(path):
            key=(r['id'],int(r['query_index']))
            assert key not in source,key
            v,f=int(r['presence_target']),yes(r['full_member'])
            assert not(v and not f)
            source[key]=dict(id=r['id'],native_corner=r['query_index'],split=r['split'],source_state='VISIBLE_TARGET'if v else'EXTERNAL_OCCLUDED_SUPPORT'if f else'OUTSIDE_TARGET_SUPPORT',exact_presence=v,M0_member=int(f),M4_member=v,visible_boundary_uncertain=r['boundary'],semantic_self='UNKNOWN',semantic_direct='UNKNOWN',image_sha256=r['image_sha256'])
    with np.load(args.cache,allow_pickle=False)as z:
        ids=z['id'].astype(str);corners=z['native_corner'].astype(int);y=z['y'];valid=z['valid'];sources=z['source'];hashes=z['image_sha256'].astype(str)
    rows=[]
    for i,(ident,k)in enumerate(zip(ids,corners)):
        r=source[ident,int(k)]
        assert int(r['exact_presence'])==y[i] and str(r['image_sha256'])==hashes[i]
        rows.append(dict(cache_row_index=i,id=ident,native_corner=int(k),source=str(sources[i]),split=r['split'],source_state=r['source_state'],exact_presence=int(y[i]),valid=bool(valid[i]),full_member=int(r['M0_member']),visible_boundary_uncertain=yes(r['visible_boundary_uncertain']),semantic_self='UNKNOWN',image_sha256=hashes[i]))
    with out.open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out.with_suffix('.BINDING.json')).write_text(json.dumps(dict(cache=dict(path=str(Path(args.cache).resolve()),sha256=sha(args.cache)),state_manifest=dict(path=str(out),sha256=sha(out)),source_labels=[dict(path=str(Path(q).resolve()),sha256=sha(q))for q in [args.synthetic_states,*args.ycb_queries]],query_count=len(rows),current_eval_read=False,label_changed=False),indent=2)+'\n')
    print(json.dumps(dict(out=str(out),queries=len(rows))))


if __name__=='__main__':main()
