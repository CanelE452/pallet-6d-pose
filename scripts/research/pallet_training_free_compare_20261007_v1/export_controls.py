"""Publish compact copies of existing control rows; no scoring or inference."""
import copy
import gzip
import json
import numpy as np
from .common import ROOT, DOC, read, write, sha, finite
from . import reporting as R


def export():
    assert not (DOC/'CONTROL_ROWS.json').exists()
    protocol=read(DOC/'PROTOCOL.json')
    baseline_path=protocol['baseline']['path']
    assert sha(baseline_path)==protocol['baseline']['sha256']
    old=read(baseline_path)
    ids=[r['id'] for r in old['rows']['RAW']]
    methods={'BASE':copy.deepcopy(old['rows']['RAW'])}
    methods.update({name:copy.deepcopy(rows) for name,rows in old['rows'].items() if name!='RAW'})
    previous=sha(DOC/'N0_LINK_RECEIPT.json')
    n0,links=R.link_N0(old,ids)
    assert previous==sha(DOC/'N0_LINK_RECEIPT.json')
    methods.update(n0)
    raw={r['id']:r['initial_points'] for r in protocol['input_manifest']}
    sources=[dict(path=str(baseline_path),sha256=sha(baseline_path)),*links]
    coordinates={'BASE':raw}
    for family,prefix in (('N3','N3_DIM_SYM'),('N0','N0_BASE_REPLAY')):
        for seed in (1,2,3):
            path=ROOT/f'data/pallet/results/pallet_dim_conditioned_p_v1/predictions/REAL_DEV/{prefix}_seed{seed}.json'
            if not path.exists():
                continue
            packet=read(path)
            coordinates[f'{family}_seed{seed}']={r['id']:None if r['selected_index'] is None else
                r['candidates'][r['selected_index']]['keypoints_xy'] for r in packet['records']}
            sources.append(dict(path=str(path),sha256=sha(path)))
    for seed in (1,2,3):
        path=ROOT/f'data/pallet/results/pallet_posefix_replay_diagnosis_v1/predictions/seed{seed}_REAL_DEV.npz'
        saved=np.load(path,allow_pickle=False)
        coordinates[f'PoseFix_seed{seed}']={str(fid):saved['points'][i,0,1] for i,fid in enumerate(saved['ids'])}
        sources.append(dict(path=str(path),sha256=sha(path)))
    path=DOC/'CONTROL_ROWS.jsonl.gz';rows_written=0;point_counts={}
    with gzip.open(path,'wt',encoding='utf-8',compresslevel=6) as stream:
        for name,rows in methods.items():
            assert len(rows)==319 and {r['id'] for r in rows}==set(ids)
            point_counts[name]=0
            for original in rows:
                row=copy.deepcopy(original);row['method']=name
                if name in coordinates:
                    row['native_points']=coordinates[name][row['id']]
                    point_counts[name]+=int(row['native_points'] is not None)
                else:
                    row['native_points']=None
                    row['native_points_status']='NOT_LINKED: existing metrics only; no model/F rerun'
                row['new_F_calls']=0;row['new_model_forwards']=0
                stream.write(json.dumps(finite(row),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
                rows_written+=1
    assert rows_written==3190
    receipt=dict(status='DONE',classification='copies of valid saved control rows; no new model/F/2D scoring',
        methods=list(methods),rows=rows_written,full_denominator_per_method=319,
        raw_rows_file=path.name,raw_rows_sha256=sha(path),raw_rows_bytes=path.stat().st_size,
        existing_coordinates_available=point_counts,source_bindings=sources,
        new_model_forwards=0,new_final_F=0,new_2D_scores=0,new_training=0)
    write(DOC/'CONTROL_ROWS.json',receipt)
    print('CONTROL_EXPORT_DONE',rows_written,path.stat().st_size,point_counts)
    return receipt


if __name__=='__main__':
    export()
