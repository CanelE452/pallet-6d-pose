"""Reference-free frozen asset status by original split and input metadata."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from . import image_batch as I,material_assets as A


def report(run,d1):
    run,d1=map(Path,(run,d1));identity_path=run/'s3_s4_production/MODEL_IDENTITY_INPUTS1241.jsonl.gz'
    identities=list(I.rows(identity_path));inputs={};bindings={}
    for domain,name in [('real','REAL238'),('synthetic','SYNTHETIC1003')]:
        path=d1/'d1_upstream_v1'/f'{name}_UPSTREAM.jsonl.gz'
        bindings[domain]=dict(path=str(path),sha256=A.sha(path))
        for row in I.rows(path):inputs[row['id']]=row
    counts=Counter();real=defaultdict(lambda:dict(count=0,cluster_counts=Counter(),partition_counts=Counter()));rows=[]
    for row in identities:
        frame=inputs[row['id']];domain=row['domain'];name=row['named_source_asset']
        status=('NAMED_USD_GEOMETRY_PRESENT_SIGNED_AXES_UNCERTIFIED'if name in('scene.usd','scene_1.usd')else
            'NAMED_GLB_MISSING_FIXED_STRUCTURED_PROXY'if name is not None else
            'REAL_KNOWN_DIMENSION_STRUCTURED_PROXY_IDENTITY_UNCERTIFIED')
        split=frame['split']if domain=='synthetic'else'REAL238'
        counts[(domain,split,status,str(name),row['proxy_key'])]+=1
        rows.append(dict(id=row['id'],domain=domain,original_split=split,named_source_asset=name,
            proxy_key=row['proxy_key'],asset_status=status,selection_reason=row['selection_reason'],
            physical_CAD_identity_verified=False,signed_axes_independently_certified=False))
        if domain=='real':
            dims=np.round(row['known_physical_dimensions_xyz_m'],12).tolist()
            key=(frame['prediction_object_type'],row['proxy_key'],I.canonical(dims),row['selection_reason'])
            record=real[key];record['count']+=1;record['cluster_counts'][frame['cluster_id']]+=1
            record['partition_counts'][frame['partition']]+=1
    expected=Counter({('real','REAL238'):238,('synthetic','train'):768,('synthetic','validation'):235})
    if Counter((r['domain'],r['original_split'])for r in rows)!=expected:
        raise ValueError('Original domain/split counts differ')
    real_records=[]
    for key,record in sorted(real.items()):
        real_records.append(dict(prediction_input_object_type=key[0],proxy_key=key[1],
            known_dimensions_xyz_m_display=__import__('json').loads(key[2]),selection_reason=key[3],
            count=record['count'],cluster_counts=dict(record['cluster_counts']),partition_counts=dict(record['partition_counts'])))
    result=dict(schema='S3_S4_REFERENCE_FREE_PROXY_ASSET_DOMAIN_COUNTS_V1',complete=True,rows=1241,
        source_asset_identity_path=str(identity_path),source_asset_identity_sha256=A.sha(identity_path),upstream_bindings=bindings,
        original_split_counts=[dict(domain=k[0],split=k[1],asset_status=k[2],named_source_asset=k[3],proxy_key=k[4],count=v)
            for k,v in sorted(counts.items())],real_known_object_metadata=real_records,
        dimensions_grouping_rounding_decimals_for_display_only=12,production_registration_and_asset_choice_unchanged=True,
        all235_validation_named_GLB_missing_uses_fixed_structured_proxy=True,named_USD_present_geometry_rows=502,named_GLB_missing_proxy_rows=501,
        all_signed_physical_axes_and_instance_CAD_identity_independently_uncertified=True,
        limits='All validation uses a known geometry/appearance mismatch proxy. Rankings remain own-development validation and do not certify actual CAD generalization or signed registration.',
        current_real_reference_reads=0,GT_pose_reads=0,new_production_fits_renders_or_pose_updates=0,report_source_sha256=A.sha(Path(__file__)))
    out=run/'s3_s4_archive_report';I.write(out/'PROXY_ASSET_DOMAIN_COUNTS.json',result)
    rowpath=out/'PROXY_ASSET_STATUS_ROWS1241.jsonl.gz'
    if rowpath.exists():
        if I.canonical(list(I.rows(rowpath)))!=I.canonical(rows):raise ValueError('Completed asset status rows differ')
    else:I.save(rowpath,rows)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--d1',type=Path,required=True)
    a=p.parse_args();print(I.canonical(report(a.run,a.d1)),flush=True)
