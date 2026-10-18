"""Additional conditional transition bounds; preserves the completed mask V1.

This counts completions of the ordered sampled binary membership sequence.
It does not assign a label to any ambiguous or outside sample, interpolate a
segment, authenticate the renderer encoding, or identify physical mesh edges.
"""
from __future__ import annotations
import argparse,csv,gzip,json,resource,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.research.pallet_risk_information_v1.common import read,write,rows,sha
from .b_mask_diagnostic import line_labels


def transition_bounds(labels):
    """Exact two-state DP over all binary completions (-1/-2 unspecified)."""
    x=[int(v) for v in labels]
    if any(v not in (-2,-1,0,1) for v in x):raise ValueError('Unknown membership encoding')
    if not x:return dict(samples=0,minimum=None,maximum=None,classification='EMPTY')
    infinity=len(x)+1
    low=[0 if x[0]<0 or x[0]==j else infinity for j in (0,1)]
    high=[0 if x[0]<0 or x[0]==j else -infinity for j in (0,1)]
    for v in x[1:]:
        low,high=([min(low[j],low[1-j]+1) if v<0 or v==j else infinity for j in (0,1)],
                  [max(high[j],high[1-j]+1) if v<0 or v==j else -infinity for j in (0,1)])
    minimum,maximum=min(low),max(high)
    kind=('CONDITIONALLY_MULTIPLE' if minimum>=2 else
          'CONDITIONALLY_EXACT_SINGLE' if minimum==maximum==1 else
          'CONDITIONALLY_EXACT_ZERO' if minimum==maximum==0 else 'UNRESOLVED_RANGE')
    return dict(samples=len(x),minimum=minimum,maximum=maximum,classification=kind)


def bounds_with_canvas(labels):
    """Separate whole-line completions from actually sampled canvas segments."""
    x=[int(v) for v in labels];segments=[];start=None
    for i,v in enumerate(x+[-2]):
        if v!=-2 and start is None:start=i
        if v==-2 and start is not None:
            segments.append(dict(start_index=start,end_index_inclusive=i-1,**transition_bounds(x[start:i])));start=None
    return dict(whole_line=transition_bounds(x),in_canvas_segments=segments,
        in_canvas_transition_minimum_sum=sum(v['minimum'] for v in segments) if segments else None,
        in_canvas_transition_maximum_sum=sum(v['maximum'] for v in segments) if segments else None,
        outside_samples=x.count(-2),ambiguous_in_canvas_samples=x.count(-1),
        no_cross_segment_transition_inserted=True,
        whole_line_outside_semantics='Unobserved outside samples are unspecified binary completions, not measured membership')


def run(query_dir,mask_dir,cache,plan,output,*,GO=False):
    if not GO:raise PermissionError('Explicit diagnostic GO required')
    query_dir,mask_dir,cache,plan,output=map(Path,(query_dir,mask_dir,cache,plan,output))
    oldseal=read(mask_dir/'B_MASK_DIAGNOSTIC_SEAL.json');qseal=read(query_dir/'B_QUERY_FACTORIAL_SEAL.json')
    invpath=mask_dir/'MASK_INPUT_INVENTORY.jsonl.gz';qpath=query_dir/'B_QUERY_FACTORIAL_ROWS.jsonl.gz'
    if sha(invpath)!=oldseal['artifacts'][invpath.name]or sha(qpath)!=qseal['artifacts'][qpath.name]:raise ValueError('Prior sealed inputs changed')
    inventory={r['id']:r for r in rows(invpath)};index={read(p)['id']:p.parent for p in(cache/'frames').glob('*/FRAME_SEAL.json')}
    output.mkdir(parents=True,exist_ok=False);start,cpu=time.monotonic(),time.process_time();groups=defaultdict(Counter)
    histograms=defaultdict(lambda:np.zeros(256,dtype=np.int64));metadata=[];line_count=0
    fields=['id','split','phase_index','family','query_variant','support_id','native_corner','mask_kind','whole_minimum','whole_maximum','whole_class',
        'in_canvas_segment_count','in_canvas_minimum_sum','in_canvas_maximum_sum','segments_json','outside_samples','ambiguous_in_canvas_samples']
    with gzip.open(output/'B_MASK_TRANSITION_BOUNDS.csv.gz','wt',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows(qpath):
            fid=row['id'];info=inventory[fid];masks={}
            for kind in ('renderer_visible','renderer_amodal'):
                path=Path(info[kind+'_path'])
                if sha(path)!=info[kind+'_sha256']:raise ValueError('Original raw mask changed')
                with Image.open(path)as image:
                    if image.mode!='L' or image.format!='PNG':raise ValueError('Do not silently convert an unrecognized raw mask format')
                    pixels=np.asarray(image).copy();exif=image.getexif()
                    metadata.append(dict(id=fid,kind=kind,original_sha256=sha(path),mode=image.mode,format=image.format,bands=list(image.getbands()),size=list(image.size),
                        info_keys=sorted(image.info),gamma=image.info.get('gamma'),srgb=image.info.get('srgb'),
                        source_blend_path=image.info.get('File'),source_date=image.info.get('Date'),source_frame=image.info.get('Frame'),
                        source_camera=image.info.get('Camera'),source_scene=image.info.get('Scene'),source_render_time=image.info.get('RenderTime'),
                        exif_tag_ids=sorted(exif),histogram_uint8=np.bincount(pixels.ravel(),minlength=256).tolist()))
                masks[kind]=pixels;histograms[kind]+=np.bincount(pixels.ravel(),minlength=256)
            path=Path(info['SAM_path'])
            if sha(path)!=info['SAM_sha256']:raise ValueError('Archived SAM mask changed')
            with np.load(path,allow_pickle=False)as archive:masks['SAM']=archive['pallet_mask_clean'].copy()
            bundle=next(rows(index[fid]/'BOUNDARY_BUNDLE.jsonl.gz'))
            qmap={q['support_id']:q for g in(bundle or{}).get('groups',[])for q in g['queries']}
            for family in row['families']:
                for query in family['profiles']:
                    original=qmap[query['support_id']]
                    xy=np.asarray(original['sample_pixel_xy'],float)if query['query_variant']=='Q00_ORIGINAL'else np.asarray(query['center_xy'])+np.asarray(original['sample_offsets_base_px'])[:,None]*bundle['pixel_scale']*np.asarray(query['outward_normal_xy'])
                    line_count+=1
                    for kind,mask in masks.items():
                        labels=line_labels(mask,xy,encoding='sam_binary'if kind=='SAM'else'renderer_uint8')
                        result=bounds_with_canvas(labels);whole=result['whole_line']
                        writer.writerow(dict(id=fid,split=row['split'],phase_index=family['phase_index'],family=family['family'],query_variant=query['query_variant'],support_id=query['support_id'],native_corner=query['native_corner'],mask_kind=kind,
                            whole_minimum=whole['minimum'],whole_maximum=whole['maximum'],whole_class=whole['classification'],in_canvas_segment_count=len(result['in_canvas_segments']),
                            in_canvas_minimum_sum=result['in_canvas_transition_minimum_sum'],in_canvas_maximum_sum=result['in_canvas_transition_maximum_sum'],segments_json=json.dumps(result['in_canvas_segments'],separators=(',',':')),
                            outside_samples=result['outside_samples'],ambiguous_in_canvas_samples=result['ambiguous_in_canvas_samples']))
                        for pop in ('all',row['split']):
                            g=groups['/'.join((pop,family['family'],query['query_variant'],kind))];g['profiles']+=1;g[whole['classification']]+=1
                            g['outside_samples']+=result['outside_samples'];g['ambiguous_in_canvas_samples']+=result['ambiguous_in_canvas_samples']
                            g['in_canvas_segment_count']+=len(result['in_canvas_segments'])
                            g['any_in_canvas_segment_conditionally_multiple']+=any(s['minimum']>=2 for s in result['in_canvas_segments'])
                            g['in_canvas_transition_minimum_sum']+=result['in_canvas_transition_minimum_sum']or 0
                            g['in_canvas_transition_maximum_sum']+=result['in_canvas_transition_maximum_sum']or 0
    if len(inventory)!=1003 or line_count!=139976:raise ValueError('Preserve original entire profile union')
    with gzip.open(output/'MASK_ENCODING_METADATA.jsonl.gz','wt')as f:
        for entry in metadata:f.write(json.dumps(entry,allow_nan=False)+'\n')
    write(output/'B_MASK_TRANSITION_BOUNDS.json',dict(schema='CONDITIONAL_MASK_TRANSITION_BOUNDS_V1',status='COMPLETE',frames=1003,B_input_frames=673,phase_family_variant_profiles=line_count,mask_line_records=line_count*3,
        groups={k:dict(v)for k,v in groups.items()},aggregate_histograms={k:v.tolist()for k,v in histograms.items()},
        semantic_contract='Conditional on exactly0/background and255/target in archived renderer L PNG; all1..254 remain unspecified. This encoding has not been authenticated by the missing original generator/compositor.',
        intermediate_values_cause='UNVERIFIED: PNG/gamma metadata and value histograms cannot certify AA versus compositing/dither/processing origin',
        renderer_mask_is_independent_physical_truth=False,SAM_is_predicted_membership_not_reference=True,
        whole_line='Exact DP minimum/maximum transitions among binary completions; outside samples remain unobserved/unspecified, never assigned',
        in_canvas='Separate maximal contiguous sampled-index runs inside canvas, with exact DP per run; no bridge/interpolation across outside runs',
        counterexample='minimum>=2 contradicts a single transition conditional on the specified mask-value contract; not independent physical-edge certification',
        conditional_single='minimum=maximum=1 fixes one sampled membership transition conditional on the contract, not a material boundary or RGB likelihood',
        previous_UNRESOLVED_V1_preserved=True,new_thresholds=0,new_forward=0,new_solver=0,new_optical_cost=0,new_rendering=0))
    write(output/'DIAGNOSTIC_SEAL.json',dict(schema='MASK_TRANSITION_BOUNDS_SEAL_V1',status='COMPLETE',source_sha256=sha(__file__),plan_sha256=sha(plan),
        previous_mask_seal_sha256=sha(mask_dir/'B_MASK_DIAGNOSTIC_SEAL.json'),query_seal_sha256=sha(query_dir/'B_QUERY_FACTORIAL_SEAL.json'),
        artifacts={p.name:sha(p)for p in output.iterdir()if p.is_file()},wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('query-dir','mask-dir','cache','plan','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--diagnostic-GO',action='store_true');a=p.parse_args();run(a.query_dir,a.mask_dir,a.cache,a.plan,a.output,GO=a.diagnostic_GO)


if __name__=='__main__':main()
