"""Private renderer-target versus saved SAM ownership-line diagnostics.

No RGB, pose solve, forward, optical cost, or automatic real type promotion.
Renderer intermediate mask values remain ambiguous, never thresholded.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv,gzip,hashlib,json
from pathlib import Path
import resource,time
import numpy as np
from PIL import Image
from scripts.research.pallet_risk_information_v1.common import read,rows,sha,write


def line_labels(mask,pixels,*,encoding):
    xy=np.asarray(pixels);h,w=mask.shape;valid=np.isfinite(xy).all(1)
    indices=np.zeros_like(xy,dtype=np.int64);indices[valid]=np.floor(xy[valid]+.5).astype(np.int64)
    valid&=(indices[:,0]>=0)&(indices[:,0]<w)&(indices[:,1]>=0)&(indices[:,1]<h)
    labels=np.full(len(xy),-2,dtype=np.int8) # outside/invalid
    value=mask[indices[valid,1],indices[valid,0]];known=np.full(len(value),-1,dtype=np.int8)
    known[value==0]=0;known[value==(255 if encoding=='renderer_uint8'else 1)]=1
    labels[valid]=known
    return labels


def transition_summary(labels):
    x=np.asarray(labels);known=(x>=0)
    adjacent=known[:-1]&known[1:]
    changes=int(np.sum(adjacent&(x[:-1]!=x[1:])))
    complete=bool(known.all())
    return dict(samples=len(x),target_samples=int(np.sum(x==1)),background_samples=int(np.sum(x==0)),
        ambiguous_samples=int(np.sum(x==-1)),outside_samples=int(np.sum(x==-2)),
        adjacent_definite_transition_count=changes,
        full_transition_count=changes if complete else None,
        transition_class=('ZERO_TRANSITION'if changes==0 else'SINGLE_TRANSITION'if changes==1 else'MULTIPLE_TRANSITIONS')if complete else'UNRESOLVED_AMBIGUOUS_OR_OUTSIDE',
        object_membership_not_RGB_brightness=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('query-dir','metadata','cache','observations','upstream','plan','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--diagnostic-GO',action='store_true');a=p.parse_args()
    if not a.diagnostic_GO:raise PermissionError('Root diagnostic GO required')
    seal=read(a.query_dir/'B_QUERY_FACTORIAL_SEAL.json');querypath=a.query_dir/'B_QUERY_FACTORIAL_ROWS.jsonl.gz'
    if sha(querypath)!=seal['artifacts'][querypath.name]:raise ValueError('Sealed query geometry changed')
    out=a.output;out.mkdir(parents=True,exist_ok=False);start,cpu=time.monotonic(),time.process_time()
    metadata={r['id']:r for r in rows(a.metadata)};up={r['id']:r for r in rows(a.upstream)}
    index={read(p)['id']:p.parent for p in (a.cache/'frames').glob('*/FRAME_SEAL.json')}
    inventory=[];counts=defaultdict(Counter);nqueries=0;canvas_disagreements=0
    fields=('id','split','phase_index','family','query_variant','support_id','native_corner','renderer_visible_transition','renderer_amodal_transition','SAM_transition',
        'renderer_visible_target_samples','renderer_visible_background_samples','renderer_ambiguous_samples','outside_samples','SAM_target_samples',
        'amodal_target_visible_nontarget_samples','SAM_target_renderer_visible_background_samples','nearest_pixel_line_in_canvas','continuous_line_in_canvas')
    with gzip.open(out/'B_MASK_QUERY_ROWS.csv.gz','wt',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows(querypath):
            fid=row['id'];info=metadata[fid];annotation=Path(info['annotation_locator'])
            if sha(annotation)!=info['annotation_sha256']:raise ValueError('Original annotation source changed')
            stem=annotation.name.replace('_label.json','.png');release=annotation.parent.parent
            visible=release/'mask_visible'/stem;amodal=release/'mask_amodal'/stem
            sam_path=a.observations/'frames'/index[fid].name/'MASK.npz'
            if not visible.is_file()or not amodal.is_file()or not sam_path.is_file():raise FileNotFoundError('Required original mask unavailable: '+fid)
            with Image.open(visible)as image:vm=np.asarray(image.convert('L'))
            with Image.open(amodal)as image:am=np.asarray(image.convert('L'))
            with np.load(sam_path,allow_pickle=False)as archive:sam=archive['pallet_mask_clean'].copy()
            hw=up[fid]['raw_hw']
            if list(vm.shape)!=hw or list(am.shape)!=hw or list(sam.shape)!=hw:raise ValueError('Raw original mask canvas mismatch')
            folder=index[fid];bundle=next(rows(folder/'BOUNDARY_BUNDLE.jsonl.gz'))
            if bundle and hashlib.sha256(np.asarray(sam,dtype=np.float64).tobytes()).hexdigest()!=bundle['target_mask_float64_sha256']:
                raise ValueError('Saved SAM mask does not match original B appearance mask')
            if not np.isin(sam,[0,1]).all():raise ValueError('Original SAM mask encoding is not binary0/1; do not silently threshold')
            inventory.append(dict(id=fid,split=row['split'],raw_hw=hw,annotation_sha256=info['annotation_sha256'],
                renderer_visible_path=str(visible),renderer_visible_sha256=sha(visible),renderer_amodal_path=str(amodal),renderer_amodal_sha256=sha(amodal),
                SAM_path=str(sam_path),SAM_sha256=sha(sam_path),original_B_mask_binding_exact=bool(bundle),
                renderer_visible_ambiguous_pixels=int(np.sum((vm!=0)&(vm!=255))),
                material_edge_ID_available=False,independent_physical_ownership_verified=False))
            qmap={q['support_id']:q for g in (bundle or{}).get('groups',[])for q in g['queries']}
            for f in row['families']:
                for q in f['profiles']:
                    original=qmap[q['support_id']]
                    if q['query_variant']=='Q00_ORIGINAL':xy=np.asarray(original['sample_pixel_xy'],float)
                    else:xy=np.asarray(q['center_xy'])+np.asarray(original['sample_offsets_base_px'])[:,None]*bundle['pixel_scale']*np.asarray(q['outward_normal_xy'])
                    vl=line_labels(vm,xy,encoding='renderer_uint8');al=line_labels(am,xy,encoding='renderer_uint8');sl=line_labels(sam,xy,encoding='sam_binary')
                    vs,ams,ss=map(transition_summary,(vl,al,sl));nearest=not bool(np.any(vl==-2))
                    canvas_disagreements+=nearest!=q['sample_line_in_canvas'];nqueries+=1
                    writer.writerow(dict(id=fid,split=row['split'],phase_index=f['phase_index'],family=f['family'],query_variant=q['query_variant'],support_id=q['support_id'],native_corner=q['native_corner'],
                        renderer_visible_transition=vs['transition_class'],renderer_amodal_transition=ams['transition_class'],SAM_transition=ss['transition_class'],
                        renderer_visible_target_samples=vs['target_samples'],renderer_visible_background_samples=vs['background_samples'],renderer_ambiguous_samples=vs['ambiguous_samples'],outside_samples=vs['outside_samples'],SAM_target_samples=ss['target_samples'],
                        amodal_target_visible_nontarget_samples=int(np.sum((al==1)&(vl==0))),SAM_target_renderer_visible_background_samples=int(np.sum((sl==1)&(vl==0))),
                        nearest_pixel_line_in_canvas=nearest,continuous_line_in_canvas=q['sample_line_in_canvas']))
                    for population in ('all',row['split']):
                        group=counts[population+'/'+f['family']+'/'+q['query_variant']]
                        group['profiles']+=1
                        for label,stat in [('renderer_visible',vs),('renderer_amodal',ams),('SAM',ss)]:
                            group[label+'/'+stat['transition_class']]+=1
                            group[label+'/ambiguous_samples']+=stat['ambiguous_samples'];group[label+'/outside_samples']+=stat['outside_samples']
                        group['SAM_target_renderer_visible_background_samples']+=int(np.sum((sl==1)&(vl==0)))
                        group['amodal_target_visible_nontarget_samples']+=int(np.sum((al==1)&(vl==0)))
    if len(inventory)!=1003 or nqueries!=139976:raise ValueError('Original full mask/frame/query denominator changed')
    with gzip.open(out/'MASK_INPUT_INVENTORY.jsonl.gz','wt')as stream:
        for r in inventory:stream.write(json.dumps(r,allow_nan=False)+'\n')
    write(out/'B_MASK_TRANSITIONS.json',dict(schema='ORIGINAL_MASK_QUERY_TRANSITION_DIAGNOSTIC_V1',status='COMPLETE',frames=1003,B_input_frames=673,
        phase_family_variant_profiles=nqueries,groups={k:dict(v)for k,v in counts.items()},
        continuous_vs_nearest_pixel_canvas_disagreements=canvas_disagreements,
        semantics='Target object membership transitions, never RGB brightness transitions; mask source assertions are not independent physical-owner verification',
        renderer_encoding='Exactly0=background,255=target;1..254 ambiguous; no thresholding or inferred face/edge ownership',
        SAM_encoding='Original bound binary clean mask0/1; predicted membership is not physical ground truth',
        sampling='Q00 exact archived sample_pixel_xy; moved-query masks sampled with original floor(x+.5); no RGB distribution or optical cost moved',
        actual_material_edge_correspondence='UNVERIFIED',new_automatic_type_supplier=False,production_bank_insertions=0,
        original_RGB_read=False,new_forward=0,new_solver_calls=0,new_optical_cost_evaluations=0))
    write(out/'B_MASK_DIAGNOSTIC_SEAL.json',dict(status='COMPLETE',plan_sha256=sha(a.plan),source_sha256=sha(__file__),
        original_query_seal_sha256=sha(a.query_dir/'B_QUERY_FACTORIAL_SEAL.json'),metadata_sha256=sha(a.metadata),
        artifacts={p.name:sha(p)for p in out.iterdir()if p.is_file()},wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,new_model_forward=0,new_solves=0,new_rendering=0))


if __name__=='__main__':main()
