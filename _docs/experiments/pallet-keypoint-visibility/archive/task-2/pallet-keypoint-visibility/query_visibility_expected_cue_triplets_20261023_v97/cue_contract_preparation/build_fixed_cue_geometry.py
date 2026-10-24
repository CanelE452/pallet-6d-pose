"""Materialize only8 fixed GT cue geometries and bounded rays from saved mesh."""
from pathlib import Path
from collections import Counter
import importlib.util,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def clean(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:clean(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v)for v in x]
    return x
def save(p,x):
    with Path(p).open('x')as f:json.dump(clean(x),f,indent=2,allow_nan=False);f.write('\n')
def main():
    contract=read(HERE/'CUE_OBSERVABILITY_CONTRACT_BEFORE_NEW_RENDER.json')
    for k in ['teacher_code','generated_proof','frozen_source_associations','V96_appearance_protocol','V93_geometry','V93_inventory']:assert sha(contract[k]['path'])==contract[k]['sha256']
    assert read(contract['generated_proof']['path'])['PASS']
    spec=importlib.util.spec_from_file_location('fixed_expected_cue_teacher',contract['teacher_code']['path']);T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
    refs=read(contract['frozen_source_associations']['path'])['references'];camera=read(contract['V96_appearance_protocol']['path'])['camera'];inventory=read(contract['V93_inventory']['path'])
    with np.load(contract['V93_geometry']['path'],allow_pickle=False)as z:
        local=z['vertices_local'];world=z['vertices_world_m'];offsets=z['mesh_vertex_offsets'];matrices=z['mesh_local_to_world_row'];triangles=z['triangles_global'];part=z['triangle_asset_local_part_id']
    orientations=np.array([1 if m['USD_orientation']=='rightHanded'else-1 for m in inventory['per_mesh']])[part];two=np.array([m['double_sided']for m in inventory['per_mesh']])[part]
    scene=T.RayScene(world[triangles],part,orientations,two,foreign_owner_min=308);cues=[];results=[]
    for r in refs:
        pid=r['association_part_id'];e=T.make_endpoint(local[offsets[pid]:offsets[pid+1]],matrices[pid],r['association_world_vertex'],r['endpoint_patch_radius_world_proposal'])
        cue=dict(aux_corner_index=r['aux_corner_index'],part_id=pid,original_q=r['original_aux_q'],L=r['L'],geometry=e,native8_mapping='UNKNOWN',human_gold=False)
        result=T.evaluate_endpoint(cue,camera,scene)
        cues.append(cue);results.append(dict(aux_corner_index=r['aux_corner_index'],part_id=pid,**result))
    assert len(cues)==8 and[c['aux_corner_index']for c in cues]==list(range(8))
    assert all(c['original_q']==r['original_aux_q']and c['L']==r['L']for c,r in zip(cues,refs))
    save(HERE/'CUE_GEOMETRY_GT_ONLY.json',dict(completed=True,contract=bind(HERE/'CUE_OBSERVABILITY_CONTRACT_BEFORE_NEW_RENDER.json'),camera=camera,references=cues,code=bind(__file__),training_admission=False))
    keys=['endpoint_world','edge_vectors','edge_lengths','face_normals','face_patch_quads_world','face_probe_points','edge_probe_points','face_edge_axes','local_endpoint_sign']
    np.savez_compressed(HERE/'CUE_GEOMETRY_GT_ONLY.npz',**{k:np.stack([c['geometry'][k]for c in cues])for k in keys},original_q=np.array([c['original_q']for c in cues]),L=np.array([c['L']for c in cues]),part_id=np.array([c['part_id']for c in cues],np.int32))
    save(HERE/'ACTUAL_TARGET_ONLY_ENDPOINT_RAY_REPLAY.json',dict(completed=True,code=bind(__file__),teacher=contract['teacher_code'],results=results,state_counts=dict(Counter(r['state']for r in results)),source_ray_count=8*37,
        source_appearance_RGB_reread=0,new_renders_models_fit_private_current_PnP=0,training_admission=False))
    target=results[7];assert target['state']=='D','Predeclared AUX7 must be reported rather than replaced if not direct'
    compact=dict(completed=True,contract=bind(HERE/'CUE_OBSERVABILITY_CONTRACT_BEFORE_NEW_RENDER.json'),teacher_code=contract['teacher_code'],cue_geometry=bind(HERE/'CUE_GEOMETRY_GT_ONLY.json'),cue_geometry_npz=bind(HERE/'CUE_GEOMETRY_GT_ONLY.npz'),target_only_replay=bind(HERE/'ACTUAL_TARGET_ONLY_ENDPOINT_RAY_REPLAY.json'),camera=camera,
        references=[dict(aux_corner_index=c['aux_corner_index'],original_aux_q=c['original_q'],L=c['L'])for c in cues],
        fixed_target_aux7=dict(aux_corner_index=7,part_id=307,projected_support_disk_center_xy=target['projected_support_disk']['center'],radius_px=target['projected_support_disk']['radius'],min_patch_camera_z=target['all_patch_camera_z_min'],all_patch_projected_vertices=target['all_patch_projected_vertices'],target_only_state=target['state']),
        caveat='Disk bounds placement only. Patch/edge/endpoint teacher, not disk or whole-part mask, determines cue state. q/L exact; nativeUNKNOWN, humanfalse. No newrender authorized by this metadata alone.')
    save(HERE/'CUE_RENDERER_INPUT_BINDING.json',compact)
    print(json.dumps(clean(dict(results=[{k:r[k]for k in ['aux_corner_index','state','reason','front_facing_faces','original_direct_witness_faces']}for r in results],fixed_target_aux7=compact['fixed_target_aux7'])),indent=2))
if __name__=='__main__':main()
