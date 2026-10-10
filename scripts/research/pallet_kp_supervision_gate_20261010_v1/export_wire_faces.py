"""Export retained physical ownership witness geometry; no casts or models."""
from pathlib import Path
import gzip,json,hashlib
import numpy as np
P=Path(__file__).parent;M=Path('/dev/shm/pallet-observation-private-20261009/actual_mesh/scene.usd.npz');R=P/'CORRECTED_TARGET_PROPOSAL_ROWS.jsonl.gz'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 protocol=P/'WIRE_FACE_WITNESSES_PROTOCOL.json';assert not protocol.exists()
 protocol.write_text(json.dumps(dict(schema='existing_actual_wire_face_export_protocol_v1',before_export=True,mesh_sha256=sha(M),proposal_rows_sha256=sha(R),code_sha256=sha(Path(__file__)),operation='Only copy normalized float64 actual vertex coordinates and triangle indices used by selected ownership face pairs; provide exact shared-coordinate wire endpoints. No new rays or owner selection.',new_rays=0,new_models=0,new_fits=0,new_training=0,new_RGB=0),indent=2)+'\n')
 rows=[json.loads(x) for x in gzip.open(R,'rt')];selected=[x['selected'] for x in rows if x['proposed_target']=='POSITIVE'];pids=sorted({p for c in selected for p in c['noncoplanar_face_witness']['triangles']});wires={c['wire_id']:c for c in selected}
 with np.load(M) as z:v=z['vertices'];tri=z['triangles']
 artifact=dict(schema='normalized_actual_triangle_wire_face_witnesses_v1',mesh_sha256=sha(M),proposal_rows_sha256=sha(R),protocol_sha256=sha(protocol),code_sha256=sha(Path(__file__)),
  normalized_actual_faces=[dict(triangle_id=p,original_vertex_ids=tri[p].tolist(),normalized_actual_vertex_coords=v[tri[p]].tolist()) for p in pids],
  normalized_actual_wires=[dict(wire_id=k,original_representative_endpoint_vertex_ids=c['actual_endpoint_vertex_ids'],normalized_endpoint_coords=v[c['actual_endpoint_vertex_ids']].tolist(),ownership_face_ids=c['noncoplanar_face_witness']['triangles']) for k,c in sorted(wires.items())],
  face_count=len(pids),wire_count=len(wires),limits='actual coords extracted from immutable already-composed mesh; USDpart/material identifiers were not retained, ownership is geometric actual shared noncoplanar wire',new_rays=0,new_models=0,new_fits=0,new_training=0,new_RGB=0)
 out=P/'WIRE_FACE_WITNESSES.json';out.write_text(json.dumps(artifact,separators=(',',':'))+'\n');print(json.dumps(dict(face_count=len(pids),wire_count=len(wires),bytes=out.stat().st_size,sha256=sha(out))))
if __name__=='__main__':main()
