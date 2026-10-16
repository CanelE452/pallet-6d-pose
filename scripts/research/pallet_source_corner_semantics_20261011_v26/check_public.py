"""Independently verify published source corner phase, geometry and summaries."""
import argparse
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

READY='_docs/experiments/pallet_kp_supervision_repair_20261010_v1/READY_SOURCE_TARGET_ROWS.jsonl.gz'
CEILING='_docs/experiments/pallet_kp_difficulty_20261010_v1/SOURCE_CEILING_ROWS.jsonl.gz'
V25='_docs/experiments/pallet_point_admission_mechanism_20261011_v25/source/CORNER_ROWS.jsonl.gz'
EDGES=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def read(p):
    with gzip.open(p,'rt') as f:return [json.loads(s) for s in f if s.strip()]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def close(a,b):assert math.isclose(a,b,abs_tol=1e-8,rel_tol=1e-8),(a,b)


def stat(values):
    v=sorted(values);n=len(v);mean=sum(v)/n
    variance=sum((x-mean)**2 for x in v)/n
    def quantile(p):
        x=(n-1)*p;low=int(x);high=min(n-1,low+1)
        return v[low]*(1-(x-low))+v[high]*(x-low)
    return dict(n=n,mean=mean,variance=variance,std=math.sqrt(variance),median=quantile(.5),P90=quantile(.9),max=max(v))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('artifact',type=Path)
    p.add_argument('--checkout',type=Path,required=True);a=p.parse_args();root=a.artifact;repo=a.checkout
    protocol=json.loads((root/'PROTOCOL.json').read_text());checks=json.loads((root/'CHECKS.json').read_text())
    assert checks['passed'];summary=json.loads((root/'SUMMARY.json').read_text())
    for name,digest in checks['output_hashes'].items():assert sha(root/name)==digest
    for key,relative in [('ready',READY),('ceiling',CEILING),('saved_corners',V25)]:
        expected=protocol['inputs'][key];assert sha(repo/relative)==expected['sha256'] and (repo/relative).stat().st_size==expected['bytes']
    source={r['id']:r for r in read(repo/READY) if r['partition']=='calibration'}
    ceiling={r['id']:r for r in read(repo/CEILING) if r['partition']=='calibration'}
    previous={(r['arm'],r['id'],r['corner'],tuple(r['edges'])):r for r in read(repo/V25)}
    frames=read(root/'FRAME_ROWS.jsonl.gz');assert len(frames)==128
    lookup={r['id']:r for r in frames};assert len(lookup)==128 and set(lookup)==set(source)
    scalar_projection_calls=0
    for f in frames:
        s=source[f['id']];assert f['dimensions']==s['dimensions'] and f['K']==s['K']
        assert f['existing_synthetic_R']==s['R'] and f['existing_synthetic_t']==s['t']
        signs=list(itertools.product((-1,1),repeat=3))
        expected=sorted(tuple(sign[j]*s['dimensions'][j]/2 for j in range(3)) for sign in signs)
        xyz=f['canonical_unordered_xyz'];assert sorted(tuple(x) for x in xyz)==expected
        uv=[]
        for x in xyz:
            camera=[sum(s['R'][i][j]*x[j] for j in range(3))+s['t'][i] for i in range(3)]
            h=[sum(s['K'][i][j]*camera[j] for j in range(3)) for i in range(3)]
            uv.append([h[0]/h[2],h[1]/h[2]]);scalar_projection_calls+=1
        ids=[min(range(8),key=lambda k:math.dist(q,uv[k])) for q in f['annotation_corner_xy']]
        assert ids==f['native_corner_phase_indices'] and len(set(ids))==8
        for q,k,projected,error in zip(f['annotation_corner_xy'],ids,f['projected_native_corner_xy'],f['source_phase_error_px']):
            close(math.dist(projected,uv[k]),0);close(math.dist(q,uv[k]),error);assert error<.05
        b=ceiling[f['id']]['source_annotation'];assert f['source_annotation_binding']==dict(relative_path=b['path'],bytes=b['bytes'],sha256=b['sha256'])
        assert not f['native_N3_predicted_H_reconstructed'] and f['physical_corner_ownership']=='UNKNOWN'
    rows=read(root/'CORNER_ROWS.jsonl.gz');assert len(rows)==573
    assert {(r['arm'],r['id'],r['corner'],tuple(r['edges'])) for r in rows}==set(previous)
    for r in rows:
        k=(r['arm'],r['id'],r['corner'],tuple(r['edges']));old=previous[k];f=lookup[r['id']]
        assert r['saved_wire_reference_xy']==old['reference_xy'] and r['predicted_xy']==old['predicted_xy']
        assert r['existing_source_canonical_projection_xy']==f['projected_native_corner_xy'][r['corner']]
        close(math.dist(r['saved_wire_reference_xy'],r['existing_source_canonical_projection_xy']),r['reference_to_canonical_px'])
        close(math.dist(r['predicted_xy'],r['existing_source_canonical_projection_xy']),r['error_to_existing_source_canonical_px'])
        close(math.dist(r['saved_wire_reference_xy'],r['existing_annotation_corner_xy']),r['reference_to_annotation_px'])
        close(abs(r['error_to_existing_source_canonical_px']-r['saved_error_to_wire_reference_px']),r['error_change_abs_px'])
        assert r['error_change_abs_px']<=r['reference_to_canonical_px']+1e-8
        assert r['saved_error_to_wire_reference_px']==old['saved_corner_error_px']
        assert r['decoder_selected']==old['decoder_selected'] and r['existing_point_admission_accepted']==old['existing_point_admission_accepted']
        for witness in r['support_witnesses']:
            assert all(source[r['id']]['queries'][q]['proposed_target']=='POSITIVE' for q in witness['queries'])
            assert witness['canonical_edge_endpoint_xy']==[f['projected_native_corner_xy'][k] for k in EDGES[witness['edge']]]
            support=[source[r['id']]['queries'][q]['actual_target_uv'] for q in witness['queries']]
            assert support==witness['actual_target_uv']
            A,B=witness['canonical_edge_endpoint_xy'];dx=B[0]-A[0];dy=B[1]-A[1];length=math.hypot(dx,dy)
            for q,saved in zip(support,witness['certified_support_canonical_line_residual_px']):
                residual=(-dy*(q[0]-A[0])+dx*(q[1]-A[1]))/length;close(residual,saved)
        assert r['physical_corner_ownership']=='UNKNOWN' and not r['real_pose_accuracy_evaluated']
    for arm,groups in summary['groups'].items():
        rr=[r for r in rows if r['arm']==arm]
        values={name:[r[name] for r in rr] for name in ('reference_to_canonical_px','reference_to_annotation_px')}
        values['reference_error_change_abs_px']=[r['error_change_abs_px'] for r in rr]
        values['canonical_line_support_abs_residual_px']=[abs(v) for r in rr for s in r['support_witnesses'] for v in s['certified_support_canonical_line_residual_px']]
        for name,v in values.items():
            for field,value in stat(v).items():assert math.isclose(value,groups[name][field],abs_tol=1e-15,rel_tol=1e-10),(name,field)
        assert sum(r['saved_error_to_wire_reference_px']>8 for r in rr)==groups['old_reference_gt8_count']
        assert sum(r['error_to_existing_source_canonical_px']>8 for r in rr)==groups['canonical_reference_gt8_count']
    unit=json.loads((root/'EXECUTION_UNIT_CLARIFICATION.json').read_text())
    assert unit['source_frames_projected']==len(frames)==128 and unit['scalar_project_function_calls_original']==scalar_projection_calls==1024
    print(json.dumps(dict(passed=True,source_frames=128,source_corners=573,upstream_READY_ceiling_V25_hash_verified=True,
        scalar_projection_calls_this_reader=scalar_projection_calls,original_labels_reauthenticated=False,
        fresh_clone=False,real_physical_ownership_certified=False,new_PnP=0,new_real_pose_accuracy=0)))


if __name__=='__main__':main()
