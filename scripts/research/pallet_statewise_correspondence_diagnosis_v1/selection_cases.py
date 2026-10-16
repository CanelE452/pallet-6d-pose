"""Two deterministic saved-candidate illustrations after the sealed oracle.

No candidate, fit, image, threshold, or production selector is created.
Only already-oracle-evaluated candidates are recounted with its fixed metric
to package exact numbers omitted from the compact non-example row summaries.
"""
from pathlib import Path
import argparse
import numpy as np

from . import analysis as A
from .common import digest, rows, write
from .oracle import pose_from_native


def prepare(*, folder, source_root, reference_manifest, output):
    folder,source_root,output=Path(folder),Path(source_root),Path(output)
    oracle=folder/'oracle_v1';seal=A.read(oracle/'ORACLE_SEAL.json')
    for name in ('ORACLE_ROWS.jsonl.gz','ORACLE_SUMMARY.json'):
        if A.sha(oracle/name)!=seal['artifacts'][name]:raise ValueError('Original sealed oracle changed')
    if A.sha(reference_manifest)!=seal['reference_manifest_sha256']:raise ValueError('Original fixed reference changed')
    scored=list(rows(oracle/'ORACLE_ROWS.jsonl.gz'));chosen=[]
    for stage in ('family_selection','within_family_final_selection'):
        choices=sorted((r for r in scored if r['partition']!='CAL' and r['earliest_observed_failure_stage']==stage),key=lambda r:r['id'])
        row=choices[0]
        cid=min(row['stages']['feasible' if stage=='family_selection' else 'family']['joint_success_candidate_ids'])
        chosen.append((row,stage,cid))
    ids={r['id'] for r,_,_ in chosen}
    manifest_path=folder/'original_candidate_archive_v1/CANDIDATE_MANIFEST.json';manifest=A.read(manifest_path)
    bank=manifest_path.with_name('ORIGINAL_E_CANDIDATE_BANK.jsonl.gz')
    if (A.sha(manifest_path)!=seal['original_candidate_manifest_sha256'] or
            A.sha(bank)!=manifest['artifacts'][bank.name]):raise ValueError('Original candidate source changed')
    originals={r['id']:r for r in rows(bank) if r['id'] in ids}
    packets_path=folder/'e_trace_v2/TYPED_PACKETS.jsonl.gz';es=A.read(packets_path.with_name('EXTRACTION_SEAL.json'))
    if A.sha(packets_path)!=es['files'][packets_path.name]['sha256']:raise ValueError('Original packet binding changed')
    packets={r['id']:r for r in rows(packets_path) if r['id'] in ids}
    references,reference_bindings,annotations=A.fixed_references(source_root,reference_manifest,scored)
    protocol=A.read(folder/'REPLAY_PROTOCOL.json');index={i:n for n,i in enumerate(protocol['ordered_ids'])}
    cases=[]
    for row,stage,cid in chosen:
        fid=row['id'];original=originals[fid];packet=packets[fid]
        _,family_name,_,sample_index=cid.split('/');sample_index=int(sample_index)
        family=next(f for f in original['stages']['G13']['families'] if f['name']==family_name)
        sample=family['feasible_samples'][sample_index];record=sample['original_record']
        candidate=pose_from_native(record['R'],record['t'],family)
        metric=A.metric(candidate,references[row['partition']][fid])
        if not metric['success_le5']:raise ValueError('Already sealed oracle joint-success membership differs')
        slot_path=folder/f'slots/{index[fid]:03d}_TYPED_MULTI.json';slot=A.read(slot_path)
        matches=[(i,c) for i,c in enumerate(slot['result']['generated_candidates']) if c['stage']=='G13'
            and c['family']==family_name and np.allclose(c['R'],record['R'],atol=1e-9,rtol=0)
            and np.allclose(c['t'],record['t'],atol=1e-9,rtol=0)
            and abs(c['original_endpoint_record'].get('objective',float('inf'))-record['objective'])<=1e-9]
        selected_name=original['original_e']['frozen_family']
        selected_family=next(f for f in original['stages']['G13']['families'] if f['name']==selected_name)
        selected_summary=row['stages']['selected']
        source_image=packet['image']
        if Path(source_image).is_absolute():
            try:source_image=str(Path(source_image).relative_to(source_root))
            except ValueError:source_image=None
        cases.append(dict(id=fid,partition=row['partition'],failure_stage=stage,candidate_id=cid,
            selection_rule='First sorted PRIMARY ID in the frozen earliest-failure category; first lexicographic existing joint-success candidate ID in its eligible retained pool',
            candidate_pose=candidate,candidate_metric=metric,candidate_objective=record.get('objective'),
            candidate_initialization=record.get('initialization'),
            candidate_objective_components=(matches[0][1]['original_endpoint_record'].get('objective_components') if matches else None),
            objective_component_source='Same-family original replay endpoint R/t and objective match at fixed1e-9; not pose recovery from a cost or name',
            objective_component_matching_replay_indices=[i for i,_ in matches],
            selected_e_pose=original['original_e']['actual_pose'],
            selected_e_metric=dict(pose_available=True,translation_cm=selected_summary['translation_min_cm'],
                rotation_deg=selected_summary['rotation_min_deg'],success_le5=selected_summary['joint_success_exists']),
            selected_e_objective=selected_family['original_objective'],
            selected_e_objective_components=selected_family['original_objective_components'],
            selected_family=selected_name,
            G10_family_objectives={f['name']:f['original_objective'] for f in original['stages']['G10']['families']},
            G13_family_objectives={f['name']:f['original_objective'] for f in original['stages']['G13']['families']},
            source=dict(archive_path='original_candidate_archive_v1/ORIGINAL_E_CANDIDATE_BANK.jsonl.gz',
                archive_sha256=A.sha(bank),source_field=f'stages.G13.families[{family_name}].feasible_samples[{sample_index}].original_record',
                family=family_name,native_xyz=family['native_xyz'],Q=family['physical_registration_Q'],
                cf_extents=family['cf_extents'],index=sample_index,raw_feasible_record=record,
                image=source_image,image_sha256=packet['image_sha256'],K=packet['solver_input']['K'],
                typed_packet_sha256=digest(packet),original_candidate_packet_sha256=digest(original),
                oracle_row_sha256=digest(row),typed_multi_slot_sha256=A.sha(slot_path)),
            offline_privileged_oracle_only=True,new_prediction_or_candidate_or_posefit=False,
            metric_recount_scope='Exact same already-evaluated and sealed candidate/metric/reference; compact non-six-case summaries omitted its numeric T/R, now deterministically recounted',
            semantic_corner_owner_relabel_or_GT_fit=False))
    value=dict(schema='TWO_SEALED_ORACLE_SELECTION_LOSS_ILLUSTRATIONS_V1',cases=cases,
        oracle_seal_sha256=A.sha(oracle/'ORACLE_SEAL.json'),reference_bindings=reference_bindings,
        reference_manifest_sha256=seal['reference_manifest_sha256'],
        reused_original_annotations=len(annotations),same_candidate_metric_numeric_recounts=2,
        new_candidates=0,new_posefit_calls=0,new_image_forwards=0,
        warning='Offline privileged reference oracle illustration; not an inference result or new selector improvement')
    write(output,value)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('folder','source-root','reference-manifest','output'):p.add_argument('--'+name,required=True)
    r=prepare(**vars(p.parse_args()))
    print([(c['id'],c['candidate_id'],c['candidate_metric'],c['candidate_objective'],c['selected_e_objective']) for c in r['cases']])
