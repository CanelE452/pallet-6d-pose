"""Generic exact frozen-payload reuse staging; mismatches never qualify."""
from pathlib import Path
from collections import Counter
import shutil
from . import fullnormal_batch as K,qcov as Q,calibration as CAL


def stage_exact_reuse(prior_run,packet_root,output,source_roots,*,arms=('C0','CP','CB'),source_arms=None,packet_seal=None):
    """Copy only rows whose complete frozen kernel payload/source still match."""
    prior_run,packet_root,output=map(Path,(prior_run,packet_root,output));output.mkdir(parents=True,exist_ok=True)
    from .fullnormal_parallel import verify_packet_bindings
    packet_seal=Path(packet_seal)if packet_seal is not None else packet_root/('PACKET_SEAL.json'if(packet_root/'PACKET_SEAL.json').exists()else'PACKET_MANIFEST.json')
    expected_frames={domain:CAL.load_bound_frames(prior_run,domain)[0]for domain in ('real','synthetic')}
    _,packet_validation=verify_packet_bindings(packet_root,packet_seal,expected_frames)
    original_config,_,_=K.D1._cpu_models();original_config=K.B.configuration(original_config)
    feature_config=Q.read(prior_run.parent/'pallet-statewise-end-to-end-recovery-v1'/'CONFIG_FROZEN.json')
    inventory=Q.read(prior_run/'P0_INPUT_CONTRACT.json');binding=K.source_binding();rows=[];excluded=[]
    for domain in ('real','synthetic'):
        frames,_=CAL.load_bound_frames(prior_run,domain)
        for ordinal,frame in enumerate(frames):
            existing=[(arm,source_arm,root,Path(root)/source_arm/domain/f'{ordinal:04d}')for arm in arms
                for source_arm in (source_arms if source_arms is not None else (arm,))for root in source_roots
                if (Path(root)/source_arm/domain/f'{ordinal:04d}'/'FRAME_SEAL.json').exists()]
            if not existing:continue
            observation_folder=Path(inventory[domain]['inventory'][ordinal]['observation_folder'])
            observation=K.jsonrow(observation_folder/'OBSERVATION.jsonl.gz');mask_sha=Q.sha(observation_folder/'MASK.npz')
            packet=Q.read(packet_root/'packets'/domain/f'{ordinal:04d}.json.gz');used=set()
            for arm,source_arm,root,source in existing:
                if arm in used:continue
                signature=Q.P.digest(K.cache_payload(frame,packet['selections'][arm],observation,mask_sha,original_config,feature_config))
                seal=Q.read(source/'FRAME_SEAL.json')
                if seal['cache_payload_sha256']!=signature or seal['source_binding']!=binding:
                    excluded.append(dict(domain=domain,ordinal=ordinal,arm=arm,source=str(source),reason='NUMERICAL_PAYLOAD_OR_SOURCE_DIFFERS'));continue
                destination=output/arm/domain/f'{ordinal:04d}';destination.mkdir(parents=True,exist_ok=True)
                for name,digest in seal['artifacts'].items():
                    if Q.sha(source/name)!=digest:raise ValueError('Previous attempt row changed')
                for name in (*seal['artifacts'],'FRAME_SEAL.json'):shutil.copy2(source/name,destination/name)
                rows.append(dict(domain=domain,ordinal=ordinal,arm=arm,source_arm=source_arm,source=str(source),source_frame_seal_sha256=Q.sha(source/'FRAME_SEAL.json'),
                    desired_cache_payload_sha256=signature,exact_dependency_and_artifact_equality_verified=True));used.add(arm)
    report=dict(schema='S2_UNIT_WEIGHT_EXACT_PRIOR_EXECUTION_REUSE_V1',rows=rows,eligible_n=len(rows),
        excluded=excluded,excluded_n=len(excluded),eligible_counts=dict(Counter(row['arm']+'/'+row['domain']for row in rows)),
        source_roots=[str(r)for r in source_roots],source_sha256=Q.sha(__file__),new_pose_optimizer_calls=0,
        packet_seal_sha256=Q.sha(packet_seal),packet_validation=packet_validation,original_receipts_preserved=True,
        driver_reuse_slot='reuse-pilot; staging includes explicitly identified old production and fixed pilot rows',
        numerical_source_binding=binding)
    Q.write(output/'REUSE_MANIFEST.json',report);return report

