"""Isolated, explicit candidate lists for manual real-support annotation.

No prediction coordinates are loaded, no legacy labels are imported, and no
canonical evaluation workspace is mutated. Callers must disable legacy fallback.
"""
import copy
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
import numpy as np

OUTPUT_REL = 'outputs/pallet_type_selftrain_v1/large_corner_recovery_v1/real_support_annotations'


def load_review_contexts(manifest, cli_args, registry, repo):
    repo=Path(repo).resolve()
    path=Path(manifest)
    if not path.is_absolute():path=repo/path
    records=json.loads(path.read_text())
    if not isinstance(records,list) or not records:
        raise ValueError('review manifest must be a nonempty candidate list')
    output=(repo/OUTPUT_REL).resolve()
    if not output.is_relative_to(repo/'outputs'):
        raise ValueError('review output may not escape isolated outputs directory')
    grouped=OrderedDict();seen=set();role_groups={}
    for row in records:
        role=row['proposed_role']
        if role not in {'proposed_support','proposed_validation'}:
            raise ValueError('unknown review role')
        image=(repo/row['image']['path']).resolve();session=(repo/row['session']).resolve()
        if not image.is_relative_to(repo) or not image.is_relative_to(session/'rgb'):
            raise ValueError('review image must stay inside its repository RGB session')
        digest=hashlib.sha256(image.read_bytes()).hexdigest()
        if digest!=row['image']['sha256'] or digest in seen:
            raise ValueError('review image hash mismatch or duplicate')
        seen.add(digest)
        gid=row['recording_id']
        if gid in role_groups and role_groups[gid]!=role:
            raise ValueError('support/validation recording overlap')
        role_groups[gid]=role
        grouped.setdefault((role,str(session)),[]).append(row)
    contexts={};sessions=[]
    for i,((role,seq),rows) in enumerate(grouped.items(),1):
        name=f'{i:02d}_{"SUPPORT" if role=="proposed_support" else "VALIDATION"}_{Path(seq).name}'
        key=f'review:{name}';args=copy.copy(cli_args)
        types={r['object_type'] for r in rows}
        if len(types)!=1:raise ValueError('mixed object types in review session')
        spec=registry.resolve(next(iter(types)))
        args.object_type=spec.object_type;args.population_role='DEV'
        args.default_split='train' if role=='proposed_support' else 'eval'
        args.capture_session_id=Path(seq).name
        args.lighting_condition='night' if Path(seq).parent.name=='night' else 'day'
        args.intrinsics_quality='UNKNOWN'
        camera=Path(seq)/'cam_K.txt';K=np.loadtxt(camera).reshape(3,3)
        for row in rows:
            if not np.allclose(K,np.array(row['K']),rtol=0,atol=1e-6):
                raise ValueError('review camera does not match frozen candidate camera')
        args.intrinsics_source=str(camera.relative_to(repo))
        out=(output/name).resolve()
        if not out.is_relative_to(output):raise ValueError('unsafe review output')
        frames=[str((repo/r['image']['path']).resolve()) for r in rows]
        contexts[key]=dict(args=args,metadata=dict(population_role='DEV',object_type=spec.object_type,
            lighting=args.lighting_condition,source_capture_session=Path(seq).name,review_role=role),
            geometry_spec=spec,out_dir=str(out),K=K,K_source=str(camera),frame_paths=frames,frame_count=len(frames),
            writable=True,workspace_scope=None,display_role='DEV',source_session_dir=seq,
            refresh_evaluation=False,force_explicit_object_type=True,active_evaluation_member=True)
        sessions.append((name,seq,key))
    return sessions,contexts
