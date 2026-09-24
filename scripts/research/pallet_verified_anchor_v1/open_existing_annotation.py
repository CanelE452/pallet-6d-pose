"""Open the fixed 18 anchors in the existing annotate.py, with normal PnP.

User-revised workflow: keypoints first, status review later. No old reference
coordinates or predictions are imported. PnP-assisted labels are not blind GT.
"""
import argparse
import copy
from collections import OrderedDict
from pathlib import Path
import sys
import numpy as np
from . import common as C

WORK = C.OUT/'keypoints_first'

def contexts(manifest, cli_args, registry, repo):
    assert str(manifest)==str(C.RAW/'ANCHOR_SELECTION.json')
    selection=C.read(C.RAW/'ANCHOR_SELECTION.json')
    assert C.sha(C.RAW/'ANCHOR_SELECTION.json')==C.read(C.DOC/'INPUT_BINDINGS.json')['selection_sha256']
    split=C.read(C.SPLIT);records={r['id']:r for r in split['heldout']}
    groups=OrderedDict();mapping=[]
    for row in selection['frames']:
        image=C.ROOT/row['image']['path'];assert C.sha(image)==row['image']['sha256']
        # Camera metadata only; do not use objects/manual/projected coordinates.
        source=records[row['frame_id']]['annotation']
        assert C.sha(C.ROOT/source['path'])==source['sha256']
        intrinsics=C.read(C.ROOT/source['path'])['camera_data']['intrinsics']
        k=(intrinsics['fx'],intrinsics['fy'],intrinsics['cx'],intrinsics['cy'])
        groups.setdefault(k,[]).append(row)
    sessions=[];result={};spec=registry.resolve('plastic')
    for i,(k,rows) in enumerate(groups.items(),1):
        name=f'ANCHOR18_camera{i}_{len(rows)}frames';key=f'review:{name}';seq=WORK/name
        out=seq/'annotations';args=copy.copy(cli_args)
        args.object_type=spec.object_type;args.population_role='DEV';args.default_split='eval'
        args.capture_session_id=name;args.lighting_condition=None
        args.intrinsics_quality='UNKNOWN';args.intrinsics_source='frozen anchor camera metadata; see WORKSPACE.json'
        K=np.array([[k[0],0,k[2]],[0,k[1],k[3]],[0,0,1.]])
        frames=[str(C.ROOT/r['image']['path']) for r in rows]
        assert len({C.ROOT/r['image']['path'] for r in rows})==len(rows)
        assert len({Path(p).stem for p in frames})==len(frames)
        result[key]=dict(args=args,metadata=dict(population_role='DEV',object_type=spec.object_type,
                    review_role='verified_anchor_keypoints_first',reference_protocol='HUMAN_PNP_ASSISTED_NOT_BLIND'),
            geometry_spec=spec,out_dir=str(out),K=K,K_source=args.intrinsics_source,
            frame_paths=frames,frame_count=len(frames),writable=True,workspace_scope=None,
            display_role='DEV',source_session_dir=str(seq),refresh_evaluation=False,
            force_explicit_object_type=True,active_evaluation_member=True)
        sessions.append((name,str(seq),key))
        for row,path in zip(rows,frames):
            mapping.append(dict(frame_id=row['frame_id'],image=row['image'],severity=row['severity'],
                recording=row['recording'],source_annotation=records[row['frame_id']]['annotation'],
                output_annotation=str((out/(Path(path).stem+'.json')).relative_to(C.ROOT)),K=K.tolist()))
    assert len(mapping)==18
    payload=dict(protocol='USER_REVISED_KEYPOINTS_FIRST_THEN_STATUS',
        selection_sha256=C.sha(C.RAW/'ANCHOR_SELECTION.json'),frames=mapping,
        original_GT_modified=False,model_predictions_imported=False,legacy_coordinates_imported=False,
        PnP_enabled=True,first_pass_model_reference_blind=False,
        note='Existing annotate.py with PnP assistance. Subsequent D/V etc separate. Do not promote PnP-filled points to direct-visible GT.')
    dest=WORK/'WORKSPACE.json'
    if dest.exists():assert C.read(dest)==payload
    else:C.save_new(dest,payload)
    return sessions,result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    sys.path.insert(0,str(C.ROOT/'scripts/annotate'))
    import annotate
    import annotate_review
    from object_geometry_registry import load_object_geometry_registry,DEFAULT_REGISTRY_PATH
    if args.prepare_only:
        sessions,result=contexts(str(C.RAW/'ANCHOR_SELECTION.json'),argparse.Namespace(),
            load_object_geometry_registry(DEFAULT_REGISTRY_PATH),C.ROOT)
        assert sum(len(c['frame_paths']) for c in result.values())==18
        assert all(Path(c['out_dir']).is_relative_to(WORK) for c in result.values())
        print('READY: existing annotate.py; 18 fixed frames; PnP enabled; isolated outputs',[(s[0],len(result[s[2]]['frame_paths'])) for s in sessions]);return
    # Process-local adapter only. Existing annotate.py and its PnP/save logic are unchanged.
    annotate_review.load_review_contexts=contexts
    annotate.main(['--review-manifest',str(C.RAW/'ANCHOR_SELECTION.json'),'--stride','1',
                   '--population-role','DEV','--default_split','eval','--object-type','plastic'])

if __name__=='__main__':main()
