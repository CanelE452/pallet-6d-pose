"""Open the existing GREEN150 raw images for minimal native severity review."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research.pallet_static_registry_review_20261003_v1.open_severity_annotation import StaticReview,sha256
from scripts.research.pallet_lifter_case_review_20261003_v1.open_severity_annotation import SeverityReview
from open_existing_annotation import write_json

SNAPSHOT=ROOT/'_docs/paper/final_dimension_v1/green150_saved_labels_v1/DATASET_SNAPSHOT.json'
OUT=ROOT/'data/pallet/results/pallet_static_registry_review_20261003_v1/existing_square150_severity_review'


def load_context():
    snapshot=json.loads(SNAPSHOT.read_text())
    frames,images={},{}
    for row in snapshot['records']:
        image=ROOT/row['image']['path']
        if not image.is_file() or sha256(image)!=row['image']['sha256']:
            raise ValueError('Raw image missing/changed: '+row['id'])
        annotation=ROOT/row['annotation']['path']
        if not annotation.is_file() or sha256(annotation)!=row['annotation']['sha256']:
            raise ValueError('Frozen annotation missing/changed: '+row['id'])
        fid='GREEN150::'+row['id']
        frames[fid]=dict(frame_id=fid,session_id=row['session'],saved_frame_index=int(image.stem),
            image_sha256=row['image']['sha256'],population='GREEN150',
            prior_severity=dict(status='UNREVIEWED',locked=False,source=None),
            existing_corner_annotation=row['annotation'])
        images[fid]=image
    if len(frames)!=150:raise ValueError('Expected exactly150 frozen images')
    return SimpleNamespace(frames=frames,images=images,bindings={'green150_snapshot_sha256':sha256(SNAPSHOT)})


class ExistingSquareReview(StaticReview):
    schema='green150_native_direct_severity_review_v1'
    window_title='Annotate - Existing square GREEN150 occlusion 1 2 3'
    display_title='기존 정사각형 · 총150장'

    def __init__(self,context,path):
        super().__init__(context,path)
        self.population='GREEN150'
        self.display_title='기존 정사각형 · 총150장'

    def contexts(self,manifest,cli_args,registry,repo):
        sessions,contexts=[],{}
        spec=registry.resolve('plastic_square')
        names=list(dict.fromkeys(row['session_id'] for row in self.ctx.frames.values()))
        # Show sessions with pending labels first, preserving membership and all raw frames.
        names.sort(key=lambda name:all(self.record_for_display(fid) for fid,row in self.ctx.frames.items() if row['session_id']==name))
        for number,name in enumerate(names):
            rows=[row for row in self.ctx.frames.values() if row['session_id']==name]
            pending=[i for i,row in enumerate(rows) if not self.record_for_display(row['frame_id'])]
            if pending:
                i=pending[0];rows=rows[i:]+rows[:i]
            args=copy.copy(cli_args)
            args.object_type=spec.object_type;args.population_role='DEV';args.default_split='eval'
            args.capture_session_id=name;args.lighting_condition=None
            args.intrinsics_quality='UNKNOWN';args.intrinsics_source='unused: severity-only; PnP disabled'
            output=self.workspace/name
            for row in rows:
                # Same numeric image names recur across captures: keep outputs session-separated.
                path=output/(self.ctx.images[row['frame_id']].stem+'.json')
                self.path_map[str(path.resolve())]=row['frame_id']
            key='review:GREEN150:'+name
            sessions.append((f'{number+1:02d}_{name}',str(output),key))
            contexts[key]=dict(args=args,metadata={'population_role':'DEV','object_type':spec.object_type},
                geometry_spec=spec,out_dir=str(output),K=np.eye(3),K_source=args.intrinsics_source,
                frame_paths=[str(self.ctx.images[row['frame_id']]) for row in rows],frame_count=len(rows),
                writable=True,workspace_scope=None,display_role='DEV',source_session_dir=str(output),
                refresh_evaluation=False,force_explicit_object_type=True,active_evaluation_member=False)
        if len(self.path_map)!=150:raise ValueError('Frame/output collision detected')
        return sessions,contexts

    def load(self,state,path,read_only=False):
        return SeverityReview.load(self,state,path,read_only)

    def persist(self):
        write_json(self.store_path,self.store)
        groups={key:[] for key in ('clean','moderate','severe','unreviewed')}
        for fid,row in self.ctx.frames.items():
            record=self.record_for_display(fid)
            groups[record['severity'] if record else 'unreviewed'].append(
                dict(frame_id=fid,session_id=row['session_id'],image_path=str(self.ctx.images[fid])))
        write_json(self.store_path.parent/'GREEN150_SEVERITY_GROUPS.json',dict(
            schema='green150_native_severity_groups_v1',input_bindings=self.store['input_bindings'],counts=self.counts(),groups=groups))

    def install(self,editor):
        SeverityReview.install(self,editor)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    context=load_context();native=ExistingSquareReview(context,OUT/'GREEN150_SEVERITY_INPUTS.json')
    print(json.dumps(dict(status='READY_EXISTING_SQUARE150',frames=150,
        sessions=len({row['session_id'] for row in context.frames.values()}),counts=native.counts()),ensure_ascii=False),flush=True)
    if args.prepare_only:return
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'native_editor.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('기존 정사각형150장 창이 이미 열려 있습니다.')
    sys.path.insert(0,str(ROOT/'scripts/annotate'))
    import annotate,annotate_review
    native.install(annotate);annotate_review.load_review_contexts=native.contexts
    annotate.main(['--review-manifest',str(SNAPSHOT),'--stride','1','--population-role','DEV',
        '--default_split','eval','--object-type','plastic_square','--geometry-registry',
        str(ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json'),'--win-w','1290','--win-h','830'])


if __name__=='__main__':main()
