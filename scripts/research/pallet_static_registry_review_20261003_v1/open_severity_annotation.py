"""The user's existing annotate.py on rectangular and separately shot square eval RGB."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'scripts/research/pallet_lifter_case_review_20261003_v1'))
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_lifter_case_review_20261003_v1.open_severity_annotation import SeverityReview, LABELS
from open_existing_annotation import write_json

MANIFEST = ROOT/'_docs/experiments/pallet_static_registry_review_20261003_v1/review/STATIC_REVIEW_MANIFEST.json'
OUT = ROOT/'data/pallet/results/pallet_static_registry_review_20261003_v1/native_severity_review'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_context():
    manifest = json.loads(MANIFEST.read_text())
    frames, images = {}, {}
    for case in manifest['cases']:
        path = ROOT/case['image']['path']
        if not path.is_file() or sha256(path) != case['image']['sha256']:
            raise ValueError('Image missing/hash mismatch: '+str(path))
        fid = case['case_id']
        frames[fid] = dict(frame_id=fid, session_id=case['session'],
            saved_frame_index=int(case['frame_id'].rsplit(':',1)[-1]),
            image_sha256=case['image']['sha256'], population=case['population'],
            prior_severity=copy.deepcopy(case['frame_severity']))
        images[fid] = path
    return SimpleNamespace(frames=frames, images=images,
        bindings={'static_manifest_sha256':sha256(MANIFEST), 'source_bindings':manifest['source_bindings']})


class StaticReview(SeverityReview):
    schema = 'static_native_direct_severity_review_v1'
    rubric = 'human_direct_three_class_static_v1'
    window_title = 'Annotate - Rectangular and Square evaluation occlusion'

    def __init__(self, context, path):
        super().__init__(context, path)
        self.population = 'GREEN0918'
        self.display_title = '정사각형 평가 원본'
        self.original = {fid:dict(severity=row['prior_severity']['status'], source=row['prior_severity'])
            for fid,row in context.frames.items() if row['prior_severity']['locked']}

    def record_for_display(self, fid):
        return self.store['records'].get(fid) or self.original.get(fid)

    def counts(self):
        result = {key:0 for key in ('clean','moderate','severe','unreviewed')}
        for fid,row in self.ctx.frames.items():
            if row['population'] != self.population: continue
            record = self.record_for_display(fid)
            result[record['severity'] if record else 'unreviewed'] += 1
        return result

    def contexts(self, manifest, cli_args, registry, repo):
        sessions, contexts = [], {}
        for population, name, geometry in [('GREEN0918','01_SQUARE_119','plastic_square'),
                                           ('DEV319','02_RECTANGULAR_319','plastic_rectangular')]:
            rows = [row for row in self.ctx.frames.values() if row['population']==population]
            pending = [i for i,row in enumerate(rows) if not self.record_for_display(row['frame_id'])]
            if pending:
                i=pending[0]; rows=rows[i:]+rows[:i]
            # The geometry here initializes the raw viewer only; pose is disabled.
            try: spec=registry.resolve(geometry)
            except (KeyError, ValueError): spec=registry.resolve('plastic_square')
            args=copy.copy(cli_args)
            args.object_type=spec.object_type; args.population_role='DEV'; args.default_split='eval'
            args.capture_session_id=population; args.lighting_condition=None
            args.intrinsics_quality='UNKNOWN'; args.intrinsics_source='unused: raw severity viewer; PnP OFF'
            output=self.workspace/population
            for row in rows:
                path=output/(self.ctx.images[row['frame_id']].stem+'.json')
                self.path_map[str(path.resolve())]=row['frame_id']
            key='review:'+population
            sessions.append((name,str(output),key))
            contexts[key]=dict(args=args, metadata={'population_role':'DEV','object_type':spec.object_type},
                geometry_spec=spec, out_dir=str(output), K=np.eye(3), K_source=args.intrinsics_source,
                frame_paths=[str(self.ctx.images[row['frame_id']]) for row in rows], frame_count=len(rows),
                writable=True, workspace_scope=None, display_role='DEV', source_session_dir=str(output),
                refresh_evaluation=False, force_explicit_object_type=True, active_evaluation_member=False)
        return sessions,contexts

    def load(self,state,path,read_only=False):
        result=super().load(state,path,read_only)
        self.population=self.ctx.frames[self.current]['population']
        self.display_title='정사각형 평가 원본' if self.population=='GREEN0918' else '직사각형 평가 원본'
        return result

    def choose(self,key,path,source='human_keyboard'):
        super().choose(key,path,source)
        record=self.store['records'][self.current]
        record['dataset_population']=self.population
        record['review_scope']='static_frame_severity_only'
        record['prior_approved_severity']=self.ctx.frames[self.current]['prior_severity']
        if self.current in self.original:
            record['classification_status']='HUMAN_REVISION_RECORDED_ORIGINAL_PRESERVED'
        self.store['history'][-1]['current']=copy.deepcopy(record)
        self.persist()
        write_json(Path(path),dict(schema='static_native_severity_input_v1',record=record))

    def persist(self):
        write_json(self.store_path,self.store)
        groups={population:{key:[] for key in ('clean','moderate','severe','unreviewed')}
                for population in ('DEV319','GREEN0918')}
        for fid,row in self.ctx.frames.items():
            record=self.record_for_display(fid)
            groups[row['population']][record['severity'] if record else 'unreviewed'].append(
                dict(frame_id=fid,image_path=str(self.ctx.images[fid]),
                     source='existing_approved_label' if fid in self.original and fid not in self.store['records'] else 'new_explicit_human_input' if record else 'unreviewed'))
        write_json(self.store_path.parent/'STATIC_SEVERITY_GROUPS.json',dict(
            schema='static_native_severity_groups_v1',input_bindings=self.store['input_bindings'],groups=groups))

    def install(self,editor):
        super().install(editor)
        original_render, original_mouse, original_key=editor.render, editor.on_mouse,editor._handle_click_key
        from PIL import Image,ImageDraw,ImageFont
        font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18)
        def render(state,*args):
            array=original_render(state,*args)
            image=Image.fromarray(array[:,:,::-1]);draw=ImageDraw.Draw(image)
            for i,(label,population) in enumerate([('정사각형 119장','GREEN0918'),('직사각형 319장','DEV319')]):
                box=(976+i*155,5,1124+i*155,43)
                draw.rounded_rectangle(box,radius=6,fill='#276b50' if population==self.population else '#3c495b')
                draw.text((box[0]+8,box[1]+6),label,font=font,fill='white')
            draw.rectangle((976,650,1289,769),fill='#17202b')
            for i,line in enumerate(('위 버튼: 정사각형 / 직사각형',
                    '직사각형: 기존 분류 표시', '1/2/3 변경은 별도 기록으로 저장',
                    '코너 입력·식별자 입력 없음')):
                draw.text((978,650+i*28),line,font=font,fill='#d9e5ee')
            return np.asarray(image)[:,:,::-1].copy()
        def mouse(event,x,y,flags,state):
            if event==editor.cv2.EVENT_LBUTTONDOWN and 5<=y<=43:
                for i in range(2):
                    if 976+i*155<=x<=1124+i*155:
                        state.sess_pick=i
                        return
            original_mouse(event,x,y,flags,state)
        def key(value,state,*args):
            if value in (ord('s'),ord('S'),13,10) and self.record_for_display(self.current):
                return 'next'
            return original_key(value,state,*args)
        editor.render,editor.on_mouse,editor._handle_click_key=render,mouse,key


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    context=load_context();native=StaticReview(context,OUT/'STATIC_SEVERITY_INPUTS.json')
    print(json.dumps(dict(status='READY_STATIC_EVAL_RGB',rectangular=319,square=119,
        square_counts=native.counts(),original_rectangular_labels=len(native.original)),ensure_ascii=False),flush=True)
    if args.prepare_only:return
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'native_editor.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('정적 평가 이미지 창이 이미 열려 있습니다.')
    sys.path.insert(0,str(ROOT/'scripts/annotate'))
    import annotate,annotate_review
    native.install(annotate);annotate_review.load_review_contexts=native.contexts
    annotate.main(['--review-manifest',str(MANIFEST),'--stride','1','--population-role','DEV',
        '--default_split','eval','--object-type','plastic_square','--geometry-registry',
        str(ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json'),'--win-w','1290','--win-h','830'])


if __name__=='__main__':main()
