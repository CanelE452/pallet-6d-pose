"""Open eight RGB-reviewed, upper-face-visible manual annotation candidates.

Original30 workspace is preserved. No GT import, training, or eval promotion.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from . import common as C
from .real_support_review import closure

sys.path.insert(0,str(C.ROOT/'scripts/annotate'))
from annotate_review import load_review_contexts
from object_geometry_registry import load_object_geometry_registry, DEFAULT_REGISTRY_PATH

OUT=C.OUT/'large_corner_recovery_v1/readable_annotation_v1'
MANIFEST=OUT/'CANDIDATES.json'
SESSION='data/pallet/raw_data/capture0403middle'
# RGB-only review of16 time-spread frames; easiest oblique views first.
STEMS=['1775201210359631360','1775201212409274368','1775201214189923584',
       '1775201164702006784','1775201188555311360','1775201198701622272',
       '1775201203001981184','1775201206899281920']
OLD_WORKSPACE=C.OUT/'large_corner_recovery_v1/real_support_annotations/WORKSPACE.json'


def contexts():
    return load_review_contexts(MANIFEST,argparse.Namespace(),
        load_object_geometry_registry(DEFAULT_REGISTRY_PATH),C.ROOT)


def prepare():
    if (OUT/'WORKSPACE.json').exists():
        w=C.read(OUT/'WORKSPACE.json')
        for b in w['sources']:C.verify(b)
        contexts()
        return w
    groups_path=C.ROOT/'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json'
    groups=C.read(groups_path)
    eval_path=C.DOC/'EVAL_PROTOCOL.json';pool_path=C.DOC/'POOL.json'
    evaluation=C.read(eval_path)['records'];pool=C.read(pool_path)
    seeds={str(Path(r['image']['path']).parent.parent) for r in evaluation}
    seeds|={s['session_key'] for g in groups['groups'] if g['recording_id'] in pool['evaluation_recording_ids'] for s in g['sessions']}
    assert SESSION not in closure(seeds,groups)
    gid=next(g['recording_id'] for g in groups['groups'] if any(s['session_key']==SESSION for s in g['sessions']))
    assert gid=='REC_036'
    K=np.loadtxt(C.ROOT/SESSION/'cam_K.txt').reshape(3,3)
    # Only dimensions/camera metadata, NEVER projected coordinates, are imported.
    metadata_path=C.ROOT/SESSION/'gt_manual/1775201155966553600.json'
    meta=C.read(metadata_path)
    assert meta['objects'][0]['dimensions_m']==dict(width=1.1,depth=1.3,height=.11)
    intr=meta['camera_data']['intrinsics']
    assert np.allclose(K,[[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]],atol=1e-6,rtol=0)
    eval_hashes={r['image']['sha256'] for r in evaluation}
    rows=[]
    for stem in STEMS:
        binding=C.bound(C.ROOT/SESSION/'rgb'/f'{stem}.png')
        assert binding['sha256'] not in eval_hashes
        rows.append(dict(id='PLASTIC__'+binding['sha256'],image=binding,
            session=SESSION,recording_id=gid,object_type=C.TYPES['PLASTIC'],kind='PLASTIC',
            K=K.tolist(),proposed_role='proposed_support',
            selection_reason='RGB upper face visible and pallet in frame; no model prediction/error selection',
            warning='Same indoor setting and nearby acquisition as eval_noapril; not independent validation. Training promotion pending label/provenance review.'))
    assert len({r['image']['sha256'] for r in rows})==8
    C.freeze(MANIFEST,rows)
    _,cc=contexts();assert len(cc)==1
    ctx=next(iter(cc.values()));assert ctx['args'].default_split=='train'
    old=C.read(OLD_WORKSPACE)
    old_paths={r['annotation'] for r in old['rows']}
    mapping=[]
    for r in rows:
        target=Path(ctx['out_dir'])/(Path(r['image']['path']).stem+'.json')
        rel=str(target.relative_to(C.ROOT))
        assert rel not in old_paths
        assert not target.exists(),'New workspace would collide with preexisting annotation'
        mapping.append(dict(id=r['id'],image=r['image'],recording_id=gid,
            role='proposed_support',split='train',annotation=rel))
    preserved=[C.bound(C.ROOT/r['annotation']) for r in old['rows'] if (C.ROOT/r['annotation']).exists()]
    w=dict(rows=mapping,original_workspace=C.bound(OLD_WORKSPACE),original_saved_at_setup=preserved,
        status='USER_AUTHORIZED_MANUAL_REVIEW_ONLY',automatic_labels_created=0,training_started=False,
        validation_candidates=0,output_directory=str(Path(ctx['out_dir']).relative_to(C.ROOT)),
        warning='train is a staging label only, not automatic training inclusion. Preserve all original evaluation exclusions. This single indoor recording cannot serve as independent validation.',
        sources=[C.bound(MANIFEST),C.bound(__file__),C.bound(groups_path),C.bound(eval_path),C.bound(pool_path),
                 C.bound(C.ROOT/SESSION/'cam_K.txt'),C.bound(metadata_path)])
    C.freeze(OUT/'WORKSPACE.json',w)
    for b in preserved:C.verify(b)
    C.write_text(OUT/'README.md','# 윗면이 보이는 수동 어노테이션 8장\n\n'
        '사선에서 윗면이 잘 보이는 사진부터 시작합니다. 기존30장 작업 목록 및 저장1장은 보존했습니다.\n\n'
        '- 저장: `s`, 다음 사진: `n`, 이전: `p`.\n'
        '- 직접 확인되는 코너만 클릭합니다. AprilTag 모서리가 아니라 팔레트 모서리입니다.\n'
        '- 가려진 점을 추측해서 찍지 마세요. PnP 자동 투영점은 수동 정답으로 세지 않습니다.\n'
        '- train 표시는 검토용 staging이며 자동 학습 편입이 아닙니다. 검증용으로 섞지 않습니다.\n\n'
        f'저장 위치: `{w["output_directory"]}`\n\n'
        '이8장은 같은 촬영 묶음입니다. 별도 실사 검증 정답은 여전히 필요합니다.\n')
    return w


def status(w):
    saved=0;corners=0
    for r in w['rows']:
        p=C.ROOT/r['annotation']
        if not p.exists():continue
        saved+=1;d=C.read(p)
        corners+=sum(k.get('source')=='manual_click' and k.get('visibility')==2
            for obj in d.get('objects',[]) for k in obj.get('keypoint_annotations',[])[:8])
    return dict(candidates=len(w['rows']),saved=saved,manual_visible_corners=corners,
        validation_candidates=0,output=w['output_directory'])


def launch():
    w=prepare();print(status(w),flush=True)
    pid_path=OUT/'EDITOR_PID.json'
    if pid_path.exists():
        pid=C.read(pid_path)['pid'];p=Path(f'/proc/{pid}/cmdline')
        if p.exists() and str(MANIFEST).encode() in p.read_bytes():
            print('ALREADY_RUNNING',pid);return
    env=os.environ.copy();env.setdefault('DISPLAY',':0')
    env.setdefault('MPLCONFIGDIR','/tmp/pallet-readable-annotation-mpl')
    argv=[sys.executable,'-u',str(C.ROOT/'scripts/annotate/annotate.py'),
        '--review-manifest',str(MANIFEST),'--stride','1','--population-role','DEV']
    with (OUT/'editor.log').open('a') as log:
        p=subprocess.Popen(argv,cwd=C.ROOT,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(2)
    if p.poll() is not None:raise RuntimeError(f'Editor exited {p.returncode}; see {OUT/"editor.log"}')
    C.freeze(pid_path,dict(pid=p.pid,command=argv))
    print('EDITOR_RUNNING',p.pid,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--status',action='store_true')
    args=parser.parse_args()
    if args.prepare_only or args.status:print(status(prepare()))
    else:launch()


if __name__=='__main__':main()
