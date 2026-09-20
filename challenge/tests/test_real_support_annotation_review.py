"""Review session routing: no GT creation and no legacy coordinate preload."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/annotate'))
sys.path.insert(0,str(ROOT/'scripts'))
from annotate_review import load_review_contexts, OUTPUT_REL
from object_geometry_registry import load_object_geometry_registry, DEFAULT_REGISTRY_PATH
from annotate_sessions import session_summary


def args():
    return argparse.Namespace(capture_timestamp=None,camera_serial=None)


def fixture(tmp_path):
    records=[]
    for i,role in enumerate(['proposed_support','proposed_validation']):
        session=tmp_path/f'raw/night/s{i}';(session/'rgb').mkdir(parents=True)
        image=session/'rgb/001.png';image.write_bytes(bytes([i]))
        K=np.array([[600+i,0,320],[0,600+i,240],[0,0,1.]])
        np.savetxt(session/'cam_K.txt',K)
        records.append(dict(proposed_role=role,session=str(session.relative_to(tmp_path)),
            image=dict(path=str(image.relative_to(tmp_path)),sha256=hashlib.sha256(image.read_bytes()).hexdigest()),
            K=K.tolist(),recording_id=str(i),object_type='plastic_standard_110x130x11'))
    p=tmp_path/'manifest.json';p.write_text(json.dumps(records));return p,records


def test_split_camera_output_and_count_are_per_session(tmp_path):
    path,rows=fixture(tmp_path);reg=load_object_geometry_registry(DEFAULT_REGISTRY_PATH)
    sessions,contexts=load_review_contexts(path,args(),reg,tmp_path)
    cc=list(contexts.values());assert [c['args'].default_split for c in cc]==['train','eval']
    assert cc[0]['K'][0,0]==600 and cc[1]['K'][0,0]==601
    assert all(Path(c['out_dir']).is_relative_to(tmp_path/OUTPUT_REL) for c in cc)
    assert [r['frames'] for r in session_summary(sessions,tmp_path,contexts=contexts)]==[1,1]
    assert not (tmp_path/OUTPUT_REL).exists()  # Discovery does not write labels.


@pytest.mark.parametrize('damage',['hash','group','camera'])
def test_changed_inputs_rejected(tmp_path,damage):
    path,rows=fixture(tmp_path)
    if damage=='hash':rows[0]['image']['sha256']='bad'
    elif damage=='group':rows[1]['recording_id']=rows[0]['recording_id']
    else:rows[1]['K'][0][0]=999
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):load_review_contexts(path,args(),load_object_geometry_registry(DEFAULT_REGISTRY_PATH),tmp_path)


def test_actual_30_frame_navigation_without_labels_or_gui(monkeypatch):
    import annotate as A
    manifest=ROOT/'outputs/pallet_type_selftrain_v1/large_corner_recovery_v1/real_support_review_v2/CANDIDATES.json'
    if not manifest.exists():pytest.skip('local review images unavailable')
    seen=[]
    for name in ['namedWindow','resizeWindow','setMouseCallback','createTrackbar','setTrackbarPos','destroyAllWindows','imshow']:
        monkeypatch.setattr(A.cv2,name,lambda *a,**k:None)
    monkeypatch.setattr(A.cv2,'getWindowProperty',lambda *a:1)
    monkeypatch.setattr(A,'load_existing_annotation',lambda *a,**k:False)
    monkeypatch.setattr(A,'update_pose',lambda *a:None)
    def render(state,cur,total,stem):
        seen.append((state.sess_name,state.split,stem,all(p is None for p in state.kps_2d),state.annotation_output_dir))
        return np.zeros((100,100,3),np.uint8)
    monkeypatch.setattr(A,'render',render)
    monkeypatch.setattr(A.cv2,'waitKey',lambda *a:ord('q') if len(seen)==30 else ord('n'))
    def fail(*a,**k):raise AssertionError('unexpected save or legacy lookup')
    monkeypatch.setattr(A,'save_frame_json',fail)
    monkeypatch.setattr(A,'_resolve_legacy_read_dir',fail)
    A.main(['--review-manifest',str(manifest),'--stride','1','--population-role','DEV'])
    assert len(seen)==30 and len({r[0] for r in seen})==7
    assert [r[1] for r in seen]==['train']*20+['eval']*10
    assert all(r[3] for r in seen)
    assert all(Path(r[4]).is_relative_to(ROOT/OUTPUT_REL) for r in seen)
