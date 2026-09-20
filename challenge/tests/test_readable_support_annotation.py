"""Actual eight-frame navigation without opening a GUI or writing labels."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/annotate'))
sys.path.insert(0,str(ROOT/'scripts'))


def test_readable_navigation_preserves_original(monkeypatch):
    import annotate as A
    from scripts.research.pallet_type_selftrain_v1.open_readable_support_annotation import prepare,MANIFEST
    from scripts.research.pallet_type_selftrain_v1 import common as C
    if not MANIFEST.exists():pytest.skip('local RGB candidates unavailable')
    w=prepare();seen=[]
    old=C.read(C.ROOT/w['original_workspace']['path'])
    assert not {r['annotation'] for r in w['rows']} & {r['annotation'] for r in old['rows']}
    for name in ['namedWindow','resizeWindow','setMouseCallback','createTrackbar','setTrackbarPos','destroyAllWindows','imshow']:
        monkeypatch.setattr(A.cv2,name,lambda *a,**k:None)
    monkeypatch.setattr(A.cv2,'getWindowProperty',lambda *a:1)
    monkeypatch.setattr(A,'load_existing_annotation',lambda *a,**k:False)
    monkeypatch.setattr(A,'update_pose',lambda *a:None)
    def render(s,cur,total,stem):
        seen.append((stem,s.split,all(p is None for p in s.kps_2d),s.annotation_output_dir))
        return np.zeros((100,100,3),np.uint8)
    monkeypatch.setattr(A,'render',render)
    monkeypatch.setattr(A.cv2,'waitKey',lambda *a:ord('q') if len(seen)==8 else ord('n'))
    def fail(*a,**k):raise AssertionError('Unexpected save or legacy load')
    monkeypatch.setattr(A,'save_frame_json',fail)
    monkeypatch.setattr(A,'_resolve_legacy_read_dir',fail)
    A.main(['--review-manifest',str(MANIFEST),'--stride','1','--population-role','DEV'])
    assert len(seen)==8 and len({r[0] for r in seen})==8
    assert all(r[1]=='train' and r[2] for r in seen)
    assert all(Path(r[3]).resolve()==C.ROOT/w['output_directory'] for r in seen)
    for b in w['original_saved_at_setup']:C.verify(b)
    C.verify(w['original_workspace'])
