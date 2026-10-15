from copy import deepcopy
import gzip,hashlib,io,json
import pytest
from . import archive_compact_clean_fresh_BI_poses_v4_32_r2_v1 as A


def row():
    return dict(id='toy:0',partition='TEST',states=['B']*8,legacy_e_pose_semantics={'centered':True},
        arms={a:dict(accepted=True,pose_available=True,actual_pose={'R_physical':[1,2,3],'native_t':[.1,.2,.3]},
            WD_family='SWAPPED_WD',solution={'objective':1.2,'native_observation_objective':.2},
            full_costplan={'not':'copied'},source_policy='toy')for a in A.ARMS})


def test_all5_exact_poses_topWD_and_flags_saved_not_full_evidence():
    r=row();before=deepcopy(r);c=A.compact(r)
    assert r==before and set(c['arms'])==set(A.ARMS)
    for a in A.ARMS:
        assert c['arms'][a]['actual_pose']==r['arms'][a]['actual_pose']and c['arms'][a]['WD_family']=='SWAPPED_WD'
        assert 'full_costplan'not in c['arms'][a]
    c['arms'][A.ARMS[0]]['actual_pose']['native_t'][0]=999
    assert r==before


def test_fallback_and_no_pose_retained_without_invented_pose():
    r=row();a=A.ARMS[1];r['arms'][a].update(accepted=False,pose_available=False,actual_pose=None)
    r['arms'][A.ARMS[2]]['accepted']=False
    c=A.compact(r)
    assert c['arms'][a]['actual_pose']is None and c['arms'][A.ARMS[2]]['actual_pose']is not None
    del r['arms'][A.ARMS[-1]]
    with pytest.raises(ValueError):A.compact(r)


def test_gzip_multimember_onepass_original_compressedSHA_and_unicode():
    payload=('한글\n'.encode());blob=gzip.compress(payload)+gzip.compress(b'2\n')
    h=A.HashReader(io.BytesIO(blob))
    with gzip.GzipFile(fileobj=h,mode='rb')as z:assert z.read()==payload+b'2\n'
    assert h.digest.hexdigest()==hashlib.sha256(blob).hexdigest()and h.byte_count==len(blob)


def test_go_and_explicit_SHA_before_IO(monkeypatch):
    def no(*a,**k):raise AssertionError('No IO')
    monkeypatch.setattr(A.C,'read',no);monkeypatch.setattr(A.C,'sha',no)
    with pytest.raises(PermissionError):A.declare()
    with pytest.raises(PermissionError):A.run()
    with pytest.raises(ValueError):A.declare(root_metadata_GO=True)
