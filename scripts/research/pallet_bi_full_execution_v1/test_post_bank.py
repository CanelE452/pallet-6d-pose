"""Constructed compressed-archive and wrong-bank provenance regressions."""
from pathlib import Path
from tempfile import TemporaryDirectory
import gzip
import json
from . import s1_union as U, post_bank as B


def run():
    with TemporaryDirectory()as temporary:
        root=Path(temporary);bank=root/'GOOD';wrong=root/'WRONG';bank.mkdir();wrong.mkdir()
        pose=dict(R_physical=[[1,0,0],[0,1,0],[0,0,1]],centroid=[0,0,1])
        for folder,label in ((bank,'A'),(wrong,'B')):
            row=folder/'0000';row.mkdir()
            with gzip.open(row/'RESULT.json.gz','wt')as stream:
                json.dump(dict(id='CONSTRUCTED',candidates=[dict(candidate_id=label,**pose)]),stream)
            U.write(row/'FRAME_SEAL.json',dict(complete=True,id='CONSTRUCTED',artifacts={'RESULT.json.gz':U.sha(row/'RESULT.json.gz')}))
        source=root/'FEATURE_SEAL.json'
        U.write(source,dict(complete=True,rows=238,frame_seal_sha256={'0000':U.sha(bank/'0000/FRAME_SEAL.json')}))
        spec=dict(backend='fullnormal',result_root=str(bank),feature_seal=str(source))
        binding={};rows=list(B.generated_rows(spec,[{'id':'CONSTRUCTED'}],root,{},binding))
        assert rows[0][0]['candidate_id']=='A'and binding['generated_frame_seals']['0000']==U.sha(bank/'0000/FRAME_SEAL.json')
        try:list(B.generated_rows(dict(spec,result_root=str(wrong)),[{'id':'CONSTRUCTED'}],root,{},{}))
        except ValueError as error:assert 'different feature bank'in str(error)
        else:raise AssertionError('Wrong bank accepted')
        union=root/'UNION_FEATURE_SEAL.json';union_row=root/'UNION_ROW.json'
        U.write(union_row,dict(dependencies=dict(core_frame_seal_sha256=U.sha(wrong/'0000/FRAME_SEAL.json'),result_sha256=U.sha(wrong/'0000/RESULT.json.gz'))))
        U.write(union,dict(schema='OWN_FULL_NORMAL_KERNEL_EXTRA_FEATURE_SEAL_V1',frame_seals=[dict(id='CONSTRUCTED',seal_path=str(union_row),sha256=U.sha(union_row))]))
        try:list(B.generated_rows(dict(spec,feature_seal=str(union),generated_feature_seal=str(source)),[{'id':'CONSTRUCTED'}],root,{},{}))
        except ValueError as error:assert 'different banks'in str(error)
        else:raise AssertionError('Wrong core-to-union ancestry accepted')
        with (bank/'0000/RESULT.json.gz').open('ab')as stream:stream.write(b'corruption')
        try:list(B.generated_rows(spec,[{'id':'CONSTRUCTED'}],root,{},{}))
        except ValueError as error:assert 'archive changed'in str(error)
        else:raise AssertionError('Changed compressed archive accepted')
    return dict(PASS=True,constructed_checks=4,checks=['gzip actual-pose archive decode','wrong arm resultroot rejected','wrong union core ancestry rejected','changed endpoint archive rejected'],new_pose_calls=0,real_reference_reads=0,new_fits=0)


if __name__=='__main__':print(run())
