"""Sealed packet provenance must precede production scheduling."""
import pytest
from scripts.research.pallet_bi_full_execution_v1 import fullnormal_parallel as P


@pytest.mark.parametrize('kind',['S2','F'])
def test_all_requested_actual_packet_bytes_bound_before_scheduling(tmp_path,kind):
    frames={'real':[{'id':'a'}],'synthetic':[{'id':'b'}]};digests={}
    for domain in frames:
        path=tmp_path/'packets'/domain/'0000.json.gz';P.Q.write(path,dict(id=domain))
        digests[domain]=P.Q.sha(path)
    seal=tmp_path/'manifest.json'
    if kind=='S2':P.Q.write(seal,dict(artifacts={f'packets/{d}/0000.json.gz':h for d,h in digests.items()}))
    else:P.Q.write(seal,dict(frames=[dict(domain=d,index=0,packet_sha256=h)for d,h in digests.items()]))
    mapping,receipt=P.verify_packet_bindings(tmp_path,seal,frames)
    assert len(mapping)==2 and receipt['all_actual_packet_bytes_verified_before_scheduling']
    P.Q.write(tmp_path/'packets'/'real'/'0000.json.gz',dict(id='changed_actual_packet'))
    with pytest.raises(ValueError,match='mismatch before scheduling'):P.verify_packet_bindings(tmp_path,seal,frames)


def test_missing_packet_mapping_rejected_before_scheduling(tmp_path):
    path=tmp_path/'packets'/'real'/'0000.json.gz';P.Q.write(path,dict(id='a'));seal=tmp_path/'manifest.json'
    P.Q.write(seal,dict(artifacts={'packets/real/0000.json.gz':P.Q.sha(path)}))
    with pytest.raises(ValueError,match='every original'):P.verify_packet_bindings(tmp_path,seal,{'real':[dict(id='a')],'synthetic':[dict(id='b')]})
