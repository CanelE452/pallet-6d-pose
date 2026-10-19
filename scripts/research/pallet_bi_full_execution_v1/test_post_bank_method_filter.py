"""Keep an unused estimator outside the reported generated candidate bound."""
from pathlib import Path
import argparse
import tempfile

from . import post_bank as P, s1_union as U


def test(output):
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);archive=root/'EXTRAS.jsonl.gz';seal=root/'SEAL.json'
        methods=['POSEFIX_RAW','POSEFIX_SUBPIX_PF_ANCHOR','POSEFIX_SUBPIX_YOLO_ANCHOR']
        records=[dict(id=fid,method=m,candidate_id=m,producer_available=m!=methods[2])
                 for fid in ('a','b')for m in methods]
        U.save(archive,records)
        U.write(seal,dict(complete=True,output_sha256=U.sha(archive)))
        before=U.sha(archive)
        source=dict(path=str(archive),seal=str(seal))
        whole,binding=P.declared_extras(source,['a','b'])
        assert [r for rr in whole.values()for r in rr]==records
        assert 'declared_methods'not in binding
        selected,binding=P.declared_extras(dict(**source,methods=methods[1:]),['a','b'])
        assert all([r['method']for r in rr]==methods[1:]for rr in selected.values())
        assert sum(map(len,selected.values()))==4 and binding['selected_rows']==4
        assert all(not rr[-1]['producer_available']for rr in selected.values())
        rejected=0
        for bad in ([],methods[1],['unknown'],[methods[1],methods[1]]):
            try:P.declared_extras(dict(**source,methods=bad),['a','b'])
            except ValueError:rejected+=1
            else:raise AssertionError('Bad method declaration accepted')
        try:P.declared_extras(dict(**source,methods=methods[1:]),['a','b','missing'])
        except ValueError:rejected+=1
        else:raise AssertionError('Missing denominator accepted')
        assert U.sha(archive)==before
        legacy=root/'LEGACY.jsonl.gz';legacy_seal=root/'LEGACY_SEAL.json'
        legacy_rows=[dict(id=fid,candidate_id='legacy',actual_pose=dict(available=False))for fid in ('a','b')]
        U.save(legacy,legacy_rows);U.write(legacy_seal,dict(complete=True,output_sha256=U.sha(legacy)))
        values,_=P.declared_extras(dict(path=str(legacy),seal=str(legacy_seal)),['a','b'])
        assert [r for rr in values.values()for r in rr]==legacy_rows
    result=dict(PASS=True,full_archive_default_preserved=True,
        unused_RAW_excluded_from_PF2_bound=True,unavailable_declared_action_preserved=True,
        methodless_legacy_default_preserved=True,
        adversarial_declarations_rejected=rejected,source_archive_unchanged=True,
        constructed_only=True,campaign_dataset_calls=0,pose_calls=0,model_forwards=0,fits=0,reference_reads=0)
    U.write(output,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    print(test(parser.parse_args().output))
