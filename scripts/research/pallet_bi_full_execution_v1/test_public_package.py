"""Small constructed packaging tests only; no actual galleries/datasets copied."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from . import public_package as P

def tests():
    with TemporaryDirectory(prefix='public-package-constructed-')as tmp:
        root=Path(tmp);source=root/'source';source.mkdir();doc=source/'PUBLIC_PACKAGE_PLAN.md';table=source/'PUBLIC_STAGE_STATUS_DRAFT.csv'
        doc.write_text('[table](PUBLIC_STAGE_STATUS_DRAFT.csv)\n');table.write_text('stage,status\nS2,INCOMPLETE\n')
        entries=[dict(source=str(p),destination=P.DOC_PREFIX+p.name,kind='documentation',bytes=p.stat().st_size,sha256=P.sha(p),evidence=None)for p in (doc,table)]
        base=dict(schema='EXPLICIT_PUBLIC_PACKAGE_ALLOWLIST_DRAFT_V1',whole_campaign_complete=False,source_roots=[str(source)],source_evidence_bindings={},entries=entries,maximum_file_bytes=4096,maximum_total_bytes=16384,declared_files=2,declared_total_bytes=sum(e['bytes']for e in entries),package_base_execution_sha='CONSTRUCTED')
        def save(value,name):
            p=root/(name+'.json');P.write_new(p,value);return p
        good=save(base,'good');assert P.verify(good)['PASS']
        try:P.materialize(good,root/'no_go')
        except PermissionError:pass
        else:raise AssertionError('Preparation copied files without root flag')
        assert not(root/'no_go').exists()
        try:P.materialize(good,root/'wrong_sha',True,'WRONG')
        except ValueError:pass
        else:raise AssertionError('Reviewed manifest hash not required')
        result=P.materialize(good,root/'constructed_copy',True,P.sha(good));assert result['files']==2
        assert (root/'constructed_copy'/P.DOC_PREFIX/doc.name).read_bytes()==doc.read_bytes()
        assert 'source'not in (root/'constructed_copy/PUBLIC_CONTENTS.json').read_text()
        for name,mutate in [('traversal',lambda x:x['entries'][0].update(destination='../escape.md')),
                            ('duplicate',lambda x:x['entries'][1].update(destination=x['entries'][0]['destination'])),
                            ('weights',lambda x:x['entries'][0].update(destination=P.DOC_PREFIX+'weights.pt')),
                            ('digest',lambda x:x['entries'][0].update(sha256='WRONG'))]:
            value=deepcopy(base);mutate(value)
            try:P.verify(save(value,name))
            except ValueError:pass
            else:raise AssertionError(name+' accepted')
        alias=source/'alias.md';alias.symlink_to(doc);value=deepcopy(base);value['entries'][0]['source']=str(alias)
        try:P.verify(save(value,'symlink'))
        except ValueError:pass
        else:raise AssertionError('Symlink source accepted')
        doc.write_text('[missing](NOT_PUBLIC.csv)\n');value=deepcopy(base);value['entries'][0].update(bytes=doc.stat().st_size,sha256=P.sha(doc));value['declared_total_bytes']=sum(e['bytes']for e in value['entries'])
        try:P.verify(save(value,'missing_link'))
        except ValueError:pass
        else:raise AssertionError('Missing public local link accepted')
        try:P.guard_json(dict(selected=dict(actual_pose=dict(R_physical=[[1]],centroid=[0,0,1]))))
        except ValueError:pass
        else:raise AssertionError('Nested pose/GT JSON accepted')
    return dict(PASS=True,constructed_only=True,actual_dataset_rows_read=0,actual_gallery_files_copied=0,
        new_metric_cost_pose_forward_fit_calls=0,reference_decodes=0,code_imports_scientific_modules=False,
        path_traversal_duplicate_symlink_weights_changed_digest_missing_local_link_rejected=True,
        nested_pose_GT_JSON_rejected=True,root_flag_and_exact_reviewed_manifest_hash_required=True,
        tiny_constructed_copy_exact=True,source_sha256=P.sha(P.__file__))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=tests();P.write_new(a.output,r);print(json.dumps(r))
