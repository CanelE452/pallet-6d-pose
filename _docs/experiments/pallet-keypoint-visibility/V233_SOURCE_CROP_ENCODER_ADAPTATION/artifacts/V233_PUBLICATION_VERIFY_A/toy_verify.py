"""Only newly invented temporary bytes; no project/source artifacts are read."""
import hashlib
import json
from pathlib import Path
import tempfile
import verify_package as verifier


def fixture():
    checks=[]
    with tempfile.TemporaryDirectory(prefix='v233_package_toy_') as temporary:
        root=Path(temporary);(root/'notes').mkdir();(root/'notes'/'a.txt').write_bytes(b'invented package text\n')
        payload=(root/'notes'/'a.txt').read_bytes()
        record={'path':'notes/a.txt','sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}
        def manifest(records):
            (root/'MANIFEST.json').write_text(json.dumps({'files':records,'invented_fixture':True}))
        def rejected(name,records):
            manifest(records)
            try:verifier.verify_package(str(root),'MANIFEST.json')
            except (verifier.VerificationError,OSError):checks.append(name)
            else:raise AssertionError(name)
        manifest([record]);result=verifier.verify_package(str(root),'MANIFEST.json')
        assert result['files_verified']==1 and result['bytes_verified']==len(payload);checks.append('valid_SHA_size')
        assert verifier.main(['--package',str(root),'--manifest','MANIFEST.json'])==0;checks.append('CLI_success')
        rejected('wrong_SHA',[dict(record,sha256='0'*64)])
        rejected('wrong_size',[dict(record,bytes=len(payload)+1)])
        for path in ['/absolute.txt','../outside.txt','notes/../a.txt','notes//a.txt','V112/never_open.txt','notes/CACHE.PYC','__pycache__/x.pyc']:
            # file_fd receives an invalid fd: every path is rejected before any filesystem operation.
            try:verifier.file_fd(-1,path)
            except verifier.VerificationError:checks.append('lexical_before_FS:'+path)
            else:raise AssertionError(path)
        rejected('self_manifest_excluded',[dict(record,path='MANIFEST.json')])
        rejected('duplicate_paths',[record,record])
        (root/'link.txt').symlink_to(root/'notes'/'a.txt')
        rejected('leaf_symlink',[dict(record,path='link.txt')])
        (root/'linked_dir').symlink_to(root/'notes',target_is_directory=True)
        rejected('ancestor_symlink',[dict(record,path='linked_dir/a.txt')])
        try:verifier.verify_package(str(root/'linked_dir'),'MANIFEST.json')
        except OSError:checks.append('package_symlink')
        else:raise AssertionError('package_symlink')
    return {'completed':True,'PASS':True,'checks':checks,'count':len(checks),'external_material_reads':0,'actual_package_reads':0,'project_payload_or_model_access':0,'temporary_fixture_only':True}


if __name__=='__main__':
    result=fixture()
    with (Path(__file__).parent/'TOY_RESULT.json').open('x') as output:json.dump(result,output,indent=2);output.write('\n')
    print(json.dumps(result))
