"""Replay saved Source CF arithmetic from exact six public input blobs; no inference."""
from pathlib import Path
import argparse,hashlib,json,os,stat,types


def safe(p):
    p=Path(p)
    if not p.is_absolute() or '..' in p.parts or any(w in str(p).lower() for w in ('v112','cache','.pyc')):
        raise ValueError('Disallowed artifact path')
    for a in [*reversed(p.parents),p]:
        try:s=os.lstat(a)
        except FileNotFoundError:continue
        if stat.S_ISLNK(s.st_mode):raise ValueError('Artifact symlink')
    return p


def read(p):return safe(p).read_bytes()
def sha(b):return hashlib.sha256(b).hexdigest()
def relpath(base,rel):
    q=Path(rel)
    if q.is_absolute() or '..' in q.parts:raise ValueError('Nonrelative public dependency')
    return safe(base/q)


def replay(package,output):
    package=safe(package);output=safe(output)
    if output.exists():raise ValueError('Fresh output directory required')
    manifest_bytes=read(package/'MANIFEST.json');manifest=json.loads(manifest_bytes)
    assert manifest['completed'] and manifest['PASS']
    for row in manifest['files']:
        b=read(relpath(package,row['relative_path']))
        assert len(b)==row['size_bytes'] and sha(b)==row['sha256']
    deps=json.loads(read(package/'runtime_dependencies.json'))
    bindings={}
    for key,row in deps['inputs6'].items():
        p=relpath(package,row['relative_path']);b=read(p)
        assert len(b)==row['size_bytes'] and sha(b)==row['sha256']
        bindings[key]={'path':str(p),'sha256':row['sha256'],'size_bytes':row['size_bytes']}
    cr=deps['code'];cp=relpath(package,cr['relative_path']);code=read(cp)
    assert sha(code)==cr['sha256'] and len(code)==cr['size_bytes']
    m=types.ModuleType('v254_saved_cf_replay');m.__file__=str(cp)
    exec(compile(code,str(cp),'exec'),m.__dict__)
    m.run(bindings,output)
    compared=[]
    for row in deps['core_outputs']:
        leaf=Path(row['name'])
        if leaf.is_absolute() or len(leaf.parts)!=1 or leaf.name!=row['name'] or '..' in leaf.parts:
            raise ValueError('Core output must be a single relative filename')
        actual=read(safe(output/leaf));expected=read(relpath(package,row['relative_path']))
        assert sha(expected)==row['sha256'] and len(expected)==row['size_bytes']
        assert actual==expected,'Replay byte mismatch: '+row['name']
        compared.append({'name':row['name'],'sha256':sha(actual),'size_bytes':len(actual),'byte_exact':True})
    result={'completed':True,'PASS':True,'package_manifest':{'path':str(package/'MANIFEST.json'),'sha256':sha(manifest_bytes),'size_bytes':len(manifest_bytes)},'all_package_files_hashed':len(manifest['files']),'replayed_core_outputs':compared,'new_training':0,'new_model_forward':0,'RGB_or_GT_decode':0,'original_absolute_path_fallback':False,'current_or_held_read':0,'PnP':0,'limits':'Saved Source arithmetic replay only, not original teacher/renderer/encoder generation or private evaluation.'}
    b=(json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode()
    with safe(output/'PUBLIC_REPLAY_QA.json').open('xb') as f:f.write(b)
    print(json.dumps({'completed':True,'PASS':True,'core_outputs_byte_exact':len(compared),'package_files_hashed':len(manifest['files'])}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    replay(a.package,a.output)
