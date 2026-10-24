"""Execute only frozen hub0.36 wheel overlay and one offline class-import probe."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,urllib.parse,urllib.request

R=Path(__file__).resolve().parent
PROTOCOL=R/'RUNTIME_OVERLAY_PROTOCOL.json'
EXPECTED='d844b2e4b4d86a6fc4188b633a1a6f1257913d8bc6f584f6e7ae875960a5de1b'
PYTHON='/home/minjae/anaconda3/bin/python'
OVERLAY=R/'overlay'
P=R.parents[1]
SAM=P/'foundation_runtime_sam21_20261023'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def b(p):return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def main_files():
    site=Path('/home/minjae/anaconda3/lib/python3.13/site-packages')
    files=[site/'huggingface_hub/__init__.py',site/'huggingface_hub-1.24.0.dist-info/METADATA',
        site/'transformers/__init__.py',site/'transformers-4.57.6.dist-info/METADATA',
        SAM/'RUNTIME_SEAL.json',SAM/'provenance/RUNTIME_VERSIONS.json',
        SAM/'venv/pyvenv.cfg']
    return [b(p) for p in files]
def main():
    assert sha(PROTOCOL)==EXPECTED
    p=json.loads(PROTOCOL.read_text())
    assert p['frozen'] is True and p['package']=='huggingface-hub==0.36.0'
    assert p['overlay_directory']==str(OVERLAY)
    for binding in p['bindings'].values():assert sha(binding['path'])==binding['sha256']
    assert not OVERLAY.exists() and not (R/'EXECUTION_STARTED.json').exists(),'Preserve attempts; no automatic retry'
    before=main_files()
    write(R/'EXECUTION_STARTED.json',dict(protocol=b(PROTOCOL),code=b(Path(__file__)),probe=b(R/'probe_overlay_offline_classes.py'),
        main_and_SAM_before=before,model_source_current_reads=0))
    stage='official_metadata'
    try:
        url='https://pypi.org/pypi/huggingface-hub/0.36.0/json'
        req=urllib.request.Request(url,headers={'User-Agent':'V81-official-wheel-metadata-only'})
        with urllib.request.urlopen(req,timeout=30) as f:
            raw=f.read(500000)
        meta=json.loads(raw);assert meta['info']['version']=='0.36.0'
        wheels=[x for x in meta['urls'] if x['packagetype']=='bdist_wheel' and x['filename']=='huggingface_hub-0.36.0-py3-none-any.whl']
        assert len(wheels)==1;wheel=wheels[0]
        assert urllib.parse.urlparse(wheel['url']).scheme=='https' and urllib.parse.urlparse(wheel['url']).hostname=='files.pythonhosted.org'
        metadata_path=R/'OFFICIAL_PYPI_METADATA.json'
        with metadata_path.open('xb') as f:f.write(raw)
        write(R/'OFFICIAL_WHEEL_SELECTION_BEFORE_DOWNLOAD.json',dict(metadata_URL=url,metadata=b(metadata_path),filename=wheel['filename'],URL=wheel['url'],
            expected_sha256=wheel['digests']['sha256'],expected_bytes=wheel['size'],no_dependencies=True,code=b(Path(__file__))))
        stage='official_wheel_download'
        target=R/wheel['filename']
        req=urllib.request.Request(wheel['url'],headers={'User-Agent':'V81-fixed-official-wheel-only'})
        with urllib.request.urlopen(req,timeout=60) as response:
            assert urllib.parse.urlparse(response.geturl()).hostname=='files.pythonhosted.org'
            body=response.read(wheel['size']+1)
        assert len(body)==wheel['size']
        with target.open('xb') as f:f.write(body)
        assert sha(target)==wheel['digests']['sha256'],'Official PyPI wheel digest mismatch'
        write(R/'WHEEL_SHA_VERIFIED_BEFORE_INSTALL.json',dict(PASS=True,wheel=b(target),expected_sha256=wheel['digests']['sha256'],metadata=b(metadata_path),protocol=b(PROTOCOL)))
        stage='overlay_install_only'
        cmd=[PYTHON,'-B','-m','pip','--isolated','install','--no-index','--no-deps','--no-compile','--no-cache-dir','--disable-pip-version-check','--target',str(OVERLAY),str(target)]
        env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
        result=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=60)
        with (R/'INSTALL_STDOUT.log').open('x') as f:f.write(result.stdout)
        with (R/'INSTALL_STDERR.log').open('x') as f:f.write(result.stderr)
        write(R/'INSTALL_RECEIPT.json',dict(command=cmd,exit_code=result.returncode,stdout=b(R/'INSTALL_STDOUT.log'),stderr=b(R/'INSTALL_STDERR.log'),
            verified_wheel=b(target),overlay=str(OVERLAY),dependencies_installed=False,base_environment_targeted=False))
        assert result.returncode==0,'pip overlay install failed; stop without dependency repair'
        stage='offline_class_import_only'
        env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',PYTHONPATH=str(OVERLAY),PYTHONDONTWRITEBYTECODE='1')
        for key in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN'):env.pop(key,None)
        cmd=[PYTHON,'-B',str(R/'probe_overlay_offline_classes.py')]
        start=time.monotonic();probe=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=60)
        with (R/'OFFLINE_PROBE_STDOUT.json').open('x') as f:f.write(probe.stdout)
        with (R/'OFFLINE_PROBE_STDERR.log').open('x') as f:f.write(probe.stderr)
        try:parsed=json.loads(probe.stdout)
        except Exception:parsed={'PASS':False,'unparsed_stdout':True}
        after=main_files();assert after==before,'Base environment or SAM records changed'
        report=dict(completed=True,PASS=bool(parsed.get('PASS')) and probe.returncode==0,stage=stage,
            protocol=b(PROTOCOL),wheel=b(target),wheel_verification=b(R/'WHEEL_SHA_VERIFIED_BEFORE_INSTALL.json'),
            install=b(R/'INSTALL_RECEIPT.json'),probe_code=b(R/'probe_overlay_offline_classes.py'),probe_exit_code=probe.returncode,seconds=time.monotonic()-start,
            stdout=b(R/'OFFLINE_PROBE_STDOUT.json'),stderr=b(R/'OFFLINE_PROBE_STDERR.log'),result=parsed,
            base_and_SAM_source_hashes_unchanged=True,main_and_SAM_before=before,main_and_SAM_after=after,
            no_bytecode_writes=True,downloaded_artifacts='One official PyPI wheel plus PyPI metadata only',
            other_dependency_or_version_repairs=0,model_constructor_weights_source_current_GPU_forward_fit_PnP=0)
        write(R/'RUNTIME_OVERLAY_RESULT.json',report)
        if not report['PASS']:
            write(R/'IMPORT_BLOCKER_PRESERVED.json',dict(stage=stage,result=b(R/'RUNTIME_OVERLAY_RESULT.json'),exception=parsed.get('exception_message'),
                automatic_repair_allowed=False,other_deps_versions_changed=False))
        print(json.dumps(dict(PASS=report['PASS'],result=b(R/'RUNTIME_OVERLAY_RESULT.json'),exception=parsed.get('exception_message'))),flush=True)
    except Exception as e:
        import traceback
        write(R/'EXECUTION_FAILURE_PRESERVED.json',dict(stage=stage,exception_type=type(e).__name__,exception_message=str(e),traceback=traceback.format_exc(),code=b(Path(__file__)),
            automatic_dependency_version_retry=False,model_source_current_reads=0))
        raise
if __name__=='__main__':main()
