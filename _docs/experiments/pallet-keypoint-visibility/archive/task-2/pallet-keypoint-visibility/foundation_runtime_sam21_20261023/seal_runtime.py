#!/usr/bin/env python3
"""Read-only asset/runtime audit and one-time seal; no model or image calls."""
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def ref(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)


def read(p):return json.loads(Path(p).read_text())


def main():
    output=ROOT/'RUNTIME_SEAL.json';assert not output.exists(),'Preserve existing seal'
    assert Path(sys.prefix).resolve()==ROOT/'venv' and sys.prefix!=sys.base_prefix
    protocol=read(ROOT/'PREPARATION_PROTOCOL.json')
    assert protocol['dataset_inference_allowed'] is False
    versions={d.metadata['Name']:d.version for d in metadata.distributions()}
    expected={'torch':'2.5.1+cu121','torchvision':'0.20.1+cu121','triton':'3.1.0','numpy':'1.26.4',
              'hydra-core':'1.3.2','omegaconf':'2.3.0','iopath':'0.1.10','SAM-2':'1.0'}
    for key,value in expected.items():assert metadata.version(key)==value
    wheels=read(ROOT/'provenance/WHEEL_MANIFEST.json')
    for x in wheels:assert sha(x['path'])==x['sha256']
    cached=read(ROOT/'provenance/CACHED_STAGE_WHEELS.json')
    assert len(cached)==30 and all(x['hardlink_same_inode'] for x in cached)
    new=read(ROOT/'provenance/NEW_DOWNLOADED_WHEELS.json')
    assert len(new)==5 and all(x['verified'] for x in new)
    repo=ROOT/'sam2_repo'
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True)
    assert revision==protocol['revision'] and not dirty
    repo_manifest=read(ROOT/'provenance/SAM2_REPO_MANIFEST.json')
    for path,digest in repo_manifest['code_files_sha256'].items():assert sha(repo/path)==digest
    assert not list((repo/'sam2').glob('_C*.so')),'No optional CUDA extension expected'
    cp=ROOT/'checkpoints/sam2.1_hiera_base_plus.pt'
    assert sha(cp)==protocol['checkpoint_SHA256'] and cp.stat().st_size==protocol['checkpoint_bytes']
    smokes={device:read(ROOT/'smoke'/device/'SMOKE.json') for device in ('CPU','CUDA')}
    for device,report in smokes.items():
        assert report['PASS'] is True and report['source_RGB_data_reads']==0
        assert report['current_RGB_GT_scores_inputs_read'] is False and report['optimizer_updates']==0
        assert report['compile_called'] is False and report['all_parameters_frozen'] is True
        assert report['checkpoint']['sha256']==protocol['checkpoint_SHA256']
        assert sha(report['code']['path'])==report['code']['sha256']
        for x in report['outputs']:assert sha(x['path'])==x['sha256']
    assert smokes['CUDA']['CUDA_peak_reserved_bytes'] < 10*1024**3
    check=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True,check=True)
    runtime=dict(python=sys.executable,python_version=sys.version,prefix=sys.prefix,base_prefix=sys.base_prefix,
        include_system_site_packages=False,pyvenv_config=(ROOT/'venv/pyvenv.cfg').read_text(),
        installed_packages=versions,pip_check=check.stdout.strip(),optional_cv2_installed=any(k.lower()=='opencv-python' for k in versions),
        dependencies_are_local_pinned_wheels=True)
    (ROOT/'provenance/RUNTIME_VERSIONS.json').write_text(json.dumps(runtime,indent=2)+'\n')
    binding_paths=[ROOT/'PREPARATION_PROTOCOL.json',ROOT/'INTEGRATION.json',ROOT/'REQUIREMENTS.lock.txt',ROOT/'smoke_blank.py',Path(__file__),
        cp,ROOT/'provenance/WHEEL_MANIFEST.json',ROOT/'provenance/CACHED_STAGE_WHEELS.json',ROOT/'provenance/NEW_DOWNLOADED_WHEELS.json',
        ROOT/'provenance/OFFICIAL_METADATA_DOWNLOADS.json',ROOT/'provenance/SAM2_REPO_MANIFEST.json',ROOT/'provenance/SAM2_APACHE2_LICENSE.txt',
        ROOT/'provenance/INSTALLED_PACKAGES.json',ROOT/'provenance/RUNTIME_VERSIONS.json',ROOT/'venv/pyvenv.cfg']
    for x in read(ROOT/'provenance/OFFICIAL_METADATA_DOWNLOADS.json'):binding_paths.append(Path(x['path']))
    for device in ('CPU','CUDA'):
        binding_paths.append(ROOT/'smoke'/device/'SMOKE.json')
        binding_paths.append(ROOT/'smoke'/device/'GENERATED_BLANK_OUTPUTS.npz')
    for p in sorted((ROOT/'logs').glob('*.log')):binding_paths.append(p)
    report=dict(schema='SAM21_BASE_PLUS_FROZEN_ISOLATED_RUNTIME_READY_SEAL_V1',created_unix=time.time(),READY=True,
        environment=str(ROOT/'venv'),python=sys.executable,repo_revision=revision,official_repository_clean=True,
        checkpoint=ref(cp),license='Apache-2.0; official code and checkpoint',config=protocol['config'],
        pinned_versions=expected,bindings=[ref(p) for p in binding_paths],wheels=wheels,
        cached_wheel_hardlink_count=len(cached),cached_wheel_bytes_reused=sum(x['bytes'] for x in cached),
        new_wheel_download_count=len(new),new_wheel_download_bytes=sum(x['bytes'] for x in new),
        checkpoint_download_bytes=cp.stat().st_size,total_new_binary_download_bytes=sum(x['bytes'] for x in new)+cp.stat().st_size,
        free_disk_bytes=shutil.disk_usage(ROOT).free,
        generated_smokes={d:dict(PASS=r['PASS'],seconds=r['seconds'],CUDA_peak_allocated_bytes=r['CUDA_peak_allocated_bytes'],
            CUDA_peak_reserved_bytes=r['CUDA_peak_reserved_bytes']) for d,r in smokes.items()},
        SAM2_BUILD_CUDA='0',torch_compile=False,postprocessing=False,source_RGB_reads=0,current_RGB_GT_scores_inputs_read=False,
        training_calls=0,optimizer_updates=0,PnP_calls=0,existing_environment_package_changes=0,
        dataset_inference_allowed=False,current_inference_allowed=False,
        next_scope='Parent must freeze and authorize a separate source-only baseline protocol before reading dataset RGB or performing dataset inference.',
        baseline_semantics='SAM2 image visible segmentation only. Query absence does not establish external/self occlusion or physical target absence.',
        portability='Local reproducibility bundle. Venv uses an existing read-only Python3.10 base interpreter; absolute paths bind this workspace.')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(READY=True,seal=ref(output),python=sys.executable,free_disk_bytes=report['free_disk_bytes'],
                          generated_smokes=report['generated_smokes'])),flush=True)


if __name__=='__main__':main()
