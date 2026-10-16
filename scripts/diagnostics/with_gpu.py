"""Run a command with process-local NVIDIA libraries when the host is mismatched.

Does not reboot, install a driver, change system links, or stop any application.
The override is only enabled for the EXACT currently loaded kernel module.
When normal nvidia-smi works, the system environment is used unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'.cache/nvidia-runtime'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def environment():
    env=dict(os.environ)
    probe=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],
                         capture_output=True,text=True,timeout=20,env=env)
    if probe.returncode==0:return env,'system or inherited working runtime'
    text=Path('/proc/driver/nvidia/version').read_text()
    found=re.search(r'Kernel Module\s+(\d+\.\d+\.\d+)',text)
    if not found:raise RuntimeError('Cannot identify loaded NVIDIA kernel module')
    version=found.group(1);bundle=CACHE/version
    manifest=json.loads((bundle/'MANIFEST.json').read_text())
    if manifest['version']!=version:raise RuntimeError('Local runtime does not match loaded kernel')
    for name,sha in manifest['files'].items():
        if digest(bundle/name)!=sha:raise RuntimeError(f'Runtime file changed: {name}')
    env['LD_LIBRARY_PATH']=str(bundle)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    probe=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],
                         capture_output=True,text=True,timeout=20,env=env)
    if probe.returncode!=0 or any(v.strip()!=version for v in probe.stdout.splitlines()):
        raise RuntimeError('Matching local runtime did not restore NVIDIA access: '+probe.stderr+probe.stdout)
    env['PALLET_NVIDIA_RUNTIME_VERSION']=version
    return env,'process-local NVIDIA '+version

def main():
    args=sys.argv[1:]
    if args and args[0]=='--':args=args[1:]
    if not args:raise SystemExit('usage: python scripts/diagnostics/with_gpu.py COMMAND [ARGS...]')
    env,mode=environment()
    print('[GPU runtime] '+mode,file=sys.stderr,flush=True)
    os.execvpe(args[0],args,env)

if __name__=='__main__':main()
