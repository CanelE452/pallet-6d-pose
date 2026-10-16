"""Read driver versions and verify real CUDA forward/backward.

Run outside the device-isolating filesystem sandbox. Does not install software,
reload modules, terminate processes or reboot the workstation.
"""
import argparse
import datetime
import json
import ctypes
import platform
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def command(args):
    try:
        p=subprocess.run(args,capture_output=True,text=True,timeout=20)
        return {'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {'returncode':None,'error':str(exc)}

def check():
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'kernel':platform.release(),'nvidia_smi':command(['nvidia-smi']),
            'installed_module':command(['modinfo','-F','version','nvidia'])}
    p=Path('/proc/driver/nvidia/version')
    text=p.read_text() if p.exists() else ''
    match=re.search(r'Kernel Module\s+(\d+\.\d+\.\d+)',text)
    result['loaded_module_version']=match.group(1) if match else None
    result['loaded_matches_installed']=(result['loaded_module_version'] is not None and
        result['loaded_module_version']==result['installed_module'].get('stdout'))
    try:
        nvml=ctypes.CDLL('libnvidia-ml.so.1')
        if nvml.nvmlInit_v2()!=0:raise RuntimeError('NVML initialization failed')
        buf=ctypes.create_string_buffer(80)
        rc=nvml.nvmlSystemGetDriverVersion(buf,len(buf))
        nvml.nvmlShutdown()
        if rc!=0:raise RuntimeError('NVML version query failed')
        result['active_nvml_driver_version']=buf.value.decode()
    except Exception as exc:result['nvml_error']=str(exc)
    try:
        import torch
        result['torch']=torch.__version__
        result['torch_cuda_runtime']=torch.version.cuda
        result['cuda_available']=torch.cuda.is_available()
        if not result['cuda_available']:raise RuntimeError('CUDA unavailable')
        result['gpu_name']=torch.cuda.get_device_name(0)
        torch.manual_seed(17)
        cpu=torch.randn(2,3,64,64)
        weight=torch.randn(8,3,3,3)
        x=cpu.cuda().requires_grad_()
        w=weight.cuda().requires_grad_()
        y=torch.nn.functional.conv2d(x,w,padding=1)
        y.square().mean().backward()
        torch.cuda.synchronize()
        reference=torch.nn.functional.conv2d(cpu,weight,padding=1)
        err=float((y.detach().cpu()-reference).abs().max())
        result['cuda_convolution_max_abs_error']=err
        result['finite_output_and_gradients']=bool(torch.isfinite(y).all() and
            torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all())
        result['nonzero_gradients']=bool(x.grad.abs().sum()>0 and w.grad.abs().sum()>0)
        result['cuda_forward_backward_passed']=bool(err<.001 and result['finite_output_and_gradients'] and
                                                   result['nonzero_gradients'])
    except Exception as exc:
        result['cuda_forward_backward_passed']=False
        result['cuda_error']=str(exc)
    result['active_runtime_matches_loaded']=(result.get('active_nvml_driver_version') is not None and
        result.get('active_nvml_driver_version')==result['loaded_module_version'])
    result['system_installation_consistent']=result['loaded_matches_installed']
    result['PASS']=bool(result['nvidia_smi']['returncode']==0 and result['active_runtime_matches_loaded'] and
                        result['cuda_forward_backward_passed'])
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise SystemExit('Refusing to overwrite an existing diagnostic')
    result=check()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['PASS'] else 1)
