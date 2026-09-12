"""Read-only process-local GPU telemetry, no driver or power-setting changes."""
import json
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.contracts import RAW,write

def snapshot():
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    query=lambda fields,kind:subprocess.check_output(['nvidia-smi',f'--query-{kind}={fields}','--format=csv'],text=True).strip()
    value=dict(utc=timestamp,gpu=query('name,temperature.gpu,memory.used,utilization.gpu,power.draw','gpu'),
        compute=query('pid,process_name,used_memory','compute-apps'),system_changes=False)
    write(RAW/'gpu'/f'{timestamp}.json',value)
    print(json.dumps(value),flush=True)
    return value

if __name__=='__main__':snapshot()
