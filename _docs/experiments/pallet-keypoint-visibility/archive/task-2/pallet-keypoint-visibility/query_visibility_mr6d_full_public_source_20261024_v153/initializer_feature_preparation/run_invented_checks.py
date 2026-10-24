"""Invented input-only math checks before actual intake/model entry."""
from pathlib import Path
import importlib.util,json,hashlib
H=Path(__file__).resolve().parent
for name in ('infer_full230','encode_full230'):
 p=H/(name+'.py');s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 m.GENERATED=H/('EARLY_'+name+'_GENERATED_CHECKS.json')
 if name=='infer_full230':m.generated()
 else:
  parent=m.read(m.PARENT);m.generated(parent,m.operators(parent))
 print(json.dumps({'producer':name,'check':str(m.GENERATED),'sha256':hashlib.sha256(m.GENERATED.read_bytes()).hexdigest()}),flush=True)
