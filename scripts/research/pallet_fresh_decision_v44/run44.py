from context44 import *
import argparse,subprocess
PHASES=dict(render=('render44.py','RENDER_COMPLETE.json'),prepare=('prepare44.py','DATA_COMPLETE.json'),parity=('features44.py','FEATURE_PARITY.json'),infer=('infer44.py','INFERENCE_COMPLETE.json'),replay=('replay44.py','UPSTREAM_REPLAY.json'),train=('train44.py','TRAINING_COMPLETE.json'),evaluate=('evaluate44.py','EVALUATION_COMPLETE.json'),audit=('audit44.py','AUDIT.json'),report=('report44.py','REPORT_COMPLETE.json'),deliver=('deliver44.py','COMPLETION.json'))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['all',*PHASES],default='all');args=parser.parse_args();register();(RUN/'logs').mkdir(exist_ok=True)
 for phase,(script,marker) in PHASES.items():
  if args.phase not in ('all',phase):continue
  if args.phase=='all' and (RUN/marker).exists():continue
  env=dict(os.environ)
  if phase in ('render','prepare','infer','replay'):
   compat=env.get('PALLET_V44_DRIVER_LIBS','/tmp/pallet-v44-cuda-libs')
   if Path(compat).exists():env['LD_LIBRARY_PATH']=compat+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
  if phase=='render':env.update(PYOPENGL_PLATFORM='egl',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
  with (RUN/'logs'/f'{phase}.log').open('w') as log:subprocess.run([str(RENDER_PY if phase=='render' else PY),str(KIT/script)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':main()
