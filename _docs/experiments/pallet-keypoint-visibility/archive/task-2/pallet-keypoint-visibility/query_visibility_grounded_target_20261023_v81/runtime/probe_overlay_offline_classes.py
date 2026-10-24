"""Exactly three class imports only; no processor/model construction."""
from pathlib import Path
import hashlib,importlib.metadata as md,json,os,sys,traceback

HERE=Path(__file__).resolve().parent
OVERLAY=HERE/'overlay'
assert os.environ.get('HF_HUB_OFFLINE')=='1'
assert os.environ.get('TRANSFORMERS_OFFLINE')=='1'
assert os.environ.get('HF_HUB_DISABLE_TELEMETRY')=='1'
assert Path(sys.path[1]).resolve()==OVERLAY.resolve(),sys.path[:3]

network_attempts=[]
def audit(event,args):
    if event in ('socket.connect','socket.getaddrinfo'):
        network_attempts.append(event)
        raise RuntimeError('Offline import probe forbids network activity')
sys.addaudithook(audit)

def bound_module(module):
    p=Path(module.__file__).resolve()
    return dict(name=module.__name__,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

result=dict(PASS=False,python=sys.executable,python_version=sys.version,
    offline_flags={k:os.environ[k] for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY')},
    overlay=str(OVERLAY),classes={},modules=[],versions={},network_attempts=network_attempts,
    constructors_from_pretrained_modelweights_source_current_reads=0,
    GPU_forward_fit_PnP=0)
try:
    for name in ('huggingface-hub','transformers','tokenizers','safetensors','torch','numpy','pillow'):
        result['versions'][name]=md.version(name)
    import huggingface_hub
    assert huggingface_hub.__version__=='0.36.0'
    assert OVERLAY.resolve() in Path(huggingface_hub.__file__).resolve().parents
    result['modules'].append(bound_module(huggingface_hub))
    import transformers
    assert transformers.__version__=='4.57.6'
    result['modules'].append(bound_module(transformers))
    for name in ('GroundingDinoConfig','GroundingDinoProcessor','GroundingDinoForObjectDetection'):
        cls=getattr(transformers,name)
        module=sys.modules[cls.__module__]
        result['classes'][name]=dict(imported=True,module=cls.__module__,source=bound_module(module))
        if not any(x['name']==module.__name__ for x in result['modules']):result['modules'].append(bound_module(module))
    result['PASS']=True
except Exception as e:
    result.update(exception_type=type(e).__name__,exception_message=str(e),traceback=traceback.format_exc())
print(json.dumps(result,indent=2),flush=True)
sys.exit(0 if result['PASS'] else 1)
