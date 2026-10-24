"""No model/processor construction; only installed metadata and class imports."""
import importlib.metadata as metadata
import json,os,sys,traceback

assert os.environ.get('HF_HUB_OFFLINE')=='1'
assert os.environ.get('TRANSFORMERS_OFFLINE')=='1'
assert os.environ.get('HF_HUB_DISABLE_TELEMETRY')=='1'
packages={}
for name in ('transformers','huggingface-hub','tokenizers','safetensors','accelerate',
             'torch','torchvision','pillow','numpy','packaging','timm'):
    try:
        d=metadata.distribution(name)
        packages[name]=dict(version=d.version,metadata_path=str(d._path),
            requires_python=d.metadata.get('Requires-Python'),requires=d.requires)
    except metadata.PackageNotFoundError:
        packages[name]=dict(installed=False)

result=dict(python=sys.executable,version=sys.version,
    offline_flags={k:os.environ[k] for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY')},
    installed_distribution_metadata=packages,imports={},
    processor_model_instantiation_from_pretrained_forward=0,
    source_weights_RGB_GT_current_reads=0,environment_changes=0)
try:
    import transformers
    result['transformers_top_level_import']=True
except Exception as e:
    result.update(PASS=False,transformers_top_level_import=False,
        failure_type=type(e).__name__,failure_message=str(e),traceback=traceback.format_exc(),
        class_import_status='BLOCKED_BY_TRANSFORMERS_TOP_LEVEL_DEPENDENCY_GUARD')
    print(json.dumps(result,indent=2),flush=True)
    sys.exit(1)

for name in ('AutoProcessor','AutoModelForZeroShotObjectDetection',
             'GroundingDinoProcessor','GroundingDinoForObjectDetection'):
    try:
        cls=getattr(transformers,name)
        result['imports'][name]=dict(PASS=True,module=cls.__module__)
    except Exception as e:
        result['imports'][name]=dict(PASS=False,failure_type=type(e).__name__,failure_message=str(e),traceback=traceback.format_exc())
result['PASS']=all(x['PASS'] for x in result['imports'].values())
print(json.dumps(result,indent=2),flush=True)
sys.exit(0 if result['PASS'] else 1)
