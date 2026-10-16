import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'pallet_correspondence_v33'))
import context33 as V33
from context33 import *
KIT=Path(__file__).resolve().parent
RUN=ROOT/'data/pallet/results/pallet_line_validation_v34'
NOTE=ROOT/'_docs/notes/point_line_fusion/04_failure_analysis/pallet_line_validation_v34.md'
ARMS=('line_only','pooled')
REPEATS=(0,1)

def deterministic():
    setup();torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    from deterministic34 import install
    install()
    return dict(enabled=torch.are_deterministic_algorithms_enabled(),warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],pooling='equivalent divisible avg_pool2d',torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())

def record(message):
    with (ROOT/'_docs/history/2026-09-10.md').open('a') as f:f.write('\n\n### V34 '+time.strftime('%H:%M:%S')+'\n\n'+message+'\n')
