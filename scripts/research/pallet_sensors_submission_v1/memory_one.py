"""Fresh-process isolated allocation measurement; no latency samples replaced."""
import numpy as np,torch,cv2
from env import *
from runtime_numeric import PointInference,DirectInference,PriorInference
from runtime_numeric_checks import coordinates
def run(name):
    torch.set_num_threads(4);assert not gpu()['foreign_compute'];assert torch.cuda.memory_allocated()==0
    plan=read(DOC/'RUNTIME_PROTOCOL.json');assert name in plan['models']
    er=old('evaluate_real');_,cache,_=er.baseline_inputs(LINE);images={k:er.load_bgr(k,cache['frame_metadata'][k]) for k in plan['keys']}
    if name=='R0':m=er.PlainBaseline(R0)
    elif name.startswith('PRIOR'):m=PriorInference(int(name[5:]))
    elif name.startswith('P'):m=PointInference(int(name[1:]))
    elif name.startswith('D'):m=DirectInference(int(name[1:]))
    else:m=old('inference').PalletLinePoseInference(LINE/f'runs/image_line_only_seed{name[1:]}/last.pt',LINE/'SELECTION.json')
    saved={r['image_key']:r['prediction'] for r in read(RAW/f'evaluation/{name}/IMAGE_PREDICTIONS.json')['records']} if name.startswith('PRIOR') else None
    m.predict(images[plan['keys'][0]]);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();numerical=[]
    for k,im in images.items():
        pred=m.predict(im)
        if saved:
            er.check_prediction(pred,cache['frames'][k],frame_key=k)
            for c,r in zip(pred['candidates'],saved[k]['candidates']):numerical.append(dict(model=name,key=k,**coordinates(c,r,name+'/'+k)))
    torch.cuda.synchronize()
    write(RAW/f'runtime/memory_{name}.json',dict(complete=True,model=name,time=now(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),scope='fresh process; one model; 1 warmup then26 untimed actual-image forwards; excludes desktop and other processes',numerical_replay=numerical,numerical_replay_scope='Separate untimed memory phase, not retroactively substituted for timed predictions',threads=dict(torch_intraop=torch.get_num_threads(),torch_interop=torch.get_num_interop_threads(),opencv=cv2.getNumThreads(),OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS'))))
    print('ISOLATED_MEMORY_COMPLETE',name,flush=True)
if __name__=='__main__':run(sys.argv[1])
