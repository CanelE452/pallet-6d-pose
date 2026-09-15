"""Read-only repeated-forward audit of the frozen prior, outside latency timing."""
import numpy as np,torch
from env import *
from prior_inference import PriorInference
from prior_model import expectation
def run():
    verify();assert not gpu()['foreign_compute'];torch.set_num_threads(4)
    er=old('evaluate_real');_,cache,_=er.baseline_inputs(LINE)
    key=read(OLD_DOC/'RUNTIME_PROTOCOL.json')['keys'][0];im=er.load_bgr(key,cache['frame_metadata'][key])
    m=PriorInference(1);saved={r['image_key']:r['prediction'] for r in read(RAW/'evaluation/PRIOR1/IMAGE_PREDICTIONS.json')['records']}[key]
    captured=[];hook=m.head.register_forward_pre_hook(lambda _,args:captured.append(tuple(x.detach().clone() for x in args)))
    p=m.predict(im);hook.remove();er.check_prediction(p,cache['frames'][key],frame_key=key)
    baseline=tuple(captured[0]);rows=[];reference=None
    with torch.no_grad(),torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False,allow_tf32=False):
        for i in range(5):
            logits,layers=m.head(*baseline,return_layers=True);v={k:t.cpu().numpy().copy() for k,t in layers.items()};q=expectation(logits).cpu().numpy()
            if reference is None:reference=(v,q)
            rows.append(dict(repeat=i,layers_max_abs_vs_first={k:float(np.max(np.abs(x-reference[0][k]))) for k,x in v.items()},crop_coordinate_max_abs_vs_first=float(np.max(np.abs(q-reference[1])))))
    delta=float(np.max(np.abs(np.asarray(p['candidates'][p['selected_index']]['keypoints_xy'])-np.asarray(saved['candidates'][saved['selected_index']]['keypoints_xy']))))
    write(DOC/'RUNTIME_FLOAT_DIAGNOSTIC.json',dict(time=now(),frame=key,checkpoint=bound(RAW/'runs/PRIOR1/last.pt'),input_tensors_fixed_across_repeats=True,model_eval=True,cudnn_deterministic=False,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,repeat_deltas=rows,saved_prediction_delta_original_px=delta,baseline_contract_exact=True,no_weight_or_data_or_metric_change=True))
    m.close();print('RUNTIME_FLOAT_DIAGNOSTIC',rows,'saved_delta',delta,flush=True)
if __name__=='__main__':run()
