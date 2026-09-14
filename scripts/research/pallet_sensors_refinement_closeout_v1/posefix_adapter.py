"""Pallet9 data adapter scaffold. No official PoseFix network inference claim."""
import time,importlib.util
import numpy as np
import cv2
from env import *
def affine(box,shape=(384,288)):
    box=np.asarray(box,float);size=box[2:]-box[:2];assert np.isfinite(box).all() and (size>0).all()
    center=(box[2:]+box[:2])*.5;ratio=shape[1]/shape[0];w,h=size
    if w>h*ratio:h=w/ratio
    else:w=h*ratio
    w*=1.25;h*=1.25
    return np.array([[shape[1]/w,0,shape[1]*(.5-center[0]/w)],[0,shape[0]/h,shape[0]*(.5-center[1]/h)]],float)
def transform(points,matrix):return np.asarray(points)@matrix[:,:2].T+matrix[:,2]
def restore(points,matrix,original):
    result=transform(points,cv2.invertAffineTransform(matrix));result[8]=original[8];return result
def gaussian(points,valid,shape=(384,288),sigma=9):
    yy,xx=np.mgrid[:shape[0],:shape[1]];grid=np.stack([xx,yy],-1)
    return (np.exp(-np.square(grid[:,:,None]-points[None,None]).sum(-1)/(2*sigma*sigma))*valid[None,None]).astype(np.float32)
def run():
    data=dataset();records=[];times=[];maximum=0;examples=[]
    for row in data.train_rows[:8]:
        record=data.source['records'][int(data.indices[row])];assert record['partition']=='train' and record['source_kind']=='synthetic'
        image=cv2.imread(record['image']);assert image is not None and sha(record['image'])==record['image_sha256']
        inp=data.arrays['input_shape'][row];gain,offset=old('features').canvas_affine(record['prepared_shape_hw'],inp)
        points=(data.arrays['points'][row]-offset)/gain;gt=(data.arrays['gt_points'][row]-offset)/gain
        box=((data.arrays['boxes'][row].reshape(2,2)-offset)/gain).reshape(4)
        m=affine(box);begin=time.perf_counter();crop=cv2.warpAffine(image,m,(288,384));q=transform(points,m)
        hm=gaussian(q,data.arrays['point_valid'][row]);rgb=crop[:,:,::-1].astype(np.float32)-np.array([123.68,116.78,103.94],np.float32)
        times.append((time.perf_counter()-begin)*1000);back=restore(q,m,points);error=float(np.max(np.abs(back-points)));maximum=max(maximum,error);assert error<1e-8
        assert hm.shape==(384,288,9) and rgb.shape==(384,288,3)
        # GT and predictor streams remain independent in the actual exported example.
        target=transform(gt,m);valid=data.arrays['gt_valid'][row].copy();valid[8]=False
        assert not np.array_equal(points,gt)
        examples.append(dict(id=record['id'],image_id=int(row),image=record['image'],source_partition='train',bbox=box.tolist(),input_pose=points.tolist(),target_pose=gt.tolist(),target_valid=valid.tolist(),affine=m.tolist(),crop_input_pose=q.tolist(),crop_target_pose=target.tolist()))
        records.append(dict(row=int(row),image_sha256=record['image_sha256'],cache_record_index=int(data.indices[row]),roundtrip_max_px=error,adapter_ms=times[-1],heatmap_shape=list(hm.shape)))
    write(RAW/'prior_adapter/SOURCE_TRAIN_EXAMPLES.json',dict(examples=examples,network_executed=False,input_pose_source='frozen R0 source-train cache, not ground truth'))
    envs=[]
    for directory in sorted(Path('/home/minjae/anaconda3/envs').glob('*')):
        matches=list(directory.glob('lib/python*/site-packages/tensorflow/__init__.py'))
        envs.append(dict(environment=directory.name,tensorflow_package_present=bool(matches)))
    repositories={}
    for name in ('PoseFix_RELEASE','CRT-6D'):
        path=RAW/'external'/name
        repositories[name]=dict(url=subprocess.check_output(['git','-C',str(path),'remote','get-url','origin'],text=True).strip(),commit=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip(),
            license=bound(path/'LICENSE'),tracked_files=subprocess.check_output(['git','-C',str(path),'ls-files'],text=True).splitlines())
    write(DOC/'PRIOR_ADAPTER_TESTS.json',dict(PASS=True,scope='8 actual synthetic source train images, affine and heatmap interface only',network_executed=False,records=records,max_roundtrip_px=maximum,CPU_adapter_median_ms=float(np.median(times)),center_preserved=True,no_DEV_noise_fit=True,repositories=repositories,environments=envs))
    write(DOC/'PRIOR_BASELINE_READINESS.json',dict(status='PRIOR_BASELINE_BLOCKED',data_adapter='SOURCE_TRAIN_8_CASES_PASS',official_network_tested=False,full_training=0,
        reasons=['Official PoseFix uses tensorflow.contrib, TF1 graph APIs and legacy Python/CUDA stack; no compatible isolated runtime verified on this RTX3080.',
        'Pallet9 integration in official network/data loader and pretrained backbone validation remain; adapter tests alone are not network implementation readiness.',
        'CRT-6D official commit contains only LICENSE and README; executable implementation unavailable.'],GPU_time_estimate=dict(status='NOT_MEASURABLE_UNTIL_NETWORK_RUNTIME_READY',hours=None,peak_memory_GB=None,CPU_adapter_median_ms=float(np.median(times)),no_proxy_model_estimate=True),source_commits=repositories))
    write(DOC/'PREDECLARED_PRIOR_BASELINE_PROTOCOL.json',dict(classification='purpose-adapted PoseFix pallet9 comparator, not faithful human-benchmark reproduction',
        source_commit=repositories['PoseFix_RELEASE']['commit'],source_license='MIT; retain original notice when distributing adapted official code',
        keypoints='17 human to 9 pallet; preserve center8 at restoration, exclude center8 loss; no human swap/inversion skeleton priors',
        image_pose_encoding='RGB mean-centered 384x288 crop; aspect-correct predicted bbox x1.25; nine Gaussian input maps sigma9; 96x72 output',
        kept='ResNet152 option, Gaussian pose encoding, heatmap CE plus coordinate L1 and expectation readout',
        changed='predicted pallet boxes and frozen R0 source train poses replace human GT box/human empirical noise; no DEV residual fit, no pose flipping until index-contract verification',
        initialization='Official tf-slim ImageNet resnet_v1_152 checkpoint referenced by README; validate download/hash and expanded input conv before run; no COCO/PoseTrack pose weights by default',
        weights_availability='Official README links Google Drive human-trained PoseFix snapshots and tf-slim ImageNet backbones; linked, not downloaded/verified here',
        additional_supervision='ImageNet class-label pretraining if checkpoint obtained; human keypoint supervision excluded in default adapted comparator',
        matched_budget=dict(seeds=[1,2,3],steps=6000,batch=16,exposures_per_seed=96000,source_order='exact P order arrays',optimizer='official Adam lr0.0005, weight regularization0.00001, default Adam betas0.9/0.999; no added warmup/clip',decay_factor=10,decay_at_updates=[3858,5143],schedule='ceil(6000*90/140) and ceil(6000*120/140); adapted matched-budget schedule, not140 completed epochs'),
        convergence_budget=dict(epochs=140,steps_per_seed=int(np.ceil(len(data.train_rows)/16))*140,separate_additional_budget=True,selection='synthetic selection only, last fixed checkpoint'),
        validation='same cal/selection/heldout membership, fixed P lambda/cap grid with exact old source corner objective and tie; no temperature sweep for coordinate output',
        runtime='full image crop+extra backbone+readout+restoration; include original R0 detector; PnP separate',
        resource_estimate=read(DOC/'PRIOR_BASELINE_READINESS.json')['GPU_time_estimate'],
        next_gate='Isolated compatible runtime, official-network forward/backward+weight loading+operator parity and measured peak VRAM/seconds per update; then bind numerical schedule and code before any full fit',
        interpretation='matched-exposure result cannot establish converged-method superiority; convergence budget is a separate comparator; official baseline performance NOT_YET_MEASURED'))
    print('PRIOR_ADAPTER_PASS_NETWORK_BLOCKED',maximum,flush=True)
if __name__=='__main__':run()
