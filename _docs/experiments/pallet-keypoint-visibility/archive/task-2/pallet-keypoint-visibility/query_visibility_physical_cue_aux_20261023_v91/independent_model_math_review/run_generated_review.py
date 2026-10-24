"""Independent V91 code and generated-tensor review; no corpus/weight IO."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import math
import sys
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = ROOT / 'local_cue_evidence_map.py'
MOSAIC = ROOT / 'spatial_mosaic.py'


def binding(p):
    p=Path(p).resolve()
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)


def write(name,d):
    with (HERE/name).open('x') as f:
        json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def sample_border(array,x,y):
    """Independent scalar bilinear with border clamping in pixel-index space."""
    h,w=array.shape
    x=min(max(float(x),0.),w-1.);y=min(max(float(y),0.),h-1.)
    x0,y0=math.floor(x),math.floor(y);x1,y1=min(x0+1,w-1),min(y0+1,h-1)
    dx,dy=x-x0,y-y0
    return ((1-dx)*(1-dy)*float(array[y0,x0])+dx*(1-dy)*float(array[y0,x1])+
            (1-dx)*dy*float(array[y1,x0])+dx*dy*float(array[y1,x1]))


def region_reference(logits,valid,normalized):
    values=[];available=[]
    for z,v,points in zip(logits,valid,normalized):
        out=[];ok=[]
        for gx,gy in points:
            # align_corners=False: normalized0 maps exactly to index4.
            x=(float(gx)+1)*9/2-.5;y=(float(gy)+1)*9/2-.5
            den=sample_border(v,x,y);num=sample_border(z*v,x,y)
            out.append(num/max(den,1e-12));ok.append(den>0)
        values.append(float(np.mean(out)));available.append(all(ok))
    return np.array(values),np.array(available)


def mosaic_reference(logits,valid,q,L,hw):
    h,w=hw;n=np.zeros(hw,np.float64);d=np.zeros(hw,np.float64)
    for y in range(h):
        for x in range(w):
            for z,v,xy,side in zip(logits,valid,q,L):
                x0,y0=xy-side/2
                if not (x0<=x<x0+side and y0<=y<y0+side):continue
                ux=2*(x-float(xy[0]))/float(side);uy=2*(y-float(xy[1]))/float(side)
                tent=max(1-abs(ux),0)*max(1-abs(uy),0)
                ix=(ux+1)*9/2-.5;iy=(uy+1)*9/2-.5
                n[y,x]+=tent*sample_border(z*v,ix,iy)
                d[y,x]+=tent*sample_border(v,ix,iy)
    a=d>0
    return np.where(a,n/np.maximum(d,1e-12),0),a,d


def main():
    scope=dict(schema='V91_INDEPENDENT_GENERATED_MODEL_AND_MOSAIC_REVIEW',created_utc=datetime.now(timezone.utc).isoformat(),
        files=[binding(MODEL),binding(MOSAIC)],scope='Code/static contracts and generated CPU tensors only.',
        forbidden=['source RGB','source feature tensors','source/model checkpoint parameter loading','current data','real-source forward','fit','PnP'],
        independent_oracle='Scalar bilinear border clamp + valid-normalized ratio; scalar native integer-pixel tent sum.',
        generated_numeric_policy=dict(float64_region_abs_max=1e-10,float32_region_abs_max=2e-5,float32_mosaic_abs_max=2e-5,
                                      purpose='Generated-operation floating summation bounds, not data or threshold tolerance.'))
    write('SCOPE_BEFORE_GENERATED_TESTS.json',scope)
    start_model=binding(MODEL);start_mosaic=binding(MOSAIC)
    m=module(MODEL,'v91_reviewed_model');s=module(MOSAIC,'v91_reviewed_mosaic')
    torch.set_num_threads(1);torch.manual_seed(20261023);rng=np.random.default_rng(20261023)
    records={};blockers=[]
    signature=inspect.signature(m.LocalCueEvidenceMap.forward)
    assert list(signature.parameters)==['self','p3_roi','local_rgb96','local_valid96']
    assert m.FORWARD_FIELDS==('p3_roi','local_rgb96','local_valid96')
    model=m.LocalCueEvidenceMap();params=sum(p.numel() for p in model.parameters());assert params==25381
    n=4;x=torch.randn(n,65,9,9);x[:,64]=(torch.rand(n,9,9)>.15)
    rgb=torch.randint(0,256,(n,3,96,96),dtype=torch.uint8);v=torch.rand(n,1,96,96)>.2
    result=model(x,rgb,v)
    assert result['physical_external_logit'].shape==(n,9,9) and not result['physical_external_logit'].any()
    assert result['map_valid'].shape==(n,9,9) and result['map_valid'].dtype==torch.bool
    expected_valid=(F.adaptive_avg_pool2d(v.float(),(9,9))[:,0]>0)&x[:,64].bool()
    assert torch.equal(result['map_valid'],expected_valid)
    assert torch.equal(result['input_available'],expected_valid.flatten(1).any(1))
    zero=model(x*0,rgb,v*False);assert not zero['input_available'].any() and not zero['map_valid'].any()
    rejected=[]
    for name,call in [
        ('GTdict',lambda:model({'reference_xy':torch.zeros(2)},rgb,v)),
        ('GTkwarg',lambda:model(x,rgb,v,reference_xy=torch.zeros(2))),
        ('GTstatekwarg',lambda:model(x,rgb,v,semantic_state=torch.ones(n))),
        ('wrongRGBdtype',lambda:model(x,rgb.float(),v)),
        ('wrongP3shape',lambda:model(x[:,:,:8],rgb,v)),
        ('nonfiniteP3',lambda:model(torch.full_like(x,float('nan')),rgb,v)),
        ('nonbinaryvalid',lambda:model(x,rgb,torch.full_like(v,.3,dtype=torch.float32))),
    ]:
        try:call()
        except (TypeError,ValueError):rejected.append(name)
        else:raise AssertionError('Not rejected: '+name)
    source={k:torch.randn_like(value) for k,value in model.state_dict().items() if k.startswith(('p3_encoder.','local_encoder.'))}
    source['classifier.0.weight']=torch.randn(3,4);source['fusion.0.weight']=torch.full_like(model.fusion[0].weight,777)
    fusion_before=model.fusion[0].weight.detach().clone();transfer=m.initialize_shared_encoders(model,source)
    assert all(torch.equal(model.state_dict()[k],source[k]) for k in transfer['copied'])
    assert torch.equal(model.fusion[0].weight,fusion_before) and not model.evidence.weight.any() and not model.evidence.bias.any()
    assert 'fusion.0.weight' in transfer['excluded_source'] and 'classifier.0.weight' in transfer['excluded_source']
    # Generated nonzero output weights solely to exercise image gradients.
    torch.nn.init.normal_(model.evidence.weight,std=.02)
    xf=x.detach().clone().requires_grad_(True)
    out=model(xf,rgb,v);points=torch.zeros(n,9,2)
    pooled,available=m.region_logit(out['physical_external_logit'],out['map_valid'],points)
    loss=F.binary_cross_entropy_with_logits(pooled,torch.tensor([0.,1.,0.,1.]),reduction='none')
    (loss*available).sum().backward()
    p3grad=float(model.p3_encoder[0].weight.grad.abs().sum());rgbgrad=float(model.local_encoder[0].weight.grad.abs().sum())
    assert p3grad>0 and rgbgrad>0 and torch.isfinite(xf.grad).all()
    records['model']=dict(parameter_count=params,three_tensor_signature=True,GT_metadata_inputs_rejected=rejected,
        zero_output_initialization=True,factual_validity_parity=True,all_invalid_input_unavailable=True,
        copied_shared_encoder_tensors=transfer['copied'],source_fusion_or_classifier_not_copied=True,
        new_fusion_default_random_and_final_output_zero=True,
        generated_nonzero_output_gradient=dict(p3_encoder_abs_sum=p3grad,RGB_encoder_abs_sum=rgbgrad),
        no_actual_weights_or_feature_arrays_loaded=True)
    maxerr64=0.;maxerr32=0.;regioncases=0
    for dtype in (np.float64,np.float32):
        for maskkind in ('all','random','none','singlecell'):
            z=rng.normal(size=(13,9,9)).astype(dtype);valid=rng.random((13,9,9))>.4
            if maskkind=='all':valid[:]=True
            elif maskkind=='none':valid[:]=False
            elif maskkind=='singlecell':valid[:]=False;valid[:,4,4]=True
            points=rng.uniform(-1.3,1.3,(13,9,2)).astype(dtype)
            points[0]=np.array([[-1,-1],[1,1],[0,0],[-8/9,-8/9],[8/9,8/9],[-1.1,.3],[1.1,-.3],[1/9,0],[0,-1/9]],dtype=dtype)
            expected,ea=region_reference(z,valid,points)
            got,ga=m.region_logit(torch.from_numpy(z),torch.from_numpy(valid),torch.from_numpy(points))
            error=float(np.max(np.abs(got.numpy()-expected)))
            assert np.array_equal(ga.numpy(),ea)
            if dtype==np.float64:maxerr64=max(maxerr64,error);assert error<1e-10
            else:maxerr32=max(maxerr32,error);assert error<2e-5
            regioncases+=len(z)
    # A partial factual hole disables ONE region unit; denominator is never a negative label.
    z=torch.ones(1,9,9,requires_grad=True);vv=torch.zeros(1,9,9,dtype=torch.bool);vv[:,4,4]=True
    pp=torch.zeros(1,9,2);pp[:,0]=1
    pool,ok=m.region_logit(z,vv,pp);assert not ok.any()
    (F.binary_cross_entropy_with_logits(pool,torch.ones(1),reduction='none')*ok).sum().backward()
    assert not z.grad.any()
    constant,ca=m.region_logit(torch.full((1,9,9),3.25),torch.ones(1,9,9,dtype=torch.bool),torch.linspace(-1,1,18).reshape(1,9,2))
    assert ca.item() and constant.item()==3.25
    records['region']=dict(independent_scalar_cases=regioncases,max_abs_error_float64=maxerr64,max_abs_error_float32=maxerr32,
        border_and_cell_boundary=True,normalized0_maps_to_center4=True,normalized_2offset_overL=True,
        nine_logits_mean_one_BCE_not_mean9BCE=True,all9_sample_support_required=True,
        unavailable_unit_mask_has_zero_gradient=True,constant_field_exact=True)
    # Independent full native-pixel scalar mosaic on small generated frames.
    q=np.array([[0,0],[8,6],[22,16],[14,10],[-10,-8]],np.float64);L=np.array([8,16,16,8,8])
    mosaicmax=0.;denmax=0.;pixelcases=0
    for kind in ('random','constant','ramp','allinvalid'):
        z=rng.normal(size=(5,9,9)).astype(np.float32);vv=rng.random((5,9,9))>.25
        if kind=='constant':z[:]=2.25
        if kind=='ramp':
            gy,gx=np.meshgrid(np.arange(9),np.arange(9),indexing='ij')
            z[:]=(gx*.2+gy*.3).astype(np.float32)
        if kind=='allinvalid':vv[:]=False
        expected,ea,ed=mosaic_reference(z,vv,q,L,(17,23))
        got,ga,gd=s.make_frame_mosaic(z,vv,q,L,(17,23))
        err=float(np.max(np.abs(got-expected)));derr=float(np.max(np.abs(gd-ed)))
        assert np.array_equal(ga,ea) and err<2e-5 and derr<2e-5
        if kind=='constant':assert np.max(np.abs(got[ga]-2.25))<1e-6
        if kind=='allinvalid':assert not ga.any() and not got.any() and not gd.any()
        mosaicmax=max(mosaicmax,err);denmax=max(denmax,derr);pixelcases+=got.size
    # Analytic 129-tile full frame from the declared grid algorithm, not source manifests.
    qs=[];Ls=[]
    for side in (128,256):
        xs=list(range(0,640,side//2));ys=list(range(0,480,side//2))
        if xs[-1]!=639:xs.append(639)
        if ys[-1]!=479:ys.append(479)
        for y in ys:
            for x0 in xs:qs.append([x0,y]);Ls.append(side)
    assert len(qs)==129
    z=np.full((129,9,9),-1.75,np.float32);vv=np.ones((129,9,9),bool)
    tiled,ta,td=s.make_frame_mosaic(z,vv,np.array(qs),np.array(Ls),(480,640))
    assert ta.all() and np.max(np.abs(tiled+1.75))<2e-6
    # Tile order cannot be score/GT-selected. Real implementation adds every index.
    records['mosaic']=dict(independent_scalar_native_pixels=pixelcases,max_abs_logit_error=mosaicmax,max_abs_den_error=denmax,
        small_frame_random_constant_ramp_invalid=True,fixed_129_algorithm_tiles=True,
        all129_constant_full_frame_covered=True,all129_constant_max_abs=float(np.max(np.abs(tiled+1.75))),
        native_output_grid='integer pixel centres0..W-1, distinct from RGB96 even cell centres',
        exact_formula='sum(tent*bilinear(logit*valid))/sum(tent*bilinear(valid)); den0 has availableFalse',
        no_GT_reference_or_label_argument=True,halfopen_windows_and_zero_tent_at_outer_boundary=True)
    # Final scoring samples completed mosaic; invalid/OOB point values are numeric-safe only.
    yy,xx=np.meshgrid(np.arange(7),np.arange(9),indexing='ij');z=(2*xx+3*yy).astype(np.float32);vv=np.ones_like(z,bool)
    pts=np.array([[0,0],[8,6],[2.5,3.5],[np.nan,1],[-.1,1],[9,2]])
    sampled,sa=s.sample_native_region(z,vv,pts)
    assert sa.tolist()==[True,True,True,False,False,False]
    assert np.max(np.abs(sampled[:3]-[0,34,15.5]))<2e-5
    z[2,2]=999;vv[2,2]=False
    sampled,sa=s.sample_native_region(z,vv,[[2,2],[2.5,2]])
    assert not sa[0] and sa[1] and np.isfinite(sampled).all()
    records['final_mosaic_scoring']=dict(native_integer_and_subpixel_ramp=True,NaN_outside_and_factual_holes_U=True,
        scoring_called_after_completed_GTfree_map=True)
    assert binding(MODEL)==start_model and binding(MOSAIC)==start_mosaic
    report=dict(completed=True,PASS=True,blockers=blockers,model=start_model,mosaic=start_mosaic,
        scope=binding(HERE/'SCOPE_BEFORE_GENERATED_TESTS.json'),review_code=binding(__file__),generated=records,
        static_native_initial_query_compatibility=dict(adapter_created=False,adapter_executed=False,
            input_shapes='N65x9x9 + N3x96x96(uint8) + N1x96x96(valid)',
            center_readout='physical_external_logit[:,4,4] with map_valid[:,4,4]; same nominal native query center for grid9.',
            caveat='This static shape compatibility is not target/native8 semantic association, a calibration claim, current admission, or observed transfer.'),
        mandatory_downstream_integration_notes=[
            'GT stays outside forward. Use only finite safe GT grids/admitted units; region available must gate the BCE and denominator. U/partial/no-support cannot become E or D.',
            'region_logit itself does not consume loss_admitted/weights/labels. Trainer must apply separately frozen view/scale/duplicate normalization; this review does not inspect an actual trainer.',
            'New fusion has default random initialization; only final evidence Conv is zero-initialized. Copied source parts are exactly P3/local encoders, not V88 classifier/fusion/target/context.',
            'map_valid is factual native/RGB support, not semantic target mask. Head has no explicit target instance/native slot identity and does not solve hidden-reference identifiability.',
            'Mosaic interpolation uses fixed border extension and validity normalization, including at native9 outer cells. This is the declared spatial readout, not a proven physical cue/pixel model.',
            'Mosaic expects already validated finite logits, binary map validity, finite integer q and positive even L from fixed metadata; primitive utility validation is not an arbitrary-input security boundary.',
        ],
        no_source_RGB_P3_GT_arrays_or_model_weights_loaded=True,
        generated_random_model_forward_only=True,actual_source_forward_training_current_PnP=0,
        runtime=dict(executable=sys.executable,torch=str(torch.__version__),numpy=np.__version__))
    write('REVIEW.json',report)
    write('REVIEW_SEAL.json',dict(completed=True,PASS=True,files=[binding(p) for p in sorted(HERE.iterdir()) if p.is_file()],
        model=start_model,mosaic=start_mosaic,actual_source_forward_training_current_PnP=0))
    print(json.dumps(dict(report=binding(HERE/'REVIEW.json'),seal=binding(HERE/'REVIEW_SEAL.json')),indent=2),flush=True)


if __name__=='__main__':main()
