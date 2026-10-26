"""Minimal tensor API for Root's four fixed SYN-only spatial-head fits.

No data discovery, image/model loading, current evaluation, or settings search.
fit_tensors reads numeric token rows only for the supplied FIT indices.
Caller must freeze both LOC checkpoints before SYN DEV crop comparison, and
all four final checkpoints/configurations before current development inference.
"""
import hashlib
import json
import numpy as np
import torch


def state_sha(state):
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()


def fixed_targets(spatial, input_available, direct_cue_known, direct_cue,
                  identity_known, identity_correct, certified_state,
                  reference_xy, q0, crop_side):
    """LOCATION states are not semantic VIS: D0/E1/S2/U-1 teacher namespace.

    Root defines NO_MATCH as no directly observable point of this native,
    not object absence. E/S never supervise an amodal reference heatmap.
    """
    if certified_state.shape != (len(q0),) or not bool(((certified_state>=-1)&(certified_state<=2)).all()):
        raise ValueError('Certified teacher D0/E1/S2/U-1 sidecar only')
    is_D=certified_state==0
    loc_valid, uv, in_range = spatial.fixed_loc_mask(
        input_available, direct_cue_known & direct_cue & is_D,
        identity_known & identity_correct, reference_xy, q0, crop_side)
    state = torch.full((len(q0),), -1, dtype=torch.int64, device=q0.device)
    no_direct=(input_available & identity_known & identity_correct &
               ((certified_state==1)|(certified_state==2)))
    state[no_direct] = spatial.LOC_NO_MATCH
    outside = (input_available & identity_known & identity_correct & is_D &
               direct_cue_known & direct_cue & torch.isfinite(uv).all(1) & ~in_range)
    state[outside] = spatial.LOC_OUT_OF_SEARCH
    state[loc_valid] = spatial.LOC_MATCH
    return loc_valid, uv, state


def token_support_from_valid(valid96):
    """Input-only nonempty support for each fixed 6x6 RAW96 patch cell."""
    if valid96.dtype != torch.bool or valid96.shape[1:] != (1,96,96):
        raise ValueError('bool N1x96x96 original crop validity')
    return torch.nn.functional.avg_pool2d(valid96.float(), 6, 6)[:,0] > 0


def extract_spatial(model, normalized_full_views):
    """Official frozen block8 normalized patch tokens; no CLS/head inputs."""
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Official frozen eval backbone required')
    with torch.inference_mode():
        tokens = model.get_intermediate_layers(normalized_full_views, n=[8],
                  reshape=False, return_class_token=False, norm=True)[0]
    if tokens.shape[1:] != (256,384) or not bool(torch.isfinite(tokens).all()):
        raise ValueError('Finite N256x384 intermediate spatial representation')
    return tokens.to(torch.float16)


def fit_tensors(spatial, tokens, support, native_slot, fit, reference_uv, loc_valid,
                visibility_target, location_target, mode, device='cuda:0'):
    """One fixed 1000-update arm; fake or Root-authorized SYN tensors only.

    tokens may be a memory-mapped NumPy NPY. Only FIT batch rows are indexed.
    All rows stay in the original denominator, including unavailable/U rows.
    Targets -1 are masked, not resampled. Both crop arms use the same PCG17 rows.
    """
    if mode not in ('LOC','VIS','JOINT'):
        raise ValueError('Root fixed LOC/VIS/JOINT only')
    fit = np.asarray(fit, dtype=bool)
    indices = np.flatnonzero(fit)
    n = len(fit)
    if not len(indices) or tuple(tokens.shape) != (n,256,384):
        raise ValueError('Nonempty FIT and N256x384 tokens')
    if tuple(support.shape) != (n,16,16):
        raise ValueError('Input-only N16x16 support')
    torch.manual_seed(17)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = spatial.SpatialPointHead().to(device)
    initial = state_sha(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = np.random.Generator(np.random.PCG64(17))
    draws = indices[generator.integers(0,len(indices),size=(1000,32))]
    rows_hash = hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()
    logs = []
    model.train()
    for step, batch in enumerate(draws, 1):
        x = torch.as_tensor(np.array(tokens[batch],copy=True),device=device)
        active = torch.as_tensor(np.array(support[batch],copy=True),dtype=torch.bool,device=device)
        uv = torch.as_tensor(np.asarray(reference_uv)[batch],dtype=torch.float32,device=device)
        loc = torch.as_tensor(np.asarray(loc_valid)[batch],dtype=torch.bool,device=device) & active.any(2).any(1)
        vis = torch.as_tensor(np.asarray(visibility_target)[batch],dtype=torch.int64,device=device)
        state = torch.as_tensor(np.asarray(location_target)[batch],dtype=torch.int64,device=device)
        # Unavailable input does not silently supervise a zero-filled crop.
        vis = torch.where(active.any(2).any(1),vis,torch.full_like(vis,-1))
        state = torch.where(active.any(2).any(1),state,torch.full_like(state,-1))
        optimizer.zero_grad(set_to_none=True)
        slot = torch.as_tensor(np.asarray(native_slot)[batch],dtype=torch.int64,device=device)
        output = model(x,active,slot)
        loss, parts = spatial.losses(output,uv,loc,vis,state,mode)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite fixed fit; no automatic retry')
        loss.backward()
        optimizer.step()
        logs.append({'step':step,'loss':float(loss.detach()),
                     **{k:float(v.detach()) for k,v in parts.items()},
                     'LOC_valid':int(loc.sum()),'VIS_known':int((vis>=0).sum()),
                     'location_state_known':int((state>=0).sum())})
    model.eval()
    result = {'completed':True,'PASS':True,'mode':mode,'seed':17,
              'optimizer_updates':1000,'batch_size':32,'learning_rate':1e-3,
              'weight_decay':1e-4,'head_parameters':sum(p.numel() for p in model.parameters()),
              'initial_state_sha256':initial,'final_state_sha256':state_sha(model.state_dict()),
              'FIT_rows':len(indices),'draws_sha256':rows_hash,
              'DEV_numeric_rows_read_by_fit':0,'current_or_held_rows_read':0,
              'LOC_state_is_not_VIS':True,'new_learned_offset_head':False}
    return model, result, logs, draws


def infer_tensors(spatial, model, tokens, support, native_slot, indices, batch_size=64):
    """Caller supplies indices only after its required checkpoint freeze barrier."""
    model.eval()
    device = next(model.parameters()).device
    outputs = {}
    with torch.inference_mode():
        for start in range(0,len(indices),batch_size):
            batch = np.asarray(indices[start:start+batch_size],dtype=np.int64)
            out = model(torch.as_tensor(np.array(tokens[batch],copy=True),device=device),
                        torch.as_tensor(np.array(support[batch],copy=True),dtype=torch.bool,device=device),
                        torch.as_tensor(np.asarray(native_slot)[batch],dtype=torch.int64,device=device))
            for key,value in out.items():
                outputs.setdefault(key,[]).append(value.cpu())
    return {k:torch.cat(v) for k,v in outputs.items()}
