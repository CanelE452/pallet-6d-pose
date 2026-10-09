"""Actual unchanged N3 inference at SUBPIX input coordinates, with a Base cap."""
import copy
import hashlib
import sys
import numpy as np
from . import common as C


def replace_coordinates(captured, points):
    """Copy only detection metadata; preserve the exact shared feature tensors."""
    changed = dict(captured)
    changed['candidates'] = copy.deepcopy(captured['candidates'])
    index = captured['selected_index']
    if index is not None:
        changed['candidates'][index]['keypoints_xy'] = np.asarray(points, np.float64).copy()
    for key in ('p3','p4'):
        assert changed[key] is captured[key]
    return changed


def tensor_hash(value):
    a = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


class InputTrace:
    """Observe one actual N3 forward; no extra learned inference or GT access."""
    def __init__(self, inf, head, captured, points):
        self.inf, self.head, self.captured, self.points = inf, head, captured, points
        self.feature = inf.E.old('features')
        self.generic = sys.modules[head.__class__.__mro__[1].__module__]
        self.result = dict(head_calls=0, sample_calls=0, sample_records=[], GT_inputs=False,
            original_shared_features=True, stale_point_cache_used=False)
        self.handles = []; self.actual_output = None

    def __enter__(self):
        import torch
        self.original_branch = self.feature.branch_inputs
        self.original_sample = self.generic.sample
        def branch(captured, *args, **kwargs):
            inputs = self.original_branch(captured, *args, **kwargs)
            if inputs is not None:
                gain, offset = self.feature.canvas_affine(captured['canvas_shape'], captured['input_shape'])
                expected = ((self.points+captured['added_border'])*gain+offset).astype(np.float32)
                assert np.array_equal(inputs['points'], expected, equal_nan=True)
                self.result.update(branch_input_points=inputs['points'], input_support_mask=inputs['point_valid'],
                    gain=float(gain), affine_offset=offset, raw_to_network_applied_once=True)
                self.inputs = inputs
            return inputs
        def before(head, args, kwargs):
            self.result['head_calls'] += 1
            p3,p4,points,boxes,valid,shape = args
            assert p3 is self.captured['p3'] and p4 is self.captured['p4']
            np.testing.assert_array_equal(points[0].detach().cpu().numpy(), self.inputs['points'])
            np.testing.assert_array_equal(boxes[0].detach().cpu().numpy(), self.inputs['boxes'])
            np.testing.assert_array_equal(valid[0].detach().cpu().numpy(), self.inputs['point_valid'])
            self.points_tensor = points.to(head.role_embedding.weight.dtype)
            self.boxes_tensor = boxes.to(head.role_embedding.weight.dtype)
            self.valid_tensor = self.generic.finite(self.points_tensor, valid)
            self.safe = torch.where(self.valid_tensor[...,None], self.points_tensor, torch.zeros_like(self.points_tensor))
            box_valid = torch.isfinite(self.boxes_tensor).all(-1) & (self.boxes_tensor[:,2:] > self.boxes_tensor[:,:2]).all(-1)
            self.bb = torch.where(box_valid[:,None], self.boxes_tensor, self.boxes_tensor.new_tensor([0,0,1,1]))
            size = self.bb[:,2:]-self.bb[:,:2]
            self.diag = size.norm(dim=-1).clamp_min(1)
            self.center = (self.bb[:,2:]+self.bb[:,:2])*.5
            self.locations = self.safe[:,:8,None] + (self.diag[:,None,None]*head.displacements[None])[:,None,:-1]
            self.expected_positions = self.locations[:,:,:,None] + self.diag[:,None,None,None,None]*head.stencil_fraction*head.stencil[None,None,None]
            self.result.update(head_input_points_hash=tensor_hash(points), head_input_matches_SUBPIX=True,
                input_shape=shape[0].detach().cpu().tolist(), context=kwargs['context'][0].detach().cpu().tolist(),
                shared_feature_dtype=str(p3.dtype), candidate_centers_first=self.locations[0,:,0].detach().cpu().tolist(),
                candidate_centers_shape=list(self.locations.shape), candidate_centers_sha256=tensor_hash(self.locations))
        def sample(feature, positions, shape, stride):
            error = float((positions-self.expected_positions).abs().max().item())
            assert error == 0, ('Candidate feature samples did not recenter on actual N3 input', error)
            self.result['sample_calls'] += 1
            self.result['sample_records'].append(dict(stride=stride, positions_shape=list(positions.shape),
                positions_sha256=tensor_hash(positions), formula_max_abs_error=error))
            return self.original_sample(feature, positions, shape, stride)
        def scorer(head, args):
            pooled_width = 2*self.head.patch_body[0].out_channels
            own = (self.safe[:,:8]-self.center[:,None])/self.diag[:,None,None]
            actual = args[0][...,pooled_width:pooled_width+2]
            expected = own[:,:,None].expand(-1,-1,actual.shape[2],-1)
            error = float((actual-expected).abs().max().item())
            assert error == 0, ('Box-relative context did not use actual N3 input', error)
            self.result.update(box_relative_own=own[0].detach().cpu().tolist(), box_context_formula_error=error)
        def after(head, args, output):
            self.actual_output = output
            self.result.update(logits_sha256=tensor_hash(output['logits']),
                output_support=output['point_support'][0].detach().cpu().tolist())
        self.feature.branch_inputs = branch; self.generic.sample = sample
        self.handles = [self.head.register_forward_pre_hook(before, with_kwargs=True),
            self.head.scorer.register_forward_pre_hook(scorer), self.head.register_forward_hook(after)]
        return self

    def __exit__(self, *args):
        self.feature.branch_inputs = self.original_branch
        self.generic.sample = self.original_sample
        for handle in self.handles: handle.remove()


def reverse_captured(inf, head, captured, dims, order, temperature, rule, raw_hw, norm, gray, *, diagnostics=False):
    """S(q0) -> cap_q0 -> actual N3(qS) -> cap_q0, in native numbering."""
    assert rule['max_move_image_diagonal_fraction'] == .01 and rule['lam'] == 1.
    index = captured['selected_index']
    if index is None:
        result, _ = inf.predict_captured(head,'N3_DIM_SYM',captured,dims,order,temperature,rule,raw_hw,norm)
        return result, dict(algorithm_corner_calls=0,N3_head_used=False,no_detection=True), None
    q0 = np.asarray(captured['candidates'][index]['keypoints_xy'],np.float64)
    support = np.isfinite(q0).all(-1) & ~(q0 == -1).all(-1)
    qS_native, subdiag = C.correct(gray,q0.copy(),support.copy(),'SUBPIX')
    h,w = raw_hw
    assert list(gray.shape) == [h,w]
    qS = C.cap_points(q0,qS_native,w,h,support)
    shifted = replace_coordinates(captured,qS)
    if diagnostics:
        with InputTrace(inf,head,shifted,qS) as trace:
            prediction, native_diag = inf.predict_captured(head,'N3_DIM_SYM',shifted,dims,order,temperature,rule,raw_hw,norm)
        trace_data = trace.result
        assert trace_data['head_calls'] == int(prediction['head_used'])
        if prediction['head_used']:
            assert trace_data['sample_calls'] == 2
            output = trace.actual_output
            # Diagnostics of the internal clamp from the actual logits; no extra
            # candidate/head/F execution, no alternative method is evaluated.
            import torch
            probability = (output['logits']/float(temperature)).softmax(-1)
            raw_delta = (probability[...,None]*output['candidate_displacements'][:,None]).sum(-2)*float(rule['lam'])
            n3_unclamped_move = torch.linalg.vector_norm(raw_delta,dim=-1)[0].detach().cpu().numpy()/trace.inputs['gain']
            internal_active = (n3_unclamped_move > .01*np.hypot(w,h)) & np.asarray(trace_data['output_support'],bool)
        else:
            n3_unclamped_move = np.zeros(8); internal_active = np.zeros(8,bool)
    else:
        prediction, native_diag = inf.predict_captured(head,'N3_DIM_SYM',shifted,dims,order,temperature,rule,raw_hw,norm)
        trace_data = None; n3_unclamped_move = None; internal_active = None
    qSN = np.asarray(prediction['candidates'][index]['keypoints_xy'],np.float64)
    qFinal = C.cap_points(q0,qSN,w,h,support)
    assert np.array_equal(qFinal[8],q0[8],equal_nan=True)
    assert np.array_equal(qFinal[~support],q0[~support],equal_nan=True)
    assert np.max(np.linalg.norm(qFinal[:8][support[:8]]-q0[:8][support[:8]],axis=-1),initial=0) <= .01*np.hypot(w,h)+1e-10
    inf.preservation(captured['candidates'],prediction['candidates'],index)
    prediction = dict(prediction); prediction['candidates'] = copy.deepcopy(prediction['candidates'])
    prediction['candidates'][index]['keypoints_xy'] = qFinal
    for key in ('p3','p4'): assert shifted[key] is captured[key]
    cap = .01*np.hypot(w,h)
    usable = support[:8]
    lengths_before = np.linalg.norm(qSN[:8]-q0[:8],axis=-1)
    diag = dict(algorithm_corner_calls=subdiag['algorithm_corner_calls'],N3_head_used=prediction['head_used'],
        SUBPIX=subdiag, q0=q0, qS_native=qS_native, qS=qS, qSN=qSN, qFinal=qFinal,
        cap_px=cap, SUBPIX_cap_active8=(np.linalg.norm(qS_native[:8]-q0[:8],axis=-1)>cap)&usable,
        N3_internal_unclamped_move_px8=n3_unclamped_move,N3_internal_cap_active8=internal_active,
        N3_move_from_SUBPIX_px8=np.linalg.norm(qSN[:8]-qS[:8],axis=-1),
        final_cap_active8=(lengths_before>cap)&usable,total_before_final_cap_px8=lengths_before,
        total_final_move_px8=np.linalg.norm(qFinal[:8]-q0[:8],axis=-1),
        input_changed_corners8=np.any(qS[:8]!=q0[:8],axis=-1), input_trace=trace_data,
        metadata_preserved=True, shared_features_preserved=True, GT_inputs=False)
    return prediction,diag,qSN
