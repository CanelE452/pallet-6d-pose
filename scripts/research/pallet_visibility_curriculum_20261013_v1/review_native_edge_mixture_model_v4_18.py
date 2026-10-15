"""Independent sealed-journal checks; no reference, image or fitting calls."""
from copy import deepcopy
from pathlib import Path
import math

import numpy as np

from . import common as C
from . import calibrate_native_edge_mixture_v4_18 as FIT


OUTPUT = FIT.OUTPUT.parent / 'independent_CAL_native_edge_mixture_model_v4_18'


def _nll(variance, groups, weights):
    """Direct density computation, separate from the fitter implementation."""
    value = 0.
    for components, weight in zip(groups, weights):
        logits = []
        for component in components:
            covariance = variance + component['floor']
            squared = float(np.sum(component['error']**2/covariance))
            robust = 18.*(math.sqrt(1.+squared/9.)-1.)
            logits.append(-.5*float(np.log(covariance).sum())-.5*robust-math.log(len(components)))
        peak = max(logits)
        value -= 2.*weight*(peak+math.log(sum(math.exp(x-peak) for x in logits)))
    return float(value)


def main(output=OUTPUT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve every independent review')
    folder = FIT.OUTPUT
    seal = C.read(folder/'MODEL_SEAL.json')
    names = {'MODEL.json':'model_sha256', 'CAL_MIXTURE_ERRORS.jsonl.gz':'CAL_errors_sha256',
        'CAL_LOGO_REPORT.json':'LOGO_sha256', 'PREPARATION_SEAL.json':'preparation_seal_sha256'}
    for name, key in names.items():
        if C.sha(folder/name) != seal[key]:
            raise ValueError('Completed model evidence seal mismatch: '+name)
    preparation = C.read(folder/'PREPARATION_SEAL.json')
    if (C.sha(folder/'PROTOCOL.json') != preparation['protocol_sha256']
            or C.sha(folder/'CAL_AUTHENTICATED_INPUTS.jsonl.gz') != preparation['CAL_inputs_sha256']):
        raise ValueError('Prefit source/input seal mismatch')
    rows = list(C.rows(folder/'CAL_MIXTURE_ERRORS.jsonl.gz'))
    model = C.read(folder/'MODEL.json')
    logo = C.read(folder/'CAL_LOGO_REPORT.json')
    if len(rows) != 46 or sum(r['compatible'] is True for r in rows) != 44:
        raise ValueError('Complete CAL46/compatible44 inventory required')
    models = [model]+[f['model'] for f in logo['LOGO_folds']]
    results = []
    for current in models:
        exported = FIT.mixture_noise_parameters(current)
        _, inventory, eligible, train, sessions, groups, weights = FIT._training_groups(
            rows, inventory_ids=current['inventory_ids'], heldout_session=current['heldout_session'])
        if (inventory != current['inventory_ids'] or eligible != current['compatible_training_ids']
                or [r['id'] for r in train] != current['fit_ids']
                or sessions != current['fit_sessions']
                or len(train) != current['fit_frame_count']
                or len(groups) != current['fit_physical_edge_count']
                or sum(map(len,groups)) != current['fit_alternative_count']):
            raise ValueError('Full/fold training population changed')
        if abs(weights.sum()-1.) > 1e-12:
            raise ValueError('Session/frame/edge weights do not sum to one')
        for session in sessions:
            expected = 1./len(sessions)
            actual = sum(1./(len(sessions)*sum(r['session']==session for r in train))
                for r in train if r['session']==session)
            if abs(actual-expected)>1e-12:
                raise ValueError('Unequal session calibration weight')
        variance = np.array([current['normal_variance_component_px2'],current['angular_variance_component_sine2']])
        if not np.allclose(np.sqrt(variance),[current['sigma_normal_px'],current['sigma_sine']],atol=1e-14,rtol=1e-13):
            raise ValueError('Variance/standard deviation export mismatch')
        nll = _nll(variance,groups,weights)
        if abs(nll-current['training_mixture_nll'])>1e-11:
            raise ValueError('Independent robust mixture NLL mismatch')
        if (exported['normal_variance_component_px2'] != variance[0]
                or exported['angular_variance_component_sine2'] != variance[1]
                or exported['noise_provenance']['observation_family'] != FIT.MIX.NOISE_OBSERVATION_FAMILY):
            raise ValueError('Runtime export must contain the same components only')
        floor_checks = 0
        for row in rows:
            if row.get('compatible') is not True:
                continue
            for group in row['groups']:
                for component in group['components']:
                    span = component['actual_inlier_span_px']
                    fn = max(1.,component['normal_floor_px'])
                    independently_computed = np.array([fn*fn,2.*fn*fn/(span*span+2.*fn*fn)])
                    if not np.allclose(independently_computed,component['conditional_variance_floor'],atol=1e-14,rtol=1e-12):
                        raise ValueError('Runtime conditional floor changed')
                    floor_checks += 1
        chosen = current['optimizer_searches'][current['selected_optimizer_start_index']]
        gradient_scaled = np.asarray(chosen['gradient_wrt_variance'])*np.asarray(current['optimizer_conditioning_variance_scales'])
        results.append(dict(heldout_session=current['heldout_session'],fit_frames=len(train),
            physical_groups=len(groups),alternatives=sum(map(len,groups)),sigma_normal_px=current['sigma_normal_px'],
            sigma_sine=current['sigma_sine'],independent_nll=nll,nll_absolute_difference=abs(nll-current['training_mixture_nll']),
            floors_checked=floor_checks,selected_optimizer_success=chosen['success'],
            scaled_gradient_max=float(np.abs(gradient_scaled).max()),
            all_optimizer_searches_success=all(s['success'] for s in current['optimizer_searches'])))
    bindings = {str((folder/name).relative_to(C.ROOT)):C.sha(folder/name)
        for name in (*names,'MODEL_SEAL.json','PROTOCOL.json','CAL_AUTHENTICATED_INPUTS.jsonl.gz')}
    sources = {str(Path(p).resolve().relative_to(C.ROOT)):C.sha(Path(p))
        for p in (__file__,FIT.__file__,FIT.MIX.__file__,FIT.MIX.M.__file__)}
    output.mkdir(parents=True)
    report = dict(schema='INDEPENDENT_CAL_NMS_MIXTURE_MODEL_REVIEW_V4_18',passed=True,
        input_bindings=bindings,source_sha256=sources,full_and_four_fold_checks=results,
        full_model_sigma_normal_px=model['sigma_normal_px'],full_model_sigma_sine=model['sigma_sine'],
        fit_inventory=dict(all_frames=46,compatible_frames=44,supported_frames=model['fit_frame_count'],
            physical_groups=model['fit_physical_edge_count'],alternatives=model['fit_alternative_count']),
        material_open_findings=[],limitations=[
            'Wood-held LOGO selected optimizer reports success=False despite near-identical converged-start NLL; do not claim all folds converged.',
            'Uniform component mass can depend on distinct subset paths; no independent pixel/alternative probability or future coverage claim.',
            'Noise is conditional on ORIGINAL D-merged old-partial applicability and the frozen image-NMS observer; no general sensor calibration claim.'],
        new_reference_decodes=0,new_RGB_or_NN_calls=0,new_noise_or_pose_fits=0,
        production_source_model_or_prediction_changes=0)
    C.write(output/'REVIEW.json',report)
    C.write(output/'REVIEW_SEAL.json',dict(review_sha256=C.sha(output/'REVIEW.json'),source_sha256=sources,input_bindings=bindings))
    print(dict(output=str(output),passed=True,models_checked=len(results),max_nll_difference=max(r['nll_absolute_difference']for r in results)))


if __name__=='__main__':
    main()
