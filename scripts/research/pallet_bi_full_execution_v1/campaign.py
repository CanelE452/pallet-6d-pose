"""Immutable contracts for the 2026-10-19 full execution campaign.

This is a new campaign; no previous campaign permission/receipt is copied.
Scientific configuration changes require a separately archived amendment.
"""
from pathlib import Path
import argparse
import hashlib
import json

SEEDS = (20261018, 20261019, 20261020)
ARMS = ('S1_UNION', 'S1_MATCHED_BASELINE', 'S2_C0', 'S2_CP', 'S2_CB',
        'S3_COMMON', 'S3_BI', 'S4_PL', 'S4_CONTOUR', 'S4_IMAGE',
        'F_COMMON', 'F_BI')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def write(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n'
    if p.exists():
        if p.read_text() != data: raise ValueError('Immutable contract changed: '+str(p))
    else: p.write_text(data)

def freeze(run, instructions):
    plan = dict(schema='PALLET_BI_FULL_EXECUTION_CAMPAIGN_V1',
        base_sha='3553202a3682faeb7be2e7df051b230683965382',
        instruction_sha256=sha(instructions), stages=['P0','S1','D_QCOV','S2','S3','S4','F','R'],
        seeds=list(SEEDS), representative_seed=SEEDS[0],
        populations=dict(real=238, PRIMARY=192, TEST=49, REMAINDER=143, CAL=46,
                         synthetic=1003, train=768, validation=235, independent_real_test_n=0),
        resources=dict(CPU_workers=3, BLAS_threads_per_worker=1, GPU_concurrent_jobs=1,
                       expensive_producer_timing_rows=24, timing_selection='first12sortedIDs per real/synthetic; no accuracy selection'),
        H1=dict(arms=list(ARMS), max_actual_fits=36, input_dim=188, parameters=16321,
                architecture='188->64ReLU->64ReLU->1', objective='frame-balanced multi-positive/all-zero BCE',
                optimizer='AdamW', learning_rate=.001, weight_decay=.0001, batch_frames=32,
                max_epochs=50, patience=5, checkpoint_selection='strict minimum own validation loss',
                normalizer='own eligible TRAIN actions only; no real/VAL fit', device='CPU1',
                reuse='only exact consumed features/labels/normalizer/protocol/checkpoint binding'),
        admission=dict(old_D1='unchanged original eligibility including existing N0/E0 exception',
                       new_complete_actions='producer available, finite proper rotation atol1e-6/det1e-6, all original-native depths>1e-5',
                       dedup='R maxabs1e-7; t maxabs1e-7*objectscale, no C2 merging',
                       features='original31 signedlog+31valid; recompute5bankgaps/all188 bank summaries',
                       feature_reuse_tolerance=1e-10, missing='original missing+valid masks, never invented zero evidence'),
        S1=dict(PF='frozen checkpoint b9670d1a42db8d4e80ae1efe55eb237f05f81f7bd089cefe4051bbedd7d0a581',
                anchors=['PF','YOLO'], old_baseline_retraining='only if old admission/features change; otherwise exact reuse'),
        D_QCOV=dict(observation_seed=SEEDS[0], cells=['Q0S0','Q1S0','Q0S1','Q1S1'],
                    max_family_solves=1904, backend='previous limited solver, not original hierarchical120',
                    strict='same physical source association, fixed deduplicated factors, partial archives unchanged',
                    operational_counterfactual='freeze COMMON logical factor graph; transport only existing native2D q/cov with explicit donor provenance; missing donor COMMON fallback',
                    source_changes='separate stratum; logical graph IDs do not certify physical evidence identity',
                    other_observation_seeds='stored observation reaggregation only'),
        S2=dict(base='LOCAL COMMON observation seed20261018', groups=['C0','CP','CB'],
                q_dimension='fixed', partials='fixed COMMON normal-only/typed partial graph; original full producer adapter explicitly registered',
                calibration='zero-centered weighted80percent ellipse coverage; session/frame/point equal weighting; 60bisection',
                positive_definite_floor_px2=1, quantile=.8, pooled='single isotropic additive total-error floor',
                role='B/I raw isotropic floors shrunk n/(n+20) toward pooled; empty role pooled',
                pseudo_count=20, reliability_weight=1, bias_correction=False,
                internal_validation='4 recording-group leave-one-out, training-only scene scale, clip .5..2',
                real_reference_scope='CAL46 only for calibration; PRIMARY forbidden to fit/policy',
                matched_C0='new full candidate production plus3H1 unless exact stored result exists'),
        S3=dict(asset='registered structured textured proxies; exact physical CAD/owner accuracy uncertified',
                registration='bboxcenter; thin->Y; long/short to known physical footprint rank; proper fixed rotation; axiswise dimension scale; no GT',
                max_edges=512, crease_degrees=35, visibility_samples_per_edge=9,
                z_tolerance='max(.005m,.005*depth)', search_radius_px=8, min_support=3,
                line_RMS_max_px=3, tangent_difference_max_deg=35,
                conditional_variance_px2='4 + measurement RMS squared',
                junction_angle_min_deg=30, junction_distance_max_px=8,
                COMMON='minimum normalized support/normal-consistency/uncertainty cue cost; RGB tie first; fullRGB+tightmask jointly',
                BI='B mask/tight first then RGB/full; I RGB/full first then mask/tight; same adoption gates/evidence access',
                dimensions='actual mesh support: two independent incident directions ->2Djunction, one ->1Dnormal, zero ->none',
                dedup='same raw source IDs once; junction replaces underlying 1D factors; archives retained',
                policies=['cue adoption','original/tight/full recovery order','evidence-driven dimension','common/role covariance'],
                generic_support='object_xyz/segment/support_id/visible raw coordinates; never relabel proxy edge as cuboid corner'),
        S4=dict(groups=['PL','CONTOUR','IMAGE'], actual_starts_max=2, updates_per_start_max=4,
                starts='N0, first distinct properpositive E0, then PF_PFanchor, then PF_YOLOanchor; no WD basis duplication',
                same_physical_starts=True, render='native frame EGL device1, textured mesh RGB/depth/mask',
                IMAGE='OpenCV4.9 DIS FAST grayscale render-to-observation, bidirectional',
                image_input='whole native grayscale image; rendered background composited from observation outside render mask',
                flow_grid_px=8, max_correspondences=2048, min_correspondences=12,
                flow_forward_backward_max_px=2, flow_displacement_max_px=64, target_mask_min=.25,
                PnP_refineLM_max_iterations_per_update=10,
                CONTOUR='visible mesh silhouette -> automatic target mask, normal residual, least_squares max10 per update',
                PL='existing native point/typed normal/outline evidence, max10 internal refinement evaluations per update',
                step_rotation_max_deg=30, step_translation_max_object_diagonal=.5,
                invalid='last valid / initialization fallback with all visited endpoints and failure reason',
                DeepIM_or_SRT3D_reproduction=False),
        F=dict(mandatory=True, primary_scores_used_to_choose=False,
               configuration_selection='availability/contracts first; highest median across3seeds own synthetic VAL joint success; tie lower medianVALloss, then cheaper actualwall; COMMON covariance CP unless CB LOGO mean NLL strictly lower and no heldout coverage loss',
               no_stage_improves_original120='still execute F on original full producer baseline with strong shared COMMON observation rules; no omission',
               final_rows=['ORIGINAL_120','MATCHED_COMMON','MATCHED_BI'],
               intervention='policies1-4; downstream B/I summaries remain identical both arms; actual own observations/banks/3H1',
               exact_reuse='identical produced packets/features/train bindings may share fits; equality proof required'),
        R=dict(metric='unchanged real C2 T<=5cm AND R<=5deg, synthetic original row C1/C2',
               outputs='192/49/143 and separate CAL46; means/sampleSD/median/P90/P95/max/tails/noPose/fallback',
               paired='all3 seeds separately, fixed31/41/38 sets, 6recording clusters,10000paired cluster bootstrap, sessions, leave-one-cluster-out',
               gallery='all recovery/damage IDs plus deterministic normal/largeerror/selectionfail/candidateabsent examples',
               real_reference='prediction/model/config seal before numeric evaluation; already-used development data not new independent test'),
        preservation='original dirty worktree/index/HEAD/default/paper/oldlocks/results untouched; new namespace and run only',
        publication='derived rights-cleared small panels/code/reports only; full private RGB/caches/weights/references excluded; main fast-forward, never force')
    write(Path(run)/'CAMPAIGN_PLAN.json', plan)
    return dict(path=str(Path(run)/'CAMPAIGN_PLAN.json'), sha256=sha(Path(run)/'CAMPAIGN_PLAN.json'))

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--instructions',type=Path,required=True)
    a=p.parse_args();print(json.dumps(freeze(a.run,a.instructions)))
