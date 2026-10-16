from pathlib import Path
import sys,os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'pallet_line_utility_v37'))
from context37 import *
from analyze37 import load_seed,features,rowstats
V37=RUN
KIT=Path(__file__).resolve().parent
RUN=ROOT/'data/pallet/results/pallet_line_uncertainty_v38'
NOTE=ROOT/'_docs/notes/point_line_fusion/04_failure_analysis/pallet_line_uncertainty_v38.md'
HANDOFF=ROOT/'_docs/handoff/2026-09-11_pallet_line_uncertainty_v38.md'
ARMS=('grid','both');SCALES=('constant','appearance','geometry')
PARAMS=dict(learning_rate=.08,max_iter=60,max_leaf_nodes=15,min_samples_leaf=64,l2_regularization=1.,max_bins=64,early_stopping=False)

def partition(p):
    groups=sorted(set(p['groups'][p['train']]));cal=[g for i,g in enumerate(groups) if i%4==3];fit=[g for g in groups if g not in cal];folds=[[g for i,g in enumerate(fit) if i%3==k] for k in range(3)]
    return dict(fit_groups=fit,calibration_groups=cal,check_groups=sorted(set(p['groups'][~p['train']])),fold_groups=folds,fit=np.isin(p['groups'],fit),calibration=np.isin(p['groups'],cal),check=~p['train'])

def register():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'PROTOCOL.json').exists():return
    p=population();part=partition(p)
    protocol=dict(question='Does prediction of uncertainty at the target point transfer to unused groups and permit beneficial conservative line corrections?',scope='Frozen V36 grid/both candidates and V34anchor3seeds. Existing600renderobservations/436detected, same2CAD. New mean/scale fit12groups, calibration4groups, reused check8groups. No neural updates/newrealGT/manualedits. Existing upstreamV36 and V35offsets previously saw all16oldtrain groups: calibration is held out of NEW mean/scale fitting only, not independent of upstream. Empirical coverage, not a distribution-free conformal guarantee.',
    partitions={k:v for k,v in part.items() if isinstance(v,list)},models='36HistGradientBoosting fits: each2arms3seeds has3groupfold mean fits+1final mean fit+2OOF residual-scale fits. Mean target signed normal correction/boxdiag, absolute-errorloss. Scale target log(abs out-of-group residual+1e-4), absolute-errorloss; constant=groupweighted90th percentile OOF absolute residual. Same mean predictor across all3uncertainty arms. Hyperparameters fixed beforecheck; CPU one thread.',hyperparameters=PARAMS,
    mean_features='V37 visual_geometry inputs plus observable cached Hough support features: center/spread/mass; point-tangent distance and spread-normalized lever; role candidate normal dispersion and target residual dispersion. No GTsupport/visibility/role matching features. Feature arrays stay associated with frozen diverse8candidate index. Cached support weighted by predictedrole map is a proxy, not exact visible endpoints.',
    scale_features='constant, appearance=V37point/query/score/role+candidatevisual, geometry=sameappearance plus normal/correction+predicted support/lever/disagreement. Uncertainty targets use3fold source-group OOF mean predictions from fit12only, never in-sample residuals. Finalmean fits12, scale fits12OOFlabels; calibration targets only set scalar90%multiplier.',
    calibration='Two prespecified variants: marginal90% weighted candidate residual/scale quantile and simultaneous point90% quantile of max over24candidates. Source groups equally weighted inside calibration. The latter addresses candidate selection multiplicity perpoint, not acrossframes/groups; no formal coverage promise. Calibrationgroups4 only. Scale clipped to fitOOF residual .01/.99 quantiles with floor1e-4 before calibration.',
    action='Keep same V37 raw bounded .15boxdiag candidate corrections a,n. Mean mu predicts desired n.(gt-anchor)/diag. Conservative lower squared-gain=2*a*mu-a^2*norm(n)^2-2*abs(a)*q*scale; choose bestpositive else no-op. mu-only control q=0. Interval truth inclusion implies nonworsening actual squared distance at selected candidate. No testGT affects inference. CheckGT only independent metrics/oracle. ExistingV37pixelgain historicalreference separately.',
    metrics='Check coverage candidate and simultaneouspoint/selectedcandidate, mean90%halfwidthpx, scale Spearman rank vsabsolute residual; calibration-vscheck groupbreakdown; retention/riskcurves thresholds calibratedonly; perpoint Euclideanerror, selectionfraction, harmful>1px, hidden/outside/clean/crops. Bootstrap8sourcegroups5000resamples,3seedsaveragedfirst.',
    gates='Primary both/geometry_point: check simultaneouscoverage>=.85 eachseed, scale Spearman>=.2 eachseed, meanhalfwidth<=.9constant_point eachseed => uncertainty usefulness. Actual transfer >=1%pointimprovement eachseed vsV34 and8groupCIupper<0 on1040allpoints; everypopulatedcheckcondition/stratum perseed<=1.01V34; selectionfraction>=.01eachseed; geometry_point beatsconstant_point pairedCIupper<0. All gates required for screen success. No-op/allabstain cannot count as improvement. Other variants diagnostic, never select bestcheckmodel.',
    decision='Report mean-estimation, uncertainty generalization, conservative selection and actual gain separately. If wide intervals cause abstention, quantify it instead of calling preservation improvement. No long end-to-end architecture training unless usable additional information established; current task is small frozen-architecture uncertainty module experiment.',
    limits='Smallreused2CADcheck,detectedonly;sourcegroupsareappearance groups not novelCAD. Fixedexistingparentproposals. Upstream trainedold16groups prevents strong calibration independence claim. HGB module is a prototype attached to frozen architecture, not an end-to-end optimized replacement. NoPCGrad.')
    write(RUN/'PROTOCOL.json',protocol);old=read(V37/'PRIOR_BINDINGS.json')['files']|read(V37/'COMPLETION.json')['files'];old[str((V37/'COMPLETION.json').relative_to(ROOT))]=sha(V37/'COMPLETION.json');write(RUN/'PRIOR_BINDINGS.json',dict(files=old));(RUN/'PURPOSE.md').write_text('# V38 point-conditioned line uncertainty\n\n'+protocol['question']+'\n\n'+protocol['scope']+'\n');HANDOFF.write_text('# V38 점 위치의 선 보정 불확실성\n\n사용자 승인. V36grid/both고정3seed,V34점기준. 새mean/scale학습12그룹/calibration4/check8,36HGBfits60rounds,신경망0. OOF잔차학습/후보별또는점별max90%경험적범위보정. 평균보정량동일,상수/영상/기하불확실성비교. 기존부모/offset은old16학습이력이있어엄밀conformal보장아님. PROTOCOL사전기준및원결과보존. 실제개선/보정률없으면전부거절도성공아님. 완료후MD/그림/Chrome/승인Discord.\n')

if __name__=='__main__':register()
