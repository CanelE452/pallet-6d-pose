from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'pallet_constraint_validity_v42'))
from context42 import *
from experiment42 import load_inputs,labels,probabilities
V42=RUN
KIT=Path(__file__).resolve().parent
RUN=ROOT/'data/pallet/results/pallet_useful_constraint_v43'
NOTE=ROOT/'_docs/notes/point_line_fusion/04_failure_analysis/pallet_useful_constraint_v43.md'
HANDOFF=ROOT/'_docs/handoff/2026-09-11_pallet_useful_constraint_v43.md'
ARMS=('valid_natural_visual','valid_weighted_visual','useful_natural_visual','useful_weighted_visual','useful_weighted_geometry')
ROUTES=('residual','probability')
PRIMARY='useful_weighted_visual_probability'

def register():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'PROTOCOL.json').exists():return
    p=population();part=partition(p)
    protocol=dict(question='Does supervising useful valid constraints and weighting their rare positives retain missed improvements, versus validity alone, and does utility ranking beat frozen residual ranking?',
    scope='Frozen V41 actualphysical3 candidates, V41direct fallback and V42 labels/features. Same2CAD12fit4calibration8reusedcheck; upstream/offsetold16reuse. 600requested436detected,check200requested130detected1040points. No neural updates/new render/newreal/manualGT changes. Mechanistic diagnostic, not independent final architecture evidence.',
    targets='valid=V42constraint label. useful=valid AND Euclidean fallback error - candidate error >1px. Valid observed slots only. GT only supervision/calibration/diagnosis. Weight natural inherited groupframe slot weights; weighted multiply useful positive slots by fixed4 for BOTH valid/useful targets then renormalize. No checkdependent balancing or inverseprevalence.',
    features='ExactV42 visual/geometry plus candidate-minus-fallback normalized displacement2D and its norm. All newmodels have these same3 observable additions; geometry removes V42appearance as before. This compares candidate versus fallback reference explicitly. Additionalfeature effect versus historicalV42 is confounded withnewcalibration; matchednewvalidnatural control is used for target/weight comparisons.',
    models='4 factorial visual arms valid/useful xnatural/weighted plus usefulweightedgeometry=5perseed x3=15HGBClassifier60rounds900rounds. InheritedPARAMS,random_state43000+seed. No newproposal or residual regressor.',
    routes='residual uses same positive V41residual expectedgain ranking relativeanchor; probability uses predicted class probability ranking with no residual-positive prerequisite. Both require valid matchedcandidate andprob>=threshold; elseV41direct fallback. No point-hardness gate. Probability classifier score forweighted fits is not calibrated probability. GTuseful oracle sameoldresidualranking vsGTcandidatebest separated.',
    calibration='For EACH arm/route choose threshold fromfixed .10,.15,...,.95 usingcal4 ONLY. Require selectedconstraintprecision>=.9,>=10actuallychangedpoints across>=2groups, and eachcalcondition mean error<=1.01fallback. Among admissible positivegroupframeweightedmeanpixelgain maximizegain, tieshigherthreshold. Otherwise1.1abstain. Threshold objective is empiricalpointutility not globalvalidslotrecall; no confidence guarantee. Report entirefixedgrid andprobability bins; never choosebestcheckarm.',
    primary=PRIMARY,gates='>=1%additionalallpointgain everyseed vsfallback andpaired5000sourcegroupbootstrapCIupper<0; allpopulatedcheckcondition/stratum<=1.01fallback; >=1%points changed eachseed; selectedconstraintprecision>=.9 eachseed. UsefulGTopportunitycoverage eachseed>=2xV42 whileprimarypointgate required; allrequiredaccepted. Matchedvalidnatural andgeometry deltas separately. Allabstain preservation isnotimprovement.',
    diagnosis='Before andafter fit/cal/check compare useful prevalence, classifier AUROC/AP, useful/harduseful recall and GTbestcoverage amongacceptedcandidatepool, frozenresidual-positive eligibility, actualchosenhelp/harm. Fixedscore bins [0,.1,...,1] and checkgroupbreakdown. No posthoccheckthresholdselection. GTbest candidates arediagnostics only.',hyperparameters=PARAMS,partitions={k:v for k,v in part.items() if isinstance(v,list)})
    write(RUN/'PROTOCOL.json',protocol)
    old=read(V42/'PRIOR_BINDINGS.json')['files']|read(V42/'COMPLETION.json')['files'];old[str((V42/'COMPLETION.json').relative_to(ROOT))]=sha(V42/'COMPLETION.json');write(RUN/'PRIOR_BINDINGS.json',dict(files=old))
    (RUN/'PURPOSE.md').write_text('# V43 useful constraint learning\n\n'+protocol['question']+'\n\n'+protocol['scope']+'\n')
    HANDOFF.write_text('# V43 改善 가능한 유효 제약 학습\n\n사용자 승인. 15HGB900rounds, V41후보/복귀 고정. valid/useful×natural/4배가중 visual + usefulweightedgeometry. cal4고정격자 선택, primary usefulweightedvisual_probability. 같은12fit/4cal/8재사용check,old16reuse. 새로운렌더/실사/GT수정/신경망0. 기존완료결과보존,그림/MD/실제Chrome/완료사실만Discord.\n'.replace('改善 가능한','개선 가능한'))

if __name__=='__main__':register()
