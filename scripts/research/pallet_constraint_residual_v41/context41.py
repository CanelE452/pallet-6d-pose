from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'pallet_segment_correspondence_v40'))
from context40 import *
from observations40 import canonical
V40=RUN
KIT=Path(__file__).resolve().parent
RUN=ROOT/'data/pallet/results/pallet_constraint_residual_v41'
NOTE=ROOT/'_docs/notes/point_line_fusion/04_failure_analysis/pallet_constraint_residual_v41.md'
HANDOFF=ROOT/'_docs/handoff/2026-09-11_pallet_constraint_residual_v41.md'
SOURCES=('pred_segment','GT_match_segment','GT_line')
FAMILIES=('legacy24','physical3')
MODES=('direct','analytic','residual')

def register():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'PROTOCOL.json').exists():return
    p=population();part=partition(p)
    protocol=dict(question='Does an analytic constraint skip with learned residual transfer physical line evidence better than direct target regression, and does choosing the actual physical-normal proposal remove the legacy candidate-axis mismatch?',
    scope='V34anchor3seeds,V40actualLSDmatched/GTmatched/idealGTline observations immutable. Same436detected/600requested,fit12/cal4/reusedcheck8groups,same2CAD. No newneuralupdates/render/real/manualGTediting. FrozenV35old16trainmedian offsets reused identically inallarms; calibrationnot upstreamindependent.',
    differences='V30 learned signed moves on earlier guided/independentDHT, V35 tested offsets/projections/GTroleSVD. V41 uses V40sameobservations and matched direct-vs-analytic-plus-residual objective with identicalfeatures/training masks, then changes proposal direction separately. Doesnot claim ordinary projection is new.',
    legacy24='Same24V37boundedcandidates andV40feature arrays. Analytic prior mu=(physicalrho+oldtrainmedian-normalphysical.anchor)*dot(normalphysical,normalcandidate)/diag when role supportpresent; zero prior whenabsent. Direct exactV40cachedmu reused. Analytic-only usesV38standard fallback forabsentroles; residual predicts target-prior onallfitcandidates then addsprior,including absentroles. Samepositive expected squaredgain/noop selector.',
    physical3='Exactlyone observedline per3incidentroles,canonicalny>=0,validonly matched/visible. Proposal anchor+physicalnormal*clip(physicalrho+oldtrainmedian-physicalnormal.anchor,+/-.15diag). Target desiredsignedphysicalnormal correction/diag. Prior unboundedphysicalresidual/diag. Direct HGBtarget versus residual HGB(target-prior)+prior use identicalfeature arrays/weights. Invalidproposals masked/noop,excludeinvalidtargets fromfitting. No tangent displacement within a singlephysicalnormal correction. Prior residualtarget equals physicalnormal.GTcorner-rho-median; independent ofanchor for fixedline.',
    physical3features='Mean andstd ofV40geometryover8legacycandidates perrole, V40source14segmentfeatures fromfirstrole slot with orientation overwritten1whenvalid, normalizedphysicalprior extra scalar. Thus bothdirect/residual haveexplicitpriorfeature andsameinputinformation. It still containslegacyfrozenpoint/linevisualcontext. NoGTpoint/features inactualinference.',
    fits='3sources3seeds: 1legacy24residual +2physical3direct/residual =27newHGBmodels*60rounds=1620boostingrounds. Allfitpoints,sourcegroupbalanced inheritedframeweights,invalidphysical3excludedthenweightsnormalized. Nohard-onlyfit/nocalibrationtuning. Family/method effects are reported within matchedcomparison; nofamilyMAEcomparison acrossdifferentnormaltargetpopulations.',
    routing='For eachfamily/mode: apply bestpositivepredictedgain candidate elsekeep; also samefrozenV39linehardnessgate; GTsafe rejects selectedpoint onlyif actualerrorworsens diagnostic. GTinputs remainprivileged regardlessrouting. GTbestcandidate+noop oracle pernew3pool andlegacy24 separate. Nooutput fromGTgate used inactualpredictions.',
    metrics='All/easy/hard>=10px/outside/hidden/condition Euclideanpoint error, harmful>1px andmovefraction. DesirednormalMAE/sign within validrole targets forlegacy24 orphysical3,hardandall. Paired5000sourcegroupbootstrap keepingseed-specific denominators. Primaryactualphysical3 residual vsdirect hardnormalMAE>=10%eachseedCIupper<0; actualresidual+frozenhardgate >=1%pointgain vsanchor eachseedCIupper<0 andeverypopulatedcondition/stratum <=1.01anchor. Rawall andothervariants diagnostic notbestcheckselection. Rawactualresidual pointvsdirect CI asmechanisticcontrast. GTsource screens separate. Nooracleacceptedassuccess.',
    decision='IfGTphysical3analytic succeeds butactualfails, errors in line validity/correspondence/location remain; residualfittingcanalsodamageanalyticgoodconstraints. Iflegacyresidual improves withoutphysical3thenlearnedtargetbias is involved; ifphysical3 improves onlywithinGT thenaxisalignment/proposalcapacity benefits notdeployable. Correlatedsmallreusedcheck cannot establish universalimpossibility or real improvement. No new longendtoend/gate/PCGradtraining onfailedscreen.',hyperparameters=PARAMS,partitions={k:v for k,v in part.items() if isinstance(v,list)})
    write(RUN/'PROTOCOL.json',protocol);old=read(V40/'PRIOR_BINDINGS.json')['files']|read(V40/'COMPLETION.json')['files'];old[str((V40/'COMPLETION.json').relative_to(ROOT))]=sha(V40/'COMPLETION.json');write(RUN/'PRIOR_BINDINGS.json',dict(files=old));(RUN/'PURPOSE.md').write_text('# V41 analytic constraint + residual\n\n'+protocol['question']+'\n\n'+protocol['scope']+'\n');HANDOFF.write_text('# V41 물리 제약 계산·잔차 학습 검증\n\n사용자 계속 승인. V40관측/GT대조불변,V34고정3시드. 기존24후보target대조에analytic+residual추가,같은물리normal3후보direct/analytic/residual별도. 3source×3seed×3newfits=27HGB60rounds,신경망0. 같은12fit/4cal/8reusedcheck,same2CAD/old16offsetreuse. GT입력/GTsafe/oracle는진단. PROTOCOL사전등록. 기존결과불변,완료후MD/그림/실제Chrome/완료사실만Discord.\n')
if __name__=='__main__':register()
