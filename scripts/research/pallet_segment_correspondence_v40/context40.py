from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'pallet_hard_direction_v39'))
from context39 import *
from experiment39 import fit_reg
from evaluate39 import metric, masks
V39=RUN
KIT=Path(__file__).resolve().parent
RUN=ROOT/'data/pallet/results/pallet_segment_correspondence_v40'
NOTE=ROOT/'_docs/notes/point_line_fusion/04_failure_analysis/pallet_segment_correspondence_v40.md'
HANDOFF=ROOT/'_docs/handoff/2026-09-11_pallet_segment_correspondence_v40.md'
SOURCES=('pred_segment','GT_match_segment','GT_line','GT_segment','GT_shuffled_segment')
FAMILIES=('all','hard')

def register():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'PROTOCOL.json').exists():return
    p=population();part=partition(p)
    protocol=dict(question='Do physical segment/endpoint correspondence observations identify hard-point normal correction beyond V39 frozen features? Separate actual image segment matching, GT correspondence to the same extracted segments, ideal GT physical line, ideal GT support envelope, and wrong-role support control.',
    scope='Same V34 anchor/V36 both candidates only,3seeds. Reused synthetic436detected of600requested, same2CAD, newfit12/calibration4/check8 exactlyV38/V39. No new real/manualGT edit/render/neuraltraining. This is a diagnostic intervention, not independent final-generalization evidence. Upstream previously trainedold16groups.',
    observation='OpenCV LSD_REFINE_STD on actual transformed grayscale. Keep length>=8px and segment midpoint within detected bbox expanded by20%boxdiag; longest128 stableordered. No manual GT input for image extraction. Physical support from immutableV35 cache is first/last visible samples of longest role segment; may span disjoint visibility and is not a certified contiguous edge or virtual corner.',
    matching='Predicted: top-score V36both candidate persemantic role; use its normal/rho and frozen support-center proxy. Segment mean absolute endpoint line distance<=max(3px,.02diag), acuteangle<=15deg, tangent center distance<=max(12px,.25diag). Min normalizeddistance+.05sinangle+.1center_distance/diag, tie longer extractedsegment order. GTmatch: same extracted pool, GTphysical role normal/rho and visible-envelope center; distance<=3px,angle<=10deg, positiveintervaloverlap. Min samecost. Invisible roles absent. Both matching algorithms are fixed before metrics; no newly learned matcher. GTmatch uses oracle role/visibility/geometry and is never deployable.',
    features='Append14candidate-relative segment features to identicalV38geometry. Availability, normal2D, signedphysical-line residual, orientation dot, 2 endpoint relativecoords4, length, tangentcenter, abs tangentcenter, normalcenter, endpoint-to-anchor minimumdistance. Normalize coordinates/length byboxdiag. Canonicalsegmentnormal ny>=0. No GTvirtual points or their residuals in observation construction. GT_line sets endpoint/tangent/length/distance fields zero but keeps availability,physicalnormal,residual,orientation. GT_segment uses exactvisible physical envelope. GT_shuffled_segment rolls role slots by4, preserving raw envelope distribution/availability; imperfect semantic null, not random independence proof.',
    models='5source variants x2fitpopulations x3seeds=30newHGB absoluteerror regressions,60rounds each=1800rounds. Allfitpoints versus only V34error>=10px, sourcegroupbalanced weights. ReuseV38standard allcenter andV39hardgeometry centers as controls, no retuning. Same predictednormal target, same bounded24lineproposals, no new correction candidates. HGB fixedV38params. GTidealobservations are privileged information diagnostic, not a deployable architecture.',
    action='all regressor uses originalpositive expected squaredgain select/noop. hard regressor routes with frozenV39both linehardness predictions/thresholds (cal4), unchanged. Also GTtruehardness route diagnostic. No new gating fit/tuning; hardness GT only fitting/strata/oracles. Candidate oracle fromV39 identical. GTinput variants remain oracle even if learned model appliedwithoutGTgate.',
    metrics='NormalMAE/sign (abs truetarget>=1px) onall/hard and selectedpoint error all/easy/hard/outside/hidden percondition,3seeds. Paired bootstrap5000sourcegroup draws withseed-specific hard denominators. Matching coverage and extracted-to-physical quality. Primarypred_segment/all comparedstandard: normalhardMAE>=10%improvement eachseed+pairedCIupper<0 and actualpoint>=1%improvement eachseed vsanchor+CIupper<0, everycondition/stratum<=1.01anchor. GTsegment vsGTline normalhardMAE>=10%eachseed+CIupper<0 identifies additional envelope/extent information under currentpredictor; GTsegment vsGTshuffled analogous. GTmatching gap comparedpredsegment separately. No bestcheckarm selection or oracle architecture success.',
    decision='Predsegment success only provisional actual observation screen; new independentgroups and endtoend training still required. GTmatchbetter indicates matching/visibility pathway loss; GTsegmentbetterthanGTmatch indicates extraction/precision/support content combined, not pureprecision causalproof. GTsegment vsGTline isolates extraextentfeatures in thisfrozen learner. EvenGTsupportfailure doesnot prove universalimpossibility: frozenfeatures, learner, limitedgroups andsameproposal family remain constraints. Do not repeat longgate/PCGradtraining fromthisdiagnostic.',hyperparameters=PARAMS,partitions={k:v for k,v in part.items() if isinstance(v,list)})
    write(RUN/'PROTOCOL.json',protocol)
    old=read(V39/'PRIOR_BINDINGS.json')['files']|read(V39/'COMPLETION.json')['files'];old[str((V39/'COMPLETION.json').relative_to(ROOT))]=sha(V39/'COMPLETION.json');write(RUN/'PRIOR_BINDINGS.json',dict(files=old))
    (RUN/'PURPOSE.md').write_text('# V40 physical segment correspondence\n\n'+protocol['question']+'\n\n'+protocol['scope']+'\n')
    HANDOFF.write_text('# V40 실제 선분·끝점 대응 정보 검증\n\n사용자 승인. 기존 V39 이후 큰 오류 방향 식별 원인 개입. V36both/V34고정3시드, 같은12fit/4cal/8reusedcheck. 실제 LSD선분 예측대응 / GT물리대응 / GT정확선 / GT지원끝점 / 역할교란끝점 ×전체/큰오류학습=30HGB×60라운드. 실제/GT입력과GT선택 별도. 새실사/렌더/수동GT수정/신경망0. 사전PROTOCOL, 이전완료결과불변. 완료후보고서/이미지/실제Chrome/완료사실만Discord.\n')
if __name__=='__main__':register()
