from pathlib import Path
import os,sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[3];KIT=Path(__file__).resolve().parent;RUN=ROOT/'data/pallet/results/pallet_fresh_decision_v44'
NOTE=ROOT/'_docs/notes/point_line_fusion/04_failure_analysis/pallet_fresh_decision_v44.md';HANDOFF=ROOT/'_docs/handoff/2026-09-11_pallet_fresh_decision_v44.md'
PY=Path('/home/minjae/anaconda3/envs/pallet-yolo26/bin/python');RENDER_PY=Path('/home/minjae/anaconda3/envs/pallet-pose/bin/python');SEEDS=(1,2,3)
PARAMS=dict(learning_rate=.08,max_iter=60,max_leaf_nodes=15,min_samples_leaf=64,l2_regularization=1.,max_bins=64,early_stopping=False)
PRIMARY='fresh_line';ARMS=('reused_point','reused_line','fresh_point','fresh_line')
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def legacy():
 sys.path.insert(0,str(KIT.parent/'pallet_useful_constraint_v43'));import context43
 return context43

def register():
 RUN.mkdir(parents=True,exist_ok=True)
 if (RUN/'PROTOCOL.json').exists():return
 protocol=dict(question='Two final tests: frozen V41direct small gain on genuinely new scenes; parent-unseen supervision and explicit line information versus matched point-feature correction. Stop this frozen-feature correction research family if stable incremental line improvement fails.',
 scope='New independent random-render scenes, same2CAD and renderer distribution, not novelCAD/real generalization. Sourcegroup is appearance/lighting+paired5views andallstressvariants. No sealedreal/GTedit/neuraltraining. Frozen V12/V25/V34/V36/V41 weights never see newfit/cal/test before this protocol.',
 data='seed440000+100asset+group; 2CADx12newgroupsx5viewsx2states=240RGB600observations. Perasset g00..05 newfit (12groups),g06..07 newcal (4groups),g08..11 finalcheck (8groups). Unique v44_ prefix; oldrender32 uses32000seed. Newfit is unseen by EVERY frozen parent, eliminating parent in-sample supervision without claiming same-data OOF. Oldfit12 is previous partition. Differentnewscenes andparent-exposure arejoint intervention, notpurecausalOOF.',
 test1='Immutable V41 actualphysical3direct prediction vs immutableV34anchor onnewfinalcheck only. Reconstruct exact features on oldcache and verify frozen output parity before new scoring. Also report base/strong forcontext. No refit or threshold tuning.',
 test2='2trainingorigins reusedoldfit12 vsnewparentunseenfit12 x2inputs point/line x3seeds. Two HGB absolute-error scalar regressions predict x/y normalized residual fromV34anchor; fixed .15diag vectornorm bound. 24fits1440rounds. No gainclassifier, gate or threshold sweep. Pointfeature: normalizedbase,strong,v34 coords6+query32+aspect1+pointonehot8=47. Linefeature adds mean/std ofV41physical3 observablefeatures over3roles andcandidateavailability3; anchor andpointfeatures identical. Pointcontrol has no additional line tokens but frozenV34anchor/query already come fromline-trained upstream; notpureYOLOarchitecture.',
 fits='HGB fixed inherited params, random_state44000+seed, groupframe weighting thenpoint replication. Newfit excludesnewcal/finalcheck; reused excludesoldcal/check. No newcal tuning; kept separate descriptive holdout. Learn actualparent errors, noGTinfeatures. All4arms matchedsteps/algorithm/outputbound, linehasmoreinputfeatures so samebudget notsameparametercount.',
 gates='Test1: >=1%gain everyseed vsV34 plus8sourcegroup5000pairedbootstrapCIupper<0 andeverypopulatedcondition/stratumwithin1%anchor. Report smaller consistentgains withoutacceptance. Test2primaryfreshline must >=1%allpointgain everyseed vsV34, CIupper<0; everycondition/stratum<=1.01V34; >=1%incrementalgain eachseed vsfreshpoint withCIupper<0. Helpful/harm>1px, hard>=10px, hidden/outside separately. Detectionfailuresreportednotrecovered. Comparefreshvsreused factorial separately; no bestcheckselection. If test2fails endthisfrozenfeature/linecandidate correctionfamily, no V45threshold/weight tweak. Passingonly warrants nextindependentrealprotocol, notdeployment.',
 execution='GPU kernel580.173.02 versusinstalled580.178.04 mismatch. OfficialNVIDIA580.173.02 archive extractedunder/tmp; processlocalLD_LIBRARY_PATH only. CUDA tensor andNVIDIAEGL verified. No systemdriverinstall/reboot. Originalrenderer geometry/convention/depth rules unchanged. Protocol before rendering/training/scoring.',hyperparameters=PARAMS)
 write(RUN/'PROTOCOL.json',protocol);(RUN/'PURPOSE.md').write_text('# V44 final fresh-scene decision\n\n'+protocol['question']+'\n\n'+protocol['data']+'\n');write(RUN/'PRIOR_BINDINGS.json',dict(files=read(ROOT/'data/pallet/results/pallet_useful_constraint_v43/PRIOR_BINDINGS.json')['files']|read(ROOT/'data/pallet/results/pallet_useful_constraint_v43/COMPLETION.json')['files']))
 HANDOFF.write_text('# V44 새 장면 두 검증·종료 판정\n\n사용자 승인. 새24그룹600관측(12fit/4cal/8check),parent전부미관측. 보존V41고정재현+old/new감독×point/line24HGB1440rounds. 신경망0. OOF가아닌새원본parent-unseen감독,원본분포효과혼재명시. 새최종check로튜닝하지않음. test2실패시현재고정특징보정계열종료,문턱/가중반복없음. GPU버전불일치실험별라이브러리로우회,시스템변경없음.\n')
if __name__=='__main__':register()
