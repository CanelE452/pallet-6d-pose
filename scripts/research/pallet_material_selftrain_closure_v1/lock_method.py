"""Freeze the existing Plastic main; no Wood results are consulted."""
from pathlib import Path
import subprocess
import torch
from scripts.research.pallet_type_selftrain_v1.recovery_pose_trainer import pose_parameter
from . import common as C

def main():
    if (C.DOC/'METHOD_LOCK.json').exists():
        for b in C.read(C.DOC/'INPUT_BINDINGS.json')['files']:C.verify(b)
        print('METHOD_ALREADY_LOCKED');return
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
    remote=subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip();assert branch=='main' and head==remote
    P=C.P;core=C.read(P.DOC/'CORE_COMPARABILITY_AUDIT.json');protocol=C.read(P.REC/'pose_only/PROTOCOL.json')
    for b in C.read(P.DOC/'INPUT_BINDINGS.json')['files']:C.verify(b)
    teacher=C.checkpoint('TEACHER');assert teacher['sha256']=='fc9b3d7b7f2e7c38b608a12461cf16a58b76669f12ef9be5c8dc35d81104a41d'
    for a in ('R0','SYN_LR5','TEACHER'):C.verify(C.checkpoint(a))
    model=torch.load(C.ROOT/C.checkpoint('R0')['path'],map_location='cpu',weights_only=False)['model'].float()
    trainable=[dict(name=n,shape=list(p.shape),numel=p.numel()) for n,p in model.named_parameters() if pose_parameter(n)]
    source_list=C.ROOT/protocol['datasets']['RAW']['train_list']['path'];source=[Path(x) for x in source_list.read_text().splitlines() if Path(x).name.startswith('syn__')]
    assert len(source)==512 and len(set(source))==512
    source_manifest=[dict(image=C.bind(p),label=C.bind(p.parent.parent/'labels'/f'{p.stem}.txt')) for p in source]
    C.save(C.RAW/'SYNTHETIC_REPLAY512.json',source_manifest,True)
    settings=dict(protocol['args'],lr0=1e-5)
    method=dict(utc=C.now(),plastic_teacher_checkpoint=teacher,teacher_manual_budget=dict(images=9,manual_corners=38,scope='Teacher coordinate adaptation only; existing material labels/metadata and66visible evaluation clicks are separate human effort.'),
        student_initial_checkpoint=C.checkpoint('R0'),trainable_inventory=trainable,protected='All other parameters and every buffer, including pose BN running stats; PoseOnlyTrainer exact preservation.',
        optimizer='AdamW',lr=1e-5,updates=320,epochs=5,batch=16,nbs=16,seed=42,args=settings,
        synthetic_replay=dict(manifest=C.bind(C.RAW/'SYNTHETIC_REPLAY512.json'),unique=512,slots_per_epoch=512,unchanged_from_plastic_main=True),
        pseudo_filter_contract=dict(raw_box_score_min=.85,raw_keypoint_conf_min=.5,min_valid_corners=6,raw_flip_and_median_LOO_max=.05,refined_all8_finite=True,refined_median_LOO_max=.05,refined_flip=False,whole_image_keep_drop=True),
        support_contract='Existing recovery_pose.paired_labels: identical intersected v2 support, v1 true-ignore with center unchanged; boxes same; only supported xy values differ.',
        inference_contract='Native RGB ->100px reflection padding once ->YOLO640 rect=True/conf.001/no augment ->highest-score candidate ->inverse padding; R0 cuDNN TF32True, ReplayFalse, matmulFalse.',
        pose_contract='Same common deployableD9/corner0..7 solver+existing center residual and registered material dimensions/K; no oracle or model-specific selector.',
        plastic_population=dict(candidate=1000,accepted=249,sampled_unique=217,eval=128,main_arms=['R0','RAW_LR5','REF_LR5']),
        primary_metrics=['full-denominator PCK10','common-D9 ADDsym AUC'],secondary='Other2D/pose metrics, severity, recording, PCK5/20/tails; no all-metric dominance requirement.',
        wood_status='PROVENANCE_AUDIT_PENDING',wood_teacher_policy='Exactly the same frozen Replay9/38; poor outcomes never justify replacement.',
        reuse_SYN_LR5=dict(checkpoint=C.checkpoint('SYN_LR5'),reason='Material-independent source-only R0 copy; same replay512 and original320-update pose-only LR5 contract; no new SYN fit.'),
        new_fit_cap=2,conditional_SYN_fit_cap=3,additional_labels=0,
        forbidden_posthoc_changes=['teacher','thresholds','lr/epoch sweep','loss','selector/router','labels','GT coordinates','severity-dependent checkpoint','DOPE','Plastic main retuning','performance-based subset exclusion'],
        material_routing='Material supplied by existing human/registry metadata. Separate students only if fitted. No new automatic classifier or unknown-material policy evaluated.')
    C.save(C.DOC/'METHOD_LOCK.json',method,True)
    C.save(C.DOC/'GOAL_LOCK.md','# Material closure 목표와 종료 조건\n\nPlastic의320-update RAW/REF 비교를 그대로 보존하고, 같은 Replay9장38코너 교사·학생 계약으로 Wood의 corrected-vs-raw 효과를 평가한다. 목적은 Wood 성능을 좋게 만드는 것이 아니라 material 내부 차이를 공정하게 비교하는 것이다.\n\nWood 평가 population은 이미지/촬영 provenance로 scoring 전에 고정한다. 교사 감독에는 Plastic night도 포함되므로 Wood와 이름이 달라도 같은 recording이면 제외한다. 실제 믿을 수 있는 직접 클릭 참조가 없으면 Q1_WOOD는UNRESOLVED로 남기며 legacy/PnP 보완점을 승격하지 않는다.\n\n기본fit2개·각320update, source-onlySYN은 기존 동일checkpoint 재사용한다. 좌표/이미지/teacher 추가 없음. 결과를 보고 threshold·교사·학습량·subset을 바꾸지 않는다. Wood 비교 불가면WOOD_UNRESOLVED로 원고를 닫는다. 표·한계·영문원고·build/tests/audit·main push 후STOP. DOPE는별도요청전까지실행하지않는다.\n',True)
    C.save(C.DOC/'MATERIAL_CLAIM_MATRIX.json',dict(plastic=dict(status='SUPPORTED_WITHIN_REUSED_DEV',main='R0/RAW_LR5/REF_LR5,128 ordinary-plastic images',verified66='Student43/66 tie retained; teacher44->50/66'),
        wood=dict(pseudo_quality='PENDING_TRUSTED_REFERENCE_AUDIT',self_training='PENDING_PROVENANCE_AND_MATCHED_PAIR',sixD='PENDING_MATCHED_PAIR'),
        decision_rules=dict(MATERIAL_GENERAL_SIGNAL='Both materials corrected>raw on locked PCK10/AUC, scoped to these two tested categories.',MATERIAL_DEPENDENT_EFFECT='Plastic positive, Wood absent/negative/materially different tradeoff.',WOOD_UNRESOLVED='No valid fair Wood pair/evaluation/coordinate contract.'),
        no_claims=['all materials','independent test','independent physical6D','automatic material identification','estimator architecture generalization']),True)
    sources=[P.DOC/n for n in ('REPORT_KO.md','CORE_COMPARABILITY_AUDIT.json','CORE_RESULTS.json','PSEUDO_LABEL_QUALITY.json','INPUT_BINDINGS.json','MANUAL_SUPERVISION_BUDGET.json','INDEPENDENT_CONFIRMATION_AUDIT.json')]
    sources += [P.REC/'pose_only/PROTOCOL.json',P.REC/'pose_only/FIT_SYN_LR5.json',C.ROOT/'data/evaluation/pallet_eval_v1/adaptation/PSEUDOLABEL_FILTER_LOCK.json',C.ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json',C.TEACHER/'PROTOCOL.json',C.TEACHER/'INPUT_LOCK.json',C.TEACHER/'FIT.json',P.META]
    sources += [C.ROOT/'scripts/research/pallet_type_selftrain_v1'/n for n in ('pool.py','pseudo.py','recovery_pose.py','recovery_pose_trainer.py','train.py','evaluate.py')]
    sources += [C.ROOT/'scripts/self_training_yolo/v3'/n for n in ('true_ignore_trainer.py','true_ignore_pose_loss.py')]
    sources += [C.ROOT/C.checkpoint(a)['path'] for a in ('R0','SYN_LR5','TEACHER')]
    sources += list(C.PAPER.glob('*.tex'))+[C.PAPER/'manuscript.pdf']
    C.save(C.DOC/'INPUT_BINDINGS.json',dict(files=[C.bind(p) for p in sources if not p.is_relative_to(C.PAPER)],paper_before=[C.bind(p) for p in sources if p.is_relative_to(C.PAPER)],synthetic_manifest=C.bind(C.RAW/'SYNTHETIC_REPLAY512.json')),True)
    C.save(C.DOC/'PREFLIGHT.json',dict(head_start=head,remote_start=remote,branch=branch,status=subprocess.check_output(['git','status','--short','--branch'],text=True),log=subprocess.check_output(['git','log','-8','--oneline'],text=True),untracked_files=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines()),True)
    print('PLASTIC_METHOD_LOCKED_320_LR5_SAME_REPLAY9_38',len(trainable),sum(r['numel'] for r in trainable),flush=True)

if __name__=='__main__':main()
