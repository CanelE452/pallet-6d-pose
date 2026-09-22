"""One authorized FULL150 fit; historical results remain read-only."""
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from scripts.research.pallet_posefix_crop_support_v1 import common as S
B=S.B; H=S.H; F=S.F; ROOT=S.ROOT; HERE=Path(__file__).resolve().parent
NAME='pallet_posefix_crop_completion_v2'
DOC=ROOT/'_docs/experiments'/NAME;RAW=ROOT/'data/pallet/results'/NAME;OUT=ROOT/'outputs'/NAME
read=S.read;bind=S.bind;verify=S.verify;state_hash=S.state_hash
EXPANSION=1.50;BASE_EXPANSION=1.25
ARMS={'A':('FULL',BASE_EXPANSION),'B':('FULL',EXPANSION),'C':('C',EXPANSION),'D':('C',BASE_EXPANSION)}

def save(path,obj):
    path=Path(path).resolve();assert any(path.is_relative_to(p) for p in (DOC,RAW,OUT));path.parent.mkdir(parents=True,exist_ok=True)
    def scalar(value):
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    text=obj if isinstance(obj,str) else json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=scalar)+'\n'
    with path.open('x') as f:f.write(text)

def tensor_save(path,obj):
    assert path.resolve().is_relative_to(RAW);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:torch.save(obj,f)

def protocol():
    verify(read(DOC/'PROTOCOL_LOCK.json'));return read(DOC/'PROTOCOL.json')

def init():
    for root in (DOC,RAW,OUT):assert not root.exists(),('Do not overwrite',root)
    p=B.protocol();assert read(S.DOC/'AUDIT.json')['PASS']
    protected={x['path']:x for x in S.lock()['protected']}
    for root in (S.DOC,S.HERE,S.RAW):
        for f in root.rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts and '.pytest_cache' not in f.parts:protected[str(f.relative_to(ROOT))]=bind(f)
    for f in (B.E.RAW/'PSEUDOLABEL_MANIFEST.json',B.E.DOC/'E2_INPUT_LOCK.json'):
        protected[str(f.relative_to(ROOT))]=bind(f)
    for b in protected.values():verify(b)
    for root in (DOC,RAW,OUT):root.mkdir(parents=True)
    old=S.lock();rows=read(S.RAW/'E0_CORNER_ROWS.json');hard=[r for r in rows if r['fixed_hard']]
    fixed={(r['id'],r['canonical']):r for r in old['fixed_hard']}
    for r in hard:r['fixed_channel']=fixed[(r['id'],r['canonical'])]['channel']
    subsets=dict(H163=hard,U29=[r for r in hard if r['old_unreachable']],
        N14=[r for r in hard if r['old_unreachable'] and r['C1']['reachable10']],
        U15=[r for r in hard if r['old_unreachable'] and not r['C1']['reachable10']],
        R134=[r for r in hard if not r['old_unreachable']])
    subsets['I11']=[r for r in subsets['N14'] if r['C1']['minimum_distance_px']<=1e-9]
    subsets['E3']=[r for r in subsets['N14'] if r['C1']['minimum_distance_px']>1e-9]
    assert len(subsets['N14'])==read(S.DOC/'E0_SUPPORT_AUDIT.json')['newly_reachable10']
    save(DOC/'SUBSET_LOCK.json',dict(groups=subsets,counts={k:len(v) for k,v in subsets.items()},source=bind(S.RAW/'E0_CORNER_ROWS.json')))
    inputlock=dict(HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],text=True),protected=list(protected.values()),
        previous_inputs=read(B.DOC/'INPUT_LOCK.json'),eval_records=old['eval_records'],populations=old['populations'],symmetry=old['symmetry'],
        subset=bind(DOC/'SUBSET_LOCK.json'),FULL_fit=read(B.DOC/'FIT_FULL.json'),PRIOR1=p['base'],
        instructions=[bind(Path('/home/minjae/Downloads')/x) for x in ('pallet_crop_completion_cli_v2.txt','pallet_crop_completion_experiment_plan_v2.txt')])
    save(DOC/'INPUT_LOCK.json',inputlock)
    spec=dict(name=NAME,arms=ARMS,new_train_runs=1,primary='C vs A',
        authorization='User: read latest two Downloads TXT and execute; later experiment_plan_v2 clarifies primary interpretation',
        historical_E0='Preserved gate failure; separately authorized post-E0 exploratory study, not retrospective gate pass',
        same_as_FULL={k:p[k] for k in ('base','seed','optimizer','lr','updates','real_batch','source_batch','microbatch','BN','regularization','source_corruption_rng','source_probe_rng')},
        expansion=EXPANSION,control_expansion=BASE_EXPANSION,relative_object_scale=BASE_EXPANSION/EXPANSION,
        mask='recover original semantic trust exactly from frozen pseudo manifest and original preparation code; preserve clean/OCC intersection; no eval dependency',
        source_corruption='regenerate baseline1.25 trace exactly; inverse-transform final perturbed float32 inputs and freeze original-coordinate points; remap same points to1.50; newly supported use original R0',
        coordinate_tolerance_px=1e-4,preserve=False,decoder='expectation unchanged',training_gate='integrity only, no E0/performance gate',
        no_rescue=True,student_selftraining=False,model_promotion=False,scoring='A/B/C/D frozen before evaluation GT',
        train_probe='path/index fixed16 real plus up to16 newly-supported; fixed16 sorted sourceTRAIN plus up to16 newly-supported; chosen before inference',
        primary_mechanism=['RECOVERY_OBSERVED','SUPPORT_ONLY','CANDIDATE_BUT_DECODE_FAIL','ADAPTATION_UNRESOLVED','REFERENCE_OR_CONTRACT_UNRESOLVED'],
        primary_system='NET_GAIN_WITH_PRESERVATION only if N14 gain>=1, primary C>=A, BASE/N2 loss no worse, GREEN/clean/source fixed counts no worse, primaryP90<=A; otherwise observed gains with losses=TRADE_OFF, else NO_OBSERVED_GAIN',
        decoder_route_fraction=.25,legacy_cli_3corner_rule='secondary descriptive only; newer plan two-axis definition primary',
        publication='standing user override: MD+images push; code publication separate from private data/checkpoints',inputs=bind(DOC/'INPUT_LOCK.json'))
    save(DOC/'PROTOCOL.json',spec);save(DOC/'PROTOCOL_LOCK.json',bind(DOC/'PROTOCOL.json'))
    save(DOC/'PURPOSE_AND_PLAN.md','# Crop completion v2\n\n과거 E0 STOP 보존. 최신 두 TXT 승인에 따라 신규 C=FULL150만 PRIOR1부터300update. A=기존FULL125, B=같은weight/infer150, D=Cweight/infer125. 기하/성능 gate 없음, 무결성 오류만 중단. 순서는 입력/원좌표/교란 parity → B freeze → C300 → C/D freeze → 평가·TRAIN fit·후보진단 → 보고.\n\n더 최근 experiment_plan_v2의 기전/시스템 두 축 판정이 primary, CLI의3corner rule은 보조 기술로만 보고. 새 코너/영상 선택·loss·decoder·배율탐색·학생학습·최종모델 변경 없음. 상시 사용자 공개 지시에 따라 MD+실제그림 push, private 매핑/응답/checkpoint 제외.\n')
    save(DOC/'CROP_CONTRACT.md','# 고정 crop 계약\n\nR0 predicted bbox, 중심/종횡비정규화/보간LINEAR/zero padding/384×288 유지. 공통 explicit expansion:1.25 또는1.50만. output96×72 expectation×4의 연속지지영역 [0,284]×[0,380], 학습 mask[0,288)×[0,384)와 다름. inverse affine로 원영상 복원, invalid점·center8·bbox·score·selected candidate pass-through.\n\n확대 시 물체 입력 scale=5/6, 원영상 grid간격=1.2배. crop좌표 loss의 원영상당 크기와 지원감독량도 바뀌므로 C−A는 crop pipeline 총효과이며 순수 support만의 인과효과가 아님. 원영상 RGB에서 다시crop, 기존crop resize 금지.\n')
    print('INIT',len(protected),{k:len(v) for k,v in subsets.items()},flush=True)

if __name__=='__main__':init()
