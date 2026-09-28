"""Read-only evidence collection and aggregate-only V2 report generation.

No fit, inference, GPU, Git, old-file write, or resource/state mutation occurs.
FINAL_SELECTION is immutable input; FINAL_DECISION is a descriptive closure,
not deployment approval. Missing declared results remain explicitly pending.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

from . import common as C
from . import metric_baseline as M

PYTHON='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python'
MAIN=('A_INPUT_OCCLUSION','B_COORDINATE_SUPPLEMENT','C_EXPOSURE')
KEYS=('translation_cm','rotation_deg','yaw_deg')


def pointer(*parts):
    return ''.join('/'+str(p).replace('~','~0').replace('/','~1') for p in parts)


def lookup(value,ptr):
    if not ptr:return value
    if not ptr.startswith('/'):raise ValueError(ptr)
    for part in ptr[1:].split('/'):
        part=part.replace('~1','/').replace('~0','~')
        value=value[int(part)] if isinstance(value,list) else value[part]
    return value


@dataclass
class Source:
    path: Path
    value: object
    binding: dict

    @classmethod
    def read(cls,path):
        path=Path(path);before=C.bind(path);value=C.read(path)
        assert C.bind(path)==before,'Input changed during report read: '+str(path)
        return cls(path,value,before)


@dataclass
class Cell:
    text: str
    refs: list


def number(source,ptr,fmt='.6f'):
    value=lookup(source.value,ptr)
    if value is None:
        parent,key=ptr.rsplit('/',1);status=lookup(source.value,parent).get(key+'_status','NA')
        return Cell('+∞' if status=='POSITIVE_INFINITY' else 'NA',[])
    rendered=format(value,fmt)
    return Cell(rendered,[dict(source=source.binding,json_pointer=ptr,format=fmt,rendered=rendered)])


def combine(*parts,sep=''):
    return Cell(sep.join(p.text if isinstance(p,Cell) else str(p) for p in parts),
                [r for p in parts if isinstance(p,Cell) for r in p.refs])


class Document:
    def __init__(self,name,trace):self.name=name;self.lines=[];self.trace=trace
    def line(self,*parts):
        cell=combine(*parts);self.lines.append(cell.text)
        for ref in cell.refs:
            self.trace.append(dict(document=self.name,line_number=len(self.lines),line_contains=cell.text,**ref))
    def para(self,*parts):self.line(*parts);self.line('')
    def title(self,text,level=2):self.para('#'*level+' '+text)
    def table(self,headers):self.line('| '+' | '.join(headers)+' |');self.line('| '+' | '.join(['---']*len(headers))+' |')
    def row(self,*cells):self.line(combine('| ',combine(*cells,sep=' | '),' |'))
    def content(self):return '\n'.join(self.lines).rstrip('\n')+'\n'


def result_key(source):
    r=source.value;return (r['cycle'],r['material'],r['seed'])


def label(source,arm):
    r=source.value;return f"{r['cycle']}/{r['material']}/S{r['seed']}/{arm}"


def pending_results(protocols,results,selection):
    available={result_key(r) for r in results};missing=[]
    for source in protocols:
        p=source.value
        for material in p['materials']:
            for seed in p.get('seeds',[p.get('seed',42)]):
                if (p['cycle'],material,seed) not in available:missing.append(f"{p['cycle']}/{material}/S{seed}")
    if selection is None:missing.append('FINAL_SELECTION.json')
    else:
        for field in ('selected_cycle','repetition_cycle','control_repeat_cycle','Wood_cycle'):
            cycle=selection.value.get(field)
            if cycle and cycle not in {r.value['cycle'] for r in results}:missing.append(field+':'+cycle)
    return sorted(set(missing))


def paired_from_private(before,after,arm,groups,sources):
    private=[]
    for source in (before,after):
        b=next(b for b in source.value['private_artifacts'] if Path(b['path']).name.startswith('POSE_METRICS_'))
        C.verify(b);p=Source.read(C.ROOT/b['path']);assert p.binding==b;sources.append(p);private.append(p.value)
    assert set(private[0][arm])==set(private[1][arm]),'Incremental comparison population differs'
    return {g:M.paired(private[0][arm],private[1][arm],ids) for g,ids in groups.items()}


def derived_results(results,selection,ledger,sources):
    members=Source.read(M.RAW/'POPULATION_LOCK_PRIVATE.json');sources.append(members)
    indexed={result_key(r):r for r in results};increments={}
    a=indexed.get((MAIN[0],'PLASTIC',42));c=indexed.get((MAIN[2],'PLASTIC',42))
    if a and c:
        increments['C-minus-A']={arm:paired_from_private(a,c,arm,members.value['PLASTIC']['groups'],sources)
                                  for arm in ('NEW_RAW','NEW_REF')}
    repeated={};repeat_status='NOT_RUN';repeat_outcome='UNRESOLVED'
    if selection:
        s=selection.value;seed=s.get('repeat_seed',43);material=s.get('selected_material',s.get('material','PLASTIC'))
        before=indexed.get((s.get('control_repeat_cycle'),material,seed));after=indexed.get((s.get('repetition_cycle'),material,seed))
        if before and after:
            for arm in ('NEW_RAW','NEW_REF'):
                repeated[arm]=dict(groups=paired_from_private(before,after,arm,members.value[material]['groups'],sources),
                    classification=M.classify_candidate(after.value['groups'][M.PRIMARY][arm],before.value['groups'][M.PRIMARY][arm]))
            repeat_outcome=repeated['NEW_REF']['classification']['label']
            repeat_status='DIRECTION_REPEATED' if repeated['NEW_REF']['classification']['eligible_joint'] else 'MIXED'
    return dict(kind='AGGREGATE_ONLY_REPORT_DERIVATIONS',created_at=C.now(),increments=increments,
        same_seed_recipe_minus_original_baseline=repeated,repeat_status=repeat_status,repeat_outcome=repeat_outcome,
        scope='Same original R0 initialization; nominal second trainer seed only. Effective worker streams were identical; deterministic reexecution, not stochastic replication or independent reference.',
        resources=dict(GPU_hours=ledger.value['totals']['gpu_seconds']/3600,
                       recorded_active_wall_hours=ledger.value['totals']['elapsed_wall_seconds']/3600),
        sources=[s.binding for s in sources])


def decision(results,selection,derived,pending,final,replication=None):
    selected=selection.value if selection else {};chosen=next((r for r in results if result_key(r)==(
        selected.get('selected_cycle'),selected.get('selected_material',selected.get('material','PLASTIC')),
        selected.get('selected_seed',selected.get('seed',42)))),None)
    numeric_repeat=derived.value['repeat_status'];seed42=chosen.value['classification']['versus'] if chosen else {}
    # Independent review established that the nominal new seed did not vary the
    # loader streams. Do not infer stochastic replication from identical signs.
    repeat='NOT_RUN'
    # A sign-only discovery is never automatically promoted to meaningful gain.
    outcome='UNRESOLVED' if pending else 'MIXED'
    if chosen and not pending and seed42['OLD_REF']['eligible_joint'] and repeat=='DIRECTION_REPEATED':
        outcome='JOINT_T_R_GAIN'
    return dict(created_at=C.now(),status='FINAL' if final else 'DRAFT',
        EXECUTION='PARTIAL_BUDGET' if final and not pending else 'PARTIAL_INFORMATION',POSE_OUTCOME=outcome,
        REPEAT=repeat,EVIDENCE='REUSED_DEV_ONLY',CAUSE='COMPETING_EXPLANATIONS',
        BEST_SUPPORTED_RECIPE='NO_NEW_RECIPE_PROMOTED',selected_for_repeat=selected.get('selected_cycle'),
        seed42_comparisons=seed42,repeat_classification=derived.value['repeat_outcome'],pending=pending,
        nominal_seed_reexecution_direction=numeric_repeat,
        stochastic_training_variation_tested=False,
        replication_limitation='Nominal different trainer seed did not vary loader RNG: identical trace hashes and all model state tensors. Four actual fits remain counted; no stochastic replication claim.',
        practical_significance_established=False,deployment_approval=False,original_paper_edited=False,
        decision_rule='Both median signs and coverage reported; repeated signs are descriptive, not a practical-effect or independent-validation threshold.',
        NEXT_ONE_DECISION='새 방법을 더 탐색하지 않고, 대표적인 큰 pose 실패가 좌표 오차인지 reference/물리 축 오차인지 구분할 독립 확인 근거를 확보할지 결정한다.',
        sources=[s.binding for s in (selection,derived,chosen,replication) if s])


def summary_cells(source,base):
    n=lambda *p,fmt='.6f':number(source,base+pointer(*p),fmt)
    return [combine(n('valid_pose',fmt='d'),'/',n('frames',fmt='d')),
            *[combine(n('conditional',k,'median'),' / ',n('conditional',k,'P90')) for k in KEYS],
            n('axis_mismatch_count',fmt='d'),n('failed_pose',fmt='d')]


def pose_table(doc,rows):
    doc.table(['조건 / 모델','valid / N','T cm median / P90','R deg median / P90','yaw deg median / P90','axis 오류','pose 실패'])
    for name,source,base in rows:doc.row(name,*summary_cells(source,base))
    doc.line('')


def contrast_table(doc,rows):
    doc.table(['조건 / after − before','common / N','Δ T 중앙값 cm','Δ R 중앙값 deg','median frame ΔT cm','median frame ΔR deg','valid→fail','fail→valid'])
    for name,source,base in rows:
        n=lambda *p,fmt='.6f':number(source,base+pointer(*p),fmt)
        doc.row(name,combine(n('common_valid_frames',fmt='d'),'/',n('frames',fmt='d')),
                *[n('difference_of_conditional_medians',k,fmt='+.6f') for k in KEYS[:2]],
                *[n('median_of_common_frame_differences',k,fmt='+.6f') for k in KEYS[:2]],
                n('available_to_failed',fmt='d'),n('failed_to_available',fmt='d'))
    doc.line('')


def direction_table(doc,rows):
    keys=[f'T_{a}__R_{b}' for a in ('IMPROVE','TIE','WORSEN') for b in ('IMPROVE','TIE','WORSEN')]
    doc.table(['조건 / 대비']+[k.replace('IMPROVE','↓').replace('WORSEN','↑').replace('TIE','=') for k in keys])
    for name,source,base in rows:
        doc.row(name,*[number(source,base+pointer('paired_direction_counts',key),'d') for key in keys])
    doc.line('')


def selected_rows(baseline,results,group=M.PRIMARY,material='PLASTIC'):
    rows=[(f'BASE/{material}/{arm}',baseline,pointer('materials',material,'groups',group,arm)) for arm in ('R0','OLD_RAW','OLD_REF','SYN')]
    rows += [(label(r,arm),r,pointer('groups',group,arm)) for r in results if r.value['material']==material for arm in ('NEW_RAW','NEW_REF')]
    return rows


def result_contrasts(results,groups=None):
    rows=[]
    for source in results:
        r=source.value
        chosen=groups if groups else r['contrasts']
        for group in chosen:
            if group not in r['contrasts']:continue
            for contrast in ('NEW_REF-minus-OLD_REF','NEW_REF-minus-R0','NEW_REF-minus-NEW_RAW'):
                rows.append((label(source,group+'/'+contrast),source,pointer('contrasts',group,contrast)))
    return rows


def derived_contrasts(derived,primary_only=False):
    rows=[]
    for contrast,arms in derived.value['increments'].items():
        for arm,groups in arms.items():
            for group in groups:
                if primary_only and group!=M.PRIMARY:continue
                rows.append((contrast+'/'+arm+'/'+group,derived,pointer('increments',contrast,arm,group)))
    for arm,value in derived.value['same_seed_recipe_minus_original_baseline'].items():
        for group in value['groups']:
            if primary_only and group!=M.PRIMARY:continue
            rows.append(('명목 추가 seed 재실행/RECIPE−BASELINE/'+arm+'/'+group,derived,
                         pointer('same_seed_recipe_minus_original_baseline',arm,'groups',group)))
    return rows


def figure_lines(doc,figures,selected=None):
    if figures is None:doc.para('그림: PENDING — FIGURE_MANIFEST가 아직 없다.');return
    relative=lambda p:Path(__import__('os').path.relpath(C.ROOT/p,(C.DOC/doc.name).parent)).as_posix()
    for figure in figures.value['figures']:
        if figure['name'] not in ('objective99_tr_scatter','plastic_severity_tr','wood_severity_tr'):continue
        C.verify(figure['file']);doc.para('![',figure['name'],'](',relative(figure['file']['path']),')')
        doc.para(figure['caption'])
    examples=[e for e in figures.value.get('examples',[]) if e.get('status')=='AVAILABLE']
    chosen=[e for e in examples if e.get('cycle')==selected] if selected else []
    if not chosen:
        chosen=examples[-2:]
        doc.para('선택 recipe의 예시가 아직 없으면 아래 예시는 명시된 완료 cycle의 기술 예시이며 선택 recipe의 그림으로 대체 해석하지 않는다.')
    for ex in chosen:
        C.verify(ex['figure']);i=figures.value['examples'].index(ex)
        doc.para('![',ex['cycle'],' ',ex['category'],'](',relative(ex['figure']['path']),')')
        doc.para(ex['cycle'],' / ',ex['category'],' / 기존 공개 승인 ID `',ex['id'],'`: T 변화 ',
                 number(figures,pointer('examples',i,'delta_translation_cm'),'+.6f'),' cm, R 변화 ',
                 number(figures,pointer('examples',i,'delta_rotation_deg'),'+.6f'),' deg. 모델 출력은 원래 검출 keypoint이며, 초록색 reference는 독립적인 물리 pose 정답 검증을 의미하지 않는다.')
        doc.table(['해당 예시 모델','T cm','R deg'])
        for j,panel in enumerate(ex.get('panel_pose_metrics',[])):
            doc.row(ex['cycle']+'/'+ex['category']+'/'+panel['arm'],number(figures,pointer('examples',i,'panel_pose_metrics',j,'translation_cm')),
                    number(figures,pointer('examples',i,'panel_pose_metrics',j,'rotation_deg')))
        doc.line('')
    doc.para('개선 예시라도 절대 오류가 클 수 있으므로 정상 작동 사례로 보지 않는다. 악화 예시도 동일 기준으로 공개한다. 전체 출처·승인·실제 패널 수치는 [FIGURE_INDEX](FIGURE_INDEX.md)와 [FIGURE_MANIFEST](FIGURE_MANIFEST.json)에 연결된다.')


def full_tables(doc,baseline,results,derived,inventory):
    doc.title('전체 집계표',1)
    doc.para('음의 Δ가 개선이다. 모든 실수는 원 JSON 값을 소수점 여섯 자리로 표시한다. 중앙값의 차이와 frame별 차이의 중앙값은 별개다. conditional은 valid pose만 사용하며, fail은 전체 모집단 표에서 +∞로 남긴다. Wood Severe는 빈 모집단이므로 NA다. RAW/REF는 좌표 조건이며 반복 baseline은 원 recipe의 같은 seed 대조군이다.')
    doc.title('주 모집단: Plastic Moderate + Severe')
    pose_table(doc,selected_rows(baseline,results))
    doc.title('모든 재료·난도·recording: conditional T / R / yaw')
    rows=[]
    for material,content in baseline.value['materials'].items():
        for group,arms in content['groups'].items():
            for arm in arms:
                suffix=' [사후 참고선, 새 후보 아님]' if arm.startswith(('C2_','C3_')) else ''
                rows.append((f'BASE/{material}/{group}/{arm}'+suffix,baseline,pointer('materials',material,'groups',group,arm)))
    for r in results:
        rows += [(label(r,group+'/'+arm),r,pointer('groups',group,arm)) for group in r.value['groups'] for arm in ('NEW_RAW','NEW_REF')]
    pose_table(doc,rows)
    doc.title('Full population 및 camera x/z: 실패 포함, 차량좌표 아님')
    doc.table(['조건 / 모델','T full median / P90 cm','R full median / P90 deg','camera |x| median / P90 cm','camera |z| median / P90 cm'])
    for name,s,base in rows:
        vals=[combine(number(s,base+pointer('full_population',k,'median')),' / ',number(s,base+pointer('full_population',k,'P90'))) for k in KEYS[:2]]
        vals += [combine(number(s,base+pointer(axis,'absolute_cm','median')),' / ',number(s,base+pointer(axis,'absolute_cm','P90'))) for axis in ('camera_x','camera_z')]
        doc.row(name,*vals)
    doc.line('');doc.title('Paired 대비: 기존 REF, R0, matched RAW 및 incremental / 반복')
    contrasts=result_contrasts(results)+derived_contrasts(derived)
    contrast_table(doc,contrasts);direction_table(doc,contrasts)
    doc.title('Recording 하나씩 제외한 민감도: 같은 primary population')
    loro=[]
    for r in results:
        loro += [(label(r,'제외:'+rec+'/'+contrast),r,pointer('leave_one_recording_out',rec,contrast))
                 for rec,values in r.value['leave_one_recording_out'].items() for contrast in values]
    contrast_table(doc,loro)
    doc.title('보조 2D / ADD: 선택 목적 아님, PCK는 비율')
    doc.table(['조건 / 모델','corner 수','matched frame','PCK5','PCK10','PCK20','full penalty median / P90 px','tail >20 px 수','ADDsym AUC'])
    for r in results:
        for group,arms in r.value['groups'].items():
            for arm in arms:
                base=pointer('groups',group,arm);p=base+'/twoD'
                if arms[arm]['twoD'] is None:
                    doc.row(label(r,group+'/'+arm),*(['NA']*8));continue
                doc.row(label(r,group+'/'+arm),number(r,p+'/corners','d'),number(r,p+'/matched','d'),
                    *[number(r,p+pointer('PCK',k)) for k in ('5','10','20')],
                    combine(number(r,p+'/full_penalty_median_px'),' / ',number(r,p+'/full_penalty_P90_px')),
                    number(r,p+'/tail_gt20_count','d'),number(r,base+'/ADDsym_AUC'))
    doc.line('');doc.title('검증된 visible point 보조 집계: 독립 물리 pose GT 아님')
    doc.table(['조건 / 그룹 / 모델','point N','median / P90 px','PCK5','PCK10','PCK20','>20 px 수'])
    for r in results:
        if not r.value.get('verified66') or 'groups' not in r.value['verified66']:continue
        for group,arms in r.value['verified66']['groups'].items():
            for arm in arms:
                base=pointer('verified66','groups',group,arm)
                doc.row(label(r,group+'/'+arm),number(r,base+'/n','d'),combine(number(r,base+'/median_px'),' / ',number(r,base+'/p90_px')),
                    *[number(r,base+pointer('PCK',k,'fraction')) for k in ('5','10','20')],number(r,base+'/gt20','d'))
    doc.line('');doc.title('동일 synthetic validation: source 보존 보조, 물리 pose 평가 아님')
    doc.table(['실행 / 모델','box mAP50–95','pose mAP50–95','val pose loss','val RLE loss'])
    for r in results:
        for arm in r.value['source_validation32']['results']:
            base=pointer('source_validation32','results',arm)
            doc.row(label(r,arm),*[number(r,base+pointer(k)) for k in ('metrics/mAP50-95(B)','metrics/mAP50-95(P)','val/pose_loss','val/rle_loss')])
    doc.line('');doc.title('실제 fit: 마지막 checkpoint만 사용')
    doc.table(['실행 / 좌표 조건','명목 seed','updates','GPU fit wall sec','학습 scalar params / tensors','새 manual','추가 추론 component'])
    for r in results:
        for arm in ('RAW','REF'):
            base=pointer('fits',arm);ip=pointer('materials',r.value['material'])
            doc.row(label(r,arm),*[number(r,base+pointer(k),'d' if k!='seconds' else '.6f') for k in ('seed','optimizer_steps','seconds')],
                combine(number(inventory,ip+'/trainable_scalar_parameters','d'),' / ',number(inventory,ip+'/trainable_parameter_tensors','d')),
                number(r,base+'/manual_added','d'),number(inventory,'/extra_inference_components','d'))
    doc.line('')


def build(final=False):
    started=time.perf_counter();sources=[]
    def get(name,optional=False):
        path=C.DOC/name
        if optional and not path.exists():return None
        s=Source.read(path);sources.append(s);return s
    baseline=get('BASELINE_POSE_RESULTS.json');oracle=get('TR_ORACLE_RESULTS.json')
    ledger=get('RESOURCE_LEDGER.json');inputs=get('INPUT_BINDINGS.json');lock=get('METRIC_AND_SELECTION_LOCK.json')
    selection=get('FINAL_SELECTION.json',True);figures=get('FIGURE_MANIFEST.json',True)
    replication=get('REPLICATION_VALIDITY_CORRECTION.json',True)
    if replication:
        assert replication.value['effective_data_variation'] is False
        assert replication.value['deterministic_reexecution_verified'] is True
    loss=get('LOSS_SIGNAL_AUDIT.json');scale=get('LOSS_SCALE_PROBE.json');fmin=get('F_MINIMUM_DIAGNOSTIC.json')
    inventory=get('TRAINABLE_PARAMETER_INVENTORY.json');discussion=get('DISCUSSION_REVIEW.json',True)
    branch=get('BRANCH_DIAGNOSTIC.json',True)
    tests=get('TEST_RESULTS.json',True)
    protocols=[get(str(p.relative_to(C.DOC))) for p in sorted((C.DOC/'cycles').glob('*/PROTOCOL.json'))]
    results=[get(str(p.relative_to(C.DOC))) for p in sorted((C.DOC/'cycles').glob('*/RESULTS_*_S*.json'))]
    pending=pending_results(protocols,results,selection)
    figure_cycles={tuple(row[k] for k in ('cycle','material','seed')) for row in figures.value['completed_cycles']} if figures else set()
    if {result_key(r) for r in results}-figure_cycles:pending.append('FIGURES_NOT_UPDATED_FOR_ALL_COMPLETED_RESULTS')
    if replication is None:pending.append('REPLICATION_VALIDITY_CORRECTION.json')
    if discussion is None:pending.append('DISCUSSION_REVIEW.json')
    if branch is None:pending.append('BRANCH_DIAGNOSTIC.json')
    if tests is None:pending.append('TEST_RESULTS.json')
    if final:
        assert not pending,'Cannot finalize incomplete evidence: '+str(pending)
        assert tests.value['status']=='PASS','Final test artifact is not PASS'
        assert all(any(r.value['cycle']==cycle for r in results) for cycle in MAIN),'Main cycle result missing'
        assert ledger.value['totals']['fits']==sum(len(r.value['fits']) for r in results),'Ledger/results fit count not reconciled'
    derived_value=derived_results(results,selection,ledger,sources)
    # Read snapshots first; detect concurrent updates instead of binding stale numbers to new bytes.
    for source in sources:assert C.bind(source.path)==source.binding,'Changed report input; rerun: '+str(source.path)
    C.save(C.DOC/'REPORT_DERIVED_RESULTS.json',derived_value);derived=Source.read(C.DOC/'REPORT_DERIVED_RESULTS.json')
    dec=decision(results,selection,derived,pending,final,replication);C.save(C.DOC/'FINAL_DECISION.json',dec)
    decsource=Source.read(C.DOC/'FINAL_DECISION.json');trace=[];documents={}
    def doc(name,title=None):
        d=Document(name,trace);documents[name]=d
        if title:d.title(title,1)
        d.para('상태: **'+dec['status']+'**. EXECUTION='+dec['EXECUTION']+'; POSE_OUTCOME='+dec['POSE_OUTCOME']+'; REPEAT='+dec['REPEAT']+'.')
        if pending:d.para('아직 완료로 주장하지 않는 항목: '+', '.join(pending)+'.')
        return d
    tables=doc('FINAL_TABLES.md');full_tables(tables,baseline,results,derived,inventory)
    report=doc('REPORT_KO.md','Translation / rotation 우선 후속 실험')
    report.title('문제')
    report.para('기존 REF의 보조 지표 향상이 실제 위치와 방향의 동시 개선인지 다시 묻는다. 주 모집단은 자연 가림 Plastic Moderate + Severe의 고정 frame 집합이며, 전체 Plastic 및 Wood의 난도·recording도 함께 공개한다. OLD_RAW / OLD_REF는 과거 원 recipe, NEW_RAW / NEW_REF는 각 실행의 matched 좌표 조건이다.')
    report.title('왜 이 개입인가')
    report.para('A는 입력에만 가림을 추가하여 보이는/가려지는 감독점의 학습 신호를 유지하는 가설이다. B는 기존 location 항의 큰 오차 감쇠를 보완하는 정규화 좌표 SmoothL1 항이다. C는 A의 실제 가림 노출이 제한적이었다는 관측에서 schedule만 높인 대조다. B가 두 주 지표를 모두 개선하지 못하여 A+B 결합은 실행하지 않았다. C 선택은 기존 REF 대비 두 부호가 아주 조금 좋아진 사실만을 근거로 한 반복 대상 지정이며 성능 승격이 아니다.')
    report.title('선행연구와 다른 점')
    report.para('가림과 easy–hard 학습 원리, 좌표 regression과 heatmap 손실을 구분했다. 현재 head의 RLE는 이미 존재하며 optical flow가 아니다. dense detector Focal과 heatmap Adaptive Wing을 현재 pose 좌표에 그대로 이식하지 않았다. 원문·공식 구현의 열람 범위와 전제 차이는 [RELATED_WORK](RELATED_WORK.md), [SOURCE_REGISTRY](SOURCE_REGISTRY.json)에 남겼다.')
    report.table(['아이디어','실제 처리와 해석'])
    for k,v in [('A','입력 가림: paired fit 및 평가 완료. 결과는 tradeoff.'),('B','좌표 보완: TRAIN runtime/gradient 진단 후 paired fit. 기존 REF 대비 두 중앙값 모두 악화.'),('C','유효 가림 노출 증가: paired fit. 작은 joint sign만 있어 같은 seed의 원 recipe 대조군을 둔 추가 반복 대상으로 지정.'),('D','동결 detector의 Focal은 적용 대상 아님. kobj/GHMR은 별도 가설로 보류; 실패로 기록하지 않음.'),('E','soft-target 경계·질량·ignore CPU fixture만 확인. heatmap/teacher 전이 fit은 실행하지 않음.'),('F','과거 저장 refiner의 제한된 자산 재사용 가능성만 CPU 확인. 현재 pool의 새 teacher/student 전이는 실행하지 않음.')]:report.row(k,v)
    report.line('');report.title('사용한 정보')
    report.para('기존 TRAIN 이미지·동결 pseudo target·source replay만 사용한다. 평가 reference는 paired native prediction 및 공통 D9 pose가 저장·잠긴 뒤에만 점수 계산에 사용한다. 2D click과 기존 geometry reference는 독립적인 물리 축 검증과 같지 않다. 새 manual 감독이나 sealed TEST 개봉은 없다. frame 좌표·K·pose 배열·checkpoint는 비공개 data 경로에 남기고, 공개 JSON은 집계 및 해시/경로 메타데이터만 담는다. 기존 teacher 노출과 재료 pool 차이는 [POOL_AND_SUPERVISION_AUDIT](POOL_AND_SUPERVISION_AUDIT.md)에 명시했다.')
    report.title('공정 대조와 지표 계약')
    report.para('각 RAW/REF 쌍은 같은 R0 초기화, RGB·이름 순서·box, source 비율, update 수 및 pose/flow 학습 범위를 공유한다. detector/backbone과 BN·buffer는 고정한다. 기존 augmentation의 경계 clipping 때문에 좌표 조건에 따라 최종 지원점이 소수 batch에서 다르므로 “모든 최종 mask가 완전히 같다”고 주장하지 않는다. 실제 batch trace 및 차이 수를 [EXPERIMENT_LOG](EXPERIMENT_LOG.md)에 공개한다.')
    report.para('실제 R0 구조와 trainer 선택식을 CPU로 확인한 학습 범위는 재료별 ',number(inventory,'/materials/PLASTIC/trainable_scalar_parameters','d'),
                ' scalar parameters / ',number(inventory,'/materials/PLASTIC/trainable_parameter_tensors','d'),
                ' parameter tensors이며, 전체 ',number(inventory,'/materials/PLASTIC/state_tensors','d'),' state tensors 중 ',
                number(inventory,'/materials/PLASTIC/protected_state_tensors','d'),'개를 동결한다. [정확한 재료별 inventory](TRAINABLE_PARAMETER_INVENTORY.json).')
    report.para('T는 같은 pallet centroid의 유클리드 거리(cm), R은 object→camera full rotation의 물리 C2 대칭 최소 geodesic(deg)이다. yaw는 보조이고 camera x/z는 차량 좌표가 아니다. 실패를 삭제한 conditional 값만 보지 않고 valid/N와 실패를 +∞로 보존한 full-population quantile을 확인한다. severity 중앙값의 평균이 아니라 실제 frame들을 합쳐 primary를 계산한다. 계약: [METRIC_CONTRACT](METRIC_CONTRACT.md).')
    report.title('실제 T · R 결과')
    pose_table(report,selected_rows(baseline,results))
    contrasts=result_contrasts(results,[M.PRIMARY])+derived_contrasts(derived,True)
    contrast_table(report,contrasts)
    report.para('C의 기존 REF 대비 변화는 수치 부호상의 작은 개선에 불과하다. R0 및 같은 C recipe의 RAW보다 두 축이 나쁘다는 반증을 함께 둔다. C−A는 T가 악화하고 R이 개선하는 tradeoff이므로 노출 증가 자체의 joint 개선으로 읽을 수 없다. 반복 recipe는 동일 추가 seed의 BASELINE_REPEAT와 비교해야 하며 과거 seed의 OLD_REF와 비교한 수치만으로 recipe 효과를 주장하지 않는다.')
    report.para('전체 Plastic/Wood, CLEAN/Moderate/Severe, recording별 N·valid·T/R/yaw·P90·full-population·camera 성분, 보조 PCK/ADD 및 source validation은 [FINAL_TABLES](FINAL_TABLES.md)에 모두 있다. Wood의 없는 Severe를 영점 성능으로 채우지 않는다. 과거 C2/C3는 사후 참고선이며 새 후보 선정에 재사용하지 않는다.')
    wood=[r for r in results if r.value['material']=='WOOD']
    if wood:
        w=wood[-1];p=pointer('groups','ALL');wc=result_contrasts([w],['ALL'])
        report.para('Wood 적용성은 선택 후의 기술적 보조 검사다. 전체 ',number(w,p+'/NEW_REF/frames','d'),
                    ' frame에서 기존 REF T ',number(w,p+'/OLD_REF/conditional/translation_cm/median'),' → ',number(w,p+'/NEW_REF/conditional/translation_cm/median'),
                    ' cm, R ',number(w,p+'/OLD_REF/conditional/rotation_deg/median'),' → ',number(w,p+'/NEW_REF/conditional/rotation_deg/median'),
                    ' deg였다. 기존 REF 및 matched RAW 대비 두 중앙값의 부호가 개선되지만 R0 대비는 T 개선·R 악화이며, Plastic 결과의 독립적 재현으로 취급하지 않는다.')
        report.para('Wood를 자연 난도로 나누면 기존 REF 대비 R 중앙값은 CLEAN에서 ',
                    number(w,pointer('contrasts','severity:CLEAN','NEW_REF-minus-OLD_REF','difference_of_conditional_medians','rotation_deg'),'+.6f'),
                    ' deg, Moderate에서 ',number(w,pointer('contrasts','severity:MODERATE_OCCLUSION','NEW_REF-minus-OLD_REF','difference_of_conditional_medians','rotation_deg'),'+.6f'),
                    ' deg로 둘 다 악화한다. 전체 pooled 중앙값의 작은 개선을 난도 전반의 회전 개선으로 해석하면 안 된다.')
        contrast_table(report,wc)
    report.title('큰 실패와 손익')
    direction_table(report,contrasts)
    figure_lines(report,figures,selection.value.get('selected_cycle') if selection else None)
    report.para('평균적인 작은 변화는 큰 절대 실패가 사라졌다는 뜻이 아니다. paired 사분면과 tie를 분리하고, recording 하나를 제외한 민감도 및 P90를 전체 표에서 함께 본다. point 기준의 검증된 visible 보조 평가는 pose의 물리적 참조 정확도를 대신하지 않는다.')
    if discussion:
        p=pointer('Plastic_comparisons','C42_REF_minus_OLD42_REF')
        report.para('독립 재집계에서는 C REF의 R 중앙값 차이가 ',number(discussion,p+'/difference_of_medians/rotation_deg','+.6f'),
                    ' deg인 것과 달리 frame별 R 변화의 중앙값은 ',number(discussion,p+'/median_of_paired_differences/rotation_deg','+.6f'),
                    ' deg였다. recording REC_007을 제외하면 ΔT ',number(discussion,pointer('Plastic_leave_one_recording_out','C42_REF_minus_OLD42_REF','REC_007','difference_of_medians','translation_cm'),'+.6f'),
                    ' cm, ΔR ',number(discussion,pointer('Plastic_leave_one_recording_out','C42_REF_minus_OLD42_REF','REC_007','difference_of_medians','rotation_deg'),'+.6f'),
                    ' deg로 두 부호 모두 악화한다. 이 민감도는 강한 성능 주장과 맞지 않는다. [독립 해석 검토](DISCUSSION_REVIEW.json).')
    report.title('원인 판정')
    report.para('원인은 COMPETING_EXPLANATIONS이다. 실제 TRAIN probe는 location 감쇠와 일부 branch의 RLE clamp를 구분했고, 큰 오류에서도 합산 좌표 신호가 남는 반례를 확인했다. 따라서 “모든 큰 오답에서 전체 gradient가 사라진다”는 원인 설명은 성립하지 않는다. B의 부정 결과는 이 보완이 현 조건에서 유익하다는 가설을 약화하지만 모든 강건 손실의 실패를 뜻하지 않는다.')
    if branch:
        pp=pointer('materials','PLASTIC',M.PRIMARY,'comparisons','NEW_REF-minus-OLD_REF')
        wp=pointer('materials','WOOD','ALL','comparisons','NEW_REF-minus-OLD_REF')
        matched=pointer('materials','PLASTIC',M.PRIMARY,'comparisons','NEW_REF-minus-NEW_RAW')
        report.para('저장 후보의 branch 분해에서 C REF−기존 REF는 Plastic primary ',number(branch,pp+'/counts/same_branch','d'),'/',number(branch,pp+'/frames','d'),
                    ', Wood ',number(branch,wp+'/counts/same_branch','d'),'/',number(branch,wp+'/frames','d'),
                    ' frame 모두 같은 branch였다. 이 recipe 변화는 router branch 전환이 아니라 같은 branch에서의 해 변화다. matched C RAW→REF의 branch 전환은 ',
                    number(branch,matched+'/counts/branch_switch','d'),' frame이다. PCK10 정답점 수가 늘어난 ',number(branch,matched+'/counts/frame_PCK10_count_gain','d'),
                    ' frame 중 ',number(branch,matched+'/counts/frame_PCK10_count_gain_both_pose_worse','d'),
                    '개는 T와 R이 모두 악화했다. 2D 향상만으로 pose 향상을 추론할 수 없다. [CPU branch 진단](BRANCH_DIAGNOSTIC.md)은 고정 출력의 사후 분해이며 새 selector·fit이 아니다.')
    report.para('단일 동결 TRAIN batch에서 보완 후 원 objective 대비 gradient norm 비율은 ',number(scale,'/combined_over_original'),
                ', 방향 cosine은 ',number(scale,'/original_supplement_cosine'),
                '였다. 크기 변화가 완전히 통제된 fit은 아니며 이 비율이 학습 전체에서 유지된다고 주장하지 않는다. [LOSS_SIGNAL_AUDIT](LOSS_SIGNAL_AUDIT.md), [LOSS_SCALE_PROBE](LOSS_SCALE_PROBE.json).')
    report.para('고정 D9 후보 oracle은 T 최소와 R 최소가 서로 다른 pose일 수 있음을 그대로 남긴다. GT 의존적인 진단 상한을 배포 selector로 사용하거나 두 최적값을 한 pose의 실현 가능한 성능으로 합치지 않았다. 기존 geometry/ID와 actual image cue의 불일치, noisy pseudo target, head 표현력은 아직 경쟁 설명이다.')
    report.table(['기존 REF 고정 후보 진단 / Plastic primary','T median cm','R median deg'])
    op=pointer('materials','PLASTIC',M.PRIMARY,'fixed_D9_candidates','REF_LR5')
    for name,key in [('기존 선택','current'),('T 최적 후보의 완전한 pose','translation_optimal'),('R 최적 후보의 완전한 pose','rotation_optimal')]:
        report.row(name,number(oracle,op+pointer(key,'conditional','translation_cm','median')),number(oracle,op+pointer(key,'conditional','rotation_deg','median')))
    report.line('');report.para('같은 한 후보가 T와 R을 모두 개선하는 frame은 ',number(oracle,op+'/frames_with_same_candidate_joint_gain','d'),
                              ', T 최적과 R 최적 후보가 다른 frame은 ',number(oracle,op+'/T_R_optimal_choices_different','d'),
                              '이다. 기존 후보 집합의 GT-dependent 진단이며 새 학생의 실제 달성 성능이나 oracle gap 회복률이 아니다.')
    report.title('기존 REF 및 R0 대비 가치')
    report.para('실행·진단 완료와 성능 개선 성공은 별개다. A는 tradeoff, B는 기존 REF 대비 양 축 NO_GAIN이다. C는 작은 joint sign 때문에 반복했지만 그 사실만으로 실용적 개선을 인정하지 않는다. 좌표 보정의 가치는 각 실행의 NEW_REF−NEW_RAW로 따로 보며, 기존 REF−R0 비교를 숨기지 않는다. 추가 teacher inference나 별도 deployment component는 없다; train-only augmentation/손실 변경과 같은 학생·D9 추론 구조다.')
    report.para('유효한 학습 변동성 반복 판정: ',dec['REPEAT'],'. 명목 추가 seed 재실행의 원 recipe 대비 수치 분류: ',dec['repeat_classification'],'. 최종 해석: ',dec['BEST_SUPPORTED_RECIPE'],'. 명목 trainer seed를 바꿨지만 data loader의 실제 worker 난수가 동일해 trace와 모든 model state가 같았다. 학습 비용을 소비한 결정론적 수치 재실행이지 training-seed 변동성 검증이 아니다. 독립적인 pretrained seed나 독립 DEV 검증도 아니다. [반복 유효성 정정](REPLICATION_VALIDITY_CORRECTION.json).')
    if replication:
        report.para('이 실패한 반복 설계에도 실제 ',number(replication,'/counted_attempts/fits','d'),' fits / ',
                    number(replication,'/counted_attempts/optimizer_updates','d'),
                    ' updates가 소모되었으며 비용을 지우지 않았다. 최초 재실행 전에 실제 loader stream 차이를 확인하지 않은 실행 검증 누락이며, 프레임워크 탓만으로 돌리지 않는다. 남은 fit 예산이 없어 유효한 학습 변동성 반복은 미완료다.')
    report.title('한계')
    report.para('고정 예산 안의 재사용 DEV 탐색이며 독립적 의미 있는 효과 크기나 배포 안전성을 확증하지 않는다. 자연 가림 TRAIN severity를 별도로 검증하지 않은 점, synthetic source 보조 validation과 물리 pose 평가의 차이, manual 정보의 기존 노출, Wood의 다른 원 pool을 유지한 적용성 검사를 공개한다. 이전 원고와 과거 실험 산출물은 바꾸지 않는다.')
    report.title('다음 결정 하나')
    report.para(dec['NEXT_ONE_DECISION'])
    chosen=next((r for r in results if selection and r.value['cycle']==selection.value['selected_cycle'] and r.value['material']=='PLASTIC'),None)
    if chosen:
        old=pointer('materials','PLASTIC','groups',M.PRIMARY,'OLD_REF','conditional');new=pointer('groups',M.PRIMARY,'NEW_REF','conditional')
        report.para('마지막 수치: Plastic Moderate + Severe의 실제 ',number(baseline,pointer('materials','PLASTIC','groups',M.PRIMARY,'OLD_REF','frames'),'d'),
                    ' frame에서 기존 REF → 선택 반복 대상 ',chosen.value['cycle'],' REF의 T 중앙값은 ',number(baseline,old+'/translation_cm/median'),' → ',number(chosen,new+'/translation_cm/median'),
                    ' cm, R 중앙값은 ',number(baseline,old+'/rotation_deg/median'),' → ',number(chosen,new+'/rotation_deg/median'),' deg이며, 실용적 joint gain으로 승격하지 않는다.')
    claim=doc('CLAIM_IMPACT.md','원고 주장에 미치는 영향 — 수정 제안만')
    claim.para('원래 원고는 수정하지 않았다. 추가할 수 있는 것은 재사용 DEV에서 pose-first 평가와 실패한/혼합된 개입을 구분한 제한적 후속 진단이다. 2D 또는 ADDsym 향상을 곧바로 translation과 full rotation의 joint 개선으로 번역하지 않는다. C의 미세한 sign만으로 새로운 우수 recipe라고 주장하지 않는다.')
    contrast_table(claim,contrasts)
    claim.para('허용되는 표현: 같은 감독·모델·추론 구조 아래의 paired 비교, 기존 REF보다 개선/악화된 축의 실제 수치, matched RAW와 R0의 반증, 반복의 원 recipe 같은-seed 대조. 허용되지 않는 표현: 독립 물리 pose 확증, 배포 안전성, heatmap/refiner/GHM의 미실행을 실패로 처리, 진단 성공을 성능 성공으로 치환.')
    nxt=doc('NEXT_DECISION.md','다음 결정 하나')
    nxt.para(dec['NEXT_ONE_DECISION']);nxt.para('후속 촬영·라벨링·sealed TEST 개봉은 이 보고서가 자동 승인하지 않는다. 현재 사용자 조치가 필요한 기술 blocker는 보고서 생성만으로 만들어 내지 않는다. 미완료 fit/eval은 위 PENDING 목록으로만 표시한다.')
    reproduce=doc('REPRODUCE.md','집계 보고서 재생성')
    reproduce.para('이 문서의 명령은 학습을 추가하지 않는다. 기존 lock이 불일치하면 원본을 고치지 말고 중단한다. 실제 fit 재실행·새 seed·미완료 run 덮어쓰기는 별도 승인 및 budget 검토가 필요하다.')
    reproduce.para('```bash\n'+PYTHON+' -m scripts.research.pallet_pose_objective_followup_v2.final_report\n'+PYTHON+' -m unittest scripts.research.pallet_pose_objective_followup_v2.test_final_report\n'+PYTHON+' -m scripts.research.pallet_pose_objective_followup_v2.final_audit\n```')
    reproduce.para('branch 분해 CPU 재계산 명령(일반 inference 또는 학습을 추가하지 않음):\n\n```bash\n'+PYTHON+' -m scripts.research.pallet_pose_objective_followup_v2.branch_diagnostic\n```')
    reproduce.para('모든 선언된 결과·최신 그림·원장 정산이 끝난 뒤에만 `final_report --final`을 사용한다. 이 명령은 CPU private pose metrics로 C−A 및 동일 반복 seed의 recipe−원 baseline 집계를 재계산한다. JSON pointer와 SHA는 REPORT_NUMERIC_TRACE에 기록된다. 원장과 STATE는 실행 조정자만 수정한다.')
    reproduce.para('완료된 평가의 frozen cache 확인/재집계 명령:')
    for r in results:
        v=r.value;reproduce.para('```bash\n'+PYTHON+' -m scripts.research.pallet_pose_objective_followup_v2.eval_student score --cycle '+v['cycle']+' --material '+v['material']+' --seed '+str(v['seed'])+'\n```')
    reproduce.title('이미 완료된 학습의 정확한 기록 명령 — 지금 추가 실행하지 말 것')
    reproduce.para('아래는 같은 원 recipe와 immutable protocol을 재현하기 위한 역사적 entrypoint 기록이다. 현 예산에서 새 fit을 허가하는 명령 목록이 아니다. 완료 checkpoint와 실패/부분 로그를 보존하며, 독립 재현이 필요하면 별도 namespace·예산·승인을 먼저 정한다. 명목 추가 seed 명령은 실제 worker stream을 바꾸지 못했던 원 실행 그대로이며, 유효한 확률적 반복 방법으로 제시하지 않는다.')
    for r in results:
        v=r.value;cycle=v['cycle'];module='loss_student' if cycle==MAIN[1] else 'exposure_student' if cycle in (MAIN[2],'RECIPE_REPEAT','WOOD_APPLICABILITY') else 'student'
        reproduce.para('실행 `'+cycle+'`의 [잠긴 protocol](cycles/'+cycle+'/PROTOCOL.json), [사전 SPEC](cycles/'+cycle+'/SPEC.md), [결과](cycles/'+cycle+f'/RESULTS_{v["material"]}_S{v["seed"]}.json).')
        commands=[]
        for target in ('RAW','REF'):
            commands.append(PYTHON+' -m scripts.research.pallet_pose_objective_followup_v2.'+module+' train --cycle '+cycle+' --material '+v['material']+' --target '+target+' --seed '+str(v['seed']))
        commands.append(PYTHON+' -m scripts.research.pallet_pose_objective_followup_v2.student parity --cycle '+cycle+' --material '+v['material']+' --seed '+str(v['seed']))
        commands.append(PYTHON+' -m scripts.research.pallet_pose_objective_followup_v2.eval_student infer --cycle '+cycle+' --material '+v['material']+' --seed '+str(v['seed']))
        reproduce.para('```bash\n'+'\n'.join(commands)+'\n```')
    reproduce.para('마지막 명목 seed 대조·recipe 재실행·Wood 실행은 [run_closure.py](../../../scripts/research/pallet_pose_objective_followup_v2/run_closure.py)가 순차 관리했다. 로그 및 각 FIT JSON은 `data/pallet/results/pallet_pose_objective_followup_v2/closure_logs/`와 `cycles/<cycle>/`에 있다. 전처리 preflight/lock의 입력·구현 SHA는 각 protocol에 잠겨 있다. 이 orchestrator도 지금 다시 실행하지 않는다.')
    if tests:
        reproduce.para('실제 검사 결과: ',number(tests,'/tests','d'),' tests ',tests.value['status'],', ',number(tests,'/seconds'),' sec. source-file SHA와 원 pytest 출력은 [TEST_RESULTS](TEST_RESULTS.json)에 보존했다.')
        reproduce.para('```bash\n'+tests.value['command']+'\n```')
    log=doc('EXPERIMENT_LOG.md','실제 실행과 비용')
    log.para('누적 fits ',number(ledger,'/totals/fits','d'),', updates ',number(ledger,'/totals/optimizer_updates','d'),
             ', GPU 경과 ',number(ledger,'/totals/gpu_seconds'),' sec (',number(derived,'/resources/GPU_hours'),
             ' h), 원장 시점 active wall ',number(ledger,'/totals/elapsed_wall_seconds'),' sec (',number(derived,'/resources/recorded_active_wall_hours'),' h). GPU 값은 CUDA event kernel 합이 아니라 해당 GPU 작업 구간의 wall-time이다. 사후 보고서 실행 시간은 자동으로 원장에 쓰지 않는다.')
    log.table(['event','wall sec','GPU sec','fits','updates'])
    for i,event in enumerate(ledger.value['events']):log.row(event['event'],*[number(ledger,pointer('events',i,k),'d' if k in ('fits','optimizer_updates') else '.6f') for k in ('wall_seconds','gpu_seconds','fits','optimizer_updates')])
    log.line('');log.title('실제 paired 노출과 지원점')
    log.table(['pair','batch','source RGB','real RGB','real 감독점','real ignore','support 차이 batch','가림 적용','REF 가린 감독점'])
    for r in results:
        p=get('cycles/'+r.value['cycle']+f'/PARITY_{r.value["material"]}_S{r.value["seed"]}.json')
        log.row(label(r,'RAW+REF'),number(p,'/batches','d'),number(p,'/exposures/SOURCE_images','d'),number(p,'/exposures/REAL_images','d'),
            number(p,'/exposures/REAL_supervised','d'),number(p,'/exposures/REAL_ignore','d'),number(p,'/differences/support','d'),
            number(p,'/occlusion/applied','d') if 'applied' in p.value.get('occlusion',{}) else '해당 없음',
            number(p,'/occlusion/REF_masked_supervised','d') if 'REF_masked_supervised' in p.value.get('occlusion',{}) else '해당 없음')
    log.line('');log.para('원래 geometric out-of-frame 처리의 RAW/REF 지원 차이는 선언된 차이이며 후처리로 공통 mask를 만든 것이 아니다. synthetic RGB 변환·샘플링은 원 recipe를 유지한다. 기술 재시도 및 자원 이벤트의 누락 여부는 최종 AUDIT와 실행 조정자의 로그를 함께 확인한다.')
    log.title('원본 / 가림 TRAIN runtime probe — 전체 TRAIN 빈도 아님')
    log.table(['checkpoint / mode / branch','supervised assigned','e max','RLE clamp 전 / 후','location head norm','RLE head norm','combined head norm'])
    for i,row in enumerate(loss.value['results']):
        for j,branch in enumerate(row['branches']):
            base=pointer('results',i,'branches',j)
            log.row(row['job']+'/'+row['mode']+'/'+branch['branch'],number(loss,base+'/supervised_assigned_points','d'),number(loss,base+'/e/max'),
                combine(number(loss,base+'/rle_before_clamp'),' / ',number(loss,base+'/rle_after_clamp')),
                *[number(loss,base+pointer('terms',term,'head_gradient_norm')) for term in ('location','rle','combined')])
    log.line('');log.para('RLE branch 전체 clamp와 location 감쇠를 구분한다. C3 실패점의 실제 좌표·assignment 및 descent 증거와 ignored channel 검산은 [LOSS_SIGNAL_AUDIT](LOSS_SIGNAL_AUDIT.md)의 private-bound 진단에서 확인한다. 여기 표를 큰 오차의 전체 gradient 소실이나 전체 TRAIN 빈도로 일반화하지 않는다.')
    cli=doc('CLI_REPORT_KO.md','CLI 종료 보고')
    cli.para('STATUS: '+dec['EXECUTION']+' / '+dec['POSE_OUTCOME']+' / '+dec['REPEAT']+' / REUSED_DEV_ONLY')
    cli.para('START_HEAD: `'+inputs.value['git']['HEAD']+'`. Git 공개 검증은 push 후 별도 `PUBLICATION.json`에 실제 results commit / remote HEAD / status를 기록한다. 이 결과 생성기는 push 완료나 자기 자신을 담을 미래 commit SHA를 추정하지 않는다. 최종 문서 commit은 CLI 인계 메시지에서 별도로 확인한다.')
    cli.para('PRIMARY_OBJECTIVE: centroid T(cm)와 C2 full R(deg) 동시 개선. PRIMARY_POPULATION: 고정 Plastic Moderate + Severe. METRIC_CONVENTION: [METRIC_CONTRACT](METRIC_CONTRACT.md).')
    cli.para('IDEA_A_TO_F: A paired input occlusion / B paired coordinate supplement / C paired increased exposure; D detector 부적합·회귀항 보류; E CPU soft-target fixture만; F 과거 자산 CPU 최소 진단만. 상세 [REPORT_KO](REPORT_KO.md).')
    cli.para('ACTUAL_FITS: ',number(ledger,'/totals/fits','d'),' / UPDATES: ',number(ledger,'/totals/optimizer_updates','d'),
             ' / GPU_TIME_SEC: ',number(ledger,'/totals/gpu_seconds'),' / WALL_TIME_SEC: ',number(ledger,'/totals/elapsed_wall_seconds'),'. TECHNICAL_RETRIES: 원장 event 및 최종 AUDIT 확인; 없다고 추정하지 않는다.')
    cli.para('BEST_SUPPORTED_RECIPE: '+dec['BEST_SUPPORTED_RECIPE']+'. TEACHER_ONLY_GAIN_VS_STUDENT_GAIN: 새 teacher fit/추론 없음. ADDED_INFERENCE_COST: 새로운 component 없음; 측정 latency 불변을 자동 주장하지 않는다.')
    pose_table(cli,selected_rows(baseline,results));contrast_table(cli,contrasts)
    cli.para('TAILS_AND_COVERAGE / PAIRED_DIRECTION_COUNTS / REPEAT_RESULTS: [FINAL_TABLES](FINAL_TABLES.md). WHAT_WAS_FIXED: 평가 목적과 손실/노출 검증. WHAT_WAS_RULED_OUT: 전체 좌표 gradient가 항상 소실된다는 설명, B의 현재 paired 이득. WHAT_REMAINS_UNKNOWN: 큰 실패의 물리 reference/축 불일치와 좌표 신호의 상대 기여.')
    cli.para('REPORT_PATH: REPORT_KO.md. REPRODUCE_COMMAND: `'+PYTHON+' -m scripts.research.pallet_pose_objective_followup_v2.final_report'+(' --final' if final else '')+'`. NEXT_ONE_DECISION: '+dec['NEXT_ONE_DECISION'])
    cli.para('HUMAN_ACTION_REQUIRED: 현재 보고서 생성만으로 새 필수 행동을 요구하지 않는다. 실제 기술 blocker는 별도 상태를 따른다.')
    if tests:cli.para('TESTS: ',number(tests,'/tests','d'),' ',tests.value['status'],' / ',number(tests,'/seconds'),' sec. [정확한 명령·출력](TEST_RESULTS.json).')
    if chosen:cli.para(report.lines[-2])  # Same final sentence; its numeric trace is copied below.
    if chosen:
        report_last=report.lines[-2]
        for entry in list(trace):
            if entry['document']=='REPORT_KO.md' and entry['line_contains']==report_last:
                trace.append(dict(entry,document='CLI_REPORT_KO.md',line_number=len(cli.lines)-1))
    for cycle in MAIN[1:]:
        rr=[r for r in results if r.value['cycle']==cycle]
        if not rr:continue
        d=doc('cycles/'+cycle+'/REPORT_KO.md',cycle+' 결과')
        d.para('사전 SPEC과 실제 RESULTS는 그대로 보존했다. 아래 표는 완료 결과의 JSON을 직접 집계한다. 음의 Δ가 개선이다. 기존 REF·R0·matched RAW 대비를 모두 공개한다.')
        for r in rr:
            group=M.PRIMARY if r.value['material']=='PLASTIC' else 'ALL'
            pose_table(d,[(label(r,a),r,pointer('groups',group,a)) for a in ('R0','OLD_REF','NEW_RAW','NEW_REF')])
        cr=result_contrasts(rr,[M.PRIMARY,'ALL']);contrast_table(d,cr);direction_table(d,cr)
        if cycle==MAIN[1]:d.para('기존 REF 대비 T와 R 모두 NO_GAIN이다. 유효 좌표 신호의 감쇠를 보완한 이 구현은 이 고정 조건에서 유리하지 않았다. 전체 gradient 소실이나 pseudo target의 정답성은 입증하지 못했으며, 보완 항의 gradient 크기 혼입도 남는다. 결합 후보로 승격하지 않는다.')
        else:
            contrast_table(d,[row for row in derived_contrasts(derived,True) if row[0].startswith('C-minus-A')])
            d.para('기존 REF 대비 작은 joint sign만 보였으나 matched RAW와 R0보다 양 축에서 나쁘다. C−A 역시 tradeoff이다. 따라서 반복 대상으로만 잠갔고, 같은 추가 seed의 원 recipe 대조와 Wood 적용성 결과는 상위 최종 보고서에서 확인한다.')
        d.para('[전체 보고서](../../REPORT_KO.md), [전체 난도·recording 표](../../FINAL_TABLES.md), [진단 한계](../../LOSS_SIGNAL_AUDIT.md).')
    # Rendering is all completed in memory before any document is replaced.
    for source in sources:assert C.bind(source.path)==source.binding,'Changed report input; rerun: '+str(source.path)
    for name,document in documents.items():C.save(C.DOC/name,document.content())
    C.save(C.DOC/'REPORT_NUMERIC_TRACE.json',dict(created_at=C.now(),status=dec['status'],entries=trace,
        scope='Generated measured numeric cells/sentences; names, protocol constants and quoted existing captions are not measured numeric claims.',
        documents=[C.bind(C.DOC/name) for name in documents],implementation=C.bind(Path(__file__)),
        source_documents=[C.bind(C.DOC/name) for name in ('LOSS_SIGNAL_AUDIT.md','RELATED_WORK.md','FIGURE_INDEX.md') if (C.DOC/name).exists()]))
    return dict(status=dec['status'],pending=pending,documents=len(documents),numeric_trace_fields=len(trace),
                CPU_wall_seconds=time.perf_counter()-started,fit_or_GPU_operations=0,ledger_or_old_file_writes=0)


def main():
    import json
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--final',action='store_true')
    args=parser.parse_args();print(json.dumps(build(args.final),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
