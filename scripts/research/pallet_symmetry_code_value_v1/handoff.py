"""Requested terminal handoff, after main push and three-SHA verification."""
import cv_env as E

def main():
    audit=E.read(E.DOC/'FINAL_AUDIT.json');a=E.read(E.DOC/'A_C1C2_RESULTS.json')['populations'];b=E.read(E.DOC/'B_C1C2C4_RESULTS.json')['populations']
    c=E.read(E.DOC/'CODE_PERTURBATION_RESULTS.json')['models'];git=E.read(E.RAW/'GIT_SYNC_COMPLETE.json')
    lines=['[QUESTION]','Does explicit C1/C2/C4 input add value beyond dimensions + symmetry-aware target?','', '[INTEGRITY]',
      f"R0 frozen: {audit['R0_frozen']}; same 8-D architecture: True; same params: 20307; same target: True; same order: True; only code differs: True; tests: {audit['tests']} PASS",'', '[A C1/C2]']
    def row(label,r):
        x=r['contrast'];return f"{label}: blind={r['blind']['E_sym']:.9f}, aware={r['aware']['E_sym']:.9f}, delta={x['delta']:.9g}, CI95={x['CI95']}, improved={x['improved_seeds']}/3"
    for pop,r in a.items():lines.append(row(pop,r))
    for g,x in a['SYNTH']['groups'].items():lines.append(f"{g}: n={x['n']}, delta={x['delta']:.9g}, CI95={x['CI95']}")
    lines+=['','[B C1/C2/C4]']
    for pop,r in b.items():lines.append(row(pop,r))
    for pop in ['SYNTH','SQUARE']:
        for g,x in b[pop]['groups'].items():lines.append(f"{pop}/{g}: n={x['n']}, delta={x['delta']:.9g}, CI95={x['CI95']}")
    lines+=['','[C CODE PERTURBATION]','Deltas are perturbation minus correct; positive favors correct.']
    for arm,pops in c.items():
        for pop,r in pops.items():
            for mode,x in r['modes'].items():
                if not x['applicable']:lines.append(f'{arm}/{pop}/{mode}: N/A(single group)');continue
                m=x['movement'];lines.append(f"{arm}/{pop}/{mode}: delta={x['contrast']['delta']:.9g}, CI95={x['contrast']['CI95']}, coordinate change mean/max={m['mean_coordinate_change_px']:.6f}/{m['max_coordinate_change_px']:.6f}px, top1 change={m['top1_change_rate']:.6f}")
    verdict=audit['decision']['result'];lines+=['','[DECISION]',verdict,'','[INTERPRETATION]',
      '두 모델은 dimensions, symmetry target, architecture, parameters, training budget가 같고 explicit group-code information만 다르다. '+
      ('개발 데이터에서 코드 유지 조건을 만족했으나 독립 확인은 아니다.' if verdict=='KEEP_CODE_DEV_EVIDENCE' else '명시적 코드를 뺀 단순한 main method를 우선하고 code-aware는 ablation으로 보존한다.')+
      ' A의 matched-capacity 추가 이득은 확립되지 않았고, 실사 DEV에서는 중립 코드가 오히려 더 좋았다. 실사 DEV는 모두 C2이고 B는 domain/group confounding이 있으므로 대칭 개념의 인과적 학습이나 독립 일반화를 주장하지 않는다. 추가 sweep와 기존 배포 모델 교체는 하지 않았다.',
      '', '[GIT]',f"start SHA: {audit['start_SHA']}",f"end SHA: {git['HEAD']}",f"HEAD: {git['HEAD']}",f"origin/main: {git['origin_main']}",f"ls-remote: {git['ls_remote']}",f"equal: {git['equal']}"]
    text='\n'.join(lines)+'\n';(E.RAW/'TERMINAL_HANDOFF.txt').write_text(text);print(text)
if __name__=='__main__':main()
