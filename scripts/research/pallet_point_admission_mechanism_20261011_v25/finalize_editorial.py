"""Record source review caveats and actual reader/figure evidence in the final report."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,r):
    with p.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    replacements=[
        ('아래는13점 전체다. remaining-GOOD는 **native N3 오차**로 계산했다.',
         '아래는13점 전체다. remaining-GOOD는 H∪{k}를 뺀 finite/sentinel-valid native pool에서 **native N3 오차**로 계산했다. 이 pool 집계에는 영상 범위 검사가 없으므로 실제 LOO eligibility의 보증으로 읽지 않는다.'),
        ('remaining GOOD 수','remaining finite GOOD 수'),
        ('BAD13 중6점은 LOO에서 남은 known-GOOD가4개 이상이었다. 그6점의 충분한 점 수만으로 정확도나 다중해 해소를 보장하지 않았다.',
         'BAD13 중6점은 이 finite native pool의 known-GOOD가4개 이상이었다. 실제 in-image eligible 점 수나 정확도·다중해 해소를 그 수만으로 보장하지 않았다.'),
    ]
    source_image='''\n![source 4개 전체의 저장 경계 support와 코너 확대](source_plot/ALL4_SOURCE_BOUNDARY_AND_CORNER.png)\n\n위 행은 각 두 선의 retained support이고 아래 행은 같은 교점 주변70px 확대다. 점은 예측 query, +는 certified source POS target, 빨간×는 예측 교점, 초록별은 저장 wire 참조, 보라원은 calibrated 반경이다. 앞3개는 기존 cap에서 이미 제외되고 마지막은 admission만 통과한다. 거의 평행한 앞3개와 거의 직교한 마지막 사례를 구분할 수 있다. 물리 자세 GT나 source N3/H/LOO 성공을 나타내는 그림이 아니다. 새 line fit은0이다.\n'''
    real_replay='''\n원 real saved join도 정책·함수 본문을 바꾸지 않고 ROOT/G/P/SELECTED 경로만 바꿔 실제 실행했다. frozen geometry 복사와 이전 Git blob을 사용한 명시적 fixture이며 fresh clone은 아니다. 전체245 join 결과의133행 gzip·SUMMARY·CHECKS는 원 실행과 바이트 단위로 같았다. [실제 real replay 검산](PUBLIC_REAL_REPLAY_CHECKS.json)과 [fixture provenance](PUBLIC_REAL_FIXTURE_PROTOCOL.json)에 남겼다. 위 V21 archive를 복원한 checkout에서는 다음 명령이 같은 입력SHA/bytes와 출력 동일성을 검사한다.\n\n```bash\npython3 -I -S -B scripts/research/pallet_point_admission_mechanism_20261011_v25/replay_public_real.py --checkout CHECKOUT --output NEW_OUTPUT\n```\n'''
    marker='**uncertainty는 어디서 쓰이고 어디서 빠지는가:**'
    end='전용 브랜치 정상commit/push·원격SHA 일치·main/사용자변경 보존은 게시 receipt에서 확인한다.'
    hashes={}
    for name in ('RESULT_KO.md','build_report.py'):
        p=ROOT/name;text=p.read_text();hashes[name]=dict(before=sha(p))
        for before,after in replacements:
            assert text.count(before)==1,(name,before);text=text.replace(before,after)
        assert text.count(marker)==1;text=text.replace(marker,source_image+'\n'+marker)
        assert text.count(end)==1;text=text.replace(end,end+'\n'+real_replay)
        p.write_text(text);hashes[name]['after']=sha(p)
    p=ROOT/'SOURCE_REVIEW.json';r=json.loads(p.read_text())
    r['findings'].append('Real native remaining pool uses finite/sentinel only; in-image LOO eligibility is not certified by remaining_GOOD_ge4.')
    r['reviewed_report_causal_wording_narrowed']=True
    p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    p=ROOT/'ROOT_VISUAL_REVIEW.json';r=json.loads(p.read_text())
    source=json.loads((ROOT/'source_plot/CHECKS.json').read_text())
    r.update(source_image_actual_ROOT_viewed=True,source_image_sha256=source['image_sha256'],
        all4_source_cases_overview_and_zoom_readable=True,source3_rejected_vs1_admitted_UNKNOWN_clearly_labeled=True)
    p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    shutil.copytree(ROOT/'source_plot',ROOT/'prepared/source_plot')
    checked=json.loads(subprocess.check_output(['/home/minjae/anaconda3/bin/python3','-I','-S','-B',
        str(ROOT/'check_public.py'),str(ROOT/'prepared')]))
    assert checked['passed'];save(ROOT/'ROOT_FINAL_DERIVED_CHECKS.json',checked)
    save(ROOT/'REPORT_EDITORIAL_CHECKS.json',dict(passed=True,report_hashes=hashes,
        causal_contribution_not_claimed_proven=True,finite_pool_not_actual_eligibility=True,
        source_and_real_replays_actual_completed=True,actual_all4_source_figure_viewed=True,
        data_changed=False,numerical_method_changed=False,new_PnP=0,new_training=0))
    print(json.dumps(dict(passed=True,report_lines=len((ROOT/'RESULT_KO.md').read_text().splitlines()),checks=checked)))


if __name__=='__main__':main()
