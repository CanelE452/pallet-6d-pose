"""Public numeric charts, plus optional private RGB case sheets."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from . import common as C
from . import verdict as V

METHODS = C.METHODS
COLORS = {"BASE": "#586b7b", "N3_DIM_SYM": "#2878a8", "SUBPIX": "#ba792d", "N3_THEN_SUBPIX": "#7a5195"}
METRICS = (("T_cm", "위치 오차 (cm)", 1), ("R_deg", "회전 오차 (°)", 1), ("ADDsym_m", "ADDsym (cm)", 100))
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


def read(path):
    return json.loads(Path(path).read_text())


def configure():
    fonts = (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
             Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"))
    for path in fonts:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white"})


def load_rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def dataset(doc):
    real = (doc / "VERDICT.json").exists()
    prefix = "" if real else "SYNTH_"
    metrics = read(doc / (prefix + "METRICS.json"))
    paired = read(doc / (prefix + "PAIRED.json"))
    failures = read(doc / (prefix + "FAILURES.json"))
    verdict = read(doc / (prefix + "VERDICT.json"))
    if real:
        all_rows = C.historical_rows()
        vis_rows = load_rows(doc / "PREDICTIONS.jsonl.gz")
    else:
        all_rows = load_rows(doc / "SYNTH_ALL.jsonl.gz")
        vis_rows = load_rows(doc / "SYNTH_PREDICTIONS.jsonl.gz")
    return real, prefix, metrics, paired, failures, verdict, all_rows, vis_rows


def save(fig, path):
    fig.savefig(path, dpi=145, bbox_inches="tight", pil_kwargs={"compress_level": 9})
    plt.close(fig)


def box(ax, x, y, text, color="#e5eff7", width=2.65, height=.9):
    ax.add_patch(FancyBboxPatch((x,y), width,height, boxstyle="round,pad=.1", facecolor=color, edgecolor="#536579"))
    ax.text(x+width/2,y+height/2,text,ha="center",va="center",fontsize=11)


def method_flow(path):
    fig, ax = plt.subplots(figsize=(13.8,6.5));ax.set_xlim(0,14);ax.set_ylim(0,6);ax.axis("off")
    box(ax,.35,4.25,"각 경로의 최종 2D 코너\nqFinal + 원래 지원 마스크")
    box(ax,3.8,4.25,"6면의 부호 넓이\n넓이 > 0: camera-facing")
    box(ax,7.25,4.25,"향한 면의 코너 유지\nunknown 면이면\n보수적으로 유지")
    box(ax,10.7,4.25,"지원 코너 < 4개이면\n원래 전체 대응점으로 fallback")
    for x in (3.1,6.55,10):ax.annotate("",xy=(x+.6,4.7),xytext=(x,4.7),arrowprops={"arrowstyle":"->","lw":1.5})
    box(ax,.35,2.45,"ALL: 원래 8점 fit\nVIS: 유지할 코너만 fit", "#e5f3e8",width=3.3)
    box(ax,4.3,2.45,"동일 SQPnP → RefineLM\nK·3D 순서·코너 좌표 보존", "#e5f3e8",width=3.3)
    box(ax,8.25,2.45,"원래 9점 재투영 점수 유지\n동일 W/D 가설 선택 → 최종 자세", "#e5f3e8",width=4.45)
    for x in (3.7,7.65):ax.annotate("",xy=(x+.5,2.9),xytext=(x,2.9),arrowprops={"arrowstyle":"->","lw":1.5})
    ax.annotate("",xy=(2,3.55),xytext=(12,4.1),arrowprops={"arrowstyle":"->","lw":1.3,"connectionstyle":"angle,angleA=-90,angleB=0"})
    ax.text(.35,1.5,"추론 마스크를 SHA 봉인한 뒤 평가 참조를 읽음 → A0 parity → A1 합성 harm gate → 통과할 때만 A2 실사",fontsize=12)
    ax.text(.35,.6,"GT·사람 가시성은 마스크 입력에 없음. 좌표는 바뀌지 않아 코너 지표도 동일해야 함.\n규정된 면 순서는 기존 3D 축에서 inward winding이며, 그대로의 2D 양수 규칙을 사용한다.",fontsize=11,color="#455365")
    fig.suptitle("VIS_RULE_V1: 기하 마스크는 PnP fit의 대응점에만 적용",fontsize=17,y=1.01)
    save(fig,path)


def accuracy(path, m, phase):
    scopes=[(str(s),m["by_seed"][str(s)]["ALL"]) for s in C.SEEDS]+[("seed-mean",m["seed_mean"]["ALL"])]
    entries=[(s,name,summary) for s,ss in scopes for base in METHODS for name in (base,base+"_VIS") for summary in [ss[name]]]
    fig,axes=plt.subplots(1,3,figsize=(19,14.3),sharey=True)
    for ax,(key,label,scale) in zip(axes,METRICS):
        for i,(seed,name,summary) in enumerate(entries):
            stat=summary["metrics"][key];y=len(entries)-1-i;mean=stat["mean"]
            if mean is None:ax.text(.02,y,"자세 미산출",transform=ax.get_yaxis_transform());continue
            lo,hi=stat["CI95"] or (mean,mean);color=COLORS[name.removesuffix("_VIS")]
            ax.errorbar(mean*scale,y,xerr=[[max(0,(mean-lo)*scale)],[max(0,(hi-mean)*scale)]],fmt="o",color=color,
                        markerfacecolor=color if name.endswith("_VIS") else "white",capsize=3)
            ax.annotate(f"{mean*scale:.2f} ± {stat['std']*scale:.2f}; n={stat['n']}",(mean*scale,y),xytext=(9,5),
                        textcoords="offset points",va="bottom",fontsize=8.5,
                        bbox={"facecolor":"white","edgecolor":"none","alpha":.88,"pad":.2})
        ax.set_xlabel(label+"; 막대=평균의 95% CI");ax.grid(axis="x",alpha=.22);ax.margins(x=.48)
    axes[0].set_yticks(range(len(entries)),[f"{s} / {name}" for s,name,_ in reversed(entries)],fontsize=9)
    for ax in axes:
        for y in (7.5,15.5,23.5):ax.axhline(y,color="#bdc5cd",lw=.8)
    fig.suptitle(f"{phase}: 4경로 ALL/VIS × 3 seed 및 영상별 seed-mean",fontsize=17,y=.995)
    fig.text(.13,.02,"빈 점=ALL, 채운 점=VIS. 숫자는 평균±표본 SD이며 CI와 구분한다. 모든 성공·미산출 수는 표와 JSON에 보존한다.",fontsize=11)
    fig.tight_layout(rect=(0,.045,1,.975));save(fig,path)


def primary_ci(path, m, p, phase):
    fig,axes=plt.subplots(1,2,figsize=(13.8,5.5),sharey=True)
    scopes=[(str(s),p["by_seed"][str(s)]["ALL"][V.PRIMARY_CONTRAST]) for s in C.SEEDS]+[("seed-mean",p["primary"])]
    for ax,key,title in zip(axes,("confusion_rate","success_rate"),("가로·깊이 혼동률: 감소가 개선","5 cm·5° 성공률: 증가가 개선")):
        for y,(seed,r) in enumerate(reversed(scopes)):
            stat=r[key];d=stat["delta"];interval=stat["CI95"]
            if d is None or interval is None:ax.text(0,y,"미산출: 판정 불가");continue
            lo,hi=interval;ax.errorbar(d*100,y,xerr=[[(d-lo)*100],[(hi-d)*100]],fmt="o",color="#7a5195",capsize=4)
            ax.annotate(f"{d*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}]",(d*100,y),xytext=(8,10),textcoords="offset points",fontsize=9)
            secondary=stat.get("scenario_cluster_secondary")
            if seed=="seed-mean" and secondary and secondary.get("CI95"):
                a,b=secondary["CI95"];ax.plot([a*100,b*100],[y-.13,y-.13],color="#708090",lw=3,alpha=.65)
        ax.axvline(0,color="black",lw=1);ax.set_title(title);ax.set_xlabel("직렬 VIS−ALL (%p); 막대=95% CI");ax.grid(axis="x",alpha=.2);ax.margins(x=.3)
    axes[0].set_yticks(range(4),[s for s,_ in reversed(scopes)])
    fig.suptitle(f"{phase} 사전등록 주비교: N3_THEN_SUBPIX_VIS − N3_THEN_SUBPIX",fontsize=15)
    footer=("실사 CI: 13세션 paired cluster bootstrap 10,000회, seed 20260917; 동일 고정 추출을 유지한다."
            if phase=="A2 실사" else "회색 추가 막대는 scenario cluster 보조 CI이며 gate는 고정 frame bootstrap을 따른다.")
    fig.text(.07,.015,"혼동: R>45° 및 |yaw|≥60°. 성공: T<5 cm 및 R<5°. seed별 이진 판정 후 동일 ID에서 평균.\n"+footer,fontsize=10)
    fig.tight_layout(rect=(0,.095,1,.95));save(fig,path)


def subgroup(path,m,phase):
    groups=[scope for scope in m["seed_mean"] if scope!="ALL"]
    if not groups:groups=["ALL"]
    fig,axes=plt.subplots(1,3,figsize=(17,6.2))
    for ax,(key,label,scale) in zip(axes,METRICS):
        for j,base in enumerate(METHODS):
            for suffix,shift,style in (("",-.015,"--"),("_VIS",.015,"-")):
                values=[m["seed_mean"][g][base+suffix]["metrics"][key]["mean"] for g in groups]
                x=np.arange(len(groups))+(j-1.5)*.14+shift
                y=[np.nan if v is None else v*scale for v in values]
                ax.plot(x,y,style,marker="o",ms=5,color=COLORS[base],alpha=.9,label=base+suffix)
        ax.set_xticks(range(len(groups)),groups,rotation=18);ax.set_ylabel(label);ax.grid(axis="y",alpha=.2)
    axes[1].legend(fontsize=8,ncol=2,loc="upper center",bbox_to_anchor=(.5,1.32))
    fig.suptitle(f"{phase} 모든 고정 하위집단: seed-mean 평균 오차",fontsize=16,y=1.02)
    fig.text(.06,.005,"하위집단은 서로 겹칠 수 있다. 부정 결과를 포함하며 범주별 분모·SD·CI는 METRICS/보고서에 공개한다.",fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.9));save(fig,path)


def diagnostics(path,failures,m,phase):
    fig,axes=plt.subplots(2,2,figsize=(14.5,8.5));entries=[(str(s),base,failures["per_seed"][str(s)][base]) for s in C.SEEDS for base in METHODS]
    x=np.arange(len(entries));labels=[f"s{s}\n{b.replace('N3_THEN_SUBPIX','직렬').replace('N3_DIM_SYM','N3')}" for s,b,_ in entries]
    hidden_keys=sorted({int(k) for _,_,r in entries for k in r["hidden_count_distribution"]});bottom=np.zeros(len(entries))
    for i,k in enumerate(hidden_keys):
        y=np.array([r["hidden_count_distribution"].get(str(k),0) for _,_,r in entries]);axes[0,0].bar(x,y,bottom=bottom,label=f"숨김 {k}개",alpha=.85);bottom+=y
    axes[0,0].legend(ncol=3,fontsize=9);axes[0,0].set_title("기하 숨김 코너 수별 영상 수")
    axes[0,1].bar(x-.18,[r["success_to_failure"] for _,_,r in entries],width=.36,color="#bd4c4c",label="ALL 성공→VIS 실패")
    axes[0,1].bar(x+.18,[r["failure_to_success"] for _,_,r in entries],width=.36,color="#438d69",label="ALL 실패→VIS 성공");axes[0,1].legend(fontsize=9);axes[0,1].set_title("영상 단위 5 cm·5° 손상 / 회복")
    axes[1,0].bar(x,[r["hypothesis_changes"] for _,_,r in entries],color=[COLORS[b] for _,b,_ in entries]);axes[1,0].set_title("원래 선택기의 W/D 가설 전환 수")
    axes[1,1].bar(x-.18,[r["fallback_lt4"] for _,_,r in entries],width=.36,color="#ba792d",label="남은 지원점<4 fallback")
    axes[1,1].bar(x+.18,[len(r["unavailable_VIS_ids"]) for _,_,r in entries],width=.36,color="#586b7b",label="VIS 자세 미산출");axes[1,1].legend(fontsize=9);axes[1,1].set_title("fallback과 후단 자세 미산출을 구분")
    for ax in axes.flat:ax.set_xticks(x,labels,fontsize=7.6);ax.set_ylabel("영상 수");ax.grid(axis="y",alpha=.2);ax.set_ylim(bottom=0)
    if all(r["fallback_lt4"]==0 and not r["unavailable_VIS_ids"] for _,_,r in entries):
        axes[1,1].set_ylim(0,1);axes[1,1].set_yticks([0,1])
        axes[1,1].text(.5,.42,"모든 seed·경로에서 0",ha="center",transform=axes[1,1].transAxes,fontsize=13)
    fig.suptitle(f"{phase}: 모든 seed·경로의 숨김, 손상, 전환, 미산출",fontsize=17)
    fig.text(.08,.007,"각 막대는 해당 seed의 같은 전체 영상 분모다. seed별 반복 결과를 독립 영상 수로 합치지 않는다.",fontsize=10)
    fig.tight_layout(rect=(0,.03,1,.95));save(fig,path)


def select_cases(all_rows,vis_rows):
    baseline={r["id"]:r for r in all_rows if int(r["seed"])==1 and r["method"]==V.PRIMARY_METHOD}
    changed={r["id"]:r for r in vis_rows if int(r["seed"])==1 and r["method"]==V.PRIMARY_METHOD+"_VIS"}
    pairs=[]
    for identifier in sorted(baseline):
        b,v=baseline[identifier],changed[identifier]
        if not (b["pose"].get("available") and v["pose"].get("available")):continue
        d=v["pose"]["translation_cm"]-b["pose"]["translation_cm"]
        pairs.append((identifier,b,v,d))
    recovery=[r for r in pairs if not V.indicators(r[1]["pose"])["success_rate"] and V.indicators(r[2]["pose"])["success_rate"]]
    damage=[r for r in pairs if V.indicators(r[1]["pose"])["success_rate"] and not V.indicators(r[2]["pose"])["success_rate"]]
    def choose(pool,sign):return min(pool,key=lambda r:(sign*r[3],r[0])) if pool else None
    return [("성공 회복 중 최대 T 개선",choose(recovery,1)),("성공 손상 중 최대 T 악화",choose(damage,-1)),
            ("전체 최대 T 개선",choose(pairs,1)),("전체 최대 T 악화",choose(pairs,-1))]


def pose_projection(row):
    """Project a saved actual pose, without solving F or using reference pose."""
    actual=row["actual_pose"]
    if not actual.get("available"):return None
    w,h,d=np.asarray(actual["cf_extents"],float)/2
    model=np.array([[-w,-h,-d],[w,-h,-d],[w,h,-d],[-w,h,-d],
                    [-w,-h,d],[w,-h,d],[w,h,d],[-w,h,d]],float)
    camera=model@np.asarray(actual["R_cf"],float).T+np.asarray(actual["centroid"],float)
    homogeneous=camera@np.asarray(row["fixed_metadata"]["K"],float).T
    with np.errstate(divide="ignore",invalid="ignore"):
        return homogeneous[:,:2]/homogeneous[:,2:3]


def draw_pose_edges(ax,row,scale,style,color,label):
    projected=pose_projection(row)
    if projected is None:return
    projected=projected/scale
    for j,(a,z) in enumerate(EDGES):
        ax.plot(projected[[a,z],0],projected[[a,z],1],style,color=color,lw=1.15,
                alpha=.8,label=label if j==0 else None)


def cases(path,all_rows,vis_rows,phase):
    fig,axes=plt.subplots(2,2,figsize=(13,11.5));records=[]
    for ax,(rule,pair) in zip(axes.flat,select_cases(all_rows,vis_rows)):
        if pair is None:ax.text(.5,.5,f"{rule}\n해당 사례 없음",ha="center",va="center",transform=ax.transAxes);ax.axis("off");continue
        identifier,b,v,delta=pair;q=np.array(v["qFinal"],float);h,w=v["raw_hw"];normalized=q/np.array([w,h]);mask=np.array(v["visibility"]["effective_mask"],bool)
        assert np.array_equal(q,np.asarray(b["qFinal"],float)),"Mask experiment must preserve coordinates"
        for a,z in EDGES:ax.plot(normalized[[a,z],0],normalized[[a,z],1],color="#bdc5cd",lw=1)
        draw_pose_edges(ax,b,np.array([w,h]),"--","#586b7b","ALL 자세 재투영")
        draw_pose_edges(ax,v,np.array([w,h]),"-","#ba792d","VIS 자세 재투영")
        ax.scatter(normalized[:8][mask,0],normalized[:8][mask,1],s=65,color="#287e5d",label="PnP fit 유지")
        ax.scatter(normalized[:8][~mask,0],normalized[:8][~mask,1],s=90,facecolors="none",edgecolors="#bd4c4c",lw=2,label="PnP fit 제외")
        for k,point in enumerate(normalized[:8]):ax.annotate(str(k),point,xytext=(5,4),textcoords="offset points",fontsize=10)
        xy=normalized[np.isfinite(normalized).all(1)];margin=max(float(np.ptp(xy,axis=0).max())*.15,.025)
        ax.set_xlim(xy[:,0].min()-margin,xy[:,0].max()+margin);ax.set_ylim(xy[:,1].max()+margin,xy[:,1].min()-margin);ax.set_aspect("equal")
        dR=v["pose"]["rotation_deg"]-b["pose"]["rotation_deg"]
        ax.set_title(f"{rule}\n{identifier} / seed1\nALL T/R={b['pose']['translation_cm']:.2f} cm/{b['pose']['rotation_deg']:.2f}° → "
                     f"VIS {v['pose']['translation_cm']:.2f} cm/{v['pose']['rotation_deg']:.2f}°\n"
                     f"ΔT={delta:+.3f} cm, ΔR={dR:+.3f}°",fontsize=10)
        ax.set_xlabel("x / raw width");ax.set_ylabel("y / raw height");ax.legend(fontsize=8,loc="best");ax.grid(alpha=.15)
        records.append(dict(rule=rule,id=identifier,seed=1,delta_T_cm=delta,delta_R_deg=dR,
                            ALL_T_cm=b["pose"]["translation_cm"],ALL_R_deg=b["pose"]["rotation_deg"],
                            VIS_T_cm=v["pose"]["translation_cm"],VIS_R_deg=v["pose"]["rotation_deg"],
                            normalized_points=normalized.tolist(),effective_mask=mask.tolist()))
    fig.suptitle(f"{phase}: 사전등록 규칙의 실제 코너·마스크 사례 (원본 RGB 없음)",fontsize=15)
    fig.text(.08,.01,"ALL/VIS의 입력 코너 좌표는 완전히 같다. 점은 fit 유지·제외, 선은 저장된 실제 자세의 재투영이다.\n표시 범위는 입력 코너에 맞춘다. 최대 악화 사례도 남기며 미산출 ID는 FAILURES에 유지한다.",fontsize=10)
    fig.tight_layout(rect=(0,.065,1,.94));fig.subplots_adjust(hspace=.65);save(fig,path);return records


def private_rgb_cases(all_rows,vis_rows,real,private_dir):
    """Read exactly the preselected images; never write RGB into public DOC."""
    import cv2
    private_dir=Path(private_dir).resolve()
    assert not private_dir.is_relative_to(C.ROOT),"RGB sheets must remain outside the public repository"
    private_dir.mkdir(parents=True,exist_ok=True)
    if real:
        bindings={r["id"]:r["image"] for r in read(C.OLD_FINAL/"INPUT_AUDIT.json")["inputs"]}
    else:
        manifest=read(C.SOURCE/"data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json")
        bindings={r["id"]:dict(path=r["image"],sha256=r["image_sha256"],pad=r["reflect_pad_px"])
                  for r in manifest["records"] if r["partition"]=="heldout"}
    fig,axes=plt.subplots(2,2,figsize=(14.5,11));receipt=[]
    for ax,(rule,pair) in zip(axes.flat,select_cases(all_rows,vis_rows)):
        if pair is None:
            ax.text(.5,.5,f"{rule}\n해당 사례 없음",ha="center",va="center",transform=ax.transAxes);ax.axis("off");continue
        identifier,b,v,delta=pair;binding=bindings[identifier];image_path=Path(binding["path"])
        if not image_path.is_absolute():image_path=C.SOURCE/image_path
        assert C.sha(image_path)==binding["sha256"]
        rgb=cv2.imread(str(image_path),cv2.IMREAD_COLOR);assert rgb is not None
        if not real:
            assert binding["pad"]==100
            rgb=rgb[100:-100,100:-100]
        h,w=v["raw_hw"];assert list(rgb.shape[:2])==[h,w]
        ax.imshow(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB));q=np.asarray(v["qFinal"],float);mask=np.asarray(v["visibility"]["effective_mask"],bool)
        assert np.array_equal(q,np.asarray(b["qFinal"],float))
        draw_pose_edges(ax,b,np.ones(2),"--","#13bcec","ALL 자세 재투영")
        draw_pose_edges(ax,v,np.ones(2),"-","#ffce45","VIS 자세 재투영")
        ax.scatter(q[:8][mask,0],q[:8][mask,1],s=48,color="#33e39b",edgecolors="black",lw=.6,label="fit 유지")
        ax.scatter(q[:8][~mask,0],q[:8][~mask,1],s=95,marker="X",color="#fa5353",edgecolors="black",lw=.6,label="fit 제외")
        for k,point in enumerate(q[:8]):ax.annotate(str(k),point,xytext=(5,4),textcoords="offset points",color="white",
                                                   bbox={"facecolor":"black","alpha":.5,"pad":1},fontsize=9)
        ax.set_xlim(-.05*w,1.05*w);ax.set_ylim(1.05*h,-.05*h);ax.set_aspect("equal")
        ax.set_title(f"{rule}: {identifier}\nALL T/R={b['pose']['translation_cm']:.3f} cm/{b['pose']['rotation_deg']:.3f}° → "
                     f"VIS {v['pose']['translation_cm']:.3f} cm/{v['pose']['rotation_deg']:.3f}°",fontsize=10)
        ax.legend(fontsize=8,loc="lower left");ax.set_xlabel("raw x (pixel)");ax.set_ylabel("raw y (pixel)")
        receipt.append(dict(rule=rule,id=identifier,seed=1,image_sha256=binding["sha256"],
                            raw_hw=[h,w],crop_reflect_pad_px=0 if real else 100,
                            delta_T_cm=delta,delta_R_deg=v["pose"]["rotation_deg"]-b["pose"]["rotation_deg"]))
        assert C.sha(image_path)==binding["sha256"]
    basename="cases_A2_REAL.png" if real else "cases_A1_SYNTH.png"
    fig.suptitle("비공개 원본 RGB 검토: 사전등록 seed 1 사례 / 동일 코너와 실제 ALL·VIS 자세",fontsize=16)
    fig.tight_layout(rect=(0,0,1,.96));save(fig,private_dir/basename)
    C.write(private_dir/(basename.removesuffix(".png")+"_receipt.json"),dict(RGB_publication=False,
        basename=basename,sha256=C.sha(private_dir/basename),cases=receipt,model_calls=0,F_calls=0))
    return dict(basename=basename,cases=len(receipt),publication=False)


def square_counts(path,audit):
    counts=audit["counts"];d=counts["manual_in_frame_count_distribution"];xs=sorted(map(int,d));ys=[d[str(k)] for k in xs]
    fig,ax=plt.subplots(figsize=(11,5.6));bars=ax.bar(xs,ys,color=["#bd4c4c" if k<4 else "#ba792d" if k<6 else "#438d69" for k in xs])
    ax.bar_label(bars,padding=4,fontsize=12);ax.set_xticks(xs);ax.set_xlabel("영상별 화면 내 직접 수동 코너 수 (0..7만)");ax.set_ylabel("영상 수");ax.set_ylim(0,max(ys)*1.24);ax.grid(axis="y",alpha=.15)
    fig.suptitle("GREEN0918: 수동 입력 감사와 참조 절차 충돌",fontsize=17)
    ax.text(.98,.97,"119장 / 600개 화면 내 수동 코너\nB1 ≥4: 118장\n기존 참조 ≥6: 36장\nB1 통과·기존 ≥6 미달: 82장\n새 참조·6D 평가: 0",ha="right",va="top",transform=ax.transAxes,fontsize=12,
            bbox={"boxstyle":"round,pad=.5","facecolor":"white","edgecolor":"#bdc5cd"})
    fig.text(.1,.015,"112개는 B1 적격 영상의 6점까지 산술 부족분 합계이며 추가 주석 요청이 아니다.\n029844는 화면 내 3점으로 B1 부적격. 4점 30장은 기존 LOO에서 남은 3점으로 검산 불가.",fontsize=10)
    fig.tight_layout(rect=(0,.09,1,.94));save(fig,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--doc",type=Path,default=C.DOC)
    parser.add_argument("--private-cases",type=Path,help="Optional local RGB sheets outside the repository")
    args=parser.parse_args();doc=args.doc
    configure();real,prefix,m,p,f,v,all_rows,vis_rows=dataset(doc);phase="A2 실사" if real else "A1 합성"
    target=doc/"figures";target.mkdir(exist_ok=True)
    filenames=["01_method_flow.png","02_all_vis_accuracy.png","03_primary_paired_ci.png","04_subgroups.png","05_hidden_damage_switch.png","06_numeric_cases.png","07_square_manual_counts.png"]
    titles=["VIS 마스크와 기존 F의 적용 범위","모든 경로·seed ALL/VIS 정확도","사전등록 주비교의 짝 95% CI","모든 고정 하위집단","숨김·손상·전환·fallback·미산출","실제 코너 좌표와 fit 마스크 사례","정사각형 수동 코너 분포와 절차 충돌"]
    method_flow(target/filenames[0]);accuracy(target/filenames[1],m,phase);primary_ci(target/filenames[2],m,p,phase)
    subgroup(target/filenames[3],m,phase);diagnostics(target/filenames[4],f,m,phase)
    examples=cases(target/filenames[5],all_rows,vis_rows,phase);square_counts(target/filenames[6],read(doc/"SQUARE_INPUT_AUDIT.json"))
    evidence_names=[prefix+k for k in ("METRICS.json","PAIRED.json","FAILURES.json","VERDICT.json")]+["METHOD_LOCK.json","A0.json","SQUARE_INPUT_AUDIT.json"]
    evidence_names+= ["PREDICTIONS.jsonl.gz"] if real else ["SYNTH_ALL.jsonl.gz","SYNTH_PREDICTIONS.jsonl.gz"]
    sources=[dict(path=name,sha256=C.sha(doc/name)) for name in evidence_names]
    if real:sources.append(dict(path="../pallet_feature_gradient_joint_20261010/PREDICTIONS.jsonl.gz",sha256=C.sha(C.OLD_DOC/"PREDICTIONS.jsonl.gz")))
    C.write(doc/"FIGURE_INDEX.json",dict(schema="pallet_vispnp_square6d_figures_v1",phase=phase,
        figures=[dict(path="figures/"+name,title=title,sha256=C.sha(target/name),bytes=(target/name).stat().st_size,
                      evidence=sources,rights_status="OWN_GENERATED_NUMERIC_CHART; NO_RGB",generated_by="scripts/research/pallet_vispnp_square6d_20261011/figures.py")
                 for name,title in zip(filenames,titles)],examples=examples,
        case_selection="METHOD_LOCK.json seed1 rule; recovery/damage subsets ranked by T change; ties ID ascending; absent subsets labeled",
        RGB_used=False,visual_inspection="PENDING_ROOT_REVIEW"))
    private_receipt=private_rgb_cases(all_rows,vis_rows,real,args.private_cases) if args.private_cases else None
    print(json.dumps(dict(figures=len(filenames),phase=phase,public_RGB_used=False,private_cases=private_receipt),ensure_ascii=False))


if __name__=="__main__":main()
