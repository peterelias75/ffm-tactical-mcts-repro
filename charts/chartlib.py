from __future__ import annotations
import json
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "paper_reproduction" / "runs"
if not RUNS.exists():
    RUNS = ROOT / "experiments" / "runs"
OUT_ROOT = ROOT / "charts"
COLORS = ["#2563EB", "#14B8A6", "#F59E0B", "#8B5CF6", "#EF4444"]

def load(path):
    return json.loads(Path(path).read_text())

def style():
    plt.rcParams.update({"figure.facecolor":"#F8FAFC","axes.facecolor":"#FFFFFF","axes.edgecolor":"#CBD5E1","axes.titleweight":"bold","axes.titlesize":16,"axes.labelcolor":"#334155","xtick.color":"#475569","ytick.color":"#475569","font.size":10,"grid.color":"#E2E8F0","grid.linewidth":0.8})

def finish(fig, ax, out, source, interim=False):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=.85, zorder=0)
    note = f"Source: {source}"
    if interim: note += "  •  INTERIM — incomplete run"
    fig.text(.01,.01,note,fontsize=7,color="#64748B")
    fig.tight_layout(rect=(0,.035,1,1))
    fig.savefig(out,dpi=200,bbox_inches="tight")
    plt.close(fig)

def labels(ax, bars, fmt="{:.2f}"):
    for b in bars:
        v=b.get_height(); ax.text(b.get_x()+b.get_width()/2,v,fmt.format(v),ha="center",va="bottom" if v>=0 else "top",fontsize=9,fontweight="bold",color="#334155")

def grouped(title, ylabel, names, series, out, source, interim=False):
    fig,ax=plt.subplots(figsize=(9,5.2)); x=np.arange(len(names)); width=.8/len(series)
    for i,(label,vals) in enumerate(series.items()):
        bars=ax.bar(x-.4+width/2+i*width,vals,width,label=label,color=COLORS[i],zorder=3); labels(ax,bars)
    ax.set_title(title,loc="left"); ax.set_ylabel(ylabel); ax.set_xticks(x,names); ax.legend(frameon=False,ncol=min(4,len(series)),loc="upper center",bbox_to_anchor=(.5,1.0)); ax.axhline(0,color="#94A3B8",lw=.8)
    finish(fig,ax,out,source,interim)

def single(title,ylabel,names,vals,out,source,colors=None,interim=False):
    fig,ax=plt.subplots(figsize=(8.5,5)); bars=ax.bar(names,vals,color=colors or COLORS[:len(vals)],width=.62,zorder=3); labels(ax,bars); ax.set_title(title,loc="left"); ax.set_ylabel(ylabel); ax.axhline(0,color="#94A3B8",lw=.9); finish(fig,ax,out,source,interim)

def summary(path): return load(path)["summary"]

def build(key):
    style(); out=OUT_ROOT/key/"chart.png"
    if key=="E01_forward_model":
        d=load(RUNS/"e01_historical_ffm/results.json")["summary"]
        single("E01 — Forward-model fidelity","MAE / MAD (native metric units)",["Possession\n(pp)","Passes","Shots","Dribbles"],[d["possession_mae_pp"],d["passes_mae"],d["shots_mae"],d["dribbles_mae"]],out,"e01_historical_ffm/results.json")
    elif key=="E02_token_augmentation":
        v=load(RUNS/"e02_token_augmentation/results.json")["variants"]; order=["no_aug","untyped_aug","typed_aug"]
        grouped("E02 — Effect of intervention-token augmentation","Mean absolute deviation (lower is better)",[v[k]["paper_label"] for k in order],{"Possession (pp)":[v[k]["possession_mad_pct"] for k in order],"Shots":[v[k]["shots_mad"]["avg"] for k in order],"Dribbles":[v[k]["dribbles_mad"]["avg"] for k in order],"Events":[v[k]["total_events_mad"] for k in order]},out,"e02_token_augmentation/results.json")
    elif key=="E03_spatial_grid":
        v=load(RUNS/"e03_spatial_grid/results.json")["variants"]; order=["coarse_4x6","regular_8x12","fine_16x24"]
        grouped("E03 — Spatial grid resolution","MAD (lower is better)",[v[k]["paper_label"] for k in order],{"Possession (pp)":[v[k]["possession_mad_pct"] for k in order],"Shots":[v[k]["shots_mad"]["avg"] for k in order],"Events":[v[k]["total_events_mad"] for k in order]},out,"e03_spatial_grid/results.json")
    elif key=="E04_repeat_stability":
        d=load(RUNS/"e04_repeat_stability/results.json")["summary"]
        single("E04 — Repeat stability across stochastic rollouts","Reported value",["Within-match\nvariance (pp²)","Absolute\nerror (pp)"],[d["mean_within_match_sample_variance_pp2"],d["mean_absolute_error_pp"]],out,"e04_repeat_stability/results.json")
    elif key=="E05_policy_comparison":
        paths={"Random":RUNS/"e05_controlled_200/random/results.json","Classic Markov":RUNS/"e05_controlled_200/classic_markov/results.json","Plain MCTS":RUNS/"e05_capped_200/plain_mcts/results.json","Priors MCTS":RUNS/"e05_capped_200/priors_mcts/results.json"}
        vals=[]
        for p in paths.values(): vals.append(summary(p)["mean_delta_pp"])
        single("E05 — Tactical policy comparison","Mean paired possession gain (pp)",list(paths),vals,out,"controlled/capped E05 result JSONs")
    elif key=="E06_intervention_frequency":
        ps=[RUNS/f"E06_plain_mcts_n50/step_minutes_{x:02d}/results.json" for x in (1,5,10)]
        single("E06 — Decision frequency","Mean paired possession gain (pp)",["Every 1 min","Every 5 min","Every 10 min"],[summary(p)["mean_delta_pp"] for p in ps],out,"E06_plain_mcts_n50")
    elif key=="E07_search_budget":
        ns=[1,5,10,30,50]; ps=[RUNS/f"E07_plain_mcts_step1_n50/num_simulations_{n:02d}/results.json" for n in ns]
        single("E07 — MCTS simulation budget","Mean paired possession gain (pp)",[str(n) for n in ns],[summary(p)["mean_delta_pp"] for p in ps],out,"E07_plain_mcts_step1_n50")
    elif key=="E08_lookahead":
        ns=[0,40,200,400]; ps=[RUNS/f"E08_plain_mcts_step1_sim5_n50/lookahead_tokens_{n:03d}/results.json" for n in ns]
        single("E08 — Candidate rollout lookahead","Mean paired possession gain (pp)",[str(n) for n in ns],[summary(p)["mean_delta_pp"] for p in ps],out,"E08_plain_mcts_step1_sim5_n50")
    elif key=="Figure7_match_context":
        rows=[]
        for i in (0,1): rows+=load(RUNS/f"Figure7_plain_mcts_step1_sim5_lookahead200_n50/shard{i}.json")["replays"]
        groups={"<45%":[],"45–55%":[],">55%":[]}
        for r in rows: groups["<45%" if r["real_home_possession"]<.45 else ("45–55%" if r["real_home_possession"]<.55 else ">55%")].append(r["coach_vs_real_pp"])
        single("Figure 7 — Policy value by match context","Coach minus real possession (pp)",list(groups),[sum(v)/len(v) for v in groups.values()],out,"Figure7 replay shards")
    elif key=="Figure8_objectives":
        names=["Shots-only","Balanced"]; rows=[load(RUNS/"Figure8_plain_mcts_step1_sim5_lookahead200_n50/shots_only/results.json")["matches"],load(RUNS/"Figure8_plain_mcts_step1_sim5_lookahead200_n50/balanced/results.json")["matches"]]
        grouped("Figure 8 — Objective trade-off","Average gain vs paired no-action",names,{"Shots (count)":[sum(r["shot_delta_vs_noop"] for r in x)/len(x) for x in rows],"Possession (pp)":[sum(r["delta_pp"] for r in x)/len(x) for x in rows]},out,"Figure8 n50 result JSONs")
    elif key=="E10_intervention_cooldown":
        p1=RUNS/"E08_plain_mcts_step1_sim5_n50/lookahead_tokens_200/results.json"; ps=[p1,RUNS/"E10_plain_mcts_intervention_cooldown_n50/cooldown_5/results.json",RUNS/"E10_plain_mcts_intervention_cooldown_n50/cooldown_10/results.json"]
        ss=[summary(p) for p in ps]; interim=not all(s.get("complete",False) for s in ss)
        names=[f"1 min\n(n={ss[0]['num_matches']})",f"5 min\n(n={ss[1]['num_matches']})",f"10 min\n(n={ss[2]['num_matches']})"]
        single("E10 — Recovery time after an active intervention","Mean paired possession gain (pp)",names,[s["mean_delta_pp"] for s in ss],out,"E08 baseline + E10 cooldown results",interim=interim)
    else: raise KeyError(key)
    print(out)
