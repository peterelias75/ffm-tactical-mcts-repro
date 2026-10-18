#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RUNS=ROOT/'paper_reproduction'/'runs'
if not RUNS.exists(): RUNS=ROOT/'experiments'/'runs'
HERE=Path(__file__).resolve().parent
BG='none'; FG='#111111'; BLUE='#63a4d5'; ORANGE='#a65300'; CORAL='#d56548'; TEAL='#258f86'; SLATE='#69758a'; RED='#ff7f7f'
plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':BG,'savefig.facecolor':BG,'text.color':FG,'axes.labelcolor':FG,'axes.edgecolor':'#b8b8b8','xtick.color':FG,'ytick.color':FG,'font.family':'DejaVu Sans','font.size':12,'font.weight':'bold','axes.labelweight':'bold','axes.titlesize':18,'axes.titleweight':'bold','grid.color':'#4b4b4f','grid.linestyle':'--','grid.alpha':.6})
def load(p): return json.loads(Path(p).read_text())
def clean(ax):
 ax.spines[['top','right']].set_visible(False); ax.set_axisbelow(True); ax.grid(True,axis='both',zorder=0); ax.axhline(0,color='#aaa',lw=.8)
def save(fig,name):
 fig.set_size_inches(8.5,6,forward=True); fig.tight_layout(pad=1.8); fig.savefig(HERE/name,dpi=220,facecolor='none',transparent=True); plt.close(fig); print(HERE/name)
def barlabels(ax,bars,fmt='{:.2f}',suffix='',stars=()):
 for i,b in enumerate(bars):
  v=b.get_height(); ax.text(b.get_x()+b.get_width()/2,v,fmt.format(v)+suffix+(' ★' if i in stars else ''),ha='center',va='bottom' if v>=0 else 'top',color=FG,fontweight='bold')

rows=[]
for i in (0,1): rows+=load(RUNS/f'Figure7_plain_mcts_step1_sim5_lookahead200_n50/shard{i}.json')['replays']
bucket=lambda x:'<45%' if x<.45 else ('45–55%' if x<.55 else '>55%')
g=defaultdict(list); mix=defaultdict(Counter)
for r in rows:
 g[bucket(r['real_home_possession'])].append(r['coach_vs_real_pp'])
 for d in r['decisions']: mix[bucket(d['simulated_possession_before_decision'])][d['intent'].replace('_',' ').title()]+=1
names=['<45%','45–55%','>55%']; vals=[sum(g[n])/len(g[n]) for n in names]
fig,a=plt.subplots(figsize=(8.5,6)); a.plot(names,vals,color=ORANGE,marker='o',lw=3,ms=8)
for i,v in enumerate(vals): a.text(i,v+.5,f'{v:+.2f}'+(' ★' if v==max(vals) else ''),ha='center')
a.set_title('Figure 7a — Possession Gain by Match Context',pad=14); a.set_ylabel('Simulated coach advantage over real match (pp)'); a.set_xlabel('Real second-half home possession'); clean(a); save(fig,'figure_7a.png')
fig,b=plt.subplots(figsize=(8.5,6)); cats=['Attacking','Defending','Neutral','No Action']; cols=[BLUE,ORANGE,TEAL,RED]; x=np.arange(3); w=.2
for j,cat in enumerate(cats): b.bar(x+(j-1.5)*w,[mix[n][cat]/sum(mix[n].values()) for n in names],w,label=cat,color=cols[j],zorder=3)
b.set_xticks(x,names); b.set_ylim(0,1); b.set_ylabel('Share of decisions'); b.set_xlabel('Simulated home possession at decision'); b.set_title('Figure 7b — Intervention Choice by Simulated Possession',pad=14); b.legend(frameon=False); clean(b); save(fig,'figure_7b.png')
