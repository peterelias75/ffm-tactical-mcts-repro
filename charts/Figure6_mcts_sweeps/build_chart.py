#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RUNS=ROOT/'paper_reproduction'/'runs'
if not RUNS.exists(): RUNS=ROOT/'experiments'/'runs'
BG='none'; FG='#111111'; GRID='#4b4b4f'; BLUE='#63a4d5'; RED='#ff7f7f'
plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':BG,'savefig.facecolor':BG,'text.color':FG,'axes.labelcolor':FG,'axes.edgecolor':'#b8b8b8','xtick.color':FG,'ytick.color':FG,'font.family':'DejaVu Sans','font.size':12,'font.weight':'bold','axes.labelweight':'bold','axes.titlesize':18,'axes.titleweight':'bold','grid.color':GRID,'grid.linestyle':'--','grid.alpha':.55})
def summary(path): return json.loads(path.read_text())['summary']
def clean(ax):
 ax.spines[['top','right']].set_visible(False); ax.set_axisbelow(True); ax.grid(True,axis='both',zorder=0); ax.axhline(0,color='#999',lw=.7)
def label_bars(ax,bars):
 best=max(b.get_height() for b in bars)
 for b in bars:
  v=b.get_height(); ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.0f}%' + (' ★' if v==best else ''),ha='center',color=FG,fontweight='bold')
# E06
intervals=[1,5,10]
e06=[summary(RUNS/f'E06_plain_mcts_n50/step_minutes_{n:02d}/results.json') for n in intervals]
# E07
sims=[1,5,10,30,50]
e07=[summary(RUNS/f'E07_plain_mcts_step1_n50/num_simulations_{n:02d}/results.json') for n in sims]
# E08
lookahead=[0,40,200,400]
e08=[summary(RUNS/f'E08_plain_mcts_step1_sim5_n50/lookahead_tokens_{n:03d}/results.json') for n in lookahead]
# Figure 6a — intervention interval
fig,a=plt.subplots(figsize=(8.5,6))
bars=a.bar([f'{n} min' for n in intervals],[s['pct_matches_improved'] for s in e06],color=BLUE,width=.45,zorder=3)
label_bars(a,bars); a.set_ylim(0,100); a.set_ylabel('Matches improved over no-action (%)'); a.set_xlabel('Decision interval (minutes)'); a.set_title('Figure 6a — Match Improvement by Decision Interval',pad=14,fontweight='bold'); clean(a)
fig.tight_layout(); fig.savefig(Path(__file__).with_name('figure_6a.png'),dpi=220,facecolor='none',transparent=True); plt.close(fig)
# Figure 6b — search budget
fig,b=plt.subplots(figsize=(8.5,6))
b.plot(sims,[s['pct_matches_improved'] for s in e07],color=BLUE,marker='o',lw=2.6,label='Improvement rate (%)')
b.set_ylabel('Matches improved (%)',color=BLUE); b.set_xlabel('MCTS simulations per decision'); b.set_xticks(sims); b.set_ylim(0,100); clean(b)
b2=b.twinx(); b2.plot(sims,[s['mean_delta_pp'] for s in e07],color=RED,marker='s',ls=':',lw=2.3,label='Mean possession gain (pp)'); b2.set_ylabel('Mean possession gain (pp)',color=RED); b.set_title('Figure 6b — MCTS Search Budget and Performance',pad=14,fontweight='bold')
lines=b.lines+b2.lines; b.legend(lines,[x.get_label() for x in lines],frameon=False,loc='lower right',fontsize=9)
fig.tight_layout(); fig.savefig(Path(__file__).with_name('figure_6b.png'),dpi=220,facecolor='none',transparent=True); plt.close(fig)
# Figure 6c — rollout lookahead
fig,c=plt.subplots(figsize=(8.5,6))
mat=np.array([[s['pct_matches_improved'] for s in e08],[s['mean_gain_when_improved_pp'] for s in e08]])
rgba=np.zeros((2,4,4)); row0=(mat[0]-mat[0].min())/(mat[0].max()-mat[0].min() or 1); row1=(mat[1]-mat[1].min())/(mat[1].max()-mat[1].min() or 1)
[rgba.__setitem__((0,j),plt.cm.Blues(.25+.7*row0[j])) for j in range(4)]; [rgba.__setitem__((1,j),plt.cm.Oranges(.25+.7*row1[j])) for j in range(4)]
c.imshow(rgba,aspect='auto'); c.set_xticks(range(4),lookahead); c.set_yticks([0,1],['% improved\nmatches','Avg gain when\nimproved (pp)']); c.set_xlabel('Rollout tokens per candidate'); c.set_ylabel('Performance metric'); c.set_title('Figure 6c — Rollout Lookahead and Performance',pad=14,fontweight='bold')
for i in range(2):
 for j in range(4): c.text(j,i,(f'{mat[i,j]:.0f}%' if i==0 else f'+{mat[i,j]:.2f}') + (' ★' if j==int(np.argmax(mat[i])) else ''),ha='center',va='center',color='black',fontweight='bold')
fig.tight_layout(); fig.savefig(Path(__file__).with_name('figure_6c.png'),dpi=220,facecolor='none',transparent=True); plt.close(fig)
print(Path(__file__).parent)
