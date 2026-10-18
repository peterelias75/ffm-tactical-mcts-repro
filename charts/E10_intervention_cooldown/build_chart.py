#!/usr/bin/env python3
"""Build two readable single-panel E10 bar charts from complete N=50 runs."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; RUNS=ROOT/'paper_reproduction'/'runs'
if not RUNS.exists(): RUNS=ROOT/'experiments'/'runs'
PATHS=[RUNS/'E08_plain_mcts_step1_sim5_n50/lookahead_tokens_200/results.json',RUNS/'E10_plain_mcts_intervention_cooldown_n50/cooldown_5/results.json',RUNS/'E10_plain_mcts_intervention_cooldown_n50/cooldown_10/results.json']
DATA=[json.loads(p.read_text()) for p in PATHS]; SUM=[d['summary'] for d in DATA]
assert all(s['complete'] and s['num_matches']==50 for s in SUM), 'Every cooldown must be complete with n=50'
LABELS=['1-minute cooldown','5-minute cooldown','10-minute cooldown']; SHORT=['1 min','5 min','10 min']; X=np.arange(3)
BLUE='#4F46E5'; TEAL='#10B981'; CORAL='#F43F5E'; ATTACK='#F97360'; DEFEND='#14B8A6'; NEUTRAL='#8B5CF6'; ORANGE='#F59E0B'; INK='#111827'; GRID='#E2E8F0'; WHITE='#FFFFFF'
plt.rcParams.update({'figure.facecolor':'none','axes.facecolor':'none','savefig.facecolor':'none','text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':'#64748B','xtick.color':INK,'ytick.color':INK,'font.family':'DejaVu Sans','font.size':12,'font.weight':'bold','axes.labelweight':'bold','axes.titlesize':18,'axes.titleweight':'bold','grid.color':'#94A3B8','grid.linestyle':'--','grid.linewidth':0.85,'grid.alpha':.55})

def save(fig,name):

 fig.savefig(HERE/name,dpi=220,transparent=True); plt.close(fig); print(HERE/name)
def clean(ax):
 ax.spines[['top','right','left']].set_visible(False); ax.spines['bottom'].set_color('#CBD5E1'); ax.set_axisbelow(True); ax.grid(True,axis='both',zorder=0); ax.axhline(0,color='#94A3B8',lw=.9)
def classify(action):
 if action.endswith('_ATTACK'): return 'Attacking'
 if action.endswith('_DEFEND'): return 'Defending'
 if '_SUB_ROLE_' in action:
  role=action.rsplit('_SUB_ROLE_',1)[1]
  if role in {'ATTACKING_MIDFIELDER','WINGER','FORWARD'}: return 'Attacking'
  if role in {'GOALKEEPER','DEFENDER','DEFENSIVE_MIDFIELDER'}: return 'Defending'
 return 'Neutral'

# Chart 1: reliability is the primary bar metric; gain is additional in-bar detail.
improved=np.array([s['pct_matches_improved'] for s in SUM]); mean_gain=np.array([s['mean_delta_pp'] for s in SUM]); conditional=np.array([s['mean_gain_when_improved_pp'] for s in SUM])
fig,ax=plt.subplots(figsize=(8.5,6),layout='constrained')
bars=ax.bar(X,improved,width=.56,color=[BLUE,TEAL,CORAL],edgecolor='none',zorder=3)
best=int(np.argmax(improved))
for i,(bar,pct) in enumerate(zip(bars,improved)):
 ax.text(bar.get_x()+bar.get_width()/2,pct+2.0,f'{pct:.0f}%'+(' ★' if i==best else ''),ha='center',va='bottom',color='black',fontweight='bold',fontsize=14)
ax.set_xticks(X,[f'{x} cooldown\n(n=50)' for x in SHORT]); ax.set_ylim(0,86)
ax.set_ylabel('Matches improved (%)'); ax.set_xlabel('Minimum wait after an active intervention')
clean(ax)
ax2=ax.twinx(); ax2.plot(X,conditional,color=ORANGE,marker='D',markersize=8,markeredgecolor='white',markeredgewidth=1.2,linewidth=2.7,zorder=5,label='Gain when improved')
for i,v in enumerate(conditional): ax2.text(i,v+.18,f'+{v:.2f} pp',ha='center',va='bottom',color='black',fontweight='bold',fontsize=10)
ax2.set_ylim(0,12.5); ax2.set_yticks([]); ax2.set_ylabel(''); ax2.spines[['top','left','right']].set_visible(False)
ax.set_title('Possession Improvement by Cooldown',loc='center',pad=14,fontweight='bold')
ax.legend([bars,ax2.lines[0]],['Matches improved','Gain when improved'],frameon=True,facecolor='white',edgecolor='#E2E8F0',framealpha=.92,loc='upper left')
save(fig,'cooldown_performance_n50.png')

# Chart 2: minute ranges on x; grouped bar colors identify cooldowns.
bins=[(45,54),(55,64),(65,74),(75,84),(85,94)]
values=np.zeros((3,len(bins)))
for ci,d in enumerate(DATA):
 for row in d['matches']:
  for e in row['interventions']:
   if e['chosen_action']=='ACTION::NOOP': continue
   minute=int(e['minute'])
   for bi,(lo,hi) in enumerate(bins):
    if lo<=minute<=hi: values[ci,bi]+=1/50; break
fig,ax=plt.subplots(figsize=(8.5,6),layout='constrained')
centers=np.arange(len(bins)); width=.23
for ci,(label,color,offset) in enumerate(zip(SHORT,[BLUE,TEAL,CORAL],[-width,0,width])):
 ax.bar(centers+offset,values[ci],width*.92,color=color,edgecolor='none',label=f'{label} cooldown',zorder=3)
ax.set_xticks(centers,[f'{lo}–{hi}' for lo,hi in bins]); ax.set_ylim(0,5.7)
ax.set_ylabel('Active interventions per match'); ax.set_xlabel('Second-half minute')
ax.set_title('Timing of Active Interventions',loc='center',pad=14,fontweight='bold')
clean(ax)
ax.legend(frameon=True,facecolor='white',edgecolor='#E2E8F0',framealpha=.92,ncol=3,loc='upper right')
save(fig,'cooldown_intervention_patterns_n50.png')
