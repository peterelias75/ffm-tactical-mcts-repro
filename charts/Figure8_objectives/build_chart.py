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

def classify(a):
 if a.endswith('_ATTACK'): return 'Attacking'
 if a.endswith('_DEFEND'): return 'Defending'
 if '_SUB_ROLE_' in a:
  role=a.rsplit('_SUB_ROLE_',1)[1]
  if role in {'ATTACKING_MIDFIELDER','WINGER','FORWARD'}: return 'Attacking'
  if role in {'GOALKEEPER','DEFENDER','DEFENSIVE_MIDFIELDER'}: return 'Defending'
 return 'Neutral'
names=['Shots-only','Balanced']; rs=[load(RUNS/'Figure8_plain_mcts_step1_sim5_lookahead200_n50/shots_only/results.json')['matches'],load(RUNS/'Figure8_plain_mcts_step1_sim5_lookahead200_n50/balanced/results.json')['matches']]; x=np.arange(2); w=.34
shots=[sum(z['shot_delta_vs_noop'] for z in r)/len(r) for r in rs]; pos=[sum(z['delta_pp'] for z in r)/len(r) for r in rs]
fig,a=plt.subplots(figsize=(8.5,6)); bs=a.bar(x-w/2,shots,w,color=ORANGE,label='Shots'); bp=a.bar(x+w/2,pos,w,color=BLUE,label='Possession (pp)'); barlabels(a,bs,stars=(int(np.argmax(shots)),)); barlabels(a,bp,stars=(int(np.argmax(pos)),)); a.set_xticks(x,names); a.set_title('Figure 8a — Average Gain by Objective',pad=14); a.set_ylabel('Average gain vs paired no-action'); a.set_xlabel('Planning objective'); a.legend(frameon=False); clean(a); save(fig,'figure_8a.png')
si=[100*sum(z['shot_delta_vs_noop']>0 for z in r)/len(r) for r in rs]; pi=[100*sum(z['delta_pp']>0 for z in r)/len(r) for r in rs]
fig,b=plt.subplots(figsize=(8.5,6)); bs=b.bar(x-w/2,si,w,color=ORANGE,label='Shots'); bp=b.bar(x+w/2,pi,w,color=BLUE,label='Possession'); barlabels(b,bs,'{:.0f}','%',stars=(int(np.argmax(si)),)); barlabels(b,bp,'{:.0f}','%',stars=(int(np.argmax(pi)),)); b.set_xticks(x,names); b.set_ylim(0,100); b.set_title('Figure 8b — Improvement Rate by Objective',pad=14); b.set_ylabel('Matches improved (%)'); b.set_xlabel('Planning objective'); b.legend(frameon=False); clean(b); save(fig,'figure_8b.png')
counts=[]
for r in rs:
 q=Counter(classify(d['chosen_action']) for z in r for d in z['interventions'] if d['chosen_action']!='ACTION::NOOP'); counts.append([q['Attacking'],q['Defending'],q['Neutral']])
fig,c=plt.subplots(figsize=(8.5,6)); c.axis('off'); c.set_title('Figure 8c — Active Intervention Mix by Objective',pad=14); piecols=[CORAL,TEAL,SLATE]
for i,(cnt,label) in enumerate(zip(counts,names)):
 ia=c.inset_axes([.03+i*.49,.16,.45,.70]); ia.pie(cnt,colors=piecols,autopct='%1.0f%%',textprops={'color':FG,'weight':'bold'},wedgeprops={'edgecolor':'white','linewidth':.7}); ia.text(0,-1.25,label,ha='center',color=FG,fontweight='bold'); ia.set_aspect('equal')
c.legend(['Attacking','Defending','Neutral'],title='Intervention type',frameon=False,loc='upper center',ncol=3); fig.set_size_inches(8.5,6,forward=True); fig.subplots_adjust(left=.06,right=.94,bottom=.10,top=.88); fig.savefig(HERE/'figure_8c.png',dpi=220,facecolor='none',transparent=True); plt.close(fig); print(HERE/'figure_8c.png')
