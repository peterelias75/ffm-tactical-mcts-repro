#!/usr/bin/env python3
"""Render E03 as ground-truth versus simulated event density on pitch grids."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle, Circle, Arc

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RESULTS=ROOT/'paper_reproduction'/'runs'/'e03_spatial_grid'/'results.json'
if not RESULTS.exists(): RESULTS=ROOT/'experiments'/'runs'/'e03_spatial_grid'/'results.json'
DATA=json.loads(RESULTS.read_text())['variants']
PITCH_L=120.; PITCH_W=80.
COLS=[
 ('coarse_4x6','Coarse 4×6 Grid'),
 ('regular_8x12','Regular 8×12 Grid'),
 ('fine_16x24','Fine 16×24 Grid'),
]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'font.weight':'bold','axes.titleweight':'bold','figure.facecolor':'none','savefig.facecolor':'none'})

def draw_pitch(ax):
 ax.set_facecolor('#23632b'); line='white'; lw=1.5
 ax.add_patch(Rectangle((0,0),120,80,fill=False,lw=lw,edgecolor=line,zorder=4))
 ax.plot([60,60],[0,80],color=line,lw=lw,zorder=4)
 ax.add_patch(Circle((60,40),9.15,fill=False,lw=lw,edgecolor=line,zorder=4)); ax.add_patch(Circle((60,40),.38,color=line,zorder=4))
 for x0,direction in ((0,1),(120,-1)):
  pen=x0+direction*18; six=x0+direction*6; spot=x0+direction*12
  ax.add_patch(Rectangle((min(x0,pen),18),abs(pen-x0),44,fill=False,lw=lw,edgecolor=line,zorder=4))
  ax.add_patch(Rectangle((min(x0,six),30),abs(six-x0),20,fill=False,lw=lw,edgecolor=line,zorder=4))
  ax.add_patch(Circle((spot,40),.38,color=line,zorder=4))
  theta=(-53,53) if direction==1 else (127,233)
  ax.add_patch(Arc((spot,40),18.3,18.3,theta1=theta[0],theta2=theta[1],lw=lw,color=line,zorder=4))
 ax.set_xlim(-1,121); ax.set_ylim(-1,81); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
 for spine in ax.spines.values(): spine.set_visible(False)

fig,axs=plt.subplots(2,3,figsize=(18,9.2),facecolor='none')
images=[]
for col,(key,title) in enumerate(COLS):
 v=DATA[key]; z=v['zone_matrices']
 gt=np.asarray(z['gt_avg_events_per_match'],dtype=float); sim=np.asarray(z['sim_avg_events_per_match'],dtype=float)
 norm=PowerNorm(gamma=.65,vmin=0,vmax=float(max(gt.max(),sim.max())))
 for row,(mat,label) in enumerate(((gt,'Ground Truth'),(sim,'Simulated'))):
  ax=axs[row,col]; draw_pitch(ax)
  im=ax.imshow(mat,extent=[0,120,0,80],origin='lower',cmap='cividis',norm=norm,alpha=.88,zorder=2,aspect='auto')
  rows,cols=mat.shape
  for gx in np.linspace(0,120,cols+1): ax.plot([gx,gx],[0,80],color='white',lw=.45,alpha=.30,zorder=3)
  for gy in np.linspace(0,80,rows+1): ax.plot([0,120],[gy,gy],color='white',lw=.45,alpha=.30,zorder=3)
  if row==0:
   ax.set_title(title,fontsize=18,pad=28,fontweight='bold')
   ax.text(.5,1.025,f'Total Zone MAD = {v["total_zone_mad"]:.2f}',transform=ax.transAxes,ha='center',va='bottom',fontsize=12,fontweight='normal',color='#334155')
 images.append(im)

fig.suptitle('Spatial Grid Resolution: Ground Truth vs Simulated Event Density',fontsize=22,fontweight='bold',y=.985)
fig.text(.018,.67,'Ground\nTruth',ha='center',va='center',fontsize=16,fontweight='bold')
fig.text(.018,.28,'Simulated',ha='center',va='center',fontsize=16,fontweight='bold')
fig.subplots_adjust(left=.055,right=.985,top=.86,bottom=.16,wspace=.10,hspace=.10)
colorbars=[]
for col in range(3):
 cb=fig.colorbar(images[col],ax=[axs[0,col],axs[1,col]],orientation='horizontal',fraction=.045,pad=.055,aspect=28)
 cb.ax.tick_params(labelsize=10)
 colorbars.append(cb)
fig.canvas.draw()
scale_y=min(cb.ax.get_position().y0 for cb in colorbars) - .040
fig.text(.52,scale_y,'Mean Events per Grid Cell',ha='center',va='top',fontsize=16,fontweight='bold',color='#111827')
out=HERE/'plot.png'; fig.savefig(out,dpi=220,facecolor='none',transparent=True,bbox_inches='tight',pad_inches=.01); plt.close(fig); print(out)
