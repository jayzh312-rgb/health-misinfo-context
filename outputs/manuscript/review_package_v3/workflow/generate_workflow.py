from pathlib import Path
import json,os,tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'mentor-workflow-mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MPath
R=Path(__file__).resolve().parent
nodes={
'A':{'box':[280,10,550,75],'lines':['CoAID News and FakeHealth HealthStory'],'branch':'shared'},
'B':{'box':[280,110,550,85],'lines':['Validate identities, versions and dataset labels'],'branch':'shared'},
'C':{'box':[35,245,440,105],'lines':['Engagement populations','Unique observed-post counts','Include zero-post entities'],'branch':'engagement'},
'D':{'box':[570,245,480,105],'lines':['Separate text populations','Quality filters and resource/text leakage groups','Engagement is excluded from predictors'],'branch':'text'},
'E':{'box':[35,390,440,100],'lines':['CoAID hurdle; FakeHealth NB2','Standardized Fake/Real','expected-post ratios'],'branch':'engagement'},
'F':{'box':[35,530,440,100],'lines':['Independent overlap-cluster','resampling within each dataset','Percentile intervals'],'branch':'engagement'},
'G':{'box':[35,670,440,110],'lines':['Cross-context ratio-of-ratios','Prespecified engagement sensitivities','RQ1'],'branch':'engagement'},
'H':{'box':[570,390,480,130],'lines':['Grouped source-only selection','Within-context fits: training pools','Evaluation: reserved holdouts','RQ2'],'branch':'text'},
'I':{'box':[570,560,480,115],'lines':['Transfer fits: all eligible source rows','Locked source-selected configurations','Cross-context evaluation; no target tuning','Matched target holdouts are primary'],'branch':'text'},
'K':{'box':[570,730,480,115],'lines':['Frozen full-source LR/SVM only','Common COVID-Rumor NEWS target','No target fitting or tuning','RQ3'],'branch':'text'},
'J':{'box':[570,910,480,110],'lines':['Separate supporting checks','Lexical audit: within-context LR models','Publisher, representation','and transformer checks'],'branch':'text'},
'N':{'box':[35,1030,440,110],'lines':['Extension: FakeNewsNet (PolitiFact, GossipCop)','Title-only LR external target (Table S13)','Engagement under frozen estimand (Table S17)','Supplementary; no frozen result changes'],'branch':'external'},
'R':{'box':[35,830,440,160],'lines':['Independent COVID-Rumor NEWS','Inherited True/False annotations','Quality and exact-overlap screening','External text evaluation only'],'branch':'external'},
}
edges=[['A','B'],['B','C'],['B','D'],['C','E'],['E','F'],['F','G'],['D','H'],['H','I'],['H','J'],['I','K'],['R','K'],['N','K'],['N','E']]
# Routes are presentation coordinates; all edges follow the corresponding Mermaid graph.
routes={
'A-B':[[555,85],[555,110]],
'B-C':[[400,195],[400,221],[255,221],[255,245]],
'B-D':[[710,195],[710,221],[810,221],[810,245]],
'C-E':[[255,350],[255,390]],'E-F':[[255,490],[255,530]],'F-G':[[255,630],[255,670]],
'D-H':[[810,350],[810,390]],'H-I':[[810,520],[810,560]],
'H-J':[[1050,455],[1090,455],[1090,965],[1050,965]],
'I-K':[[810,675],[810,730]],
'R-K':[[475,900],[520,900],[520,787],[570,787]],
'N-K':[[475,1085],[540,1085],[540,787],[570,787]],
'N-E':[[22,1085],[22,440],[35,440]],
}
layout={'canvas':[1120,1160],'nodes':nodes,'edges':edges,'routes':routes,'protocol_mapping':{'C':'Methods 2.2/2.4: distinct engagement populations and zero-inclusive unique observed-post counts','E':'Methods 2.5: CoAID hurdle, FakeHealth NB2 and common standardized ratio estimand','F':'Methods 2.4/2.5: overlap-component bootstrap independently within datasets','H':'Methods 2.6/2.7: training-pool source selection and reserved holdouts','I':'Methods 2.8: refits on all eligible source rows; locked settings and transform-only targets','J':'Methods 2.8–2.10: lexical models are within-context fits; other checks retain their own stated protocols','R':'Methods 2.11: inherited T/F NEWS target, quality/overlap restrictions; excluded from engagement','K':'Methods 2.11: frozen natural full-source LR/SVM only; no target adaptation','N':'Supplementary S8/S13/S17: FakeNewsNet extension; title-only LR scoring and engagement under the frozen estimand; populations and labels inherited'}}
(R/'workflow_layout.json').write_text(json.dumps(layout,indent=2))
mermaid='flowchart TD\n'
for k,n in nodes.items():mermaid+='    '+k+'["'+'<br/>'.join(n['lines'])+'"]\n'
for a,b in edges:mermaid+='    '+a+' --> '+b+'\n'
mermaid+='    %% Engagement and text populations remain distinct; there is no mediation edge.\n'
(R/'workflow.mmd').write_text(mermaid)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(11.2,11.6));fig.subplots_adjust(left=0,right=1,bottom=0,top=1);ax.set_xlim(0,1120);ax.set_ylim(1160,0);ax.set_aspect('equal');ax.axis('off')
colors={'shared':('#f2f4f6','#65717a'),'engagement':('#edf5f8','#31627a'),'text':('#f2f5f9','#536987'),'external':('#f6f1f8','#85618c')}
for a,b in edges:
 points=routes[a+'-'+b];path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1));ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=13,linewidth=1.25,color='#71808c',zorder=1))
for k,n in nodes.items():
 x,y,w,h=n['box'];fill,line=colors[n['branch']];ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=10',facecolor=fill,edgecolor=line,linewidth=1.15,zorder=2))
 gap=25;mid=y+h/2
 for i,text in enumerate(n['lines']):
  fontsize=12.2 if len(text)>44 else 14.3
  ax.text(x+w/2,mid+(i-(len(n['lines'])-1)/2)*gap,text,ha='center',va='center',fontsize=fontsize,fontweight='bold' if i==0 else 'normal',color='#18242e',zorder=3)
for x,label,color in [(35,'ENGAGEMENT',colors['engagement'][1]),(570,'TEXT',colors['text'][1])]:ax.text(x,236,label,ha='left',va='bottom',fontsize=11.3,fontweight='bold',color=color)
ax.text(1100,704,'Supporting checks',ha='center',va='center',rotation=90,fontsize=10.5,color='#536987')
for ext in ['svg','pdf','png']:
 fig.savefig(R/f'figure_M1_workflow.{ext}',dpi=300,facecolor='white',metadata={'Title':'Figure M1. Analytical workflow'} if ext!='png' else None)
svg=R/'figure_M1_workflow.svg'
svg.write_text(svg.read_text().replace("'DejaVu Sans'",'Arial, sans-serif'))
plt.close(fig)
print(json.dumps({'workflow_nodes':len(nodes),'workflow_edges':len(edges),'exports':['SVG','PDF','300-dpi PNG']}))
