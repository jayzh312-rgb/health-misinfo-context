"""First within-dataset engagement EDA and bootstrap effect sizes; no tests/models."""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROCESSED=ROOT/"data/processed";TABLES=ROOT/"outputs/tables";FIGURES=ROOT/"outputs/figures/engagement_eda"
FAKE_PATH=PROCESSED/"fakehealth_healthstory_primary.csv";COAID_PATH=PROCESSED/"coaid_news_primary.csv";COAID_OBSERVATIONS=PROCESSED/"coaid_content_level.csv"
DESC_PATH=TABLES/"primary_engagement_descriptives.csv";EFFECT_PATH=TABLES/"primary_engagement_effect_sizes.csv";DIAG_PATH=TABLES/"engagement_distribution_diagnostics.csv";SENS_PATH=TABLES/"coaid_engagement_sensitivity_results.csv"
SEED=42;BOOTSTRAP_RESAMPLES=5000;CI_METHOD="stratified nonparametric percentile bootstrap"
COUNTS=("unique_post_count","unique_reply_count")

def load_samples():
    fake=pd.read_csv(FAKE_PATH);coaid=pd.read_csv(COAID_PATH)
    fake["analysis_group"]=fake.fakehealth_within_dataset_analysis_group
    coaid["analysis_group"]=coaid.coaid_within_dataset_analysis_group
    return {"FakeHealth HealthStory":fake,"CoAID News":coaid}

def describe(dataset,frame):
    rows=[]
    for group in ("fake","real"):
        part=frame[frame.analysis_group.eq(group)]
        for outcome in (*COUNTS,"engagement_present"):
            x=pd.to_numeric(part[outcome],errors="coerce").dropna().astype(float);n=len(x);mean=x.mean();var=x.var(ddof=1)
            row={"dataset":dataset,"veracity_group":group,"outcome":outcome,"N":n,"mean":mean,"median":x.median(),"standard_deviation":x.std(ddof=1),"minimum":x.min(),"percentile_25":x.quantile(.25),"percentile_75":x.quantile(.75),"IQR":x.quantile(.75)-x.quantile(.25),"percentile_90":x.quantile(.90),"percentile_95":x.quantile(.95),"maximum":x.max(),"variance":var,"variance_to_mean_ratio":var/mean if mean else np.nan,"number_zeros":int((x==0).sum()),"percentage_zeros":100*(x==0).mean(),"engaged_count":int((x>0).sum()) if outcome=="engagement_present" else "","non_engaged_count":int((x==0).sum()) if outcome=="engagement_present" else "","engaged_percentage":100*(x>0).mean() if outcome=="engagement_present" else "","non_engaged_percentage":100*(x==0).mean() if outcome=="engagement_present" else ""}
            rows.append(row)
    return rows

def concentration(x,proportion):
    x=np.asarray(x,dtype=float);n=len(x);k=max(1,math.ceil(n*proportion));top=np.sort(x)[-k:];total=x.sum()
    return float(top.min()),float(top.sum()/total) if total else np.nan,k

def classify(mean,var,zero):
    if mean==0:return "uncertain"
    over=var/mean>2;heavy=zero>=30
    if over and heavy:return "both overdispersed and zero-heavy"
    if over:return "overdispersed"
    if heavy:return "highly zero-heavy"
    if .5<=var/mean<=2:return "approximately Poisson-like"
    return "uncertain"

def diagnostics(samples):
    rows=[]
    for dataset,frame in samples.items():
        for outcome in COUNTS:
            x=pd.to_numeric(frame[outcome],errors="coerce").dropna().to_numpy(float);mean=x.mean();var=x.var(ddof=1);zero=100*np.mean(x==0);t1,s1,k1=concentration(x,.01);t5,s5,k5=concentration(x,.05)
            rows.append({"dataset":dataset,"outcome":outcome,"N":len(x),"mean":mean,"variance":var,"variance_to_mean":var/mean if mean else np.nan,"zero_percentage":zero,"top_1_percent_item_count":k1,"top_1_percent_threshold":t1,"top_1_percent_share":s1,"top_5_percent_item_count":k5,"top_5_percent_threshold":t5,"top_5_percent_share":s5,"maximum":x.max(),"diagnostic_classification":classify(mean,var,zero),"outliers_removed":False})
    return pd.DataFrame(rows)

def bootstrap_count(fake,real,resamples=BOOTSTRAP_RESAMPLES,seed=SEED):
    fake=np.asarray(fake,float);real=np.asarray(real,float);rng=np.random.default_rng(seed);stats={"mean_difference":[],"mean_ratio":[],"median_difference":[]};batch=100
    for start in range(0,resamples,batch):
        b=min(batch,resamples-start);fs=fake[rng.integers(0,len(fake),(b,len(fake)))];rs=real[rng.integers(0,len(real),(b,len(real)))];fm=fs.mean(1);rm=rs.mean(1)
        stats["mean_difference"].extend(fm-rm);stats["mean_ratio"].extend(np.divide(fm,rm,out=np.full(b,np.nan),where=rm!=0));stats["median_difference"].extend(np.median(fs,1)-np.median(rs,1))
    return {k:np.asarray(v,float) for k,v in stats.items()}

def bootstrap_proportion(fake,real,resamples=BOOTSTRAP_RESAMPLES,seed=SEED):
    fake=np.asarray(fake,float);real=np.asarray(real,float);rng=np.random.default_rng(seed);values=[];batch=200
    for start in range(0,resamples,batch):
        b=min(batch,resamples-start);fs=fake[rng.integers(0,len(fake),(b,len(fake)))];rs=real[rng.integers(0,len(real),(b,len(real)))];values.extend(fs.mean(1)-rs.mean(1))
    return np.asarray(values,float)

def ci(values):
    values=np.asarray(values);values=values[np.isfinite(values)]
    return tuple(np.quantile(values,[.025,.975])) if len(values) else (np.nan,np.nan)

def effects_for(dataset,frame,resamples=BOOTSTRAP_RESAMPLES,seed=SEED,extra=None):
    fake=frame[frame.analysis_group.eq("fake")];real=frame[frame.analysis_group.eq("real")];rows=[]
    for outcome in COUNTS:
        f=fake[outcome].to_numpy(float);r=real[outcome].to_numpy(float);boots=bootstrap_count(f,r,resamples,seed)
        estimates={"mean_difference_fake_minus_real":f.mean()-r.mean(),"mean_ratio_fake_over_real":f.mean()/r.mean() if r.mean() else np.nan,"median_difference_fake_minus_real":np.median(f)-np.median(r)}
        mapping={"mean_difference_fake_minus_real":"mean_difference","mean_ratio_fake_over_real":"mean_ratio","median_difference_fake_minus_real":"median_difference"}
        for effect,estimate in estimates.items():
            lo,hi=ci(boots[mapping[effect]]);row={"dataset":dataset,"outcome":outcome,"effect_type":effect,"estimate":estimate,"lower_95_CI":lo,"upper_95_CI":hi,"fake_N":len(f),"real_N":len(r),"bootstrap_resamples":resamples,"CI_method":CI_METHOD,"random_seed":seed};row.update(extra or {});rows.append(row)
    f=fake.engagement_present.astype(float).to_numpy();r=real.engagement_present.astype(float).to_numpy();boots=bootstrap_proportion(f,r,resamples,seed);lo,hi=ci(boots);row={"dataset":dataset,"outcome":"engagement_present","effect_type":"engagement_proportion_difference_fake_minus_real","estimate":f.mean()-r.mean(),"lower_95_CI":lo,"upper_95_CI":hi,"fake_N":len(f),"real_N":len(r),"bootstrap_resamples":resamples,"CI_method":CI_METHOD,"random_seed":seed};row.update(extra or {});rows.append(row)
    return rows

def safe_name(value):return value.casefold().replace(" ","_")
def plot_distributions(samples):
    FIGURES.mkdir(parents=True,exist_ok=True)
    colors={"fake":"#c23b22","real":"#2878b5"}
    for dataset,frame in samples.items():
        for outcome in COUNTS:
            groups={g:frame.loc[frame.analysis_group.eq(g),outcome].to_numpy(float) for g in ("fake","real")};base=f"{safe_name(dataset)}_{outcome}"
            fig,ax=plt.subplots(figsize=(7,4.5))
            for g,x in groups.items():ax.hist(x,bins=50,alpha=.52,label=g,color=colors[g])
            ax.set(title=f"{dataset}: {outcome} (raw scale)",xlabel=outcome,ylabel="Items");ax.legend();fig.tight_layout();fig.savefig(FIGURES/f"{base}_hist_raw.png",dpi=180);plt.close(fig)
            fig,ax=plt.subplots(figsize=(7,4.5))
            for g,x in groups.items():ax.hist(np.log1p(x),bins=50,alpha=.52,label=g,color=colors[g])
            ax.set(title=f"{dataset}: log1p({outcome})",xlabel=f"log1p({outcome})",ylabel="Items");ax.legend();fig.tight_layout();fig.savefig(FIGURES/f"{base}_hist_log1p.png",dpi=180);plt.close(fig)
            fig,ax=plt.subplots(figsize=(7,4.5))
            for g,x in groups.items():
                sx=np.sort(x);y=np.arange(1,len(sx)+1)/len(sx);ax.step(sx,y,where="post",label=g,color=colors[g])
            ax.set_xscale("symlog",linthresh=1);ax.set(title=f"{dataset}: ECDF of {outcome}",xlabel=f"{outcome} (symlog scale)",ylabel="Cumulative proportion");ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(FIGURES/f"{base}_ecdf.png",dpi=180);plt.close(fig)
            fig,ax=plt.subplots(figsize=(7,4.5));data=[np.log1p(groups[g]) for g in ("fake","real")];box=ax.boxplot(data,tick_labels=["fake","real"],patch_artist=True,showfliers=True)
            for patch,g in zip(box["boxes"],("fake","real")):patch.set_facecolor(colors[g]);patch.set_alpha(.55)
            ax.set(title=f"{dataset}: {outcome} boxplot",xlabel="Veracity group",ylabel=f"log1p({outcome}); all outliers retained");fig.tight_layout();fig.savefig(FIGURES/f"{base}_boxplot_log1p.png",dpi=180);plt.close(fig)

def sensitivity(coaid):
    rows=[]
    rows.extend(effects_for("CoAID News",coaid,extra={"specification":"A Primary entity-resolved","release":"all","excluded_possible_duplicates":0}))
    conservative=coaid[~coaid.sensitivity_exclude_candidate.astype(bool)].copy();rows.extend(effects_for("CoAID News",conservative,extra={"specification":"B Conservative","release":"all","excluded_possible_duplicates":len(coaid)-len(conservative)}))
    observations=pd.read_csv(COAID_OBSERVATIONS);observations=observations[observations.content_category.eq("News")].copy();observations["analysis_group"]=observations.veracity_original.str.casefold()
    for release,part in observations.groupby("source_release_or_subset"):
        if {"fake","real"}<=set(part.analysis_group):rows.extend(effects_for("CoAID News",part,extra={"specification":"C Release-specific uncollapsed","release":release,"excluded_possible_duplicates":0}))
    return pd.DataFrame(rows)

def main():
    for path in (FAKE_PATH,COAID_PATH,COAID_OBSERVATIONS):
        if not path.exists():raise FileNotFoundError(path)
    TABLES.mkdir(parents=True,exist_ok=True);samples=load_samples()
    descriptives=[]
    for dataset,frame in samples.items():descriptives.extend(describe(dataset,frame))
    pd.DataFrame(descriptives).to_csv(DESC_PATH,index=False)
    primary=[]
    for dataset,frame in samples.items():primary.extend(effects_for(dataset,frame))
    pd.DataFrame(primary).to_csv(EFFECT_PATH,index=False);diagnostics(samples).to_csv(DIAG_PATH,index=False);sensitivity(samples["CoAID News"]).to_csv(SENS_PATH,index=False);plot_distributions(samples)
    print(f"Rows: FakeHealth={len(samples['FakeHealth HealthStory']):,}, CoAID={len(samples['CoAID News']):,}; bootstrap={BOOTSTRAP_RESAMPLES:,}; figures={len(list(FIGURES.glob('*.png')))}")
    return 0
if __name__=="__main__":raise SystemExit(main())
