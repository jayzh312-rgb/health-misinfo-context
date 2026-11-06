"""Build separate primary samples without running EDA or modifying raw data."""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PROCESSED=ROOT/"data/processed";TABLES=ROOT/"outputs/tables"
COAID_INPUT=PROCESSED/"coaid_content_level.csv";FAKE_INPUT=PROCESSED/"fakehealth_content_level.csv";DUP_INPUT=TABLES/"coaid_duplicate_audit.csv"
COAID_OUTPUT=PROCESSED/"coaid_news_primary.csv";FAKE_OUTPUT=PROCESSED/"fakehealth_healthstory_primary.csv"
SUMMARY_OUTPUT=TABLES/"primary_sample_summary.csv";ENTITY_OUTPUT=TABLES/"coaid_news_entity_resolution_audit.csv";SENSITIVITY_OUTPUT=TABLES/"primary_analysis_sensitivity_plan.csv"
SNOWFLAKE_EPOCH_MS=1288834974657

class DSU:
    def __init__(self,items):self.parent={x:x for x in items}
    def find(self,x):
        while self.parent[x]!=x:self.parent[x]=self.parent[self.parent[x]];x=self.parent[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)
    def groups(self):
        out=defaultdict(list)
        for x in self.parent:out[self.find(x)].append(x)
        return list(out.values())

def uid(prefix,items):
    raw="\x1f".join(sorted(map(str,items)));return prefix+hashlib.sha256(raw.encode()).hexdigest()[:24]
def idnorm(value):
    text=str(value).strip()
    return text[:-2] if re.fullmatch(r"\d+\.0",text) else text
def nonmissing(series):return series.notna()&series.astype(str).str.strip().ne("")
def decode(identifier):
    try:return pd.to_datetime((int(identifier)>>22)+SNOWFLAKE_EPOCH_MS,unit="ms",utc=True,errors="coerce")
    except (ValueError,TypeError,OverflowError):return pd.NaT

def raw_ids(row,caches):
    provenance=json.loads(row.provenance)
    oid=idnorm(provenance["original_content_id"])
    result={}
    for kind,path_key,map_key,id_col in (("posts","raw_tweet_file","index","tweet_id"),("replies","raw_reply_file","news_id","reply_id")):
        path=ROOT/provenance[path_key]
        if path not in caches:
            frame=pd.read_csv(path,dtype=str,encoding="utf-8-sig")
            grouped=defaultdict(set)
            for key,identifier in zip(frame[map_key],frame[id_col]):
                if pd.notna(key) and pd.notna(identifier):grouped[idnorm(key)].add(idnorm(identifier))
            caches[path]=dict(grouped)
        result[kind]=set(caches[path].get(oid,set()))
    return result

def build_fakehealth():
    frame=pd.read_csv(FAKE_INPUT,dtype={"original_content_id":str})
    out=frame[frame.source_release_or_subset.eq("HealthStory")].copy()
    out.insert(1,"subset","HealthStory")
    out["content_id"]=out.original_content_id
    out["stable_content_uid"]=out.content_uid
    out["original_dataset_label"]=out.veracity_original
    out["fakehealth_within_dataset_analysis_group"]=out.fakehealth_author_classification.map({"fake_rating_lt_3":"fake","real_rating_ge_3":"real","missing_rating":"missing"})
    out["timestamp_role"]="secondary_sensitivity_only"
    out["primary_outcomes"]="unique_post_count | unique_reply_count | engagement_present"
    return out

def build_coaid():
    all_rows=pd.read_csv(COAID_INPUT,dtype={"original_content_id":str});news=all_rows[all_rows.content_category.eq("News")].copy().set_index("content_uid",drop=False)
    audit=pd.read_csv(DUP_INPUT);audit=audit[audit.left_content_uid.isin(news.index)&audit.right_content_uid.isin(news.index)].copy()
    clear=audit[audit.classification.eq("clear duplicate/re-observation")];weak=audit[audit.classification.isin(["probable duplicate","unresolved"])]
    dsu=DSU(news.index)
    for x in clear.itertuples():dsu.union(x.left_content_uid,x.right_content_uid)
    components=[sorted(g) for g in dsu.groups()];member_to_entity={m:uid("coaid_news_entity_",g) for g in components for m in g}
    # Weak links create flags/groups over final entities but never collapse them.
    entities=sorted(set(member_to_entity.values()));weak_dsu=DSU(entities);weak_labels=defaultdict(set);weak_evidence=defaultdict(set)
    for x in weak.itertuples():
        a,b=member_to_entity[x.left_content_uid],member_to_entity[x.right_content_uid]
        if a==b:continue
        weak_dsu.union(a,b);weak_labels[a].add(x.classification);weak_labels[b].add(x.classification);weak_evidence[a].add(x.matching_evidence);weak_evidence[b].add(x.matching_evidence)
    possible_group={}
    for group in weak_dsu.groups():
        if len(group)>1:
            gid=uid("coaid_possible_",group)
            for entity in group:possible_group[entity]=gid
    caches={};records=[];audits=[]
    for members in components:
        eid=member_to_entity[members[0]];member_rows=news.loc[members]
        if isinstance(member_rows,pd.Series):member_rows=member_rows.to_frame().T
        reconstructed={m:raw_ids(news.loc[m],caches) for m in members}
        post_union=set().union(*(x["posts"] for x in reconstructed.values()));reply_union=set().union(*(x["replies"] for x in reconstructed.values()))
        # Richest text/title is the deterministic representative; release and UID break ties.
        candidates=list(member_rows.itertuples(index=False));representative=sorted(candidates,key=lambda x:(-len(str(x.text)) if pd.notna(x.text) else 0,-len(str(x.title)) if pd.notna(x.title) else 0,str(x.source_release_or_subset),x.content_uid))[0]
        releases=sorted(member_rows.source_release_or_subset.astype(str).unique());labels=sorted(member_rows.veracity_original.astype(str).unique())
        post_times=[decode(x) for x in post_union];post_times=[x for x in post_times if pd.notna(x)]
        first=min(post_times).isoformat() if post_times else "";last=max(post_times).isoformat() if post_times else "";lifespan=(max(post_times)-min(post_times)).total_seconds() if post_times else ""
        delay=""
        if first and pd.notna(representative.publish_date_utc) and str(representative.publish_date_utc).strip():delay=(pd.Timestamp(first)-pd.Timestamp(representative.publish_date_utc)).total_seconds()
        clear_pairs=clear[clear.left_content_uid.isin(members)&clear.right_content_uid.isin(members)]
        weak_status="+".join(sorted(weak_labels[eid]));collapsed=len(members)>1
        status=("clear_collapsed" if collapsed else "distinct")+(f"+possible_{weak_status}" if weak_status else "")
        provenance={"entity_rule":"connected component of clear duplicate/re-observation edges among CoAID News rows","member_content_uids":members,"member_release_periods":releases,"representative_content_uid":representative.content_uid,"raw_member_provenance":[json.loads(x) for x in member_rows.provenance]}
        record={c:getattr(representative,c) for c in member_rows.columns if c not in {"content_uid","observed_post_count","unique_post_count","reply_count","unique_reply_count","engagement_present","first_observed_post_time","last_observed_post_time","observed_lifespan","publication_to_first_post_delay","provenance","duplicate_classification","duplicate_group_id"}}
        record.update({"dataset":"CoAID","subset":"News","content_uid":eid,"stable_entity_uid":eid,"content_id":eid,"original_dataset_label":labels[0] if len(labels)==1 else json.dumps(labels),"coaid_within_dataset_analysis_group":labels[0].casefold() if len(labels)==1 else "label_conflict","source_release_or_subset":" | ".join(releases),"observed_post_count":len(post_union),"unique_post_count":len(post_union),"reply_count":len(reply_union),"unique_reply_count":len(reply_union),"engagement_present":bool(post_union or reply_union),"first_observed_post_time":first,"last_observed_post_time":last,"observed_lifespan":lifespan,"publication_to_first_post_delay":delay,"provenance":json.dumps(provenance,sort_keys=True,separators=(",",":")),"member_original_uids_json":json.dumps(members),"member_original_content_ids_json":json.dumps(member_rows.original_content_id.astype(str).tolist()),"member_release_periods_json":json.dumps(releases),"release_observation_count":len(members),"duplicate_status":status,"possible_duplicate_group":possible_group.get(eid,""),"sensitivity_exclude_candidate":bool(weak_status),"raw_observed_post_membership_sum":int(member_rows.observed_post_count.sum()),"raw_unique_post_count_sum":int(member_rows.unique_post_count.sum()),"raw_reply_membership_sum":int(member_rows.reply_count.sum()),"raw_unique_reply_count_sum":int(member_rows.unique_reply_count.sum()),"timestamp_role":"secondary_sensitivity_only","primary_outcomes":"unique_post_count | unique_reply_count | engagement_present"})
        records.append(record)
        member_post_counts={m:len(reconstructed[m]["posts"]) for m in members};member_reply_counts={m:len(reconstructed[m]["replies"]) for m in members}
        processed_post_counts={m:int(news.loc[m].unique_post_count) for m in members};processed_reply_counts={m:int(news.loc[m].unique_reply_count) for m in members}
        evidence=sorted(set(clear_pairs.matching_evidence.dropna())|weak_evidence[eid])
        audits.append({"final_entity_uid":eid,"member_original_uids":" | ".join(members),"member_release_periods":" | ".join(releases),"number_release_observations":len(members),"duplicate_classification":"clear duplicate/re-observation collapsed" if collapsed else "singleton","possible_duplicate_status":weak_status or "none","possible_duplicate_group":possible_group.get(eid,""),"raw_post_counts_by_member_json":json.dumps(member_post_counts,sort_keys=True),"processed_unique_post_counts_by_member_json":json.dumps(processed_post_counts,sort_keys=True),"raw_post_membership_sum":sum(member_post_counts.values()),"unioned_unique_post_count":len(post_union),"duplicate_post_memberships_removed":sum(member_post_counts.values())-len(post_union),"raw_reply_counts_by_member_json":json.dumps(member_reply_counts,sort_keys=True),"processed_unique_reply_counts_by_member_json":json.dumps(processed_reply_counts,sort_keys=True),"raw_reply_membership_sum":sum(member_reply_counts.values()),"unioned_unique_reply_count":len(reply_union),"duplicate_reply_memberships_removed":sum(member_reply_counts.values())-len(reply_union),"title_url_matching_evidence":" | ".join(evidence),"member_post_counts_match_processed":member_post_counts==processed_post_counts,"member_reply_counts_match_processed":member_reply_counts==processed_reply_counts,"all_post_ids_preserved_once":len(post_union)==record["unique_post_count"],"all_reply_ids_preserved_once":len(reply_union)==record["unique_reply_count"],"member_labels_consistent":len(labels)==1,"notes":"No ID summed across releases; final counts are set unions. Probable/unresolved links flag sensitivity only."})
    return pd.DataFrame(records),pd.DataFrame(audits),audit

def coverage(frame,column):
    available=nonmissing(frame[column]);return int(available.sum()),100*available.mean()
def summary_rows(fake,coaid):
    rows=[]
    configs=[("FakeHealth","HealthStory",fake,"fakehealth_within_dataset_analysis_group"),("CoAID","News",coaid,"coaid_within_dataset_analysis_group")]
    for dataset,subset,frame,label in configs:
        for group in ("all","fake","real"):
            part=frame if group=="all" else frame[frame[label].eq(group)]
            fake_n=int(part[label].eq("fake").sum());real_n=int(part[label].eq("real").sum());n=len(part)
            row={"dataset":dataset,"subset":subset,"group":group,"total_rows":n,"fake_false_rows":fake_n,"real_true_rows":real_n,"fake_false_percent":100*fake_n/n if n else 0,"real_true_percent":100*real_n/n if n else 0,"rows_with_text":int(nonmissing(part.text).sum()),"rows_with_publication_dates":int(nonmissing(part.publish_date_utc).sum()),"rows_with_any_engagement":int(part.engagement_present.fillna(False).astype(bool).sum()),"median_unique_post_count":part.unique_post_count.median() if n else "","median_reply_count":part.unique_reply_count.median() if n else "","number_flagged_possible_duplicates":int(part.get("sensitivity_exclude_candidate",pd.Series(False,index=part.index)).fillna(False).astype(bool).sum()),"number_collapsed_coaid_entities":int(part.get("release_observation_count",pd.Series(1,index=part.index)).gt(1).sum())}
            for col in ("title","text","publish_date_utc","unique_post_count","unique_reply_count","engagement_present","first_observed_post_time","observed_lifespan","publication_to_first_post_delay"):
                cn,cp=coverage(part,col);row[f"{col}_nonmissing_n"]=cn;row[f"{col}_coverage_pct"]=cp
            rows.append(row)
    return pd.DataFrame(rows)

def sensitivity_plan():
    return pd.DataFrame([
      {"specification":"A Primary","coaid_entity_rule":"Collapse connected components formed only by clear duplicate/re-observation edges; union unique post/reply IDs","probable_unresolved_rule":"Retain as separate flagged entities","release_handling":"All releases represented through entity provenance","run_now":False},
      {"specification":"B Conservative","coaid_entity_rule":"Use the same clear-collapsed entities","probable_unresolved_rule":"Exclude sensitivity_exclude_candidate=True","release_handling":"All releases represented through entity provenance","run_now":False},
      {"specification":"C Release-specific","coaid_entity_rule":"Do not collapse across releases","probable_unresolved_rule":"Retain and flag within each release","release_handling":"Analyze each CoAID release separately","run_now":False},
    ])

def main():
    for path in (COAID_INPUT,FAKE_INPUT,DUP_INPUT):
        if not path.exists():raise FileNotFoundError(path)
    fake=build_fakehealth();coaid,entity_audit,pair_audit=build_coaid()
    fake.to_csv(FAKE_OUTPUT,index=False);coaid.to_csv(COAID_OUTPUT,index=False);summary_rows(fake,coaid).to_csv(SUMMARY_OUTPUT,index=False);entity_audit.to_csv(ENTITY_OUTPUT,index=False);sensitivity_plan().to_csv(SENSITIVITY_OUTPUT,index=False)
    checks=("member_post_counts_match_processed","member_reply_counts_match_processed","all_post_ids_preserved_once","all_reply_ids_preserved_once","member_labels_consistent")
    if not entity_audit[list(checks)].all().all():raise RuntimeError("Engagement union/member validation failed")
    print(f"FakeHealth HealthStory rows={len(fake):,}; CoAID News entities={len(coaid):,}; clear groups={(entity_audit.number_release_observations>1).sum():,}; validation=passed")
    return 0
if __name__=="__main__":raise SystemExit(main())
