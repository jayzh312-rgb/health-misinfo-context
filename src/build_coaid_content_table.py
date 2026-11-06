"""Build a provenance-specific CoAID content table; never alters raw data."""
from __future__ import annotations

import itertools
from collections import Counter
from pathlib import Path
import pandas as pd

from content_table_utils import COMMON_COLUMNS, clean_scalar, json_text, normalize_text, normalize_url, parse_date, project_root, stable_uid, timing

ROOT = project_root(__file__)
RAW = ROOT / "data/raw/coaid/CoAID-master"
OUT = ROOT / "data/processed/coaid_content_level.csv"
DUP_OUT = ROOT / "outputs/tables/coaid_duplicate_audit.csv"

def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try: return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError: pass
    raise UnicodeDecodeError("unknown", b"", 0, 1, str(path))

def label_parts(stem: str) -> tuple[str,str]:
    return ("News" if stem.startswith("News") else "Claim", "Fake" if "Fake" in stem else "Real")

def id_strings(series: pd.Series | None) -> list[str]:
    if series is None: return []
    return [str(int(x)) if isinstance(x, float) and x.is_integer() else str(x) for x in series.dropna()]

def id_key(value) -> str:
    return str(int(value)) if pd.notna(value) and isinstance(value, float) and value.is_integer() else clean_scalar(value)

def title_signature(value) -> str:
    signature=normalize_text(value)
    boilerplate={"403 forbidden","404 not found","access denied","page not found","just a moment","error"}
    return "" if signature in boilerplate or len(signature)<8 else signature

def build() -> tuple[pd.DataFrame,pd.DataFrame]:
    records=[]; post_sets={}
    for path in sorted(RAW.glob("*/*.csv")):
        if "_tweets" in path.stem: continue
        release=path.parent.name; stem=path.stem; category,veracity=label_parts(stem); frame=read_csv(path)
        tweets_path=path.with_name(stem+"_tweets.csv"); replies_path=path.with_name(stem+"_tweets_replies.csv")
        tweets=read_csv(tweets_path) if tweets_path.exists() else pd.DataFrame()
        replies=read_csv(replies_path) if replies_path.exists() else pd.DataFrame()
        tweet_groups={id_key(key):id_strings(group["tweet_id"]) for key,group in tweets.groupby(tweets["index"].map(id_key),dropna=False)} if "index" in tweets and "tweet_id" in tweets else {}
        reply_groups={id_key(key):id_strings(group["reply_id"]) for key,group in replies.groupby(replies["news_id"].map(id_key),dropna=False)} if "news_id" in replies and "reply_id" in replies else {}
        for _,raw in frame.iterrows():
            oid=raw.get("Unnamed: 0"); oid_key=id_key(oid)
            post_ids=tweet_groups.get(oid_key,[]); reply_ids=reply_groups.get(oid_key,[])
            date_raw=clean_scalar(raw.get("publish_date")); date_utc,date_quality=parse_date(raw.get("publish_date"))
            title=clean_scalar(raw.get("newstitle")) or clean_scalar(raw.get("title"))
            text=clean_scalar(raw.get("content")) or clean_scalar(raw.get("abstract"))
            uid="coaid_"+stable_uid(release,path.name,oid_key)
            post_sets[uid]=set(post_ids)
            provenance={"raw_content_file":path.relative_to(ROOT).as_posix(),"raw_tweet_file":tweets_path.relative_to(ROOT).as_posix() if tweets_path.exists() else "","raw_reply_file":replies_path.relative_to(ROOT).as_posix() if replies_path.exists() else "","raw_row_index":int(raw.name),"original_content_id":oid_key}
            record={"dataset":"CoAID","source_release_or_subset":release,"content_uid":uid,"content_category":category,"veracity_original":veracity,"title":title,"text":text,"publish_date_raw":date_raw,"publish_date_utc":date_utc,"publish_date_quality":date_quality,"observed_post_count":len(post_ids),"unique_post_count":len(set(post_ids)),"reply_count":len(reply_ids),"unique_reply_count":len(set(reply_ids)),"engagement_present":bool(post_ids or reply_ids),**timing(post_ids,date_utc),"provenance":json_text(provenance),"original_content_id":oid_key,"original_filename_category":stem,"raw_content_file":provenance["raw_content_file"],"news_url":clean_scalar(raw.get("news_url")),"fact_check_url":clean_scalar(raw.get("fact_check_url")),"source_title":clean_scalar(raw.get("title")),"duplicate_classification":"distinct","duplicate_group_id":""}
            records.append(record)
    result=pd.DataFrame(records)
    audits=[]; candidate_uids=set(); clear_graph={uid:set() for uid in result.content_uid}
    signatures={}
    for _,row in result.iterrows():
        signatures[row.content_uid]={"url":normalize_url(row.news_url),"fact":normalize_url(row.fact_check_url),"title":title_signature(row.title),"source_title":title_signature(row.source_title),"text":normalize_text(row.text),"posts":post_sets[row.content_uid]}
    title_frequency=Counter((row.source_release_or_subset,signatures[row.content_uid]["title"]) for row in result.itertuples(index=False) if signatures[row.content_uid]["title"])
    source_title_frequency=Counter((row.source_release_or_subset,signatures[row.content_uid]["source_title"]) for row in result.itertuples(index=False) if signatures[row.content_uid]["source_title"])
    rows=list(result.itertuples(index=False))
    for left,right in itertools.combinations(rows,2):
        if left.source_release_or_subset==right.source_release_or_subset: continue
        a,b=signatures[left.content_uid],signatures[right.content_uid]
        evidence=[]
        if a["url"] and a["url"]==b["url"]: evidence.append("news_url_exact")
        if a["fact"] and a["fact"]==b["fact"]: evidence.append("fact_check_url_exact")
        if a["title"] and a["title"]==b["title"] and title_frequency[(left.source_release_or_subset,a["title"])]<=5 and title_frequency[(right.source_release_or_subset,b["title"])]<=5: evidence.append("title_exact")
        if a["source_title"] and a["source_title"]==b["source_title"] and source_title_frequency[(left.source_release_or_subset,a["source_title"])]<=5 and source_title_frequency[(right.source_release_or_subset,b["source_title"])]<=5: evidence.append("source_title_exact")
        if len(a["text"])>=100 and a["text"]==b["text"]: evidence.append("content_text_exact")
        shared_posts=len(a["posts"]&b["posts"])
        if shared_posts: evidence.append("shared_social_ids")
        if not evidence or evidence==["fact_check_url_exact"]: continue
        conflict=left.veracity_original!=right.veracity_original or left.content_category!=right.content_category
        if conflict: classification="unresolved"
        elif "news_url_exact" in evidence or "content_text_exact" in evidence or ({"title_exact","fact_check_url_exact"}<=set(evidence)): classification="clear duplicate/re-observation"
        elif "title_exact" in evidence or "source_title_exact" in evidence or shared_posts: classification="probable duplicate"
        else: classification="unresolved"
        candidate_uids.update((left.content_uid,right.content_uid))
        if classification=="clear duplicate/re-observation": clear_graph[left.content_uid].add(right.content_uid);clear_graph[right.content_uid].add(left.content_uid)
        audits.append({"left_content_uid":left.content_uid,"right_content_uid":right.content_uid,"left_release":left.source_release_or_subset,"right_release":right.source_release_or_subset,"left_category":left.content_category,"right_category":right.content_category,"left_veracity":left.veracity_original,"right_veracity":right.veracity_original,"classification":classification,"matching_evidence":" | ".join(evidence),"shared_social_id_count":shared_posts,"automatic_action":"none; rows preserved"})
    # Connected components only for clear duplicates; flags never remove rows.
    visited=set(); groups={}
    for uid in clear_graph:
        if uid in visited or not clear_graph[uid]: continue
        stack=[uid]; component=[]
        while stack:
            x=stack.pop()
            if x in visited: continue
            visited.add(x);component.append(x);stack.extend(clear_graph[x]-visited)
        gid="coaid_dup_"+stable_uid(*sorted(component))
        for x in component: groups[x]=gid
    severity={"distinct":0,"probable duplicate":1,"clear duplicate/re-observation":2,"unresolved":3}
    status={uid:"distinct" for uid in result.content_uid}
    for audit in audits:
        for uid in (audit["left_content_uid"],audit["right_content_uid"]):
            if severity[audit["classification"]]>severity[status[uid]]: status[uid]=audit["classification"]
    result["duplicate_classification"]=result.content_uid.map(status)
    result["duplicate_group_id"]=result.content_uid.map(groups).fillna("")
    audit=pd.DataFrame(audits,columns=["left_content_uid","right_content_uid","left_release","right_release","left_category","right_category","left_veracity","right_veracity","classification","matching_evidence","shared_social_id_count","automatic_action"])
    columns=COMMON_COLUMNS+[c for c in result.columns if c not in COMMON_COLUMNS]
    return result[columns],audit

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True);DUP_OUT.parent.mkdir(parents=True,exist_ok=True)
    table,audit=build();table.to_csv(OUT,index=False);audit.to_csv(DUP_OUT,index=False)
    print(f"CoAID rows: {len(table):,}; duplicate candidate pairs: {len(audit):,}")
    return 0
if __name__=="__main__": raise SystemExit(main())
