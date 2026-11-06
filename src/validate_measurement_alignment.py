"""Validate cross-dataset measurement alignment without merging analytical rows."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from content_table_utils import COMMON_COLUMNS, clean_scalar, project_root, snowflake_time

ROOT=project_root(__file__);RAW=ROOT/"data/raw";PROCESSED=ROOT/"data/processed";OUT=ROOT/"outputs/tables"

def read_csv(path):
    for enc in ("utf-8","utf-8-sig","latin-1"):
        try:return pd.read_csv(path,encoding=enc)
        except UnicodeDecodeError:pass

def load_json(path):
    with path.open(encoding="utf-8") as f:return json.load(f)

def nonmissing(series):
    return series.notna() & series.astype(str).str.strip().ne("")

def all_social_ids():
    rows=[]
    coaid=RAW/"coaid/CoAID-master"
    for path in sorted(coaid.glob("*/*_tweets.csv")):
        frame=read_csv(path)
        for value in frame.get("tweet_id",[]):rows.append(("CoAID",path.parent.name,"post",value))
    for path in sorted(coaid.glob("*/*_tweets_replies.csv")):
        frame=read_csv(path)
        for value in frame.get("reply_id",[]):rows.append(("CoAID",path.parent.name,"reply",value))
        for value in frame.get("tweet_id",[]):
            if pd.notna(value):rows.append(("CoAID",path.parent.name,"reply_parent_reference",value))
    fh=RAW/"FakeHealth-master/dataset/engagements"
    for subset in ("HealthStory","HealthRelease"):
        mapping=load_json(fh/f"{subset}.json")
        for values in mapping.values():
            for kind,key in (("post","tweets"),("reply","replies"),("retweet","retweets")):
                for value in values.get(key,[]):rows.append(("FakeHealth",subset,kind,value))
    return rows

def snowflake_audit():
    details=[]
    for dataset,subset,kind,value in all_social_ids():
        ts,quality=snowflake_time(value);parsed=pd.Timestamp(ts) if ts else pd.NaT
        plausible=(pd.Timestamp("2019-01-01",tz="UTC")<=parsed<=pd.Timestamp("2021-01-31",tz="UTC")) if dataset=="CoAID" and pd.notna(parsed) else (pd.Timestamp("2009-01-01",tz="UTC")<=parsed<=pd.Timestamp("2020-12-31",tz="UTC")) if pd.notna(parsed) else False
        details.append({"dataset":dataset,"subset_or_release":subset,"id_type":kind,"social_id":str(value),"decoded_timestamp_utc":ts,"decode_quality":quality,"within_broad_plausible_collection_period":plausible})
    detail=pd.DataFrame(details);summary=[]
    for keys,g in detail.groupby(["dataset","subset_or_release","id_type"],dropna=False):
        parsed=pd.to_datetime(g.decoded_timestamp_utc,utc=True,errors="coerce",format="mixed");bad=g[(g.decode_quality!="valid")|(~g.within_broad_plausible_collection_period)]
        summary.append({"dataset":keys[0],"subset_or_release":keys[1],"id_type":keys[2],"total_ids":len(g),"unique_ids":g.social_id.nunique(),"decoded_n":parsed.notna().sum(),"decoded_percent":100*parsed.notna().mean(),"impossible_date_n":(g.decode_quality=="impossible_date").sum(),"outside_broad_plausible_collection_period_n":(~g.within_broad_plausible_collection_period).sum(),"minimum_timestamp_utc":parsed.min().isoformat() if parsed.notna().any() else "","maximum_timestamp_utc":parsed.max().isoformat() if parsed.notna().any() else "","median_timestamp_utc":parsed.median().isoformat() if parsed.notna().any() else "","suspicious_id_examples_json":json.dumps(bad.social_id.head(20).tolist()),"plausibility_rule":"CoAID 2019-01-01..2021-01-31; FakeHealth 2009-01-01..2020-12-31 (broad validation windows, not collection cutoffs)"})
    return pd.DataFrame(summary),detail

def publication_audit(tables):
    rows=[]
    for dataset,frame in tables.items():
        for subset,g in frame.groupby("source_release_or_subset"):
            pub=pd.to_datetime(g.publish_date_utc,utc=True,errors="coerce",format="mixed");first=pd.to_datetime(g.first_observed_post_time,utc=True,errors="coerce",format="mixed")
            raw_present=nonmissing(g.publish_date_raw)
            invalid=(g.publish_date_quality=="unparsable").sum();impl=(g.publish_date_quality=="implausible_range").sum()
            after=(pub.notna()&first.notna()&(pub>first))
            rows.append({"dataset":dataset,"subset_or_release":subset,"total_n":len(g),"valid_dates":(g.publish_date_quality=="valid").sum(),"missing_dates":(~raw_present).sum(),"unparsable_dates":invalid,"implausible_range_dates":impl,"dates_after_first_observed_post":after.sum(),"dated_items_with_posts":(pub.notna()&first.notna()).sum(),"minimum_valid_date_utc":pub[g.publish_date_quality=="valid"].min().isoformat() if (g.publish_date_quality=="valid").any() else "","maximum_valid_date_utc":pub[g.publish_date_quality=="valid"].max().isoformat() if (g.publish_date_quality=="valid").any() else "","imputation_performed":False})
    return pd.DataFrame(rows)

def missingness(tables):
    rows=[]
    for dataset,frame in tables.items():
        for subset,g in frame.groupby("source_release_or_subset"):
            for var in COMMON_COLUMNS:
                available=nonmissing(g[var]);rows.append({"variable":var,"dataset":dataset,"subset_or_release":subset,"total_n":len(g),"nonmissing_n":available.sum(),"missing_n":len(g)-available.sum(),"missing_percent":100*(len(g)-available.sum())/len(g)})
    return pd.DataFrame(rows)

DEFS={
"dataset":("constant CoAID","constant FakeHealth","yes","none","Dataset identity, not an outcome.","yes"),
"source_release_or_subset":("dated CoAID release","HealthStory/HealthRelease","no","retain as provenance/stratum","Release and subset are different constructs.","yes"),
"content_uid":("hash of release, raw file, original ID","hash of subset and original ID","yes","none","Synthetic stable identifier only.","yes"),
"content_category":("News or Claim from filename","HealthStory or HealthRelease","no","analyze/stratify, do not equate","Categories are not equivalent.","yes"),
"veracity_original":("Fake/Real filename category","original 0-5 rating representation","no","within-dataset grouping only","Ground-truth constructs differ.","yes"),
"title":("newstitle, fallback title","content title, fallback review titles","partial","document fallback","Field sources differ.","yes"),
"text":("content, fallback abstract","locally crawled text","partial","document fallback and missingness","Claims often lack bodies; crawlers differ.","sensitivity"),
"publish_date_raw":("raw publish_date","raw Unix-seconds publish_date","partial","parse to UTC separately","Missingness/precision differ.","yes"),
"publish_date_utc":("parsed UTC","parsed UTC","partial","validated parsing","Substantial FakeHealth missingness.","sensitivity"),
"publish_date_quality":("missing/unparsable/valid/range flag","same quality vocabulary","yes","none","Quality distribution differs.","yes"),
"observed_post_count":("rows mapped in *_tweets","length of tweets ID list","yes","none","Search strategies/windows differ.","yes"),
"unique_post_count":("distinct mapped tweet_id","distinct tweets-list ID","yes","none","Observation mechanisms differ.","yes"),
"reply_count":("reply rows mapped by news_id","length of replies ID list","partial","aggregate at content level","Only CoAID has explicit parent tweet field.","yes"),
"unique_reply_count":("distinct reply_id by content","distinct replies-list ID by content","partial","none","Parent attribution differs.","yes"),
"engagement_present":("any mapped post or reply","any mapped post or reply","yes","none","Zero may mean unobserved rather than truly none.","yes"),
"first_observed_post_time":("minimum decoded original-post ID","minimum decoded tweets-list ID","partial","Snowflake decode","Coverage/search timing differ.","sensitivity"),
"last_observed_post_time":("maximum decoded original-post ID","maximum decoded tweets-list ID","partial","Snowflake decode","Right censoring differs.","sensitivity"),
"observed_lifespan":("last-first original-post seconds","last-first original-post seconds","partial","Snowflake decode","Observed window, not full lifespan.","sensitivity"),
"publication_to_first_post_delay":("first post minus publication seconds","first post minus publication seconds","partial","requires both valid dates","Negative delays and date missingness require review.","sensitivity"),
"provenance":("JSON raw file/row/ID pointers","JSON review/content/engagement pointers","yes","none","Do not use as predictor.","yes")}

def alignment(tables,miss):
    rows=[]
    for var in COMMON_COLUMNS:
        cov={}
        for ds,frame in tables.items():cov[ds]=f"{nonmissing(frame[var]).sum()}/{len(frame)} ({100*nonmissing(frame[var]).mean():.2f}%)"
        cdef,fdef,comp,trans,caution,recommend=DEFS[var]
        rows.append({"variable":var,"CoAID_definition":cdef,"FakeHealth_definition":fdef,"CoAID_coverage":cov["CoAID"],"FakeHealth_coverage":cov["FakeHealth"],"directly_comparable":comp,"transformation_needed":trans,"major_caution":caution,"recommended_for_primary_analysis":recommend})
    return pd.DataFrame(rows)

def linkage_validation(tables):
    rows=[]
    # Recompute raw relationship totals and compare with processed sums.
    coaid_raw=RAW/"coaid/CoAID-master"
    for path in sorted(coaid_raw.glob("*/*.csv")):
        if "_tweets" in path.stem:continue
        release=path.parent.name;stem=path.stem;content=read_csv(path);tweets=read_csv(path.with_name(stem+"_tweets.csv"));replies=read_csv(path.with_name(stem+"_tweets_replies.csv"))
        cids={clean_scalar(int(x) if isinstance(x,float) and x.is_integer() else x) for x in content["Unnamed: 0"].dropna()}
        tweet_keys={clean_scalar(int(x) if isinstance(x,float) and x.is_integer() else x) for x in tweets["index"].dropna()};reply_keys={clean_scalar(int(x) if isinstance(x,float) and x.is_integer() else x) for x in replies["news_id"].dropna()}
        subset=tables["CoAID"][(tables["CoAID"].source_release_or_subset==release)&(tables["CoAID"].original_filename_category==stem)]
        rows.append({"dataset":"CoAID","subset_or_release":release,"table_family":stem,"raw_content_rows":len(content),"processed_content_rows":len(subset),"raw_post_rows":len(tweets),"processed_post_count_sum":subset.observed_post_count.sum(),"raw_reply_rows":len(replies),"processed_reply_count_sum":subset.reply_count.sum(),"post_mapping_keys_linked_to_content":len(tweet_keys&cids),"post_mapping_unique_keys":len(tweet_keys),"reply_mapping_keys_linked_to_content":len(reply_keys&cids),"reply_mapping_unique_keys":len(reply_keys),"counts_match_raw":len(content)==len(subset) and len(tweets)==subset.observed_post_count.sum() and len(replies)==subset.reply_count.sum()})
    fhraw=RAW/"FakeHealth-master/dataset"
    for subset_name in ("HealthStory","HealthRelease"):
        reviews=load_json(fhraw/"reviews"/f"{subset_name}.json");eng=load_json(fhraw/"engagements"/f"{subset_name}.json");frame=tables["FakeHealth"][tables["FakeHealth"].source_release_or_subset==subset_name]
        rids={str(x.get("news_id")) for x in reviews};eids=set(eng)
        rows.append({"dataset":"FakeHealth","subset_or_release":subset_name,"table_family":subset_name,"raw_content_rows":len(reviews),"processed_content_rows":len(frame),"raw_post_rows":sum(len(x.get("tweets",[])) for x in eng.values()),"processed_post_count_sum":frame.observed_post_count.sum(),"raw_reply_rows":sum(len(x.get("replies",[])) for x in eng.values()),"processed_reply_count_sum":frame.reply_count.sum(),"post_mapping_keys_linked_to_content":len(eids&rids),"post_mapping_unique_keys":len(eids),"reply_mapping_keys_linked_to_content":len(eids&rids),"reply_mapping_unique_keys":len(eids),"counts_match_raw":len(reviews)==len(frame) and sum(len(x.get("tweets",[])) for x in eng.values())==frame.observed_post_count.sum() and sum(len(x.get("replies",[])) for x in eng.values())==frame.reply_count.sum()})
    return pd.DataFrame(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tables={"CoAID":pd.read_csv(PROCESSED/"coaid_content_level.csv",dtype={"original_content_id":str}),"FakeHealth":pd.read_csv(PROCESSED/"fakehealth_content_level.csv",dtype={"original_content_id":str})}
    snow_summary,snow_detail=snowflake_audit();pub=publication_audit(tables);miss=missingness(tables);align=alignment(tables,miss);link=linkage_validation(tables)
    snow_summary.to_csv(OUT/"coaid_fakehealth_snowflake_audit.csv",index=False);snow_detail.to_csv(OUT/"coaid_fakehealth_snowflake_id_details.csv",index=False)
    pub.to_csv(OUT/"coaid_fakehealth_publication_date_audit.csv",index=False);miss.to_csv(OUT/"coaid_fakehealth_missingness.csv",index=False);align.to_csv(OUT/"coaid_fakehealth_measurement_alignment.csv",index=False);link.to_csv(OUT/"coaid_fakehealth_linkage_count_validation.csv",index=False)
    print(f"Validation rows: snowflake summary={len(snow_summary)}, publication={len(pub)}, missingness={len(miss)}, alignment={len(align)}, linkage={len(link)}")
    return 0
if __name__=="__main__":raise SystemExit(main())
