"""Build a reviewed-item FakeHealth table without harmonizing its labels."""
from __future__ import annotations

import json
import pandas as pd

from content_table_utils import COMMON_COLUMNS, clean_scalar, json_text, parse_date, project_root, stable_uid, timing

ROOT=project_root(__file__)
RAW=ROOT/"data/raw/FakeHealth-master/dataset"
OUT=ROOT/"data/processed/fakehealth_content_level.csv"
SECONDARY=ROOT/"data/processed/fakehealth_secondary_engagement.csv"

def load(path):
    with path.open(encoding="utf-8") as f:return json.load(f)

def build():
    rows=[];secondary=[]
    for subset in ("HealthStory","HealthRelease"):
        reviews=load(RAW/"reviews"/f"{subset}.json"); engagements=load(RAW/"engagements"/f"{subset}.json")
        content_dir=RAW/"content"/subset
        for review_index,review in enumerate(reviews):
            oid=str(review.get("news_id"));content_path=content_dir/f"{oid}.json";content=load(content_path) if content_path.exists() else {}
            engagement=engagements.get(oid,{})
            post_ids=[str(x) for x in engagement.get("tweets",[])];reply_ids=[str(x) for x in engagement.get("replies",[])];retweet_ids=[str(x) for x in engagement.get("retweets",[])]
            date_raw=content.get("publish_date");date_utc,date_quality=parse_date(date_raw)
            rating=review.get("rating"); author_class="fake_rating_lt_3" if rating is not None and rating<3 else "real_rating_ge_3" if rating is not None else "missing_rating"
            uid="fakehealth_"+stable_uid(subset,oid)
            provenance={"raw_review_file":(RAW/"reviews"/f"{subset}.json").relative_to(ROOT).as_posix(),"raw_review_array_index":review_index,"raw_content_file":content_path.relative_to(ROOT).as_posix() if content_path.exists() else "","raw_engagement_file":(RAW/"engagements"/f"{subset}.json").relative_to(ROOT).as_posix(),"engagement_mapping_key":oid,"original_content_id":oid}
            title=clean_scalar(content.get("title")) or clean_scalar(review.get("original_title")) or clean_scalar(review.get("title"))
            text=clean_scalar(content.get("text"))
            rows.append({"dataset":"FakeHealth","source_release_or_subset":subset,"content_uid":uid,"content_category":subset,"veracity_original":f"rating={rating}" if rating is not None else "rating=missing","title":title,"text":text,"publish_date_raw":clean_scalar(date_raw),"publish_date_utc":date_utc,"publish_date_quality":date_quality,"observed_post_count":len(post_ids),"unique_post_count":len(set(post_ids)),"reply_count":len(reply_ids),"unique_reply_count":len(set(reply_ids)),"engagement_present":bool(post_ids or reply_ids),**timing(post_ids,date_utc),"provenance":json_text(provenance),"original_content_id":oid,"original_review_rating":rating,"fakehealth_author_classification":author_class,"review_title":clean_scalar(review.get("title")),"review_original_title":clean_scalar(review.get("original_title")),"review_link":clean_scalar(review.get("link")),"source_link":clean_scalar(review.get("source_link")),"news_source":clean_scalar(review.get("news_source")),"review_category":clean_scalar(review.get("category")),"review_description":clean_scalar(review.get("description")),"reviewers_json":json_text(review.get("reviewers",[])),"review_tags_json":json_text(review.get("tags",[])),"review_summary_json":json_text(review.get("summary",{})),"review_criteria_json":json_text(review.get("criteria",[])),"content_available":content_path.exists()})
            secondary.append({"dataset":"FakeHealth","subset":subset,"content_uid":uid,"original_content_id":oid,"retweet_count":len(retweet_ids),"unique_retweet_count":len(set(retweet_ids)),"retweet_ids_json":json_text(retweet_ids),"note":"FakeHealth-only; excluded from primary common core because CoAID lacks equivalent raw retweet mappings."})
    frame=pd.DataFrame(rows);columns=COMMON_COLUMNS+[c for c in frame.columns if c not in COMMON_COLUMNS]
    return frame[columns],pd.DataFrame(secondary)

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True);table,secondary=build();table.to_csv(OUT,index=False);secondary.to_csv(SECONDARY,index=False)
    print(f"FakeHealth rows: {len(table):,}; secondary engagement rows: {len(secondary):,}")
    return 0
if __name__=="__main__":raise SystemExit(main())
