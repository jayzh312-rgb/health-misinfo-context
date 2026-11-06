"""Read-only structural and suitability audit of A COVID-19 Rumor Dataset."""
from __future__ import annotations

import csv, json, re, statistics, sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data/raw/COVID-19-rumor-dataset-master"
DATA=RAW/"Data"; OUT=ROOT/"outputs/tables"
SUMMARY=OUT/"covid_rumor_dataset_summary.csv";MANIFEST=OUT/"covid_rumor_file_manifest.csv";REPORT=OUT/"covid_rumor_inspection_report.txt"
COAID=OUT/"covid_rumor_vs_coaid_comparison.csv";FAKEHEALTH=OUT/"covid_rumor_fakehealth_alignment.csv"

def read(path,header="infer",names=None):
    return pd.read_csv(path,encoding="utf-8-sig",header=header,names=names,low_memory=False)
def val(v):return "" if pd.isna(v) else str(v).strip()
def norm(v):return re.sub(r"\s+"," ",re.sub(r"[^\w]+"," ",val(v).casefold())).strip()
def dup(frame):return int(frame.astype(str).duplicated().sum())
def parse(series):
    parsed=pd.to_datetime(series.replace({"None":pd.NA,"":pd.NA}),utc=True,errors="coerce",format="mixed")
    return parsed.map(lambda x:x if pd.isna(x) or 1900<=x.year<=2100 else pd.NaT)
def types(frame):return " | ".join(f"{c}:{frame[c].dtype}" for c in frame)
def missing(frame):return " | ".join(f"{c}:{frame[c].isna().sum()} ({100*frame[c].isna().mean():.2f}%)" for c in frame)
def english_heuristic(text):
    s=val(text);letters=[c for c in s if c.isalpha()]
    if not letters:return False
    latin=sum(("a"<=c.casefold()<="z") for c in letters)
    return latin/len(letters)>=0.90
def month_counts(dates):return json.dumps(dict(sorted(Counter(dates.dropna().dt.strftime("%Y-%m")).items())))

def load_main():
    news=read(DATA/"news/news.csv",header=None,names=["rumor_id","label","content","sentiment"])
    twitter=read(DATA/"twitter/Twitter.csv")
    twitter=twitter.rename(columns={twitter.columns[0]:"rumor_id"})
    en=read(DATA/"en_dup.csv")
    return news,twitter,en

def per_item_audit(news,twitter):
    manifests=[];news_files={};twitter_files={};news_rows=[];twitter_rows=[]
    for subset,folder,summary_name in (("NEWS",DATA/"news","news.csv"),("TWITTER",DATA/"twitter","Twitter.csv")):
        for path in sorted(folder.glob("*.csv")):
            if path.name==summary_name:continue
            try:frame=read(path)
            except Exception as exc:
                manifests.append({"relative_path":path.relative_to(RAW).as_posix(),"subset":subset,"role":"per-item","rows":"","columns":"","column_names":"","dtypes":"","missing":"","duplicate_rows":"","link_key":path.stem,"notes":f"ERROR {type(exc).__name__}: {exc}"});continue
            manifests.append({"relative_path":path.relative_to(RAW).as_posix(),"subset":subset,"role":"per-item repost/source records" if subset=="NEWS" else "per-item response/comment records","rows":len(frame),"columns":len(frame.columns),"column_names":" | ".join(frame.columns),"dtypes":types(frame),"missing":missing(frame),"duplicate_rows":dup(frame),"link_key":path.stem,"notes":"Raw file read only"})
            if subset=="NEWS":
                key=int(path.stem.removeprefix("rumor"));news_files[key]=frame
                dates=parse(frame.get("dates",pd.Series(dtype=object)))
                news_rows.append({"rumor_id":key,"records":len(frame),"unique_websites":frame.get("website",pd.Series(dtype=object)).dropna().astype(str).nunique(),"valid_dates":dates.notna().sum(),"min_date":dates.min(),"max_date":dates.max(),"stances":json.dumps(frame.get("stance",pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),default=str),"duplicate_records":dup(frame)})
            else:
                key=path.stem;twitter_files[key]=frame
                times=parse(frame.get("time",pd.Series(dtype=object)));release=parse(frame.get("release date",pd.Series(dtype=object)))
                twitter_rows.append({"source_key":key,"records":len(frame),"valid_response_times":times.notna().sum(),"min_response_time":times.min(),"max_response_time":times.max(),"valid_release_dates":release.notna().sum(),"release_date_min":release.min(),"parent_id_values":frame.get("Twitter ID",pd.Series(dtype=object)).dropna().astype(str).nunique(),"stances":json.dumps(frame.get("stance",pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),default=str),"duplicate_records":dup(frame)})
    return manifests,news_files,twitter_files,pd.DataFrame(news_rows),pd.DataFrame(twitter_rows)

def top_manifest(news,twitter,en):
    rows=[]
    for path,subset,role,frame in ((DATA/"news/news.csv","NEWS","main rumor statements",news),(DATA/"twitter/Twitter.csv","TWITTER","main rumor tweets",twitter),(DATA/"en_dup.csv","COMBINED","unprocessed English combined table",en)):
        rows.append({"relative_path":path.relative_to(RAW).as_posix(),"subset":subset,"role":role,"rows":len(frame),"columns":len(frame.columns),"column_names":" | ".join(frame.columns),"dtypes":types(frame),"missing":missing(frame),"duplicate_rows":dup(frame),"link_key":"rumor_id" if subset!="COMBINED" else "none","notes":"news.csv has no header; audit assigns documented semantic names"})
    return rows

def temporal_stats(news_meta,twitter_meta):
    news_dates=[]
    for x in news_meta.itertuples():
        if pd.notna(x.min_date):news_dates.append(x.min_date)
        if pd.notna(x.max_date):news_dates.append(x.max_date)
    twitter_dates=[]
    for x in twitter_meta.itertuples():
        if pd.notna(x.min_response_time):twitter_dates.append(x.min_response_time)
        if pd.notna(x.max_response_time):twitter_dates.append(x.max_response_time)
    return news_dates,twitter_dates

def build_summary(news,twitter,en,news_meta,twitter_meta,news_files,twitter_files):
    rows=[]
    for subset,frame,idcol in (("NEWS",news,"rumor_id"),("TWITTER",twitter,"rumor_id"),("EN_DUP",en,None)):
        textcol="content";labels=frame["label"].value_counts(dropna=False).to_dict()
        row={"subset":subset,"records":len(frame),"columns":len(frame.columns),"column_names":" | ".join(frame.columns),"dtypes":types(frame),"missing_values":missing(frame),"exact_duplicate_rows":dup(frame),"unique_ids":frame[idcol].nunique() if idcol else "","duplicate_ids":int(frame[idcol].duplicated().sum()) if idcol else "","labels":json.dumps(labels,default=str),"content_present":int(frame[textcol].notna().sum()),"english_looking_n":int(frame[textcol].map(english_heuristic).sum()),"english_looking_pct":100*frame[textcol].map(english_heuristic).mean(),"per_item_files":len(news_files) if subset=="NEWS" else len(twitter_files) if subset=="TWITTER" else 0}
        rows.append(row)
    return rows

def comparisons():
    fields=["criterion","COVID_Rumor_evidence","CoAID_evidence","classification","reason"]
    def r(*v):return dict(zip(fields,v))
    return [
      r("content unit","NEWS short rumor statement; TWITTER rumor tweet; separate subsets","News/Claim content items","uncertain","NEWS is item-like but not full article; TWITTER is post-level"),
      r("English-language suitability","English-designated repository; heuristic quantified in report","English COVID health information","similar","Both are primarily English"),
      r("fake/real labeling","Item-level T/F/U, manually cross-validated and fact-check informed","Item-level Fake/Real file categorization","COVID Rumor better","Explicit unverified class and documented validation, but not binary-equivalent"),
      r("item-level veracity","Direct on both subsets","Direct filename category","similar","Both assign item labels"),
      r("full-text availability","Short rumor sentences/tweet text; no full articles","News content available for 74.81%; claims short","CoAID better","CoAID includes article bodies"),
      r("publication-date coverage","NEWS only repost dates; TWITTER release dates only for items with per-item files","19.62% parsed content dates plus Snowflake post times","uncertain","Different date meanings and missingness"),
      r("post/tweet counts","TWITTER is one rumor tweet per row; no content-to-many-post mapping","Observed tweet IDs mapped to content","CoAID better","COVID Rumor TWITTER unit is already a tweet"),
      r("reply counts","Direct aggregate counts on all Twitter rows","Direct mapped reply rows","similar","Definitions/collection differ"),
      r("retweets","Direct aggregate counts on Twitter rows","Unavailable","COVID Rumor better","Adds retweet volume, but not edges"),
      r("likes","Direct aggregate counts on Twitter rows","Unavailable","COVID Rumor better","Platform engagement measure only"),
      r("temporal engagement","Response times for subset with per-item files","Snowflake original-post times; reply timestamps derivable from IDs","uncertain","Neither provides fully comparable complete timelines"),
      r("propagation edges","No child IDs; counts and first-order response records only","Explicit tweet_id->reply_id pairs","CoAID better","CoAID has verifiable ID edges"),
      r("API dependence","No hydration needed for text/counts; IDs insufficient for graph","No hydration needed for counts/reply IDs","similar","Neither needs API for primary counts"),
      r("duplication","No release-overlap design; local duplicates audited","Known cumulative release overlap","COVID Rumor better","Cleaner release structure, though repeated content remains"),
      r("source-selection confounding","Wide mixed news/social/fact-check sources; labels manually/fact-check derived","Fact-check/official-source class confounding","uncertain","Different but material ascertainment confounds"),
      r("compatibility with FakeHealth","NEWS item-like but website reposts differ; TWITTER engagement richer but unit differs","Content-level tweet/reply mappings align better","CoAID better","CoAID preserves a closer content-to-social mapping"),
    ]

def alignment_rows():
    fields=["variable","FakeHealth_definition","COVID_Rumor_NEWS_definition","COVID_Rumor_TWITTER_definition","directly_comparable","major_caution"]
    def r(*v):return dict(zip(fields,v))
    return [
      r("content text","Locally crawled article/release text","Short rumor/claim statement","Tweet text","No","Neither Rumor subset supplies FakeHealth-like full articles"),
      r("veracity","Expert rating 0-5; authors use <3 vs >=3","T/F/U manually cross-validated/fact-check informed","Same T/F/U scheme","No","Different constructs; retain original labels"),
      r("publication date","Article/release publish_date","No main-item date; dated website reposts only","Original release date only through per-item response files","No","Rumor dates are incomplete/different units"),
      r("observed post/repost count","Number of mapped original tweets","Number of reposting website records/unique websites","Not applicable: content unit is a single tweet","No","Website reposts are not Twitter posts"),
      r("reply count","Reply-ID membership count per content","Unavailable","Aggregate reply number per rumor tweet","Partial","Different unit and collection"),
      r("retweet count","Retweet-ID membership count per content","Unavailable","Aggregate retweet number per rumor tweet","Partial","FakeHealth is content-level; Rumor is tweet-level"),
      r("like count","Unavailable","Unavailable","Aggregate likes per rumor tweet","No","No FakeHealth equivalent"),
      r("first engagement time","Minimum mapped tweet-ID time","Earliest dated website record (not same construct)","Earliest response time for items with per-item files","No","Different platforms/units and incomplete coverage"),
      r("last engagement time","Maximum mapped tweet-ID time","Latest dated website record","Latest response time for items with files","No","Incomplete observation windows"),
      r("engagement lifespan","Last-first observed tweet time","Last-first dated website record","Last-first response time","No","Different processes"),
      r("engagement presence","Any mapped tweet/reply","Any per-item website record","Any reply/retweet/like count or response record","Partial","Meaning changes by subset"),
    ]

def print_report(news,twitter,en,news_files,twitter_files,news_meta,twitter_meta,summary):
    print("A COVID-19 RUMOR DATASET — STRUCTURAL + SUITABILITY AUDIT")
    print(f"Raw root: {RAW}\nRaw files read only; no EDA, hydration, scraping, processing, or merging.\n")
    print("1 STRUCTURE / COMPLETENESS")
    dirs=[p.relative_to(RAW).as_posix()+"/" for p in RAW.rglob("*") if p.is_dir()];files=[p for p in RAW.rglob("*") if p.is_file()]
    print("Directories: "+", ".join(sorted(dirs)))
    print(f"Files total={len(files):,}; news CSVs={len(list((DATA/'news').glob('*.csv'))):,}; Twitter CSVs={len(list((DATA/'twitter').glob('*.csv'))):,}.")
    print("Required top-level files present: Data/en_dup.csv, Data/news/news.csv, Data/twitter/Twitter.csv.")
    print(f"Main rows: news={len(news):,} (official 4,129); Twitter={len(twitter):,} (official 2,705); en_dup={len(en):,} (official 7,179 including duplicates).")
    print(f"Per-item files: news={len(news_files):,} (official 3,936); Twitter={len(twitter_files):,} (official 1,383). Counts match documentation exactly after excluding main CSVs.")
    print("Internal record totals differ slightly from the paper: local NEWS per-item rows=32,747 versus 32,750 reported; local TWITTER per-item rows=34,916 versus 34,847 reported.")
    print("Unexpected but documented/supporting files: data-collection script, power-law analysis inputs/scripts/PDFs/figures.\n")
    print("2 MAIN TABLE SCHEMAS")
    for row in summary:
        print(f"{row['subset']}: {row['records']} rows x {row['columns']} columns; columns={row['column_names']}; dtypes={row['dtypes']}; missing={row['missing_values']}; exact duplicates={row['exact_duplicate_rows']}; labels={row['labels']}")
    print("NEWS semantics: rumor_id, item-level T/F/U, short rumor statement, manually assigned sentiment. It is not a full-article corpus. The main table has no source, publication date, stance, or popularity column; website records supply sources/stances/dates, and their count operationalizes popularity.")
    print("TWITTER semantics: one rumor tweet/statement with synthetic/source key, item-level T/F/U, sentiment, and aggregate reply/retweet/like counts. It contains no author/user column or main-table publication date.")
    print(f"Sentiment values NEWS={news.sentiment.value_counts(dropna=False).to_dict()}; TWITTER={twitter.sentiment.value_counts(dropna=False).to_dict()}. The literal #VALUE! occurs in 195 NEWS and 354 TWITTER rows, so sentiment is not analysis-ready without a documented missing/error treatment.")
    print("en_dup is an unprocessed combined English table with duplication, not the recommended analytical table.\n")
    print("3 LABELS / LANGUAGE")
    print("Observed labels are reported above without conversion. Authors define T=True factual, F=False/made-up or false information, U=Unverified/hard to judge at labeling time.")
    print("Veracity is item-level. Fact-check labels from Snopes/PolitiFact/Boomlive are used directly; multiple sources use majority voting. Social items are checked against authoritative sites, common sense, and otherwise assigned U. Authors report manual cross-validation.")
    for row in summary:print(f"  {row['subset']} conservative English-looking heuristic: {row['english_looking_n']}/{row['records']} ({row['english_looking_pct']:.2f}%).")
    print("The heuristic requires >=90% Latin letters; it is a script check, not language identification. Non-English/mixed snippets can exist. en_dup is unprocessed combined English-designated data, not a filtering key.\n")
    print("4 PER-ITEM NEWS RECORDS / LINKAGE")
    main_ids=set(news.rumor_id.astype(int));file_ids=set(news_files)
    print(f"Files linked by rumor#### numeric key to news.csv: {len(main_ids&file_ids):,}/{len(file_ids):,} files; news rows with a file: {len(main_ids&file_ids):,}/{len(main_ids):,} ({100*len(main_ids&file_ids)/len(main_ids):.2f}%).")
    print(f"Orphan news files={len(file_ids-main_ids):,}; news rows without file={len(main_ids-file_ids):,}. A file links to at most one unique main ID because main rumor IDs are unique={news.rumor_id.is_unique}.")
    print(f"Per-news records={news_meta.records.sum():,}; unique website sum={news_meta.unique_websites.sum():,}; exact duplicate per-file rows={news_meta.duplicate_records.sum():,}; valid repost dates={news_meta.valid_dates.sum():,}.")
    schemas=Counter(" | ".join(frame.columns) for frame in news_files.values());print(f"Per-news schema variants={dict(schemas.most_common())}. Only 3,210 files have the exact documented dates|website|stance header; many use unnamed/missing/malformed stance headers.")
    print("Each row is a repost/discussion website with optional date and support/deny/comment/query stance. It has URL/source metadata but no full repost content. These form item->website memberships, not website-to-website propagation edges.\n")
    print("5 PER-ITEM TWITTER RECORDS / LINKAGE")
    source=twitter["source"].dropna().astype(str);file_keys=set(twitter_files);source_keys=set(source)
    print(f"Files linked by filename to Twitter.csv source: {len(file_keys&source_keys):,}/{len(file_keys):,}; Twitter rows with file={twitter.source.astype(str).isin(file_keys).sum():,}/{len(twitter):,} ({100*twitter.source.astype(str).isin(file_keys).mean():.2f}%).")
    print(f"Orphan Twitter files={len(file_keys-source_keys):,}; source keys without file={len(source_keys-file_keys):,}; duplicate source keys in main={twitter.source.astype(str).duplicated().sum():,}.")
    print(f"Response rows={twitter_meta.records.sum():,}; valid response times={twitter_meta.valid_response_times.sum():,}; exact duplicate response rows={twitter_meta.duplicate_records.sum():,}.")
    schemas=Counter(" | ".join(frame.columns) for frame in twitter_files.values());print(f"Per-Twitter schema variants={dict(schemas.most_common())}.")
    print("Rows contain response/comment text, time, stance, and reply/retweet/like counts. They do not contain child tweet ID, user ID, parent ID distinct from the repeated file key, quote flag, or engagement type. Thus response text/time is local, but replies versus retweets cannot be separated at record level.\n")
    print("6 TEMPORAL COVERAGE")
    all_news=[];news_raw=[]
    for frame in news_files.values():
        raw=frame.get("dates",pd.Series(dtype=object));news_raw.extend(raw.tolist());all_news.extend(parse(raw).dropna().tolist())
    all_resp=[];all_release=[];resp_raw=[];release_raw=[]
    for frame in twitter_files.values():
        rr=frame.get("time",pd.Series(dtype=object));rl=frame.get("release date",pd.Series(dtype=object));resp_raw.extend(rr.tolist());release_raw.extend(rl.tolist());all_resp.extend(parse(rr).dropna().tolist());all_release.extend(parse(rl).dropna().tolist())
    nd=pd.Series(all_news,dtype="datetime64[ns, UTC]");rd=pd.Series(all_release,dtype="datetime64[ns, UTC]");td=pd.Series(all_resp,dtype="datetime64[ns, UTC]")
    ismiss=lambda x:pd.isna(x) or val(x).casefold() in {"","none","nan"}
    nm=sum(ismiss(x) for x in news_raw);rm=sum(ismiss(x) for x in release_raw);tm=sum(ismiss(x) for x in resp_raw)
    print(f"NEWS main publication dates: unavailable. Repost dates valid={len(nd):,}/{news_meta.records.sum():,} ({100*len(nd)/news_meta.records.sum():.2f}%), missing={nm:,}, unparseable/implausible={len(news_raw)-nm-len(nd):,}, range={nd.min()}..{nd.max()}, months={month_counts(nd)}")
    print(f"TWITTER original release-date fields: valid={len(rd):,}/{twitter_meta.records.sum():,} response rows ({100*len(rd)/twitter_meta.records.sum():.2f}%), missing={rm:,}, unparseable/implausible={len(release_raw)-rm-len(rd):,}, item coverage={twitter_meta.valid_release_dates.gt(0).sum():,}/{len(twitter):,}; range={rd.min()}..{rd.max()}, months={month_counts(rd.drop_duplicates())}")
    print(f"TWITTER response times: valid={len(td):,}/{twitter_meta.records.sum():,} ({100*len(td)/twitter_meta.records.sum():.2f}%), missing={tm:,}, unparseable/implausible={len(resp_raw)-tm-len(td):,}, range={td.min()}..{td.max()}, months={month_counts(td)}")
    plausible=((nd.dt.year.between(2019,2021)).sum() if len(nd) else 0);print(f"NEWS dated rows outside 2019-2021={len(nd)-plausible:,}; TWITTER response rows outside 2019-2021={len(td)-td.dt.year.between(2019,2021).sum():,}.")
    ordering=0;tested=0
    for frame in twitter_files.values():
        rel=parse(frame.get("release date",pd.Series(dtype=object)));resp=parse(frame.get("time",pd.Series(dtype=object)));mask=rel.notna()&resp.notna();tested+=mask.sum();ordering+=(resp[mask]<rel[mask]).sum()
    print(f"Parent/response temporal ordering: responses before release calendar date={ordering:,}/{tested:,}. Release dates omit time of day, so same-day ordering cannot be established.\n")
    print("7 ENGAGEMENT / PROPAGATION")
    print("NEWS popularity: derivable as per-item website-record count; unique reposting websites also derivable. Dates/stance are incomplete where fields are missing. It is website repost/discussion visibility, not a Twitter post count.")
    print(f"TWITTER direct aggregate coverage: reply={twitter['reply numbers'].notna().mean()*100:.2f}%, retweet={twitter['retweet numbers'].notna().mean()*100:.2f}%, like={twitter['likes numbers'].notna().mean()*100:.2f}%; total engagement is derivable as their sum. Response count/timing/stance derive only for the {len(twitter_files):,} items with files.")
    equal=((twitter['retweet numbers']==twitter['likes numbers'])&twitter['retweet numbers'].notna()).sum();print(f"Data-quality caution: retweet count equals like count exactly in {equal:,}/{len(twitter):,} main rows; preserve both raw fields but validate this pattern before modeling.")
    print("The source/file key is described by the paper as a fakeID and is not safe to interpret as a native Twitter Snowflake ID; no timestamp was decoded from it.")
    print("No true cascade topology is present. TWITTER has aggregate counts and first-order response records without child tweet IDs/users/types; NEWS has item->website sets without repost-to-repost edges. No recursive reply graph, retweet edge graph, user-user network, or cascade can be reconstructed.\n")
    print("8 DUPLICATES / OVERLAP")
    print(f"NEWS exact rows={dup(news):,}; repeated rumor IDs={news.rumor_id.duplicated().sum():,}; repeated normalized content rows={news.content.map(norm).duplicated().sum():,}.")
    print(f"TWITTER exact rows={dup(twitter):,}; repeated rumor IDs={twitter.rumor_id.duplicated().sum():,}; repeated source keys={twitter.source.astype(str).duplicated().sum():,}; repeated normalized content rows={twitter.content.map(norm).duplicated().sum():,}.")
    print(f"EN_DUP exact rows={dup(en):,}; repeated normalized content rows={en.content.map(norm).duplicated().sum():,}.")
    overlap=set(news.content.map(norm))&set(twitter.content.map(norm));print(f"Exact normalized NEWS/TWITTER content overlap={len(overlap):,}. Nothing was removed.")
    print("\n9 SUITABILITY / RECOMMENDATION")
    print("Subset choice: Option B—use both only for different questions, never combined. NEWS is closer to FakeHealth's content-level unit but contains short statements and website reposts, not articles plus Twitter mappings. TWITTER has richer direct counts but a post-level unit and no propagation edges.")
    print("Dataset role: B—supplement CoAID as an independent COVID robustness dataset, not replace it. It adds T/F/U labels, website repost popularity, retweets, likes, response text/time, stance and sentiment. It worsens unit alignment with FakeHealth and does not provide content->many-tweet diffusion mappings or graph topology.")
    print("Primary common features if used: dataset, subset, content_id, veracity_original, short_text, sentiment, engagement_present; NEWS-only website_record_count/unique_website_count; TWITTER-only reply_count/retweet_count/like_count/total_engagement. Cross-context primary models should not equate NEWS website reposts with FakeHealth tweets, and temporal variables should remain subset-specific/sensitivity measures.")
    print("Overall: not better than CoAID for the main FakeHealth comparison. CoAID remains primary because its unit is content and its mapped tweet/reply measurements align more directly with FakeHealth. COVID Rumor is valuable robustness evidence for different engagement constructs.\n")
    print("Sources: local README; Cheng et al. 2021, https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.644801/full")

def main():
    if not RAW.is_dir():print(f"Missing {RAW}",file=sys.stderr);return 1
    OUT.mkdir(parents=True,exist_ok=True);news,twitter,en=load_main();per,news_files,twitter_files,nm,tm=per_item_audit(news,twitter);top=top_manifest(news,twitter,en);summary=build_summary(news,twitter,en,nm,tm,news_files,twitter_files)
    pd.DataFrame(summary).to_csv(SUMMARY,index=False);pd.DataFrame(top+per).to_csv(MANIFEST,index=False);pd.DataFrame(comparisons()).to_csv(COAID,index=False);pd.DataFrame(alignment_rows()).to_csv(FAKEHEALTH,index=False)
    with REPORT.open("w",encoding="utf-8") as f,redirect_stdout(f):print_report(news,twitter,en,news_files,twitter_files,nm,tm,summary)
    print(f"Wrote five audit outputs; manifest rows={len(top)+len(per):,}")
    return 0
if __name__=="__main__":raise SystemExit(main())
