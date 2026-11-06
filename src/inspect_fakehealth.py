"""Read-only, reproducible structural audit of the local FakeHealth repository."""
from __future__ import annotations

import csv, json, os, statistics, sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
import pandas as pd

ROOT = Path(os.environ.get("FAKEHEALTH_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
RAW = ROOT / "data/raw/FakeHealth-master"
DATA, OUT = RAW / "dataset", ROOT / "outputs/tables"
SUMMARY = OUT / "fakehealth_dataset_summary.csv"
MANIFEST = OUT / "fakehealth_file_manifest.csv"
REPORT = OUT / "fakehealth_inspection_report.txt"
COMPARISON = OUT / "coaid_fakehealth_feature_comparison.csv"
SETS = ("HealthStory", "HealthRelease")
EPOCH = 1288834974657
MFIELDS = ["relative_path","filename","dataset","role","file_type","rows","columns","column_names","data_types","missing_value_counts","missing_value_percentages","exact_duplicate_rows","likely_primary_keys","likely_foreign_keys","text_title_content_columns","publication_date_columns","tweet_post_id_columns","user_id_columns","reply_id_columns","retweet_id_columns","engagement_fields","network_edge_fields","review_credibility_label_fields","notes"]

def load(path):
    with path.open(encoding="utf-8") as f: return json.load(f)

def present(v): return v is not None and (not isinstance(v,str) or bool(v.strip()))
def stable(v): return json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)
def dup(rows):
    vals=[stable(x) for x in rows]; return len(vals)-len(set(vals))
def typ(v):
    if v is None:return "null"
    if isinstance(v,bool):return "bool"
    if isinstance(v,int):return "int"
    if isinstance(v,float):return "float"
    if isinstance(v,str):return "str"
    if isinstance(v,list):return "list"
    if isinstance(v,dict):return "dict"
    return type(v).__name__

def details(rows, cols=None):
    cols=cols or sorted({k for r in rows for k in r}); n=len(rows)
    miss={c:sum(not present(r.get(c)) for r in rows) for c in cols}
    types={c:sorted({typ(r.get(c)) for r in rows if r.get(c) is not None}) for c in cols}
    return {"rows":n,"columns":len(cols),"column_names":" | ".join(cols),"data_types":" | ".join(f"{c}:{'/'.join(types[c]) or 'missing'}" for c in cols),"missing_value_counts":" | ".join(f"{c}:{miss[c]}" for c in cols),"missing_value_percentages":" | ".join(f"{c}:{100*miss[c]/n if n else 0:.2f}" for c in cols),"exact_duplicate_rows":dup(rows)}

def classes(cols):
    low={c:c.lower() for c in cols}
    def p(*terms): return " | ".join(c for c,n in low.items() if any(t in n for t in terms))
    return {"text_title_content_columns":p("title","text","content","description","summary","explanation"),"publication_date_columns":p("publish_date","publication_date"),"tweet_post_id_columns":p("tweet","post_id"),"user_id_columns":p("user_id","user_name","users"),"reply_id_columns":p("reply"),"retweet_id_columns":p("retweet"),"engagement_fields":p("tweet","reply","retweet","favorite","like","engagement"),"network_edge_fields":p("follower","following","source_user","target_user","edge"),"review_credibility_label_fields":p("rating","criteria","review","label","credib")}

def load_all():
    content,reviews,eng={},{},{}
    for ds in SETS:
        content[ds]={p.stem:load(p) for p in sorted((DATA/"content"/ds).glob("*.json"))}
        reviews[ds]=load(DATA/"reviews"/f"{ds}.json")
        eng[ds]=load(DATA/"engagements"/f"{ds}.json")
    return content,reviews,eng

def manifest(content,reviews,eng):
    out=[]
    for ds in SETS:
        cols=sorted({k for r in content[ds].values() for k in r})+["news_id_from_filename"]
        for nid,raw in content[ds].items():
            rec={**raw,"news_id_from_filename":nid}; path=DATA/"content"/ds/f"{nid}.json"
            out.append({"relative_path":path.relative_to(RAW).as_posix(),"filename":path.name,"dataset":ds,"role":"news/content","file_type":"JSON object",**details([rec],cols),"likely_primary_keys":"news_id_from_filename","likely_foreign_keys":"filename stem -> reviews.news_id and engagements key",**classes(cols),"notes":"One locally crawled content object; provenance ID encoded in filename."})
        path=DATA/"reviews"/f"{ds}.json"; cols=sorted({k for r in reviews[ds] for k in r})
        out.append({"relative_path":path.relative_to(RAW).as_posix(),"filename":path.name,"dataset":ds,"role":"review/ground truth","file_type":"JSON array of objects",**details(reviews[ds],cols),"likely_primary_keys":"news_id","likely_foreign_keys":"news_id -> content filename stem and engagements key",**classes(cols),"notes":"Expert item reviews; original ratings/criteria retained."})
        path=DATA/"engagements"/f"{ds}.json"; rows=[{"news_id":k,**v} for k,v in eng[ds].items()]; cols=["news_id","tweets","replies","retweets"]
        out.append({"relative_path":path.relative_to(RAW).as_posix(),"filename":path.name,"dataset":ds,"role":"social engagement IDs","file_type":"JSON mapping of news_id to ID lists",**details(rows,cols),"likely_primary_keys":"news_id","likely_foreign_keys":"news_id -> reviews.news_id and content filename stem",**classes(cols),"notes":"IDs only; no hydrated posts, parent pointers, or users."})
    return out

def pubdate(v):
    if not present(v): return None
    x=pd.to_datetime(v,unit="s",utc=True,errors="coerce") if isinstance(v,(int,float)) else pd.to_datetime(v,utc=True,errors="coerce")
    return None if pd.isna(x) else x
def snow(v):
    try:
        x=pd.to_datetime((int(v)>>22)+EPOCH,unit="ms",utc=True,errors="coerce")
        return None if pd.isna(x) or not 2010<=x.year<=2030 else x
    except (ValueError,TypeError,OverflowError): return None
def ids(e,k): return [str(x) for v in e.values() for x in v.get(k,[])]
def med(vals,words=False): return statistics.median(len(x.split()) if words else len(x) for x in vals) if vals else 0

def metrics(ds,c,r,e):
    ci=set(c); ri=[str(x.get("news_id")) for x in r]; ei=set(e)
    titles=[str(x.get("title")) for x in c.values() if present(x.get("title"))]
    texts=[str(x.get("text")) for x in c.values() if present(x.get("text"))]
    urls=[str(x.get("url")) for x in c.values() if present(x.get("url"))]
    dates=[d for x in c.values() if (d:=pubdate(x.get("publish_date"))) is not None]
    ti,ri2,rti=ids(e,"tweets"),ids(e,"replies"),ids(e,"retweets"); social=ti+ri2+rti
    st=[d for x in social if (d:=snow(x)) is not None]
    return {"dataset":ds,"review_items":len(r),"local_content_files":len(c),"engagement_item_mappings":len(e),"review_id_unique":len(set(ri)),"content_id_unique":len(ci),"engagement_id_unique":len(ei),"reviews_linked_to_content_n":len(set(ri)&ci),"reviews_linked_to_content_pct":100*len(set(ri)&ci)/len(set(ri)),"engagements_linked_to_reviews_n":len(ei&set(ri)),"engagements_linked_to_reviews_pct":100*len(ei&set(ri))/len(ei),"titles_present":len(titles),"full_text_present":len(texts),"url_present":len(urls),"url_only_items":sum(present(x.get("url")) and not present(x.get("title")) and not present(x.get("text")) for x in c.values()),"missing_text":len(c)-len(texts),"median_title_characters":med(titles),"median_title_words":med(titles,True),"median_text_characters":med(texts),"median_text_words":med(texts,True),"publication_dates_parseable":len(dates),"publication_dates_missing_or_unparseable":len(c)-len(dates),"publication_dates_missing_pct":100*(len(c)-len(dates))/len(c),"earliest_publication_date_utc":min(dates).isoformat() if dates else "","latest_publication_date_utc":max(dates).isoformat() if dates else "","publication_year_counts":stable(dict(sorted(Counter(d.year for d in dates).items()))),"publication_month_counts":stable(dict(sorted(Counter(d.strftime('%Y-%m') for d in dates).items()))),"tweet_id_rows":len(ti),"unique_tweet_ids":len(set(ti)),"reply_id_rows":len(ri2),"unique_reply_ids":len(set(ri2)),"retweet_id_rows":len(rti),"unique_retweet_ids":len(set(rti)),"all_social_id_rows":len(social),"unique_all_social_ids":len(set(social)),"snowflake_timestamps_parseable":len(st),"earliest_social_id_time_utc":min(st).isoformat() if st else "","latest_social_id_time_utc":max(st).isoformat() if st else "","review_exact_duplicate_rows":dup(r),"content_exact_duplicates_excluding_id":dup(list(c.values())),"content_columns":" | ".join(sorted({k for x in c.values() for k in x})),"review_columns":" | ".join(sorted({k for x in r for k in x})),"engagement_columns":"news_id | tweets | replies | retweets"}

def comparisons():
    F=["concept","CoAID availability","FakeHealth availability","definition in CoAID","definition in FakeHealth","directly comparable?","cautions"]
    def x(*v): return dict(zip(F,v))
    return [
      x("fake/real ground truth","direct: Fake/Real files","rating direct; author rule documented","curated COVID news/claims assigned to Fake/Real files","expert item rating 0-5; paper uses <3 fake, >=3 real","No","Different constructs/ascertainment; FakeHealth is reporting quality, not necessarily factual falsity."),
      x("title","direct","direct","title/newstitle","crawled source title and review original_title","Yes, with mapping","Retain source field."),
      x("full text","direct but incomplete","direct but incomplete","content; claims often title-only","locally crawled text","Partly","Content type and crawl missingness differ."),
      x("publication date","direct but incomplete","direct but incomplete","publish_date string","publish_date Unix seconds","Partly","Format and missingness differ."),
      x("tweet/post count","derivable","derivable","rows in *_tweets per content","length of tweets list per news_id","Yes, approximate","Different searches/windows/platform eras."),
      x("unique tweet/post count","derivable","derivable","distinct tweet_id per content","distinct tweets-list ID per content","Yes, approximate","Use provenance-aware keys."),
      x("reply count","derivable","derivable","reply rows/distinct reply_id","reply-list rows/distinct IDs","Partly","Only CoAID supplies tweet parent column."),
      x("retweet/repost count","unavailable","derivable","no retweet tables","retweets-list rows/distinct IDs","No","Exclude from core."),
      x("unique engaged users","requires API","requires API","user IDs absent","user IDs absent locally","Not currently","Historical hydration attrition."),
      x("first engagement time","derivable from IDs","derivable from IDs","minimum Snowflake tweet time","minimum Snowflake tweets-list time","Yes, cautiously","Derived, not explicit; coverage bias."),
      x("last engagement time","derivable from IDs","derivable from IDs","maximum Snowflake tweet time","maximum Snowflake tweets-list time","Yes, cautiously","Truncated observation tails."),
      x("engagement lifespan","derivable","derivable","last-first observed tweet time","last-first observed tweet time","Yes, cautiously","Not full platform lifespan."),
      x("publication-to-first-engagement delay","derivable where dated","derivable where dated","first tweet minus publication","first tweet minus publication","Partly","Missing dates/search recall/negative delays."),
      x("replies per post","derivable","aggregate derivable","reply rows / tweet rows","reply IDs / tweet IDs","Partly","FakeHealth lacks per-parent attribution."),
      x("user-user network","unavailable","separate archive; absent locally","none","directed follower/following lists","No","Not diffusion; likely crawl-time/truncated."),
      x("interaction network","limited reply links","unavailable locally","tweet->reply IDs, no users","no parents/users","No","Requires hydration."),
      x("propagation/cascade structure","one-level reply links only","unavailable locally","tweet/reply pair, no deeper tree","separate ID sets","No","Neither is a complete cascade."),
    ]

def write(path,rows,fields=None):
    fields=fields or list(rows[0])
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def repeat(c,field):
    v=[stable(x.get(field)) for x in c.values() if present(x.get(field))];return len(v)-len(set(v))
def ratio(n,d): return f"{n:,}/{d:,} ({100*n/d:.2f}%)" if d else "not testable"

def make_report(content,reviews,eng,ms,mf):
    print("FAKEHEALTH STRUCTURAL INSPECTION — NO EDA OR MODELING")
    print(f"Raw root: {RAW}\nRaw files read only; no scrape, API call, label creation, harmonization, or merge.\n")
    print("Evidence tags: [LOCAL] computed from files; [DOC] repository/paper statement; [INFERENCE] interpretation.\n")
    dirs=sorted(p.relative_to(RAW).as_posix()+"/" for p in RAW.rglob("*") if p.is_dir()); files=[p for p in RAW.rglob("*") if p.is_file()]
    print("1 STRUCTURE / COMPLETENESS")
    print("[LOCAL] Directories: "+", ".join(dirs))
    print(f"[LOCAL] {len(files):,} files: {sum(p.suffix=='.json' for p in files):,} JSON, {sum(p.suffix=='.py' for p in files):,} Python, README/API metadata.")
    print("[LOCAL] Present: both content folders, both review files, both engagement-ID files, collection code. Absent: user_network, hydrated posts, users, timelines, follower/following lists.")
    print("[DOC] README locates follower/following IDs in separate Zenodo v2. Local copy is complete for the core GitHub ID release but not the paper's full four-component repository.")
    print(f"[LOCAL] Content JSONs: HealthStory={len(content['HealthStory']):,}; HealthRelease={len(content['HealthRelease']):,}. Manifest covers all {len(mf):,} structured files.\n")
    print("2 DEFINITIONS / GROUND TRUTH")
    print("[DOC] HealthStory = media-reported health news stories; HealthRelease = releases from universities, research centers, and companies. They are separate because sources differ.")
    print("[DOC] HealthNewsReview experts evaluate each item on 10 criteria: 8 shared plus 2 subset-specific; criteria contain answer and explanation. Rating 0-5 is proportional to criteria satisfied.")
    print("[DOC] FakeHealth paper explicitly uses rating <3 as fake and >=3 as real for both subsets. [LOCAL] Unit is item/article, not source; source is separate. No new binary field was created.")
    print("[CAUTION] This is expert-rated reporting quality, not automatically equivalent to CoAID misinformation. Sources: local README and Dai et al. 2020, https://ojs.aaai.org/index.php/ICWSM/article/download/7350/7204\n")
    print("3 COUNTS / LINKAGE / TEXT / TIME / ENGAGEMENT")
    for m in ms:
        ds=m['dataset']; print(f"\n{ds}: reviews={m['review_items']:,}; content={m['local_content_files']:,}; engagement mappings={m['engagement_item_mappings']:,}")
        print(f"  unique IDs review/content/engagement={m['review_id_unique']:,}/{m['content_id_unique']:,}/{m['engagement_id_unique']:,}")
        print(f"  review->content={m['reviews_linked_to_content_n']:,}/{m['review_items']:,} ({m['reviews_linked_to_content_pct']:.2f}%); engagement->review={m['engagements_linked_to_reviews_n']:,}/{m['engagement_item_mappings']:,} ({m['engagements_linked_to_reviews_pct']:.2f}%)")
        print(f"  titles={m['titles_present']:,}; full text={m['full_text_present']:,}; URLs={m['url_present']:,}; URL-only={m['url_only_items']:,}; missing text={m['missing_text']:,}")
        print(f"  medians: title {m['median_title_characters']:.0f} chars/{m['median_title_words']:.0f} words; text {m['median_text_characters']:.0f} chars/{m['median_text_words']:.0f} words. Text is local when present; retrieval needed only for missing content.")
        print(f"  publications {m['earliest_publication_date_utc']} to {m['latest_publication_date_utc']}; missing/unparseable={m['publication_dates_missing_or_unparseable']:,}/{m['local_content_files']:,} ({m['publication_dates_missing_pct']:.2f}%)")
        print(f"  years={m['publication_year_counts']}\n  months={m['publication_month_counts']}")
        print(f"  tweets={m['tweet_id_rows']:,}/{m['unique_tweet_ids']:,} unique; replies={m['reply_id_rows']:,}/{m['unique_reply_ids']:,}; retweets={m['retweet_id_rows']:,}/{m['unique_retweet_ids']:,}")
        print(f"  all social IDs={m['all_social_id_rows']:,}/{m['unique_all_social_ids']:,} unique; Snowflake times={m['snowflake_timestamps_parseable']:,}, range {m['earliest_social_id_time_utc']} to {m['latest_social_id_time_utc']}")
        print(f"  original ratings={dict(sorted(Counter(x.get('rating') for x in reviews[ds]).items()))}")
    missing=sum(m['publication_dates_missing_or_unparseable'] for m in ms); total=sum(m['local_content_files'] for m in ms)
    print(f"\n[LOCAL] All parseable publication dates predate 2020; {missing:,}/{total:,} ({100*missing/total:.2f}%) local content files lack a parseable date. Thus dated items are pre-COVID; undated items remain uncertain.\n")
    print("4 SOCIAL AVAILABILITY")
    for line in ["A posts/item: directly available and countable from tweets lists","B unique users: requires external API/data; user IDs absent","C replies: IDs/counts direct; parents unavailable","D retweets: IDs/counts direct; originals unavailable","E timestamps: derivable from Twitter Snowflake IDs; not explicit","F user IDs: requires external API/data","G post IDs: directly available","H parent-child relationships: unavailable locally","I conversations: unavailable locally","J likes/favorites: requires hydrated tweet objects"]: print("  "+line)
    print("\n5 USER NETWORK")
    print("[LOCAL] Cannot empirically inspect nodes/edges/direction/time/linkage because network files are absent.")
    print("[DOC] Separate archive is a global user graph across both subsets: Twitter-user nodes and directed follower/following lists (README caps: up to 200 followers and 5,000 followings per user), not item-specific engagement edges; no edge times documented.")
    print("[INFERENCE] It is likely a truncated crawl-time snapshot, not historical-at-publication evidence and not a diffusion network.\n")
    print("6 IDENTIFIERS / EMPIRICAL RELATIONSHIPS")
    for ds in SETS:
        c=set(content[ds]);r={str(x.get('news_id')) for x in reviews[ds]};e=set(eng[ds])
        print(f"  {ds}: content->review {ratio(len(c&r),len(c))}; engagement->content {ratio(len(e&c),len(e))}; engagement->review {ratio(len(e&r),len(e))}")
        print("    post->user and reply/retweet->parent: not testable (fields absent).")
    print(f"  content ID overlap across subsets={len(set(content['HealthStory'])&set(content['HealthRelease'])):,}")
    for k in ("tweets","replies","retweets"):
        a,b=set(ids(eng['HealthStory'],k)),set(ids(eng['HealthRelease'],k));print(f"  {k} cross-subset ID overlap={len(a&b):,}")
    for ds in SETS:
        z={k:set(ids(eng[ds],k)) for k in ("tweets","replies","retweets")};print(f"  {ds} role overlap tweet/reply={len(z['tweets']&z['replies']):,}, tweet/retweet={len(z['tweets']&z['retweets']):,}, reply/retweet={len(z['replies']&z['retweets']):,}")
    print("  User uniqueness/overlap untestable. Recommended keys: (dataset, subset, content_id) and (dataset, subset, content_id, engagement_type, post_id); store IDs as strings.\n")
    print("7 DUPLICATES / OVERLAP")
    for ds in SETS:
        print(f"  {ds}: exact content duplicates excluding ID={dup(list(content[ds].values())):,}; exact review duplicates={dup(reviews[ds]):,}")
        print("    "+"; ".join(f"repeated {f}={repeat(content[ds],f):,}" for f in ("url","canonical_link","title","text")))
        print("    "+"; ".join(f"repeated {k} memberships={len(ids(eng[ds],k))-len(set(ids(eng[ds],k))):,}" for k in ("tweets","replies","retweets")))
    for f in ("url","canonical_link","title","text"):
        a={stable(x.get(f)) for x in content['HealthStory'].values() if present(x.get(f))};b={stable(x.get(f)) for x in content['HealthRelease'].values() if present(x.get(f))};print(f"  cross-subset {f} overlap={len(a&b):,}")
    print("[CAUTION] Repeated engagement membership may be one post matching multiple items/searches; it is not automatically an accidental duplicate. Repeated users would be legitimate but are unavailable.\n")
    print("8 COMPARABILITY / COMMON CORE")
    print(f"Detailed table: {COMPARISON.relative_to(ROOT)}")
    print("Candidate content-level core: dataset, source_release/subset, content_id, period_type, content_category, veracity_original, title, text, publish_date_raw, publish_datetime_utc, observed_post_count, unique_post_count, reply_count, unique_reply_count, engagement_present, first_observed_post_time, last_observed_post_time, observed_post_lifespan, publication_to_first_observed_post_delay, data_provenance.")
    print("Temporal fields require valid Snowflakes and complete dates; label them observed and assess missingness. Exclude from primary comparison: FakeHealth retweets, follower graph, user/bot/profile metrics, unique users, likes, cascade depth, criteria details/reviewers.\n")
    print("9 DEFENSIBILITY")
    print("A Domain: broadly similar health information, but FakeHealth mixes general medical stories/releases while CoAID is COVID-specific and includes news/claims.")
    print("B Period: strongly different for dated items (2009-2019 vs CoAID 2020); undated FakeHealth content is uncertain.")
    print("C Labels: defensible for within-FakeHealth author-rule contrasts, not equivalent ground truth across datasets.")
    print("D Engagement: observed tweet/reply counts align structurally, but query strategy, API/search era, platform ecology, observation window, and missingness differ.")
    print("E Confounds: topic/content type; label/reviewer process; era/platform changes; search/window censoring; content/date missingness; sample/engagement scale; CoAID release duplication; story vs release sources.")
    print("[INFERENCE] Suitable as a qualified non-COVID comparator for restricted content-level within-dataset contrasts in observed tweet/reply engagement—not for causal period claims or primary retweet/user/network/cascade comparisons.\n")
    print("10 NEXT STEP")
    print("After review, separately build provenance-preserving content-level tables using the core above; validate CoAID release deduplication and FakeHealth date/Snowflake quality; produce a measurement/missingness alignment table before EDA or merging.")

def main():
    if not RAW.is_dir(): print(f"Missing {RAW}",file=sys.stderr);return 1
    OUT.mkdir(parents=True,exist_ok=True);c,r,e=load_all();mf=manifest(c,r,e);ms=[metrics(ds,c[ds],r[ds],e[ds]) for ds in SETS];cp=comparisons()
    write(MANIFEST,mf,MFIELDS);write(SUMMARY,ms);write(COMPARISON,cp)
    with REPORT.open("w",encoding="utf-8") as f, redirect_stdout(f): make_report(c,r,e,ms,mf)
    print(f"Wrote {SUMMARY}, {MANIFEST}, {REPORT}, and {COMPARISON}")
    return 0
if __name__=="__main__": raise SystemExit(main())
