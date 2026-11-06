"""Audit captured content versions, without rebuilding data or changing prior outputs.

Run: .venv/bin/python src/audit_coaid_content_version_stability.py
TF-IDF is used only for transparent within-resource similarity, not prediction.
Stored bodies are excerpts, so classifications are qualified evidence judgments.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from audit_coaid_entity_resolution import ROOT, js, md_table, specific_url, text_error, url_key
from content_table_utils import clean_scalar, normalize_text

TABLES = ROOT / "outputs/tables"
REPORT = ROOT / "outputs/reports/coaid_content_version_stability.md"
SEED = 42
LABELS = ["A. SAME CONTENT", "B. MINOR UPDATE", "C. MATERIAL CONTENT UPDATE",
          "D. REPURPOSED RESOURCE", "E. INSUFFICIENT EVIDENCE"]
PREVIOUS = [TABLES / f"coaid_entity_resolution_{name}.csv" for name in
            ("validation", "member_evidence", "proposed_groups")]
RELEASES = ["05-01-2020", "07-01-2020", "09-01-2020", "11-01-2020"]
FIELDS = ["title", "newstitle", "content", "abstract", "news_url", "fact_check_url", "publish_date",
          "archive", "type", "meta_keywords", "news_url2", "news_url3", "news_url4", "news_url5"]

# Qualitative review of the stored excerpts, not completed researcher review.
# Fingerprints bind each decision to the exact evidence inspected in this audit;
# changed data fails explicitly rather than silently inheriting a stale decision.
EXCERPT_REVIEW = {
    "coaid_news_entity_f2de5df0b3a8172d293ebd3f": (
        "3a4cd7245618501a78d6bef7f5bd223a754ac62423c4dc71aaaec7e42dbc8661", LABELS[1],
        "The three introductions retain the same outbreak background/Wuhan narrative. Wording and embedded last-updated dates change, but no material claim change is visible in these excerpts. Broader FAQ changes remain unobserved."),
    "coaid_news_entity_00c4b938ae668289032087f5": (
        "4a38fcecec36d3866e8b1d88c23b19d4250c7cd4d3422f43c3c93f3736108e9d", LABELS[4],
        "A healthcare-professional index yields a MIS-C reporting block in one capture and a Q&A/contact block in another. Different selected blocks do not establish a material version or repurposing of the entire index."),
    "coaid_news_entity_6d17d76a36f97aca68de4f88": (
        "ca27de46465d650bd2f98c4aa07f68b4c12006293254abcfafc37898f2b18621", LABELS[4],
        "Park guidance excerpts select a preparation section, a checklist, and an introductory physical-activity section. The available text does not distinguish page editing from extraction of different sections."),
    "coaid_news_entity_26dabeeb366c6fa0f7f8e6c6": (
        "8e0cd8af7b9765a5a38ce0e8df0849e4e1e63be57e92828b67d0435f2ceba257", LABELS[4],
        "Older-adult guidance captures a care-plan section versus an infection-prevention section. Broad abstract differences do not establish a changed recommendation rather than a changed excerpt region."),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tok(value):
    return set(value.split())


def jac(a, b):
    a, b = tok(a), tok(b)
    return len(a & b) / len(a | b) if a and b else np.nan


def edit(a, b):
    return SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else np.nan


def behavior(key):
    host, path = urlsplit("https://" + key).netloc, urlsplit("https://" + key).path.lower()
    government = host.endswith(".gov") or host.endswith("who.int")
    if key.startswith(("twitter.com/status/", "facebook.com/")):
        return "social post", government
    if "/media/releases/" in path or "/news-events/press-announcements/" in path:
        return "dated institutional article", government
    if government and any(s in path for s in ("cases-updates", "covid-data", "covidview", "surveillance", "data-tracker", "mortality-report")):
        return "dashboard/surveillance (likely evolving)", government
    if government and (path.endswith("/index.html") or any(s in path for s in ("resources", "publications", "communication"))):
        return "index/resource page (likely maintained)", government
    if government:
        return "institutional guidance (likely maintained)", government
    return "article/news page (candidate)", government


def load_members():
    validation = pd.read_csv(PREVIOUS[0])
    prior_members = pd.read_csv(PREVIOUS[1], keep_default_na=False)
    prior_groups = pd.read_csv(PREVIOUS[2])
    frames, hashes = {}, {p.relative_to(ROOT).as_posix(): digest(p) for p in PREVIOUS}
    records = {}
    for row in prior_members.itertuples():
        relative = row.raw_content_file
        if relative not in frames:
            path = ROOT / relative
            hashes[relative] = digest(path)
            frames[relative] = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        raw = frames[relative].iloc[int(row.raw_row_index)]
        assert str(raw["Unnamed: 0"]) == str(row.original_id)
        record = {field: clean_scalar(raw.get(field)) for field in FIELDS}
        record.update({"member_uid": row.member_uid, "current_entity_uid": row.entity_uid,
                       "release": row.release, "original_id": str(row.original_id),
                       "veracity": row.veracity, "raw_content_file": relative, "raw_row_index": int(row.raw_row_index),
                       "resource_key": url_key(record["news_url"]),
                       "post_ids": set(json.loads(row.observed_post_ids_json)),
                       "reply_ids": set(json.loads(row.observed_reply_ids_json)),
                       "prior_boilerplate_body": bool(row.boilerplate_text is True or str(row.boilerplate_text).lower() == "true")})
        for field in ("title", "newstitle", "content", "abstract", "meta_keywords"):
            record["norm_" + field] = normalize_text(record[field])
        records[row.member_uid] = record
    # Template frequency across all audited sources. These are diagnostic
    # flags; a matching short footer never becomes affirmative content identity.
    repeated = {field: defaultdict(set) for field in ("newstitle", "abstract")}
    for m in records.values():
        for field in repeated:
            repeated[field][m["norm_" + field]].add(m["resource_key"])
    for m in records.values():
        title, abstract, body = m["norm_newstitle"], m["norm_abstract"], m["norm_content"]
        m["generic_scraped_title"] = bool(len(tok(title)) < 3 or text_error(title) or len(repeated["newstitle"][title]) >= 3)
        m["body_informative"] = bool(len(tok(body)) >= 25 and len(body) >= 150 and not m["prior_boilerplate_body"] and not text_error(body))
        abstract_template = len(repeated["abstract"][abstract]) >= 3 and len(tok(abstract)) < 60
        m["abstract_informative"] = bool(len(tok(abstract)) >= 12 and len(abstract) >= 80 and not text_error(abstract) and not abstract_template)
        m["body_excerpt_length"] = len(m["content"])
        m["body_likely_truncated"] = len(m["content"]) >= 450
        m["publication_date_nonempty"] = bool(m["publish_date"])
        m["resource_behavior"], m["government_institutional"] = behavior(m["resource_key"])
    return validation, prior_groups, records, hashes


def candidates(validation, prior_groups, records):
    units, inventory = [], []
    for row in validation.itertuples():
        if row.validation_category.startswith(("D.", "E.")):
            continue
        members = [x["uid"] for x in json.loads(row.original_ids_json)]
        units.append({"resource_uid": row.entity_uid, "current_entity_uid": row.entity_uid,
                      "members": sorted(members), "cohort": "same_resource_primary" if row.validation_category.startswith("A.") else "unresolved_identity_context",
                      "prior_entity_category": row.validation_category,
                      "prior_source_title_inconsistency": bool(row.same_url_source_metadata_inconsistent)})
    for row in prior_groups.itertuples():
        members = sorted(json.loads(row.members_json))
        releases = {records[u]["release"] for u in members}
        eligible = len(members) > 1 and len(releases) > 1
        inventory.append({"current_entity_uid": row.current_entity_uid, "proposed_resource_uid": row.proposed_subgroup,
                          "member_count": len(members), "member_uids_json": js(members),
                          "release_count": len(releases), "eligible_cross_release_version_comparison": eligible,
                          "notes": "Corrected-resource control proposal from prior audit; not applied. Single observations and within-release-only repeats cannot establish cross-release version stability."})
        if eligible:
            assert len({records[u]["resource_key"] for u in members}) == 1
            units.append({"resource_uid": row.proposed_subgroup, "current_entity_uid": row.current_entity_uid,
                          "members": members, "cohort": "corrected_resource_control",
                          "prior_entity_category": "E. DEFINITELY OVER-MERGED (parent only)",
                          "prior_source_title_inconsistency": False})
    assert sum(u["cohort"] == "same_resource_primary" for u in units) == 185
    assert sum(u["cohort"] == "unresolved_identity_context" for u in units) == 6
    return units, inventory


class Similarity:
    def __init__(self, records):
        self.lookup, docs = {}, []
        for uid, m in sorted(records.items()):
            for field in ("content", "abstract"):
                self.lookup[uid, field] = len(docs)
                docs.append(m["norm_" + field] if m["body_informative" if field == "content" else "abstract_informative"] else "")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True, norm="l2", min_df=1)
        self.matrix = self.vectorizer.fit_transform(docs)

    def cosine(self, a, b, field):
        x = self.matrix[self.lookup[a, field]]
        y = self.matrix[self.lookup[b, field]]
        return float(x.multiply(y).sum()) if x.nnz and y.nnz else np.nan


def classify_pair(p):
    """Multi-signal audit rubric; no engagement/count fields are used here."""
    if not p["same_resource_key"]:
        return LABELS[4], "Resource identity unresolved across different URLs; establish alias/translation identity before judging versions.", "low"
    if not p["both_bodies_informative"]:
        return LABELS[4], "At least one body is missing, boilerplate, or too short; titles/abstracts alone do not establish version equivalence.", "low"
    body_close = sum([p["content_cosine"] >= .88, p["content_edit_similarity"] >= .83, p["content_token_jaccard"] >= .75]) >= 2
    abstract_compatible = not p["both_abstracts_informative"] or p["abstract_exact"] or p["abstract_cosine"] >= .80
    title_compatible = p["scraped_title_exact"] or p["either_scraped_title_generic"] or p["scraped_title_jaccard"] >= .70
    distinct_title = not p["either_scraped_title_generic"] and p["scraped_title_jaccard"] < .25
    abstract_divergent = p["both_abstracts_informative"] and p["abstract_cosine"] < .55 and p["abstract_token_jaccard"] < .50
    if distinct_title and abstract_divergent and p["content_cosine"] < .25 and p["content_token_jaccard"] < .20:
        return LABELS[3], "Specific scraped title, informative abstract, and body all shift sharply; candidate repurposing requires archive confirmation.", "medium"
    if p["content_exact"] and abstract_compatible:
        if p["all_available_metadata_equal"] and (not p["both_abstracts_informative"] or p["abstract_exact"]):
            return LABELS[0], "Informative stored body and available metadata/abstract agree exactly after normalization; full-page identity remains unverified.", "medium"
        return LABELS[1], "Informative body is unchanged; differences are confined to metadata or a compatible abstract. Source-title disagreement alone is not a content update.", "medium"
    if p.get("both_retired_resource_notices", False) and abstract_compatible:
        return LABELS[1], "Both captures explicitly retain the same retired-data/reference notice; shortened wording and an updated destination label do not demonstrate a new substantive version.", "medium"
    numeric_update = p["dashboard_page"] and p["substantive_numbers_changed"] and not p["content_exact"]
    substantial_body = sum([p["content_cosine"] < .75, p["content_edit_similarity"] < .70, p["content_token_jaccard"] < .60]) >= 2
    independent_signal = abstract_divergent or distinct_title or p["explicit_update_language"] or (p["maintained_page"] and p["publication_date_changed"])
    if numeric_update or substantial_body and independent_signal:
        return LABELS[2], "Substantial excerpt change plus abstract/title/update-language/date evidence, or surveillance-number changes; propose versions pending full-capture review.", "medium"
    if body_close and abstract_compatible and title_compatible:
        return LABELS[1], "At least two body similarity measures support small edits, with compatible title and abstract evidence.", "medium"
    if p["content_cosine"] >= .75 and p["content_edit_similarity"] >= .65 and p["both_abstracts_informative"] and p["abstract_cosine"] >= .90 and title_compatible:
        return LABELS[1], "Moderately edited excerpt retains a near-identical informative abstract and compatible title.", "medium"
    return LABELS[4], "Excerpts differ but corroborating version evidence is inadequate; a changed excerpt/capture region cannot alone prove a material page update.", "low"


def pair(a, b, similarity):
    both_body = a["body_informative"] and b["body_informative"]
    both_abstract = a["abstract_informative"] and b["abstract_informative"]
    body_a, body_b = a["norm_content"], b["norm_content"]
    numbers = [set(re.findall(r"\b\d+(?:[,.]\d+)*\b", body)) - {"2019", "2020", "19"} for body in (body_a, body_b)]
    update_phrases = ("guidance has been updated", "guidance has been revised", "summary of changes", "updated recommendations",
                      "restart elective", "no longer recommended", "new recommendations", "changes made", "updated guidance")
    p = {"left_member_uid": a["member_uid"], "right_member_uid": b["member_uid"],
         "left_release": a["release"], "right_release": b["release"],
         "cross_release_pair": a["release"] != b["release"],
         "same_resource_key": a["resource_key"] == b["resource_key"],
         "scraped_title_exact": bool(a["norm_newstitle"] and a["norm_newstitle"] == b["norm_newstitle"]),
         "scraped_title_jaccard": jac(a["norm_newstitle"], b["norm_newstitle"]),
         "scraped_title_edit_similarity": edit(a["norm_newstitle"], b["norm_newstitle"]),
         "source_title_exact": bool(a["norm_title"] and a["norm_title"] == b["norm_title"]),
         "source_title_jaccard": jac(a["norm_title"], b["norm_title"]),
         "either_scraped_title_generic": a["generic_scraped_title"] or b["generic_scraped_title"],
         "both_bodies_informative": both_body, "both_abstracts_informative": both_abstract,
         "content_exact": bool(body_a and body_a == body_b),
         "content_cosine": similarity.cosine(a["member_uid"], b["member_uid"], "content"),
         "content_edit_similarity": edit(body_a, body_b), "content_token_jaccard": jac(body_a, body_b),
         "abstract_exact": bool(a["norm_abstract"] and a["norm_abstract"] == b["norm_abstract"]),
         "abstract_cosine": similarity.cosine(a["member_uid"], b["member_uid"], "abstract"),
         "abstract_token_jaccard": jac(a["norm_abstract"], b["norm_abstract"]),
         "keyword_jaccard": jac(a["norm_meta_keywords"], b["norm_meta_keywords"]),
         "publication_date_changed": bool(a["publish_date"] and b["publish_date"] and a["publish_date"] != b["publish_date"]),
         "same_factcheck_key": bool(a["fact_check_url"] and url_key(a["fact_check_url"]) == url_key(b["fact_check_url"])),
         "same_archive_url": bool(a["archive"] and a["archive"] == b["archive"]),
         "type_changed": a["type"] != b["type"],
         "dashboard_page": "dashboard" in a["resource_behavior"],
         "maintained_page": a["government_institutional"] and a["resource_behavior"] != "dated institutional article",
         "substantive_numbers_changed": bool(numbers[0] and numbers[1] and numbers[0] != numbers[1]),
         "explicit_update_language": any(phrase in body_a or phrase in body_b for phrase in update_phrases),
         "both_retired_resource_notices": all("no longer being updated" in body and "reference" in body for body in (body_a, body_b)),
         "all_available_metadata_equal": all(normalize_text(a[field]) == normalize_text(b[field]) for field in
                                             ("title", "newstitle", "abstract", "publish_date", "type", "meta_keywords")),
         "left_title": a["newstitle"], "right_title": b["newstitle"],
         "left_source_title": a["title"], "right_source_title": b["title"],
         "left_body": a["content"], "right_body": b["content"],
         "left_abstract": a["abstract"], "right_abstract": b["abstract"],
         "left_publish_date": a["publish_date"], "right_publish_date": b["publish_date"],
         "left_archive": a["archive"], "right_archive": b["archive"],
         "left_url": a["news_url"], "right_url": b["news_url"]}
    p["classification"], p["reason"], p["confidence"] = classify_pair(p)
    return p


def complete_link_groups(member_uids, pairs, records):
    lookup = {frozenset((p["left_member_uid"], p["right_member_uid"])): p for p in pairs}
    groups = []
    for uid in sorted(member_uids, key=lambda u: (RELEASES.index(records[u]["release"]), u)):
        for group in groups:
            if all(lookup[frozenset((uid, other))]["classification"] in LABELS[:2] for other in group):
                group.append(uid)
                break
        else:
            groups.append([uid])
    return groups


def component(unit, pairs, records):
    members = [records[u] for u in unit["members"]]
    keys = sorted({m["resource_key"] for m in members})
    pair_labels = {p["classification"] for p in pairs}
    if unit["cohort"] == "unresolved_identity_context":
        classification = LABELS[4]
        note = "Original component has unresolved multi-URL identity; pair comparisons are contextual, not proof of version change. Resolve resource identity first."
    elif LABELS[3] in pair_labels:
        classification, note = LABELS[3], "At least one pair has independently corroborated topic shift; verify historical captures before confirming repurposing."
    elif LABELS[2] in pair_labels:
        classification, note = LABELS[2], "At least one pair has multi-signal material-version evidence. The proposed complete-link grouping leaves uncertain observations separate."
    elif LABELS[4] in pair_labels:
        classification, note = LABELS[4], "One or more pairs lack enough corroborating evidence; do not infer equivalence from URL or boilerplate identity."
    elif pair_labels == {LABELS[0]}:
        classification, note = LABELS[0], "All observed pairs agree on informative captured text and available metadata; this is observed-excerpt stability, not full-page proof."
    else:
        classification, note = LABELS[1], "All pairs support equivalence with only small wording/metadata edits; retain full capture provenance if collapsed."
    action = {LABELS[0]: "Collapse provisionally, preserving every release/version capture",
              LABELS[1]: "Collapse provisionally, preserving edits and every release/version capture",
              LABELS[2]: "Separate candidate versioned entities; manual confirmation required before applying",
              LABELS[3]: "Treat versions as distinct content entities; confirm historical repurposing before applying",
              LABELS[4]: "Manual review or conservative sensitivity exclusion; no version assignment yet"}[classification]
    posts = set().union(*(m["post_ids"] for m in members))
    replies = set().union(*(m["reply_ids"] for m in members))
    sims = [p["content_cosine"] for p in pairs if np.isfinite(p["content_cosine"])]
    dates = sorted({m["publish_date"] for m in members if m["publish_date"]})
    title_consistency = {"scraped_exact_pair_fraction": np.mean([p["scraped_title_exact"] for p in pairs]),
                         "source_exact_pair_fraction": np.mean([p["source_title_exact"] for p in pairs])}
    version_groups = complete_link_groups(unit["members"], pairs, records) if classification in LABELS[2:4] else [unit["members"]]
    return {"resource_uid": unit["resource_uid"], "current_entity_uid": unit["current_entity_uid"],
            "canonical_resource_uid": "coaid_resource_" + hashlib.sha256(keys[0].encode()).hexdigest()[:24] if len(keys) == 1 else "",
            "cohort": unit["cohort"], "prior_entity_category": unit["prior_entity_category"],
            "prior_source_title_inconsistency": unit["prior_source_title_inconsistency"],
            "member_count": len(members), "member_source_uids_json": js(unit["members"]),
            "releases": " | ".join(sorted({m["release"] for m in members}, key=RELEASES.index)),
            "veracity": members[0]["veracity"], "resource_key": " | ".join(keys),
            "title_consistency": js(title_consistency), "text_similarity_summary": js({"minimum_cosine": min(sims) if sims else None, "median_cosine": float(np.median(sims)) if sims else None}),
            "minimum_content_cosine": min(sims) if sims else np.nan,
            "median_content_cosine": np.median(sims) if sims else np.nan,
            "exact_informative_body_pair_fraction": np.mean([p["content_exact"] and p["both_bodies_informative"] for p in pairs]),
            "informative_body_members": sum(m["body_informative"] for m in members),
            "informative_abstract_members": sum(m["abstract_informative"] for m in members),
            "publication_date_consistency": "missing" if not dates else "consistent among available" if len(dates) == 1 else "different recorded dates",
            "publication_dates_json": js(dates), "publication_date_missing_members": sum(not m["publish_date"] for m in members),
            "archive_url_members": sum(bool(m["archive"]) for m in members),
            "source_publisher": " | ".join(sorted({urlsplit("https://" + m["resource_key"]).netloc for m in members})),
            "resource_behavior": " | ".join(sorted({m["resource_behavior"] for m in members})),
            "government_institutional": any(m["government_institutional"] for m in members),
            "classification": classification, "proposed_action": action,
            "confidence": "medium (stored excerpts only)" if classification != LABELS[4] else "low",
            "observed_unique_posts": len(posts), "observed_unique_replies": len(replies),
            "candidate_version_count": len(version_groups) if classification != LABELS[4] else np.nan,
            "insufficient_pair_count": sum(p["classification"] == LABELS[4] for p in pairs),
            "pair_count": len(pairs), "notes": note}, version_groups


def summarize(validation, primary):
    rows = []
    for cohort in ("same_resource_primary", "corrected_resource_control", "unresolved_identity_context"):
        base = validation.loc[validation.cohort.eq(cohort)]
        for group in ("all", "fake", "real"):
            denominator = base if group == "all" else base.loc[base.veracity.eq(group)]
            for label in LABELS:
                part = denominator.loc[denominator.classification.eq(label)]
                rows.append({"cohort": cohort, "veracity": group, "classification": label,
                             "resources": len(part), "audited_group_resources": len(denominator),
                             "within_audited_group_percent": 100*len(part)/len(denominator) if len(denominator) else np.nan,
                             "source_rows": int(part.member_count.sum()), "sum_resource_unique_posts": int(part.observed_unique_posts.sum()),
                             "sum_resource_unique_replies": int(part.observed_unique_replies.sum()),
                             "post_share_previous_primary_total": part.observed_unique_posts.sum()/primary.unique_post_count.sum(),
                             "reply_share_previous_primary_total": part.observed_unique_replies.sum()/primary.unique_reply_count.sum(),
                             "notes": "CoAID prior-primary denominator is descriptive and unrepaired; corrected resource splits may share social IDs. Cohorts reported separately; no population error-rate or causal interpretation."})
    return pd.DataFrame(rows)


def manual_review(validation, units, pair_lookup, records):
    chosen, reasons = set(), defaultdict(set)
    def take(frame, n, why):
        for uid in frame.resource_uid.head(n):
            chosen.add(uid)
            reasons[uid].add(why)
    take(validation.sort_values("observed_unique_posts", ascending=False), 15, "high_engagement_priority_only")
    take(validation.loc[validation.veracity.eq("fake")], 10, "fake_examples")
    take(validation.loc[validation.classification.isin(LABELS[2:4])].sort_values("observed_unique_posts", ascending=False), 12, "candidate_material_or_repurposed")
    take(validation.loc[validation.cohort.eq("unresolved_identity_context")], 6, "unresolved_identity")
    take(validation.loc[validation.cohort.eq("corrected_resource_control")].sort_values("member_count", ascending=False), 4, "large_parent_control")
    candidates = validation.loc[~validation.resource_uid.isin(chosen) & validation.government_institutional & validation.classification.eq(LABELS[4])].sort_values("observed_unique_posts", ascending=False)
    take(candidates, max(0, 45-len(chosen)), "institutional_insufficient_evidence")
    remaining = validation.loc[~validation.resource_uid.isin(chosen)].sample(frac=1, random_state=SEED)
    take(remaining, max(0, 45-len(chosen)), "stratified_fill")
    assert 30 <= len(chosen) <= 50
    rows = []
    for row in validation.loc[validation.resource_uid.isin(chosen)].to_dict("records"):
        pairs = pair_lookup[row["resource_uid"]]
        exemplar = sorted(pairs, key=lambda p: (p["classification"] in LABELS[2:4], p["classification"] == LABELS[4], -p["content_cosine"] if np.isfinite(p["content_cosine"]) else 1), reverse=True)[0]
        rows.append({**row, "selection_reasons": " | ".join(sorted(reasons[row["resource_uid"]])),
                     "member_evidence_json": js([{k: records[u][k] for k in ("member_uid", "release", "newstitle", "title", "content", "abstract", "publish_date", "archive", "news_url", "meta_keywords")} for u in json.loads(row["member_source_uids_json"])]),
                     "representative_pair_evidence_json": js(exemplar), "human_classification": "", "human_notes": "",
                     "review_status": "PENDING; do not apply proposed assignments"})
    return pd.DataFrame(rows)


def main():
    # Snapshot all prior direct processed/tables/report outputs, not only inputs,
    # and confirm this stage writes exclusively to its new output names.
    new_prefix = "coaid_content_version_"
    protected = {p.relative_to(ROOT).as_posix(): digest(p) for folder in
                 (ROOT / "data/processed", TABLES, ROOT / "outputs/reports") for p in folder.glob("*")
                 if p.is_file() and not p.name.startswith(new_prefix)}
    prior, controls, records, input_hashes = load_members()
    units, inventory = candidates(prior, controls, records)
    similarity = Similarity(records)
    rows, pair_rows, proposal_rows, pair_lookup = [], [], [], {}
    for unit in units:
        pairs = [pair(records[a], records[b], similarity) for a, b in itertools.combinations(unit["members"], 2)]
        for p in pairs:
            p["rubric_classification_before_excerpt_review"] = p["classification"]
            p["rubric_reason_before_excerpt_review"] = p["reason"]
            p["qualitative_excerpt_review_applied"] = False
        if unit["resource_uid"] in EXCERPT_REVIEW:
            expected, decision, reason = EXCERPT_REVIEW[unit["resource_uid"]]
            captured = [{k: records[u][k] for k in ("member_uid", "content", "abstract", "newstitle")} for u in sorted(unit["members"])]
            assert hashlib.sha256(js(captured).encode()).hexdigest() == expected, "Reviewed evidence changed: re-review required"
            for p in pairs:
                p["classification"], p["reason"] = decision, "Qualitative excerpt review: " + reason
                p["confidence"] = "medium" if decision == LABELS[1] else "low"
                p["qualitative_excerpt_review_applied"] = True
        for p in pairs:
            p.update({"resource_uid": unit["resource_uid"], "cohort": unit["cohort"]})
        row, groups = component(unit, pairs, records)
        row["qualitative_excerpt_review_applied"] = unit["resource_uid"] in EXCERPT_REVIEW
        if row["qualitative_excerpt_review_applied"]:
            row["notes"] += " " + EXCERPT_REVIEW[unit["resource_uid"]][2]
        rows.append(row)
        pair_rows.extend(pairs)
        pair_lookup[unit["resource_uid"]] = pairs
        for index, group in enumerate(groups, 1):
            unresolved = row["classification"] == LABELS[4]
            proposal_rows.append({"current_resource_uid": unit["resource_uid"], "current_entity_uid": unit["current_entity_uid"],
                                  "canonical_resource_uid": row["canonical_resource_uid"],
                                  "proposed_version_uid": "" if unresolved else row["canonical_resource_uid"] + f"_v{index:03d}",
                                  "member_source_uids_json": js(group), "releases": " | ".join(sorted({records[u]["release"] for u in group}, key=RELEASES.index)),
                                  "titles_json": js([{k: records[u][k] for k in ("member_uid", "newstitle", "title")} for u in group]),
                                  "content_similarity_evidence": row["text_similarity_summary"], "classification": row["classification"],
                                  "reason_for_split_or_collapse": row["proposed_action"] + "; " + row["notes"],
                                  "proposal_status": "DEFERRED: no version assignment" if unresolved else "PROPOSED ONLY; not applied",
                                  "grouping_rule": "Complete-link compatibility: every within-version pair must be SAME CONTENT or MINOR UPDATE; no transitive chaining through uncertain/material pairs."})
    validation = pd.DataFrame(rows)
    primary = pd.read_csv(ROOT / "data/processed/coaid_news_primary.csv")
    summary = summarize(validation, primary)
    review = manual_review(validation, units, pair_lookup, records)
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(TABLES / "coaid_content_version_validation.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(TABLES / "coaid_content_version_pair_evidence.csv", index=False)
    pd.DataFrame(proposal_rows).to_csv(TABLES / "coaid_content_version_proposed_groups.csv", index=False)
    review.to_csv(TABLES / "coaid_content_version_manual_review.csv", index=False)
    summary.to_csv(TABLES / "coaid_content_version_summary.csv", index=False)
    pd.DataFrame(inventory).to_csv(TABLES / "coaid_content_version_control_inventory.csv", index=False)
    pd.DataFrame([{k: v for k, v in m.items() if k not in {"post_ids", "reply_ids"}} for m in records.values()]).to_csv(TABLES / "coaid_content_version_member_evidence.csv", index=False)
    write_report(validation, summary, review, records, inventory)
    for relative, checksum in {**protected, **input_hashes}.items():
        assert digest(ROOT / relative) == checksum, f"Protected input/output changed: {relative}"
    metadata = {"seed": SEED, "audit_rows": len(validation), "cohort_counts": validation.cohort.value_counts().to_dict(),
                "pair_comparisons": len(pair_rows), "manual_review_cases": len(review),
                "raw_input_sha256": input_hashes, "protected_previous_output_sha256": protected,
                "tfidf": {"corpus": "all 584 previously audited member bodies and abstracts; noninformative documents represented as empty", "ngram_range": [1, 2], "stop_words": "english", "sublinear_tf": True, "norm": "l2", "vocabulary_size": len(similarity.vectorizer.vocabulary_)},
                "script_sha256": digest(Path(__file__)), "models_or_eda_rerun": False, "dataset_rebuilt": False,
                "qualitative_excerpt_reviews": EXCERPT_REVIEW,
                "prior_outputs_and_raw_inputs_unchanged": True}
    (TABLES / "coaid_content_version_metadata.json").write_text(js(metadata) + "\n")
    print(summary.loc[summary.veracity.eq("all"), ["cohort", "classification", "resources", "source_rows", "sum_resource_unique_posts", "sum_resource_unique_replies"]].to_string(index=False))
    print(f"Saved {len(validation)} audit rows, {len(pair_rows)} pairs, {len(review)} review cases. Prior outputs and raw inputs unchanged.")


def write_report(validation, summary, review, records, inventory):
    primary = validation.loc[validation.cohort.eq("same_resource_primary")]
    stable = primary.loc[primary.classification.isin(LABELS[:2])]
    material = primary.loc[primary.classification.isin(LABELS[2:4])]
    insufficient = primary.loc[primary.classification.eq(LABELS[4])]
    focus = primary.loc[primary.prior_source_title_inconsistency]
    all_resource = validation.loc[validation.cohort.ne("unresolved_identity_context")]
    main_table = summary.loc[summary.cohort.eq("same_resource_primary") & summary.veracity.eq("all")]
    rate_rows = []
    for cohort, frame in [("same-resource primary", primary), ("corrected resource controls", all_resource.loc[all_resource.cohort.eq("corrected_resource_control")])]:
        for group in ("fake", "real"):
            sub = frame.loc[frame.veracity.eq(group)]
            n = int(sub.classification.isin(LABELS[2:4]).sum())
            rate_rows.append({"cohort": cohort, "veracity": group, "audited_resources": len(sub),
                              "material_or_repurposed": n, "within_group_percent": 100*n/len(sub) if len(sub) else np.nan,
                              "insufficient": int(sub.classification.eq(LABELS[4]).sum())})
    examples = material.sort_values("observed_unique_posts", ascending=False).head(8)
    # Excerpt equality is evidence about these captures only, particularly for
    # institutional pages where other unobserved portions may have changed.
    lines = ["# CoAID content-version stability audit", "",
             "Same item-specific URL is not sufficient to establish the same immutable content version. The stored content fields are excerpts (maximum 500 raw characters), and source-title inconsistencies are often metadata discrepancies rather than evidence of a changed article. The audit compares those fields separately and does not rebuild data or rerun EDA/models.", "",
             f"Scope: all 185 previous high-confidence same-resource components, including the 121 flagged source-title cases and 64 unflagged controls; six probable/ambiguous multi-URL identity contexts; and {int((validation.cohort == 'corrected_resource_control').sum())} cross-release resource groups proposed inside the nine over-merged parents. The remaining {sum(not r['eligible_cross_release_version_comparison'] for r in inventory)} proposed groups have only one observation or only within-release repeats and are inventoried without claiming cross-release stability. Do not pool these cohorts as a random sample of CoAID.", "",
             "## Same-resource primary results", "",
             md_table(main_table[["classification", "resources", "source_rows", "sum_resource_unique_posts", "sum_resource_unique_replies"]]), "",
             f"Of 185 resource components, {len(material)} have candidate material/repurposed versions, {int(primary.classification.eq(LABELS[1]).sum())} appear to have only minor updates, and {len(insufficient)} have insufficient evidence. Among the 121 source-title-flagged components, the classification counts are {js(focus.classification.value_counts().to_dict())}. These are evidence-based audit labels awaiting manual review, not validated whole-page equivalence judgments.", "",
             f"Candidate materially changing/repurposed primary resources contain {int(material.member_count.sum())} source rows, {int(material.observed_unique_posts.sum()):,} unique-post memberships and {int(material.observed_unique_replies.sum()):,} unique-reply memberships. These are associated with changing resources, not counts attributable to a specific version or known erroneous counts. Corrected controls and multi-URL identity contexts are quantified separately in the summary CSV.", "",
             "## Similarity evidence and limits", "",
             "All raw source title, scraped newstitle, body, abstract, URL, fact-check URL, publication date, archive URL, type, publisher host, and keyword fields are preserved in the new member-evidence CSV. Pair evidence contains exact normalized equality, token Jaccard, normalized edit similarity and body/abstract TF-IDF cosine (word unigrams/bigrams, English stop words, sublinear TF, L2 normalization). Engagement is used only to prioritize human review and quantify consequences, never to classify identity.", "",
             "The rubric combines fields rather than applying one cosine cutoff. Identical informative body plus unchanged metadata supports SAME CONTENT in the observed capture. Body identity with metadata-only changes supports MINOR UPDATE. Small edits require agreement from multiple body measures plus compatible title/abstract. Material candidates require substantial body change with corroborating abstract/title/update-language/date evidence, or evolving surveillance numbers. Repurposing requires strong divergence in specific scraped title, body and informative abstract together. Missing/error/boilerplate captures, uncorroborated excerpt changes, and unresolved resource aliases are INSUFFICIENT EVIDENCE. Dates are metadata, not capture timestamps, and repeated dates such as 2/11/20 do not prove stability.", "",
             "Low similarity can reflect different extracted portions of a page, not a different topic. Conversely, an identical 500-character introduction does not establish that later sections, recommendations, or data stayed unchanged. Source-title shifts alone never establish material change or repurposing. Generic CDC headers and error pages never establish sameness.", "",
             "Qualitative inspection refined four resource decisions: the outbreak FAQ introductions reflect wording/date edits, while healthcare-index, park-guidance, and older-adult captures may select different page sections and remain insufficient. These decisions are bound to exact evidence fingerprints; initial rubric labels/reasons are preserved beside the final decisions. A repeated retired-data notice is treated as a minor notice edit rather than a new surveillance version. This review is of stored excerpts, not a completed human archive adjudication.", "",
             "## Institutional resource behavior", "",
             md_table(primary.groupby(["resource_behavior", "classification"]).size().rename("resources").reset_index()), "",
             "Resource types are transparent URL/publisher heuristics, not verified historical diagnoses. CDC guidance is often maintained; dashboards/surveillance report changing data; index/resource pages can retain an identical introduction while their contents change. Dated press releases are more article-like but still require capture comparison. A continuously updated page must not be silently treated as an immutable article. Stored archive references are exported where present; no live archives were retrieved, so these findings concern available dataset captures only.", "",
             "## Veracity proportions", "", md_table(pd.DataFrame(rate_rows)), "",
             "Interpret within-group proportions and insufficient-evidence counts, not just the number of Real cases. The primary same-resource cohort has only two Fake resources, one of which has insufficient evidence. All 28 eligible cross-release corrected controls are Real; the repeated Fake control has only within-release observations and is inventoried outside the version-comparison denominator. Thus apparent concentration among Real resources is descriptive and confounded with resource type and data availability; this audit cannot establish a reliable population-level Fake/Real difference in version instability.", "",
             "## Proposed version groups and review", "",
             md_table(examples[["resource_uid", "resource_behavior", "member_count", "classification", "candidate_version_count", "observed_unique_posts"]]), "",
             "SAME CONTENT/MINOR UPDATE resources receive provisional collapse recommendations with all capture provenance retained. MATERIAL CONTENT UPDATE receives canonical_resource_uid + version proposals. REPURPOSED RESOURCE, if supported, receives distinct content-version groups even at the same URL. Version clustering is complete-link: every member pair in a proposed version must support same/minor equivalence, so uncertain or materially different pairs cannot join through a transitive chain. INSUFFICIENT EVIDENCE rows have no assigned version UID and require manual review or conservative sensitivity exclusion. None of these proposals is applied.", "",
             f"The manual-review CSV contains {len(review)} selected high-impact/difficult cases, including institutional pages, material candidates, unresolved aliases, controls from large over-merged parents, and both veracity groups. Complete side-by-side captured evidence is embedded and human adjudication fields remain blank. The six probable/ambiguous original identity contexts must resolve URL aliases/translations first; their conservative E labels are not a claim that those resources changed materially.", "",
             "## Analytical unit and rebuild decision", "",
             "Use **validated content resource + validated content version**, with a separate source-record/capture layer recording release/file/row IDs and observed engagement links. A URL is an identity candidate; a release is a capture label, not automatically a content version. Preserve version uncertainty rather than assigning every capture to an immutable resource entity.", "",
             "Rebuild rule: (1) resolve the prior over-merges into item-specific resource groups and adjudicate cross-URL aliases; (2) validate raw source-title/URL/social-link alignment; (3) mask boilerplate/error text and assess informative body/abstract/title/version evidence; (4) collapse only same/minor version-equivalent captures with provenance; (5) separate materially updated/repurposed versions using complete-link compatibility; (6) keep insufficient cases unresolved, with a pre-specified conservative exclusion sensitivity. Union social IDs only after item/version linkage is justified.", "",
             "We are not yet ready for an unconditional primary rebuild. The audit supplies executable proposals, but material/insufficient cases need adjudication or an explicit conservative inclusion policy. Release-level post/reply links may not reveal which historical version a post actually referenced; a capture's content version cannot automatically be assigned to every associated social ID. Keep that linkage uncertainty visible before aggregating version-specific engagement. Stable/minor proposals can form a conservative core once prior resource and metadata issues are resolved. No primary data or previous audit outputs were changed."]
    REPORT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
