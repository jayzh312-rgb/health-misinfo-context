"""Apply the existing CoAID identity/version audits; no new matching or modeling.

Run from any directory: <project>/.venv/bin/python <project>/src/rebuild_coaid_version_aware_samples.py
All counts are rebuilt from raw string-ID sets. Existing inputs are read-only.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from audit_coaid_entity_resolution import RawTrace, specific_url, url_key
from content_table_utils import timing

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs/tables"
PROCESSED = ROOT / "data/processed"
SPECS = {
    "A_primary_strict": "coaid_news_primary_version_aware.csv",
    "B_resource_sensitivity": "coaid_news_resource_level_sensitivity.csv",
    "C_high_certainty": "coaid_news_high_certainty_sensitivity.csv",
}
GIANT = "coaid_news_entity_518045a5c665be041af4c5bb"


def js(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def partition_check(groups, expected, label):
    flat = [m for g in groups for m in g]
    require(len(flat) == len(set(flat)), f"Duplicate assignment: {label}")
    require(set(flat) == set(expected), f"Incomplete partition: {label}")


def entity_uid(resource, mode, members):
    # Member source UIDs encode release/capture identity, never dataframe position.
    return "coaid_version_aware_" + digest(js([resource, mode, sorted(members)]))


def read_table(name):
    return pd.read_csv(TABLES / (name + ".csv"), keep_default_na=False)


def load_inputs():
    news = pd.read_csv(PROCESSED / "coaid_content_level.csv", keep_default_na=False,
                       dtype={"original_content_id": str})
    news = news[news.content_category.eq("News")].set_index("content_uid", drop=False)
    old = pd.read_csv(PROCESSED / "coaid_news_primary.csv", keep_default_na=False)
    names = ["coaid_entity_resolution_validation", "coaid_entity_resolution_proposed_groups",
             "coaid_content_version_validation", "coaid_content_version_proposed_groups",
             "coaid_entity_resolution_member_evidence", "coaid_entity_resolution_link_evidence",
             "coaid_content_version_member_evidence", "coaid_content_version_pair_evidence",
             "coaid_content_version_control_inventory"]
    tables = {n: read_table(n) for n in names}
    old_groups = {r.content_uid: sorted(json.loads(r.member_original_uids_json)) for r in old.itertuples()}
    partition_check(old_groups.values(), news.index, "old primary lineage")
    require(news.index.is_unique and old.content_uid.is_unique, "Nonunique input UIDs")
    source_to_old = {m: p for p, members in old_groups.items() for m in members}
    trace = RawTrace()
    raw = {r.content_uid: trace.member(r) for r in news.itertuples(index=False)}
    ev = tables["coaid_entity_resolution_member_evidence"]
    for r in ev.itertuples():
        require(raw[r.member_uid]["posts"] == set(json.loads(r.observed_post_ids_json)), "Post evidence drift")
        require(raw[r.member_uid]["replies"] == set(json.loads(r.observed_reply_ids_json)), "Reply evidence drift")
        require(source_to_old[r.member_uid] == r.entity_uid, "Identity member evidence drift")
    vm = tables["coaid_content_version_member_evidence"]
    require(set(vm.member_uid) == set(ev.member_uid), "Version member coverage drift")
    for r in vm.itertuples():
        require(r.resource_key == raw[r.member_uid]["news_url_key"], "Version resource evidence drift")
    for r in tables["coaid_entity_resolution_link_evidence"].itertuples():
        require(r.entity_uid in old_groups, "Unknown link evidence parent")
    for r in old.itertuples():
        members = old_groups[r.content_uid]
        require(len(set().union(*(raw[m]["posts"] for m in members))) == r.unique_post_count, "Old post union drift")
        require(len(set().union(*(raw[m]["replies"] for m in members))) == r.unique_reply_count, "Old reply union drift")
    input_paths = [PROCESSED / "coaid_content_level.csv", PROCESSED / "coaid_news_primary.csv"]
    input_paths += [TABLES / (n + ".csv") for n in names]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
    hashes.update(trace.hashes)
    return news, old, old_groups, source_to_old, raw, tables, hashes


def build_plans(old_groups, raw, tables):
    identity = tables["coaid_entity_resolution_validation"].set_index("entity_uid")
    versions = tables["coaid_content_version_validation"].set_index("resource_uid")
    proposals = tables["coaid_entity_resolution_proposed_groups"]
    vp = tables["coaid_content_version_proposed_groups"]
    pairs = tables["coaid_content_version_pair_evidence"]
    pair_classes = {(r.resource_uid, frozenset((r.left_member_uid, r.right_member_uid))): r.classification[0]
                    for r in pairs.itertuples()}
    require(identity.index.is_unique and versions.index.is_unique, "Nonunique audit UIDs")
    require(set(identity.index) == {p for p, g in old_groups.items() if len(g) > 1}, "Audit parent coverage drift")
    corrected = set(identity[identity.validation_category.str.startswith("E.")].index)
    require(len(corrected) == 9 and len(old_groups[GIANT]) == 110, "Expected nine corrected parents and giant")
    require(set(proposals.current_entity_uid) == corrected, "Corrected proposal coverage drift")
    require(set(vp.current_resource_uid) == set(versions.index), "Version proposal coverage drift")
    plans = {s: [] for s in SPECS}
    exclusions = []
    used_version_audits = set()

    def add(spec, parent, resource_audit, members, mode, classification, evidence, proposed_version=""):
        members = sorted(members)
        keys = {raw[m]["news_url_key"] for m in members}
        resource = next(iter(keys)) if len(keys) == 1 and specific_url(next(iter(keys))) else ""
        if len(members) > 1:
            require(bool(resource), f"Merged group without one specific resource: {resource_audit}")
        plans[spec].append(dict(parent=parent, resource_audit=resource_audit, members=members,
                                mode=mode, version_classification=classification, resource=resource,
                                canonical_resource_uid="coaid_resource_" + digest(resource) if resource else "",
                                proposed_version_uid=proposed_version, identity_evidence=evidence,
                                entity_resolution_classification=identity.loc[parent, "validation_category"]
                                if parent in identity.index else "SINGLE OBSERVATION; no merge proposed",
                                corrected_overmerge=parent in corrected,
                                uid=entity_uid(resource, mode, members)))

    for parent, members in sorted(old_groups.items()):
        if parent not in identity.index:
            require(len(members) == 1, "Unaudited merged parent")
            for spec in SPECS:
                add(spec, parent, "", members, "single_observed_capture", "NOT APPLICABLE: single observation",
                    "Retained original singleton; no new cross-record equivalence asserted")
            continue
        if parent in corrected:
            pg = proposals[proposals.current_entity_uid.eq(parent)]
            partition_check([json.loads(x) for x in pg.members_json], members, parent)
            resources = [(r.proposed_subgroup, json.loads(r.members_json), r.supporting_evidence) for r in pg.itertuples()]
        else:
            resources = [(parent, members, identity.loc[parent, "strongest_match_evidence"])]
        for rid, group, evidence in resources:
            if rid not in versions.index:
                # Explicit corrected groups outside the cross-release version audit:
                # 98 singletons and one same-release repeated specific resource.
                require(parent in corrected, "Missing version audit")
                require(len({raw[m]["release"] for m in group}) == 1, "Unaudited cross-release collapse")
                for spec in SPECS:
                    add(spec, parent, rid, group, "audited_same_release_resource" if len(group) > 1 else "single_observed_capture",
                        "NOT ASSESSED: single-release corrected resource", evidence)
                continue
            used_version_audits.add(rid)
            vr = versions.loc[rid]
            require(set(json.loads(vr.member_source_uids_json)) == set(group), "Version/member partition drift")
            require(vr.current_entity_uid == parent, "Version parent drift")
            if vr.cohort != "unresolved_identity_context":
                require({raw[m]["news_url_key"] for m in group} == {vr.resource_key}, "Audited resource key drift")
                require(vr.canonical_resource_uid == "coaid_resource_" + digest(vr.resource_key), "Canonical UID drift")
            vgroups = vp[vp.current_resource_uid.eq(rid)]
            partition_check([json.loads(x) for x in vgroups.member_source_uids_json], group, rid)
            classification = vr.classification
            require(vgroups.classification.eq(classification).all(), "Version proposal classification drift")
            if vr.cohort == "unresolved_identity_context":
                require(identity.loc[parent, "validation_category"][0] in "BC", "Unexpected unresolved identity")
                mode = "unresolved_identity_capture"
            elif classification.startswith("E."):
                mode = "insufficient_version_capture"
            else:
                require(classification[0] in "ABC", "Unsupported audited classification")
                for vg in vgroups.itertuples():
                    gm = json.loads(vg.member_source_uids_json)
                    require(bool(vg.proposed_version_uid), "Missing approved version UID")
                    for a, b in itertools.combinations(gm, 2):
                        require(pair_classes.get((rid, frozenset((a, b)))) in {"A", "B"}, "Uncertain/transitive within-version edge")
                    for spec in SPECS:
                        add(spec, parent, rid, gm, "validated_content_version", classification, evidence, vg.proposed_version_uid)
                continue
            for m in group:
                exclusions.append(dict(specification="C_high_certainty", member_source_uid=m,
                                       previous_entity_uid=parent, audited_resource_uid=rid,
                                       reason=mode, version_classification=classification))
            for spec in ("A_primary_strict", "B_resource_sensitivity"):
                if spec.startswith("B") and mode == "insufficient_version_capture":
                    require(specific_url(vr.resource_key), "Nonspecific resource sensitivity merge")
                    add(spec, parent, rid, group, "assumed_stable_resource", classification, evidence)
                else:
                    for m in group:
                        add(spec, parent, rid, [m], mode, classification, evidence)
    require(used_version_audits == set(versions.index), "Unused version audits")
    excluded = [r["member_source_uid"] for r in exclusions]
    require(len(excluded) == len(set(excluded)), "Duplicate exclusion")
    for spec, groups in plans.items():
        expected = set(raw) - (set(excluded) if spec.startswith("C") else set())
        partition_check([g["members"] for g in groups], expected, spec)
        require(len({g["uid"] for g in groups}) == len(groups), "Duplicate final UID")
    return plans, pd.DataFrame(exclusions)


def materialize(spec, groups, news, old, raw):
    old_index = old.set_index("content_uid")
    records, lineage, socials = [], [], {}
    for g in groups:
        members = g["members"]
        source = news.loc[members]
        candidates = source.to_dict("records")
        rep = sorted(candidates, key=lambda r: (-len(r["text"]), -len(r["title"]), r["source_release_or_subset"], r["content_uid"]))[0]
        posts = set().union(*(raw[m]["posts"] for m in members))
        replies = set().union(*(raw[m]["replies"] for m in members))
        socials[g["uid"]] = {"posts": posts, "replies": replies}
        labels = {raw[m]["veracity"] for m in members}
        require(len(labels) == 1 and labels <= {"fake", "real"}, "Mixed/unknown veracity")
        label = next(iter(labels))
        releases = sorted({raw[m]["release"] for m in members})
        original_ids = [raw[m]["original_id"] for m in members]
        metadata = [{"source_uid": m, "release": raw[m]["release"], "original_content_id": raw[m]["original_id"],
                     "title": raw[m]["title"], "source_title": raw[m]["source_title"], "text": raw[m]["text"],
                     "news_url": raw[m]["news_url"], "publish_date_raw": raw[m]["publish_date_raw"],
                     "raw_provenance": json.loads(news.loc[m, "provenance"])} for m in members]
        rule = "Raw post/reply ID set union within explicitly audited group only"
        note = ("Single observed capture is not proof of historical stability. " if g["mode"] == "single_observed_capture" else
                "Same-release corrected resource; no cross-release content-stability claim. " if g["mode"] == "audited_same_release_resource" else
                "Greater cross-release stability assumed for sensitivity only. " if g["mode"] == "assumed_stable_resource" else
                "Uncertain captures remain separate; overlapping social IDs are retained. " if g["mode"].endswith("capture") else
                "Audited candidate version, not a perfectly resolved historical version. ")
        record = {c: rep[c] for c in old.columns if c in rep}
        record.update(dataset="CoAID", subset="News", content_category="News", specification=spec,
                      content_uid=g["uid"], stable_entity_uid=g["uid"], content_id=g["uid"],
                      original_dataset_label=rep["veracity_original"], coaid_within_dataset_analysis_group=label,
                      source_release_or_subset=" | ".join(releases), unique_post_count=len(posts), observed_post_count=len(posts),
                      unique_reply_count=len(replies), reply_count=len(replies), engagement_present=bool(posts or replies),
                      member_original_uids_json=js(members), member_original_content_ids_json=js(original_ids),
                      member_release_periods_json=js(releases), release_observation_count=len(members), release_count=len(releases),
                      raw_observed_post_membership_sum=int(source.observed_post_count.sum()),
                      raw_unique_post_count_sum=sum(len(raw[m]["posts"]) for m in members),
                      raw_reply_membership_sum=int(source.reply_count.sum()),
                      raw_unique_reply_count_sum=sum(len(raw[m]["replies"]) for m in members),
                      previous_entity_uid=g["parent"], canonical_resource_key=g["resource"],
                      canonical_resource_uid=g["canonical_resource_uid"], content_version_uid=g["proposed_version_uid"],
                      audited_resource_uid=g["resource_audit"], version_classification=g["version_classification"],
                      entity_resolution_classification=g["entity_resolution_classification"], aggregation_mode=g["mode"],
                      corrected_overmerge=g["corrected_overmerge"], identity_evidence=g["identity_evidence"],
                      version_uncertainty_flag=g["mode"] in {"insufficient_version_capture", "assumed_stable_resource"},
                      unresolved_identity_flag=g["mode"] == "unresolved_identity_capture",
                      member_capture_metadata_json=js(metadata), representative_source_uid=rep["content_uid"],
                      duplicate_status="audited_group_collapsed" if len(members) > 1 else "separate_capture",
                      possible_duplicate_group=old_index.loc[g["parent"], "possible_duplicate_group"],
                      sensitivity_exclude_candidate=bool(old_index.loc[g["parent"], "sensitivity_exclude_candidate"]),
                      legacy_duplicate_flag_scope="Inherited parent-level flag; not the definition of specification C",
                      timestamp_role="secondary_sensitivity_only",
                      primary_outcomes="unique_post_count | unique_reply_count | engagement_present",
                      provenance=js(dict(aggregation_rule=rule, aggregation_mode=g["mode"], previous_entity_uid=g["parent"],
                                         representative_content_uid=rep["content_uid"], member_content_uids=members,
                                         raw_member_provenance=[x["raw_provenance"] for x in metadata])),
                      aggregation_notes=note)
        record.update(timing(posts, rep["publish_date_utc"]))
        require(set(old.columns) <= set(record), "Missing legacy pipeline columns")
        records.append(record)
        lineage.append(dict(specification=spec, final_entity_uid=g["uid"], previous_entity_uid=g["parent"],
                            member_source_uids=js(members), releases=js(releases), original_content_ids=js(original_ids),
                            veracity=label, canonical_resource=g["resource"], canonical_resource_uid=g["canonical_resource_uid"],
                            content_version=g["proposed_version_uid"], audited_resource_uid=g["resource_audit"],
                            identity_evidence=g["identity_evidence"], version_classification=g["version_classification"],
                            entity_resolution_classification=g["entity_resolution_classification"],
                            raw_observation_count=len(members), aggregation_mode=g["mode"], aggregation_rule=rule, notes=note))
    frame = pd.DataFrame(records).sort_values("content_uid").reset_index(drop=True)
    overlap = []
    for kind, singular in (("posts", "post"), ("replies", "reply")):
        owners = defaultdict(list)
        for eid, sets in socials.items():
            for social_id in sets[kind]:
                owners[social_id].append(eid)
        by_uid = frame.set_index("content_uid")
        counts = Counter()
        for social_id, entities in sorted(owners.items()):
            if len(entities) < 2:
                continue
            counts.update(entities)
            involved = by_uid.loc[entities]
            overlap.append(dict(specification=spec, social_type=singular, social_id=social_id,
                                final_entity_count=len(entities), excess_memberships=len(entities) - 1,
                                final_entity_uids_json=js(sorted(entities)),
                                previous_entity_uids_json=js(sorted(set(involved.previous_entity_uid))),
                                canonical_resource_keys_json=js(sorted(set(involved.canonical_resource_key))),
                                includes_insufficient_version_capture=bool(involved.aggregation_mode.eq("insufficient_version_capture").any()),
                                uncertain_capture_entity_count=int(involved.aggregation_mode.eq("insufficient_version_capture").sum()),
                                veracity_groups_json=js(sorted(set(involved.coaid_within_dataset_analysis_group)))))
        frame[f"cross_entity_{singular}_overlap_count"] = frame.content_uid.map(counts).fillna(0).astype(int)
        frame[f"cross_entity_{singular}_overlap_flag"] = frame[f"cross_entity_{singular}_overlap_count"].gt(0)
    return frame, lineage, overlap, socials


def validate_components(spec, groups, socials, old_groups, tables):
    checks = []
    cv = tables["coaid_content_version_validation"]
    vp = tables["coaid_content_version_proposed_groups"]
    material = cv[cv.classification.str.startswith("C.")]
    require(len(material) == 5, "Expected five material-update candidates")
    for r in material.itertuples():
        actual = [g for g in groups if g["resource_audit"] == r.resource_uid]
        proposed = vp[vp.current_resource_uid.eq(r.resource_uid)]
        expected = {tuple(sorted(json.loads(x))) for x in proposed.member_source_uids_json}
        require({tuple(g["members"]) for g in actual} == expected, "Material partition changed")
        checks.append(dict(specification=spec, check_type="material_version_partition", previous_entity_uid=r.current_entity_uid,
                           audited_resource_uid=r.resource_uid, final_entity_count=len(actual), passed=True))
    er = tables["coaid_entity_resolution_validation"]
    corrected = er[er.validation_category.str.startswith("E.")]
    proposal_members = {r.proposed_subgroup: set(json.loads(r.members_json))
                        for r in tables["coaid_entity_resolution_proposed_groups"].itertuples()}
    for r in corrected.itertuples():
        actual = [g for g in groups if g["parent"] == r.entity_uid]
        require(all(set(g["members"]) <= proposal_members[g["resource_audit"]] for g in actual),
                "Final group crosses audited corrected resources")
        require(all(set(g["members"]) != set(old_groups[r.entity_uid]) for g in actual), "Old overmerge survived")
        require(all(g["uid"] != r.entity_uid for g in actual), "Old overmerge UID survived")
        require(all(len(g["members"]) == 1 or specific_url(g["resource"]) for g in actual), "Domain merge survived")
        checks.append(dict(specification=spec, check_type="corrected_overmerge", previous_entity_uid=r.entity_uid,
                           audited_resource_uid="", final_entity_count=len(actual),
                           included_raw_observations=sum(len(g["members"]) for g in actual),
                           excluded_raw_observations=len(old_groups[r.entity_uid]) - sum(len(g["members"]) for g in actual),
                           post_memberships=sum(len(socials[g["uid"]]["posts"]) for g in actual),
                           reply_memberships=sum(len(socials[g["uid"]]["replies"]) for g in actual),
                           global_unique_posts=len(set().union(*(socials[g["uid"]]["posts"] for g in actual))),
                           global_unique_replies=len(set().union(*(socials[g["uid"]]["replies"] for g in actual))),
                           invalid_domain_merge_remains=False, passed=True))
    return checks


def summary_row(spec, frame, groups, socials, overlap, old_groups, raw):
    included = {m for g in groups for m in g["members"]}
    unchanged = {g["parent"] for g in groups if set(g["members"]) == set(old_groups[g["parent"]])}
    uncertain = frame[frame.aggregation_mode.eq("insufficient_version_capture")]
    unresolved = frame[frame.aggregation_mode.eq("unresolved_identity_capture")]
    row = dict(specification=spec, final_entity_count=len(frame),
               fake_count=int(frame.coaid_within_dataset_analysis_group.eq("fake").sum()),
               real_count=int(frame.coaid_within_dataset_analysis_group.eq("real").sum()),
               included_raw_observations=len(included), collapsed_raw_observations=len(included) - len(frame),
               version_split_entities=int(frame.version_classification.str.startswith("C.").sum()),
               validated_content_version_entities=int(frame.aggregation_mode.eq("validated_content_version").sum()),
               uncertain_captures_retained_separately=len(uncertain), unresolved_identity_captures=len(unresolved),
               all_uncertainty_captures_retained=len(uncertain) + len(unresolved),
               excluded_raw_observations=len(raw) - len(included),
               multi_release_entities=int(frame.release_count.gt(1).sum()),
               total_unique_post_memberships=int(frame.unique_post_count.sum()),
               total_unique_reply_memberships=int(frame.unique_reply_count.sum()),
               entities_with_zero_posts=int(frame.unique_post_count.eq(0).sum()),
               entities_with_zero_replies=int(frame.unique_reply_count.eq(0).sum()),
               previous_entities_membership_unchanged=len(unchanged),
               previous_entities_changed_or_excluded=len(old_groups) - len(unchanged),
               net_entity_count_change=len(frame) - len(old_groups),
               uncertain_capture_post_memberships=int(uncertain.unique_post_count.sum()),
               uncertain_capture_reply_memberships=int(uncertain.unique_reply_count.sum()))
    for kind, singular in (("posts", "post"), ("replies", "reply")):
        actual = set().union(*(v[kind] for v in socials.values()))
        expected = set().union(*(raw[m][kind] for m in included))
        require(actual == expected, "Global included social ID coverage mismatch")
        row[f"global_unique_{kind}"] = len(actual)
        overlaps = [r for r in overlap if r["social_type"] == singular]
        row[f"cross_entity_shared_{singular}_ids"] = len(overlaps)
        row[f"cross_entity_excess_{singular}_memberships"] = sum(r["excess_memberships"] for r in overlaps)
        uncertain_overlaps = [r for r in overlaps if r["includes_insufficient_version_capture"]]
        row[f"shared_{singular}_ids_touching_uncertain_captures"] = len(uncertain_overlaps)
        row[f"excess_{singular}_memberships_among_uncertain_captures"] = sum(max(0, r["uncertain_capture_entity_count"] - 1) for r in overlaps)
        require(row[f"total_unique_{singular}_memberships"] == len(actual) + row[f"cross_entity_excess_{singular}_memberships"], "Overlap reconciliation failed")
        for g in groups:
            require(socials[g["uid"]][kind] == set().union(*(raw[m][kind] for m in g["members"])), "Within-entity union mismatch")
    return row


def class_comparison(frames, old, corrected):
    rows = []
    for scope in ("all_news", "nine_corrected_parents_only"):
        baseline = old if scope == "all_news" else old[old.content_uid.isin(corrected)]
        for spec, frame in {"previous_primary": old, **frames}.items():
            part = frame if scope == "all_news" else frame[frame["content_uid" if spec == "previous_primary" else "previous_entity_uid"].isin(corrected)]
            for label in ("fake", "real"):
                p = part[part.coaid_within_dataset_analysis_group.eq(label)]
                b = baseline[baseline.coaid_within_dataset_analysis_group.eq(label)]
                r = dict(scope=scope, specification=spec, veracity=label, entity_count=len(p),
                         post_memberships=int(p.unique_post_count.sum()), reply_memberships=int(p.unique_reply_count.sum()))
                for metric, before in (("entity_count", len(b)), ("post_memberships", int(b.unique_post_count.sum())),
                                       ("reply_memberships", int(b.unique_reply_count.sum()))):
                    r[metric + "_change"] = r[metric] - before
                    r[metric + "_pct_change"] = 100 * (r[metric] - before) / before if before else 0.0
                rows.append(r)
    return pd.DataFrame(rows)


def markdown(frame):
    # Avoid optional tabulate dependency and keep table content deterministic.
    def display(x):
        if isinstance(x, float):
            return str(int(x)) if x.is_integer() else f"{x:.3f}"
        return str(x).replace("|", "\\|").replace("\n", " ")
    vals = [[display(x) for x in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                     ["| " + " | ".join(row) + " |" for row in vals])


def report(summary, comparison, checks, hashes):
    giant = checks[checks.previous_entity_uid.eq(GIANT)]
    material = checks[checks.check_type.eq("material_version_partition")]
    return "\n\n".join([
        "# CoAID News version-aware rebuild",
        "## Decision and scope",
        "Specification A is PRIMARY: it does not assume cross-release content equivalence when the audit found insufficient evidence. B assumes stability only within the same validated specific resource, while still splitting material versions. C excludes audited insufficient-version and unresolved cross-URL contexts. No raw data or previous primary sample was modified; no EDA, regressions, hurdle models, or cross-context comparisons were run.",
        "## Rebuilt samples",
        markdown(summary[["specification", "final_entity_count", "fake_count", "real_count", "included_raw_observations", "collapsed_raw_observations", "excluded_raw_observations", "multi_release_entities"]]),
        "## Identity and version rules",
        "The nine overmerged parents are replaced by exactly the existing corrected-resource proposals, then the content-version audit is applied within each eligible resource. Same/minor groups use the existing proposed version partitions; all five material candidates use their exact complete-link proposed partitions in every specification. Every within-version pair is checked against A/B pair evidence. Six unresolved cross-URL contexts have no validated aliases: A/B retain each source capture separately and C excludes them. No domain, generic title, boilerplate, branding, social overlap, or newly inferred/transitive alias creates a merge.",
        "The audit covers repeated candidate groups, not every conceivable duplicate in the dataset. Original singletons are retained without asserting historical stability. Of the 127 corrected-resource groups, 98 are single observations and one is a two-observation, single-release resource: these remain in all specifications under the explicit corrected-resource rule, not as validated cross-release versions. C is high-certainty relative to the audited cross-release decisions, not a claim that every singleton has verified content or every possible duplicate has been adjudicated. Legacy parent-level possible-duplicate flags are preserved as provenance, not used to redefine C.",
        "Resource keys reuse the prior offline, item-specific URL normalization; they are not newly verified redirects. Nonspecific keys remain blank and cannot merge records. Final UIDs hash the specific resource, aggregation mode, and sorted original source UIDs, so they are independent of dataframe order and consistent across specifications for identical units. Audited version UIDs are also preserved. All rows receive the new UID namespace; membership-change counts below do not count mere UID renaming as a substantive change.",
        "## Versions and uncertainty",
        markdown(summary[["specification", "validated_content_version_entities", "version_split_entities", "uncertain_captures_retained_separately", "unresolved_identity_captures", "uncertain_capture_post_memberships", "uncertain_capture_reply_memberships"]]),
        "Validated content-version entities include same/minor collapses and the material candidate partitions; the material candidates produce 11 entities (six more than their five former parents). Insufficient-version capture counts exclude the 14 separate cross-URL identity-uncertain captures. Capture-attached engagement is not proof that each social event refers to the historical page text shown in that release; the stored bodies and linkage do not resolve that attribution.",
        "## Membership changes and engagement accounting",
        markdown(summary[["specification", "previous_entities_membership_unchanged", "previous_entities_changed_or_excluded", "net_entity_count_change", "total_unique_post_memberships", "total_unique_reply_memberships", "global_unique_posts", "global_unique_replies", "entities_with_zero_posts", "entities_with_zero_replies"]]),
        "A changed previous record means its exact source-member set is no longer an output entity, including full/partial exclusions. Excluded counts refer to raw source observations, not final entities that never existed. Collapsed observations equal included raw observations minus final entity count. Counts are unique ID set unions within each legitimate entity, never sums of old aggregate counts. Summed entity-level memberships can increase after splitting because a social ID may legitimately remain attached to multiple analytical captures. Global distinct ID coverage is separately reconciled; A/B preserve every original News-linked social ID.",
        "## Cross-entity overlap (retained, never silently deleted)",
        markdown(summary[["specification", "cross_entity_shared_post_ids", "cross_entity_shared_reply_ids", "cross_entity_excess_post_memberships", "cross_entity_excess_reply_memberships", "shared_post_ids_touching_uncertain_captures", "shared_reply_ids_touching_uncertain_captures", "excess_post_memberships_among_uncertain_captures", "excess_reply_memberships_among_uncertain_captures"]]),
        "One overlap-audit row represents one social ID/type/specification with multiple owners. Excess memberships equal owner count minus one. The 'among uncertain captures' columns count excess ownership within that subset only; 'touching' columns also include overlaps with other types of entity. These overlap definitions are not interchangeable. Boolean/count flags are present on every final entity. Shared IDs and repeated resources imply dependence concerns for later inference; structural validity does not establish independent observations.",
        "## Class composition: descriptive correction only",
        markdown(comparison[comparison.scope.eq("all_news")].drop(columns="scope")),
        "## Effect attributable to the nine corrected parents",
        markdown(comparison[comparison.scope.eq("nine_corrected_parents_only")].drop(columns="scope")),
        "These are assignment and sample-composition corrections, not evidence of a substantive Fake/Real engagement effect. In A/B the same underlying source IDs and veracity labels are preserved; changed membership totals arise from overlaps exposed by splits. C additionally removes uncertain observations and their memberships. The table isolates corrected parents from other version splits/exclusions so these mechanisms are not conflated.",
        "## Previous 110-member failure",
        markdown(giant[["specification", "final_entity_count", "included_raw_observations", "excluded_raw_observations", "post_memberships", "reply_memberships", "global_unique_posts", "global_unique_replies", "invalid_domain_merge_remains", "passed"]]),
        "The invalid connected component is removed in every specification. Any multi-member descendant must have one specific resource key and remain inside a single audited corrected-resource proposal. C excludes uncertain descendants rather than forcing them into a resource-level entity. This fixes the domain-level overmerge, not the unresolved historical attribution of individual social events.",
        "## Individual material-version validation",
        markdown(material[["specification", "previous_entity_uid", "final_entity_count", "passed"]]),
        "All nine overmerge checks per specification are recorded alongside the five material-version checks in `outputs/tables/coaid_version_aware_component_validation.csv`. All pass.",
        "## Integrity and pipeline handoff",
        "Passed: unique final UIDs; complete/disjoint source lineage and explicit exclusions; raw-to-processed-to-audit social-set agreement; final within-entity set uniqueness; no source assigned twice; exact approved version partitions; no surviving old overmerged component; no multi-resource/domain-only merge; global social coverage and overlap reconciliation; all legacy primary columns retained; written CSV count/flag/lineage round-trip validation; unchanged input SHA-256 hashes. The entity-lineage CSV and per-row capture metadata preserve raw file/row references, original IDs, releases, titles, text and representative selection. Timestamps are recomputed from final post unions and remain secondary sensitivity variables.",
        "The three CSVs are internally valid and schema-ready. Existing EDA/model scripts still hard-code `coaid_news_primary.csv`, so they will NOT automatically select these samples. A later authorized pipeline run must explicitly select the new input path/specification and respect its overlap/uncertainty flags; legacy sensitivity branches do not define these new A/B/C specifications. No pipeline scripts were changed or executed here.",
        "## Reproduction and artifacts",
        "Run `.venv/bin/python src/rebuild_coaid_version_aware_samples.py`. Run the accompanying tests with `.venv/bin/python -m unittest discover -s src -p 'test_coaid_version_aware_rebuild.py'`. Input hashes and validation results are in `outputs/tables/coaid_version_aware_rebuild_metadata.json`. Existing source/audit inputs remain unchanged. The script writes only the three requested new samples and version-aware rebuild artifacts, and refuses to proceed when authoritative audit partitions no longer agree.",
        f"Fingerprint manifest covers {len(hashes)} input files, including every raw News content/social mapping read."
    ]) + "\n"


def main():
    news, old, old_groups, source_to_old, raw, tables, hashes = load_inputs()
    plans, excluded = build_plans(old_groups, raw, tables)
    frames, summaries, lineage, overlaps, checks = {}, [], [], [], []
    for spec, groups in plans.items():
        frame, lin, ov, socials = materialize(spec, groups, news, old, raw)
        summaries.append(summary_row(spec, frame, groups, socials, ov, old_groups, raw))
        checks.extend(validate_components(spec, groups, socials, old_groups, tables))
        frames[spec] = frame
        lineage.extend(lin)
        overlaps.extend(ov)
    summary = pd.DataFrame(summaries)
    check_frame = pd.DataFrame(checks)
    corrected = set(tables["coaid_entity_resolution_proposed_groups"].current_entity_uid)
    comparison = class_comparison(frames, old, corrected)
    # Validate all inputs and output frames before writing any deliverables.
    for relative, original_hash in hashes.items():
        require(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == original_hash, f"Input changed: {relative}")
    for spec, frame in frames.items():
        path = PROCESSED / SPECS[spec]
        frame.to_csv(path, index=False)
        reread = pd.read_csv(path, keep_default_na=False)
        require(reread.content_uid.is_unique and len(reread) == len(frame), "CSV UID/count roundtrip failed")
        for col in ("unique_post_count", "unique_reply_count", "engagement_present", "member_original_uids_json",
                    "cross_entity_post_overlap_flag", "cross_entity_reply_overlap_flag"):
            require(reread[col].tolist() == frame[col].tolist(), f"CSV roundtrip failed: {col}")
    outputs = {"coaid_version_aware_rebuild_summary": summary,
               "coaid_version_aware_entity_lineage": pd.DataFrame(lineage),
               "coaid_cross_entity_social_overlap": pd.DataFrame(overlaps),
               "coaid_version_aware_excluded_observations": excluded,
               "coaid_version_aware_component_validation": check_frame,
               "coaid_version_aware_class_comparison": comparison}
    for name, frame in outputs.items():
        frame.to_csv(TABLES / (name + ".csv"), index=False)
    (ROOT / "outputs/reports/coaid_version_aware_rebuild.md").write_text(report(summary, comparison, check_frame, hashes))
    metadata = dict(input_sha256=hashes, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    specifications=SPECS, raw_news_observations=len(news), previous_primary_entities=len(old),
                    integrity_checks_passed=True, component_checks=len(checks),
                    previous_primary_preserved=True, raw_files_modified=False, models_or_eda_run=False)
    (TABLES / "coaid_version_aware_rebuild_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False))
    print(f"All integrity checks passed; {len(checks)} component checks; {len(hashes)} unchanged input hashes.")


if __name__ == "__main__":
    main()
