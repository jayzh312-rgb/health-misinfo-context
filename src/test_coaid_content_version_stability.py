"""Checks prevent metadata/error captures and transitive links becoming versions."""
import unittest

from audit_coaid_content_version_stability import LABELS, classify_pair, complete_link_groups


def evidence(**updates):
    p = {"same_resource_key": True, "both_bodies_informative": True,
         "content_cosine": 1., "content_edit_similarity": 1., "content_token_jaccard": 1.,
         "both_abstracts_informative": True, "abstract_exact": True, "abstract_cosine": 1.,
         "abstract_token_jaccard": 1., "scraped_title_exact": True, "either_scraped_title_generic": False,
         "scraped_title_jaccard": 1., "content_exact": True, "all_available_metadata_equal": True,
         "dashboard_page": False, "substantive_numbers_changed": False, "explicit_update_language": False,
         "maintained_page": False, "publication_date_changed": False}
    p.update(updates)
    return p


class ContentVersionChecks(unittest.TestCase):
    def test_identical_informative_capture(self):
        self.assertEqual(classify_pair(evidence())[0], LABELS[0])

    def test_metadata_only_change_is_minor(self):
        self.assertEqual(classify_pair(evidence(all_available_metadata_equal=False))[0], LABELS[1])

    def test_identical_boilerplate_is_insufficient(self):
        self.assertEqual(classify_pair(evidence(both_bodies_informative=False))[0], LABELS[4])

    def test_low_cosine_alone_cannot_prove_material_update(self):
        p = evidence(content_exact=False, content_cosine=.1, content_edit_similarity=.3, content_token_jaccard=.2)
        self.assertEqual(classify_pair(p)[0], LABELS[4])

    def test_multiple_signals_support_material_candidate(self):
        p = evidence(content_exact=False, content_cosine=.1, content_edit_similarity=.3,
                     content_token_jaccard=.2, explicit_update_language=True)
        self.assertEqual(classify_pair(p)[0], LABELS[2])

    def test_repuposing_requires_independent_fields(self):
        p = evidence(content_exact=False, content_cosine=.1, content_edit_similarity=.3,
                     content_token_jaccard=.1, abstract_exact=False, abstract_cosine=.1,
                     abstract_token_jaccard=.1, scraped_title_exact=False, scraped_title_jaccard=.1)
        self.assertEqual(classify_pair(p)[0], LABELS[3])

    def test_retired_notice_wording_is_not_new_surveillance_version(self):
        p = evidence(content_exact=False, content_cosine=.6, content_edit_similarity=.6,
                     content_token_jaccard=.5, maintained_page=True, publication_date_changed=True,
                     both_retired_resource_notices=True)
        self.assertEqual(classify_pair(p)[0], LABELS[1])

    def test_complete_link_prevents_chain_merging(self):
        pairs = [{"left_member_uid": "a", "right_member_uid": "b", "classification": LABELS[1]},
                 {"left_member_uid": "b", "right_member_uid": "c", "classification": LABELS[1]},
                 {"left_member_uid": "a", "right_member_uid": "c", "classification": LABELS[2]}]
        records = {u: {"release": "05-01-2020"} for u in "abc"}
        self.assertEqual(complete_link_groups(list("abc"), pairs, records), [["a", "b"], ["c"]])

    def test_unresolved_alias_precedes_version_classification(self):
        self.assertEqual(classify_pair(evidence(same_resource_key=False))[0], LABELS[4])


if __name__ == "__main__":
    unittest.main()
