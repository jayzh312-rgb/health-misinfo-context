"""Round 4 Check 5a (transformer environment): label-blind character offsets where the frozen 512-token pair packing cuts each FakeHealth body."""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT/'src'))
import audit_transformer_feasibility as audit
tok = audit.load_tokenizer(); special = tok.num_special_tokens_to_add(pair=True); assert special == 4
d = pd.read_csv(ROOT/'data/processed/fakehealth_text_modeling_ready.csv', usecols=['content_uid','title_text','body_text'], keep_default_na=False)
rows = []
for r in d.itertuples(index=False):
    t = tok(r.title_text, add_special_tokens=False, truncation=False)['input_ids']
    b = tok(r.body_text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    alloc = audit.title_body_allocation(len(t), len(b['input_ids']), 512, special); assert alloc['feasible']
    kept = alloc['body_kept']; cut = b['offset_mapping'][kept-1][1] if kept > 0 else 0
    if kept == len(b['input_ids']): cut = len(r.body_text)
    rows.append(dict(content_uid=r.content_uid, title_tokens=len(t), body_tokens=len(b['input_ids']), body_kept=kept, cut_char_offset=int(cut), body_chars=len(r.body_text), truncated=kept < len(b['input_ids'])))
out = pd.DataFrame(rows); out.to_csv(ROOT/'outputs/extension_v1/tables/round4_fakehealth_cut_offsets.csv', index=False)
print('rows', len(out), '| truncated share', round(out.truncated.mean(), 4), '| median body tokens', out.body_tokens.median(), '| median kept', out.body_kept.median())
