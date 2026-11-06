"""Render review_package_v3 (step 2 of 2): Markdown -> HTML (Python Markdown, latex2mathml) -> PDF (Chrome headless), then PyMuPDF checks and link repair.
Run with a Python environment satisfying requirements-render.txt (Markdown 3.11, latex2mathml 3.81.1, PyMuPDF 1.28.2). Set REVIEW_CHROME to a Chrome/Chromium executable if it is not at the macOS default."""
import re, json, subprocess, hashlib, sys, os, html
from urllib.parse import unquote
from pathlib import Path
import markdown
from latex2mathml.converter import convert as tex2mml
import pymupdf as fitz
V3 = Path(__file__).resolve().parents[1]
CHROME = os.environ.get('REVIEW_CHROME', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
EXT = ['tables', 'md_in_html', 'attr_list', 'toc']
EQUATION_COUNTS = {}
def md_inline(s): return re.sub(r'^<p>|</p>$', '', markdown.markdown(s, extensions=['attr_list']).strip())
def prepare(text, kind):
    # equations
    counter = {'n': 0}; prefix = 'S' if kind == 'supplement' else ''
    def eq(m):
        counter['n'] += 1
        return f'\n<div class="equation" id="eq-{prefix}{counter["n"]}"><span class="eq-body">' + tex2mml(m.group(1).strip()) + f'</span><span class="eq-no">({prefix}{counter["n"]})</span></div>\n'
    text = re.sub(r'\$\$\n(.*?)\n\$\$', eq, text, flags=re.S)
    EQUATION_COUNTS[kind] = counter['n']
    # figures: image line followed by italic caption paragraph
    def fig(m):
        alt, src, cap = m.group(1), m.group(2), m.group(3)
        img = f'<img alt="{html.escape(alt)}" src="{src}" />'
        if src.startswith('workflow/'):
            svg = src.replace('.png', '.svg'); img = f'<picture><source srcset="{svg}" type="image/svg+xml">{img}</picture>'
            return f'\n<figure class="workflow">{img}<figcaption>{md_inline(cap)}</figcaption></figure>\n'
        return f'\n<figure>{img}<figcaption>{md_inline(cap)}</figcaption></figure>\n'
    text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)\n\n\*(.+?)\*\n', fig, text, flags=re.S)
    body = markdown.markdown(text, extensions=EXT, extension_configs={'toc': {'toc_depth': '1-4'}})
    title = 'Manuscript review' if kind == 'manuscript' else 'Supplementary material'
    css = '<link rel="stylesheet" href="assets/review.css">' + ('<link rel="stylesheet" href="assets/mentor_review.css">' if kind == 'manuscript' else '')
    return f'<!doctype html><html><head><meta charset="UTF-8"><title>{title}</title>{css}</head><body class="{kind}">{body}</body></html>'
def print_pdf(html_path, pdf_path):
    cmd = [CHROME, '--headless=new', '--disable-gpu', '--no-sandbox', '--no-pdf-header-footer', '--virtual-time-budget=20000', f'--print-to-pdf={pdf_path}', html_path.resolve().as_uri()]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert pdf_path.exists() and pdf_path.stat().st_size > 1000, r.stderr[-800:]
def heading_pages(pdf):
    doc = fitz.open(pdf); pages = {}
    for i, page in enumerate(doc, 1):
        for b in page.get_text('dict')['blocks']:
            for l in b.get('lines', []):
                t = ''.join(s['text'] for s in l['spans']).strip()
                m = re.match(r'^(S\d)\. |^(Table S\d+)\. ', t)
                if m and any(s['size'] >= 12.5 for s in l['spans']):
                    key = m.group(1) or m.group(2); pages.setdefault(key, i)
    doc.close(); return pages
def checks(pdf):
    doc = fitz.open(pdf); out = dict(pages=len(doc), image_pages=[], out_of_bounds_text=[], blank_pages=[], replacement_glyph_pages=[], min_words=None, links=[])
    minw = 10**9
    for i, page in enumerate(doc, 1):
        r = page.rect; words = page.get_text('words'); minw = min(minw, len(words))
        if page.get_images(): out['image_pages'].append(i)
        if any(w[0] < r.x0 - 1 or w[1] < r.y0 - 1 or w[2] > r.x1 + 1 or w[3] > r.y1 + 1 for w in words): out['out_of_bounds_text'].append(i)
        if len(words) < 15 and not page.get_images(): out['blank_pages'].append(i)
        if '�' in page.get_text(): out['replacement_glyph_pages'].append(i)
        for ln in page.get_links():
            if ln.get('kind') in (fitz.LINK_GOTOR, fitz.LINK_LAUNCH): out['links'].append(dict(page=i, target=ln.get('file'), to_page=ln.get('page')))
    out['min_words'] = minw; doc.close(); return out
def repair_links(pdf, pkg_dir, supp_pages):
    doc = fitz.open(pdf); fixed = 0
    for page in doc:
        for ln in page.get_links():
            from urllib.parse import unquote
            uri = ln.get('uri') or ln.get('file') or ''
            if not uri or uri.startswith(('http://', 'https://', 'mailto:')): continue
            uri = unquote(uri); path = re.sub(r'^file://', '', uri); path = re.sub(r'#.*$', '', path); frag = uri.split('#', 1)[1] if '#' in uri else ''
            if ln.get('kind') == fitz.LINK_GOTOR and ln.get('page') is not None and not frag: frag = f"page={ln['page'] + 1}"
            if not path.startswith('/'): path = str(pkg_dir / path)
            try: rel = str(Path(path).resolve().relative_to(pkg_dir.resolve()))
            except ValueError: continue
            new = dict(from_=ln['from'])
            if rel == 'supplement_review.pdf' and frag.startswith('page='): new.update(kind=fitz.LINK_GOTOR, file='supplement_review.pdf', page=int(frag.split('=')[1]) - 1)
            elif rel == 'supplementary_material.md':
                new.update(kind=fitz.LINK_GOTOR, file='supplement_review.pdf', page=max(supp_pages.get(frag, 1) - 1, 0))
            else: new.update(kind=fitz.LINK_LAUNCH, file=rel)
            new['from'] = new.pop('from_'); page.delete_link(ln); page.insert_link(new); fixed += 1
    # force relative file names in the written GoToR/Launch actions (PyMuPDF resolves them against the document path)
    for page in doc:
        for ln in page.get_links():
            if ln.get('kind') in (fitz.LINK_GOTOR, fitz.LINK_LAUNCH) and ln.get('xref'):
                f = unquote(ln.get('file', ''))
                try: rel = str(Path(f).resolve().relative_to(pkg_dir.resolve()))
                except ValueError: rel = f
                doc.xref_set_key(ln['xref'], 'A/F', fitz.get_pdf_str(rel))
    doc.save(str(pdf) + '.tmp', garbage=3, deflate=True); doc.close(); os.replace(str(pdf) + '.tmp', pdf); return fixed
# ---- supplement first
supp_md = (V3/'supplementary_material.md').read_text()
(V3/'supplementary_material.html').write_text(prepare(supp_md, 'supplement'))
print_pdf(V3/'supplementary_material.html', V3/'supplement_review.pdf')
hp = heading_pages(V3/'supplement_review.pdf')
# anchor slug -> page map for manuscript links
def slug(h):
    import unicodedata; s = unicodedata.normalize('NFKD', h).encode('ascii', 'ignore').decode().lower(); s = re.sub(r'[^\w\s-]', '', s); return re.sub(r'[\s]+', '-', s).strip('-')
slug_pages = {}
for m in re.finditer(r'^## (S(\d)\..*|Table S(\d+)\..*)$', supp_md, re.M):
    key = ('S' + m.group(2)) if m.group(2) else ('Table S' + m.group(3))
    if key in hp: slug_pages[slug(m.group(1))] = hp[key]
# ---- manuscript: rewrite supplement anchors to PDF page links, render
man_md = (V3/'manuscript.md').read_text()
def to_page(m): s = m.group(1); return f'(supplement_review.pdf#page={slug_pages[s]})' if s in slug_pages else m.group(0)
man_html_md = re.sub(r'\(supplementary_material\.md#([a-z0-9_-]+)\)', to_page, man_md)
(V3/'manuscript.html').write_text(prepare(man_html_md, 'manuscript'))
print_pdf(V3/'manuscript.html', V3/'manuscript_review.pdf')
fixed_m = repair_links(V3/'manuscript_review.pdf', V3, slug_pages); fixed_s = repair_links(V3/'supplement_review.pdf', V3, slug_pages)
cm, cs = checks(V3/'manuscript_review.pdf'), checks(V3/'supplement_review.pdf')
# figure/caption pages in manuscript
doc = fitz.open(V3/'manuscript_review.pdf'); cap_pages = {}
for i, page in enumerate(doc, 1):
    t = page.get_text()
    for lab in ['Figure M1.', 'Figure 1.', 'Figure 2.', 'Figure 3.', 'Figure 4.', 'Table 1.', 'Table 2.']:
        if lab in t and lab not in cap_pages: cap_pages[lab] = i
doc.close()
rec = dict(equations_numbered=EQUATION_COUNTS, equation_numbering='explicit tag: each $$ block becomes <div class="equation"><span class="eq-body">MathML</span><span class="eq-no">(n)</span></div>, n sequential per document, S prefix in the supplement; .eq-no is absolutely positioned at the right edge of the equation line (assets/review.css)', toolchain=dict(python=sys.version.split()[0], markdown=markdown.__version__, latex2mathml=__import__('latex2mathml').__version__, pymupdf=fitz.version[0], chrome=subprocess.run([CHROME, '--version'], capture_output=True, text=True).stdout.strip()),
           manuscript=cm, supplement=cs, supplement_heading_pages=hp, caption_pages=cap_pages, links_repaired=dict(manuscript=fixed_m, supplement=fixed_s),
           outputs={p: sha(V3/p) for p in ['manuscript.html', 'supplementary_material.html', 'manuscript_review.pdf', 'supplement_review.pdf']})
(V3/'internal/rendering_record.json').write_text(json.dumps(rec, indent=1))
insp = V3/'internal/page_inspection'; insp.mkdir(exist_ok=True)
for name, pages in [('manuscript_review.pdf', [1, 3, cap_pages.get('Table 2.', 7), cap_pages.get('Figure 1.', 8), cm['pages']]), ('supplement_review.pdf', [1, hp.get('S8', 7), hp.get('Table S12', 58), hp.get('Table S17', 77), cs['pages']])]:
    d = fitz.open(V3/name)
    for pno in sorted(set(pages)):
        d[pno - 1].get_pixmap(dpi=70).save(insp/f"{name.split('_')[0]}_p{pno:02d}.png")
    d.close()
print(json.dumps({k: v for k, v in rec.items() if k not in ('supplement_heading_pages',)}, indent=0)[:3000]); print('supplement headings:', hp)
