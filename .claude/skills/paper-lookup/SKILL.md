---
name: paper-lookup
description: Look up a paper's real bibliographic record and abstract from a DOI, an arXiv id, a title, or a PDF on disk -- via DOI content negotiation, Crossref, OpenAlex, arXiv and dblp, plus pdftotext for the document's own front matter. Use whenever writing or correcting a BibTeX entry, filling in a volume/issue/page/article number, identifying what an unnamed or badly-named PDF actually is, quoting an abstract, or checking that a citation is real. Also the mechanics behind literature/ADDING.md's sync procedure. Do NOT answer these from memory -- author lists and years are the fields memory gets wrong most confidently.
---

# Looking up bibliographic information

The rule this exists to enforce: **never write a citation field from memory.**
Titles and author lists are what recall is worst at while feeling most certain,
and a plausible-looking wrong page range is invisible on review. Every field
below comes from a machine-readable record or from the document itself.

Everything here needs only `curl`, `jq`, `python3` and `pdftotext`
(poppler-utils). No API keys. Be polite: put a contact address in the
User-Agent for bulk work (`-A 'yourtool (mailto:you@example.com)'`).

## Pick the entry point

| you have | start with |
|---|---|
| a DOI | DOI content negotiation (below) |
| an arXiv id | the arXiv API |
| a title | Crossref `query.title`, then follow the DOI |
| a PDF and nothing else | `pdfinfo`, then page 1, then a title search |
| a DOI that 404s on Crossref | content negotiation anyway -- it is probably DataCite |
| a missing article number | dblp |
| missing page ranges | OpenAlex |

## DOI content negotiation -- the first thing to try

Resolves against whichever registration agency owns the prefix, so it works for
Crossref (10.1145, 10.1109, 10.1016, 10.1007, 10.1111) *and* DataCite-ish
prefixes like Eurographics 10.2312, which `api.crossref.org` returns 404 for.

```bash
DOI=10.1145/3284980
curl -sL -H "Accept: application/vnd.citationstyles.csl+json" "https://doi.org/$DOI" | jq .
```

Useful fields: `.title`, `.author[]`, `.container-title`, `.volume`, `.issue`,
`.page`, `.issued.date-parts`, `.abstract`, `.type`.

BibTeX directly, if you want a starting point rather than fields:

```bash
curl -sL -H "Accept: application/x-bibtex" "https://doi.org/$DOI"
```

> **Do not paste that BibTeX in unread.** The Eurographics generator in
> particular emits garbage: for `10.2312/vmv.20191323` it splits the venue
> across two fields as `journal = {Vision}, pages = {Modeling and
> Visualization}`. Treat the BibTeX endpoint as a hint and the CSL JSON as the
> data.

## Crossref

Richer than content negotiation, and the only one of these with a good title
search. Full record:

```bash
curl -s "https://api.crossref.org/works/10.1145/3284980" | jq '.message | {title,volume,issue,page,"container-title",issued,publisher}'
```

Find a DOI from a title -- use `query.title`, **not** `query.bibliographic`,
which is fuzzy enough to return unrelated papers at rank 1:

```bash
curl -s -G "https://api.crossref.org/works" \
  --data-urlencode 'query.title=Ghost SPH for animating water' \
  --data 'rows=3&select=DOI,title,container-title,volume,issue,page,author' |
  jq -r '.message.items[] | "\(.DOI)  \(.title[0])  \(."container-title"[0]) \(.volume)(\(.issue)):\(.page)"'
```

Always eyeball the returned title and first author before adopting the DOI. A
title search *always* returns something; that something is often wrong.

**Crossref abstracts are JATS-tagged XML**, not plain text:

```bash
curl -s "https://api.crossref.org/works/$DOI" | jq -r '.message.abstract // empty' |
  python3 -c "import sys,re,html;t=sys.stdin.read();
t=re.sub(r'<jats:title>.*?</jats:title>','',t,flags=re.S);
print(re.sub(r'\s+',' ',html.unescape(re.sub(r'<[^>]+>','',t))).strip())"
```

Also normalise the typography it carries: Wiley records are full of U+2010
non-breaking hyphens (`neighbourhood‐queries`) that look like ordinary hyphens
and break every later grep.

Many records have **no** abstract (Elsevier and IEEE mostly do not deposit
them). That is not an error -- fall back to the PDF.

## OpenAlex -- when Crossref is missing pages

```bash
curl -s "https://api.openalex.org/works/https://doi.org/10.2312/vmv.20191323" |
  jq '{title, publication_year, biblio}'
```

`biblio.first_page` / `last_page` are frequently populated where Crossref's
`page` is null -- that is how VMV 2019's `99--107` was recovered.

## dblp -- when you need an ACM article number

ACM proceedings and TOG cite as `62:1-62:8`, and Crossref reports that as
`page: "1-8"` with the article number nowhere. dblp has it:

```bash
curl -s -A 'lit-lookup' "https://dblp.org/search/publ/api?q=Ghost+SPH+animating+water&format=json&h=3" |
  jq -r '.result.hits.hit[]?.info | "\(.title) | \(.venue) \(.volume)(\(.number)) \(.pages) \(.year)"'
# Ghost SPH for animating water. | ACM Trans. Graph. 31(4) 61:1-61:8 2012
```

> **dblp rate-limits hard and unhelpfully.** Several queries in quick
> succession give `429`, then plain `Connection reset by peer` for a while
> afterwards -- which looks like a network fault, not a rate limit. Space
> queries by seconds, send a User-Agent, and batch nothing. If you need twenty
> article numbers, read them off the PDFs' own ACM reference blocks instead
> (`grep -ohE 'ACM Trans\. Graph\.[^.]*\.' *.txt`), which is faster and free.

## arXiv

```bash
curl -s "https://export.arxiv.org/api/query?id_list=2507.21684" |
  python3 -c "import sys,re,html;x=sys.stdin.read()
for tag in ('title','summary'):
    m=re.findall(r'<%s>(.*?)</%s>'%(tag,tag),x,re.S)
    print(re.sub(r'\s+',' ',html.unescape(m[-1])).strip())"
```

Use `https://`, not `http://` -- the plaintext host intermittently returns an
empty body rather than redirecting. `<summary>` is the abstract, already plain
text. Note the *first* `<title>` in the feed is the query echo, so take the
last.

## Starting from a PDF

**Never trust the filename.** In one 36-paper sync, filenames were wrong about
the first author once, about the year twice, about publication status once, and
carried no information at all twice (`1-s2.0-S002199911200229X-main.pdf`).

Cheapest probe first -- the embedded metadata often carries the whole citation:

```bash
pdfinfo paper.pdf | grep -iE '^(Title|Author|Subject|Pages|CreationDate)'
# Subject: IEEE Transactions on Visualization and Computer Graphics;2014;20;3;10.1109/TVCG.2013.105
```

Elsevier and IEEE frequently put venue, year, volume, issue and DOI in
`Subject`. Wiley puts the volume and page range there. LaTeX-built preprints
usually put nothing useful, and `Creator: PDFium` means the file was re-saved by
a viewer and all original metadata is gone.

Then the front matter, and any DOI printed on it:

```bash
pdftotext -f 1 -l 2 -layout paper.pdf - | head -50
pdftotext -f 1 -l 2 paper.pdf - | grep -ohiE '(doi\.org/|DOI:? ?)10\.[0-9]{4,5}/[^ ,;)]+|arXiv:[0-9]{4}\.[0-9]{4,5}'
```

`-layout` preserves the visual arrangement, which is what you want for *reading*
a two-column first page. Plain `pdftotext` gives reading order, which is what
you want for *extracting* text. They disagree constantly; use both.

If no DOI is printed, take the title from page 1 and go to Crossref's
`query.title`.

### Last resort: scan a proceedings DOI range

Eurographics DOIs are sequential within a proceedings volume and are often not
printed in the paper. If you know the volume (`sca.2016….`), scan it:

```bash
for i in $(seq 20 45); do d=10.2312/sca.201612$i
  t=$(curl -sL -m 15 -H "Accept: application/vnd.citationstyles.csl+json" "https://doi.org/$d" | jq -r .title 2>/dev/null)
  echo "$d :: $t"
done
```

That found `10.2312/sca.20161222` in about twenty seconds. It also gives you the
whole session's table of contents, which is often worth having anyway.

## Extracting an abstract from a PDF

Quote it; do not summarise it. Three layout hazards, all of which silently
corrupt a naive extraction:

1. **Two-column abstracts scramble in reading order.** `pdftotext` may emit the
   left column, then the *last* paragraph, then the middle. Springer's
   author-version template does exactly this. Locate each paragraph
   individually and reassemble in the order the layout implies -- check against
   `-layout` output.
2. **ACM drops the copyright block into the middle of the abstract.** SCA and
   MIG papers put ~8 lines of "Permission to make digital or hard copies…"
   between two halves of the abstract text. Strip the block, then join.
3. **Ligatures and hyphenation.** Elsevier PDFs carry real ligature codepoints
   (`ﬁ ﬂ ﬀ`), and every publisher hyphenates across line breaks. Fix both before
   matching anything:

```python
for k, v in {'ﬁ':'fi','ﬂ':'fl','ﬀ':'ff','ﬃ':'ffi','ﬄ':'ffl'}.items():
    text = text.replace(k, v)
text = re.sub(r'-\n(?=[a-z])', '', text)   # join words split across lines
text = re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()
```

**Then verify the result is actually verbatim.** Normalise both the extracted
abstract and the raw page text to lowercase alphanumerics, and check the
abstract decomposes into a handful of contiguous runs that each appear in the
page. More than ~4 runs, or a run shorter than ~8 words, means the text drifted
into paraphrase. This repo ships that check as
`scripts/check_literature.py`; it caught a hand-transcribed DFSPH abstract
that had acquired three sentences the paper does not contain.

## Cross-checks worth doing

- **Does the record match the document in hand?** Normalise both and measure
  word overlap between the DOI-supplied abstract and the PDF's page 1. Real
  matches land at 94-100%. A low score means you followed the wrong DOI --
  usually to a same-titled extended version.
- **Conference paper or its journal extension?** These share titles, abstracts
  and author lists, and differ only in venue and year. Check the running header
  on page 2, not the title.
- **Author's-version PDFs carry placeholder fields.** ACM's template prints
  `ACM Trans. Graph. 1, 1, Article 1 (January 2020)` until typesetting fills it
  in. If the volume, issue and article number are all `1`, they are all wrong;
  take them from the DOI record and note in the entry that the copy is an
  author's version.
- **Two papers, one author-year key.** Common in this field (Band 2018 has two;
  Bender 2019 has two). Suffix the key with a topic word rather than `a`/`b`,
  which nobody can decode six months later.
