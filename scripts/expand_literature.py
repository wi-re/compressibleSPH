"""Pull the reference lists of the core literature set from OpenAlex and
aggregate them into a ranked list of one-step-removed candidate papers.

Usage:
    python scripts/expand_literature.py > literature/EXPANSION_CANDIDATES.md

Reads DOIs/arXiv ids straight out of literature/references.bib so the seed
set never has to be maintained twice. OpenAlex is used (not Semantic Scholar)
because ACM/Elsevier/IEEE publishers instruct Semantic Scholar to elide their
reference lists, but OpenAlex's `referenced_works` still carries them.
"""
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BIB = Path(__file__).resolve().parent.parent / "literature" / "references.bib"
API = "https://api.openalex.org/works"
MAILTO = "research@example.com"  # OpenAlex "polite pool" contact
UA = f"warpSPH-literature-expansion (mailto:{MAILTO})"


def parse_seed_papers(bib_text: str):
    """Return [(bibkey, id_type, id_value), ...] for every entry in the .bib."""
    seeds = []
    for block in re.split(r"\n(?=@)", bib_text):
        m_key = re.match(r"@\w+\{([^,]+),", block)
        if not m_key:
            continue
        key = m_key.group(1).strip()
        m_doi = re.search(r"doi\s*=\s*\{([^}]+)\}", block)
        if m_doi:
            seeds.append((key, "doi", m_doi.group(1).strip()))
            continue
        m_arxiv = re.search(r"eprint\s*=\s*\{([^}]+)\}", block)
        if m_arxiv:
            seeds.append((key, "arxiv", m_arxiv.group(1).strip()))
            continue
        # no identifier (e.g. some SPHERIC proceedings papers) -- skip


    return seeds


def fetch_json(url: str, retries: int = 4):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            sys.stderr.write(f"HTTP {e.code} for {url}\n")
            return None
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"error for {url}: {e}\n")
            time.sleep(2)
    return None


def arxiv_title(arxiv_id: str):
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            xml = resp.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return None
    m = re.findall(r"<title>(.*?)</title>", xml, re.S)
    return re.sub(r"\s+", " ", m[-1]).strip() if len(m) > 1 else None


def resolve_seed_work(id_type: str, id_value: str):
    if id_type == "doi":
        url = f"{API}/https://doi.org/{urllib.parse.quote(id_value)}?mailto={MAILTO}"
        return fetch_json(url)
    # arXiv preprints too new to be in OpenAlex's id index -- fall back to a title search
    title = arxiv_title(id_value)
    if not title:
        return None
    url = f"{API}?filter=title.search:{urllib.parse.quote(title)}&mailto={MAILTO}"
    data = fetch_json(url)
    if not data or not data.get("results"):
        return None
    return data["results"][0]


def main():
    seeds = parse_seed_papers(BIB.read_text())

    seed_work_ids = {}  # bibkey -> openalex work id
    seed_referenced = {}  # bibkey -> list of referenced_works ids
    unresolved = []

    for key, id_type, id_value in seeds:
        data = resolve_seed_work(id_type, id_value)
        time.sleep(0.15)
        if not data or "id" not in data:
            unresolved.append(key)
            continue
        seed_work_ids[key] = data["id"]
        seed_referenced[key] = data.get("referenced_works", [])

    all_seed_ids = set(seed_work_ids.values())

    # collect every unique referenced work id, remembering which seeds cite it
    cited_by = {}  # work_id -> set(bibkeys)
    for key, refs in seed_referenced.items():
        for wid in refs:
            if wid in all_seed_ids:
                continue  # a seed citing another seed -- already in our collection
            cited_by.setdefault(wid, set()).add(key)

    # batch-fetch details for all candidate works, 50 at a time (OpenAlex filter limit)
    work_ids = list(cited_by.keys())
    details = {}
    for i in range(0, len(work_ids), 50):
        batch = work_ids[i : i + 50]
        filt = "|".join(w.rsplit("/", 1)[-1] for w in batch)
        url = f"{API}?filter=openalex_id:{filt}&per-page=50&mailto={MAILTO}"
        data = fetch_json(url)
        time.sleep(0.2)
        if not data:
            continue
        for w in data.get("results", []):
            details[w["id"]] = w

    ranked = []
    for wid, keys in cited_by.items():
        w = details.get(wid)
        if not w:
            continue
        ids = w.get("ids", {})
        primary_loc = w.get("primary_location") or {}
        source = primary_loc.get("source") or {}
        ranked.append(
            {
                "title": w.get("title"),
                "year": w.get("publication_year"),
                "venue": source.get("display_name"),
                "authors": [
                    a.get("author", {}).get("display_name")
                    for a in w.get("authorships", [])
                ],
                "doi": ids.get("doi", "").replace("https://doi.org/", "") if ids.get("doi") else None,
                "oa_url": (w.get("open_access") or {}).get("oa_url"),
                "citations": w.get("cited_by_count") or 0,
                "cited_by": keys,
            }
        )
    # papers don't have their own h-index (that's an author/venue metric); citation
    # count is the per-paper analog, so sort on that within each core-citation tier
    ranked.sort(key=lambda e: (-len(e["cited_by"]), -e["citations"], -(e["year"] or 0)))

    print("# Literature expansion candidates\n")
    print(
        f"{len(ranked)} distinct papers cited by the {len(seed_work_ids)} resolved "
        f"seeds (of {len(seeds)} seeds with a DOI/arXiv id), excluding anything "
        "already in the core set.\n"
    )
    if unresolved:
        print(f"Seeds OpenAlex could not resolve: {', '.join(unresolved)}\n")
    print(
        "Split into three tiers by how many core papers cite the candidate -- a "
        "rough proxy for how central it is to this literature, not for its own "
        "quality. Tier 1 and 2 get full detail; tier 3 (cited once) is a compact "
        "table since it is mostly each seed's own background citations. Within a "
        "tier, papers are sorted by their own OpenAlex citation count (a paper has "
        "no h-index of its own -- that's an author/venue statistic -- so citation "
        "count is the per-paper stand-in for impact).\n"
    )

    tier1 = [e for e in ranked if len(e["cited_by"]) >= 3]
    tier2 = [e for e in ranked if len(e["cited_by"]) == 2]
    tier3 = [e for e in ranked if len(e["cited_by"]) == 1]

    def full_block(e):
        authors = ", ".join(a for a in e["authors"][:4] if a)
        if len(e["authors"]) > 4:
            authors += " et al."
        ident = f"doi:{e['doi']}" if e["doi"] else "no DOI"
        oa = f" | open access: {e['oa_url']}" if e["oa_url"] else ""
        print(f"## {e['title']} ({e['year']})")
        print(f"- authors: {authors}")
        print(f"- venue: {e['venue']}")
        print(f"- {ident}{oa}")
        print(f"- citations: {e['citations']}")
        print(f"- cited by ({len(e['cited_by'])}): {', '.join(sorted(e['cited_by']))}")
        print()

    print(f"## Tier 1 -- cited by 3 or more core papers ({len(tier1)})\n")
    for e in tier1:
        full_block(e)

    print(f"## Tier 2 -- cited by exactly 2 core papers ({len(tier2)})\n")
    for e in tier2:
        full_block(e)

    print(f"## Tier 3 -- cited by exactly 1 core paper ({len(tier3)})\n")
    print("| citations | year | title | doi | cited by |")
    print("|---|---|---|---|---|")
    for e in sorted(tier3, key=lambda e: (-e["citations"], -(e["year"] or 0))):
        title = (e["title"] or "").replace("|", "/")
        ident = e["doi"] or ""
        print(f"| {e['citations']} | {e['year']} | {title} | {ident} | {sorted(e['cited_by'])[0]} |")


if __name__ == "__main__":
    main()
