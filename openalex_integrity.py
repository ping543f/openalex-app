"""
Citation Integrity Screener (OpenAlex)
======================================

Screens one researcher's citation record in OpenAlex for patterns that are
often associated with citation manipulation:

  * incoming self-citation and co-author citation rates
  * citations concentrated in a few citing authors or journals
  * reciprocal "you cite me, I cite you" pairs (possible citation cartels)
  * outgoing self-citation in the researcher's own reference lists
  * retracted works, hyper-prolific years and sudden citation spikes

These are SCREENING SIGNALS, not proof of misconduct. Norms differ by field,
team size and career stage, and OpenAlex author disambiguation is imperfect.
Always read the underlying papers before drawing conclusions.

Run:
    pip install streamlit requests pandas
    export OPENALEX_API_KEY=your_key        # free at openalex.org/settings/api
    streamlit run openalex_integrity.py

Data: OpenAlex (CC0), https://openalex.org
"""

from __future__ import annotations

import html
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import pandas as pd
import requests

BASE = "https://api.openalex.org"
OR_BATCH = 50  # IDs per OR-filter; OpenAlex accepts up to 100, 50 keeps URLs short

# --------------------------------------------------------------------------
# Thresholds (amber, red). Tune these for your field.
# Self-citation benchmarks: Ioannidis et al., PLoS Biol 2019 (median ~12.7%,
# extreme > 40%). Hyper-prolific: Ioannidis et al., Nature 2018 (>= 72 papers
# in one year, i.e. one every five days).
# --------------------------------------------------------------------------
THRESHOLDS = {
    "self_in_pct": (25, 40),
    "coauthor_in_pct": (35, 50),
    "top_author_share_pct": (10, 20),
    "top_source_share_pct": (20, 35),
    "mutual_pairs": (1, 3),
    "self_out_pct": (25, 40),
    "hyperprolific_papers": (72, 120),
    "spike_ratio": (3, 5),
}
MUTUAL_MIN = 5          # citations each way before a pair counts as reciprocal
MIN_CITES_FOR_SHARE = 20  # don't judge concentration on tiny samples


class OAError(RuntimeError):
    pass


class BudgetExceeded(OAError):
    pass


def short(oid: str | None) -> str | None:
    """'https://openalex.org/W123' -> 'W123'."""
    return oid.rsplit("/", 1)[-1] if oid else oid


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# --------------------------------------------------------------------------
# API client
# --------------------------------------------------------------------------
class OpenAlex:
    def __init__(self, api_key: str | None = None, mailto: str | None = None, budget: int = 500):
        self.api_key = (api_key or "").strip() or None
        self.mailto = (mailto or "").strip() or None
        self.budget = budget
        self.calls = 0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "citation-integrity-screener/1.0"

    def get(self, path: str, params: dict | None = None) -> dict:
        if self.calls >= self.budget:
            raise BudgetExceeded(
                f"Stopped at the call budget ({self.budget} requests). "
                "Raise the budget in the sidebar or lower the caps."
            )
        p = dict(params or {})
        if self.api_key:
            p["api_key"] = self.api_key
        if self.mailto:
            p["mailto"] = self.mailto
        last = None
        for attempt in range(5):
            self.calls += 1
            try:
                r = self.session.get(BASE + path, params=p, timeout=60)
            except requests.RequestException as e:  # network hiccup
                last = str(e)
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (401, 403):
                raise OAError(
                    "OpenAlex refused the request. Check your API key "
                    "(get a free one at openalex.org/settings/api) or your daily allowance."
                )
            if r.status_code == 404:
                raise OAError("Not found in OpenAlex. Check the ID or ORCID.")
            if r.status_code >= 400:
                raise OAError(f"OpenAlex returned HTTP {r.status_code}: {r.text[:300]}")
            return r.json()
        raise OAError(f"OpenAlex did not respond after 5 tries ({last}).")

    def paged(self, path: str, params: dict, max_items: int) -> list[dict]:
        out, cursor = [], "*"
        while cursor and len(out) < max_items:
            per = min(200, max(1, max_items - len(out)))
            d = self.get(path, {**params, "per_page": per, "cursor": cursor})
            res = d.get("results") or []
            out.extend(res)
            cursor = (d.get("meta") or {}).get("next_cursor")
            if not res:
                break
        return out[:max_items]

    def works_by_ids(self, ids: list[str], select: str, max_items: int) -> list[dict]:
        """Fetch works by OpenAlex ID in OR-batches."""
        out = []
        for batch in chunks(ids[:max_items], OR_BATCH):
            d = self.get("/works", {"filter": "openalex:" + "|".join(batch),
                                    "select": select, "per_page": len(batch)})
            out.extend(d.get("results") or [])
        return out


# --------------------------------------------------------------------------
# Author lookup
# --------------------------------------------------------------------------
ORCID_RE = re.compile(r"(\d{4}-\d{4}-\d{4}-\d{3}[\dXx])")
OAID_RE = re.compile(r"\b(A\d{5,})\b", re.I)


def parse_query(q: str) -> tuple[str, str]:
    q = q.strip()
    m = ORCID_RE.search(q)
    if m:
        return "orcid", m.group(1).upper()
    m = OAID_RE.search(q)
    if m:
        return "openalex", m.group(1).upper()
    return "name", q


def institutions_of(a: dict) -> str:
    insts = a.get("last_known_institutions") or []
    if not insts and a.get("last_known_institution"):
        insts = [a["last_known_institution"]]
    return "; ".join(
        f"{i.get('display_name', '?')} ({i.get('country_code') or '?'})" for i in insts if i
    ) or "No institution listed"


def lookup(oa: OpenAlex, q: str) -> list[dict]:
    kind, val = parse_query(q)
    if kind == "orcid":
        return [oa.get(f"/authors/orcid:{val}")]
    if kind == "openalex":
        return [oa.get(f"/authors/{val}")]
    d = oa.get("/authors", {"search": val, "per_page": 10})
    return d.get("results") or []


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
@dataclass
class Report:
    author: dict
    metrics: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    partners: pd.DataFrame | None = None
    sources: pd.DataFrame | None = None
    coauthors: pd.DataFrame | None = None
    retracted: pd.DataFrame | None = None
    yearly: pd.DataFrame | None = None
    self_papers: pd.DataFrame | None = None   # one row per own paper: self vs external citations
    self_links: pd.DataFrame | None = None    # one row per self-citation (citing paper -> cited paper)
    self_by_year: pd.DataFrame | None = None  # self vs external citations by citing year
    notes: list = field(default_factory=list)
    calls: int = 0


def level(value, key):
    amber, red = THRESHOLDS[key]
    if value is None:
        return "grey"
    return "red" if value >= red else "amber" if value >= amber else "green"


def analyze(oa: OpenAlex, author: dict, max_works=300, max_citing=3000, max_refs=3000,
            progress=lambda msg, frac: None) -> Report:
    aid = short(author["id"])
    rep = Report(author=author)

    # 1) The researcher's own works -------------------------------------------
    progress("Fetching the researcher's works", 0.05)
    works = oa.paged(
        "/works",
        {"filter": f"author.id:{aid}", "sort": "cited_by_count:desc",
         "select": "id,display_name,publication_year,cited_by_count,referenced_works,"
                   "authorships,is_retracted,primary_location,doi,type"},
        max_works,
    )
    own = {short(w["id"]) for w in works}
    total_works = author.get("works_count") or len(works)
    if total_works > len(works):
        rep.notes.append(f"Analysed the {len(works)} most-cited of {total_works} works (cap).")

    coauthor_n, names = Counter(), {}
    for w in works:
        for au in w.get("authorships") or []:
            a = au.get("author") or {}
            bid = short(a.get("id"))
            if bid and bid != aid:
                coauthor_n[bid] += 1
                names[bid] = a.get("display_name") or bid
    coauthors = set(coauthor_n)

    # 2) Incoming citations ---------------------------------------------------
    progress("Fetching citing works", 0.2)
    cited_ids = [short(w["id"]) for w in works if (w.get("cited_by_count") or 0) > 0]
    expected = sum(w.get("cited_by_count") or 0 for w in works)
    citing: dict[str, dict] = {}
    for i, batch in enumerate(chunks(cited_ids, OR_BATCH)):
        if len(citing) >= max_citing:
            break
        res = oa.paged(
            "/works",
            {"filter": "cites:" + "|".join(batch),
             "select": "id,display_name,doi,publication_year,authorships,primary_location,referenced_works"},
            max_citing - len(citing),
        )
        for c in res:
            citing[short(c["id"])] = c
        progress("Fetching citing works", 0.2 + 0.4 * min(1, (i + 1) * OR_BATCH / max(1, len(cited_ids))))
    if len(citing) >= max_citing:
        rep.notes.append(f"Stopped at {max_citing} citing works (cap); results are a sample.")

    own_info = {}
    for w in works:
        src = ((w.get("primary_location") or {}).get("source") or {})
        own_info[short(w["id"])] = {
            "title": w.get("display_name") or "Untitled", "year": w.get("publication_year"),
            "venue": src.get("display_name") or "", "doi": w.get("doi") or "",
            "citations": w.get("cited_by_count") or 0}
    self_rows = []
    self_per_paper, ext_per_paper = Counter(), Counter()
    year_self, year_ext = Counter(), Counter()

    in_links = self_links = co_links = 0
    cites_in, cites_in_works = Counter(), Counter()
    by_source, source_names = Counter(), {}
    for cid, c in citing.items():
        refs = {short(r) for r in (c.get("referenced_works") or [])}
        targets = refs & own
        links = max(1, len(targets))
        authors_c = set()
        for au in c.get("authorships") or []:
            a = au.get("author") or {}
            bid = short(a.get("id"))
            if bid:
                authors_c.add(bid)
                names.setdefault(bid, a.get("display_name") or bid)
        in_links += links
        cyear = c.get("publication_year")
        csrc = ((c.get("primary_location") or {}).get("source") or {})
        if aid in authors_c:
            self_links += links
            if cyear:
                year_self[cyear] += links
            for t in targets:
                self_per_paper[t] += 1
                ti = own_info.get(t, {})
                self_rows.append({
                    "cited_paper": ti.get("title"), "cited_year": ti.get("year"),
                    "citing_paper": c.get("display_name") or "Untitled", "citing_year": cyear,
                    "years_later": (cyear - ti["year"]) if cyear and ti.get("year") else None,
                    "citing_venue": csrc.get("display_name") or "",
                    "citing_authors": len(authors_c),
                    "citing_doi": c.get("doi") or "",
                    "cited_work_id": t, "citing_work_id": cid})
            continue
        if cyear:
            year_ext[cyear] += links
        for t in targets:
            ext_per_paper[t] += 1
        if authors_c & coauthors:
            co_links += links
        for bid in authors_c:
            cites_in[bid] += links
            cites_in_works[bid] += 1
        src = ((c.get("primary_location") or {}).get("source") or {})
        sid = short(src.get("id")) or "unknown"
        by_source[sid] += links
        source_names[sid] = src.get("display_name") or "Unknown / not indexed"
    ext_links = in_links - self_links

    # 3) Outgoing references --------------------------------------------------
    progress("Fetching the researcher's references", 0.65)
    ref_count = Counter()
    for w in works:
        for r in {short(x) for x in (w.get("referenced_works") or [])}:
            ref_count[r] += 1
    ref_ids = [r for r, _ in ref_count.most_common()]
    if len(ref_ids) > max_refs:
        rep.notes.append(f"Resolved {max_refs} of {len(ref_ids)} referenced works (cap).")
    ref_works = oa.works_by_ids(ref_ids, "id,authorships", max_refs)
    out_links = self_out = 0
    cites_out = Counter()
    for rw in ref_works:
        n = ref_count[short(rw["id"])]
        authors_r = set()
        for au in rw.get("authorships") or []:
            a = au.get("author") or {}
            bid = short(a.get("id"))
            if bid:
                authors_r.add(bid)
                names.setdefault(bid, a.get("display_name") or bid)
        out_links += n
        if aid in authors_r:
            self_out += n
        for bid in authors_r - {aid}:
            cites_out[bid] += n

    # 4) Partners & reciprocity ----------------------------------------------
    progress("Scoring citation partners", 0.9)
    rows = []
    for bid in set(cites_in) | set(cites_out):
        cin, cout = cites_in[bid], cites_out[bid]
        if cin + cout < 2:
            continue
        mutual = min(cin, cout)
        rows.append({
            "author_id": bid,
            "name": names.get(bid, bid),
            "cites_researcher": cin,
            "citing_papers": cites_in_works[bid],
            "cited_by_researcher": cout,
            "reciprocal": mutual,
            "coauthor": bid in coauthors,
            "shared_papers": coauthor_n.get(bid, 0),
            "share_of_external_%": round(100 * cin / ext_links, 1) if ext_links else 0.0,
        })
    partners = pd.DataFrame(rows)
    if not partners.empty:
        partners = partners.sort_values(["reciprocal", "cites_researcher"], ascending=False)
        partners["pattern"] = partners.apply(_pattern, axis=1)
    rep.partners = partners

    rep.sources = pd.DataFrame(
        [{"source_id": s, "journal / venue": source_names[s], "citations": n,
          "share_%": round(100 * n / ext_links, 1) if ext_links else 0.0}
         for s, n in by_source.most_common(25)]
    )
    rep.coauthors = pd.DataFrame(
        [{"author_id": b, "name": names.get(b, b), "shared_papers": n,
          "cites_researcher": cites_in[b], "cited_by_researcher": cites_out[b]}
         for b, n in coauthor_n.most_common(40)]
    )
    rep.retracted = pd.DataFrame(
        [{"title": w.get("display_name"), "year": w.get("publication_year"),
          "doi": w.get("doi"), "citations": w.get("cited_by_count")}
         for w in works if w.get("is_retracted")]
    )
    cby = sorted(author.get("counts_by_year") or [], key=lambda r: r["year"])
    rep.yearly = pd.DataFrame(cby)

    sp = []
    for wid, info in own_info.items():
        s_, e_ = self_per_paper[wid], ext_per_paper[wid]
        if s_ + e_ == 0:
            continue
        sp.append({"work_id": wid, "title": info["title"], "year": info["year"], "venue": info["venue"],
                   "doi": info["doi"], "openalex_citations": info["citations"],
                   "self_citations": s_, "external_citations": e_,
                   "self_share_%": round(100 * s_ / (s_ + e_), 1)})
    rep.self_papers = (pd.DataFrame(sp).sort_values(["self_citations", "self_share_%"], ascending=False)
                       .reset_index(drop=True) if sp else pd.DataFrame())
    rep.self_links = (pd.DataFrame(self_rows).sort_values(["cited_year", "cited_paper", "citing_year"])
                      .reset_index(drop=True) if self_rows else pd.DataFrame())
    yrs = sorted(set(year_self) | set(year_ext))
    rep.self_by_year = pd.DataFrame({"year": yrs, "self": [year_self[y] for y in yrs],
                                     "external": [year_ext[y] for y in yrs]})

    # 5) Metrics ----------------------------------------------------------------
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    top_author = partners.sort_values("cites_researcher", ascending=False).iloc[0] if not partners.empty else None
    top_source = rep.sources.iloc[0] if not rep.sources.empty else None
    mutual_noncoauthor = 0 if partners.empty else int(
        ((partners.cites_researcher >= MUTUAL_MIN) & (partners.cited_by_researcher >= MUTUAL_MIN)
         & (~partners.coauthor)).sum())
    mutual_coauthor = 0 if partners.empty else int(
        ((partners.cites_researcher >= MUTUAL_MIN) & (partners.cited_by_researcher >= MUTUAL_MIN)
         & (partners.coauthor)).sum())
    max_year_works = max((r.get("works_count") or 0 for r in cby), default=0)
    spike, spike_year = _citation_spike(cby)

    m = rep.metrics
    m["works_analysed"] = len(works)
    m["citing_works"] = len(citing)
    m["citation_links"] = in_links
    m["citations_expected"] = expected
    m["self_in_pct"] = pct(self_links, in_links)
    m["coauthor_in_pct"] = pct(co_links, ext_links)
    m["top_author_share_pct"] = float(top_author["share_of_external_%"]) if top_author is not None and ext_links >= MIN_CITES_FOR_SHARE else None
    m["top_author_name"] = top_author["name"] if top_author is not None else None
    m["top_source_share_pct"] = float(top_source["share_%"]) if top_source is not None and ext_links >= MIN_CITES_FOR_SHARE else None
    m["top_source_name"] = top_source["journal / venue"] if top_source is not None else None
    m["mutual_pairs"] = mutual_noncoauthor
    m["mutual_coauthor_pairs"] = mutual_coauthor
    m["self_out_pct"] = pct(self_out, out_links)
    m["references_resolved"] = out_links
    m["retracted"] = len(rep.retracted)
    m["hyperprolific_papers"] = max_year_works
    m["spike_ratio"] = spike
    m["spike_year"] = spike_year
    rep.calls = oa.calls

    rep.flags = _flags(m)
    progress("Done", 1.0)
    return rep


def _pattern(r) -> str:
    if r.cites_researcher >= MUTUAL_MIN and r.cited_by_researcher >= MUTUAL_MIN:
        return "Reciprocal (co-author)" if r.coauthor else "Reciprocal, not co-author"
    if r.cites_researcher >= MUTUAL_MIN and r.cited_by_researcher <= 1:
        return "One-way: cites researcher heavily"
    if r.cited_by_researcher >= MUTUAL_MIN and r.cites_researcher <= 1:
        return "One-way: researcher cites them heavily"
    return "Ordinary"


def _citation_spike(cby):
    """Largest year-on-year jump in citations received (ratio), ignoring the current partial year."""
    rows = [r for r in cby if r.get("year")]
    this_year = time.gmtime().tm_year
    rows = [r for r in rows if r["year"] < this_year]
    best, best_year = None, None
    for prev, cur in zip(rows, rows[1:]):
        p, c = prev.get("cited_by_count") or 0, cur.get("cited_by_count") or 0
        if c >= 50 and p > 0:
            ratio = round(c / p, 1)
            if best is None or ratio > best:
                best, best_year = ratio, cur["year"]
    return best, best_year


def _flags(m):
    def f(key, title, value_txt, why):
        return {"key": key, "level": level(m.get(key), key), "title": title,
                "value": value_txt, "why": why}

    out = [
        f("self_in_pct", "Self-citation (incoming)",
          "n/a" if m["self_in_pct"] is None else f"{m['self_in_pct']}% of citations",
          "Share of citations to this researcher that come from papers they co-wrote."),
        f("coauthor_in_pct", "Co-author citation (incoming)",
          "n/a" if m["coauthor_in_pct"] is None else f"{m['coauthor_in_pct']}% of external citations",
          "Citations from papers written by past co-authors. High in large teams; read with context."),
        f("top_author_share_pct", "Single-author concentration",
          "too few citations" if m["top_author_share_pct"] is None else f"{m['top_author_share_pct']}% from {m['top_author_name']}",
          "One outside author supplying a large share of citations can indicate citation pushing."),
        f("top_source_share_pct", "Single-journal concentration",
          "too few citations" if m["top_source_share_pct"] is None else f"{m['top_source_share_pct']}% from {m['top_source_name']}",
          "Citations clustered in one venue can indicate citation stacking by a journal."),
        f("mutual_pairs", "Reciprocal citation pairs (non-co-authors)",
          f"{m['mutual_pairs']} pair(s)" + (f", plus {m['mutual_coauthor_pairs']} with co-authors" if m['mutual_coauthor_pairs'] else ""),
          f"Outside authors who cite this researcher and are cited back at least {MUTUAL_MIN} times each way. The core cartel signal."),
        f("self_out_pct", "Self-citation (in own reference lists)",
          "n/a" if m["self_out_pct"] is None else f"{m['self_out_pct']}% of references",
          "Share of the researcher's references that point to their own work."),
        f("hyperprolific_papers", "Peak yearly output",
          f"{m['hyperprolific_papers']} works in one year",
          "72+ papers a year (one every five days) is the published hyper-prolific threshold."),
        f("spike_ratio", "Citation spike",
          "none" if m["spike_ratio"] is None else f"×{m['spike_ratio']} in {m['spike_year']}",
          "A sudden multiple-fold jump in citations received in one year."),
    ]
    out.append({"key": "retracted", "level": "red" if m["retracted"] else "green",
                "title": "Retracted works", "value": f"{m['retracted']} found",
                "why": "Works OpenAlex marks as retracted (sourced from Retraction Watch data)."})
    return out


# --------------------------------------------------------------------------
# Network diagram: radial ego network drawn as self-contained SVG
#   centre  = the researcher
#   ring    = top citation partners, grouped by pattern, clockwise
#   arrows  = "in" (partner cites researcher) and "out" (researcher cites
#             partner), each curved to its own side so both stay readable
#   labels  = exact in / out counts on every partner
# --------------------------------------------------------------------------

NET_W, NET_H = 1120, 880
RANK_OPTIONS = ["Reciprocity first", "Total citations", "Incoming", "Outgoing"]
_CAT_ORDER = {"Reciprocal, not co-author": 0, "Reciprocal (co-author)": 1,
              "One-way: cites researcher heavily": 2,
              "One-way: researcher cites them heavily": 3, "Ordinary": 4}


def select_partners(p: pd.DataFrame | None, top_n: int = 14, rank: str = "Reciprocity first",
                    hide_coauthors: bool = False) -> pd.DataFrame:
    if p is None or p.empty:
        return pd.DataFrame()
    df = p.copy()
    df["cites_researcher"] = pd.to_numeric(df["cites_researcher"], errors="coerce").fillna(0).astype(int)
    df["cited_by_researcher"] = pd.to_numeric(df["cited_by_researcher"], errors="coerce").fillna(0).astype(int)
    df = df[(df.cites_researcher + df.cited_by_researcher) > 0]
    if hide_coauthors:
        df = df[~df.coauthor.astype(bool)]
    df["total"] = df.cites_researcher + df.cited_by_researcher
    df["reciprocal"] = df[["cites_researcher", "cited_by_researcher"]].min(axis=1)
    keys = {"Reciprocity first": ["reciprocal", "total"], "Total citations": ["total", "reciprocal"],
            "Incoming": ["cites_researcher", "total"], "Outgoing": ["cited_by_researcher", "total"]}
    df = df.sort_values(keys.get(rank, keys["Reciprocity first"]), ascending=False).head(max(1, top_n))
    if "pattern" not in df:
        df["pattern"] = df.apply(_pattern, axis=1)
    df["_cat"] = df["pattern"].map(_CAT_ORDER).fillna(4)
    return df.sort_values(["_cat", "total"], ascending=[True, False]).reset_index(drop=True)


def _xml(s) -> str:
    return html.escape(str(s), quote=True)


def _trim(s, n=24) -> str:
    s = str(s or "Unknown")
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _wrap2(name: str, n=16) -> list[str]:
    words, lines, cur = str(name).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    if len(lines) > 2:
        lines = [lines[0], _trim(" ".join(lines[1:]), n)]
    return lines


def network_svg(rep: Report, top_n: int = 14, rank: str = "Reciprocity first",
                hide_coauthors: bool = False) -> str:
    """Self-contained SVG (inline styles + light/dark tokens)."""
    df = select_partners(rep.partners, top_n, rank, hide_coauthors)
    grow = max(0, len(df) - 14)          # more partners -> larger canvas, same spacing
    W, H = NET_W + grow * 46, NET_H + grow * 30
    cx, cy = W / 2, H / 2 + 14
    rx, ry = 300 + grow * 13, 250 + grow * 14
    hub_r = 60
    me = rep.author.get("display_name") or "Researcher"

    css = """
    .net{--bg:#fbfbf9;--fg:#1d232b;--muted:#5d6672;--hair:#e3e5e8;--node:#ffffff;--hub:#1f3b57;--hubfg:#ffffff;
         --in:#1f7a8c;--out:#8a5a9e;--red:#b3261e;--amber:#b86e00;--plain:#a3abb5;
         font-family:"Source Sans 3","Segoe UI",Helvetica,Arial,sans-serif}
    @media (prefers-color-scheme: dark){.net{--bg:#171b21;--fg:#e8eaed;--muted:#9aa3ae;--hair:#2c323a;--node:#222831;
         --hub:#3d6b94;--hubfg:#ffffff;--in:#4fb3c6;--out:#b98bd0;--red:#ef6a5f;--amber:#e3a03a;--plain:#6b7480}}
    .net text{fill:var(--fg)} .net .muted{fill:var(--muted)} .net .nin{fill:var(--in);font-weight:700}
    .net .nout{fill:var(--out);font-weight:700} .net .tag-red{fill:var(--red);font-weight:600}
    .net .tag-amber{fill:var(--amber);font-weight:600}
    .net .edge{fill:none;stroke-linecap:butt;opacity:.78} .net .edge.in{stroke:var(--in)}
    .net .edge.out{stroke:var(--out);stroke-dasharray:7 5}
    .net g.p:hover .edge{opacity:1} .net g.p:hover circle.dot{stroke-width:4}
    """
    out = [f'<svg class="net" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'width="100%" preserveAspectRatio="xMidYMid meet" role="img" '
           f'aria-label="Citation network of {_xml(me)}">',
           f"<style>{css}</style>",
           '<defs>'
           '<marker id="ah-in" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="11" markerHeight="11" '
           'markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--in)"/></marker>'
           '<marker id="ah-out" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="11" markerHeight="11" '
           'markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--out)"/></marker>'
           '</defs>',
           f'<rect width="{W}" height="{H}" rx="14" style="fill:var(--bg)"/>']

    # header
    tot_in = int(df.cites_researcher.sum()) if not df.empty else 0
    tot_out = int(df.cited_by_researcher.sum()) if not df.empty else 0
    out.append(f'<text x="32" y="44" style="font-size:22px;font-weight:700">Citation partners of {_xml(_trim(me, 40))}</text>')
    out.append(f'<text x="32" y="70" class="muted" style="font-size:14px">'
               f'Top {len(df)} by {_xml(rank.lower())}{" · co-authors hidden" if hide_coauthors else ""} · '
               f'shown partners: <tspan class="nin">{tot_in} in</tspan> · <tspan class="nout">{tot_out} out</tspan></text>')

    if df.empty:
        out.append(f'<text x="{cx}" y="{cy}" text-anchor="middle" class="muted" style="font-size:16px">'
                   'No citation partners to show for this record.</text></svg>')
        return "".join(out)

    n = len(df)
    max_tot = max(1, int(df.total.max()))
    max_edge = max(1, int(max(df.cites_researcher.max(), df.cited_by_researcher.max())))
    stagger = 46 if n > 10 else 0
    start = -math.pi / 2 + (math.pi / n)  # avoid a node sitting exactly at 12 o'clock

    # faint guide ring
    out.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" style="fill:none;stroke:var(--hair);stroke-width:1"/>')

    def ew(c):
        return 1.4 + 5.6 * math.sqrt(c / max_edge)

    nodes = []
    for i, r in df.iterrows():
        t = start + 2 * math.pi * i / n
        k = stagger if i % 2 else 0
        x, y = cx + (rx + k) * math.cos(t), cy + (ry + k * 0.8) * math.sin(t)
        nr = 8 + 13 * math.sqrt(r.total / max_tot)
        nodes.append((r, t, x, y, nr))

    # ---- label placement (computed first so collisions can be resolved) ----
    labels = []
    for r, t, x, y, nr in nodes:
        c = math.cos(t)
        anchor = "start" if c > 0.25 else "end" if c < -0.25 else "middle"
        lx = x + (nr + 10) * (1 if anchor == "start" else -1 if anchor == "end" else 0)
        if anchor == "middle":
            ly = y + (nr + 18 if math.sin(t) > 0 else -(nr + 26))
        else:
            ly = y - 4
        if r.pattern == "Reciprocal, not co-author":
            extra = " · reciprocal"
        elif r.coauthor:
            extra = f" · co-author ({int(r.get('shared_papers', 0))})"
        else:
            extra = ""
        line2 = f"{r.cites_researcher} in · {r.cited_by_researcher} out{extra}"
        w = max(len(_trim(r['name'])) * 8.2, len(line2) * 7.2)
        x0 = lx if anchor == "start" else lx - w if anchor == "end" else lx - w / 2
        labels.append({"anchor": anchor, "lx": lx, "ly": ly, "ly0": ly, "x0": x0, "x1": x0 + w, "extra": extra})

    for _ in range(120):  # simple vertical relaxation of overlapping label boxes
        moved = False
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                if a["x1"] < b["x0"] - 6 or b["x1"] < a["x0"] - 6:
                    continue
                at, ab, bt, bb = a["ly"] - 15, a["ly"] + 23, b["ly"] - 15, b["ly"] + 23
                ov = min(ab, bb) - max(at, bt)
                if ov > -4:
                    shift = (ov + 4) / 2 + 0.5
                    up, dn = (a, b) if a["ly"] <= b["ly"] else (b, a)
                    up["ly"] -= shift
                    dn["ly"] += shift
                    moved = True
        for k, lb in enumerate(labels):  # keep labels off other partners' nodes
            for m_, (_, _, ox, oy, onr) in enumerate(nodes):
                if m_ == k or lb["x1"] < ox - onr - 4 or lb["x0"] > ox + onr + 4:
                    continue
                top, bot = lb["ly"] - 15, lb["ly"] + 23
                ov = min(bot, oy + onr + 4) - max(top, oy - onr - 4)
                if ov > 0:
                    lb["ly"] += (ov + 1) if lb["ly"] > oy else -(ov + 1)
                    moved = True
            lb["ly"] = min(max(lb["ly"], 104), H - 84)
        if not moved:
            break

    # ---- partners: edges, node, label (grouped so hover highlights a partner) ----
    for (r, t, x, y, nr), lb in zip(nodes, labels):
        recip_nc = r.pattern == "Reciprocal, not co-author"
        recip_co = r.pattern == "Reciprocal (co-author)"
        ring = "var(--red)" if recip_nc else "var(--amber)" if recip_co else "var(--plain)"
        dx, dy = cx - x, cy - y
        d = math.hypot(dx, dy) or 1
        ux, uy = dx / d, dy / d          # partner -> hub
        px, py = -uy, ux                 # perpendicular
        ang = math.atan2(-dy, -dx)       # hub -> partner angle
        tip = (
            f'{r["name"]}\nCites the researcher: {r.cites_researcher} '
            f'({int(r.get("citing_papers", 0) or 0)} papers)\nCited by the researcher: {r.cited_by_researcher}'
            f'\n{"Co-author, " + str(int(r.get("shared_papers", 0) or 0)) + " shared papers" if r.coauthor else "Not a co-author"}'
            f'\nPattern: {r.pattern}')
        g = [f'<g class="p"><title>{_xml(tip)}</title>']
        for direction, count in (("in", r.cites_researcher), ("out", r.cited_by_researcher)):
            if count <= 0:
                continue
            side = 1 if direction == "in" else -1
            off = 0.09 * side
            # endpoints nudged around each circle so the two arrows never share a point
            hx, hy = cx + (hub_r + 4) * math.cos(ang + off), cy + (hub_r + 4) * math.sin(ang + off)
            nx_, ny_ = x - (nr + 4) * ux + px * 3 * side, y - (nr + 4) * uy + py * 3 * side
            mx, my = (hx + nx_) / 2 + px * 22 * side, (hy + ny_) / 2 + py * 22 * side
            if direction == "in":
                path = f"M{nx_:.1f},{ny_:.1f} Q{mx:.1f},{my:.1f} {hx:.1f},{hy:.1f}"
            else:
                path = f"M{hx:.1f},{hy:.1f} Q{mx:.1f},{my:.1f} {nx_:.1f},{ny_:.1f}"
            g.append(f'<path class="edge {direction}" d="{path}" style="stroke-width:{ew(count):.2f}" '
                     f'marker-end="url(#ah-{direction})"/>')
        lx, ly = lb["lx"], lb["ly"]
        if abs(ly - lb["ly0"]) > 10:  # leader line when a label had to move
            tx = lx - 4 if lb["anchor"] == "start" else lx + 4 if lb["anchor"] == "end" else lx
            g.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{tx:.1f}" y2="{ly - 5:.1f}" '
                     f'style="stroke:var(--plain);stroke-width:1;opacity:.7"/>')
        g.append(f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="{nr:.1f}" '
                 f'style="fill:var(--node);stroke:{ring};stroke-width:{3 if (recip_nc or recip_co) else 2}"/>')
        if recip_nc:
            tag = '<tspan class="tag-red"> · reciprocal</tspan>'
        elif lb["extra"]:
            tag = f'<tspan class="muted">{_xml(lb["extra"])}</tspan>'
        else:
            tag = ""
        g.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{lb["anchor"]}" style="font-size:14px">'
            f'<tspan style="font-weight:600">{_xml(_trim(r["name"]))}</tspan>'
            f'<tspan x="{lx:.1f}" dy="18" style="font-size:13px">'
            f'<tspan class="nin">{r.cites_researcher} in</tspan><tspan class="muted"> · </tspan>'
            f'<tspan class="nout">{r.cited_by_researcher} out</tspan>{tag}</tspan></text>')
        g.append("</g>")
        out.extend(g)

    # hub on top
    out.append(f'<circle cx="{cx}" cy="{cy}" r="{hub_r}" style="fill:var(--hub)"/>')
    lines = _wrap2(me)
    y0 = cy - (len(lines) - 1) * 9 + 5
    for j, ln in enumerate(lines):
        out.append(f'<text x="{cx}" y="{y0 + j * 18:.1f}" text-anchor="middle" '
                   f'style="fill:var(--hubfg);font-size:15px;font-weight:700">{_xml(ln)}</text>')

    # legend
    ly = H - 34
    out.append(
        f'<g style="font-size:13px">'
        f'<line x1="32" y1="{ly}" x2="68" y2="{ly}" class="edge in" style="stroke-width:3" marker-end="url(#ah-in)"/>'
        f'<text x="76" y="{ly + 4}" class="muted">in: partner cites researcher</text>'
        f'<line x1="290" y1="{ly}" x2="326" y2="{ly}" class="edge out" style="stroke-width:3" marker-end="url(#ah-out)"/>'
        f'<text x="334" y="{ly + 4}" class="muted">out: researcher cites partner</text>'
        f'<circle cx="566" cy="{ly}" r="8" style="fill:var(--node);stroke:var(--red);stroke-width:3"/>'
        f'<text x="580" y="{ly + 4}" class="muted">reciprocal, not co-author (≥{MUTUAL_MIN} each way)</text>'
        f'<circle cx="876" cy="{ly}" r="8" style="fill:var(--node);stroke:var(--amber);stroke-width:3"/>'
        f'<text x="890" y="{ly + 4}" class="muted">reciprocal co-author</text>'
        f'</g>')
    out.append(f'<text x="{W - 32}" y="{H - 12}" text-anchor="end" class="muted" style="font-size:11px">'
               'Node size = total citations exchanged · line width = citations in that direction · data: OpenAlex</text>')
    out.append("</svg>")
    return "".join(out)


SELF_SORT_OPTIONS = ["Most self-citations", "Highest self-share", "Newest first"]


def select_self_papers(sp: pd.DataFrame | None, top_n: int = 15, sort: str = "Most self-citations",
                       only_with_self: bool = True) -> pd.DataFrame:
    if sp is None or sp.empty:
        return pd.DataFrame()
    df = sp.copy()
    if only_with_self:
        df = df[df.self_citations > 0]
    keys = {"Most self-citations": (["self_citations", "self_share_%"], False),
            "Highest self-share": (["self_share_%", "self_citations"], False),
            "Newest first": (["year", "self_citations"], False)}
    cols, asc = keys.get(sort, keys["Most self-citations"])
    return df.sort_values(cols, ascending=asc).head(max(1, top_n)).reset_index(drop=True)


def _nice_ticks(vmax: float, n: int = 5) -> list[int]:
    if vmax <= 0:
        return [0]
    raw = vmax / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    step = max(1, int(round(step)))
    return list(range(0, int(vmax) + step, step))


def self_citation_svg(rep: Report, top_n: int = 15, sort: str = "Most self-citations",
                      only_with_self: bool = True) -> str:
    """Per-paper self-citation chart: title, year, venue and an exact self / external split."""
    df = select_self_papers(rep.self_papers, top_n, sort, only_with_self)
    me = rep.author.get("display_name") or "Researcher"
    W = 1120
    left, bar_x0, bar_x1 = 32, 600, 930
    row_h, top = 54, 128
    H = top + max(1, len(df)) * row_h + 92

    css = """
    .sc{--bg:#fbfbf9;--fg:#1d232b;--muted:#5d6672;--hair:#e6e8eb;--track:#eef0f2;
        --self:#c4442b;--ext:#9fb4c7;--amber:#b86e00;--red:#b3261e;--ok:#2f7a55;
        font-family:"Source Sans 3","Segoe UI",Helvetica,Arial,sans-serif}
    @media (prefers-color-scheme: dark){.sc{--bg:#171b21;--fg:#e8eaed;--muted:#9aa3ae;--hair:#2c323a;--track:#222831;
        --self:#f0765d;--ext:#4e6478;--amber:#e3a03a;--red:#ef6a5f;--ok:#62b88a}}
    .sc text{fill:var(--fg)} .sc .muted{fill:var(--muted)}
    .sc .self{fill:var(--self)} .sc .ext{fill:var(--ext)}
    .sc g.row:hover rect.hl{fill:var(--track)}
    """
    out = [f'<svg class="sc" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" '
           f'preserveAspectRatio="xMidYMin meet" role="img" aria-label="Self-citation by paper for {_xml(me)}">',
           f"<style>{css}</style>",
           f'<rect width="{W}" height="{H}" rx="14" style="fill:var(--bg)"/>']

    sp = rep.self_papers if rep.self_papers is not None else pd.DataFrame()
    tot_self = int(sp.self_citations.sum()) if not sp.empty else 0
    tot_ext = int(sp.external_citations.sum()) if not sp.empty else 0
    n_papers = int((sp.self_citations > 0).sum()) if not sp.empty else 0
    share = round(100 * tot_self / (tot_self + tot_ext), 1) if tot_self + tot_ext else 0
    out.append(f'<text x="{left}" y="44" style="font-size:22px;font-weight:700">Self-citation by paper · {_xml(_trim(me, 40))}</text>')
    out.append(f'<text x="{left}" y="70" class="muted" style="font-size:14px">'
               f'<tspan class="self" style="font-weight:700">{tot_self} self-citations</tspan> to '
               f'{n_papers} papers · {tot_ext} external citations in the sample · '
               f'overall self-share <tspan style="font-weight:700">{share}%</tspan></text>')

    if df.empty:
        out.append(f'<text x="{W/2}" y="{top + 20}" text-anchor="middle" class="muted" style="font-size:16px">'
                   'No self-citations found in the analysed sample.</text></svg>')
        return "".join(out)

    vmax = max(1, int((df.self_citations + df.external_citations).max()))
    ticks = _nice_ticks(vmax)
    vmax = max(vmax, ticks[-1])
    sx = lambda v: bar_x0 + (bar_x1 - bar_x0) * v / vmax

    # column heads + grid
    out.append(f'<text x="{left}" y="{top - 22}" class="muted" style="font-size:12px;letter-spacing:.06em">'
               f'PAPER · YEAR · VENUE</text>')
    out.append(f'<text x="{bar_x0}" y="{top - 22}" class="muted" style="font-size:12px;letter-spacing:.06em">'
               f'CITATIONS IN SAMPLE</text>')
    out.append(f'<text x="{W - 32}" y="{top - 22}" text-anchor="end" class="muted" '
               f'style="font-size:12px;letter-spacing:.06em">SELF · TOTAL · SHARE</text>')
    grid_bottom = top + len(df) * row_h
    for t in ticks:
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{grid_bottom}" '
                   f'style="stroke:var(--hair);stroke-width:1"/>')
        out.append(f'<text x="{x:.1f}" y="{grid_bottom + 18}" text-anchor="middle" class="muted" '
                   f'style="font-size:11px;font-variant-numeric:tabular-nums">{t}</text>')

    links = rep.self_links if rep.self_links is not None else pd.DataFrame()
    for i, r in df.iterrows():
        y = top + i * row_h
        s_, e_ = int(r.self_citations), int(r.external_citations)
        pct = float(r["self_share_%"])
        pct_col = "var(--red)" if pct >= 40 else "var(--amber)" if pct >= 25 else "var(--muted)"
        meta = " · ".join(str(v) for v in [r.year if pd.notna(r.year) else "", _trim(r.venue, 46),
                                           f"{int(r.openalex_citations)} citations in OpenAlex"] if v)
        tip_lines = [f"{r.title}", f"{r.year} · {r.venue}".strip(" ·"),
                     f"Self-citations: {s_} · External: {e_} · Self-share: {pct}%"]
        if not links.empty:
            cites = links[links.cited_work_id == r.work_id].sort_values("citing_year")
            if len(cites):
                tip_lines.append("Cited by the researcher's own papers:")
                for _, c in cites.head(10).iterrows():
                    tip_lines.append(f"  {c.citing_year} · {_trim(c.citing_paper, 80)}")
                if len(cites) > 10:
                    tip_lines.append(f"  … and {len(cites) - 10} more (see table)")
        out.append(f'<g class="row"><title>{_xml(chr(10).join(tip_lines))}</title>')
        out.append(f'<rect class="hl" x="16" y="{y}" width="{W - 32}" height="{row_h - 4}" rx="6" '
                   f'style="fill:transparent"/>')
        out.append(f'<text x="{left}" y="{y + 21}" style="font-size:14px;font-weight:600">{_xml(_trim(r.title, 66))}</text>')
        out.append(f'<text x="{left}" y="{y + 40}" class="muted" style="font-size:12.5px">{_xml(meta)}</text>')
        bh, by = 16, y + 17
        out.append(f'<rect x="{bar_x0}" y="{by}" width="{bar_x1 - bar_x0}" height="{bh}" rx="3" style="fill:var(--track)"/>')
        if s_:
            out.append(f'<rect class="self" x="{bar_x0}" y="{by}" width="{max(2, sx(s_) - bar_x0):.1f}" height="{bh}" rx="3"/>')
        if e_:
            out.append(f'<rect class="ext" x="{sx(s_) + (1 if s_ else 0):.1f}" y="{by}" '
                       f'width="{max(2, sx(s_ + e_) - sx(s_) - (1 if s_ else 0)):.1f}" height="{bh}" rx="3"/>')
        out.append(f'<text x="{W - 32}" y="{by + 13}" text-anchor="end" '
                   f'style="font-size:14px;font-variant-numeric:tabular-nums">'
                   f'<tspan class="self" style="font-weight:700">{s_}</tspan>'
                   f'<tspan class="muted"> of {s_ + e_} · </tspan>'
                   f'<tspan style="fill:{pct_col};font-weight:700">{pct:g}%</tspan></text>')
        out.append("</g>")

    ly = H - 34
    out.append(
        f'<g style="font-size:13px">'
        f'<rect x="{left}" y="{ly - 7}" width="22" height="12" rx="2" class="self"/>'
        f'<text x="{left + 30}" y="{ly + 4}" class="muted">self-citation (citing paper includes the researcher)</text>'
        f'<rect x="{left + 400}" y="{ly - 7}" width="22" height="12" rx="2" class="ext"/>'
        f'<text x="{left + 430}" y="{ly + 4}" class="muted">external citation</text>'
        f'<text x="{left + 580}" y="{ly + 4}" class="muted">share ≥25% amber · ≥40% red</text></g>')
    out.append(f'<text x="{W - 32}" y="{H - 12}" text-anchor="end" class="muted" style="font-size:11px">'
               'Counts are citing papers in the analysed sample · hover a row for the citing papers · data: OpenAlex</text>')
    out.append("</svg>")
    return "".join(out)


def network_html(svg: str) -> str:
    """Wrap the SVG for st.components (its own document, so the dark-mode query works)."""
    return ('<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;background:transparent}'
            'svg{display:block}</style></head><body>' + svg + '</body></html>')




# --------------------------------------------------------------------------
# HTML report (for download / sharing with a committee)
# --------------------------------------------------------------------------
def html_report(rep: Report) -> str:
    a = rep.author
    colors = {"green": "#1e7b4f", "amber": "#a86400", "red": "#b3261e", "grey": "#6b7280"}
    flag_rows = "".join(
        f"<tr><td><b style='color:{colors[f['level']]}'>{f['level'].upper()}</b></td>"
        f"<td>{html.escape(f['title'])}</td><td>{html.escape(str(f['value']))}</td>"
        f"<td>{html.escape(f['why'])}</td></tr>" for f in rep.flags)
    tbl = lambda df: "<p>None.</p>" if df is None or df.empty else df.to_html(index=False, escape=True, border=0)
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in rep.notes)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Citation screen: {html.escape(a.get('display_name',''))}</title>
<style>body{{font-family:Georgia,serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#1d232b}}
table{{border-collapse:collapse;width:100%;font:13px Helvetica,Arial,sans-serif;margin:.5rem 0 1.5rem}}
th,td{{border-bottom:1px solid #dde2e8;padding:6px 8px;text-align:left;vertical-align:top}}th{{background:#f3f5f8}}
h1{{margin-bottom:.2rem}} .sub{{color:#5b6573;font:14px Helvetica,Arial,sans-serif}}</style></head><body>
<h1>{html.escape(a.get('display_name',''))}</h1>
<p class="sub">{html.escape(institutions_of(a))} · OpenAlex {short(a.get('id'))} · ORCID {html.escape(str(a.get('orcid') or 'none'))}<br>
Generated {time.strftime('%Y-%m-%d')} from OpenAlex. Screening indicators only, not a finding of misconduct.</p>
<h2>Indicators</h2><table><tr><th>Level</th><th>Indicator</th><th>Value</th><th>What it means</th></tr>{flag_rows}</table>
<h2>Citation network</h2>{network_svg(rep)}
<h2>Citation partners</h2>{tbl(rep.partners.head(40) if rep.partners is not None else None)}
<h2>Self-citation by paper</h2>{self_citation_svg(rep, 25)}
<h3>Self-citation links</h3>{tbl(rep.self_links.drop(columns=["cited_work_id", "citing_work_id"]).head(200) if rep.self_links is not None and not rep.self_links.empty else None)}
<h2>Citing journals</h2>{tbl(rep.sources)}
<h2>Retracted works</h2>{tbl(rep.retracted)}
<h2>Scope notes</h2><ul>{notes or '<li>Full record analysed.</li>'}<li>{rep.calls} API requests.</li></ul>
</body></html>"""


# ==========================================================================
# Streamlit UI
# ==========================================================================
def run_app():
    import streamlit as st

    st.set_page_config(page_title="Citation Integrity Screener", page_icon="🔎", layout="wide")
    st.markdown("""<style>
      .flag{border-left:5px solid var(--c);padding:.55rem .8rem;margin:.35rem 0;border-radius:4px;background:rgba(127,127,127,.06)}
      .flag b{color:var(--c)} .flag small{opacity:.75}
    </style>""", unsafe_allow_html=True)

    st.title("Citation Integrity Screener")
    st.caption("Screens a researcher's OpenAlex record for self-citation, citation concentration "
               "and reciprocal citation patterns. Indicators are prompts for closer reading, not verdicts.")

    with st.sidebar:
        st.header("OpenAlex access")
        key = st.text_input("API key", value=os.environ.get("OPENALEX_API_KEY", ""), type="password",
                            help="Free key at openalex.org/settings/api. Required since Feb 2026.")
        mailto = st.text_input("Contact email (optional)", value=os.environ.get("OPENALEX_MAILTO", ""))
        st.header("Scope")
        max_works = st.slider("Max works to analyse (most-cited first)", 50, 1000, 300, 50)
        max_citing = st.slider("Max citing works", 500, 10000, 3000, 500)
        max_refs = st.slider("Max referenced works to resolve", 500, 10000, 3000, 500)
        budget = st.slider("Request budget per run", 50, 2000, 500, 50,
                           help="Each list request costs about 10 OpenAlex credits.")
        st.caption("Bigger caps give a fuller picture but use more of your daily allowance.")

    if not key:
        st.info("Add your OpenAlex API key in the sidebar to begin.")

    q = st.text_input("Researcher name, ORCID, or OpenAlex author ID",
                      placeholder="e.g. 0000-0002-1825-0097, A5023888391, or a full name")
    if st.button("Find researcher", type="primary", disabled=not (q and key)):
        try:
            oa = OpenAlex(key, mailto, budget)
            st.session_state.candidates = lookup(oa, q)
            st.session_state.pop("report", None)
        except OAError as e:
            st.error(str(e))

    cands = st.session_state.get("candidates") or []
    if "candidates" in st.session_state and not cands:
        st.warning("No matching researcher. Try the ORCID, a different spelling, or the OpenAlex ID.")
    if cands:
        labels = [f"{c.get('display_name')} — {institutions_of(c)} — {c.get('works_count', 0)} works, "
                  f"{c.get('cited_by_count', 0)} citations — {short(c.get('id'))}"
                  + (f" — ORCID {c['orcid'].rsplit('/',1)[-1]}" if c.get('orcid') else "")
                  for c in cands]
        idx = st.radio("Pick the right profile. OpenAlex sometimes splits one person into several profiles.",
                       range(len(cands)), format_func=lambda i: labels[i])
        if st.button("Run integrity screen", type="primary"):
            oa = OpenAlex(key, mailto, budget)
            bar = st.progress(0.0, text="Starting")
            try:
                author = oa.get(f"/authors/{short(cands[idx]['id'])}")
                st.session_state.report = analyze(oa, author, max_works, max_citing, max_refs,
                                                  progress=lambda m, f: bar.progress(min(1.0, f), text=m))
            except OAError as e:
                st.error(str(e))
            bar.empty()

    rep: Report | None = st.session_state.get("report")
    if not rep:
        return

    a, m = rep.author, rep.metrics
    st.divider()
    st.subheader(a.get("display_name"))
    stats = a.get("summary_stats") or {}
    st.caption(f"{institutions_of(a)} · OpenAlex {short(a['id'])}"
               + (f" · ORCID {a['orcid'].rsplit('/',1)[-1]}" if a.get('orcid') else ""))
    c = st.columns(5)
    c[0].metric("Works", a.get("works_count"))
    c[1].metric("Citations", a.get("cited_by_count"))
    c[2].metric("h-index", stats.get("h_index"))
    c[3].metric("Citing works analysed", m["citing_works"])
    c[4].metric("API requests used", rep.calls)
    for n in rep.notes:
        st.caption("ℹ️ " + n)

    colors = {"green": "#1e7b4f", "amber": "#c77700", "red": "#b3261e", "grey": "#6b7280"}
    order = {"red": 0, "amber": 1, "grey": 2, "green": 3}
    st.markdown("#### Indicators")
    cols = st.columns(2)
    for i, f in enumerate(sorted(rep.flags, key=lambda f: order[f["level"]])):
        cols[i % 2].markdown(
            f"<div class='flag' style='--c:{colors[f['level']]}'><b>{f['level'].upper()}</b> · "
            f"{html.escape(f['title'])}: <b>{html.escape(str(f['value']))}</b><br><small>{html.escape(f['why'])}</small></div>",
            unsafe_allow_html=True)

    t1, t2, ts, t3, t4, t5, t6 = st.tabs(["Citation partners", "Network", "Self-citation", "Citing journals",
                                          "Co-authors", "Output & retractions", "Method"])
    with t1:
        st.caption(f"'Reciprocal' = min(citations each way). Rows with ≥{MUTUAL_MIN} each way and no shared papers "
                   "are the pairs most worth reading closely.")
        st.dataframe(rep.partners, use_container_width=True, hide_index=True)
    with t2:
        import streamlit.components.v1 as components
        n1, n2, n3 = st.columns([2, 2, 1])
        top_n = n1.slider("Partners shown", 4, 24, 12, key="net_n",
                          help="Fewer partners = a sparser, easier-to-read graph.")
        rank = n2.selectbox("Rank partners by", RANK_OPTIONS, key="net_rank")
        hide_co = n3.checkbox("Hide co-authors", key="net_hideco")
        svg = network_svg(rep, top_n, rank, hide_co)
        components.html(network_html(svg), height=820 + max(0, top_n - 14) * 26, scrolling=False)
        st.caption("Each partner shows exact counts: **in** = citations from that author's papers to the researcher, "
                   "**out** = the researcher's citations to that author. Hover a partner for details.")
        slug_n = re.sub(r"\W+", "_", rep.author.get("display_name", "researcher")).strip("_")
        st.download_button("Download graph (SVG)", svg, f"citation_network_{slug_n}.svg", "image/svg+xml")
    with ts:
        import streamlit.components.v1 as components
        sp = rep.self_papers if rep.self_papers is not None else pd.DataFrame()
        sl = rep.self_links if rep.self_links is not None else pd.DataFrame()
        s1, s2, s3 = st.columns([2, 2, 1])
        n_self = s1.slider("Papers shown", 5, 40, 15, key="self_n")
        sort_self = s2.selectbox("Sort papers by", SELF_SORT_OPTIONS, key="self_sort")
        all_papers = s3.checkbox("Include papers with no self-citations", key="self_all")
        svg_s = self_citation_svg(rep, n_self, sort_self, not all_papers)
        shown = len(select_self_papers(sp, n_self, sort_self, not all_papers))
        components.html(network_html(svg_s), height=int((128 + max(1, shown) * 54 + 92) * 0.93) + 30,
                        scrolling=True)
        st.caption("Each bar splits one paper's citations in the analysed sample into **self-citations** "
                   "(the citing paper includes this researcher) and **external** citations. "
                   "Hover a row to see which of the researcher's own papers cite it.")

        if rep.self_by_year is not None and not rep.self_by_year.empty:
            st.markdown("**Self vs external citations by year they were made**")
            st.bar_chart(rep.self_by_year.set_index("year")[["self", "external"]])

        st.markdown("**Every self-citation, paper by paper**")
        if sl.empty:
            st.info("No self-citations found in the analysed sample.")
        else:
            opts = ["All papers"] + list(dict.fromkeys(
                f"{t} ({y})" for t, y in zip(sl.cited_paper, sl.cited_year)))
            pick = st.selectbox("Show self-citations of", opts, key="self_pick")
            view = sl if pick == "All papers" else sl[
                [f"{t} ({y})" == pick for t, y in zip(sl.cited_paper, sl.cited_year)]]
            st.dataframe(
                view[["cited_paper", "cited_year", "citing_paper", "citing_year", "years_later",
                      "citing_venue", "citing_authors", "citing_doi"]].rename(columns={
                    "cited_paper": "Cited paper (researcher's)", "cited_year": "Cited year",
                    "citing_paper": "Citing paper (also researcher's)", "citing_year": "Citing year",
                    "years_later": "Years later", "citing_venue": "Citing venue",
                    "citing_authors": "Authors on citing paper", "citing_doi": "Citing DOI"}),
                use_container_width=True, hide_index=True)
        slug_s = re.sub(r"\W+", "_", rep.author.get("display_name", "researcher")).strip("_")
        d1, d2, d3 = st.columns(3)
        d1.download_button("Download graph (SVG)", svg_s, f"self_citation_{slug_s}.svg", "image/svg+xml")
        if not sp.empty:
            d2.download_button("Papers CSV", sp.to_csv(index=False), f"self_citation_papers_{slug_s}.csv", "text/csv")
        if not sl.empty:
            d3.download_button("Self-citation links CSV", sl.to_csv(index=False),
                               f"self_citation_links_{slug_s}.csv", "text/csv")
    with t3:
        st.dataframe(rep.sources, use_container_width=True, hide_index=True)
    with t4:
        st.dataframe(rep.coauthors, use_container_width=True, hide_index=True)
    with t5:
        if rep.yearly is not None and not rep.yearly.empty:
            y = rep.yearly.set_index("year")
            st.markdown("Works per year")
            st.bar_chart(y["works_count"])
            st.markdown("Citations received per year")
            st.line_chart(y["cited_by_count"])
        st.markdown("Retracted works")
        st.dataframe(rep.retracted, use_container_width=True, hide_index=True)
    with t6:
        st.markdown(f"""
* **Incoming self-citation**: citation links from citing papers that include this author, ÷ all citation links.
* **Co-author citation**: citation links from papers (without the author) that include any past co-author, ÷ external links.
* **Concentration**: share of external citation links from the single largest citing author / journal (only judged with ≥{MIN_CITES_FOR_SHARE} external citations).
* **Reciprocity**: for each outside author B, citations B→researcher and researcher→B. A pair counts when both directions reach {MUTUAL_MIN}.
* **Outgoing self-citation**: share of the researcher's reference links that point to works they co-wrote.
* **Thresholds** (amber / red) are editable at the top of the script: `{THRESHOLDS}`.

**Limitations.** OpenAlex merges or splits author profiles imperfectly; check the co-author list for strangers.
Large consortia papers inflate co-author and reciprocal counts. Self-citation norms vary by field
(often higher in mathematics, engineering and niche areas). Caps mean large records are sampled.
Use this to decide what to read, then read the papers.
""")

    st.markdown("#### Export")
    e = st.columns(3)
    slug = re.sub(r"\W+", "_", a.get("display_name", "researcher")).strip("_")
    e[0].download_button("HTML report", html_report(rep), f"citation_screen_{slug}.html", "text/html")
    if rep.partners is not None and not rep.partners.empty:
        e[1].download_button("Partners CSV", rep.partners.to_csv(index=False), f"partners_{slug}.csv", "text/csv")
    if rep.sources is not None and not rep.sources.empty:
        e[2].download_button("Journals CSV", rep.sources.to_csv(index=False), f"journals_{slug}.csv", "text/csv")


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _in_streamlit():
    run_app()
elif __name__ == "__main__":
    # Headless mode: python openalex_integrity.py "<name | ORCID | OpenAlex ID>"
    import argparse
    ap = argparse.ArgumentParser(description="Screen a researcher's citations in OpenAlex.")
    ap.add_argument("who", help="ORCID, OpenAlex author ID (A...), or a name (first match is used)")
    ap.add_argument("--key", default=os.environ.get("OPENALEX_API_KEY"))
    ap.add_argument("--mailto", default=os.environ.get("OPENALEX_MAILTO"))
    ap.add_argument("--max-works", type=int, default=300)
    ap.add_argument("--max-citing", type=int, default=3000)
    ap.add_argument("--max-refs", type=int, default=3000)
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--out", default=None, help="Write an HTML report to this path")
    args = ap.parse_args()
    oa = OpenAlex(args.key, args.mailto, args.budget)
    found = lookup(oa, args.who)
    if not found:
        raise SystemExit("No matching researcher.")
    author = oa.get(f"/authors/{short(found[0]['id'])}")
    rep = analyze(oa, author, args.max_works, args.max_citing, args.max_refs,
                  progress=lambda msg, f: print(f"[{int(f*100):3d}%] {msg}"))
    print(f"\n{author['display_name']} — {institutions_of(author)}")
    for f in rep.flags:
        print(f"  {f['level'].upper():6} {f['title']}: {f['value']}")
    out = args.out or f"citation_screen_{short(author['id'])}.html"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html_report(rep))
    print(f"\nReport written to {out}  ({rep.calls} API requests)")
