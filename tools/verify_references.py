"""Checks every entry of references/references.json against Crossref and OpenAlex.

Entries with a DOI are resolved through Crossref and their title is compared with the title in the entry.
Entries without a DOI are searched by title in OpenAlex. The script only reports: It never fails the build,
because a missing match can also mean that a book or report is not indexed.

Usage: python tools/verify_references.py --report references/VERIFICATION_REPORT.md
"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "course-reference-check/1.0 (mailto:utkukose@sdu.edu.tr)"}


def get_json(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 * (i + 1))
    return None


def words(s):
    return set(w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(w) > 2)


def title_of(text):
    m = re.match(r".+?\(\d{4}\)\.\s(.+?)\.\s", text)
    return m.group(1) if m else text[:120]


def similarity(a, b):
    A, B = words(a), words(b)
    return len(A & B) / max(1, min(len(A), len(B)))


def check(key, entry):
    title = title_of(entry["text"])
    link = entry.get("link", "")
    if "doi.org/" in link:
        doi = link.split("doi.org/", 1)[1]
        js = get_json("https://api.crossref.org/works/" + urllib.parse.quote(doi))
        if not js:
            return "doi unreachable", doi
        found = " ".join(js["message"].get("title", [""]))
        s = similarity(title, found)
        return ("match" if s >= 0.6 else "title differs"), f"{doi} ({s:.2f})"
    js = get_json("https://api.openalex.org/works?per-page=1&search=" + urllib.parse.quote(title))
    if not js or not js.get("results"):
        return "not indexed", "-"
    found = js["results"][0].get("title") or ""
    s = similarity(title, found)
    return ("match" if s >= 0.6 else "uncertain"), f"OpenAlex ({s:.2f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="references/references.json")
    ap.add_argument("--report", default="references/VERIFICATION_REPORT.md")
    a = ap.parse_args()
    refs = json.load(open(a.refs, encoding="utf-8"))
    rows, counts = [], {}
    for k, e in refs.items():
        status, detail = check(k, e)
        counts[status] = counts.get(status, 0) + 1
        rows.append(f"| `{k}` | {status} | {detail} |")
        time.sleep(0.3)
    with open(a.report, "w", encoding="utf-8") as f:
        f.write("# Reference verification report\n\n" +
                ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) +
                "\n\n| Key | Status | Detail |\n|---|---|---|\n" + "\n".join(rows) + "\n")
    print(", ".join(f"{v} {k}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
