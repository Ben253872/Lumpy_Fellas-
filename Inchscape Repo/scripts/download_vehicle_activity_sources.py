"""Download official ANAC monthly market reports used by the vehicle signal builder.

MOP workbooks are indexed separately because the official page publishes legacy XLS
files.  This downloader deliberately keeps raw source files unchanged.
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT if (ROOT / "data" / "external").exists() else ROOT / "lumpy_fellas_reapply_backup_2026-07-14"
SOURCE_DIR = BASE / "data" / "external" / "External source files"
ANAC_DIR = SOURCE_DIR / "ANAC monthly market reports"
USER_AGENT = "Mozilla/5.0 vehicle-forecast-research/1.0"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def discover_anac_reports(year: int) -> list[tuple[str, str]]:
    url = f"https://www.anac.cl/category/estudio-de-mercado/?anno={year}"
    soup = BeautifulSoup(fetch(url), "html.parser")
    reports: list[tuple[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        title = anchor.get_text(" ", strip=True)
        href = anchor["href"].strip()
        if "Informe del Mercado Automotor" not in title:
            continue
        if not href.lower().endswith(".pdf"):
            continue
        match = re.search(r"(Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre)[- ](20\d{2})", title, re.I)
        if match and int(match.group(2)) == year:
            reports.append((title, href))
    return sorted(set(reports))


def main() -> None:
    ANAC_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    discovered = 0
    for year in range(2021, 2027):
        reports = discover_anac_reports(year)
        discovered += len(reports)
        for title, url in reports:
            filename = f"{year}_{Path(url).name}"
            target = ANAC_DIR / filename
            if not target.exists() or target.stat().st_size == 0:
                target.write_bytes(fetch(url))
                downloaded += 1
            print(f"{title}: {target.name}")
    print(f"Discovered {discovered} reports; downloaded {downloaded} new files")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        raise
