"""Master-level over-the-board Chess960 games (human play case studies).

Games from elite Chess960 / Freestyle events, as PGN files from The Week in
Chess (TWIC). The files are listed in config/masters960_sources.csv (add rows
for events found later; `verified` marks URLs checked by hand). These events
usually draw one position per round for every board, so the data are many
games on a few dozen positions: case studies rather than a full ranking.

Subcommands
-----------
download  Fetch every listed file into data/masters960/pgn/ (needs internet;
          skips files already there). Files can also be saved there by hand.
parse     Read every .pgn in data/masters960/pgn/ and write
          data/masters960/masters960_games.csv: one row per game with the
          event (from the sources table), players, FIDE ratings, result,
          starting position (0-959 from the FEN), plies, clocks and the
          time-control class. Armageddon games (unequal clocks or named in the
          round/event) are flagged.

The analysis is in analyse_masters960.py.

    python code/masters960.py download
    python code/masters960.py parse
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special, stats

from lichess960 import iter_games, position_from_fen
from utils import DATA_DIR, REPO_ROOT, STANDARD_POSITION_ID, setup_logging

log = logging.getLogger("masters960")

SOURCES = REPO_ROOT / "config" / "masters960_sources.csv"
DATA = DATA_DIR / "masters960" / "pgn"          # downloaded PGN files (not committed)
OUT = DATA_DIR / "masters960"                    # masters960_games.csv

CLK_RE = re.compile(r"\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]")
COMMENT_RE = re.compile(r"\{[^}]*\}")
VARIATION_RE = re.compile(r"\([^()]*\)")
MOVE_NO_RE = re.compile(r"\d+\.(\.\.)?")
RESULTS = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
ARMAGEDDON_RATIO = 0.25          # first clocks differing by more than this = unequal time
ARMAGEDDON_RATIO_BLITZ = 0.15    # ... in blitz-length games (Armageddon is e.g. 5 vs 4 min)
OWN_CLOCK_BELOW = 0.6            # first clock below this share of the event base: faster tiebreak


# ---------------------------------------------------------------------------
# Download and parse
# ---------------------------------------------------------------------------

def load_sources(path: Path = SOURCES) -> pd.DataFrame:
    return pd.read_csv(path)


def download(sources: pd.DataFrame, out_dir: Path = DATA) -> None:
    """Fetch every file in the sources table. The URL comes from the table, so
    adding an event only needs a new row there."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(sources)
    print(f"Downloading {n} PGN files into {out_dir}")
    done, skipped, failed = [], [], []
    for k, r in enumerate(sources.itertuples(), 1):
        dest = out_dir / r.file
        tag = f"[{k:>2}/{n}] {r.file:<24}"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"{tag} already here ({dest.stat().st_size / 1024:,.0f} KB, "
                  f"{count_games(dest)} games), skipped")
            skipped.append(r.file)
            continue
        try:
            if r.url.startswith("https://lichess.org/broadcast/"):
                body = lichess_broadcast_pgn(r.url)
            else:
                body = fetch(r.url)
            if r.url.lower().endswith(".zip"):      # a TWIC weekly issue: keep its PGN
                body = pgn_from_zip(body)
            if b"[Event" not in body[:5000]:
                raise ValueError("response is not a PGN file (wrong URL or an error page?)")
            dest.write_bytes(body)
            print(f"{tag} downloaded {len(body) / 1024:,.0f} KB, {count_games(dest)} games "
                  f"({r.event} {r.year}, {r.stage})")
            done.append(r.file)
        except Exception as exc:          # noqa: BLE001 - report and carry on
            print(f"{tag} FAILED: {exc}")
            failed.append(r)
    print(f"\nDone: {len(done)} downloaded, {len(skipped)} already present, {len(failed)} failed.")
    if failed:
        print("Save these by hand into", out_dir, "(same file names), then run `parse`:")
        for r in failed:
            print(f"  {r.file}: {r.url}")


def fetch(url: str, accept: str | None = None) -> bytes:
    headers = {"User-Agent": "chess960-fairness research"}
    if accept:
        headers["Accept"] = accept
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as resp:
        return resp.read()


def lichess_broadcast_pgn(url: str) -> bytes:
    """All games of a Lichess broadcast, from the URL of any of its rounds
    (https://lichess.org/broadcast/<tour-slug>/<round-slug>/<round-id>[/<game-id>]).
    The round's tournament is exported with /api/broadcast/<tour-id>.pgn. With
    '#group' at the end of the URL, every tournament in the round's broadcast
    group is exported (e.g. both round-robin groups and the knockout)."""
    import json
    group = url.endswith("#group")
    parts = url.split("#")[0].rstrip("/").split("/broadcast/")[1].split("/")
    if len(parts) < 3:
        raise ValueError("expected a round URL: /broadcast/<tour-slug>/<round-slug>/<round-id>")
    tslug, rslug, round_id = parts[:3]
    info = json.loads(fetch(f"https://lichess.org/api/broadcast/{tslug}/{rslug}/{round_id}",
                            accept="application/json"))
    tours = [(info["tour"]["id"], info["tour"].get("name", ""))]
    if group:
        g = info.get("group") or {}
        if g.get("tours"):
            tours = [(t["id"], t.get("name", "")) for t in g["tours"]]
        else:
            print("    (no broadcast group found; exporting this tournament only)")
    chunks = []
    for tid, name in tours:
        pgn = fetch(f"https://lichess.org/api/broadcast/{tid}.pgn")
        print(f"    Lichess broadcast {tid} {name}: {pgn.count(b'[Event ')} games")
        chunks.append(pgn)
    return b"\n\n".join(chunks)


def pgn_from_zip(body: bytes) -> bytes:
    import io
    import zipfile
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".pgn")]
        if not names:
            raise ValueError("zip has no .pgn file")
        return b"\n".join(z.read(n) for n in names)


def count_games(path: Path) -> int:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return sum(1 for line in fh if line.startswith("[Event "))


def tc_class(base_s: float) -> str:
    if base_s >= 3600:
        return "classical"
    if base_s >= 480:
        return "rapid"
    return "blitz"


def clocks(movetext: str) -> list[float]:
    return [int(h) * 3600 + int(m) * 60 + float(s) for h, m, s in CLK_RE.findall(movetext)]


def count_plies(movetext: str) -> int:
    t = COMMENT_RE.sub(" ", movetext)
    while VARIATION_RE.search(t):
        t = VARIATION_RE.sub(" ", t)
    t = MOVE_NO_RE.sub(" ", t)
    toks = [x for x in t.split() if x not in RESULTS and x != "*" and not x.startswith("$")]
    return len(toks)


VARIANT_960_RE = re.compile(r"960|fischer\s*-?\s*random|fischerandom|freestyle|9lx", re.I)


def is_960_variant(variant: str) -> bool:
    """Variant header naming Chess960 (spellings seen: Chess960, chess 960,
    Fischerandom, Fischer Random, Freestyle, 9LX)."""
    return bool(VARIANT_960_RE.search(variant or ""))


def game_record(h: dict[str, str], movetext: str, src: dict) -> dict:
    fen = h.get("FEN", "")
    clk = clocks(movetext)
    w0 = clk[0] if len(clk) > 0 else np.nan
    b0 = clk[1] if len(clk) > 1 else np.nan
    source_base = float(src.get("base_minutes", np.nan)) * 60
    first = np.nanmax([w0, b0]) if np.isfinite([w0, b0]).any() else np.nan
    # A game in a file whose first clocks are far below the event's base time
    # is a tiebreak at a faster control: classify it by its own clocks.
    # A game whose first clocks are far from the event's base time (a faster
    # tiebreak, or a classical game in a mixed file) is classed by its own clocks.
    base = (first if np.isfinite(first) and (first < OWN_CLOCK_BELOW * source_base
                                             or first > 1.5 * source_base)
            else source_base)
    text = f"{h.get('Event', '')} {h.get('Round', '')}".lower()
    limit = ARMAGEDDON_RATIO_BLITZ if np.isfinite(base) and base < 480 else ARMAGEDDON_RATIO
    unequal = (np.isfinite(w0) and np.isfinite(b0) and max(w0, b0) > 0
               and abs(w0 - b0) / max(w0, b0) > limit)
    def elo(k):
        try:
            v = float(h.get(k, ""))
            return v if v > 0 else np.nan
        except ValueError:
            return np.nan
    return {
        "source": src.get("file", ""), "event": src.get("event", h.get("Event", "")),
        "year": src.get("year", np.nan), "stage": src.get("stage", ""),
        "pgn_event": h.get("Event", ""), "site": h.get("Site", ""), "date": h.get("Date", ""),
        "round": h.get("Round", ""), "white": h.get("White", ""), "black": h.get("Black", ""),
        "white_elo": elo("WhiteElo"), "black_elo": elo("BlackElo"),
        "white_title": h.get("WhiteTitle", ""), "black_title": h.get("BlackTitle", ""),
        "result": h.get("Result", ""), "score_white": RESULTS.get(h.get("Result", ""), np.nan),
        "variant": h.get("Variant", ""), "variant_960": is_960_variant(h.get("Variant", "")),
        "fen": fen, "position_id": position_from_fen(fen) if fen else -1,
        "plies": count_plies(movetext), "first_clock_white": w0, "first_clock_black": b0,
        "base_s": base, "tc_class": tc_class(base) if np.isfinite(base) else "",
        "armageddon": bool("armageddon" in text or unequal),
    }


def parse(sources: pd.DataFrame, data_dir: Path = DATA, out: Path = OUT) -> pd.DataFrame:
    meta = {r["file"]: r for r in sources.to_dict("records")}
    files = sorted(data_dir.glob("*.pgn"))
    if not files:
        raise SystemExit(f"no PGN files in {data_dir}; run `download` first")
    print(f"Parsing {len(files)} PGN files from {data_dir}\n")
    print(f"{'file':<24} {'games':>6} {'variant':>7} {'960 pos':>7} {'positions':>9} {'classical':>9} "
          f"{'rapid':>6} {'blitz':>6} {'armag.':>6} {'no Elo':>6}  event")
    rows = []
    for f in files:
        src = meta.get(f.name)
        if src is None:
            print(f"{f.name:<24} not in {SOURCES.name}: event details unknown, add a row for it")
            src = {"file": f.name}
        with open(f, encoding="utf-8", errors="replace") as fh:
            flt = src.get("event_filter")
            flt = re.compile(flt, re.I) if isinstance(flt, str) and flt else None
            recs = [game_record(h, mt, src) for h, mt in iter_games(fh)
                    if flt is None or flt.search(h.get("Event", ""))]
        rows += recs
        r = pd.DataFrame(recs)
        if r.empty:
            print(f"{f.name:<24} {0:>6}  (no games found)")
            continue
        tc = r.tc_class.value_counts()
        print(f"{f.name:<24} {len(r):>6} {int(r.variant_960.sum()):>7} {(r.position_id >= 0).sum():>7} "
              f"{r.position_id[r.position_id >= 0].nunique():>9} {tc.get('classical', 0):>9} "
              f"{tc.get('rapid', 0):>6} {tc.get('blitz', 0):>6} {int(r.armageddon.sum()):>6} "
              f"{int((r.white_elo.isna() | r.black_elo.isna()).sum()):>6}  "
              f"{src.get('event', '?')} {src.get('year', '')}")
    g = pd.DataFrame(rows)
    before = len(g)
    g = g.drop_duplicates(subset=["white", "black", "date", "round", "result", "fen", "plies"])
    unknown = g[g.position_id < 0]
    no_tag = g[(g.position_id >= 0) & ~g.variant_960]
    tag_no_pos = g[g.variant_960 & (g.position_id < 0)]
    print(f"Variant tag says Chess960: {int(g.variant_960.sum()):,}; Chess960 FEN without the tag: "
          f"{len(no_tag)}; tag without a usable FEN: {len(tag_no_pos)}")
    if len(no_tag[no_tag.position_id == STANDARD_POSITION_ID]):
        print(f"  standard start (#518) without a 960 tag: "
              f"{int((no_tag.position_id == STANDARD_POSITION_ID).sum())} games (excluded as standard chess)")
    print(f"\nTotal: {before:,} games; {before - len(g)} duplicates removed; {len(g):,} kept")
    print(f"Starting position identified: {(g.position_id >= 0).sum():,}; distinct positions: "
          f"{g.position_id[g.position_id >= 0].nunique()}")
    if len(unknown):
        print(f"No Chess960 position for {len(unknown)} games (missing or unusual FEN), e.g. "
              f"{unknown.source.value_counts().head(3).to_dict()}")
    top = g[g.position_id >= 0].position_id.value_counts().head(10)
    print("Most-played positions: " + ", ".join(f"#{p} ({n})" for p, n in top.items()))
    out.mkdir(parents=True, exist_ok=True)
    g.to_csv(out / "masters960_games.csv", index=False)
    print(f"Wrote {out / 'masters960_games.csv'}")
    return g


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    for name, h in (("download", "fetch the listed PGN files into data/masters960/pgn/"),
                    ("parse", "PGN files to the games table")):
        sub.add_parser(name, parents=[common], help=h)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    src = load_sources()
    if args.command == "download":
        download(src)
    else:
        parse(src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
