"""Build the Decision Index dev set: raw held-out records + their kit-adapter conversions.

For every benchmark in scope, 100 samples (or all available, when fewer exist) are drawn from data
the Decision Index 0.2 suite does not score, and stored twice:

  evals/decision-index-dev/raw/<NN>-<slug>/        the source records, unchanged, in the source's
                                                   own file format, + provenance.jsonl per sample
  evals/decision-index-dev/converted/<NN>-<slug>.jsonl
                                                   the same samples as Decision Index requests,
                                                   produced by the kit's own adapters (unmodified)

Source kinds (recorded per sample):
  heldout-split        a train/dev/validation split the suite does not use
  heldout-category     a published category the suite's adapter does not read (BFCL)
  heldout-window       the same files, outside the suite's date window (ForecastBench)
  suite-unselected     a row of the same test pool that Decision Index 0.2 does not score
  generator-dev        the kit's own generator's dev rows (home appliance simulator)
  handwritten          written for this dev set (50 legitimate PhishNChips emails)

How conversion works: held-out records are written, in their original format, into the input slot
the adapter reads, inside a separate dev workspace (other inputs are linked from the real build);
then the adapter runs unchanged. Suite-unselected rows are taken from the adapters' output for the
real build. `_evaluation` is attached with the same formula as the kit's freeze step, with run ids
prefixed `dev:` so they can never be mistaken for suite rows.

Run with the kit's Python (needs pyarrow, pandas, huggingface_hub, tiktoken from the kit's extras):
    K=~/Documents/dev/open-system-one/decision-index
    $K/.venv/bin/python scripts/decision_index_dev.py [--only 24 59 ...]
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KIT = Path(os.environ.get("DI_KIT", "~/Documents/dev/open-system-one/decision-index")).expanduser()
WORK = KIT / "work"  # the real suite build (sources + adapter outputs)
SUITE = KIT / "suite-0.2"  # the scored 0.2 suite
DEVWORK = KIT / "work-dev"  # adapter workspace for held-out conversions (rebuildable)
CACHE = KIT / "work-dev-downloads"  # held-out files fetched from pinned sources
OUT = REPO / "evals/decision-index-dev"
SEED = 20261002
N = 100

sys.path.insert(0, str(KIT))
from decision_index.suite.build import adapters_added as AA
from decision_index.suite.build import freeze as FZ
from decision_index.suite.build.layout import DATASETS, NAMES, Layout
from decision_index.suite.build.rebuild import BUILDERS
from decision_index.suite.io import dumps, read_jsonl

REAL = Layout(WORK)

# Benchmarks whose terms forbid redistribution or ask that items not be posted publicly: their raw
# and converted files are written but gitignored (see evals/decision-index-dev/.gitignore).
RESTRICTED = {
    59: "RAGTruth contexts include Yelp Open Dataset text (no redistribution) and MS MARCO (non-commercial)",
    25: "GPQA authors ask that the questions not be posted in plain text",
}


def bench_name(n):
    return DATASETS.get(n) or AA.SPECS[n]


def slug(n):
    return f"{n:02d}-" + bench_name(n).split(" ")[0].replace("/", "-").replace("+", "plus")


def rng(n):
    return random.Random(f"{SEED}:{n}")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------- sampling helpers


def balanced(items, label, n, r, labels=(True, False)):
    """n items split evenly over two labels (fewer if a side runs short), deterministic order."""
    pools = {lab: [x for x in items if label(x) == lab] for lab in labels}
    for p in pools.values():
        r.shuffle(p)
    per = n // len(labels)
    chosen = [x for lab in labels for x in pools[lab][:per]]
    r.shuffle(chosen)
    return chosen


def proportional(groups, n):
    """Split n over groups in proportion to their sizes (largest remainder)."""
    total = sum(groups.values())
    raw = {k: n * v / total for k, v in groups.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: -(raw[k] - out[k]))[: n - sum(out.values())]:
        out[k] += 1
    return out


# ----------------------------------------------------------------------------- output helpers


class Bench:
    def __init__(self, n):
        self.n, self.slug = n, slug(n)
        self.raw_dir = OUT / "raw" / self.slug
        if self.raw_dir.exists():
            shutil.rmtree(self.raw_dir)
        self.raw_dir.mkdir(parents=True)
        self.samples = []  # provenance records

    def raw_path(self, name):
        p = self.raw_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def add(self, sample_key, kind, source, label=None, raw_file=None):
        self.samples.append(
            {
                "sample_id": f"{self.slug}:{len(self.samples):03d}",
                "key": sample_key,
                "benchmark": bench_name(self.n),
                "catalog_id": self.n,
                "source_kind": kind,
                "source": source,
                "label": label,
                "raw_file": raw_file,
            }
        )


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def write_parquet_subset(src, rows_idx, dest, transform=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    t = pq.read_table(src).take(pa.array(rows_idx, pa.int64()))
    if transform:
        t = transform(t)
    pq.write_table(t, dest)
    return t


def link(src, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    dest.symlink_to(Path(src).resolve())


def evaluation(n, track, row, rel_source):
    """The kit freeze step's `_evaluation`, with a `dev:` run id prefix."""
    payload = {"model": FZ.REFERENCE_MODEL, "state": row["state"], "questions": row["questions"]}
    digest = hashlib.sha256(dumps(payload).encode()).hexdigest()
    return dict(
        run_id=f"dev:{n}:{track}:{row['id']}",
        catalog_id=n,
        dataset=DATASETS[n],
        group_id=FZ.gid(row),
        track=track,
        source_path=rel_source,
        payload_sha256=digest,
        proxy_tokens=max(1, len(dumps(payload)) // 4),
        benchmark_origin=FZ.provenance(n),
    )


def finish(b, rows):
    """Write converted rows + provenance; link each sample to its converted run ids."""
    by_key = defaultdict(list)
    for r in rows:
        by_key[r["_dev"]["key"]].append(r["_evaluation"]["run_id"])
    kept = []
    for s in b.samples:
        s["converted_run_ids"] = by_key.get(s["key"], [])
        assert s["converted_run_ids"], f"{b.slug}: sample {s['key']} produced no converted row"
        kept.append(s)
    for r in rows:
        r["_dev"]["sample_id"] = next(
            s["sample_id"] for s in b.samples if s["key"] == r["_dev"]["key"]
        )
    write_jsonl(b.raw_dir / "provenance.jsonl", kept)
    conv = OUT / "converted" / f"{b.slug}.jsonl.gz"
    conv.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(conv, "wt", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    labels = Counter(str(s["label"]) for s in kept if s["label"] is not None)
    kinds = Counter(s["source_kind"] for s in kept)
    return {
        "catalog_id": b.n,
        "benchmark": bench_name(b.n),
        "slug": b.slug,
        "samples": len(kept),
        "converted_requests": len(rows),
        "source_kinds": dict(kinds),
        "labels": dict(labels),
        "restricted": RESTRICTED.get(b.n),
    }


# ----------------------------------------------------------------------------- suite-unselected


def suite_run_ids(n):
    ids = set()
    for fn in ("selected-rows.jsonl.gz", "added-rows.jsonl.gz"):
        for r in read_jsonl(SUITE / fn):
            if r["_evaluation"]["catalog_id"] == n:
                ids.add(r["_evaluation"]["run_id"])
    return ids


_SUITE_HASHES = None


def suite_payload_hashes():
    global _SUITE_HASHES
    if _SUITE_HASHES is None:
        _SUITE_HASHES = {
            r["_evaluation"]["payload_sha256"]
            for fn in ("selected-rows.jsonl.gz", "added-rows.jsonl.gz")
            for r in read_jsonl(SUITE / fn)
        }
    return _SUITE_HASHES


def payload_hash(row):
    payload = {"model": FZ.REFERENCE_MODEL, "state": row["state"], "questions": row["questions"]}
    return hashlib.sha256(dumps(payload).encode()).hexdigest()


def unselected(n, label=None, raw=None, balance_labels=(True, False), one_row=False):
    """Sample N groups of the real build's adapter output that the 0.2 suite does not score.

    one_row: a sample is one row from a distinct group (BPoMP: one original/perturbed pair per
    poem) instead of the whole group."""
    b = Bench(n)
    scored = suite_run_ids(n)
    groups = defaultdict(list)
    for name in NAMES[n]:
        p = REAL.normalized / f"{name}.jsonl"
        for r in read_jsonl(p):
            if f"{n}:{name}:{r['id']}" not in scored:
                groups[FZ.gid(r)].append((name, r))
    # Source datasets contain duplicates under different ids (ToolRet, CLadder): drop any group
    # whose request is identical to a scored suite request.
    suite = suite_payload_hashes()
    keys = sorted(g for g, rs in groups.items() if not any(payload_hash(r) in suite for _, r in rs))
    r_ = rng(n)
    if one_row:
        for g in keys:
            groups[g] = [r_.choice(groups[g])]
    if label:
        chosen = balanced(keys, lambda g: label(groups[g][0][1]), N, r_, balance_labels)
    else:
        r_.shuffle(keys)
        chosen = keys[:N]
    rows = []
    raw_records = []
    for g in chosen:
        first = groups[g][0][1]
        rec, source = raw(first) if raw else (None, {})
        if rec is not None:
            raw_records.append({"sample_key": g, **rec})
        b.add(
            g,
            "suite-unselected",
            {"note": "same test pool as the suite; not scored in Decision Index 0.2", **source},
            label=label(first) if label else None,
            raw_file="records.jsonl",
        )
        for name, row in groups[g]:
            row["_evaluation"] = evaluation(
                n, name, row, f"artifacts/benchmark-suite/normalized/{name}.jsonl"
            )
            row["_dev"] = {"key": g, "source_kind": "suite-unselected"}
            rows.append(row)
    write_jsonl(b.raw_path("records.jsonl"), raw_records)
    return finish(b, rows)


def u_bpomp():
    files = {}

    def raw(row):
        prov = row["metadata"]["provenance"]
        p = WORK / prov["source"]
        if p not in files:
            files[p] = json.loads(p.read_text())
        pair = files[p][row["metadata"]["variant"]][prov["source_index"]]
        return {
            "file": p.name,
            "variant": row["metadata"]["variant"],
            "index": prov["source_index"],
            "record": pair,
        }, {
            "dataset": "BPoMP (zenodo 7299879)",
            "file": p.name,
            "variant": row["metadata"]["variant"],
            "index": prov["source_index"],
        }

    return unselected(
        20,
        label=lambda r: r["expected"]["answer"],
        raw=raw,
        balance_labels=("option_0", "option_1"),
        one_row=True,
    )


def u_cladder():
    with zipfile.ZipFile(REAL.raw / "cladder/data/cladder-v1.zip") as z:
        qs = {q["question_id"]: q for q in json.loads(z.read("cladder-v1-q-balanced.json"))}

    def raw(row):
        qid = (
            row["provenance"]["source_question_id"]
            if "provenance" in row
            else row["metadata"]["source_question_id"]
        )
        return {"file": "cladder-v1.zip:cladder-v1-q-balanced.json", "record": qs[qid]}, {
            "dataset": "CLadder v1 (causalNLP/cladder @3d2d116)",
            "file": "cladder-v1-q-balanced.json",
            "question_id": qid,
        }

    def label(r):
        return r["expected"]["q1"]

    return unselected(44, label=label, raw=raw, balance_labels=("A", "B"))


def u_cfcolor():
    def raw(row):
        m = row["metadata"]
        return {
            "record": {
                k: m[k]
                for k in ("user_id", "rating_gap", "target_palette_ids", "history_palette_ids")
            },
            "file": "cfcolor.zip: allMTurkRatings.mat + themeData.mat",
        }, {
            "dataset": "cfcolor (dgp.toronto.edu)",
            "user_id": m["user_id"],
            "target_palette_ids": m["target_palette_ids"],
        }

    return unselected(
        23, label=lambda r: r["expected"]["preference"], raw=raw, balance_labels=("A", "B")
    )


def u_routerbench():
    import pandas as pd

    frames = {}

    def raw(row):
        m = row["metadata"]
        p = WORK / m["source_file"]
        if p not in frames:
            frames[p] = pd.read_pickle(p)
        rec = frames[p].iloc[m["source_index"]].to_dict()
        return {
            "file": p.name,
            "index": m["source_index"],
            "record": json.loads(json.dumps(rec, default=str)),
        }, {
            "dataset": "withmartian/routerbench @7840214",
            "file": p.name,
            "index": m["source_index"],
            "sample_id": m.get("sample_id"),
        }

    return unselected(6, raw=raw)


def u_pop909():
    def raw(row):
        m = row["metadata"]
        midi = REAL.repos / "pop909cl/POP909_processed" / f"{m['song_id']}.mid"
        dest = OUT / "raw" / slug(22) / "midi" / midi.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(midi, dest)
        return {
            "song_id": m["song_id"],
            "midi": f"midi/{midi.name}",
            "beat": row["id"].split(":")[-1],
        }, {"dataset": "POP909-CL @be90943", "song_id": m["song_id"], "case": row["id"]}

    return unselected(22, raw=raw)


def u_chess():
    import base64

    from decision_index.suite.build import adapters_scored as SC

    holder = {}

    def raw(row):
        m = row["metadata"]
        holder.setdefault("wanted", set()).add(m["group_id"])
        return {
            "position_hash": m["group_id"],
            "fen": row["state"]["fen"],
            "file": "chessbench-test-action-value.bag",
            "records": None,
        }, {
            "dataset": "ChessBench test action-value bag (searchless_chess @90ae0e6)",
            "file": "chessbench-test-action-value.bag",
            "position_hash": m["group_id"],
        }

    summary = unselected(31, raw=raw)
    # Second pass: fill each chosen position with its original bag records (decoded + raw bytes).
    path = OUT / "raw" / slug(31) / "records.jsonl"
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    want = {r["fen"]: r for r in recs}
    for record in SC.bag_records(REAL.downloads / "chessbench-test-action-value.bag"):
        fen, move, value = SC.decode_bag(record)
        if fen in want:
            want[fen].setdefault("bag_records", []).append(
                {
                    "fen": fen,
                    "move": move,
                    "win_prob": value,
                    "raw_base64": base64.b64encode(bytes(record)).decode(),
                }
            )
    for r in recs:
        r.pop("records", None)
    write_jsonl(path, recs)
    return summary


def u_retrieval(n):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    family = "ToolRet-retrieval" if n == 2 else "BRIGHT-retrieval"
    sidecar = {}
    for line in (WORK / "artifacts/benchmark-suite/retrieval-queries" / f"{family}.jsonl").open():
        rec = json.loads(line)
        sidecar[rec["id"]] = rec
    corpora_cache = {}

    def corpus_rows(row, ids):
        m = row["metadata"]
        if n == 2:
            paths = sorted((REAL.raw / "toolret_tools" / m["category"]).glob("*.parquet"))
        else:
            paths = [REAL.raw / "bright/documents" / Path(m["source"]).name]
        out = []
        for p in paths:
            t = corpora_cache.get(p) or corpora_cache.setdefault(p, pq.read_table(p))
            hit = t.filter(
                pc.is_in(
                    pc.cast(t["id"], "string"), value_set=__import__("pyarrow").array(sorted(ids))
                )
            )
            out += hit.to_pylist()
        return out

    def raw(row):
        m = row["metadata"]
        domain, qid = m["group_id"].split(":", 1)
        src = WORK / m["source"]
        t = pq.read_table(src)
        q = t.filter(pc.equal(pc.cast(t["id"], "string"), qid)).to_pylist()
        side = sidecar[m["group_id"]]
        docs = corpus_rows(row, set(side["retrieved_ids"]))
        rec = {
            "file": src.name,
            "query_record": json.loads(json.dumps(q[0], default=str)) if q else None,
            "retrieved_ids": side["retrieved_ids"],
            "qrels": side["qrels"],
            "candidate_documents": json.loads(json.dumps(docs, default=str)),
        }
        return rec, {
            "dataset": "mangopy/ToolRet-Queries @b8c76ad + ToolRet-Tools @e06c38c"
            if n == 2
            else "xlangai/BRIGHT @3066d29",
            "file": src.name,
            "query_group": m["group_id"],
        }

    return unselected(n, raw=raw)


def generator_home():
    b = Bench(9)
    rows = []
    dev = [
        json.loads(line)
        for line in (WORK / "artifacts/benchmark-suite/home/dev.jsonl").read_text().splitlines()
        if line.strip()
    ]
    write_jsonl(b.raw_path("dev.jsonl"), dev)
    for i, row in enumerate(dev):
        key = row["id"]
        b.add(
            key,
            "generator-dev",
            {
                "dataset": "kit home-appliance generator (purpose-built benchmark)",
                "file": "home/dev.jsonl",
                "index": i,
                "seed": json.loads(
                    (WORK / "artifacts/benchmark-suite/home/generation-manifest.json").read_text()
                )["seed"],
            },
            raw_file="dev.jsonl",
        )
        row = json.loads(json.dumps(row))
        row["_evaluation"] = evaluation(
            9, "Home-Appliance", row, "artifacts/benchmark-suite/home/dev.jsonl"
        )
        row["_dev"] = {"key": key, "source_kind": "generator-dev"}
        rows.append(row)
    return finish(b, rows)


UNSELECTED = {
    20: u_bpomp,
    44: u_cladder,
    23: u_cfcolor,
    6: u_routerbench,
    22: u_pop909,
    31: u_chess,
    2: lambda: u_retrieval(2),
    36: lambda: u_retrieval(36),
    9: generator_home,
}


# ----------------------------------------------------------------------------- held-out sources


def hf_file(repo, path, rev):
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    return Path(
        hf_hub_download(repo, path, revision=rev, repo_type="dataset", cache_dir=str(CACHE / "hf"))
    )


def hf_rev(repo):
    from huggingface_hub import HfApi

    return HfApi().dataset_info(repo).sha


def url_file(url, name):
    import urllib.request

    dest = CACHE / name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def dev_layout():
    if DEVWORK.exists():
        shutil.rmtree(DEVWORK)
    return Layout(DEVWORK)


def git_link(dl, repo):
    """Adapters record `git rev-parse HEAD`; point the dev repo dir at the real checkout's .git."""
    link(REAL.repos / repo / ".git", dl.repos / repo / ".git")


def collect(b, dl, n, name, key_of, kind):
    rows = []
    wanted = {s["key"] for s in b.samples}
    for r in read_jsonl(dl.normalized / f"{name}.jsonl"):
        k = key_of(r)
        if k in wanted:
            r["_evaluation"] = evaluation(
                n, name, r, f"work-dev/artifacts/benchmark-suite/normalized/{name}.jsonl"
            )
            r["_dev"] = {"key": k, "source_kind": kind}
            rows.append(r)
    return rows


def parquet_pick(path, idx, dest_raw, slot, transform_slot=None):
    """Write the chosen rows of a parquet file to the raw store (unchanged) and to the adapter slot."""
    t = write_parquet_subset(path, idx, dest_raw)
    slot.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq

    pq.write_table(transform_slot(t) if transform_slot else t, slot)
    return t


def h_knowledge(dl, only):
    """MMLU (validation), ARC-Easy/Challenge (validation), WinoGrande (train), HellaSwag (train)."""
    import pyarrow.parquet as pq

    arc_rev = hf_rev("allenai/ai2_arc")
    specs = {
        24: (
            "cais/mmlu",
            "c30699e8356da336a370243923dbaf21066bb9fe",
            "all/validation-00000-of-00001.parquet",
            "mmlu/all/test-00000-of-00001.parquet",
            "validation",
        ),
        26: (
            "allenai/ai2_arc",
            arc_rev,
            "ARC-Easy/validation-00000-of-00001.parquet",
            "arc/ARC-Easy/test-00000-of-00001.parquet",
            "validation",
        ),
        27: (
            "allenai/ai2_arc",
            arc_rev,
            "ARC-Challenge/validation-00000-of-00001.parquet",
            "arc/ARC-Challenge/test-00000-of-00001.parquet",
            "validation",
        ),
        28: (
            "allenai/winogrande",
            "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
            "winogrande_xl/train-00000-of-00001.parquet",
            "winogrande/winogrande_xl/validation-00000-of-00001.parquet",
            "train",
        ),
        29: (
            "Rowan/hellaswag",
            "218ec52e09a7e7462a5400043bb9a69a41d06b76",
            "data/train-00000-of-00001.parquet",
            "hellaswag/data/validation-00000-of-00001.parquet",
            "train",
        ),
    }
    names = {24: "MMLU", 26: "ARC-Easy", 27: "ARC-Challenge", 28: "WinoGrande", 29: "HellaSwag"}
    # Held-out items whose text also appears in the suite's own source file are excluded
    # (MMLU, for one, repeats some questions between validation and test).
    text_field = {24: "question", 26: "question", 27: "question", 28: "sentence", 29: "ctx"}
    benches, maps = {}, {}
    for n, (repo, rev, path, slot, split) in specs.items():
        src = hf_file(repo, path, rev)
        rows = pq.read_table(src).to_pylist()
        suite_texts = {
            " ".join(str(x[text_field[n]]).split())
            for x in pq.read_table(REAL.sources / slot).to_pylist()
        }
        r_ = rng(n)
        idx = [
            i
            for i in range(len(rows))
            if " ".join(str(rows[i][text_field[n]]).split()) not in suite_texts
        ]
        if n == 28:
            idx = balanced(idx, lambda i: rows[i]["answer"], N, r_, ("1", "2"))
        else:
            r_.shuffle(idx)
            idx = idx[:N]
        b = Bench(n)
        parquet_pick(src, idx, b.raw_path(Path(path).name), dl.sources / slot)
        for slot_i, src_i in enumerate(idx):
            rid = rows[src_i].get("id") if n in (26, 27) else None
            key = f"{split}:{rid if rid is not None else src_i}"
            label = rows[src_i]["answer"] if n == 28 else None
            b.add(
                key,
                "heldout-split",
                {
                    "dataset": repo,
                    "revision": rev,
                    "file": path,
                    "split": split,
                    "row_index": src_i,
                    "record_id": rid,
                },
                label=label,
                raw_file=Path(path).name,
            )
            maps[(n, slot_i, rid)] = key
        benches[n] = b
    BUILDERS[24](dl)

    def key_for(n):
        def k(r):
            sid = r["id"].split(":", 2)[2]
            if n in (26, 27):
                return next(v for (m, _, rid), v in maps.items() if m == n and rid == sid)
            slot_i = int(sid.split(":")[0])
            return next(v for (m, si, _), v in maps.items() if m == n and si == slot_i)

        return k

    return [
        finish(benches[n], collect(benches[n], dl, n, names[n], key_for(n), "heldout-split"))
        for n in specs
        if not only or n in only
    ]


def h_anli(dl):
    import pyarrow.parquet as pq

    rev = "8e4813d81f46d313dac7892e1c28076917cfcdf9"
    b = Bench(12)
    alloc = {"r1": 31, "r2": 31, "r3": 38}  # suite test rounds: 1000 / 1000 / 1200
    r_ = rng(12)
    for rnd, k in alloc.items():
        path = f"plain_text/dev_{rnd}-00000-of-00001.parquet"
        src = hf_file("facebook/anli", path, rev)
        rows = pq.read_table(src).to_pylist()
        idx = list(range(len(rows)))
        r_.shuffle(idx)
        idx = idx[:k]
        parquet_pick(
            src,
            idx,
            b.raw_path(Path(path).name),
            dl.raw / "anli/plain_text" / f"test_{rnd}-00000-of-00001.parquet",
        )
        for i in idx:
            b.add(
                rows[i]["uid"],
                "heldout-split",
                {
                    "dataset": "facebook/anli",
                    "revision": rev,
                    "file": path,
                    "split": f"dev_{rnd}",
                    "row_index": i,
                    "record_id": rows[i]["uid"],
                },
                raw_file=Path(path).name,
            )
    BUILDERS[12](dl)
    return [
        finish(b, collect(b, dl, 12, "ANLI", lambda r: r["id"].split(":", 2)[2], "heldout-split"))
    ]


def h_banking77(dl):
    url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
    src = url_file(url, "banking77/train.csv")
    rows = list(csv.DictReader(src.open(newline="", encoding="utf-8")))
    idx = list(range(len(rows)))
    rng(4).shuffle(idx)
    idx = idx[:N]
    b = Bench(4)
    for path in (b.raw_path("train.csv"), dl.raw / "banking77/test.csv"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows[i] for i in idx)
    link(REAL.raw / "banking77/categories.json", dl.raw / "banking77/categories.json")
    keys = {}
    for slot_i, i in enumerate(idx):
        keys[slot_i] = f"train:{i}"
        b.add(
            keys[slot_i],
            "heldout-split",
            {
                "dataset": "PolyAI BANKING77",
                "url": url,
                "file_sha256": sha(src),
                "split": "train",
                "row_index": i,
            },
            raw_file="train.csv",
        )
    BUILDERS[4](dl)
    return [
        finish(
            b,
            collect(
                b, dl, 4, "BANKING77", lambda r: keys[int(r["id"].split(":")[-1])], "heldout-split"
            ),
        )
    ]


def h_clinc(dl):
    full = json.loads((REAL.raw / "clinc150/data_full.json").read_text())
    r_ = rng(5)
    picks = {}
    for split, k in (("val", 82), ("oos_val", 18)):  # suite: test 4500 / oos_test 1000
        idx = list(range(len(full[split])))
        r_.shuffle(idx)
        picks[split] = idx[:k]
    b = Bench(5)
    raw = {s: [full[s][i] for i in ix] for s, ix in picks.items()}
    (b.raw_path("data_full.subset.json")).write_text(json.dumps(raw, ensure_ascii=False))
    slot = dict(full)
    slot["test"], slot["oos_test"] = raw["val"], raw["oos_val"]
    p = dl.raw / "clinc150/data_full.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(slot))
    keys = {}
    for split, slot_split in (("val", "test"), ("oos_val", "oos_test")):
        for slot_i, i in enumerate(picks[split]):
            keys[(slot_split, slot_i)] = f"{split}:{i}"
            b.add(
                f"{split}:{i}",
                "heldout-split",
                {
                    "dataset": "clinc/oos-eval data_full.json",
                    "file_sha256": sha(REAL.raw / "clinc150/data_full.json"),
                    "split": split,
                    "row_index": i,
                },
                raw_file="data_full.subset.json",
            )
    BUILDERS[5](dl)
    return [
        finish(
            b,
            collect(
                b,
                dl,
                5,
                "CLINC150+OOS",
                lambda r: keys[(r["id"].split(":")[1], int(r["id"].split(":")[2]))],
                "heldout-split",
            ),
        )
    ]


def h_nli_docs(dl, only):
    """ContractNLI (dev, topped up from train) and NLI4CT (practice/dev set, 50/50)."""
    out = []
    # ContractNLI
    zp = REAL.repos / "contractnli/resources/contract-nli.zip"
    with zipfile.ZipFile(zp) as z:
        dev = json.loads(z.read("contract-nli/dev.json"))
        train = json.loads(z.read("contract-nli/train.json"))
    r_ = rng(11)
    dev_docs, train_docs = list(dev["documents"]), list(train["documents"])
    r_.shuffle(dev_docs)
    r_.shuffle(train_docs)
    chosen = [("dev", d) for d in dev_docs[:N]] + [
        ("train", d) for d in train_docs[: max(0, N - len(dev_docs))]
    ]
    b11 = Bench(11)
    subset = {"documents": [d for _, d in chosen], "labels": dev["labels"]}
    b11.raw_path("contract-nli.subset.json").write_text(json.dumps(subset, ensure_ascii=False))
    slot = dl.repos / "contractnli/resources/contract-nli.zip"
    slot.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(slot, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("contract-nli/test.json", json.dumps(subset))
    git_link(dl, "contractnli")
    for split, d in chosen:
        b11.add(
            str(d["id"]),
            "heldout-split",
            {
                "dataset": "stanfordnlp/contract-nli @eced652",
                "file": f"contract-nli.zip:contract-nli/{split}.json",
                "split": split,
                "record_id": d["id"],
            },
            raw_file="contract-nli.subset.json",
        )
    # NLI4CT
    repo = REAL.repos / "nli4ct"
    rows = json.loads((repo / "practice_test.json").read_text())
    gold = json.loads((repo / "gold_practice_test.json").read_text())
    with zipfile.ZipFile(repo / "training_data.zip") as z:
        have = set(z.namelist())
    ok = [
        u
        for u in rows
        if u in gold
        and all(
            ("CT json/" + rows[u][k] + ".json") in have
            for k in ("Primary_id", "Secondary_id")
            if k in rows[u]
        )
    ]
    pick = balanced(ok, lambda u: gold[u]["Label"], N, rng(42), ("Entailment", "Contradiction"))
    b42 = Bench(42)
    b42.raw_path("practice_test.subset.json").write_text(
        json.dumps({u: rows[u] for u in pick}, ensure_ascii=False)
    )
    b42.raw_path("gold_practice_test.subset.json").write_text(
        json.dumps({u: gold[u] for u in pick}, ensure_ascii=False)
    )
    (dl.repos / "nli4ct").mkdir(parents=True, exist_ok=True)
    (dl.repos / "nli4ct/test.json").write_text(json.dumps({u: rows[u] for u in pick}))
    (dl.repos / "nli4ct/gold_test.json").write_text(json.dumps({u: gold[u] for u in pick}))
    link(repo / "training_data.zip", dl.repos / "nli4ct/training_data.zip")
    git_link(dl, "nli4ct")
    for u in pick:
        b42.add(
            u,
            "heldout-split",
            {
                "dataset": "ai-systems/Task-2-SemEval-2024 @7f32fa6",
                "file": "practice_test.json + gold_practice_test.json",
                "split": "practice (dev)",
                "record_id": u,
            },
            label=gold[u]["Label"],
            raw_file="practice_test.subset.json",
        )
    BUILDERS[11](dl)
    if not only or 11 in only:
        out.append(
            finish(
                b11,
                collect(
                    b11, dl, 11, "ContractNLI", lambda r: r["id"].split(":", 2)[2], "heldout-split"
                ),
            )
        )
    if not only or 42 in only:
        out.append(
            finish(
                b42,
                collect(
                    b42, dl, 42, "NLI4CT-2024", lambda r: r["id"].split(":", 2)[2], "heldout-split"
                ),
            )
        )
    return out


def h_humor_gpqa(dl, only):
    out = []
    # Humicroedit: subtask-2 dev, 50/50 on which headline is funnier (ties excluded, as the adapter does)
    zp = REAL.downloads / "humicroedit-full.zip"
    with zipfile.ZipFile(zp) as z:
        text = z.read("semeval-2020-task-7-dataset/subtask-2/dev.csv").decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    pick = balanced(
        [r for r in rows if r["label"] in ("1", "2")], lambda r: r["label"], N, rng(21), ("1", "2")
    )
    b21 = Bench(21)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(pick)
    b21.raw_path("subtask-2-dev.subset.csv").write_text(buf.getvalue(), encoding="utf-8")
    slot = dl.downloads / "humicroedit-full.zip"
    slot.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(slot, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("semeval-2020-task-7-dataset/subtask-2/test.csv", buf.getvalue())
    for r in pick:
        b21.add(
            r["id"],
            "heldout-split",
            {
                "dataset": "SemEval-2020 Task 7 (Humicroedit)",
                "file": "subtask-2/dev.csv",
                "split": "dev",
                "record_id": r["id"],
            },
            label=r["label"],
            raw_file="subtask-2-dev.subset.csv",
        )
    # GPQA: main/extended questions that are not in Diamond. The adapter asserts 198 rows, so the
    # slot holds 198 held-out rows and only the first 100 (the samples) are kept.
    gp = REAL.sources / "gpqa/dataset.zip"
    with zipfile.ZipFile(gp) as z:
        read = lambda m: list(
            csv.DictReader(io.StringIO(z.read(m, pwd=b"deserted-untie-orchid").decode("utf-8-sig")))
        )
        diamond = read("dataset/gpqa_diamond.csv")
        main = read("dataset/gpqa_main.csv")
    dq = {r["Question"].strip() for r in diamond}
    pool = [r for r in main if r["Question"].strip() not in dq]
    r_ = rng(25)
    r_.shuffle(pool)
    slot_rows = pool[:198]
    b25 = Bench(25)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(main[0]))
    w.writeheader()
    w.writerows(slot_rows[:N])
    b25.raw_path("gpqa_main.not-diamond.subset.csv").write_text(buf.getvalue(), encoding="utf-8")
    full = io.StringIO()
    w = csv.DictWriter(full, fieldnames=list(main[0]))
    w.writeheader()
    w.writerows(slot_rows)
    gslot = dl.sources / "gpqa/dataset.zip"
    gslot.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(gslot, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("dataset/gpqa_diamond.csv", full.getvalue())
    for i, r in enumerate(slot_rows[:N]):
        b25.add(
            f"slot:{i}",
            "heldout-split",
            {
                "dataset": "idavidrein/gpqa dataset.zip",
                "file": "dataset/gpqa_main.csv",
                "split": "main (not in Diamond)",
                "record_id": r.get("Record ID"),
            },
            raw_file="gpqa_main.not-diamond.subset.csv",
        )
    BUILDERS[21](dl)
    if not only or 21 in only:
        out.append(
            finish(
                b21,
                collect(
                    b21, dl, 21, "Humicroedit", lambda r: r["id"].split(":", 2)[2], "heldout-split"
                ),
            )
        )
    if not only or 25 in only:
        out.append(
            finish(
                b25,
                collect(
                    b25,
                    dl,
                    25,
                    "GPQA-Diamond",
                    lambda r: f"slot:{r['id'].split(':')[-1]}",
                    "heldout-split",
                ),
            )
        )
    return out


def h_vast(dl):
    src = REAL.raw / "vast/data/VAST/vast_dev.csv"
    rows = list(csv.DictReader(src.open(newline="", encoding="utf-8")))
    seen, uniq = set(), []
    for r in rows:
        if r["new_id"] not in seen:
            seen.add(r["new_id"])
            uniq.append(r)
    rng(41).shuffle(uniq)
    pick = uniq[:N]
    b = Bench(41)
    for path in (b.raw_path("vast_dev.subset.csv"), dl.raw / "vast/data/VAST/vast_test.csv"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(pick)
    link(
        REAL.raw / "cladder/data/cladder-v1.zip", dl.raw / "cladder/data/cladder-v1.zip"
    )  # same builder
    for r in pick:
        b.add(
            r["new_id"],
            "heldout-split",
            {
                "dataset": "emilyallaway/zero-shot-stance (VAST) @e7c4775",
                "file": "vast_dev.csv",
                "split": "dev",
                "record_id": r["new_id"],
            },
            raw_file="vast_dev.subset.csv",
        )
    BUILDERS[41](dl)
    return [
        finish(b, collect(b, dl, 41, "VAST", lambda r: r["id"].split(":", 2)[2], "heldout-split"))
    ]


def h_isarcasm(dl):
    """Tasks A (En, Ar) and B (En) from the train files; task C has no train-side pairs file."""
    repo = REAL.repos / "isarcasm/train"
    en = list(csv.DictReader((repo / "train.En.csv").open(newline="", encoding="utf-8-sig")))
    ar = list(csv.DictReader((repo / "train.Ar.csv").open(newline="", encoding="utf-8-sig")))
    suite = Counter(
        r["_evaluation"]["track"]
        for r in read_jsonl(SUITE / "selected-rows.jsonl.gz")
        if r["_evaluation"]["catalog_id"] == 40
    )
    alloc = proportional(
        {k: suite.get(f"iSarcasmEval-{k}", 0) for k in ("A-En", "A-Ar", "B-En")}, N
    )
    for k in ("A-En", "A-Ar"):  # binary tracks are split 50/50, so they need even counts
        if alloc[k] % 2:
            alloc[k] -= 1
            alloc["B-En"] += 1
    r_ = rng(40)
    b = Bench(40)
    labels6 = [
        "sarcasm",
        "irony",
        "satire",
        "understatement",
        "overstatement",
        "rhetorical_question",
    ]
    plans = {
        "A-En": (
            en,
            lambda r: r["sarcastic"],
            [r for r in en if r["sarcastic"] in ("0", "1") and r["tweet"].strip()],
            lambda r: {"text": r["tweet"], "sarcastic": r["sarcastic"]},
            "train.En.csv",
        ),
        "A-Ar": (
            ar,
            lambda r: r["sarcastic"],
            [r for r in ar if r["sarcastic"] in ("0", "1") and r["text"].strip()],
            lambda r: {"text": r["text"], "dialect": r["dialect"], "sarcastic": r["sarcastic"]},
            "train.Ar.csv",
        ),
        "B-En": (
            en,
            None,
            [
                r
                for r in en
                if r["sarcastic"] == "1"
                and all(r.get(x) in ("0", "1") for x in labels6)
                and r["tweet"].strip()
            ],
            lambda r: {"text": r["tweet"], **{x: r[x] for x in labels6}},
            "train.En.csv",
        ),
    }
    keys = {}
    for fam, (_, lab, pool, shim, fname) in plans.items():
        k = alloc[fam]
        pick = balanced(pool, lab, k, r_, ("1", "0")) if lab else r_.sample(pool, k)
        task, lang = fam.split("-")
        raw_name = f"{fname[:-4]}.{fam}.subset.csv"
        with b.raw_path(raw_name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(pool[0]))
            w.writeheader()
            w.writerows(pick)
        slot = dl.repos / f"isarcasm/test/task_{task}_{lang}_test.csv"
        slot.parent.mkdir(parents=True, exist_ok=True)
        shimmed = [shim(r) for r in pick]
        with slot.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(shimmed[0]))
            w.writeheader()
            w.writerows(shimmed)
        src_rows = plans[fam][0]
        for slot_i, r in enumerate(pick):
            src_i = src_rows.index(r)
            key = f"{fam}:{src_i}"
            keys[(f"iSarcasmEval-{task}-{lang}", slot_i)] = key
            b.add(
                key,
                "heldout-split",
                {
                    "dataset": "iabufarha/iSarcasmEval @dfc708b",
                    "file": f"train/{fname}",
                    "split": "train",
                    "row_index": src_i,
                    "task": fam,
                    "shim": "column tweet->text (En)"
                    if lang == "En"
                    else "columns text,dialect,sarcastic",
                },
                label=r["sarcastic"] if lab else None,
                raw_file=raw_name,
            )
    git_link(dl, "isarcasm")
    BUILDERS[40](dl)
    rows = []
    for fam in ("A-En", "A-Ar", "B-En"):
        task, lang = fam.split("-")
        name = f"iSarcasmEval-{task}-{lang}"
        rows += collect(
            b,
            dl,
            40,
            name,
            lambda r, name=name: keys[(name, int(r["id"].split(":")[-1]))],
            "heldout-split",
        )
    return [finish(b, rows)]


def h_esci(dl):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    base = REAL.repos / "esci/shopping_queries_dataset"
    src = base / "shopping_queries_dataset_examples.parquet"
    t = pq.read_table(src)
    train_idx = pc.indices_nonzero(pc.equal(t["split"], "train")).to_pylist()
    loc, lab = t["product_locale"].to_pylist(), t["esci_label"].to_pylist()
    strata = defaultdict(list)
    for i in train_idx:
        strata[(loc[i], lab[i])].append(i)
    suite_strata = Counter(
        (r["metadata"]["locale"], r["expected"]["answer"])
        for r in read_jsonl(SUITE / "selected-rows.jsonl.gz")
        if r["_evaluation"]["catalog_id"] == 37
    )
    alloc = proportional(dict(suite_strata), N)
    r_ = rng(37)
    pick = [i for k, c in sorted(alloc.items()) for i in r_.sample(strata[k], c)]
    b = Bench(37)
    write_parquet_subset(src, pick, b.raw_path("examples.train.subset.parquet"))
    slot = dl.repos / "esci/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"
    slot.parent.mkdir(parents=True, exist_ok=True)
    sub = t.take(pa.array(pick, pa.int64()))
    sub = sub.set_column(
        sub.schema.get_field_index("split"), "split", pa.array(["test"] * sub.num_rows)
    )
    pq.write_table(sub, slot)
    link(
        base / "shopping_queries_dataset_products.parquet",
        slot.parent / "shopping_queries_dataset_products.parquet",
    )
    git_link(dl, "esci")
    ex = t["example_id"].to_pylist()
    for i in pick:
        b.add(
            str(ex[i]),
            "heldout-split",
            {
                "dataset": "amazon-science/esci-data @7916cdf",
                "file": "shopping_queries_dataset_examples.parquet",
                "split": "train",
                "record_id": ex[i],
                "shim": "split column set to 'test' in the adapter's input copy only",
            },
            label=None,
            raw_file="examples.train.subset.parquet",
        )
    BUILDERS[37](dl)
    return [
        finish(
            b,
            collect(b, dl, 37, "Amazon-ESCI", lambda r: r["id"].split(":", 2)[2], "heldout-split"),
        )
    ]


def h_acos(dl):
    real = REAL.repos / "acos/data"
    suite_dom = Counter(
        r["_evaluation"]["group_id"].split(":")[0]
        for r in read_jsonl(SUITE / "selected-rows.jsonl.gz")
        if r["_evaluation"]["catalog_id"] == 38
    )
    groups = Counter({d: len({g for g in []}) for d in suite_dom})
    doms = {}
    for r in read_jsonl(SUITE / "selected-rows.jsonl.gz"):
        if r["_evaluation"]["catalog_id"] == 38:
            doms.setdefault(r["_evaluation"]["group_id"].split(":")[0], set()).add(
                r["_evaluation"]["group_id"]
            )
    alloc = proportional({d: len(v) for d, v in doms.items()}, N)
    del groups, suite_dom
    b = Bench(38)
    r_ = rng(38)
    for domain_dir in sorted(p for p in real.iterdir() if p.is_dir()):
        for f in domain_dir.glob("*.tsv"):
            link(f, dl.repos / "acos/data" / domain_dir.name / f.name)
        dev = next(domain_dir.glob("*_quad_dev.tsv"))
        lines = [line for line in dev.read_text(encoding="utf-8").splitlines() if line]
        idx = list(range(len(lines)))
        r_.shuffle(idx)
        idx = sorted(idx[: alloc[domain_dir.name]])
        stem = dev.stem.replace("_dev", "_devsample_test")
        text = "".join(lines[i] + "\n" for i in idx)
        b.raw_path(f"{domain_dir.name}/{dev.stem}.subset.tsv").write_text(text, encoding="utf-8")
        (dl.repos / "acos/data" / domain_dir.name / f"{stem}.tsv").write_text(
            text, encoding="utf-8"
        )
        for slot_line, i in enumerate(idx, 1):
            b.add(
                f"{domain_dir.name}:{stem}:line-{slot_line}",
                "heldout-split",
                {
                    "dataset": "NUSTM/ACOS @45d179a",
                    "file": f"{domain_dir.name}/{dev.name}",
                    "split": "dev",
                    "line": i + 1,
                },
                raw_file=f"{domain_dir.name}/{dev.stem}.subset.tsv",
            )
    BUILDERS[38](dl)
    return [
        finish(
            b,
            collect(
                b,
                dl,
                38,
                "ACOS-category-sentiment",
                lambda r: r["metadata"]["group_id"],
                "heldout-split",
            ),
        )
    ]


def h_sgd(dl):
    sha_ = "e852981ae34990f4358979625854259302feaa78"
    base = f"https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/{sha_}/dev"
    schema = json.loads(url_file(f"{base}/schema.json", "sgd/dev/schema.json").read_text())
    dialogues = []
    for k in range(1, 21):
        p = url_file(f"{base}/dialogues_{k:03d}.json", f"sgd/dev/dialogues_{k:03d}.json")
        dialogues += [(p.name, d) for d in json.loads(p.read_text())]
    r_ = rng(10)
    r_.shuffle(dialogues)
    chosen = dialogues[:40]
    slot = dl.repos / "sgd/test"
    slot.mkdir(parents=True, exist_ok=True)
    (slot / "schema.json").write_text(json.dumps(schema))
    (slot / "dialogues_001.json").write_text(json.dumps([d for _, d in chosen]))
    BUILDERS[10](dl)
    rows = list(read_jsonl(dl.normalized / "SGD-service-given-intent.jsonl"))
    r_.shuffle(rows)
    rows = rows[:N]
    b = Bench(10)
    used = {r["metadata"]["provenance"]["dialogue_id"] for r in rows}
    b.raw_path("dev/schema.json").write_text(json.dumps(schema, ensure_ascii=False))
    write_jsonl(
        b.raw_path("dev/dialogues.subset.jsonl"),
        [{"file": f, "dialogue": d} for f, d in chosen if d["dialogue_id"] in used],
    )
    src_file = {d["dialogue_id"]: f for f, d in chosen}
    for r in rows:
        pv = r["metadata"]["provenance"]
        b.add(
            r["id"],
            "heldout-split",
            {
                "dataset": f"google-research-datasets/dstc8-schema-guided-dialogue @{sha_[:7]}",
                "file": f"dev/{src_file[pv['dialogue_id']]}",
                "split": "dev",
                "dialogue_id": pv["dialogue_id"],
                "turn_index": pv["turn_index"],
                "service": pv["service"],
            },
            raw_file="dev/dialogues.subset.jsonl",
        )
    return [
        finish(
            b, collect(b, dl, 10, "SGD-service-given-intent", lambda r: r["id"], "heldout-split")
        )
    ]


def h_bfcl(dl):
    base = REAL.repos / "bfcl/berkeley-function-call-leaderboard/data"
    items, gold = [], {}
    for cat in ("multiple", "parallel_multiple"):
        for line in (base / f"BFCL_v3_{cat}.json").open():
            items.append((cat, json.loads(line)))
        for line in (base / f"possible_answer/BFCL_v3_{cat}.json").open():
            g = json.loads(line)
            gold[g["id"]] = g
    slot = dl.repos / "bfcl/berkeley-function-call-leaderboard/data"
    (slot / "possible_answer").mkdir(parents=True, exist_ok=True)
    write_jsonl(slot / "BFCL_v3_simple.json", [x for _, x in items])
    write_jsonl(slot / "possible_answer/BFCL_v3_simple.json", [gold[x["id"]] for _, x in items])
    BUILDERS[1](dl)
    out_rows = list(read_jsonl(dl.normalized / "BFCL-tool-selection.jsonl"))
    by_cat = defaultdict(list)
    cat_of = {x["id"]: c for c, x in items}
    for r in out_rows:
        by_cat[cat_of[r["metadata"]["provenance"]["source_id"]]].append(
            r["metadata"]["provenance"]["source_id"]
        )
    r_ = rng(1)
    pick = []
    for c in ("multiple", "parallel_multiple"):
        ids = sorted(by_cat[c])
        r_.shuffle(ids)
        pick += ids[: N // 2]
    b = Bench(1)
    byid = {x["id"]: (c, x) for c, x in items}
    write_jsonl(b.raw_path("BFCL_v3.heldout-categories.jsonl"), [byid[i][1] for i in pick])
    write_jsonl(b.raw_path("possible_answer.heldout-categories.jsonl"), [gold[i] for i in pick])
    for i in pick:
        b.add(
            i,
            "heldout-category",
            {
                "dataset": "gorilla-llm/gorilla BFCL v3 @916260d",
                "file": f"data/BFCL_v3_{byid[i][0]}.json",
                "category": byid[i][0],
                "note": "category not read by the suite's adapter (simple, live_simple, live_multiple)",
                "record_id": i,
            },
            raw_file="BFCL_v3.heldout-categories.jsonl",
        )
    return [
        finish(
            b,
            collect(
                b,
                dl,
                1,
                "BFCL-tool-selection",
                lambda r: r["metadata"]["provenance"]["source_id"],
                "heldout-category",
            ),
        )
    ]


def h_forecast(dl):
    from decision_index.suite.build import adapters_scored as SC

    link(REAL.repos / "forecastbench-datasets", dl.repos / "forecastbench-datasets")
    lo, hi = SC.MIN_DATE, SC.MAX_DATE
    import datetime

    SC.MIN_DATE = "2000-01-01"
    SC.MAX_DATE = (datetime.date.fromisoformat(lo) - datetime.timedelta(days=1)).isoformat()
    try:
        BUILDERS[48](dl)
    finally:
        window = (SC.MIN_DATE, SC.MAX_DATE)
        SC.MIN_DATE, SC.MAX_DATE = lo, hi
    rows = list(read_jsonl(dl.normalized / "ForecastBench-binary.jsonl"))
    pick = balanced(rows, lambda r: r["expected"]["answer"], N, rng(48), ("yes", "no"))
    b = Bench(48)
    recs = []
    for r in pick:
        stem, ix = r["id"].split(":")[1], int(r["id"].split(":")[2])
        rp = REAL.repos / "forecastbench-datasets/datasets/resolution_sets" / f"{stem}.json"
        rr = json.loads(rp.read_text())
        recs.append(
            {
                "resolution_set": rp.name,
                "index": ix,
                "resolution": rr["resolutions"][ix],
                "question_set": rr["question_set"],
            }
        )
        b.add(
            r["id"],
            "heldout-window",
            {
                "dataset": "forecastingresearch/forecastbench-datasets @da48cfb",
                "file": f"datasets/resolution_sets/{rp.name}",
                "index": ix,
                "resolution_date": r["state"].get("resolution_date"),
                "note": f"resolved {window[0]}..{window[1]}, before the suite window {lo}..{hi}",
            },
            label=r["expected"]["answer"],
            raw_file="resolutions.subset.jsonl",
        )
    write_jsonl(b.raw_path("resolutions.subset.jsonl"), recs)
    return [
        finish(b, collect(b, dl, 48, "ForecastBench-binary", lambda r: r["id"], "heldout-window"))
    ]


# ----------------------------------------------------------------------------- 0.2 additions


def added(dl, n, rows_by_key, kind):
    """Run the kit's own 0.2 builder on the dev layout and keep the dev samples."""
    tmp = DEVWORK / f"added-{n}.jsonl"
    AA.build(dl, tmp, numbers=[n])
    out = []
    for r in read_jsonl(tmp):
        k = rows_by_key(r)
        if k is not None:
            r["_evaluation"]["run_id"] = "dev:" + r["_evaluation"]["run_id"]
            r["_dev"] = {"key": k, "source_kind": kind}
            out.append(r)
    return out


def by_source(keys):
    """Key for 0.2-addition rows: the builder's metadata.source_id, compared as a string."""

    def key(r):
        k = str(r["metadata"]["source_id"])
        return k if k in keys else None

    return key


def h_mmlu_pro(dl):
    rev = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
    import pyarrow.parquet as pq

    src = hf_file("TIGER-Lab/MMLU-Pro", "data/validation-00000-of-00001.parquet", rev)
    rows = pq.read_table(src).to_pylist()
    b = Bench(57)
    parquet_pick(
        src,
        list(range(len(rows))),
        b.raw_path("validation-00000-of-00001.parquet"),
        dl.raw / "mmlu_pro" / AA.HF["mmlu_pro"][2],
    )
    for i, r in enumerate(rows):
        b.add(
            str(r["question_id"]),
            "heldout-split",
            {
                "dataset": "TIGER-Lab/MMLU-Pro",
                "revision": rev,
                "file": "data/validation-00000-of-00001.parquet",
                "split": "validation",
                "row_index": i,
                "record_id": r["question_id"],
                "note": "the whole validation split (70 items); no hand-written top-up",
            },
            raw_file="validation-00000-of-00001.parquet",
        )
    keys = {s["key"] for s in b.samples}
    return [
        finish(
            b,
            added(dl, 57, by_source(keys), "heldout-split"),
        )
    ]


def h_ragtruth(dl):
    folder = REAL.repos / "cand-RAGTruth/dataset"
    sources = {
        x["source_id"]: x
        for x in map(json.loads, (folder / "source_info.jsonl").open(encoding="utf-8"))
    }
    train = [
        r
        for r in map(json.loads, (folder / "response.jsonl").open(encoding="utf-8"))
        if r["split"] == "train"
    ]
    pick = balanced(train, lambda r: len(r["labels"]) > 0, N, rng(59))
    b = Bench(59)
    write_jsonl(b.raw_path("response.train.subset.jsonl"), pick)
    write_jsonl(
        b.raw_path("source_info.subset.jsonl"),
        [sources[s] for s in sorted({r["source_id"] for r in pick})],
    )
    slot = dl.repos / "cand-RAGTruth/dataset"
    slot.mkdir(parents=True, exist_ok=True)
    write_jsonl(slot / "response.jsonl", [{**r, "split": "test"} for r in pick])
    write_jsonl(
        slot / "source_info.jsonl", [sources[s] for s in sorted({r["source_id"] for r in pick})]
    )
    for r in pick:
        b.add(
            str(r["id"]),
            "heldout-split",
            {
                "dataset": "ParticleMedia/RAGTruth @c103204",
                "file": "dataset/response.jsonl",
                "split": "train",
                "record_id": r["id"],
                "source_id": r["source_id"],
                "shim": "split set to 'test' in the adapter's input copy only",
            },
            label=len(r["labels"]) > 0,
            raw_file="response.train.subset.jsonl",
        )
    keys = {s["key"] for s in b.samples}
    return [
        finish(
            b,
            added(dl, 59, by_source(keys), "heldout-split"),
        )
    ]


def h_hover(dl):
    import sqlite3
    import unicodedata

    path = REAL.repos / "cand-hover/data/hover/hover_train_release_v1.1.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    db = sqlite3.connect(f"file:{REAL.raw / AA.HOVER_DB[0]}?mode=ro", uri=True)

    def resolvable(r):
        return all(
            len(
                db.execute(
                    "SELECT id FROM documents WHERE id=?", (unicodedata.normalize("NFD", t),)
                ).fetchall()
            )
            == 1
            for t in dict.fromkeys(t for t, _ in r["supporting_facts"])
        )

    r_ = rng(61)
    cand = list(rows)
    r_.shuffle(cand)
    cand = [r for r in cand[:2000] if resolvable(r)]
    db.close()
    pick = balanced(cand, lambda r: r["label"], N, r_, ("SUPPORTED", "NOT_SUPPORTED"))
    b = Bench(61)
    b.raw_path("hover_train_release_v1.1.subset.json").write_text(
        json.dumps(pick, ensure_ascii=False)
    )
    slot = dl.repos / "cand-hover/data/hover/hover_dev_release_v1.1.json"
    slot.parent.mkdir(parents=True, exist_ok=True)
    slot.write_text(json.dumps(pick))
    link(REAL.raw / AA.HOVER_DB[0], dl.raw / AA.HOVER_DB[0])
    for r in pick:
        b.add(
            str(r["uid"]),
            "heldout-split",
            {
                "dataset": "hover-nlp/hover @39b8469",
                "file": "data/hover/hover_train_release_v1.1.json",
                "split": "train",
                "record_id": r["uid"],
            },
            label=r["label"],
            raw_file="hover_train_release_v1.1.subset.json",
        )
    keys = {s["key"] for s in b.samples}
    return [
        finish(
            b,
            added(dl, 61, by_source(keys), "heldout-split"),
        )
    ]


def h_newyorker(dl):
    rev = "d81cbab7d0392708d5371d3a4960e69261824db4"
    import pyarrow.parquet as pq

    src = hf_file(
        "jmhessel/newyorker_caption_contest", "matching/validation-00000-of-00001.parquet", rev
    )
    t = pq.read_table(src)
    idx = list(range(t.num_rows))
    rng(64).shuffle(idx)
    idx = idx[:N]
    b = Bench(64)
    drop_image = lambda tb: tb.drop(["image"]) if "image" in tb.column_names else tb
    import pyarrow as pa

    sub = drop_image(t.take(pa.array(idx, pa.int64())))
    pq.write_table(sub, b.raw_path("matching-validation.subset.parquet"))
    slot = dl.raw / "newyorker" / AA.HF["newyorker"][2]
    slot.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t.take(pa.array(idx, pa.int64())), slot)
    ids = t["instance_id"].to_pylist()
    for i in idx:
        b.add(
            str(ids[i]),
            "heldout-split",
            {
                "dataset": "jmhessel/newyorker_caption_contest",
                "revision": rev,
                "file": "matching/validation-00000-of-00001.parquet",
                "split": "validation",
                "record_id": ids[i],
                "note": "image column not stored (the adapter does not use it)",
            },
            raw_file="matching-validation.subset.parquet",
        )
    keys = {s["key"] for s in b.samples}
    return [
        finish(
            b,
            added(dl, 64, by_source(keys), "heldout-split"),
        )
    ]


def h_phish(dl):
    rev = AA.HF["phishnchips"][1]
    real = list(
        csv.DictReader(
            hf_file("AreLit/PhishNChips", "real_phishing_validation.csv", rev).open(
                newline="", encoding="utf-8"
            )
        )
    )
    hand = list(
        csv.DictReader(
            (OUT / "handwritten/phishnchips_legitimate.csv").open(newline="", encoding="utf-8")
        )
    )
    r_ = rng(56)
    phish = r_.sample(real, N // 2)
    rows = phish + hand[: N // 2]
    r_.shuffle(rows)
    b = Bench(56)
    fields = list(real[0])
    for path in (b.raw_path("emails.subset.csv"), dl.raw / "phishnchips/core_emails.csv"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    link(REAL.repos / "jev-phishing-bench", dl.repos / "jev-phishing-bench")
    for r in rows:
        if r["datasource"] == "handwritten_dev_v1":
            src = {
                "dataset": "handwritten for this dev set",
                "file": "evals/decision-index-dev/handwritten/phishnchips_legitimate.csv",
                "record_id": r["id"],
            }
            kind = "handwritten"
        else:
            src = {
                "dataset": "AreLit/PhishNChips",
                "revision": rev,
                "file": "real_phishing_validation.csv",
                "split": "validation (not in core_emails.csv)",
                "record_id": r["id"],
            }
            kind = "heldout-split"
        b.add(
            r["id"],
            kind,
            src,
            label="phishing" if r["phish_label"] == "1" else "legitimate",
            raw_file="emails.subset.csv",
        )
    pinned = AA.HF["phishnchips"]
    AA.HF["phishnchips"] = pinned[:3] + (
        sha(dl.raw / "phishnchips/core_emails.csv"),
    )  # the adapter's input hash check, in memory only
    try:
        rows_out = added(dl, 56, lambda r: r["metadata"]["source_id"], "mixed")
    finally:
        AA.HF["phishnchips"] = pinned
    kinds = {s["key"]: s["source_kind"] for s in b.samples}
    for r in rows_out:
        r["_dev"]["source_kind"] = kinds[r["_dev"]["key"]]
    return [finish(b, rows_out)]


def h_gsm8k(dl):
    """GSM8K train split; the adapter derives a 4-choice and a 10-choice request per problem."""
    import pyarrow.parquet as pq

    rev = "740312add88f781978c0658806c59bc2815b9866"
    path = "main/train-00000-of-00001.parquet"
    src = hf_file("openai/gsm8k", path, rev)
    rows = pq.read_table(src).to_pylist()
    test_q = {
        " ".join(x["question"].split())
        for x in pq.read_table(REAL.sources / "gsm8k/main/test-00000-of-00001.parquet").to_pylist()
    }
    from decimal import Decimal

    def convertible(r):  # the adapter's own distractor pool must yield 9 wrong answers
        gold = Decimal(r["answer"].split("####")[-1].strip().replace(",", ""))
        pool = (
            {gold + Decimal(d) for d in [-10, -5, -2, -1, 1, 2, 5, 10]}
            | {gold * 2, gold * 10, gold // 2}
        ) - {gold}
        return len(pool) >= 9

    idx = [
        i
        for i in range(len(rows))
        if " ".join(rows[i]["question"].split()) not in test_q and convertible(rows[i])
    ]
    rng(30).shuffle(idx)
    idx = idx[:N]
    b = Bench(30)
    parquet_pick(
        src, idx, b.raw_path(Path(path).name), dl.sources / "gsm8k/main/test-00000-of-00001.parquet"
    )
    link(REAL.repos / "cruxeval", dl.repos / "cruxeval")  # the same builder also converts CRUXEval
    keys = {}
    for slot_i, i in enumerate(idx):
        keys[slot_i] = f"train:{i}"
        b.add(
            keys[slot_i],
            "heldout-split",
            {
                "dataset": "openai/gsm8k",
                "revision": rev,
                "file": path,
                "split": "train",
                "row_index": i,
            },
            raw_file=Path(path).name,
        )
    BUILDERS[30](dl)
    rows_out = []
    for name in NAMES[30]:
        rows_out += collect(
            b, dl, 30, name, lambda r: keys[int(r["id"].split(":")[-1])], "heldout-split"
        )
    return [finish(b, rows_out)]


HELDOUT = {
    "knowledge": ((24, 26, 27, 28, 29), h_knowledge),
    "anli": ((12,), h_anli),
    "gsm8k": ((30,), h_gsm8k),
    "banking77": ((4,), h_banking77),
    "clinc": ((5,), h_clinc),
    "nli_docs": ((11, 42), h_nli_docs),
    "humor_gpqa": ((21, 25), h_humor_gpqa),
    "vast": ((41,), h_vast),
    "isarcasm": ((40,), h_isarcasm),
    "esci": ((37,), h_esci),
    "acos": ((38,), h_acos),
    "sgd": ((10,), h_sgd),
    "bfcl": ((1,), h_bfcl),
    "forecast": ((48,), h_forecast),
    "mmlu_pro": ((57,), h_mmlu_pro),
    "ragtruth": ((59,), h_ragtruth),
    "hover": ((61,), h_hover),
    "newyorker": ((64,), h_newyorker),
    "phish": ((56,), h_phish),
}
MULTI = {"knowledge", "nli_docs", "humor_gpqa"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*", help="catalog ids to (re)build")
    args = ap.parse_args()
    only = set(args.only or [])
    results = []
    for n, fn in UNSELECTED.items():
        if not only or n in only:
            print(f"[{n}] {bench_name(n)} (suite-unselected/generator)", flush=True)
            results.append(fn())
    for name, (ids, fn) in HELDOUT.items():
        if only and not (set(ids) & only):
            continue
        print(f"[{'/'.join(map(str, ids))}] held-out via adapter", flush=True)
        dl = dev_layout()
        results += fn(dl, only) if name in MULTI else fn(dl)
    manifest_path = OUT / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {"benchmarks": {}}
    )
    for r in results:
        manifest["benchmarks"][str(r["catalog_id"])] = r
    manifest.update(
        seed=SEED,
        kit=str(KIT),
        suite="Decision Index 0.2 (kit commit 19ad28e)",
        samples_per_benchmark=N,
    )
    for entry in manifest["benchmarks"].values():
        entry["restricted"] = RESTRICTED.get(entry["catalog_id"])
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # One file with every converted request, for running (local only: includes restricted rows).
    with gzip.open(OUT / "dev-rows.jsonl.gz", "wt", encoding="utf-8", newline="\n") as out:
        for conv in sorted((OUT / "converted").glob("*.jsonl.gz")):
            with gzip.open(conv, "rt", encoding="utf-8") as f:
                shutil.copyfileobj(f, out)
    for r in results:
        print(
            json.dumps(
                {
                    k: r[k]
                    for k in (
                        "catalog_id",
                        "benchmark",
                        "samples",
                        "converted_requests",
                        "source_kinds",
                        "labels",
                    )
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
