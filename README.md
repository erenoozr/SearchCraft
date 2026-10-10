# SearchCraft

A from-scratch full-text search engine in pure Python (3.10+, standard library only) featuring Okapi BM25 ranking, a positional inverted index, and a Boolean/phrase query language.

## Features

- **Text pipeline** — lowercasing, punctuation and digit removal, English stop words, a light suffix-stripping stemmer (`learning`, `learned` → `learn`), and syllable-based n-grams (`computer → com-pu, pu-ter`).
- **Positional inverted index** — `term -> {doc_id: [positions]}`, plus document lengths, average document length and document frequencies.
- **Okapi BM25** — configurable `k1` (default 1.5) and `b` (default 0.75); IDF uses `ln((N − df + 0.5) / (df + 0.5) + 1)` so it never goes negative.
- **Query language** — `"exact phrases"`, `AND`, `OR`, `NOT`, parentheses and implicit AND. Precedence: `NOT > AND > OR`. Phrases are matched by consecutive positions and scored as a single pseudo-term.
- **CLI** — interactive shell that shows rank, doc ID, score, and a snippet with matches in `[brackets]`.

## Usage

```bash
python searchcraft.py                  # interactive shell (:q to quit)
python searchcraft.py --demo           # run the 4 demo queries
python searchcraft.py -q '"machine learning" AND python NOT java'
python searchcraft.py --k1 1.2 --b 0.5 --top 3
```

Example output:

```
Query: "machine learning" AND python NOT java
  Rank DocID     Score  Title
  1    1        3.6741  Introduction to Machine Learning with Python
                      ↳ Introduction to [Machine] [Learning] with [Python]. [Machine] [learning] is the…
  2    3        2.2602  Deep Learning with PyTorch
  3    10       2.1124  Natural Language Processing and Information Retrieval
```

## Library use

```python
from searchcraft import SearchEngine

engine = SearchEngine(k1=1.5, b=0.75)
engine.add_document("Docker packages apps into containers.", title="Docker 101", tags=["devops"])
for r in engine.search("docker OR kubernetes"):
    print(r.rank, r.doc_id, round(r.score, 3), r.snippet)
```

## Project layout

| File | Contents |
|---|---|
| `searchcraft.py` | Pipeline, inverted index, BM25 scorer, query parser, CLI |
| `dataset.py` | 10 technical demo documents |
| `test_searchcraft.py` | Unit tests, including a hand-computed BM25 check and ranking tests for the demo queries |

## Tests

```bash
python -m unittest -v
```

## Limitations

- The stemmer is rule-based and can over-stem (e.g. `kubernetes → kubernet`). Queries and documents go through the same pipeline, so matching stays consistent.
- The index lives in memory only; there is no persistence or document deletion.
