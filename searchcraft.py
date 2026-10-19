"""SearchCraft — a BM25 + Boolean/phrase search engine in pure Python (3.10+).

Stages:
  1. TextPipeline   : normalize -> strip digits -> stop words -> light stemming (+ syllable n-grams)
  2. InvertedIndex  : term -> {doc_id: [positions]}, doc_lengths, avg_doc_len, DF
  3. BM25Scorer     : Okapi BM25 (k1, b), term and phrase scores
  4. QueryParser    : "phrases", AND / OR / NOT, parentheses, implicit AND -> AST
  5. CLI            : interactive shell, single query (-q) and demo (--demo)
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field

# ============================================================================
# STAGE 1 — Text processing pipeline
# ============================================================================

_WORD_RE = re.compile(r"\w+")
_DIGIT_RE = re.compile(r"[\d_]+")
_VOWELS = frozenset("aeiouy")

STOP_WORDS = """
a an and are as at be but by can do does for from has have how if in into is it its no
not of on or so such than that the their then there these they this those to was were
what when where which while who will with
""".split()


class LightStemmer:
    """Dictionary-free light suffix stripper for English.

    shortcut: rule-based, no exception list, may over-stem —
    switch to a real stemmer (Porter/Snowball) if precision matters.
    """

    SUFFIXES = tuple(sorted({
        "ations", "ation", "ments", "ment", "ings", "ing", "edly", "ed", "ies", "es", "ly", "s",
    }, key=len, reverse=True))

    def __init__(self, min_stem: int = 3, max_passes: int = 3) -> None:
        self.min_stem = min_stem
        self.max_passes = max_passes

    def stem(self, word: str) -> str:
        for _ in range(self.max_passes):
            for suf in self.SUFFIXES:
                if word.endswith(suf) and len(word) - len(suf) >= self.min_stem:
                    word = word[: -len(suf)]
                    break
            else:
                break  # nothing was stripped in this pass
        return word


def syllabify(word: str) -> list[str]:
    """Vowel-based syllable split: one vowel per syllable; the last consonant
    between two vowels starts the next syllable.

    shortcut: approximate for English (no silent-e or digraph rules), fine for n-gram features.
    """
    w = word.lower()
    vowels = [i for i, c in enumerate(w) if c in _VOWELS]
    if len(vowels) < 2:
        return [w]
    cuts = [b if b - a == 1 else b - 1 for a, b in zip(vowels, vowels[1:])]
    return [w[i:j] for i, j in zip([0, *cuts], [*cuts, len(w)])]


def syllable_ngrams(word: str, n: int = 2) -> list[str]:
    """Syllable-based n-grams: computer, n=2 -> ['com-pu', 'pu-ter']."""
    syl = syllabify(word)
    if len(syl) < n:
        return ["-".join(syl)]
    return ["-".join(syl[i:i + n]) for i in range(len(syl) - n + 1)]


@dataclass(frozen=True, slots=True)
class Token:
    term: str      # processed term
    position: int  # word index in the original text (stop words are counted too)


class TextPipeline:
    """Modular pipeline: each filter maps str -> str; an empty result drops the token."""

    def __init__(self, stop_words: list[str] | None = None, stemmer: LightStemmer | None = None) -> None:
        self.stop_words = {w.lower() for w in (stop_words or STOP_WORDS)}
        self.stemmer = stemmer or LightStemmer()
        self.filters = (str.lower, self.strip_digits, self.drop_stop_word, self.stemmer.stem)

    @staticmethod
    def strip_digits(token: str) -> str:
        return _DIGIT_RE.sub("", token)

    def drop_stop_word(self, token: str) -> str:
        return "" if token in self.stop_words else token

    @staticmethod
    def word_spans(text: str) -> list[tuple[int, int]]:
        """Word position -> (start, end) character span; used for snippets."""
        return [m.span() for m in _WORD_RE.finditer(text)]

    def analyze(self, text: str) -> list[Token]:
        tokens = []
        for pos, m in enumerate(_WORD_RE.finditer(text)):  # \w+ already drops punctuation
            term = m.group()
            for f in self.filters:
                term = f(term)
                if not term:
                    break
            else:
                tokens.append(Token(term, pos))
        return tokens

    def terms(self, text: str) -> list[str]:
        return [t.term for t in self.analyze(text)]


# ============================================================================
# STAGE 2 — Inverted index
# ============================================================================

@dataclass(slots=True)
class Document:
    doc_id: int
    title: str
    text: str
    metadata: dict
    spans: list[tuple[int, int]]


class InvertedIndex:
    def __init__(self, pipeline: TextPipeline | None = None) -> None:
        self.pipeline = pipeline or TextPipeline()
        self.postings: dict[str, dict[int, list[int]]] = defaultdict(dict)
        self.docs: dict[int, Document] = {}
        self.doc_lengths: dict[int, int] = {}
        self._total_len = 0
        self._next_id = 1

    def add_document(self, text: str, title: str = "", **metadata) -> int:
        doc_id = self._next_id
        self._next_id += 1
        # the title is searchable content too
        full = f"{title}. {text}" if title else text
        tokens = self.pipeline.analyze(full)
        for tok in tokens:
            self.postings[tok.term].setdefault(doc_id, []).append(tok.position)
        self.docs[doc_id] = Document(doc_id, title, full, metadata, self.pipeline.word_spans(full))
        self.doc_lengths[doc_id] = len(tokens)
        self._total_len += len(tokens)
        return doc_id

    @property
    def num_docs(self) -> int:
        return len(self.docs)

    @property
    def avg_doc_len(self) -> float:
        return self._total_len / self.num_docs if self.docs else 0.0

    def df(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def doc_postings(self, term: str) -> dict[int, list[int]]:
        return self.postings.get(term, {})  # .get so lookups don't insert empty entries

    def phrase_matches(self, terms: list[tuple[str, int]]) -> dict[int, list[int]]:
        """Phrase matches: terms = [(term, offset relative to the first term)].
        Returns doc_id -> positions where the phrase starts."""
        posting_lists = [self.doc_postings(t) for t, _ in terms]
        if not all(posting_lists):
            return {}
        candidates = set.intersection(*(set(p) for p in posting_lists))
        out = {}
        for d in candidates:
            pos_sets = [set(p[d]) for p in posting_lists]
            starts = [s for s in posting_lists[0][d]
                      if all(s + off in ps for (_, off), ps in zip(terms[1:], pos_sets[1:]))]
            if starts:
                out[d] = starts
        return out


# ============================================================================
# STAGE 3 — Okapi BM25
# ============================================================================

class BM25Scorer:
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be >= 0 and b must be in [0, 1]")
        self.index, self.k1, self.b = index, k1, b

    def idf(self, term: str) -> float:
        # +1 correction keeps IDF positive for terms with df > N/2 (Lucene's BM25 variant)
        n, df = self.index.num_docs, self.index.df(term)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def tf_component(self, tf: int, doc_id: int) -> float:
        dl, avgdl = self.index.doc_lengths[doc_id], self.index.avg_doc_len
        return tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / avgdl))

    def term_scores(self, term: str) -> dict[int, float]:
        idf = self.idf(term)
        return {d: idf * self.tf_component(len(p), d) for d, p in self.index.doc_postings(term).items()}

    def phrase_scores(self, terms: list[tuple[str, int]]) -> dict[int, float]:
        # A phrase is scored as one pseudo-term: f = phrase occurrences, IDF = sum of term IDFs
        idf = sum(self.idf(t) for t, _ in terms)
        return {d: idf * self.tf_component(len(s), d) for d, s in self.index.phrase_matches(terms).items()}


# ============================================================================
# STAGE 4 — Boolean / phrase query parser
# ============================================================================

class QuerySyntaxError(ValueError):
    pass


@dataclass
class TermNode:
    term: str

    def evaluate(self, s: BM25Scorer) -> dict[int, float]:
        return s.term_scores(self.term)

    def highlight(self) -> set[str]:
        return {self.term}


@dataclass
class PhraseNode:
    terms: list[tuple[str, int]]

    def evaluate(self, s: BM25Scorer) -> dict[int, float]:
        return s.phrase_scores(self.terms)

    def highlight(self) -> set[str]:
        return {t for t, _ in self.terms}


@dataclass
class NotNode:
    child: object

    def evaluate(self, s: BM25Scorer) -> dict[int, float]:
        excluded = self.child.evaluate(s)
        return {d: 0.0 for d in s.index.docs if d not in excluded}

    def highlight(self) -> set[str]:
        return set()


@dataclass
class AndNode:
    children: list = field(default_factory=list)

    def evaluate(self, s: BM25Scorer) -> dict[int, float]:
        # subtract NOT children directly instead of building their complement
        pos = [c.evaluate(s) for c in self.children if not isinstance(c, NotNode)]
        neg = [c.child.evaluate(s) for c in self.children if isinstance(c, NotNode)]
        result = dict(pos[0]) if pos else dict.fromkeys(s.index.docs, 0.0)
        for r in pos[1:]:
            result = {d: sc + r[d] for d, sc in result.items() if d in r}
        for r in neg:
            for d in r:
                result.pop(d, None)
        return result

    def highlight(self) -> set[str]:
        return set().union(*(c.highlight() for c in self.children))


@dataclass
class OrNode:
    children: list = field(default_factory=list)

    def evaluate(self, s: BM25Scorer) -> dict[int, float]:
        result: dict[int, float] = defaultdict(float)
        for c in self.children:
            for d, sc in c.evaluate(s).items():
                result[d] += sc  # documents satisfying several branches rank higher
        return dict(result)

    def highlight(self) -> set[str]:
        return set().union(*(c.highlight() for c in self.children))


_QUERY_TOKEN_RE = re.compile(r'"([^"]*)"|([()])|([^\s()"]+)')
_OPERATORS = {"AND", "OR", "NOT"}


class QueryParser:
    """Grammar (precedence NOT > AND > OR; adjacent terms mean implicit AND):
        or_expr  := and_expr ("OR" and_expr)*
        and_expr := not_expr (["AND"] not_expr)*
        not_expr := "NOT" not_expr | atom
        atom     := '"' phrase '"' | word | "(" or_expr ")"
    Operators must be uppercase; lowercase "and/or/not" are ordinary words.
    """

    def __init__(self, pipeline: TextPipeline) -> None:
        self.pipeline = pipeline

    def parse(self, query: str):
        if query.count('"') % 2:
            raise QuerySyntaxError("Unclosed quotation mark")
        self._toks = self._lex(query)
        self._i = 0
        node = self._or()
        if self._i < len(self._toks):
            raise QuerySyntaxError(f"Unexpected token: {self._toks[self._i][1]!r}")
        return node

    @staticmethod
    def _lex(query: str) -> list[tuple[str, str]]:
        toks = []
        for m in _QUERY_TOKEN_RE.finditer(query):
            phrase, paren, word = m.groups()
            if phrase is not None:
                toks.append(("PHRASE", phrase))
            elif paren:
                toks.append((paren, paren))
            elif word in _OPERATORS:
                toks.append((word, word))
            else:
                toks.append(("WORD", word))
        return toks

    def _peek(self) -> str | None:
        return self._toks[self._i][0] if self._i < len(self._toks) else None

    def _or(self):
        children = [self._and()]
        while self._peek() == "OR":
            self._i += 1
            children.append(self._and())
        return self._combine(OrNode, children)

    def _and(self):
        children = [self._not()]
        while self._peek() not in (None, "OR", ")"):
            if self._peek() == "AND":
                self._i += 1
            children.append(self._not())
        return self._combine(AndNode, children)

    def _not(self):
        if self._peek() == "NOT":
            self._i += 1
            child = self._not()
            return NotNode(child) if child else None
        return self._atom()

    def _atom(self):
        kind = self._peek()
        if kind is None or kind in (")", "AND", "OR"):
            raise QuerySyntaxError("Missing term (after an operator or inside empty parentheses)")
        value = self._toks[self._i][1]
        self._i += 1
        if kind == "(":
            node = self._or()
            if self._peek() != ")":
                raise QuerySyntaxError("Unclosed parenthesis")
            self._i += 1
            return node
        # a WORD may produce several tokens (e.g. "machine-learning") -> treat it as a phrase
        tokens = self.pipeline.analyze(value)
        if not tokens:
            return None  # stop words / digits only: dropped from the query
        if len(tokens) == 1:
            return TermNode(tokens[0].term)
        first = tokens[0].position
        return PhraseNode([(t.term, t.position - first) for t in tokens])

    @staticmethod
    def _combine(cls, children):
        children = [c for c in children if c is not None]
        if not children:
            return None
        return children[0] if len(children) == 1 else cls(children)


# ============================================================================
# Engine facade + snippets
# ============================================================================

@dataclass(frozen=True, slots=True)
class SearchResult:
    rank: int
    doc_id: int
    score: float
    title: str
    snippet: str


class SearchEngine:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.index = InvertedIndex()
        self.scorer = BM25Scorer(self.index, k1, b)
        self.parser = QueryParser(self.index.pipeline)

    def add_document(self, text: str, title: str = "", **metadata) -> int:
        return self.index.add_document(text, title, **metadata)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        ast = self.parser.parse(query)
        if ast is None:
            return []
        scores = ast.evaluate(self.scorer)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        terms = ast.highlight()
        return [SearchResult(i, d, sc, self.index.docs[d].title, self.snippet(d, terms))
                for i, (d, sc) in enumerate(ranked, 1)]

    def snippet(self, doc_id: int, terms: set[str], window: int = 7) -> str:
        doc = self.index.docs[doc_id]
        if not doc.spans:
            return ""  # an empty document can still be returned by a pure NOT query
        hits = {p for t in terms for p in self.index.doc_postings(t).get(doc_id, ())}
        center = min(hits) if hits else 0
        lo, hi = max(0, center - window), min(len(doc.spans) - 1, center + window)
        parts, cursor = [], doc.spans[lo][0]
        for p in range(lo, hi + 1):
            s, e = doc.spans[p]
            word = doc.text[s:e]
            parts += [doc.text[cursor:s], f"[{word}]" if p in hits else word]
            cursor = e
        text = re.sub(r"\s+", " ", "".join(parts))
        return ("…" if lo > 0 else "") + text + ("…" if hi < len(doc.spans) - 1 else "")


# ============================================================================
# STAGE 5 — CLI
# ============================================================================

def build_demo_engine(k1: float = 1.5, b: float = 0.75) -> SearchEngine:
    from dataset import DOCUMENTS
    engine = SearchEngine(k1, b)
    for d in DOCUMENTS:
        engine.add_document(d["text"], d["title"], tags=d["tags"])
    return engine


def print_results(query: str, results: list[SearchResult]) -> None:
    print(f"\nQuery: {query}")
    if not results:
        print("  (no results)")
        return
    print(f"  {'Rank':<5}{'DocID':<7}{'Score':>8}  Title")
    print("  " + "-" * 70)
    for r in results:
        print(f"  {r.rank:<5}{r.doc_id:<7}{r.score:>8.4f}  {r.title}")
        print(f"  {'':<20}↳ {r.snippet}")


DEMO_QUERIES = [
    '"machine learning" AND python NOT java',
    "(docker OR kubernetes) AND deployment",
    '"relational database" OR (database AND java) NOT nosql',
    '"deep learning" (pytorch OR tensorflow) OR "natural language"',
]


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # non-ASCII output on Windows consoles
    ap = argparse.ArgumentParser(description="SearchCraft — BM25 + Boolean search shell")
    ap.add_argument("-q", "--query", help="run a single query and exit")
    ap.add_argument("--demo", action="store_true", help="run the 4 demo queries")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args(argv)

    engine = build_demo_engine(args.k1, args.b)
    idx = engine.index
    print(f"SearchCraft: {idx.num_docs} documents, {len(idx.postings)} terms, "
          f"avgdl={idx.avg_doc_len:.2f}, k1={args.k1}, b={args.b}")

    def run(q: str) -> None:
        try:
            print_results(q, engine.search(q, args.top))
        except QuerySyntaxError as e:
            print(f"  Query error: {e}")

    if args.demo or args.query:
        for q in [args.query] if args.query else DEMO_QUERIES:
            run(q)
        return 0

    print('Type a query (e.g. "machine learning" AND python NOT java). Quit with :q')
    while True:
        try:
            q = input("\nsearchcraft> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if q in (":q", ":quit", "exit"):
            return 0
        if q:
            run(q)


if __name__ == "__main__":
    raise SystemExit(main())
