"""SearchCraft validation tests: python -m unittest -v"""
import math
import unittest

from searchcraft import (DEMO_QUERIES, BM25Scorer, InvertedIndex, QuerySyntaxError, SearchEngine,
                         TextPipeline, build_demo_engine, syllabify, syllable_ngrams)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.p = TextPipeline()

    def test_normalize_stopwords_digits(self):
        self.assertEqual(self.p.terms("Python3 and the 2024 LONDON!"), ["python", "london"])

    def test_stemming_is_consistent_across_inflections(self):
        self.assertEqual({self.p.terms(w)[0] for w in ("learn", "learning", "learned", "learns")}, {"learn"})
        self.assertEqual(self.p.terms("deployments"), self.p.terms("deployment"))

    def test_positions_keep_stopword_gaps(self):
        self.assertEqual([t.position for t in self.p.analyze("data and model")], [0, 2])

    def test_syllables(self):
        self.assertEqual(syllabify("computer"), ["com", "pu", "ter"])
        self.assertEqual(syllable_ngrams("computer"), ["com-pu", "pu-ter"])


class TestIndexAndBM25(unittest.TestCase):
    def setUp(self):
        self.idx = InvertedIndex()
        self.idx.add_document("apple pear apple")  # 1: len 3
        self.idx.add_document("pear plum")         # 2: len 2
        self.idx.add_document("fig")               # 3: len 1
        self.bm25 = BM25Scorer(self.idx)

    def test_stats(self):
        self.assertEqual(self.idx.postings["apple"], {1: [0, 2]})
        self.assertEqual(self.idx.df("pear"), 2)
        self.assertAlmostEqual(self.idx.avg_doc_len, 2.0)

    def test_bm25_matches_hand_computation(self):
        idf = math.log((3 - 1 + 0.5) / (1 + 0.5) + 1)
        tf, k1, b, dl, avgdl = 2, 1.5, 0.75, 3, 2.0
        expected = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
        self.assertAlmostEqual(self.bm25.term_scores("apple")[1], expected)

    def test_idf_never_negative(self):
        idx = InvertedIndex()
        for _ in range(5):
            idx.add_document("common")
        self.assertGreater(BM25Scorer(idx).idf("common"), 0)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            BM25Scorer(self.idx, b=1.5)


class TestQueries(unittest.TestCase):
    def setUp(self):
        self.e = SearchEngine()
        self.e.add_document("machine learning done with python")      # 1
        self.e.add_document("learning is hard on a machine, python")  # 2: phrase not adjacent
        self.e.add_document("java for machine learning")              # 3

    def ids(self, q):
        return [r.doc_id for r in self.e.search(q)]

    def test_phrase_requires_adjacency(self):
        self.assertCountEqual(self.ids('"machine learning"'), [1, 3])

    def test_phrase_matches_across_inflections(self):
        self.assertCountEqual(self.ids('"machine learned"'), [1, 3])

    def test_not_and_implicit_and(self):
        self.assertCountEqual(self.ids("machine NOT java"), [1, 2])
        self.assertCountEqual(self.ids("NOT java"), [1, 2])
        self.assertCountEqual(self.ids("python machine"), [1, 2])

    def test_and_binds_tighter_than_or(self):
        self.assertCountEqual(self.ids("java OR python AND hard"), [2, 3])

    def test_lowercase_operators_are_words(self):
        self.assertEqual(self.ids("java or"), [3])  # "or" is a stop word and gets dropped

    def test_syntax_errors(self):
        for q in ('"open quote', "(python", "python AND", "python )", "OR java"):
            with self.subTest(q=q), self.assertRaises(QuerySyntaxError):
                self.e.search(q)

    def test_stopword_only_query(self):
        self.assertEqual(self.e.search("and the with"), [])


class TestDemoRanking(unittest.TestCase):
    """Ranking correctness of the 4 complex demo queries on the 10-document dataset."""

    @classmethod
    def setUpClass(cls):
        cls.e = build_demo_engine()

    def ranked(self, q):
        res = self.e.search(q, top_k=10)
        scores = [r.score for r in res]
        self.assertEqual(scores, sorted(scores, reverse=True))
        return [r.doc_id for r in res]

    def test_q1_phrase_and_not(self):
        # doc 1: phrase 2x + python 3x -> top; doc 2 (java) must be excluded
        self.assertEqual(self.ranked(DEMO_QUERIES[0]), [1, 3, 10])

    def test_q2_grouped_or(self):
        self.assertEqual(self.ranked(DEMO_QUERIES[1]), [8, 7])

    def test_q3_not_excludes_nosql(self):
        self.assertEqual(self.ranked(DEMO_QUERIES[2]), [5, 9])

    def test_q4_precedence_and_phrases(self):
        self.assertEqual(self.ranked(DEMO_QUERIES[3]), [3, 4, 10])

    def test_snippet_highlights_match(self):
        self.assertIn("[python]", self.e.search(DEMO_QUERIES[0])[0].snippet.lower())


if __name__ == "__main__":
    unittest.main()
