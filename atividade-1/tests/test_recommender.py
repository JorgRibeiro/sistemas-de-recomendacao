"""Regressões de dados esparsos, seleção de vizinhos e avaliação honesta."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from recommender.cli import main
from recommender.data import load_builtin, load_csv, load_movielens
from recommender.evaluation import evaluate, train_test_split
from recommender.model import NoNeighborsError, UserBasedRecommender
from recommender.similarity import average, pearson


class SimilarityTests(unittest.TestCase):
    def test_missing_keys_and_none_use_only_common_observations(self):
        a = {"a": 1, "b": 3, "c": None, "only_a": 5}
        b = {"a": 2, "b": 4, "c": 5, "only_b": 1}
        self.assertAlmostEqual(pearson(a, b), 1)
        self.assertAlmostEqual(pearson(b, a), 1)
        self.assertEqual(pearson(a, b, min_common=3), 0)

    def test_negative_correlation_and_disjoint_items(self):
        self.assertAlmostEqual(pearson({"a": 1, "b": 5}, {"a": 5, "b": 1}), -1)
        self.assertEqual(pearson({"a": 1}, {"b": 2}), 0)

    def test_empty_average_and_invalid_minimum(self):
        for row in ({}, {"a": None}):
            with self.assertRaises(ValueError):
                average(row)
        with self.assertRaises(ValueError):
            pearson({}, {}, min_common=1)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.dataset = load_builtin("book")

    def test_automatic_neighbors_reproduce_book_without_mutation(self):
        model = UserBasedRecommender(self.dataset.ratings, k=2)
        prediction = model.predict("Alice", "item5")
        self.assertAlmostEqual(prediction.score, 4.87, places=2)
        self.assertEqual([n.user for n in prediction.neighbors], ["User1", "User2"])
        self.assertIsNone(self.dataset.ratings["Alice"]["item5"])

    def test_k_is_applied_after_filtering_neighbors_who_rated_item(self):
        ratings = {
            "target": {"a": 1, "b": 3},
            "best_without_item": {"a": 1, "b": 3},
            "eligible": {"a": 2, "b": 5, "c": 4},
        }
        prediction = UserBasedRecommender(ratings, k=1).predict("target", "c")
        self.assertEqual([n.user for n in prediction.neighbors], ["eligible"])
        self.assertAlmostEqual(prediction.score, 2 + 4 - 11 / 3)

    def test_clips_extreme_prediction_but_exposes_raw_score(self):
        ratings = {"a": {"x": 4, "y": 5}, "b": {"x": 1, "y": 2, "z": 5}}
        clipped = UserBasedRecommender(ratings).predict("a", "z")
        raw = UserBasedRecommender(ratings, clip=False).predict("a", "z")
        self.assertEqual(clipped.score, 5)
        self.assertGreater(clipped.raw_score, 5)
        self.assertEqual(raw.score, clipped.raw_score)

    def test_no_neighbors_and_unknown_entities_have_explicit_errors(self):
        model = UserBasedRecommender(self.dataset.ratings, min_common=5)
        with self.assertRaises(NoNeighborsError):
            model.predict("Alice", "item5")
        with self.assertRaisesRegex(ValueError, "Usuário desconhecido"):
            model.predict("nobody", "item5")
        with self.assertRaisesRegex(ValueError, "Item desconhecido"):
            model.predict("Alice", "unknown")
        with self.assertRaisesRegex(ValueError, "não possui avaliações"):
            UserBasedRecommender({"empty": {}}).recommend("empty")

    def test_input_snapshot_is_not_affected_by_later_mutation(self):
        model = UserBasedRecommender(self.dataset.ratings)
        before = model.predict("Alice", "item5")
        self.dataset.ratings["User1"]["item5"] = 1
        self.assertEqual(model.predict("Alice", "item5"), before)

    def test_recommendations_are_sorted_unseen_and_repeatable(self):
        dataset = load_builtin("movies")
        model = UserBasedRecommender(dataset.ratings)
        results = model.recommend("Ana", 5)
        self.assertEqual(results, model.recommend("Ana", 5))
        self.assertEqual(len(results), 5)
        self.assertEqual(
            [p.score for p in results], sorted([p.score for p in results], reverse=True)
        )
        self.assertTrue(
            all(dataset.ratings["Ana"].get(p.item) is None for p in results)
        )
        self.assertEqual(UserBasedRecommender({"a": {"x": 1}}).recommend("a"), [])

    def test_invalid_parameters_and_ratings(self):
        for kwargs in ({"k": 0}, {"min_common": 1}):
            with self.assertRaises(ValueError):
                UserBasedRecommender(self.dataset.ratings, **kwargs)
        for value in (0, 6, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                UserBasedRecommender({"a": {"x": value}})
        with self.assertRaises(ValueError):
            UserBasedRecommender(self.dataset.ratings).recommend("Alice", 0)


class DataTests(unittest.TestCase):
    def load_text(self, text):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.csv"
            path.write_text(text, encoding="utf-8")
            return load_csv(path)

    def test_csv_missing_values_titles_and_decimal_ratings(self):
        dataset = self.load_text(
            'user_id,item_id,rating,title\nAna,x,4.5,"Filme, parte 2"\nBia,x,,\n'
        )
        self.assertEqual(dataset.ratings, {"Ana": {"x": 4.5}, "Bia": {"x": None}})
        self.assertEqual(dataset.titles["x"], "Filme, parte 2")
        self.assertEqual(dataset.count, 1)

    def test_csv_reports_invalid_input(self):
        invalid = [
            "wrong,columns\na,b\n",
            "user_id,item_id,rating\n",
            "user_id,item_id,rating\na,x,6\n",
            "user_id,item_id,rating\na,x,NaN\n",
            "user_id,item_id,rating\na,x,abc\n",
            "user_id,item_id,rating\n,x,3\n",
            "user_id,item_id,rating\na,x,3\na,x,4\n",
            "user_id,item_id,rating\na,x\n",
            "user_id,item_id,rating\na,x,4,extra\n",
            "user_id,item_id,rating,title\na,x,4,First\nb,x,3,Second\n",
        ]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load_text(text)

    def test_movielens_format_and_latin1_title(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "u.data").write_text("1\t2\t4\t874965758\n", encoding="ascii")
            (path / "u.item").write_text(
                "2|Comédia (1997)|01-Jan-1997||url|0\n", encoding="latin-1"
            )
            dataset = load_movielens(path)
            self.assertEqual(dataset.ratings, {"1": {"2": 4}})
            self.assertEqual(dataset.titles["2"], "Comédia (1997)")

    def test_builtins_have_expected_size(self):
        book, movies = load_builtin("book"), load_builtin("movies")
        self.assertEqual((len(book.ratings), len(book.items), book.count), (5, 5, 24))
        self.assertEqual(
            (len(movies.ratings), len(movies.items), movies.count), (12, 12, 112)
        )


class EvaluationTests(unittest.TestCase):
    def test_split_is_reproducible_disjoint_and_keeps_training_history(self):
        ratings = load_builtin("movies").ratings
        train, test = train_test_split(ratings, seed=42)
        self.assertEqual((train, test), train_test_split(ratings, seed=42))
        self.assertNotEqual(test, train_test_split(ratings, seed=7)[1])
        self.assertEqual(sum(len(row) for row in train.values()) + len(test), 112)
        self.assertTrue(all(train.values()))
        for user, item, value in test:
            self.assertNotIn(item, train[user])
            self.assertEqual(ratings[user][item], value)

    def test_split_handles_empty_and_single_rating_users(self):
        train, test = train_test_split({"a": {"x": 5}, "b": {"y": None}})
        self.assertEqual(train, {"a": {"x": 5}, "b": {}})
        self.assertEqual(test, [])
        for fraction in (0, 1, -0.2):
            with self.assertRaises(ValueError):
                train_test_split({}, fraction)

    def test_metrics_and_baseline_have_known_manual_values(self):
        train = {"a": {"x": 1, "y": 3}, "b": {"x": 1, "y": 3, "z": 5}}
        result = evaluate(UserBasedRecommender(train), [("a", "z", 3)], train)
        # previsão = 2 + (5 - 3) = 4; baseline = 2
        self.assertEqual(
            (result.attempted, result.predicted, result.coverage), (1, 1, 1)
        )
        self.assertAlmostEqual(result.mae, 1)
        self.assertAlmostEqual(result.rmse, 1)
        self.assertAlmostEqual(result.baseline_mae, 1)
        self.assertAlmostEqual(result.baseline_rmse, 1)

    def test_unpredictable_items_reduce_coverage_without_zero_error(self):
        train = {"a": {"x": 1, "y": 3}, "b": {"x": 1, "y": 3, "z": 5}}
        model = UserBasedRecommender(train)
        result = evaluate(model, [("a", "z", 3), ("a", "unknown", 5)], train)
        self.assertEqual(result.coverage, 0.5)
        self.assertEqual(result.mae, 1)
        result = evaluate(
            UserBasedRecommender(train, min_common=3), [("a", "z", 3)], train
        )
        self.assertEqual(result.predicted, 0)
        self.assertIsNone(result.mae)
        self.assertIsNone(result.baseline_mae)
        self.assertEqual(evaluate(model, [], train).coverage, 0)


class CliTests(unittest.TestCase):
    def call(self, args):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            status = main(args)
        return status, out.getvalue(), err.getvalue()

    def test_default_demo_explains_prediction(self):
        status, out, err = self.call([])
        self.assertEqual((status, err), (0, ""))
        self.assertIn("4.87 / 5", out)
        self.assertIn("User1", out)

    def test_movie_recommendation_and_evaluation(self):
        for args in (
            ["inspect", "--dataset", "movies"],
            ["neighbors", "--user", "Alice"],
            ["predict", "--user", "Alice", "--item", "item5"],
            ["recommend", "--dataset", "movies", "--user", "Ana"],
            ["evaluate", "--dataset", "movies", "--limit", "0"],
        ):
            with self.subTest(args=args):
                status, out, err = self.call(args)
                self.assertEqual((status, err), (0, ""))
                self.assertTrue(out)

    def test_invalid_user_and_missing_file_return_readable_errors(self):
        for args in (
            ["recommend", "--user", "unknown"],
            ["inspect", "--csv", "/nonexistent/ratings.csv"],
            ["evaluate", "--limit", "-1"],
        ):
            with self.subTest(args=args):
                status, _, err = self.call(args)
                self.assertEqual(status, 2)
                self.assertIn("Erro:", err)
