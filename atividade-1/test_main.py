"""Testes do exercício de recomendação colaborativa baseado no livro.

Execute na pasta atividade-1 com:
    python -m unittest discover -v

O teste principal valida a sequência completa: base de avaliações, média,
similaridade de Pearson, escolha dos vizinhos e previsão de item5 para Alice.
Os cenários adicionais do projeto modular estão em tests/test_recommender.py.
"""

import unittest

from main import average, pearson, predict, ratings


class CollaborativeFilteringExerciseTests(unittest.TestCase):
    def test_rating_matrix_matches_book_example(self):
        """Confere as avaliações copiadas do exemplo do livro."""
        self.assertEqual(
            ratings["Alice"],
            {"item1": 5, "item2": 3, "item3": 4, "item4": 4, "item5": None},
        )
        self.assertEqual(
            ratings["User1"],
            {"item1": 3, "item2": 1, "item3": 2, "item4": 3, "item5": 3},
        )
        self.assertEqual(
            ratings["User2"],
            {"item1": 4, "item2": 3, "item3": 4, "item4": 3, "item5": 5},
        )
        self.assertEqual(
            ratings["User3"],
            {"item1": 3, "item2": 3, "item3": 1, "item4": 5, "item5": 4},
        )
        self.assertEqual(
            ratings["User4"],
            {"item1": 1, "item2": 5, "item3": 5, "item4": 2, "item5": 1},
        )

    def test_user_averages_ignore_unrated_items(self):
        """A média de Alice usa somente item1 a item4; item5 não foi avaliado."""
        self.assertAlmostEqual(average(ratings["Alice"]), 4.0)
        self.assertAlmostEqual(average(ratings["User1"]), 2.4)
        self.assertAlmostEqual(average(ratings["User2"]), 3.8)

    def test_pearson_similarities_match_book_values(self):
        """Valida correlações conhecidas do exemplo (antes do arredondamento)."""
        alice = ratings["Alice"]
        self.assertAlmostEqual(pearson(alice, ratings["User1"]), 0.8528, places=3)
        self.assertAlmostEqual(pearson(alice, ratings["User2"]), 0.7071, places=3)
        self.assertAlmostEqual(pearson(alice, ratings["User3"]), 0.0, places=7)
        self.assertAlmostEqual(pearson(alice, ratings["User4"]), -0.7921, places=3)

    def test_prediction_for_alice_item5_matches_book(self):
        """Usa User1 e User2, vizinhos selecionados no livro, para prever item5."""
        alice = ratings["Alice"]
        similarities = {
            name: pearson(alice, user_ratings)
            for name, user_ratings in ratings.items()
            if name != "Alice"
        }

        predicted_rating = predict(
            user_name="Alice",
            item="item5",
            ratings=ratings,
            similarities=similarities,
            neighbors=["User1", "User2"],
        )

        # Fórmula 2.3: média(Alice) + soma(sim * desvio do vizinho) / soma(sim).
        self.assertAlmostEqual(predicted_rating, 4.87, places=2)
        self.assertIsNone(ratings["Alice"]["item5"])  # não altera a nota original

    def test_prediction_ignores_nonpositive_neighbors(self):
        """Correlação zero/negativa não deve contribuir para a previsão."""
        result = predict(
            "Alice",
            "item5",
            ratings,
            {"User1": 0.85, "User2": 0.70, "User3": 0.0, "User4": -0.79},
            ["User1", "User2", "User3", "User4"],
        )
        expected = 4 + (0.85 * (3 - 2.4) + 0.70 * (5 - 3.8)) / (0.85 + 0.70)
        self.assertAlmostEqual(result, expected)

    def test_prediction_fails_when_no_positive_neighbor_can_contribute(self):
        """Sem vizinho positivo que avaliou o item, não há previsão confiável."""
        with self.assertRaises(ValueError):
            predict(
                "Alice",
                "item5",
                ratings,
                {"User3": 0.0, "User4": -0.79},
                ["User3", "User4"],
            )

    def test_pearson_handles_insufficient_or_constant_data(self):
        """Casos em que a correlação é indefinida retornam zero com segurança."""
        self.assertEqual(pearson({"item1": 5}, {"item1": 3}), 0.0)
        self.assertEqual(
            pearson({"item1": 4, "item2": 4}, {"item1": 1, "item2": 2}),
            0.0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
