"""Vizinhos mais próximos e previsão centrada na média (capítulo 2)."""

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
from math import isfinite

from .similarity import Ratings, average, pearson


class NoNeighborsError(ValueError):
    """Não há evidência suficiente para uma previsão colaborativa."""


@dataclass(frozen=True)
class Neighbor:
    user: str
    similarity: float


@dataclass(frozen=True)
class Prediction:
    item: str
    score: float
    raw_score: float
    user_mean: float
    neighbors: tuple[Neighbor, ...]


def predict(
    user_name: str,
    item: str,
    ratings: Ratings,
    similarities: Mapping[str, float],
    neighbors: Iterable[str],
) -> float:
    """API original: previsão sem limitar a nota à escala, com vizinhos explícitos."""
    user_mean = average(ratings[user_name])
    selected = [
        (other, similarities[other])
        for other in dict.fromkeys(neighbors)
        if other != user_name
        and ratings[other].get(item) is not None
        and similarities[other] > 0
    ]
    if not selected:
        raise NoNeighborsError(
            f"Nenhum vizinho com similaridade positiva avaliou {item}."
        )
    deviations = sum(
        sim * (ratings[other][item] - average(ratings[other]))
        for other, sim in selected
    )
    return user_mean + deviations / sum(sim for _, sim in selected)


class UserBasedRecommender:
    """Modelo para notas de 1 a 5; guarda uma cópia das avaliações observadas.

    Seleciona até k vizinhos positivos por item, após filtrar quem o avaliou.
    A previsão usa médias de todas as notas observadas de cada usuário.
    """

    def __init__(
        self, ratings: Ratings, k: int = 20, min_common: int = 2, clip: bool = True
    ) -> None:
        if k < 1 or min_common < 2:
            raise ValueError("k deve ser positivo e min_common deve ser pelo menos 2.")
        self.k, self.min_common, self.clip = k, min_common, clip
        self._ratings: dict[str, dict[str, float]] = {}
        for user, row in ratings.items():
            observed = {item: value for item, value in row.items() if value is not None}
            if any(
                not isfinite(value) or not 1 <= value <= 5
                for value in observed.values()
            ):
                raise ValueError("As notas devem ser finitas e estar entre 1 e 5.")
            self._ratings[user] = observed
        self._means = {user: average(row) for user, row in self._ratings.items() if row}
        self._items = {item for row in self._ratings.values() for item in row}
        self._similarities: dict[str, tuple[Neighbor, ...]] = {}

    def similarities(self, user: str) -> tuple[Neighbor, ...]:
        if user not in self._ratings:
            raise ValueError(f"Usuário desconhecido: {user}.")
        if user not in self._means:
            raise ValueError(f"O usuário {user} ainda não possui avaliações.")
        if user not in self._similarities:
            neighbors = [
                Neighbor(other, pearson(self._ratings[user], row, self.min_common))
                for other, row in self._ratings.items()
                if other != user
            ]
            self._similarities[user] = tuple(
                sorted(neighbors, key=lambda n: (-n.similarity, n.user))
            )
        return self._similarities[user]

    def predict(self, user: str, item: str) -> Prediction:
        similarities = self.similarities(user)
        if item not in self._items:
            raise ValueError(f"Item desconhecido ou sem avaliações: {item}.")
        neighbors = tuple(
            n
            for n in similarities
            if n.similarity > 0 and item in self._ratings[n.user]
        )[: self.k]
        if not neighbors:
            raise NoNeighborsError(
                f"Nenhum vizinho com similaridade positiva avaliou {item}."
            )
        mean = self._means[user]
        deviation = sum(
            n.similarity * (self._ratings[n.user][item] - self._means[n.user])
            for n in neighbors
        )
        raw = mean + deviation / sum(n.similarity for n in neighbors)
        score = max(1.0, min(5.0, raw)) if self.clip else raw
        return Prediction(item, score, raw, mean, neighbors)

    def recommend(self, user: str, n: int = 5) -> list[Prediction]:
        if n < 1:
            raise ValueError("n deve ser positivo.")
        self.similarities(user)
        predictions = []
        for item in sorted(self._items - self._ratings[user].keys()):
            try:
                predictions.append(self.predict(user, item))
            except NoNeighborsError:
                continue
        return sorted(predictions, key=lambda p: (-p.score, p.item))[:n]
