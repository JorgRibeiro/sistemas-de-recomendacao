"""Holdout por usuário, sem usar as notas de teste na escolha dos vizinhos."""

from dataclasses import dataclass
from collections.abc import Iterable
from math import sqrt
from random import Random

from .model import NoNeighborsError, UserBasedRecommender
from .similarity import Ratings, average

TestRating = tuple[str, str, float]


def train_test_split(
    ratings: Ratings, test_size: float = 0.2, seed: int = 42
) -> tuple[dict[str, dict[str, float]], list[TestRating]]:
    """Retém ao menos uma nota no treino por usuário; ignora ausências."""
    if not 0 < test_size < 1:
        raise ValueError("test_size deve estar entre 0 e 1, sem incluir os extremos.")
    rng = Random(seed)
    train, test = {}, []
    for user in sorted(ratings):
        items = sorted(
            (item, value) for item, value in ratings[user].items() if value is not None
        )
        rng.shuffle(items)
        count = (
            min(len(items) - 1, max(1, int(len(items) * test_size)))
            if len(items) > 1
            else 0
        )
        test.extend((user, item, value) for item, value in items[:count])
        train[user] = dict(items[count:])
    return train, test


@dataclass(frozen=True)
class Evaluation:
    attempted: int
    predicted: int
    mae: float | None
    rmse: float | None
    baseline_mae: float | None
    baseline_rmse: float | None

    @property
    def coverage(self) -> float:
        return self.predicted / self.attempted if self.attempted else 0.0


def evaluate(
    model: UserBasedRecommender, test: Iterable[TestRating], train: Ratings
) -> Evaluation:
    """MAE/RMSE e baseline de média no mesmo subconjunto com previsão.

    Sem vizinhos ou com item ausente do treino, conta como falta de cobertura.
    Não substitui silenciosamente essas falhas pela média do usuário.
    """
    errors, baseline_errors = [], []
    attempted = 0
    known_items = {
        item
        for row in train.values()
        for item, value in row.items()
        if value is not None
    }
    for user, item, actual in test:
        attempted += 1
        if item not in known_items:
            continue
        try:
            result = model.predict(user, item)
        except NoNeighborsError:
            continue
        errors.append(result.score - actual)
        baseline_errors.append(average(train[user]) - actual)

    def metrics(values):
        if not values:
            return None, None
        return sum(abs(e) for e in values) / len(values), sqrt(
            sum(e * e for e in values) / len(values)
        )

    mae, rmse = metrics(errors)
    baseline_mae, baseline_rmse = metrics(baseline_errors)
    return Evaluation(attempted, len(errors), mae, rmse, baseline_mae, baseline_rmse)
