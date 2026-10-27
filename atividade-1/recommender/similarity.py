"""Estatística sobre avaliações observadas; ausência nunca equivale a zero."""

from collections.abc import Mapping
from math import sqrt

UserRatings = Mapping[str, float | None]
Ratings = Mapping[str, UserRatings]


def average(user_ratings: UserRatings) -> float:
    """Média das notas disponíveis. Um usuário sem notas não tem média."""
    values = [value for value in user_ratings.values() if value is not None]
    if not values:
        raise ValueError("O usuário não possui avaliações para calcular a média.")
    return sum(values) / len(values)


def pearson(user_a: UserRatings, user_b: UserRatings, min_common: int = 2) -> float:
    """Pearson com médias calculadas apenas nos itens avaliados em comum.

    Retorna zero se faltarem pares ou se uma das séries for constante.
    Chaves ausentes e valores None representam itens não avaliados.
    """
    if min_common < 2:
        raise ValueError("min_common deve ser pelo menos 2.")
    pairs = [
        (value, user_b[item])
        for item, value in user_a.items()
        if value is not None and user_b.get(item) is not None
    ]
    if len(pairs) < min_common:
        return 0.0
    mean_a = sum(a for a, _ in pairs) / len(pairs)
    mean_b = sum(b for _, b in pairs) / len(pairs)
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in pairs)
    denominator = sqrt(
        sum((a - mean_a) ** 2 for a, _ in pairs)
        * sum((b - mean_b) ** 2 for _, b in pairs)
    )
    return max(-1.0, min(1.0, numerator / denominator)) if denominator else 0.0
