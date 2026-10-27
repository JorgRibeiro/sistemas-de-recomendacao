"""Similaridade de vetores; execução direta para as peças de Shakespeare."""

import csv
from collections.abc import Mapping
from math import sqrt
from pathlib import Path

UserRatings = Mapping[str, float | None]
Ratings = Mapping[str, UserRatings]


def cosine(vector_a: Mapping[str, float], vector_b: Mapping[str, float]) -> float:
    """Cosseno entre vetores de termos; vetor nulo retorna zero."""
    product = sum(value * vector_b.get(term, 0) for term, value in vector_a.items())
    norm_a = sqrt(sum(value ** 2 for value in vector_a.values()))
    norm_b = sqrt(sum(value ** 2 for value in vector_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(-1.0, min(1.0, product / (norm_a * norm_b)))


def load_vectors(path: Path) -> dict[str, dict[str, float]]:
    """Converte o CSV termo × peça em um vetor de termos por peça."""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        documents = reader.fieldnames[1:]
        vectors = {document: {} for document in documents}
        for row in reader:
            for document in documents:
                vectors[document][row["term"]] = float(row[document])
    return vectors


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


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[1] / "data"
    vectors = load_vectors(data_dir / "shakespeare_similarity.csv")
    output = data_dir / "shakespeare_cosine_similarity.csv"

    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["document", *vectors])
        for document, vector in vectors.items():
            scores = [cosine(vector, other) for other in vectors.values()]
            writer.writerow([document, *scores])
            print(f"\n{document}:")
            for other, score in zip(vectors, scores):
                if other != document:
                    print(f"  {other}: {score:.4f}")

    print(f"\nCSV salvo em: {output}")

