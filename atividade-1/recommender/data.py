"""Carregadores de CSV e do formato oficial MovieLens 100K."""

import csv
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Dataset:
    name: str
    ratings: dict[str, dict[str, float | None]]
    titles: dict[str, str] = field(default_factory=dict)

    @property
    def items(self) -> set[str]:
        return {item for row in self.ratings.values() for item in row}

    @property
    def count(self) -> int:
        return sum(
            value is not None for row in self.ratings.values() for value in row.values()
        )


def _add_rating(ratings, user, item, raw, location):
    if not user or not item:
        raise ValueError(f"{location}: user_id e item_id não podem ficar vazios.")
    try:
        value = float(raw) if raw.strip() else None
    except ValueError as exc:
        raise ValueError(f"{location}: nota inválida: {raw!r}.") from exc
    if value is not None and (not isfinite(value) or not 1 <= value <= 5):
        raise ValueError(f"{location}: a nota deve estar entre 1 e 5.")
    row = ratings.setdefault(user, {})
    if item in row:
        raise ValueError(f"{location}: avaliação duplicada para {user}/{item}.")
    row[item] = value


def load_csv(path: str | Path) -> Dataset:
    """CSV UTF-8: user_id,item_id,rating e coluna opcional title."""
    path = Path(path)
    ratings, titles = {}, {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"user_id", "item_id", "rating"}.issubset(reader.fieldnames or []):
            raise ValueError("O CSV precisa das colunas user_id,item_id,rating.")
        for line, row in enumerate(reader, 2):
            if None in row or any(row.get(key) is None for key in reader.fieldnames):
                raise ValueError(f"Linha {line}: quantidade incorreta de colunas.")
            user, item = row["user_id"].strip(), row["item_id"].strip()
            _add_rating(ratings, user, item, row["rating"], f"Linha {line}")
            title = row.get("title", "").strip()
            if title:
                if item in titles and titles[item] != title:
                    raise ValueError(f"Linha {line}: títulos diferentes para {item}.")
                titles[item] = title
    if not ratings:
        raise ValueError("A base de avaliações está vazia.")
    return Dataset(path.stem, ratings, titles)


def load_builtin(name: str) -> Dataset:
    if name not in {"book", "movies"}:
        raise ValueError(f"Base desconhecida: {name}.")
    return load_csv(DATA_DIR / f"{name}.csv")


def load_movielens(directory: str | Path) -> Dataset:
    """Lê u.data (tabulação) e u.item (Latin-1), preservando IDs oficiais."""
    directory = Path(directory)
    ratings, titles = {}, {}
    with (directory / "u.data").open(encoding="ascii") as stream:
        for line, text in enumerate(stream, 1):
            parts = text.rstrip("\n").split("\t")
            if len(parts) != 4:
                raise ValueError(f"u.data, linha {line}: esperadas quatro colunas.")
            user, item, rating, _ = parts
            _add_rating(ratings, user, item, rating, f"u.data, linha {line}")
    with (directory / "u.item").open(encoding="latin-1") as stream:
        for line, text in enumerate(stream, 1):
            parts = text.rstrip("\n").split("|")
            if len(parts) < 2:
                raise ValueError(f"u.item, linha {line}: registro inválido.")
            titles[parts[0]] = parts[1]
    if not ratings:
        raise ValueError("A base MovieLens está vazia.")
    return Dataset("MovieLens 100K", ratings, titles)
