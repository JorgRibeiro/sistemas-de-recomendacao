"""Entrada da CLI e compatibilidade com as funções do exercício original."""

from recommender.data import load_builtin
from recommender.similarity import average, pearson
from recommender.model import predict

__all__ = ["average", "pearson", "predict", "ratings"]

ratings = load_builtin("book").ratings


if __name__ == "__main__":
    from recommender.cli import main

    raise SystemExit(main())
