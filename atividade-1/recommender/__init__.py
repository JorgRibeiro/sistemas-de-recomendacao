"""Filtragem colaborativa baseada em usuários, sem dependências externas."""

from .model import NoNeighborsError, UserBasedRecommender
from .similarity import average, pearson

__all__ = ["NoNeighborsError", "UserBasedRecommender", "average", "pearson"]
