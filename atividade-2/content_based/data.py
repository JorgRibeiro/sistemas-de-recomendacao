"""Carregamento dos dados originais, sem tokenização ou ponderação TF-IDF."""

import csv
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class TermMatrix:
    document_ids: tuple[str, ...]
    terms: tuple[str, ...]
    # Linhas = termos; colunas = documentos, na ordem dos IDs acima.
    values: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    category: str  # Metadado da notícia; não entra na vetorização.


def load_shakespeare() -> TermMatrix:
    with (DATA_DIR / "shakespeare.csv").open(encoding="utf-8", newline="") as file:
        rows = csv.reader(file)
        document_ids = tuple(next(rows)[1:])
        records = list(rows)
    return TermMatrix(
        document_ids,
        tuple(row[0] for row in records),
        tuple(tuple(float(value) for value in row[1:]) for row in records),
    )


def load_bbc(path: Path | None = None) -> list[Document]:
    path = Path(path) if path is not None else DATA_DIR / "bbc-fulltext.zip"
    if not path.is_file():
        raise FileNotFoundError(
            "BBC não encontrada. Execute: python scripts/download_bbc.py"
        )
    documents = []
    with ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            parts = name.split("/")
            if len(parts) != 3 or not name.endswith(".txt"):
                continue
            text = archive.read(name).decode("latin-1").strip()
            documents.append(Document(
                id=f"{parts[1]}/{Path(parts[2]).stem}",
                title=text.splitlines()[0],
                text=text,
                category=parts[1],
            ))
    if len(documents) != 2225:
        raise ValueError(f"Esperadas 2225 notícias; encontradas {len(documents)}.")
    return documents
