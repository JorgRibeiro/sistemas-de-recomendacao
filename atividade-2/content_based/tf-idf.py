"""Cálculo das funcões tf-idf"""

import csv
from math import log10

from data import DATA_DIR, load_shakespeare


def calcular_tfidf():
    base = load_shakespeare()
    total_documentos = len(base.document_ids)

    # Maior frequência de cada documento (coluna).
    maiores_frequencias = [
        max(linha[coluna] for linha in base.values)
        for coluna in range(total_documentos)
    ]

    resultado = {}

    for termo, frequencias in zip(base.terms, base.values):
        documentos_com_termo = sum(
            frequencia > 0 for frequencia in frequencias
        )

        idf = (
            log10(total_documentos / documentos_com_termo)
            if documentos_com_termo > 0
            else 0
        )

        resultado[termo] = {}

        for coluna, documento in enumerate(base.document_ids):
            maior = maiores_frequencias[coluna]
            tf = frequencias[coluna] / maior if maior > 0 else 0

            resultado[termo][documento] = tf * idf

    return resultado


if __name__ == "__main__":
    pesos = calcular_tfidf()
    caminho = DATA_DIR / "shakespeare_tfidf.csv"

    with caminho.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.writer(arquivo)
        documentos = load_shakespeare().document_ids
        escritor.writerow(["term", *documentos])
        for termo, valores in pesos.items():
            escritor.writerow([termo, *(valores[documento] for documento in documentos)])

    print(f"CSV salvo em: {caminho}")

