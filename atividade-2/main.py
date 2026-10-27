"""Inspeção das bases de dados; não executa TF-IDF."""

import argparse
from collections import Counter
import json
from content_based.data import load_bbc, load_shakespeare


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("shakespeare", "bbc"), default="shakespeare")
    args = parser.parse_args()
    try:
        if args.dataset == "shakespeare":
            matrix = load_shakespeare()
            summary = {"documents": len(matrix.document_ids), "terms": len(matrix.terms)}
        else:
            documents = load_bbc()
            summary = {"documents": len(documents), "categories": dict(Counter(d.category for d in documents))}
        output = {"dataset": args.dataset, **summary, "tfidf": "não implementado"}
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
