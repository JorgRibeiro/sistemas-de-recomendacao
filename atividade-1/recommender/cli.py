"""Interface de terminal em português para explorar o recomendador."""

import argparse
import sys

from .data import DATA_DIR, load_builtin, load_csv, load_movielens
from .model import UserBasedRecommender
from .similarity import average


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recomendação colaborativa baseada em usuários."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "demo": "Reproduzir a previsão de Alice no exemplo do livro",
        "inspect": "Mostrar tamanho da base, usuários e itens",
        "neighbors": "Mostrar as similaridades de um usuário",
        "predict": "Prever a nota para um item e mostrar os vizinhos",
        "recommend": "Recomendar itens ainda não avaliados",
    }
    for name, description in descriptions.items():
        command = commands.add_parser(name, help=description, description=description)
        if name == "demo":
            continue
        sources = command.add_mutually_exclusive_group()
        sources.add_argument(
            "--dataset",
            choices=["book", "movies", "movielens"],
            default="book",
            help="Base de dados (padrão: book)",
        )
        sources.add_argument("--csv", help="Caminho de um CSV próprio")
        command.add_argument(
            "--movielens-dir",
            default=str(DATA_DIR / "ml-100k"),
            help="Pasta com u.data e u.item",
        )
        if name in {"neighbors", "predict", "recommend"}:
            command.add_argument(
                "--k", type=int, default=20, help="Máximo de vizinhos por item (20)"
            )
            command.add_argument(
                "--min-common", type=int, default=2, help="Mínimo de itens em comum (2)"
            )
        if name in {"neighbors", "predict", "recommend"}:
            command.add_argument(
                "--user",
                required=True,
                help="ID do usuário; use inspect para consultar",
            )
        if name == "predict":
            command.add_argument("--item", required=True, help="ID do item")
        if name in {"recommend", "neighbors"}:
            command.add_argument(
                "--top", type=int, default=5, help="Quantidade de resultados (5)"
            )
    return parser


def _load(args):
    if args.csv:
        return load_csv(args.csv)
    if args.dataset == "movielens":
        return load_movielens(args.movielens_dir)
    return load_builtin(args.dataset)


def _show_prediction(prediction, dataset):
    title = dataset.titles.get(prediction.item, prediction.item)
    print(f"Previsão para {title} [{prediction.item}]: {prediction.score:.2f} / 5")
    print(
        f"Média do usuário: {prediction.user_mean:.2f}; resultado sem limite: {prediction.raw_score:.4f}"
    )
    print("Vizinhos usados (similaridade; nota; média; contribuição ponderada):")
    for neighbor in prediction.neighbors:
        row = dataset.ratings[neighbor.user]
        mean = average(row)
        contribution = neighbor.similarity * (row[prediction.item] - mean)
        print(
            f"  {neighbor.user}: {neighbor.similarity:.4f}; {row[prediction.item]:.1f}; "
            f"{mean:.2f}; {contribution:+.4f}"
        )


def main(argv=None) -> int:
    parser = build_parser()
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments or ["demo"])
    try:
        if args.command == "demo":
            dataset = load_builtin("book")
            model = UserBasedRecommender(dataset.ratings, k=2)
            print("Similaridade de Pearson com Alice:")
            for neighbor in model.similarities("Alice"):
                print(f"  {neighbor.user}: {neighbor.similarity:.4f}")
            print()
            _show_prediction(model.predict("Alice", "item5"), dataset)
            return 0
        dataset = _load(args)
        if args.command == "inspect":
            capacity = len(dataset.ratings) * len(dataset.items)
            print(f"Base: {dataset.name}")
            print(
                f"Usuários: {len(dataset.ratings)} | Itens: {len(dataset.items)} | Notas: {dataset.count}"
            )
            print(f"Esparsidade: {1 - dataset.count / capacity:.1%}")
            print("Usuários (até 15): " + ", ".join(sorted(dataset.ratings)[:15]))
            print("Itens (até 15):")
            for item in sorted(dataset.items)[:15]:
                print(f"  {item}: {dataset.titles.get(item, item)}")
            return 0
        model = UserBasedRecommender(
            dataset.ratings, k=args.k, min_common=args.min_common
        )
        if args.command == "predict":
            _show_prediction(model.predict(args.user, args.item), dataset)
        elif args.command == "neighbors":
            if args.top < 1:
                raise ValueError("top deve ser positivo.")
            for neighbor in model.similarities(args.user)[: args.top]:
                print(f"{neighbor.user}: {neighbor.similarity:+.4f}")
        elif args.command == "recommend":
            results = model.recommend(args.user, args.top)
            if not results:
                print(
                    "Nenhum item ainda não avaliado pode ser recomendado com os vizinhos disponíveis."
                )
            for index, result in enumerate(results, 1):
                title = dataset.titles.get(result.item, result.item)
                print(
                    f"{index}. {title} [{result.item}]: {result.score:.2f} / 5 "
                    f"({len(result.neighbors)} vizinhos)"
                )
        return 0
    except (ValueError, OSError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 2
