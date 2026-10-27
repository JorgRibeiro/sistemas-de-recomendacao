"""Baixa os dados oficiais sem redistribuí-los no repositório."""

import argparse
from io import BytesIO
from pathlib import Path
import sys
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DEFAULT_DESTINATION = Path(__file__).resolve().parents[1] / "data" / "ml-100k"


def download(destination: Path) -> None:
    names = ("u.data", "u.item", "README")
    if all((destination / name).is_file() for name in names):
        print(f"Base já disponível em {destination}.")
        return
    print(f"Baixando {URL} ...")
    with urlopen(URL, timeout=60) as response:
        archive = response.read()
    with ZipFile(BytesIO(archive)) as zipped:
        # Lê só os arquivos necessários e usa nomes fixos, sem extractall.
        contents = {name: zipped.read(f"ml-100k/{name}") for name in names}
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        temporary = destination / f"{name}.part"
        temporary.write_bytes(content)
        temporary.replace(destination / name)
    print(f"MovieLens 100K salvo em {destination}.")
    print(f"Fonte, condições de uso e citação: {destination / 'README'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    try:
        download(args.destination)
    except (OSError, URLError, BadZipFile, KeyError) as exc:
        print(f"Não foi possível baixar a base: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
