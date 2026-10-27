"""Baixa uma versão fixa do corpus; valida integridade antes de salvar."""

from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

URL = (
    "https://raw.githubusercontent.com/derekgreene/bbc-datasets/"
    "d81c6d9543cb90683b6927f35f6d06ccb4f59b55/raw/bbc-fulltext.zip"
)
SHA256 = "a351031a62325b3d7ff13146eeb035c22feca8290913b5183d1ff3be35109495"
DESTINATION = Path(__file__).resolve().parents[1] / "data" / "bbc-fulltext.zip"


def main():
    if DESTINATION.exists() and sha256(DESTINATION.read_bytes()).hexdigest() == SHA256:
        print(f"BBC já disponível e verificada: {DESTINATION}")
        return
    with urlopen(URL, timeout=60) as response:
        content = response.read()
    if sha256(content).hexdigest() != SHA256:
        raise ValueError("SHA-256 inesperado; arquivo não foi salvo.")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix(".zip.part")
    temporary.write_bytes(content)
    temporary.replace(DESTINATION)
    print(f"BBC salva em {DESTINATION}")


if __name__ == "__main__":
    main()
