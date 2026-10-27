# Atividade 1 — Recomendação colaborativa

Sistema didático que encontra usuários com notas semelhantes, calcula a
similaridade de Pearson e usa os vizinhos para prever notas e recomendar itens.
Requer Python 3.10 ou superior, sem instalar pacotes.

## Executar

Na raiz do repositório:

```bash
cd atividade-1
python main.py
```

Esse comando executa o exemplo do livro: usa User1 e User2 para prever
**4,87 / 5 para Alice no item5**, mostrando a contribuição dos vizinhos.

Para experimentar outras operações:

```bash
# Ver usuários e itens da base de filmes
python main.py inspect --dataset movies

# Ver usuários semelhantes a Ana
python main.py neighbors --dataset movies --user Ana

# Prever a nota de um filme usando até três vizinhos
python main.py predict --dataset movies --user Ana --item m12 --k 3

# Recomendar cinco filmes ainda não avaliados
python main.py recommend --dataset movies --user Ana --top 5 --k 3
```

Sem indicar uma base, o sistema usa `book`. Consulte as opções com
`python main.py --help` ou `python main.py predict --help`.

## Bases de dados

- `book`: exemplo de Alice do livro, com 5 usuários e 5 itens.
- `movies`: base fictícia de filmes, com 12 usuários e 12 itens.
- `movielens`: MovieLens 100K, com avaliações reais de filmes, obtida por download.

Para usar MovieLens:

```bash
python scripts/download_movielens.py
python main.py inspect --dataset movielens
python main.py recommend --dataset movielens --user 1 --min-common 5
```

O download fica em `data/ml-100k/`, fora do Git. A descrição e as condições de
uso acompanham os arquivos. Fonte: [GroupLens](https://grouplens.org/datasets/movielens/100k/).

Também é possível usar um CSV próprio em UTF-8:

```csv
user_id,item_id,rating,title
Ana,filme1,5,Aventura
Ana,filme2,3,Comédia
Bruno,filme1,4,Aventura
Bruno,filme2,2,Comédia
Bruno,filme3,5,Suspense
```

```bash
python main.py recommend --csv minhas_notas.csv --user Ana
```

`title` é opcional. As notas vão de 1 a 5; um campo vazio ou um par
usuário/item omitido representa uma nota ausente. Não use zero para ausência.

## Como funciona

1. Calcula Pearson usando apenas os itens avaliados pelos dois usuários.
2. Para prever um item, seleciona até `k` vizinhos com similaridade positiva
   que tenham avaliado esse item.
3. Calcula a previsão pela média do usuário mais os desvios ponderados dos vizinhos:

```text
previsão = média do usuário
         + soma(similaridade × (nota do vizinho − média do vizinho))
           / soma(similaridades)
```

4. Para recomendar, prevê as notas dos itens ainda não avaliados e retorna
   as maiores. O modelo limita as notas previstas ao intervalo de 1 a 5.

Sem vizinhos que possam contribuir, o sistema informa que não há previsão.
`--k` controla a quantidade de vizinhos; `--min-common` controla quantos itens
os usuários precisam ter avaliado em comum (padrão: 2).

## Arquivos principais

- `main.py`: entrada do programa.
- `recommender/cli.py`: comandos do terminal.
- `recommender/data.py`: leitura das bases.
- `recommender/similarity.py`: média e Pearson.
- `recommender/model.py`: vizinhos, previsões e recomendações.
- `data/`: bases de avaliações.
- `scripts/download_movielens.py`: download do MovieLens.

## Testes

Na pasta `atividade-1`:

```bash
python -m unittest discover -v
```

Os testes verificam os cálculos, o carregamento dos dados e os comandos.

Referências locais: [slides](Slide.pdf) e [livro, capítulo 2](../Livro-RecommenderSystems.pdf).
