# Atividade 1 — Recomendação colaborativa

Implementação didática de **filtragem colaborativa baseada em usuários**, seguindo o capítulo 2 de *Recommender Systems: An Introduction*, de Jannach, Zanker, Felfernig e Friedrich.

A ideia é encontrar pessoas com padrões de notas semelhantes aos de um usuário e usar suas avaliações para prever quais itens ele pode gostar. O projeto reproduz o exemplo de Alice, permite testar outras bases e mostra quais vizinhos contribuíram para cada previsão.

## Comece por aqui

Requisito: **Python 3.10+**. Todos os comandos abaixo usam somente a biblioteca padrão; não é necessário instalar pacotes. Execute a partir da pasta `atividade-1`:

```bash
cd atividade-1  # se estiver na raiz do repositório
python main.py
```

A demonstração seleciona automaticamente User1 e User2 e prevê **4,87 / 5 para Alice em item5**. Também mostra similaridades, média do usuário e contribuição de cada vizinho. `python main.py demo` executa a mesma demonstração.

Em seguida, experimente uma base de filmes:

```bash
# Conhecer os IDs disponíveis e o tamanho da base
python main.py inspect --dataset movies

# Consultar similaridades, inclusive as negativas
python main.py neighbors --dataset movies --user Ana --top 5

# Prever uma nota e acompanhar o cálculo
python main.py predict --dataset movies --user Ana --item m12 --k 3

# Listar os cinco melhores itens ainda não avaliados
python main.py recommend --dataset movies --user Ana --top 5 --k 3

# Separar treino e teste e medir os erros
python main.py evaluate --dataset movies --seed 42 --limit 0
```

Use `python main.py --help` para listar comandos e `python main.py predict --help` para consultar as opções de cada um. Coloque as opções depois do nome do comando. Sem `--dataset` ou `--csv`, a base utilizada é `book`.

## Bases de dados

| Base | Usuários | Itens | Notas observadas | Uso |
| --- | ---: | ---: | ---: | --- |
| `book` | 5 | 5 | 24 | Conferir as contas do exemplo de Alice |
| `movies` | 12 | 12 | 112 | Experimentar perfis e avaliações ausentes sem internet |
| `movielens` | 943 | 1.682 | 100.000 | Testar com avaliações reais de filmes |

`data/book.csv` transcreve a matriz do livro. `data/movies.csv` contém avaliações **inteiramente fictícias**, elaboradas para este exercício: os títulos identificam os itens, mas as notas não representam pesquisas ou preferências de pessoas reais. Essa base permite estudar o comportamento do código; seus resultados não demonstram qualidade em dados reais.

### MovieLens 100K

O [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) é uma base do GroupLens com avaliações de 1 a 5. O script baixa o arquivo oficial e salva apenas `u.data`, `u.item` e o `README` original com as condições de uso:

```bash
python scripts/download_movielens.py
python main.py inspect --dataset movielens
python main.py recommend --dataset movielens --user 1 --top 5 --k 20 --min-common 5
python main.py predict --dataset movielens --user 1 --item 313 --k 20 --min-common 5
python main.py evaluate --dataset movielens --k 20 --min-common 5 --seed 42 --limit 200
```

O download exige internet e usa cerca de 5 MB de transferência. Depois dele, os comandos funcionam sem internet. Os arquivos ficam em `data/ml-100k/`, ignorada pelo Git; os termos da base exigem autorização para redistribuição. Consulte o README original antes de publicar dados ou usá-los em outro contexto. O script também pode ser executado novamente: se os três arquivos já existirem, reutiliza a cópia local.

Para usar uma cópia existente em outra pasta:

```bash
python main.py inspect --dataset movielens --movielens-dir /caminho/ml-100k
```

O carregador preserva os IDs oficiais como strings e lê os títulos na codificação Latin-1 do arquivo `u.item`. Os caminhos das bases incluídas são resolvidos em relação ao código; caminhos de CSV e caminhos fornecidos pelo usuário são relativos à pasta em que o comando é executado.

### Seu próprio CSV

Crie um arquivo UTF-8 com uma avaliação por linha:

```csv
user_id,item_id,rating,title
Ana,filme1,5,Filme de aventura
Ana,filme2,3,Filme de comédia
Bruno,filme1,4,Filme de aventura
Bruno,filme2,2,Filme de comédia
Bruno,filme3,5,Filme de suspense
```

```bash
python main.py inspect --csv minhas_notas.csv
python main.py predict --csv minhas_notas.csv --user Ana --item filme3
python main.py recommend --csv minhas_notas.csv --user Ana
```

As colunas `user_id`, `item_id` e `rating` são obrigatórias; `title` é opcional. Notas podem ter casas decimais com ponto, mas precisam ser finitas e estar entre 1 e 5. Um item não avaliado pode ser omitido ou ter `rating` vazio; em memória, a ausência pode ser uma chave ausente ou `None`. **Não use zero para representar ausência.**

Pares usuário/item duplicados, IDs vazios, notas inválidas, linhas incompletas e títulos conflitantes para o mesmo item geram uma mensagem com a localização do problema. IDs diferenciam maiúsculas de minúsculas. Nenhuma linha é descartada silenciosamente.

## Como o algoritmo funciona

### 1. Similaridade de Pearson

Considere `Iuv`, o conjunto dos itens avaliados tanto por `u` quanto por `v`. Calculamos a média de cada um **nesse conjunto em comum**:

```text
sim(u,v) = Σ [(r(u,i) − média_u_Iuv) × (r(v,i) − média_v_Iuv)]
           ──────────────────────────────────────────────────────
           √Σ (r(u,i) − média_u_Iuv)² × √Σ (r(v,i) − média_v_Iuv)²
```

Todas as somas percorrem `i ∈ Iuv`. A correlação varia de −1 a +1: valores positivos indicam padrões semelhantes; valores negativos indicam padrões opostos. Se houver menos de `min_common` itens compartilhados ou uma das séries for constante, o código retorna zero, pois não há correlação utilizável.

A implementação preserva a convenção do código original: Pearson sobre os pares em comum. A notação do livro apresenta a média do usuário; usar a média de todo o histórico dentro de Pearson é outra convenção de centralização e pode produzir valores diferentes, sobretudo em matrizes esparsas. Na etapa de previsão abaixo, as médias usam todo o histórico disponível no treino.

### 2. Escolha dos vizinhos

Para cada item que desejamos prever:

1. Calculamos a similaridade do usuário com os demais, sem incluí-lo como próprio vizinho.
2. Mantemos somente usuários com correlação **estritamente positiva** que avaliaram esse item.
3. Ordenamos por similaridade decrescente e selecionamos até `k` vizinhos.

O limite é aplicado **depois** de filtrar quem avaliou o item. Assim, um usuário muito semelhante que não conhece o item não ocupa a vaga de quem pode contribuir. Empates são resolvidos pelo ID em ordem lexicográfica, para resultados reproduzíveis.

### 3. Previsão centrada na média

Aplicamos a fórmula 2.3 do livro:

```text
previsão(u,i) = média(u) + Σ [sim(u,v) × (r(v,i) − média(v))]
                          ─────────────────────────────────
                                       Σ sim(u,v)
```

As somas percorrem os vizinhos selecionados para o item; as médias usam todas as notas observadas de cada usuário. Isso permite considerar que algumas pessoas dão notas mais altas que outras.

Para Alice, `média(Alice) = 4`, `média(User1) = 2,4` e `média(User2) = 3,8`:

```text
4 + [0,8528 × (3 − 2,4) + 0,7071 × (5 − 3,8)] / (0,8528 + 0,7071)
≈ 4,87
```

O modelo limita a nota final a `[1, 5]`, mas conserva o valor bruto em `Prediction.raw_score`. A função legada `predict`, disponível em `main.py`, continua retornando a previsão sem esse limite. Sem vizinhos válidos, a previsão gera `NoNeighborsError`; não inventa uma nota pela média.

**Conferência dos materiais:** no slide numerado 9, as similaridades de User2 e User3 aparecem trocadas. A matriz e o cálculo resultam em User1 ≈ 0,85, User2 ≈ 0,70, User3 = 0 e User4 ≈ −0,79, como descrito no livro, páginas impressas 14–16. Os testes conferem esses valores sem arredondar os cálculos intermediários.

### 4. Recomendações

O método `recommend` prevê notas apenas para itens ainda não avaliados. Itens sem vizinhos válidos são ignorados. O resultado é ordenado por nota prevista decrescente, com desempate pelo ID; uma lista pode conter menos resultados que o solicitado. A saída informa quantos vizinhos sustentam cada sugestão.

## Como avaliar os resultados

`evaluate` reserva por padrão 20% das notas de cada usuário para teste, mantendo ao menos uma nota no treino. Usuários com apenas uma avaliação permanecem integralmente no treino. Por arredondamento e por garantir ao menos uma nota de teste quando possível, a proporção global pode diferir de 20%.

O modelo é construído **somente com o treino**: médias, similaridades e seleção dos vizinhos não usam as notas reservadas. A semente controla a divisão e a amostragem, permitindo repetir o experimento.

| Medida | Interpretação |
| --- | --- |
| MAE | Média do erro absoluto entre nota prevista e real; menor é melhor |
| RMSE | Raiz da média dos erros ao quadrado; penaliza erros grandes |
| Cobertura | Fração das notas tentadas para as quais o modelo conseguiu prever |
| Média do usuário | Baseline que usa a média do histórico de treino daquele usuário |

O baseline e o modelo são comparados **nas mesmas notas com previsão colaborativa**. Notas sem vizinhos ou de itens ausentes no treino reduzem a cobertura e não entram nas métricas de erro. Se nenhuma previsão for possível, MAE e RMSE ficam indisponíveis. Por isso, avalie os erros junto com a cobertura.

Por padrão são avaliadas até **200 notas amostradas** do teste reservado, para uma execução rápida no MovieLens. `--limit 0` avalia todas; a saída distingue o tamanho total do teste da quantidade efetivamente tentada.

```bash
# Comparar configurações mantendo divisão e amostra iguais
python main.py evaluate --dataset movielens --k 10 --min-common 5 --seed 42 --limit 200
python main.py evaluate --dataset movielens --k 30 --min-common 5 --seed 42 --limit 200
```

Exemplo verificado com `k=20`, `min_common=5`, `seed=42` e limite de 200 no MovieLens: 80.367 notas de treino, 19.633 reservadas, cobertura de 100% na amostra, MAE de 0,7561 e RMSE de 0,9685. O baseline apresentou MAE de 0,8125 e RMSE de 1,0119. Isso descreve essa configuração e amostra; não é uma garantia para outras divisões ou dados.

Esse holdout é aleatório, sem usar timestamps. Ele mede previsão de notas e não qualidade do ranking, evolução temporal ou desempenho para usuários novos. Para escolher parâmetros e depois reportar um resultado final, acrescente uma divisão de validação e preserve um teste independente.

## Organização do código

```text
atividade-1/
├── main.py                     # Entrada do terminal e API original
├── recommender/
│   ├── __init__.py              # API pública do pacote
│   ├── data.py                  # Dataset e leitura/validação dos dados
│   ├── similarity.py            # Média e Pearson
│   ├── model.py                 # Vizinhos, previsões e recomendações
│   ├── evaluation.py            # Separação treino/teste e métricas
│   └── cli.py                   # Comandos e apresentação dos resultados
├── data/
│   ├── book.csv                 # Matriz original
│   ├── movies.csv               # Base fictícia de filmes
│   └── ml-100k/                 # Download local, fora do Git
├── scripts/
│   └── download_movielens.py    # Obtenção da base oficial e termos de uso
├── test_main.py                 # Regressões do exercício original
└── tests/
    └── test_recommender.py      # Dados esparsos, modelo, métricas e CLI
```

Os PDFs de referência permanecem na pasta da atividade. A lógica matemática não depende da interface de terminal; é possível usar o modelo diretamente:

```python
from recommender.data import load_builtin
from recommender import UserBasedRecommender

base = load_builtin("movies")
modelo = UserBasedRecommender(base.ratings, k=3, min_common=2)

previsao = modelo.predict("Ana", "m12")
print(previsao.score, previsao.neighbors)

for recomendacao in modelo.recommend("Ana", n=5):
    print(base.titles[recomendacao.item], recomendacao.score)
```

O construtor copia as avaliações observadas, evitando que alterações posteriores no dicionário original mudem um modelo já construído. As similaridades são calculadas sob demanda e reaproveitadas por usuário. Para trocar os dados ou os parâmetros, construa um novo modelo. `average`, `pearson`, `predict` e `ratings` continuam importáveis de `main` para manter a compatibilidade com o exercício original.

## Testes

Dentro de `atividade-1`:

```bash
python -m unittest discover -v
```

Ou a partir da raiz do repositório:

```bash
python -m unittest discover -s atividade-1 -v
```

Os testes são independentes de internet e do download do MovieLens. Cobrem o exemplo de Alice, dados com chaves diferentes, ausência de notas, correlação indefinida, seleção dos vizinhos por item, limitação da escala, validação de CSV, formato MovieLens com títulos acentuados, recomendações e métricas com resultados calculados manualmente.

## Limitações e próximos experimentos

- Usuários sem histórico e itens nunca avaliados não têm previsão colaborativa. Não há fallback automático para popularidade ou média.
- Correlações com poucos itens em comum podem parecer fortes por acaso. Experimente aumentar `--min-common`; isso pode diminuir a cobertura.
- Uma nota alta sustentada por um único vizinho não significa alta confiança. A quantidade de vizinhos é exibida para ajudar a interpretar as sugestões.
- Correlações negativas são descartadas por escolha didática. O projeto não implementa filtragem baseada em itens, cosseno ajustado, Slope One ou fatoração de matrizes, embora o capítulo também discuta essas técnicas.
- O algoritmo percorre os outros usuários para calcular similaridades; avaliar todo o MovieLens demora mais que uma amostra. Ele foi organizado para estudo e bases pequenas ou médias, sem infraestrutura de produção.

## Referências

- Jannach, D.; Zanker, M.; Felfernig, A.; Friedrich, G. *Recommender Systems: An Introduction*. Capítulo 2, em especial a seção 2.1 e as fórmulas 2.1–2.4. [PDF local](1jannach_dietmar_zanker_markus_felfernig_alexander_friedrich.pdf).
- [Slides locais — Collaborative recommendation](Recommender_Systems_An_Introduction_Chapter02_Collaborative_recommendation.pdf).
- GroupLens. [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/). Os termos e a descrição dos arquivos acompanham o download em `data/ml-100k/README`.
- Harper, F. M.; Konstan, J. A. (2015). *The MovieLens Datasets: History and Context*. ACM TiiS, 5(4), artigo 19. [DOI: 10.1145/2827872](https://doi.org/10.1145/2827872).
