# Atividade 2 — preparação para TF-IDF

Base em Python 3.10+, usando apenas a biblioteca padrão. Há dois tipos de
entrada: uma matriz termo-documento pequena e um corpus de textos reais.
O cálculo de TF, IDF, normalização, similaridade e ranking **não está implementado**.
Os pontos de entrada em `content_based/tfidf.py` lançam `NotImplementedError`.

## Executar

Na raiz do repositório:

```bash
cd atividade-2
python main.py
python scripts/download_bbc.py
python main.py --dataset bbc
```

O download verifica SHA-256 e utiliza uma versão fixa da fonte. O ZIP fica em
`data/bbc-fulltext.zip`, fora do Git; após o primeiro download, a base funciona
sem internet. Não é necessário extrair o arquivo nem instalar pacotes.

## 1. Shakespeare: exemplo de Hamlet

`data/shakespeare.csv` transcreve a **página 7 do PDF fornecido**
`Recommender_Systems_An_Introduction_Chapter03_Content-based_recommendation.pdf`.
São seis peças: Antony and Cleopatra, Julius Caesar, The Tempest, Hamlet,
Othello e Macbeth. Os sete termos são Antony, Brutus, Caesar, Calpurnia,
Cleopatra, mercy e worser. `load_shakespeare()` devolve `TermMatrix`, com
linhas de termos e colunas de documentos, mantendo a ordem do CSV.

**Inconsistência do material:** embora o slide chame a tabela de frequência
de termos e diga que os vetores são contagens naturais, a primeira coluna
contém `mercy = 1.51` e `worser = 1.37`. Esses valores foram preservados como
decimais, sem correção por suposição. Antes de implementar TF-IDF, precisamos
definir se usaremos esses valores como fornecidos ou uma versão corrigida
documentada. Não há textos completos das peças nessa tabela; não foram
inventados textos para reproduzir as frequências. A matriz sobreposta na
página 8 não é usada como gabarito numérico de TF-IDF.

## 2. BBC: textos encontrados na internet

Fonte primária: [repositório de Derek Greene](https://github.com/derekgreene/bbc-datasets).
São 2.225 notícias em inglês de 2004–2005, divididas em cinco categorias:
business (510), entertainment (386), politics (417), sport (511) e tech (401).
Usamos o arquivo de **textos completos**, sem TF-IDF pré-calculado.

Versão utilizada: `d81c6d9543cb90683b6927f35f6d06ccb4f59b55`,
arquivo `raw/bbc-fulltext.zip`.
SHA-256: `a351031a62325b3d7ff13146eeb035c22feca8290913b5183d1ff3be35109495`.

`load_bbc()` lê o ZIP em Latin-1, preserva os textos e devolve objetos
`Document(id, title, text, category)`. Exemplo de ID: `business/001`.
O título já está no texto; não o concatene novamente.

As categorias são metadados das notícias e não entram na vetorização dos textos.

Segundo a fonte, a base é disponibilizada para pesquisa e uso não comercial;
os direitos dos artigos pertencem à BBC. Referência solicitada pelos autores:
D. Greene e P. Cunningham, *Practical Solutions to the Problem of Diagonal
Dominance in Kernel Document Clustering*, ICML 2006.

## Onde continuar

1. Definir a variante de TF/IDF e resolver o tratamento dos decimais do slide.
2. Implementar `fit_transform_matrix` e `fit_transform_texts` em
   `content_based/tfidf.py`, com saída que preserve os IDs dos documentos.
3. Implementar transformação de consultas e similaridade em `rank`.
