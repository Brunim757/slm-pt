# slm-pt

Uma **SLM (Small Language Model) de terminal, escrita do zero em C++** — sem
nenhum framework de IA, só a STL e o SO. Byte-level (guarante a acentuação do
português), arquitetura GPT (decoder-only transformer) com forward e backward
implementados à mão, AdamW próprio, e treino que roda no **GitHub Actions**.

## Estrutura

```
include/slmpt/   headers da biblioteca
src/             implementação da biblioteca (modelo, otimizador, tokenizer, ...)
apps/            executáveis: train, chat, gradcheck
scripts/         fetch_corpus.py (baixa corpus PT e formata o diálogo)
.github/         workflow de treino no CI
```

## Como funciona

* **Tokenização byte-level**: cada byte UTF-8 é um token (vocabulário = 256).
  Nada de biblioteca de tokenizer, e nenhum acento é perdido.
* **Modelo**: GPT pré-norm, embeddings amarrados (tied), GELU, atenção causal
  multi-head, LayerNorm — tudo escrito à mão.
* **Treino**: cross-entropy + AdamW (weight decay desacoplado, clip de norma),
  warmup + cosine, checkpoints, amostras periódicas.
* **Formato de diálogo**: `### Usuário: ... ### Assistente: ...`, para ele
  aprender a **responder**, não só a copiar texto.
* **Dados** (`scripts/fetch_corpus.py`) misturam três fontes em português:
  `dominguesm/alpaca-data-pt-br` (instruções), conversas humanas reais do
  `OpenAssistant/oasst1` (só as `pt-BR`) e um conjunto **gerado** de small talk /
  QA determinístico, pra "olá", "tudo bem?" etc. terem resposta decente. Os
  pesos (`--chat-weight`, `--oasst-weight`) controlam quanto cada fonte pesa; e
  `--no-net 1` gera o corpus de chat sem baixar nada.
* **Decodificação**: temperatura + top-k + top-p + **penalidade de repetição** e
  parada automática em `### Usuário:`.

## Build local

No Termux/Android o CMake entra em "modo Android" e quebra; use o **Makefile**:

```bash
make -j"$(nproc)"     # binários em build/
```

No PC (Linux/macOS) qualquer um dos dois serve:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

## Treinar

```bash
python3 scripts/fetch_corpus.py --out data/corpus.txt   # baixa o corpus PT
./build/slmpt_train --data data/corpus.txt --out runs \
  --embd 192 --layer 5 --head 6 --block 192 --batch 32 \
  --steps 100000 --max-minutes 300 --save-every 500 --sample-every 250
```

Retomar de um checkpoint:

```bash
./build/slmpt_train --data data/corpus.txt --out runs --resume runs/model.bin ...
```

## Conversar

```bash
./build/slmpt_chat --model runs/model.bin --temp 0.7 --top-k 40 --top-p 0.9 --rep 1.15
```

## Validar a matemática

```bash
./build/slmpt_gradcheck   # compara gradiente analítico x diferenças finitas
```

## Treinar no GitHub Actions

O workflow `.github/workflows/train.yml` faz tudo no runner (mais forte que o
celular): baixa o corpus, compila, roda o gradcheck, faz um smoke test e treina
respeitando um orçamento de tempo, publicando `runs/model.bin` e
`runs/samples.txt` como **artifact**.

```bash
# dispara um treino manual
gh workflow run train.yml -f steps=100000 -f max_minutes=300 -f examples=12000

# acompanha e baixa o resultado
gh run watch
gh run download --name slm-pt-run
```

Como o repositório é **público**, os minutos do Actions são ilimitados. Dá para
treinar por **dias** encadeando runs: cada run publica `runs/model.bin`, e o
próximo continua de onde parou com o input `resume_run=<run id anterior>`:

```bash
gh workflow run train.yml -f resume_run=44908829121 -f max_minutes=300
```

Ou, de uma vez, baixe o melhor checkpoint e converse com ele:

```bash
scripts/get_model.sh          # pega o último run bem-sucedido e abre o chat
```
