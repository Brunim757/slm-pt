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

## Build local

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Treinar

```bash
python3 scripts/fetch_corpus.py --out data/corpus.txt   # baixa o corpus PT
./build/slmpt_train --data data/corpus.txt --out runs \
  --embd 192 --layer 4 --head 6 --block 192 --batch 48 \
  --steps 20000 --max-minutes 300 --save-every 500 --sample-every 500
```

Retomar de um checkpoint:

```bash
./build/slmpt_train --data data/corpus.txt --out runs --resume runs/model.bin ...
```

## Conversar

```bash
./build/slmpt_chat --model runs/model.bin --temp 0.8 --top-k 40
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
gh workflow run train.yml -f steps=20000 -f max_minutes=300

# acompanha e baixa o resultado
gh run watch
gh run download --name slm-pt-run
```

Para treinar por **dias**, baixe o artifact, e re-dispare passando
`--resume runs/model.bin` (via inputs ou ajuste o workflow).
