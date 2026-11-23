# KizuSmall

O **KizuSmall** é uma **SLM (Small Language Model) de terminal, escrita do zero
em C++** — sem nenhum framework de IA, só a STL e o SO. Tokenizador **BPE
byte-level** (garante a acentuação do português), transformer decoder-only com
**RMSNorm + RoPE + SwiGLU**, forward/backward implementados à mão, AdamW
próprio, **KV cache** para inferência rápida, **quantização int8/int4/int2** e
**RAG local**. O treino roda no **GitHub Actions**.

## Estrutura

```
include/slmpt/   headers da biblioteca
src/             implementação (modelo, otimizador, tokenizer BPE, quant, rag, ...)
apps/            treino/bpe/chat/eval/quantize/rag/gradcheck
scripts/         fetch_corpus.py (baixa o corpus PT) e get_model.sh
.github/         workflows de treino e CI
```

## Arquitetura

* **Tokenizador BPE byte-level** (`apps/bpe.cpp`): alfabeto = 256 bytes crus
  (nada de acento perdido), merges aprendidos do corpus; `token id = 256 + rank`.
  Reduz o texto para ~1/3–1/2 do número de tokens do byte puro, então o modelo
  aprende mais por passo e gera mais rápido.
* **Modelo** (`src/model.cpp`): GPT pré-norm com **RMSNorm**, **RoPE**
  (posição rotativa, sem embedding posicional aprendido), **SwiGLU** e
  embeddings amarrados (tied). Gradientes analíticos verificados por diferenças
  finitas (`slmpt_gradcheck`), incluindo a consistência do KV cache.
* **Treino**: cross-entropy + AdamW (weight decay desacoplado, clip de norma),
  warmup + cosine, checkpoints, amostras periódicas.
* **Inferência**: **KV cache** incremental (cada passo novo é O(T), não O(T²)),
  temperatura + top-k + top-p + penalidade de repetição, parada em `### Usuário:`.
* **Compressão**: quantização simétrica por grupo (8/4/2 bits) dos tensores de
  peso, mantendo as normas em fp32; relatório de tamanho × perplexidade.
* **RAG local**: indexa seus arquivos usando o estado oculto médio do modelo e
  busca por similaridade de cosseno; o chat pode usar isso como contexto.
* **SIMD**: produto interno vetorizado em NEON (ARM/Termux) e AVX2 (CI/PC).

## Build local

No Termux/Android o CMake entra em "modo Android" e quebra; use o **Makefile**:

```bash
make -j"$(nproc)"     # binários em build/
```

No PC (Linux/macOS) qualquer um dos dois serve:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

## Preparar dados (corpus + tokenizador)

```bash
python3 scripts/fetch_corpus.py --out data/corpus.txt --limit 40000
./build/slmpt_bpe --in data/corpus.txt --out data/tokenizer.txt \
  --ids data/corpus.ids --vocab 8192
```

## Treinar

```bash
./build/slmpt_train --data data/corpus.txt --ids data/corpus.ids \
  --tokenizer data/tokenizer.txt --out runs \
  --embd 384 --layer 8 --head 8 --block 256 --batch 32 \
  --steps 40000 --max-minutes 300 --save-every 500 --sample-every 500
```

Retomar de um checkpoint (o tokenizador vai embutido no arquivo):

```bash
./build/slmpt_train ... --resume runs/model.bin
```

## Conversar

```bash
./build/slmpt_chat --model runs/model.bin --temp 0.3 --top-k 40 --top-p 0.9 --rep 1.15
```

Temperatura baixa (0,2–0,4) deixa as respostas mais precisas; temperatura mais
alta (0,6–0,8) dá mais variação, mas inventa mais.

Com RAG local (responder sobre os seus arquivos):

```bash
./build/slmpt_rag build --model runs/model.bin --in notas.txt --index runs/rag.bin
./build/slmpt_chat --model runs/model.bin --rag runs/rag.bin
```

## Avaliar e comprimir

```bash
./build/slmpt_eval     --model runs/model.bin --ids data/corpus.ids --val 0.05
./build/slmpt_quantize --model runs/model.bin --ids data/corpus.ids \
  --bits 4 --group 64 --out runs/model_q4.bin --val 0.02
./build/slmpt_chat     --model runs/model_q4.bin      # roda o modelo comprimido
```

## Validar a matemática

```bash
./build/slmpt_gradcheck   # analítico x diferenças finitas + KV cache
```

## Treinar no GitHub Actions

O workflow `.github/workflows/train.yml` faz tudo no runner: baixa o corpus,
compila, treina o BPE, roda o gradcheck, um smoke test, treina respeitando um
orçamento de tempo, e publica o checkpoint + relatórios (perplexidade e
compressão) como **artifact**.

```bash
# dispara um treino manual
gh workflow run train.yml -f steps=3000 -f max_minutes=300 -f canarim_limit=30000

# acompanha e baixa o resultado
gh run watch
gh run download --name kizu-small-run
```

Como o repositório é **público**, os minutos do Actions são ilimitados. Dá para
treinar por **dias** encadeando runs:

```bash
gh workflow run train.yml -f resume_run=44908829121 -f max_minutes=300
# ou, automático, retomando do último run bem-sucedido:
gh workflow run train.yml -f chain=1 -f max_minutes=300
```

Há ainda um agendamento noturno (`schedule`) que continua o treino sozinho
(`chain=1`), melhorando o modelo enquanto você dorme.

Ou, de uma vez, baixe o melhor checkpoint e converse com ele:

```bash
scripts/get_model.sh          # pega o último run bem-sucedido e abre o chat
```
