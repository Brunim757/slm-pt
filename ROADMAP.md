# KizuSmall — Roadmap

## Status atual
- Modelo único: `embd=192 layer=5 head=6 block=192 bpe_vocab=2048` (2,61 M params), treinado
  para sempre — cada run retoma o último checkpoint com sucesso, nunca do zero.
- Corpus: 1 shard do Wikipedia PT (`wiki_chars=1500000000`, ~1,5 GB de texto, ~450 M tokens).
- Tokenizer congelado no resume (`slmpt_bpe --load`) → o vocabulário nunca muda, então o
  corpus pode crescer sem invalidar o checkpoint.
- Treino contínuo: `schedule '0 */6 * * *'` (a cada 6h), run de 5h que retoma o anterior.

---

## PRÓXIMO OBJETIVO (implementado nesta rodada)
**Otimização extrema para que o treinamento seja feito de forma ultra-absurda e eficiente.**

Meta: muito mais throughput e capacidade por run, mantendo o modelo único e o resume válido.

### Status
- [x] **(1) Loader** — ids em `uint16` + `mmap` (zero-copy): metade da RAM/disco, sem pico 2x, corpus deixa de ser limitado por RAM.
- [x] **(2) `mmap`** dos ids — o SO pagina sob demanda.
- [x] **(3) Encode BPE paralelo** em streaming (segmentos alinhados a `\n`): 50 MB medido em 6,4s → 1,98s (3,2x, 8 threads).
- [x] **(5) Cache do corpus** entre runs via `actions/cache` (texto cru determinístico).
- [x] **(6) Resume real do AdamW** — m/v + passo global gravados no checkpoint (retrocompatível com checkpoints antigos).
- [x] **(8) Inferência** — `dot_f32`/`dot4_f32` com caminho AVX-512 16-wide (antes caíam no AVX2 8-wide).
- [~] **(4) Kernels de treino** — GEMM/backward já usavam AVX-512; ganho extra de packing/fusão fica para depois (medir antes).
- [~] **(7) batch/block** — o modelo é *compute-bound* (~74% do pico), batch maior não ajuda; manter 32.

### Alavancas ainda em aberto

1. **Loader de dados — teto de memória (maior ganho de capacidade).**
   Hoje `Dataset::load_ids` carrega tudo como `int32` com pico de 2× (buffer + cópia).
   Pareto: guardar ids como **`uint16`** (vocab 2048 cabe) e ler direto em `tokens_`,
   sem o `buf` temporário. Efeito: ~metade da RAM e do arquivo; teto de corpus
   ~1,5 GB → ~3 GB (e sem OOM a 7 GB de runner).

2. **`mmap` do arquivo de ids (zero-copy).**
   Elimina o limite de corpus por RAM de vez — o SO pagina sob demanda. Combina com (1).

3. **Encode BPE mais rápido.**
   Hoje ~8,2 MB/s no encode do corpus (gargalo do prep, não do treino). Paralelizar por
   shard/bloco e melhorar a estrutura de busca de merges.

4. **Kernels de treino (GEMM/atenção).**
   AVX-512: empacotamento (packing) para cache, fusão de ops, redução de tráfego de memória;
   avaliar acumulação em bf16/fp16 onde for seguro (sem quebrar convergência).

5. **Reaproveitar o corpus entre runs.**
   Hoje cada run baixa + re-encoda do zero. Opção: cachear `corpus.ids` como artefato
   (ou split em shards) para o run começar a treinar direto.

6. **Resume verdadeiramente contínuo (AdamW).**
   Hoje o optimizer reinicia a frio no resume. Salvar/restaurar estado do AdamW para
   continuidade real.

7. **Tuning de batch/block + gradient accumulation** para saturar o runner.

8. **Velocidade de inferência/sampling** (chat local, `rodar.sh`).

### Regras que NÃO podem ser quebradas
- Nunca treinar do zero; sempre retomar o checkpoint.
- Nunca mudar o vocabulário/tokenizer (congelado); resume precisa continuar válido.
- Testar local (Termux) antes de gastar runner; medir, não adivinhar.
