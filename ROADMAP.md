# KizuSmall — Roadmap

## Status atual
- Modelo único: `embd=192 layer=5 head=6 block=192 bpe_vocab=2048` (2,61 M params), treinado
  para sempre — cada run retoma o último checkpoint com sucesso, nunca do zero.
- Corpus: 1 shard do Wikipedia PT (`wiki_chars=1500000000`, ~1,5 GB de texto, ~450 M tokens).
- Tokenizer congelado no resume (`slmpt_bpe --load`) → o vocabulário nunca muda, então o
  corpus pode crescer sem invalidar o checkpoint.
- Treino contínuo: `schedule '0 */6 * * *'` (a cada 6h), run de 5h que retoma o anterior.

---

## PRÓXIMO OBJETIVO (anotado, NÃO iniciado)
**Otimização extrema para que o treinamento seja feito de forma ultra-absurda e eficiente.**

Meta: muito mais throughput e capacidade por run, mantendo o modelo único e o resume válido.
Nada disto deve ser implementado agora — só registro para atacar quando decidirmos.

### Alavancas candidatas (ordem de impacto estimado)

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
