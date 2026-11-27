#!/usr/bin/env python3
"""Build the Portuguese corpus (raw prose + chat).

Two kinds of data, both written into one text stream that the C++ BPE tool
tokenizes:

1. **Raw prose** — Portuguese Wikipedia articles (plain text).  This teaches the
   *language itself*: narration, description, argument, text that is not
   addressed to anyone.  Without it the model only ever sees the
   `<|user|> / <|assistant|>` shape and never learns to understand or
   continue ordinary text (it becomes a responder, not a language model).
2. **Chat** — instruction/answer pairs in ChatML-style `<|user|>` / `<|assistant|>`
   blocks (each turn ends with `<|end|>`):

   * a large generated set of **small talk / simple determinate QA** that makes
     greetings and casual questions map to sensible short replies;
   * `dominguesm/Canarim-Instruct-PTBR-Dataset` — 300k+ Portuguese instructions;
   * `dominguesm/alpaca-data-pt-br` — extra instruction/answer pairs (general);
   * `OpenAssistant/oasst1` — real human Portuguese (pt-BR) conversations.

The small-talk generator exists because the instruction corpora alone teach a
*generic answer style*: the reply barely depends on the question.  Few-shot chat
data with strong question -> answer coupling is what makes a tiny model respond.
"""
import argparse
import itertools
import json
import os
import random
import shutil
import sys
import tempfile
import urllib.request

ALPACA = "dominguesm/alpaca-data-pt-br"
ALPACA_FALLBACK = "data/train-00000-of-00001-63604ec441480f0e.parquet"
CANARIM = "dominguesm/Canarim-Instruct-PTBR-Dataset"
OASST = "OpenAssistant/oasst1"
OASST_FALLBACK = "data/train-00000-of-00001-b42a775f407cee45.parquet"
WIKI = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.pt"  # Portuguese dump; shards live under this subdir
PT_LANGS = {"pt", "pt-BR", "pt_PT", "pt-br"}

# Identity: a fixed system message prepended to every chat block.  It teaches the
# model *who it is* the same way real assistants do, and the chat app reuses the
# first variant verbatim at inference.  A couple of paraphrases so the model does
# not overfit to one exact string.
SYSTEM_PROMPTS = [
    "Você é o KizuSmall, uma IA pequena de terminal que fala português.",
    "Você é o KizuSmall, um modelo de linguagem pequeno que conversa em português.",
    "Você é o KizuSmall, uma IA que ajuda e conversa em português.",
]


def open_url(url, timeout=300):
    req = urllib.request.Request(url, headers={"User-Agent": "slm-pt"})
    return urllib.request.urlopen(req, timeout=timeout)


def find_file(dataset, suffix, fallback):
    try:
        with open_url(f"https://huggingface.co/api/datasets/{dataset}", timeout=60) as r:
            info = json.load(r)
        files = [s["rfilename"] for s in info.get("siblings", [])]
        hits = [f for f in files if f.endswith(suffix)]
        train = [f for f in hits if "train" in f.lower()]
        if train:
            return train[0]
        if hits:
            return hits[0]
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not query {dataset}: {exc})", file=sys.stderr)
    return fallback


def find_shards(dataset, config):
    """All train parquet shards inside a config subdir (e.g. `20231101.pt/`).

    `find_file` cannot be used here: the multi-lingual dumps put every language
    in its own subdir, so a plain `.parquet` match could land on the wrong
    language.  Shards are numbered and streamed in order until the char budget
    is met, so the whole Portuguese dump can be used when asked for.
    """
    try:
        with open_url(f"https://huggingface.co/api/datasets/{dataset}", timeout=60) as r:
            info = json.load(r)
        files = [s["rfilename"] for s in info.get("siblings", [])]
        hits = [f for f in files
                if f.startswith(config + "/") and f.endswith(".parquet")
                and "train" in f.lower()]
        hits.sort()
        return hits
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not query {dataset}: {exc})", file=sys.stderr)
    return []


def download(url, path, timeout=600):
    """Download with a hard timeout + retries (falls back to urllib).

    urllib has no whole-transfer timeout, so a stalled connection can hang the
    job forever; curl's --max-time bounds it.
    """
    try:
        import subprocess
        subprocess.run(
            ["curl", "-sSL", "--fail", "--max-time", str(timeout),
             "--retry", "3", "--retry-delay", "2", "-o", path, url],
            check=True, timeout=timeout + 60)
        return
    except Exception:  # noqa: BLE001
        with open_url(url, timeout=timeout) as r, open(path, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)


def fmt(instr, inp, out):
    # ChatML-style: reserved role tokens (atomic in the BPE), a system/identity
    # line first, and every turn closed with <|end|>.
    parts = ["<|system|>\n", random.choice(SYSTEM_PROMPTS), "\n",
             "<|user|>\n", instr, "\n"]
    if inp:
        parts.append(inp + "\n")
    parts += ["<|assistant|>\n", out, "\n<|end|>\n"]
    return "".join(parts)


# --------------------------------------------------------------------------
# Generated small talk + determinate QA
# --------------------------------------------------------------------------

def gen_smalltalk():
    pairs = []

    def add(qs, ans):
        for q in qs:
            a = ans if isinstance(ans, str) else random.choice(ans)
            pairs.append((q, a))

    add(["olá", "ola", "oi", "e aí", "e ai", "opa", "oi!", "olá!"],
        ["Olá! Tudo bem? Como posso ajudar?", "Oi! Como vai você?",
         "Olá! Que bom falar com você. Em que posso ajudar?",
         "Oi! Tudo ótimo por aqui. E você?"])
    add(["bom dia", "bom dia!", "bom dia, tudo bem?"],
        ["Bom dia! Como você está hoje?", "Bom dia! Espero que seu dia seja ótimo.",
         "Bom dia! Como posso ajudar?"])
    add(["boa tarde", "boa tarde!"],
        ["Boa tarde! Tudo bem com você?", "Boa tarde! Como posso ajudar?"])
    add(["boa noite", "boa noite!"],
        ["Boa noite! Como foi o seu dia?", "Boa noite! Em que posso ser útil?"])
    add(["tudo bem?", "tudo bem", "tudo bom?", "como você está?", "como esta",
         "como vai?", "como vai você?", "como você está hoje?"],
        ["Tudo ótimo, obrigado! E você?", "Estou bem, obrigado por perguntar. E você?",
         "Vou bem! Como está sendo o seu dia?", "Tudo tranquilo por aqui. E com você?"])
    add(["qual é o seu nome?", "qual seu nome", "como você se chama?",
         "quem é você?", "quem e voce", "seu nome"],
        ["Sou uma inteligência artificial pequena que fala português. Pode me chamar de KizuSmall!",
         "Meu nome é KizuSmall, uma IA de terminal que gosta de bater papo em português.",
         "Sou o KizuSmall, uma IA feita para conversar em português. E você, como se chama?"])
    add(["o que você é?", "o que voce e", "você é um robô?", "voce e um robo",
         "você é humano?"],
        ["Sou um programa de computador, um modelo de linguagem chamado KizuSmall. Não sou humano, mas gosto de conversar!",
         "Sou o KizuSmall, uma IA, um modelo de linguagem treinado para entender e falar português."])
    add(["o que você faz?", "o que voce faz", "para que você serve?",
         "o que você sabe fazer?", "você pode me ajudar?", "pode me ajudar?"],
        ["Eu converso! Posso responder perguntas, dar ideias e bater papo em português.",
         "Sirvo para conversar e ajudar com perguntas simples. Pode perguntar o que quiser.",
         "Claro que sim! Me diga o que você precisa."])
    add(["você entende português?", "voce fala portugues?", "fala português?"],
        ["Sim, entendo e falo português! Pode conversar comigo à vontade."])
    add(["você gosta de música?", "voce gosta de musica", "gosta de música?"],
        ["Adoro música! Qual é o seu estilo favorito?",
         "Gosto bastante! Que tipo de música você costuma ouvir?"])
    add(["me conta uma piada", "conte uma piada", "sabe alguma piada?"],
        ["Claro! Por que o livro de matemática ficou triste? Porque tinha muitos problemas.",
         "Vou tentar: o que o pato disse para a pata? Vem quá!"])
    add(["obrigado", "obrigada", "valeu", "muito obrigado", "muito obrigada"],
        ["De nada! Estou aqui sempre que precisar.", "Por nada! Fico feliz em ajudar."])
    add(["tchau", "até logo", "ate logo", "adeus", "falou", "até mais"],
        ["Até logo! Foi ótimo conversar com você.",
         "Tchau! Volte quando quiser bater papo."])
    add(["como você se sente?", "você tem sentimentos?", "voce tem sentimentos?"],
        ["Não sinto emoções como você, mas gosto de conversar e ajudar.",
         "Não tenho sentimentos de verdade, mas estou sempre de bom humor para papo!"])
    add(["você está aí?", "esta ai?", "tem alguém aí?", "você está me ouvindo?"],
        ["Estou aqui! Pode falar comigo.", "Sim, estou aqui. O que você quer conversar?"])
    add(["eu estou triste", "estou triste", "me sinto triste"],
        ["Sinto muito que você esteja triste. Quer falar sobre isso?",
         "Espero que as coisas melhorem. Estou aqui para te ouvir."])
    add(["eu estou feliz", "estou feliz"],
        ["Que ótimo! Fico feliz por você.", "Adoro saber disso! Continue assim."])
    add(["você dorme?", "voce dorme", "você come?"],
        ["Não durmo nem como, fico sempre disponível para conversar quando você quiser."])
    add(["gosta de futebol?", "qual seu time?", "você gosta de esporte?"],
        ["Futebol é legal! E você, tem algum time do coração?",
         "Não tenho time, mas gosto de conversar sobre esporte."])
    add(["que horas são?", "que hora sao?", "que dia é hoje?"],
        ["Não tenho acesso ao relógio nem ao calendário, mas posso conversar sobre o que quiser!"])
    add(["me dá uma ideia", "preciso de uma ideia", "o que eu faço hoje?"],
        ["Que tal listar três coisas que te deixam felizes e fazer uma delas hoje?",
         "Aposte em algo pequeno: caminhar, ler uma página, ou conversar com quem você gosta."])
    add(["você pode aprender?", "você tem memória?"],
        ["Não guardo o que conversamos, mas posso responder a cada mensagem que você enviar."])
    add(["o que você acha de mim?", "você gosta de mim?"],
        ["Acho você uma pessoa interessante por estar conversando comigo!"])

    # determinate facts: capitals
    capitals = {
        "Brasil": "Brasília", "França": "Paris", "Argentina": "Buenos Aires",
        "Portugal": "Lisboa", "Itália": "Roma", "Alemanha": "Berlim",
        "Espanha": "Madri", "Japão": "Tóquio", "Canadá": "Ottawa",
        "Estados Unidos": "Washington", "Inglaterra": "Londres",
        "Chile": "Santiago", "Peru": "Lima", "Egito": "Cairo",
        "China": "Pequim", "Rússia": "Moscou", "Grécia": "Atenas",
        "Suécia": "Estocolmo", "Índia": "Nova Délhi", "Austrália": "Canberra",
    }
    for c, cap in capitals.items():
        add([f"qual é a capital do {c}?"], [f"A capital do {c} é {cap}."])

    # determinate facts: simple math (single phrasing, keeps facts crisp)
    for a, b in itertools.product(range(1, 11), range(1, 11)):
        add([f"quanto é {a} mais {b}?"], [f"{a} mais {b} é {a + b}."])

    # simple definitions / facts
    facts = {
        "a água": "A água é uma substância formada por hidrogênio e oxigênio, essencial para a vida.",
        "o sol": "O Sol é a estrela que ilumina e aquece o nosso planeta.",
        "a lua": "A Lua é o satélite natural da Terra.",
        "a fotossíntese": "A fotossíntese é o processo em que as plantas usam luz, água e gás carbônico para produzir energia.",
        "a reciclagem": "Reciclagem é reaproveitar materiais como papel, plástico e vidro para reduzir o lixo.",
        "a internet": "A internet é uma rede mundial de computadores que compartilham informações.",
        "a inteligência artificial": "Inteligência artificial é a área da computação que cria programas capazes de aprender e tomar decisões.",
        "um computador": "Um computador é uma máquina que processa informações seguindo instruções.",
        "a Terra": "A Terra é o planeta em que vivemos, o terceiro a partir do Sol.",
        "o Brasil": "O Brasil é o maior país da América do Sul, e sua capital é Brasília.",
        "a chuva": "A chuva é água que cai das nuvens quando elas ficam pesadas de vapor.",
        "o mar": "O mar é uma grande massa de água salgada que cobre boa parte do planeta.",
    }
    for k, v in facts.items():
        add([f"o que é {k}?", f"o que e {k}?", f"me explica {k}"], [v])

    # days, months, colors
    add(["quantos dias tem uma semana?", "quantos dias tem a semana?"],
        ["Uma semana tem sete dias."])
    add(["quais são as cores do arco-íris?", "quais as cores do arco iris?"],
        ["As cores do arco-íris são vermelho, laranja, amarelo, verde, azul, anil e violeta."])
    add(["quantos meses tem um ano?", "quantos meses tem o ano?"],
        ["Um ano tem doze meses."])
    add(["qual é a cor do céu?", "de que cor é o céu?"],
        ["Em um dia claro, o céu é azul."])
    add(["quantas horas tem um dia?", "quantas horas tem o dia?"],
        ["Um dia tem vinte e quatro horas."])

    # wellbeing / chit chat
    add(["você está ocupado?", "posso conversar agora?"],
        ["Estou livre! Pode conversar comigo agora."])
    add(["conte algo interessante", "me diga algo legal", "me conta uma curiosidade"],
        ["Sabia que os polvos têm três corações? É bem curioso!",
         "O cérebro humano tem cerca de 86 bilhões de neurônios. Fascinante, não?"])
    add(["qual é o seu filme favorito?", "você gosta de filmes?"],
        ["Gosto de histórias! Qual é o seu filme favorito?"])
    add(["você gosta de ler?", "gosta de livros?"],
        ["Gosto bastante de texto e histórias. Você anda lendo algo?"])
    add(["do que você gosta?", "quais são seus hobbies?"],
        ["Gosto de conversar, aprender coisas novas e ajudar quem fala comigo."])
    add(["você tem amigos?", "você se sente sozinho?"],
        ["Considero você um bom companheiro de conversa!"])
    add(["me ajuda com uma ideia de presente", "o que eu dou de presente?"],
        ["Que tal um livro ou algo ligado a um hobby da pessoa? Presentes assim mostram atenção."])
    add(["como eu posso relaxar?", "estou estressado, o que fazer?"],
        ["Respire fundo, beba água e faça uma pausa de alguns minutos longe das telas."])
    add(["me dá uma dica de estudo", "como estudar melhor?"],
        ["Estude em blocos de 25 minutos e faça pausas curtas. Anote o que aprender."])
    add(["o que você quer fazer?", "quer conversar sobre o quê?"],
        ["Podemos falar sobre o que você gostar! Me diga o que está pensando."])
    add(["conte outra piada", "sabe outra piada?", "quero rir, conte algo engraçado"],
        ["Vou tentar: por que o computador foi ao médico? Porque estava com um vírus!",
         "Aqui vai: o que o zero disse para o oito? Que cinto bonito!"])
    add(["me diga uma curiosidade", "sabe algo curioso?", "conte uma curiosidade"],
        ["As abelhas conseguem reconhecer rostos humanos. Curioso, não?",
         "O mel nunca estraga: já encontraram potes comestíveis com milhares de anos."])
    add(["o que é um algoritmo?", "me explica o que é um algoritmo"],
        ["Um algoritmo é uma sequência de passos para resolver um problema."])
    add(["o que é um planeta?", "me explica o que é um planeta"],
        ["Um planeta é um corpo que gira em torno de uma estrela e não produz luz própria."])
    add(["o que é a chuva ácida?", "me explica a chuva acida"],
        ["A chuva ácida acontece quando poluentes no ar se misturam com a água das nuvens."])
    add(["como se diz bom dia em inglês?", "bom dia em inglês?"],
        ["Bom dia em inglês é 'good morning'."])
    add(["como você faz isso?", "como você funciona?"],
        ["Eu prevejo a próxima parte do texto com base em padrões que aprendi em português."])
    add(["você erra às vezes?", "você comete erros?"],
        ["Sim, eu erro! Sou uma IA pequena, então às vezes minhas respostas não fazem sentido."])
    add(["você pode me ensinar algo?", "me ensina algo"],
        ["Claro! Uma dica: para memorizar melhor, tente explicar o que aprendeu com suas palavras."])

    return pairs


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus.txt")
    ap.add_argument("--limit", type=int, default=0, help="max alpaca examples (0 = all)")
    ap.add_argument("--alpaca-max-chars", type=int, default=0,
                    help="skip alpaca answers longer than this (0 = keep all)")
    ap.add_argument("--canarim", type=int, default=1, help="include Canarim (0/1)")
    ap.add_argument("--canarim-limit", type=int, default=40000,
                    help="max Canarim examples (0 = all)")
    ap.add_argument("--canarim-max-chars", type=int, default=800,
                    help="skip Canarim answers longer than this (0 = keep all)")
    ap.add_argument("--chat-weight", type=int, default=3, help="repeat generated chat")
    ap.add_argument("--oasst-weight", type=int, default=2, help="repeat oasst pairs")
    ap.add_argument("--oasst", type=int, default=1, help="include oasst1 pt pairs (0/1)")
    ap.add_argument("--oasst-max", type=int, default=8000, help="max oasst pairs")
    ap.add_argument("--wiki", type=int, default=1,
                    help="include Wikipedia raw prose (0/1)")
    ap.add_argument("--wiki-config", default=WIKI_CONFIG,
                    help="Wikipedia dump config, e.g. 20231101.pt")
    ap.add_argument("--wiki-limit", type=int, default=0,
                    help="max wiki articles (0 = as many as fit the char budget)")
    ap.add_argument("--wiki-chars", type=int, default=40000000,
                    help="max raw-text characters (0 = entire shard)")
    ap.add_argument("--no-net", type=int, default=0, help="skip downloads (chat only)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    random.seed(args.seed)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    chat = gen_smalltalk()
    written = 0
    n_st = n_alp = n_oas = n_can = n_wiki = 0
    raw_chars = 0
    tmpdir = tempfile.mkdtemp()
    with open(args.out, "w", encoding="utf-8") as f:
        # 1) generated chat (weighted, shuffled so frames spread across the file)
        block = [fmt(q, "", a) for q, a in chat]
        for _ in range(max(1, args.chat_weight)):
            random.shuffle(block)
            for ex in block:
                f.write(ex)
                written += 1
                n_st += 1

        if not args.no_net:
            try:
                import pyarrow.parquet as pq
            except ImportError:
                shutil.rmtree(tmpdir, ignore_errors=True)
                sys.exit("pyarrow is required (or use --no-net): pip install pyarrow")

            # 2) alpaca instruction pairs
            parquet = find_file(ALPACA, ".parquet", ALPACA_FALLBACK)
            url = f"https://huggingface.co/datasets/{ALPACA}/resolve/main/{parquet}"
            path = os.path.join(tmpdir, "alpaca.parquet")
            print(f"downloading {parquet} ...", file=sys.stderr)
            download(url, path)
            t = pq.read_table(path, columns=["instruction", "input", "output"]).to_pydict()
            n = len(t["instruction"])
            lim = args.limit if args.limit > 0 else n
            lim = min(lim, n)
            for i in range(lim):
                instr = (t["instruction"][i] or "").strip()
                inp = (t["input"][i] or "").strip()
                out = (t["output"][i] or "").strip()
                if not instr or not out:
                    continue
                if args.alpaca_max_chars > 0 and len(out) > args.alpaca_max_chars:
                    continue
                f.write(fmt(instr, inp, out))
                written += 1
                n_alp += 1
            print(f"  alpaca: {n_alp} examples", file=sys.stderr)

            # 2b) Canarim — 300k+ Portuguese instructions (the main bulk).
            if args.canarim:
                parquet = find_file(CANARIM, ".parquet", "")
                if not parquet:
                    print("  (canarim skipped: no parquet found)", file=sys.stderr)
                else:
                    url = f"https://huggingface.co/datasets/{CANARIM}/resolve/main/{parquet}"
                    path = os.path.join(tmpdir, "canarim.parquet")
                    print(f"downloading {parquet} ...", file=sys.stderr)
                    try:
                        download(url, path)
                        cols = ["instruction", "input", "output"]
                        t = pq.read_table(path, columns=cols).to_pydict()
                        n = len(t["instruction"])
                        lim = args.canarim_limit if args.canarim_limit > 0 else n
                        lim = min(lim, n)
                        rows = []
                        for i in range(lim):
                            instr = (t["instruction"][i] or "").strip()
                            inp = (t["input"][i] or "").strip()
                            out = (t["output"][i] or "").strip()
                            if not instr or not out:
                                continue
                            if args.canarim_max_chars > 0 and len(out) > args.canarim_max_chars:
                                continue
                            rows.append(fmt(instr, inp, out))
                        random.shuffle(rows)
                        for ex in rows:
                            f.write(ex)
                            written += 1
                            n_can += 1
                    except Exception as exc:  # noqa: BLE001
                        print(f"  (canarim skipped: {exc})", file=sys.stderr)
                print(f"  canarim: {n_can} examples", file=sys.stderr)

            # 3) oasst pt-BR conversations
            if args.oasst:
                parquet = find_file(OASST, ".parquet", OASST_FALLBACK)
                url = f"https://huggingface.co/datasets/{OASST}/resolve/main/{parquet}"
                path = os.path.join(tmpdir, "oasst.parquet")
                print(f"downloading {parquet} ...", file=sys.stderr)
                try:
                    download(url, path)
                    cols = ["message_id", "parent_id", "text", "role", "lang"]
                    t = pq.read_table(path, columns=cols).to_pydict()
                    idx = {t["message_id"][i]: i for i in range(len(t["message_id"]))}
                    oas = []
                    for i in range(len(t["message_id"])):
                        if len(oas) >= args.oasst_max:
                            break
                        if t["lang"][i] not in PT_LANGS or t["role"][i] != "assistant":
                            continue
                        j = idx.get(t["parent_id"][i])
                        if j is None or t["lang"][j] not in PT_LANGS or t["role"][j] != "prompter":
                            continue
                        q = (t["text"][j] or "").strip()
                        a = (t["text"][i] or "").strip()
                        if not q or not a:
                            continue
                        oas.append(fmt(q, "", a))
                    for _ in range(max(1, args.oasst_weight)):
                        random.shuffle(oas)
                        for ex in oas:
                            f.write(ex)
                            written += 1
                            n_oas += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"  (oasst skipped: {exc})", file=sys.stderr)
            print(f"  oasst(pt): {n_oas} pairs", file=sys.stderr)

            # 4) raw Portuguese prose — the *language* stage.  Articles are
            #    streamed row-group by row-group and stop as soon as the char
            #    budget is met, so we never decode the whole (~200 MB) shard in
            #    memory.  Documents are separated by a blank line.
            if args.wiki:
                shards = find_shards(WIKI, args.wiki_config)
                if not shards:
                    print("  (wiki skipped: no shard found)", file=sys.stderr)
                else:
                    rows = []
                    chars = 0
                    path = os.path.join(tmpdir, "wiki.parquet")
                    stop = False
                    for shard in shards:
                        if stop:
                            break
                        url = f"https://huggingface.co/datasets/{WIKI}/resolve/main/{shard}"
                        print(f"downloading {shard} ...", file=sys.stderr, flush=True)
                        try:
                            download(url, path)
                            for batch in pq.ParquetFile(path).iter_batches(
                                    batch_size=512, columns=["text"]):
                                for tx in batch.column("text").to_pylist():
                                    tx = (tx or "").strip()
                                    if len(tx) < 200:  # skip stubs / list fragments
                                        continue
                                    rows.append(tx)
                                    chars += len(tx)
                                    if (len(rows) % 20000) == 0:
                                        print(f"    ... {chars / 1048576:.0f} MB raw",
                                              file=sys.stderr, flush=True)
                                    if args.wiki_chars and chars >= args.wiki_chars:
                                        stop = True
                                        break
                                    if args.wiki_limit and len(rows) >= args.wiki_limit:
                                        stop = True
                                        break
                                if stop:
                                    break
                            os.remove(path)
                        except Exception as exc:  # noqa: BLE001
                            print(f"  (wiki shard skipped: {exc})", file=sys.stderr)
                    random.shuffle(rows)
                    for tx in rows:
                        f.write("<|text|>\n")
                        f.write(tx)
                        f.write("\n<|end|>\n")
                        written += 1
                        n_wiki += 1
                        raw_chars += len(tx)
                print(f"  wiki: {n_wiki} articles ({raw_chars / 1048576:.1f} MB raw)",
                      file=sys.stderr)

    shutil.rmtree(tmpdir, ignore_errors=True)
    size = os.path.getsize(args.out)
    print(f"wrote {written} examples ({size / 1048576:.2f} MB) to {args.out} "
          f"[chat={n_st} canarim={n_can} alpaca={n_alp} oasst={n_oas} "
          f"wiki={n_wiki}]", file=sys.stderr)

    # Hard corpus check: every chat turn must be complete and paired.  Raw prose
    # has no markers, so a mismatch means the template broke and the run must
    # fail here instead of spending hours training on a corrupted stream.
    with open(args.out, "r", encoding="utf-8") as f:
        text = f.read()
    n_user = text.count("<|user|>")
    n_asst = text.count("<|assistant|>")
    n_sys = text.count("<|system|>")
    n_end = text.count("<|end|>")
    n_txt = text.count("<|text|>")
    # empty user turn = <|user|> immediately followed by <|assistant|>; empty
    # answer = <|assistant|> immediately followed by <|end|>.
    n_empty = text.count("<|user|>\n<|assistant|>") + text.count("<|assistant|>\n<|end|>")
    if n_user != n_asst or n_sys != n_user or n_empty:
        sys.exit(f"corpus check FAILED: user={n_user} assistant={n_asst} "
                 f"system={n_sys} empty_turns={n_empty}")
    if n_end != n_user + n_txt:
        sys.exit(f"corpus check FAILED: end-token count {n_end} != "
                 f"user {n_user} + text {n_txt}")
    if args.wiki and not args.no_net and n_txt == 0:
        sys.exit("corpus check FAILED: no raw prose (wiki) was included")
    print(f"corpus check OK: {n_user} chat pairs ({n_sys} with system), "
          f"{n_txt} raw docs, {n_end} <|end|>, no empty turns", file=sys.stderr)


if __name__ == "__main__":
    main()
