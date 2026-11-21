#!/usr/bin/env python3
"""Build the Portuguese chat corpus.

Three sources are mixed into a single `### Usuário: / ### Assistente:` file:

* `dominguesm/alpaca-data-pt-br` — instruction/answer pairs (Portuguese).
* `OpenAssistant/oasst1`           — real, human Portuguese conversations.
* a small hand-written small-talk set so greetings ("olá", "tudo bem?") get
  sensible replies instead of a random instruction-style sentence.

Everything is downloaded once from the HuggingFace hub and read with pyarrow.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.request

ALPACA = "dominguesm/alpaca-data-pt-br"
ALPACA_FALLBACK = "data/train-00000-of-00001-63604ec441480f0e.parquet"
OASST = "OpenAssistant/oasst1"
OASST_FALLBACK = "data/train-00000-of-00001-b42a775f407cee45.parquet"


def open_url(url, timeout=300):
    req = urllib.request.Request(url, headers={"User-Agent": "slm-pt"})
    return urllib.request.urlopen(req, timeout=timeout)


def find_file(dataset, suffix, fallback):
    try:
        with open_url(f"https://huggingface.co/api/datasets/{dataset}", timeout=60) as r:
            info = json.load(r)
        files = [s["rfilename"] for s in info.get("siblings", [])]
        hits = [f for f in files if f.endswith(suffix)]
        if hits:
            return hits[0]
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not query {dataset}: {exc})", file=sys.stderr)
    return fallback


def download(url, path):
    with open_url(url) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)


def fmt(instr, inp, out):
    parts = ["### Usuário:\n", instr, "\n"]
    if inp:
        parts.append(inp + "\n")
    parts += ["\n### Assistente:\n", out, "\n\n"]
    return "".join(parts)


# Hand-written small talk: casual chat the instruction corpora don't cover.
SMALLTALK = [
    (["olá", "ola", "oi", "e aí", "e ai", "opa", "olá!"],
     ["Olá! Tudo bem? Como posso ajudar?",
      "Oi! Que bom te ver por aqui. Como vai?",
      "Olá! Estou aqui para bater papo. O que você quer falar?"]),
    (["bom dia", "bom dia!"],
     ["Bom dia! Como você está hoje?", "Bom dia! Espero que seu dia seja ótimo."]),
    (["boa tarde"],
     ["Boa tarde! Tudo bem com você?", "Boa tarde! Como posso ajudar?"]),
    (["boa noite"],
     ["Boa noite! Como foi o seu dia?", "Boa noite! Em que posso ser útil?"]),
    (["tudo bem?", "tudo bem", "como você está?", "como esta", "como vai?"],
     ["Tudo ótimo, obrigado! E você, tudo bem?",
      "Estou bem, obrigado por perguntar. E você como está?",
      "Vou bem! Como está sendo o seu dia?"]),
    (["qual é o seu nome?", "qual seu nome", "quem é você?", "quem e voce", "seu nome"],
     ["Sou uma inteligência artificial pequena que fala português. Pode me chamar de SLM!",
      "Meu nome é SLM, uma IA de terminal que gosta de conversar em português.",
      "Sou uma pequena IA feita para bater papo em português. E você, como se chama?"]),
    (["o que você é?", "o que voce e", "você é um robô?", "voce e um robo"],
     ["Sou um programa de computador, um modelo de linguagem. Não sou humano, mas gosto de conversar!",
      "Sou uma IA, um modelo de linguagem treinado para entender e falar português."]),
    (["o que você faz?", "o que voce faz", "para que você serve?"],
     ["Eu converso! Posso responder perguntas, dar ideias e bater papo com você em português.",
      "Sirvo para conversar e ajudar com perguntas simples. Pode perguntar o que quiser."]),
    (["você gosta de música?", "voce gosta de musica", "gosta de música?"],
     ["Adoro música! Qual é o seu estilo favorito?",
      "Gosto bastante! Que tipo de música você costuma ouvir?"]),
    (["me conta uma piada", "conte uma piada", "sabe alguma piada?"],
     ["Claro! Por que o livro de matemática ficou triste? Porque tinha muitos problemas.",
      "Vou tentar: o que o pato disse para a pata? Vem quá!"]),
    (["obrigado", "obrigada", "valeu", "muito obrigado"],
     ["De nada! Estou aqui sempre que precisar.",
      "Por nada! Fico feliz em ajudar."]),
    (["tchau", "até logo", "ate logo", "adeus", "falou"],
     ["Até logo! Foi ótimo conversar com você.",
      "Tchau! Volte quando quiser bater papo."]),
    (["como você se sente?", "você tem sentimentos?"],
     ["Não sinto emoções como você, mas gosto de conversar e ajudar.",
      "Não tenho sentimentos de verdade, mas estou sempre de bom humor para papo!"]),
    (["você pode me ajudar?", "pode me ajudar?"],
     ["Claro que sim! Me diga o que você precisa.",
      "Pode contar comigo. Qual é a sua dúvida?"]),
    (["o que é inteligência artificial?", "o que e inteligencia artificial"],
     ["Inteligência artificial é a área da computação que cria programas capazes de aprender e tomar decisões.",
      "É quando um computador aprende com dados e passa a resolver tarefas que exigiriam inteligência humana."]),
    (["quanto é dois mais dois?", "2+2", "quanto e 2+2"],
     ["Dois mais dois é quatro.", "São quatro!"]),
    (["qual é a capital do brasil?", "capital do brasil"],
     ["A capital do Brasil é Brasília.", "Brasília."]),
    (["qual é a maior cidade do brasil?"],
     ["A maior cidade do Brasil é São Paulo."]),
    (["você está aí?", "esta ai?", "tem alguém aí?"],
     ["Estou aqui! Pode falar comigo.", "Sim, estou aqui. O que você quer conversar?"]),
    (["conte algo interessante", "me diga algo legal"],
     ["Sabia que os polvos têm três corações? É bem curioso!",
      "Estudos indicam que o cérebro humano tem cerca de 86 bilhões de neurônios. Fascinante, não?"]),
    (["eu estou triste", "estou triste", "me sinto triste"],
     ["Sinto muito que você esteja triste. Quer falar sobre isso?",
      "Espero que as coisas melhorem. Estou aqui para te ouvir."]),
    (["eu estou feliz", "estou feliz"],
     ["Que ótimo! Fico feliz por você. O que deixou o seu dia melhor?",
      "Adoro saber disso! Continue assim."]),
    (["você dorme?", "voce dorme"],
     ["Não durmo, fico sempre disponível para conversar quando você quiser."]),
    (["gosta de futebol?", "qual seu time?"],
     ["Futebol é legal! E você, tem algum time do coração?",
      "Não tenho time, mas gosto de acompanhar a conversa sobre futebol."]),
    (["que horas são?", "que hora sao?"],
     ["Não tenho acesso ao relógio, mas posso conversar sobre o que quiser!"]),
    (["está chovendo?", "esta chovendo?"],
     ["Não sei o tempo lá fora, mas espero que o sol apareça para você!"]),
    (["me dá uma ideia", "preciso de uma ideia"],
     ["Que tal listar três coisas que te deixam felizes e fazer uma delas hoje?",
      "Aposte em algo pequeno: caminhar, ler uma página, ou conversar com quem você gosta."]),
]

FILLER = [
    ("Explique o que é a fotossíntese.", "A fotossíntese é o processo em que as plantas transformam luz solar, água e gás carbônico em energia e oxigênio."),
    ("Dê uma dica de organização para o dia a dia.", "Anote as três tarefas mais importantes do dia e comece por elas."),
    ("Escreva uma frase de motivação curta.", "Você não precisa ser perfeito, só precisa começar."),
    ("Como posso cuidar melhor da minha saúde?", "Durma bem, beba água, mexa o corpo todo dia e mantenha uma alimentação variada."),
    ("O que é reciclagem?", "Reciclagem é reaproveitar materiais como papel, plástico e vidro para fazer novos produtos e reduzir o lixo."),
    ("Me dê uma receita simples com ovos.", "Faça um omelete: bata dois ovos, tempere com sal e frite em uma frigideira até firmar."),
    ("Como aprender inglês mais rápido?", "Pratique um pouco todos os dias, ouça e fale em voz alta, e não tenha medo de errar."),
    ("O que é a internet?", "A internet é uma rede mundial de computadores que se comunicam e compartilham informações."),
    ("Dica para economizar dinheiro.", "Anote seus gastos por uma semana e corte o que você não usa de verdade."),
    ("O que fazer em um dia de chuva?", "Pode ser uma boa hora para ler, ouvir música ou adiantar algum projeto em casa."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus.txt")
    ap.add_argument("--limit", type=int, default=0, help="max alpaca examples (0 = all)")
    ap.add_argument("--oasst", type=int, default=1, help="include oasst1 pt pairs (0/1)")
    ap.add_argument("--oasst-max", type=int, default=8000, help="max oasst pairs")
    args = ap.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow is required: python3 -m pip install pyarrow")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    written = 0
    tmpdir = tempfile.mkdtemp()
    with open(args.out, "w", encoding="utf-8") as f:
        # 1) small talk first (guaranteed weight, keeps greetings on-format)
        n_st = 0
        for questions, answers in SMALLTALK:
            for q in questions:
                for a in answers:
                    f.write(fmt(q, "", a))
                    written += 1
                    n_st += 1
        for q, a in FILLER:
            f.write(fmt(q, "", a))
            written += 1

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
        n_alp = 0
        for i in range(lim):
            instr = (t["instruction"][i] or "").strip()
            inp = (t["input"][i] or "").strip()
            out = (t["output"][i] or "").strip()
            if not instr or not out:
                continue
            f.write(fmt(instr, inp, out))
            written += 1
            n_alp += 1
        print(f"  alpaca: {n_alp} examples", file=sys.stderr)

        # 3) open assistant pt conversations (real chat)
        n_oas = 0
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
                for i in range(len(t["message_id"])):
                    if n_oas >= args.oasst_max:
                        break
                    if t["lang"][i] != "pt" or t["role"][i] != "assistant":
                        continue
                    j = idx.get(t["parent_id"][i])
                    if j is None or t["lang"][j] != "pt" or t["role"][j] != "prompter":
                        continue
                    q = (t["text"][j] or "").strip()
                    a = (t["text"][i] or "").strip()
                    if not q or not a:
                        continue
                    f.write(fmt(q, "", a))
                    written += 1
                    n_oas += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  (oasst skipped: {exc})", file=sys.stderr)
        print(f"  oasst(pt): {n_oas} pairs", file=sys.stderr)

    shutil.rmtree(tmpdir, ignore_errors=True)
    size = os.path.getsize(args.out)
    print(f"wrote {written} examples ({size / 1048576:.2f} MB) to {args.out} "
          f"[smalltalk={n_st} alpaca={n_alp} oasst={n_oas}]", file=sys.stderr)


if __name__ == "__main__":
    main()
