#!/usr/bin/env python3
"""
gera_qa_async.py
Gera até 3 perguntas distintas por contrato usando Ollama (llama4) e asyncio.

Uso:
    python3 gera_qa_async.py contratos_clean.csv --n-paraphrases 3 --concurrency 8

Variáveis de ambiente opcionais:
    LLAMA_URL   – URL /api/generate  (default: http://164.41.75.221:11434/api/generate)
    MODEL_NAME  – nome do modelo    (default: llama4)
"""
import argparse, asyncio, csv, json, os, re, time, shutil, httpx, pathlib

# ───────── CONFIG ───────── #
API_URL   = os.getenv("LLAMA_URL",  "http://164.41.75.221:11434/api/generate")
MODEL     = os.getenv("MODEL_NAME", "llama4")
TEMP      = 0.5
OBJ_LIMIT = 350
RETRIES   = 3
OUTFILE   = "qa_pairs.jsonl"

PROMPT_TMPL = """Você é um sistema gerador de dados sintéticos para QA.
Dado um OBJETO e um VALOR, crie {k} perguntas distintas, claras e sem repetir estrutura,
cuja resposta exata seja o VALOR. Formato:
P1: <pergunta 1>
P2: <pergunta 2>
...
Resposta: <valor>
OBJETO: {obj}
VALOR: {val}"""

def slug(txt, n=60):
    return re.sub(r"\W+","_",txt.lower()).strip("_")[:n]

def prepare_output(path=OUTFILE):
    """
    1) Lê todos os IDs já gerados em done_ids.
    2) Faz backup *por cópia* do arquivo atual (sem removê-lo).
    3) Abre para append e retorna (file_handle, done_ids).
    """
    done_ids = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for ln in f:
                try:
                    done_ids.add(json.loads(ln)["id"])
                except:
                    continue
        # backup por cópia
        ts = time.strftime("%Y%m%d-%H%M%S")
        bak = f"{path}.bak-{ts}"
        shutil.copy(path, bak)
        print(f"🔒 backup salvo em {bak}")
    # abrir em modo append: não trunca o arquivo
    f_out = open(path, "a", encoding="utf-8")
    return f_out, done_ids

async def call_llm(client, obj, val, k):
    obj_trim = (obj[:OBJ_LIMIT] + "…") if len(obj) > OBJ_LIMIT else obj
    prompt = PROMPT_TMPL.format(obj=obj_trim, val=val, k=k)
    payload = {"model": MODEL, "prompt": prompt,
               "temperature": TEMP, "stream": False}

    for attempt in range(1, RETRIES + 1):
        try:
            r = await client.post(API_URL, json=payload, timeout=60)
            if r.status_code == 500:
                raise httpx.HTTPStatusError("500 from Ollama",
                                            request=r.request, response=r)
            r.raise_for_status()
            return r.json()["response"]
        except Exception as e:
            if attempt == RETRIES:
                raise
            await asyncio.sleep(3 * attempt)

def extract_questions(text, k):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    qs = [l.split(":",1)[1].strip()
          for l in lines if re.match(r"^P\d+\s*:", l, re.I)]
    if not qs:
        raise ValueError("Resposta sem perguntas:\n" + text)
    return qs[:k]

async def worker(queue, writer, done_ids, k):
    async with httpx.AsyncClient() as client:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break

            uid_base, obj, val, versao = item
            if f"{uid_base}_v0" in done_ids:
                queue.task_done()
                continue

            try:
                resp = await call_llm(client, obj, val, k)
                qs = extract_questions(resp, k)
                for i, q in enumerate(qs):
                    uid = f"{uid_base}_{versao:02d}_v{i}"
                    json.dump({
                        "id":       uid,
                        "question": q,
                        "answer":   val,
                        "objeto":   obj,
                        "valor":    val
                    }, writer, ensure_ascii=False)
                    writer.write("\n")
                print(".", end="", flush=True)
            except Exception as e:
                print(f"\n⚠️ erro {type(e).__name__}: {e} – {obj[:60]}…")

            done_ids.add(f"{uid_base}_v0")
            queue.task_done()

async def main(csv_in, n_para, conc):
    # prepara saída e done_ids
    out, done_ids = prepare_output(OUTFILE)

    # enfileira
    queue = asyncio.Queue()
    total = 0
    with open(csv_in, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            base = slug(row["objeto_contrato"])
            vers = int(row.get("versao_idx", 0))
            queue.put_nowait((base,
                              row["objeto_contrato"],
                              row["valor_contrato"],
                              vers))

    # dispara workers
    workers = [asyncio.create_task(worker(queue, out, done_ids, n_para))
               for _ in range(conc)]

    await queue.join()
    for _ in workers:
        queue.put_nowait(None)
    await asyncio.gather(*workers)

    out.close()
    print(f"\n✅ concluído. Lidos: {total}, QA distintos agora: {len(done_ids)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv_in")
    p.add_argument("--n-paraphrases", "-k", type=int, default=3)
    p.add_argument("--concurrency",   "-c", type=int, default=8)
    args = p.parse_args()
    asyncio.run(main(args.csv_in, args.n_paraphrases, args.concurrency))
