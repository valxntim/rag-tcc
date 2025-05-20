import hashlib
import pickle
import re

import pandas as pd

PICKLE_IN = "data/corpus_by_atos_contratos.pkl"
CSV_OUT = "contratos_clean_composite_key.csv"

def normalize_valor(valor_raw: str) -> str | None:
    """Transforma qualquer 'R$ 1.000,50' ou '1000.50' em 'R$ 1.000,50'."""
    v = re.sub(r"[^\d,\.]", "", valor_raw).replace(".", "").replace(",", ".")
    try:
        num = float(v)
        return f"R$ {num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except ValueError:
        return None

def sha12(txt: str) -> str:
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()[:12]

def make_composite_key(meta: dict) -> str:
    obj = re.sub(r"\s+", " ", meta.get("objeto_contrato", "")).strip()
    obj_hash = sha12(obj) if obj else "no_obj_hash"
    processo = meta.get("processo_gdf", "unknown_processo")
    numero = meta.get("numero_contrato", "unknown_numero")
    return f"{obj_hash}_{processo}_{numero}"

def sha_full_raw_text(entry) -> str:
    raw_text = entry[0]
    return hashlib.sha1(raw_text.encode("utf-8")).hexdigest()[:12]

def main():
    with open(PICKLE_IN, "rb") as f:
        dataset = pickle.load(f)["EXTRATO_CONTRATO"]

    rows = []
    for entry in dataset:
        meta = dict(entry[1:])
        obj = meta.get("objeto_contrato")
        val = meta.get("valor_contrato")
        if not (obj and val):
            continue

        obj_norm = re.sub(r"\s+", " ", obj).strip()
        val_norm = normalize_valor(val)
        if not val_norm:
            continue

        composite_key = make_composite_key(meta)
        raw_text_hash = sha_full_raw_text(entry)

        rows.append({
            "composite_key": composite_key,
            "objeto_contrato": obj_norm,
            "valor_contrato": val_norm,
            "processo_gdf": meta.get("processo_gdf", ""),
            "numero_contrato": meta.get("numero_contrato", ""),
            "raw_text_hash": raw_text_hash
        })

    df = pd.DataFrame(rows)

    # Drop duplicates keeping first occurrence
    df = df.drop_duplicates(subset=["composite_key", "raw_text_hash"])

    # Mark version index within each composite_key group
    df["versao_idx"] = df.groupby("composite_key").cumcount()

    df.to_csv(CSV_OUT, index=False, encoding="utf-8")
    print(f"✅ {len(df)} rows saved to {CSV_OUT}")

if __name__ == "__main__":
    main()
