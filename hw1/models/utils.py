"""Only past data determines the catalog, user cohort and all model features."""
from pathlib import Path
import hashlib
import shutil
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from scipy import sparse

URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
SHA256 = "f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b"
SOURCE = "https://archive.ics.uci.edu/dataset/352/online+retail"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_data(root="data/raw"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "online_retail.zip"
    if not archive.exists():
        partial = archive.with_suffix(".part")
        with urllib.request.urlopen(URL, timeout=120) as r, partial.open("wb") as f:
            shutil.copyfileobj(r, f)
        partial.replace(archive)
    actual = sha256(archive)
    if actual != SHA256:
        raise ValueError(f"Unexpected source archive SHA256: {actual}; verify the source before use")
    # Extract only the named member, never arbitrary paths from an archive.
    with zipfile.ZipFile(archive) as z:
        with z.open("Online Retail.xlsx") as src, (root / "Online Retail.xlsx").open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return root / "Online Retail.xlsx"


def load_raw(root="data/raw"):
    return pd.read_excel(download_data(root), engine="openpyxl")


def clean_data(raw):
    """A purchase is a positive priced merchandise invoice line, not a review."""
    report = {"raw_rows": len(raw), "raw_missing": raw.isna().sum().to_dict(),
              "raw_exact_duplicates": int(raw.duplicated().sum())}
    d = raw.copy()
    stages = []

    def keep(mask, reason):
        nonlocal d
        before = len(d)
        d = d.loc[mask].copy()
        stages.append({"step": reason, "removed": before - len(d), "remaining": len(d)})

    keep(~d.duplicated(), "exact_duplicates")
    keep(d[["CustomerID", "StockCode", "InvoiceDate"]].notna().all(axis=1), "missing_keys")
    keep(~d.InvoiceNo.astype(str).str.upper().str.startswith("C"), "cancellations")
    keep((d.Quantity > 0) & (d.UnitPrice > 0), "nonpositive_quantity_or_price")
    # Standard product code: five digits, optionally a variant letter.
    # Excludes postage, fees, manual adjustments, gift vouchers and special services.
    keep(d.StockCode.astype(str).str.fullmatch(r"\d{5}[A-Za-z]?"), "nonstandard_stock_codes")
    d["CustomerID"] = d.CustomerID.astype("int64").astype(str)
    d["StockCode"] = d.StockCode.astype(str).str.upper()
    d["InvoiceNo"] = d.InvoiceNo.astype(str)
    d["Description"] = d.Description.fillna("UNKNOWN").astype(str).str.strip().str.upper()
    d["Country"] = d.Country.fillna("UNKNOWN").astype(str)
    d = d.sort_values(["InvoiceDate", "InvoiceNo", "CustomerID", "StockCode"], kind="stable").reset_index(drop=True)
    report.update({"cleaning_steps": stages, "clean_rows": len(d),
                   "clean_users": d.CustomerID.nunique(), "clean_items": d.StockCode.nunique(),
                   "clean_pairs": len(d.drop_duplicates(["CustomerID", "StockCode"])),
                   "date_min": str(d.InvoiceDate.min()), "date_max": str(d.InvoiceDate.max())})
    return d, report


def interaction_matrices(history, shape, cutoff):
    # Invoice counts, not quantities: limit the effect of wholesale basket sizes.
    grouped = history.groupby(["u", "i"], sort=True).agg(n=("InvoiceNo", "nunique"), last=("InvoiceDate", "max"))
    u, i = grouped.index.get_level_values(0), grouped.index.get_level_values(1)
    n = sparse.csr_matrix((grouped.n.to_numpy(dtype=float), (u, i)), shape=shape)
    x = n.copy()
    x.data[:] = 1.0
    age = (pd.Timestamp(cutoff) - grouped["last"]).dt.total_seconds().to_numpy() / 86400
    if len(age) and age.min() < 0:
        raise ValueError("Future event reached the training matrix")
    ages = sparse.csr_matrix((age, (u, i)), shape=shape)
    return x, n, ages


"""FM features are derived solely from the current training-history snapshot."""
from collections import Counter
import re
import numpy as np
from scipy import sparse


def tokens(text):
    return sorted(set(re.findall(r"[A-Z]{3,}", str(text).upper())))


def build_features(history, shape, metadata=True, max_tokens=1000):
    n_users, n_items = shape
    past = history.sort_values(["InvoiceDate", "InvoiceNo"], kind="stable")
    user_meta = past.drop_duplicates("u", keep="last").set_index("u").reindex(range(n_users))
    item_meta = past.drop_duplicates("i", keep="last").set_index("i").reindex(range(n_items))
    user_rows, user_cols, user_vals = list(range(n_users)), list(range(n_users)), [1.] * n_users
    item_rows, item_cols, item_vals = list(range(n_items)), list(range(n_users, n_users + n_items)), [1.] * n_items
    offset = n_users + n_items
    audit = {"metadata": metadata, "users": n_users, "items": n_items, "as_of": str(past.InvoiceDate.max())}
    if metadata:
        countries = user_meta.Country.fillna("UNKNOWN").astype(str)
        country_map = {v: j for j, v in enumerate(sorted(countries.unique()))}
        for u, country in enumerate(countries):
            user_rows.append(u); user_cols.append(offset + country_map[country]); user_vals.append(1.)
        offset += len(country_map)
        # Price is the historical median, not the price in the future purchase.
        prices = history.groupby("i").UnitPrice.median().reindex(range(n_items))
        boundaries = np.array([1., 2., 5., 10., 20.])
        bins = np.searchsorted(boundaries, prices.to_numpy(), side="right")
        for i, price_bin in enumerate(bins):
            item_rows.append(i); item_cols.append(offset + int(price_bin)); item_vals.append(1.)
        offset += len(boundaries) + 1
        text = [tokens(t) for t in item_meta.Description.fillna("UNKNOWN")]
        frequency = Counter(t for row in text for t in row)
        vocabulary = [t for t, count in sorted(frequency.items(), key=lambda kv: (-kv[1], kv[0])) if count >= 3][:max_tokens]
        token_map = {t: j for j, t in enumerate(vocabulary)}
        for i, row in enumerate(text):
            selected = [token_map[t] for t in row if t in token_map]
            for t in selected:
                item_rows.append(i); item_cols.append(offset + t); item_vals.append(1 / np.sqrt(len(selected)))
        offset += len(token_map)
        audit.update({"countries": len(country_map), "price_boundaries_gbp": boundaries.tolist(),
                      "title_vocabulary_size": len(token_map), "items_without_title_features": sum(not any(t in token_map for t in r) for r in text)})
    uf = sparse.csr_matrix((user_vals, (user_rows, user_cols)), shape=(n_users, offset))
    itf = sparse.csr_matrix((item_vals, (item_rows, item_cols)), shape=(n_items, offset))
    audit["total_features"] = offset
    return uf, itf, audit


"""Full-catalog evaluation; all models use this exact same implementation."""
import numpy as np
import pandas as pd
from scipy import sparse


def prepare_evaluation(holdout, known, k=10):
    target = sparse.csr_matrix((np.ones(len(holdout)), (holdout.u, holdout.i)), shape=known.shape)
    target.data[:] = 1
    before = target.nnz
    target = target - target.multiply(known > 0)
    target.eliminate_zeros()
    candidates = known.shape[1] - np.diff(known.indptr)
    counts = np.diff(target.indptr)
    users = np.flatnonzero((counts > 0) & (candidates >= k))
    report = {"holdout_unique_pairs": before, "repeat_pairs_excluded": before - target.nnz,
              "novel_relevant_pairs": target.nnz, "eligible_users": len(users),
              "cohort_users": known.shape[0], "catalog_items": known.shape[1],
              "users_with_no_novel_positives": int((counts == 0).sum()),
              "users_with_less_than_k_candidates": int((candidates < k).sum())}
    if not len(users):
        raise ValueError("No evaluable users")
    return {"users": users, "target": target.tocsr(), "known": known, "k": k, "audit": report}


def recommend(model, users, known, k=10):
    scores = np.array(model.score(users), dtype=float, copy=True)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite model scores")
    rows, cols = known[users].nonzero()
    scores[rows, cols] = -np.inf
    if np.any(np.isfinite(scores).sum(axis=1) < k):
        raise ValueError("Too few candidates for top-k")
    # Stable sorting: equal scores are resolved by the ascending item index.
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def ranking_metrics(recs, truth):
    n, k = recs.shape
    selected = truth[np.repeat(np.arange(n), k), recs.ravel()]
    if sparse.issparse(selected):
        selected = selected.toarray()
    hits = np.asarray(selected, dtype=float).reshape(n, k)
    rel_count = np.diff(truth.indptr)
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideal = np.cumsum(discounts)[np.minimum(rel_count, k) - 1]
    if np.any(rel_count == 0):
        raise ValueError("Metrics require at least one relevant item per user")
    return {f"NDCG@{k}": (hits @ discounts) / ideal,
            f"Recall@{k}": hits.sum(axis=1) / rel_count,
            f"Precision@{k}": hits.mean(axis=1)}


def evaluate(model, protocol, batch_size=128):
    users, known, k = protocol["users"], protocol["known"], protocol["k"]
    recs = np.vstack([recommend(model, users[a:a + batch_size], known, k) for a in range(0, len(users), batch_size)])
    individual = pd.DataFrame(ranking_metrics(recs, protocol["target"][users]), index=users)
    individual.index.name = "user_index"
    summary = individual.mean().to_dict()
    summary[f"Coverage@{k}"] = len(np.unique(recs)) / known.shape[1]
    summary["evaluated_users"] = len(users)
    return summary, individual




from . import ItemKNN, IALS, EASE, FactorizationMachine, CustomIALS

def history_bundle(history, shape, cutoff, metadata=True):
    x, n, age = interaction_matrices(history, shape, cutoff)
    uf, itf, feature_audit = build_features(history, shape, metadata)
    return {"x": x, "n": n, "age": age, "uf": uf, "itf": itf, "feature_audit": feature_audit}


def train_model(name, params, bundle, seed):
    if name == "ItemKNN":
        return ItemKNN(**params).fit(bundle["x"])
    if name == "iALS":
        return IALS(**params, seed=seed).fit(bundle["n"])
    if name == "EASE":
        return EASE(**params).fit(bundle["x"])
    if name in {"FM", "FM_ID_only"}:
        return FactorizationMachine(**params, seed=seed).fit(bundle["x"], bundle["uf"], bundle["itf"])
    if name == "CustomiALS":
        return CustomIALS(**params, seed=seed).fit(bundle["n"], bundle["age"])
    raise ValueError(name)


