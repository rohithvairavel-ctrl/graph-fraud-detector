#!/usr/bin/env python3
"""Generate synthetic payment graph with planted fraud rings (shared device/IP + cycles)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DATA_DIR, N_ACCOUNTS, N_FRAUD_RINGS, N_LEGIT_EDGES, RING_SIZE, SAMPLE_DIR, SEED

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def generate_accounts(n: int, rng: np.random.Generator) -> pd.DataFrame:
    ages = rng.integers(30, 1800, size=n)
    countries = rng.choice(["US", "UK", "IN", "BR", "DE", "NG", "PH"], size=n, p=[0.35, 0.12, 0.2, 0.1, 0.08, 0.08, 0.07])
    return pd.DataFrame({"account_id": [f"A{i:05d}" for i in range(n)], "account_age_days": ages, "country": countries, "is_fraud": 0, "fraud_ring_id": -1})

def plant_fraud_rings(accounts, n_rings, ring_size, rng):
    ids = accounts["account_id"].tolist()
    used, rings, device_map, ip_map = set(), [], {}, {}
    n = len(ids)
    for i, aid in enumerate(ids):
        device_map[aid] = f"D{rng.integers(0, n // 3):05d}"
        ip_map[aid] = f"IP{rng.integers(0, n // 2):05d}"
    for r in range(n_rings):
        size = int(rng.integers(ring_size[0], ring_size[1] + 1))
        candidates = [a for a in ids if a not in used]
        if len(candidates) < size:
            break
        members = list(rng.choice(candidates, size=size, replace=False))
        used.update(members)
        rings.append(members)
        shared_dev, shared_ip = f"D_RING_{r:02d}", f"IP_RING_{r:02d}"
        for j, m in enumerate(members):
            if j < max(2, size - 2):
                device_map[m], ip_map[m] = shared_dev, shared_ip
        mask = accounts["account_id"].isin(members)
        accounts.loc[mask, "is_fraud"] = 1
        accounts.loc[mask, "fraud_ring_id"] = r
    return accounts, rings, device_map, ip_map

def generate_transfers(accounts, rings, n_legit, rng):
    """Tabular signals are noisy; rings differ via shared IDs + dense cycles."""
    ids = accounts["account_id"].tolist()
    n = len(ids)
    weights = rng.pareto(1.5, size=n) + 0.1
    weights /= weights.sum()
    rows, tx_id = [], 0
    def add_tx(src, dst, amount, hour, day, kind):
        nonlocal tx_id
        rows.append({"tx_id": f"T{tx_id:07d}", "src_account": src, "dst_account": dst, "amount": round(float(amount), 2), "hour": int(hour), "day": int(day), "kind": kind})
        tx_id += 1
    for _ in range(n_legit):
        src, dst = rng.choice(ids, size=2, replace=False, p=weights)
        amount = float(rng.choice([50, 100, 200, 500, 1000])) if rng.random() < 0.12 else float(rng.lognormal(mean=3.5, sigma=1.2))
        hour = int(rng.choice([0, 1, 2, 3, 4, 22, 23])) if rng.random() < 0.18 else int(rng.integers(7, 22))
        add_tx(src, dst, amount, hour, int(rng.integers(0, 90)), "legit")
    for r, members in enumerate(rings):
        m = len(members)
        for i in range(m):
            src, dst = members[i], members[(i + 1) % m]
            if rng.random() < 0.35:
                amount, hour = float(rng.choice([100, 250, 500, 1000])), int(rng.choice([0, 1, 2, 3, 22, 23]))
            else:
                amount, hour = float(rng.lognormal(mean=3.6, sigma=1.1)), int(rng.integers(8, 21))
            add_tx(src, dst, amount, hour, int(rng.integers(0, 90)), f"ring_{r}_cycle")
        for _ in range(m * 3):
            src, dst = rng.choice(members, size=2, replace=False)
            add_tx(src, dst, float(rng.lognormal(mean=3.4, sigma=1.0)), int(rng.integers(0, 24)), int(rng.integers(0, 90)), f"ring_{r}_dense")
        outsiders = [a for a in ids if a not in members]
        for _ in range(max(2, m // 3)):
            add_tx(rng.choice(members), rng.choice(outsiders), float(rng.lognormal(mean=4.0, sigma=0.8)), int(rng.integers(0, 24)), int(rng.integers(0, 90)), f"ring_{r}_cashout")
    return pd.DataFrame(rows)

def build_account_tabular(accounts, transfers, device_map, ip_map):
    sent = transfers.rename(columns={"src_account": "account_id"})
    recv = transfers.rename(columns={"dst_account": "account_id"})
    all_tx = pd.concat([sent[["account_id", "amount", "hour", "day"]], recv[["account_id", "amount", "hour", "day"]]], ignore_index=True)
    cp_src = transfers.groupby("src_account")["dst_account"].nunique()
    cp_dst = transfers.groupby("dst_account")["src_account"].nunique()
    counterparties = (cp_src.add(cp_dst, fill_value=0)).astype(int)
    agg = all_tx.groupby("account_id").agg(tx_count=("amount", "count"), tx_amount_sum=("amount", "sum"), tx_amount_mean=("amount", "mean"), tx_amount_std=("amount", "std"), tx_amount_max=("amount", "max"), avg_tx_hour=("hour", "mean"))
    agg["tx_amount_std"] = agg["tx_amount_std"].fillna(0.0)
    night = all_tx.assign(is_night=all_tx["hour"].isin([0, 1, 2, 3, 4, 22, 23]).astype(int))
    agg["night_tx_ratio"] = night.groupby("account_id")["is_night"].mean()
    agg["unique_counterparties"] = counterparties.reindex(agg.index).fillna(0).astype(int)
    feat = accounts.set_index("account_id").join(agg, how="left")
    for c in ["tx_count", "tx_amount_sum", "tx_amount_mean", "tx_amount_std", "tx_amount_max", "unique_counterparties", "avg_tx_hour", "night_tx_ratio"]:
        feat[c] = feat[c].fillna(0.0)
    feat["device_id"] = feat.index.map(device_map)
    feat["ip_id"] = feat.index.map(ip_map)
    return feat.reset_index()

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic payment fraud graph")
    parser.add_argument("--n-accounts", type=int, default=N_ACCOUNTS)
    parser.add_argument("--n-legit-edges", type=int, default=N_LEGIT_EDGES)
    parser.add_argument("--n-rings", type=int, default=N_FRAUD_RINGS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    rng = _rng(args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    accounts = generate_accounts(args.n_accounts, rng)
    accounts, rings, device_map, ip_map = plant_fraud_rings(accounts, args.n_rings, RING_SIZE, rng)
    transfers = generate_transfers(accounts, rings, args.n_legit_edges, rng)
    features = build_account_tabular(accounts, transfers, device_map, ip_map)
    accounts.to_csv(DATA_DIR / "accounts.csv", index=False)
    transfers.to_csv(DATA_DIR / "transfers.csv", index=False)
    features.to_csv(DATA_DIR / "account_features.csv", index=False)
    meta = {"n_accounts": int(len(accounts)), "n_transfers": int(len(transfers)), "n_fraud_accounts": int(accounts["is_fraud"].sum()), "fraud_rate": float(accounts["is_fraud"].mean()), "n_rings": len(rings), "ring_sizes": [len(r) for r in rings], "seed": args.seed, "description": "Synthetic payment graph with planted fraud rings sharing devices/IPs and dense cycles."}
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    sample_ids = set(accounts.head(min(200, len(accounts)))["account_id"])
    fraud_ids = set(accounts.loc[accounts["is_fraud"] == 1, "account_id"])
    keep = sample_ids | set(list(fraud_ids)[:40])
    sample_acc = accounts[accounts["account_id"].isin(keep)].copy()
    sample_tx = transfers[transfers["src_account"].isin(keep) & transfers["dst_account"].isin(keep)].copy()
    sample_feat = features[features["account_id"].isin(keep)].copy()
    sample_acc.to_csv(SAMPLE_DIR / "accounts.csv", index=False)
    sample_tx.to_csv(SAMPLE_DIR / "transfers.csv", index=False)
    sample_feat.to_csv(SAMPLE_DIR / "account_features.csv", index=False)
    (SAMPLE_DIR / "meta.json").write_text(json.dumps({"note": "Small sample subset; full data in data/", "n_accounts": len(sample_acc), "n_transfers": len(sample_tx), "n_fraud": int(sample_acc["is_fraud"].sum())}, indent=2))
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
