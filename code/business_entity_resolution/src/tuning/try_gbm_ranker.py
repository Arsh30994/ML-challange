import json, numpy as np, polars as pl, lightgbm as lgb, time
tot_all = 304818
u = pl.read_parquet('/workspace/amlc/work/tune_union_m30r5.parquet')
F = ["cos_word", "cos_char", "cos_sk", "cos_addr"]
g = ["s1_idx", "src"]
u = u.with_columns([(pl.col(f).max().over(g) - pl.col(f)).alias(f"gap_{f}") for f in F] +
                   [pl.col(f).rank("ordinal", descending=True).over(g).cast(pl.Float32).alias(f"rk_{f}") for f in F] +
                   [pl.len().over(g).cast(pl.Float32).alias("grp_n"), (pl.col("methods") & 16 > 0).cast(pl.Float32).alias("rev"),
                    pl.col("methods").cast(pl.Float32).alias("mbits")])
feats = F + [f"gap_{f}" for f in F] + [f"rk_{f}" for f in F] + ["grp_n", "rev", "mbits"]
tr = (u["s1_idx"].hash(3) % 2 == 0).to_numpy()
X = u.select(feats).to_numpy().astype(np.float32); y = u["y"].to_numpy()
t = time.time()
m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, seed=42, verbose=-1, num_threads=6,
                   feature_fraction=0.9, bagging_fraction=0.5, bagging_freq=1), lgb.Dataset(X[tr], y[tr]), 300)
s = m.predict(X, num_threads=6).astype(np.float32)
lin = X[:, :4] @ np.array([-1.1758, 4.3119, 5.2392, 13.1152], dtype=np.float32)
res = {"train_s": round(time.time()-t,1)}
ev = u.select(g + ["y"]).with_columns(pl.Series("gbm", s), pl.Series("lin", lin), pl.Series("te", ~tr)).filter(pl.col("te"))
# true pairs of held-out S1 (denominator must include pairs missed by the union): approximate by halves of the S1 subset
te_true = int(ev["y"].sum())
for name in ("gbm", "lin"):
    e = ev.with_columns(pl.col(name).rank("ordinal", descending=True).over(g).alias("r"))
    res[name] = {k: int(e.filter((pl.col("r") <= k) & pl.col("y")).height) for k in (5, 10, 15, 20, 30)}
res["heldout_true_in_union"] = te_true
print(json.dumps(res, indent=1))
m.save_model('/workspace/amlc/work/tmp/tune_gbm.txt')
