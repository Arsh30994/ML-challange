import sys, json, numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
tot = 304818
u = pl.read_parquet('/workspace/amlc/work/tune_union_m30r5.parquet')
n1 = pl.read_parquet('/workspace/amlc/work/norm_tune_s1.parquet', columns=["idx", "addr_n"])
e1 = n1.select(pl.col("idx").alias("s1_idx"), (pl.col("addr_n") == "").alias("e1"))
es = pl.concat([pl.read_parquet(f'/workspace/amlc/work/norm_tune_s{s}.parquet', columns=["idx", "addr_n"])
                .select(pl.lit(s, dtype=pl.Int8).alias("src"), pl.col("idx").alias("cand_idx"), (pl.col("addr_n") == "").alias("e2")) for s in (2, 3)])
u = u.join(e1, on="s1_idx", how="left").join(es, on=["src", "cand_idx"], how="left")
u = u.with_columns((pl.col("e1") | pl.col("e2")).cast(pl.Float32).alias("addr_missing"))
print("true pairs with addr_missing:", u.filter(pl.col("y"))["addr_missing"].mean(), "all:", u["addr_missing"].mean())
F = ["cos_word", "cos_char", "cos_sk", "cos_addr"]
u = u.with_columns([(pl.col("addr_missing") * pl.col(f)).alias(f"miss_{f}") for f in F[:3]])
G = F + ["addr_missing", "miss_cos_word", "miss_cos_char", "miss_cos_sk"]
X = u.select(G).to_numpy(); y = u["y"].to_numpy()
idx = np.random.default_rng(0).choice(len(u), size=3_000_000, replace=False)
lr = LogisticRegression(max_iter=1000).fit(X[idx], y[idx]); w = lr.coef_[0]
s = X @ w.astype(np.float32)
d = u.select("s1_idx", "src", "y").with_columns(pl.Series("s", s)).with_columns(pl.col("s").rank("ordinal", descending=True).over("s1_idx", "src").alias("r"))
res = {"weights": dict(zip(G, map(float, w))), "intercept": float(lr.intercept_[0]),
       "recall": {k: round(int(d.filter((pl.col("r") <= k) & pl.col("y")).height) / tot, 5) for k in (5, 10, 15, 20, 30)}}
nS1 = u["s1_idx"].n_unique()
res["curve_k15"] = []
for p in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
    t = np.log(p / (1 - p)) - lr.intercept_[0]
    m = d.filter((pl.col("r") <= 15) & (pl.col("s") >= t))
    res["curve_k15"].append((p, round(float(t), 3), round(int(m["y"].sum()) / tot, 5), round(m.height / nS1, 3)))
print(json.dumps(res, indent=1))
json.dump(res, open('/workspace/amlc/work/tmp/tune_eval2.json', 'w'), indent=1)
