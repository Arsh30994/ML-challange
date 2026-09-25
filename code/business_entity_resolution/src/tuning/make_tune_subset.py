# train-side tuning subset: 25% of train S1 (seeded) + their matches + 25% of train-side unreferenced
import sys, time
sys.path.insert(0, '/workspace/amlc/gt_checks/code')
import numpy as np, polars as pl
from business_entity_resolution.src.loader import load_source, load_ground_truth, explode_ground_truth
from business_entity_resolution.src.prepare import normalize_frame
D='/workspace/amlc/dataset/dataset'
gt=load_ground_truth(D)
s1sp=pl.read_parquet('/workspace/amlc/work/split/s1_split.parquet').filter(pl.col('side')=='train')
sel=s1sp.filter(pl.col('entity_id').hash(7) % 4 == 0).select('entity_id')
gts=gt.join(sel.rename({'entity_id':'source1_entity_id'}), on='source1_entity_id', how='semi')
gts.write_parquet('/workspace/amlc/work/tune_gt.parquet')
matched=explode_ground_truth(gt).select('matched_id')
msel=explode_ground_truth(gts).select('matched_id')
for n in (1,2,3):
    df=load_source(D,'train',n)
    sp_=pl.read_parquet(f'/workspace/amlc/work/split/s{n}_split.parquet').filter(pl.col('side')=='train')
    df=df.join(sp_.select('entity_id'), on='entity_id', how='semi')
    if n==1:
        df=df.join(sel, on='entity_id', how='semi')
    else:
        unref=df.join(matched.rename({'matched_id':'entity_id'}), on='entity_id', how='anti').filter(pl.col('entity_id').hash(7)%4==0)
        df=pl.concat([df.join(msel.rename({'matched_id':'entity_id'}), on='entity_id', how='semi'), unref])
    normalize_frame(df).write_parquet(f'/workspace/amlc/work/norm_tune_s{n}.parquet'); print(n, df.height, flush=True)
