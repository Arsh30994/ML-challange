import duckdb, json, time, os
T0=time.time()
D='/workspace/amlc/dataset/dataset'
P='/workspace/amlc/profile'
DB=P+'/tmp/prof.duckdb'
if os.path.exists(DB): os.remove(DB)
c=duckdb.connect(DB)
c.execute("SET memory_limit='3GB'; SET temp_directory='/workspace/amlc/profile/tmp/spill'; SET threads=4; SET preserve_insertion_order=false;")
R={}
def q(s): return c.execute(s).fetchall()
def q1(s): return c.execute(s).fetchone()
def rd(path): return f"read_csv('{path}', delim='\t', header=true, quote='', escape='', all_varchar=true, strict_mode=true, null_padding=false)"
files={}
for sp in ['train','test']:
    for s in [1,2,3]:
        files[f'{sp}_s{s}']=f'{D}/{sp}/{sp}_source{s}.tsv'
gt=f'{D}/train/train_ground_truth.tsv'
# load
timing={}
for t,p in files.items():
    t1=time.time()
    c.execute(f"CREATE TABLE {t} AS SELECT * FROM {rd(p)}")
    timing['load_'+t]=round(time.time()-t1,1)
c.execute(f"CREATE TABLE gt AS SELECT * FROM {rd(gt)}")
R['wc_l_lines_incl_header']={}
for t,p in list(files.items())+[('gt',gt)]:
    R['wc_l_lines_incl_header'][t]=int(os.popen(f"wc -l < '{p}'").read())
# 1 counts / dups / columns
R['files']={}
for t in list(files)+['gt']:
    cols=[r[0] for r in q(f"DESCRIBE {t}")]
    idc=cols[0]
    n,nd=q1(f"SELECT count(*), count(DISTINCT {idc}) FROM {t}")
    dupids=q1(f"SELECT count(*), coalesce(max(k),0), coalesce(sum(k),0) FROM (SELECT {idc}, count(*) k FROM {t} GROUP BY 1 HAVING k>1)")
    info={'columns':cols,'rows':n,'distinct_ids':nd,'duplicate_id_values':dupids[0],'max_rows_per_dup_id':dupids[1],'rows_involved_in_dups':int(dupids[2]),
          'null_id':q1(f"SELECT count(*) FROM {t} WHERE {idc} IS NULL")[0]}
    if t!='gt':
        pref='S'+t[-1]+'-'
        info['ids_bad_prefix']=q1(f"SELECT count(*) FROM {t} WHERE NOT starts_with(entity_id,'{pref}')")[0]
        info['exact_duplicate_rows']=n-q1(f"SELECT count(*) FROM (SELECT DISTINCT * FROM {t})")[0]
        # null/empty
        ne={}
        for col in cols:
            nn,ee=q1(f"SELECT count(*) FILTER (WHERE {col} IS NULL), count(*) FILTER (WHERE {col} IS NOT NULL AND trim({col})='') FROM {t}")
            ne[col]={'null':nn,'blank_whitespace_only':ee,'null_or_blank_rate':round((nn+ee)/n,6)}
        info['null_empty']=ne
        info['distinct_countries']=q1(f"SELECT count(DISTINCT country) FROM {t}")[0]
        info['country_top15']=[[k,v] for k,v in q(f"SELECT coalesce(country,'<NULL>') k, count(*) v FROM {t} GROUP BY 1 ORDER BY 2 DESC LIMIT 15")]
        nl,dv,nm=q1(rf"""SELECT count(*) FILTER (WHERE regexp_matches(business_name,'[^\p{{Latin}}\P{{L}}]')),
                              count(*) FILTER (WHERE regexp_matches(business_name,'\p{{Devanagari}}')),
                              count(business_name) FROM {t}""")
        info['names_nonlatin_letter']={'count':nl,'rate_of_nonnull_names':round(nl/nm,6)}
        info['names_devanagari']={'count':dv,'rate_of_nonnull_names':round(dv/nm,6)}
        info['name_nonlatin_top_scripts']={}
        for scr in ['Devanagari','Han','Arabic','Cyrillic','Hangul','Hiragana','Katakana','Greek','Thai','Hebrew','Bengali','Tamil','Telugu','Gujarati','Gurmukhi','Kannada','Malayalam']:
            k=q1(rf"SELECT count(*) FROM {t} WHERE regexp_matches(business_name,'\p{{{scr}}}')")[0]
            if k: info['name_nonlatin_top_scripts'][scr]=k
        ls={}
        for col in ['business_name','business_address']:
            r=q1(f"SELECT quantile_disc(length({col}),0.5), quantile_disc(length({col}),0.95), avg(length({col})), max(length({col})), min(length({col})) FROM {t} WHERE {col} IS NOT NULL")
            ls[col]={'median_chars':r[0],'p95_chars':r[1],'mean_chars':round(r[2],2),'max_chars':r[3],'min_chars':r[4],'note':'non-null values only, length in Unicode chars'}
        info['length_stats']=ls
    R['files'][t]=info
    print(t,'done',round(time.time()-T0,1),flush=True)
# cross-split overlap
ov={}
for s in [1,2,3]:
    for a in [1,2,3]:
        k=q1(f"SELECT count(*) FROM (SELECT DISTINCT entity_id FROM train_s{s}) x JOIN (SELECT DISTINCT entity_id FROM test_s{a}) y USING(entity_id)")[0]
        ov[f'train_s{s}&test_s{a}']=k
ov['within_train_cross_source']={f's{a}&s{b}':q1(f"SELECT count(*) FROM (SELECT DISTINCT entity_id FROM train_s{a}) JOIN (SELECT DISTINCT entity_id FROM train_s{b}) USING(entity_id)")[0] for a,b in [(1,2),(1,3),(2,3)]}
R['train_test_id_overlap']=ov
# 4 GT
c.execute("""CREATE TABLE gtp AS SELECT source1_entity_id s1, trim(m) mid FROM
  (SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) m FROM gt WHERE matched_entity_ids IS NOT NULL AND trim(matched_entity_ids)<>'')""")
g={}
g['gt_rows']=q1("SELECT count(*) FROM gt")[0]
g['empty_token_in_lists']=q1("SELECT count(*) FROM gtp WHERE mid=''")[0]
c.execute("DELETE FROM gtp WHERE mid=''")
g['token_with_whitespace']=q1("SELECT count(*) FROM gtp WHERE mid<>regexp_replace(mid,'\\s','','g')")[0]
g['intra_list_duplicate_ids']=q1("SELECT count(*)-count(DISTINCT (s1,mid)) FROM gtp")[0]
g['bad_prefix_tokens']=q1("SELECT count(*) FROM gtp WHERE NOT (starts_with(mid,'S2-') OR starts_with(mid,'S3-'))")[0]
c.execute("CREATE TABLE cnt AS SELECT g.source1_entity_id s1, count(DISTINCT p.mid) n, count(DISTINCT p.mid) FILTER (WHERE starts_with(p.mid,'S2-')) n2, count(DISTINCT p.mid) FILTER (WHERE starts_with(p.mid,'S3-')) n3 FROM gt g LEFT JOIN gtp p ON g.source1_entity_id=p.s1 GROUP BY 1")
N=q1("SELECT count(*) FROM cnt")[0]
sing=q1("SELECT count(*) FROM cnt WHERE n=0")[0]
g['distinct_s1_in_gt']=N; g['singletons']=sing; g['singleton_rate']=round(sing/N,6)
dist={}
for lab,cond in [('0','n=0'),('1','n=1'),('2','n=2'),('3','n=3'),('4','n=4'),('5','n=5'),('6-10','n BETWEEN 6 AND 10'),('>10','n>10')]:
    k=q1(f"SELECT count(*) FROM cnt WHERE {cond}")[0]; dist[lab]=[k,round(k/N,6)]
g['matches_per_s1_distribution_[count,frac]']=dist
m=q1("SELECT avg(n), max(n), avg(n) FILTER (WHERE n>0), median(n) FILTER (WHERE n>0) FROM cnt")
g['mean_matches_all_s1']=round(m[0],4); g['max_matches']=m[1]; g['mean_matches_nonsingleton']=round(m[2],4); g['median_matches_nonsingleton']=m[3]
t2,t3=q1("SELECT count(*) FILTER (WHERE starts_with(mid,'S2-')), count(*) FILTER (WHERE starts_with(mid,'S3-')) FROM (SELECT DISTINCT s1,mid FROM gtp)")
g['total_pairs']=t2+t3; g['pairs_S2']=t2; g['pairs_S3']=t3; g['frac_S2']=round(t2/(t2+t3),6)
g['nonsingleton_s1_with']={'S2_only':q1("SELECT count(*) FROM cnt WHERE n2>0 AND n3=0")[0],'S3_only':q1("SELECT count(*) FROM cnt WHERE n3>0 AND n2=0")[0],'both':q1("SELECT count(*) FROM cnt WHERE n2>0 AND n3>0")[0]}
g['per_s1_S2_count_dist']=[list(r) for r in q("SELECT n2, count(*) FROM cnt WHERE n>0 GROUP BY 1 ORDER BY 1 LIMIT 12")]
g['per_s1_S3_count_dist']=[list(r) for r in q("SELECT n3, count(*) FROM cnt WHERE n>0 GROUP BY 1 ORDER BY 1 LIMIT 12")]
# 5 multiplicity
c.execute("CREATE TABLE mult AS SELECT mid, count(DISTINCT s1) k FROM gtp GROUP BY 1")
r=q1("SELECT count(*), count(*) FILTER (WHERE k>1), max(k), count(*) FILTER (WHERE k>1 AND starts_with(mid,'S2-')), count(*) FILTER (WHERE k>1 AND starts_with(mid,'S3-')) FROM mult")
g['check_a_multi_parent']={'distinct_matched_ids':r[0],'ids_under_more_than_one_s1':r[1],'max_multiplicity':r[2],'S2':r[3],'S3':r[4],
   'multiplicity_distribution':[list(x) for x in q("SELECT k, count(*) FROM mult GROUP BY 1 ORDER BY 1 LIMIT 15")]}
# 6 unreferenced + orphans
b={}
for s in [2,3]:
    tot=q1(f"SELECT count(DISTINCT entity_id) FROM train_s{s}")[0]
    un=q1(f"SELECT count(*) FROM (SELECT DISTINCT entity_id FROM train_s{s}) a ANTI JOIN mult m ON a.entity_id=m.mid")[0]
    orph=q1(f"SELECT count(*) FROM mult m ANTI JOIN train_s{s} a ON a.entity_id=m.mid WHERE starts_with(m.mid,'S{s}-')")[0]
    orph_test=q1(f"SELECT count(*) FROM (SELECT mid FROM mult m ANTI JOIN train_s{s} a ON a.entity_id=m.mid WHERE starts_with(m.mid,'S{s}-')) o SEMI JOIN test_s{s} t ON t.entity_id=o.mid")[0]
    b[f'S{s}']={'train_distinct_ids':tot,'never_in_gt':un,'never_in_gt_pct':round(100*un/tot,4),'gt_ids_missing_from_train_source':orph,'of_those_found_in_test_source':orph_test}
g['check_b_unreferenced_and_orphans']=b
# 7 S1 set equality
g['s1_set']={'gt_s1_not_in_train_s1':q1("SELECT count(*) FROM (SELECT DISTINCT source1_entity_id id FROM gt) ANTI JOIN train_s1 ON id=entity_id")[0],
  'train_s1_not_in_gt':q1("SELECT count(*) FROM (SELECT DISTINCT entity_id id FROM train_s1) ANTI JOIN gt ON id=source1_entity_id")[0]}
g['s1_set']['identical']= g['s1_set']['gt_s1_not_in_train_s1']==0 and g['s1_set']['train_s1_not_in_gt']==0
# 8 country agreement
c.execute("SELECT setseed(0.42)")
c.execute("CREATE TABLE samp AS SELECT * FROM (SELECT DISTINCT s1,mid FROM gtp) USING SAMPLE reservoir(500000 ROWS) REPEATABLE (42)")
c.execute("""CREATE TABLE sj AS SELECT p.s1, p.mid, a.country c1, coalesce(b2.country,b3.country) c2, (b2.entity_id IS NOT NULL OR b3.entity_id IS NOT NULL) found
 FROM samp p JOIN (SELECT DISTINCT ON (entity_id) entity_id,country FROM train_s1) a ON a.entity_id=p.s1
 LEFT JOIN (SELECT DISTINCT ON (entity_id) entity_id,country FROM train_s2) b2 ON b2.entity_id=p.mid
 LEFT JOIN (SELECT DISTINCT ON (entity_id) entity_id,country FROM train_s3) b3 ON b3.entity_id=p.mid""")
r=q1("""SELECT count(*), count(*) FILTER (WHERE found), count(*) FILTER (WHERE found AND c1 IS NOT NULL AND c2 IS NOT NULL),
  count(*) FILTER (WHERE found AND c1=c2), count(*) FILTER (WHERE found AND lower(trim(c1))=lower(trim(c2))),
  count(*) FILTER (WHERE found AND starts_with(mid,'S2-') AND c1 IS NOT NULL AND c2 IS NOT NULL), count(*) FILTER (WHERE starts_with(mid,'S2-') AND c1=c2),
  count(*) FILTER (WHERE found AND starts_with(mid,'S3-') AND c1 IS NOT NULL AND c2 IS NOT NULL), count(*) FILTER (WHERE starts_with(mid,'S3-') AND c1=c2) FROM sj""")
g['country_agreement_sample']={'sampled_pairs':r[0],'pairs_with_both_records_found':r[1],'both_country_nonnull':r[2],'exact_equal':r[3],
  'exact_rate_of_both_nonnull':round(r[3]/r[2],6),'case_trim_insensitive_equal':r[4],'ci_rate':round(r[4]/r[2],6),
  'S2_rate':round(r[6]/r[5],6) if r[5] else None,'S3_rate':round(r[8]/r[7],6) if r[7] else None,
  'top_disagreeing_country_pairs':[list(x) for x in q("SELECT c1,c2,count(*) FROM sj WHERE found AND c1<>c2 GROUP BY 1,2 ORDER BY 3 DESC LIMIT 15")]}
R['ground_truth']=g
R['timing_s']=timing; R['runtime_s']=round(time.time()-T0,1)
R['duckdb_version']=duckdb.__version__
json.dump(R,open(P+'/profile.json','w'),indent=1,ensure_ascii=False)
print('TOTAL',R['runtime_s'])
