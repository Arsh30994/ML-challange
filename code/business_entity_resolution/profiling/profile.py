import argparse, duckdb, json, time, os, re, shutil, subprocess
ap=argparse.ArgumentParser(description='Read-only data profile of the AMLC 2026 entity-resolution dataset')
ap.add_argument('--data-dir',default='dataset',help='directory containing train/ and test/ (default: %(default)s)')
ap.add_argument('--output-dir',default='reports/data_quality',help='where profile.json is written (default: %(default)s)')
ap.add_argument('--tmp-dir',default=None,help='duckdb database + spill directory (default: <output-dir>/tmp)')
ap.add_argument('--validator',default=None,help='path to validate_submission.py (default: <data-dir>/../validate_submission.py)')
ap.add_argument('--memory-limit',default='3GB'); ap.add_argument('--threads',type=int,default=4)
ap.add_argument('--keep-db',action='store_true',help='keep the temporary duckdb file afterwards')
A=ap.parse_args()
T0=time.time()
D=os.path.abspath(A.data_dir); P=os.path.abspath(A.output_dir)
TMP=os.path.abspath(A.tmp_dir or os.path.join(P,'tmp'))
VAL=os.path.abspath(A.validator or os.path.join(D,os.pardir,'validate_submission.py'))
os.makedirs(P,exist_ok=True); os.makedirs(os.path.join(TMP,'spill'),exist_ok=True)
DB=os.path.join(TMP,'prof.duckdb')
if os.path.exists(DB): os.remove(DB)
c=duckdb.connect(DB)
c.execute(f"SET memory_limit='{A.memory_limit}'; SET temp_directory='{os.path.join(TMP,'spill')}'; SET threads={A.threads}; SET preserve_insertion_order=false;")
# scripts measured in business_name (RE2 Unicode script names)
SCRIPTS=['Devanagari','Bengali','Gurmukhi','Gujarati','Oriya','Tamil','Telugu','Kannada','Malayalam',
         'Han','Arabic','Cyrillic','Hangul','Hiragana','Katakana','Greek','Thai','Hebrew']
NONLATIN=r"regexp_matches(business_name,'[^\p{Latin}\P{L}]')"
ANY_LISTED="regexp_matches(business_name,'"+'|'.join(r'\p{%s}'%s for s in SCRIPTS)+"')"
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
    R['wc_l_lines_incl_header'][t]=int(subprocess.run(['wc','-l',p],capture_output=True,text=True,check=True).stdout.split()[0])
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
        # (a) overlapping "contains" counts: a name with two scripts counts in both
        cont={}
        for scr in SCRIPTS:
            k=q1(rf"SELECT count(*) FROM {t} WHERE regexp_matches(business_name,'\p{{{scr}}}')")[0]
            if k: cont[scr]=k
        cont['other_nonlatin']=q1(f"SELECT count(*) FROM {t} WHERE {NONLATIN} AND NOT {ANY_LISTED}")[0]
        info['name_script_contains_counts']={'note':'overlapping: a name containing several scripts is counted once per script; other_nonlatin = non-Latin names containing none of the listed scripts','counts':cont}
        # (b) exclusive primary-script buckets: each non-Latin name assigned to the listed script with most chars (ties -> list order), else other_nonlatin
        cnts=','.join(rf"length(business_name)-length(regexp_replace(business_name,'\p{{{s}}}','','g'))" for s in SCRIPTS)
        lab="['"+"','".join(SCRIPTS)+"']"
        rows=q(f"""SELECT CASE WHEN list_max(cs)=0 THEN 'other_nonlatin' ELSE {lab}[list_position(cs,list_max(cs))] END b, count(*)
                  FROM (SELECT [{cnts}] cs FROM {t} WHERE {NONLATIN}) GROUP BY 1""")
        prim={s:0 for s in SCRIPTS}; prim['other_nonlatin']=0
        for b_,k in rows: prim[b_]=k
        prim={k:v for k,v in prim.items() if v or k=='other_nonlatin'}
        info['name_script_primary_counts']={'note':'exclusive: each non-Latin name counted once, under the listed script with the most characters (ties broken by list order); other_nonlatin = none of the listed scripts',
            'scripts_measured':SCRIPTS,'counts':prim,'sum':sum(prim.values()),'nonlatin_total':nl,'sum_equals_nonlatin_total':sum(prim.values())==nl}
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
# deterministic sample: the 500k pairs with the smallest hash(s1,mid)
c.execute("CREATE TABLE samp AS SELECT s1,mid FROM (SELECT DISTINCT s1,mid FROM gtp) ORDER BY hash(s1,mid), s1, mid LIMIT 500000")
c.execute("""CREATE TABLE sj AS SELECT p.s1, p.mid, a.country c1, coalesce(b2.country,b3.country) c2, (b2.entity_id IS NOT NULL OR b3.entity_id IS NOT NULL) found
 FROM samp p JOIN (SELECT DISTINCT ON (entity_id) entity_id,country FROM train_s1) a ON a.entity_id=p.s1
 LEFT JOIN (SELECT DISTINCT ON (entity_id) entity_id,country FROM train_s2) b2 ON b2.entity_id=p.mid
 LEFT JOIN (SELECT DISTINCT ON (entity_id) entity_id,country FROM train_s3) b3 ON b3.entity_id=p.mid""")
r=q1("""SELECT count(*), count(*) FILTER (WHERE found), count(*) FILTER (WHERE found AND c1 IS NOT NULL AND c2 IS NOT NULL),
  count(*) FILTER (WHERE found AND c1=c2), count(*) FILTER (WHERE found AND lower(trim(c1))=lower(trim(c2))),
  count(*) FILTER (WHERE found AND starts_with(mid,'S2-') AND c1 IS NOT NULL AND c2 IS NOT NULL), count(*) FILTER (WHERE starts_with(mid,'S2-') AND c1=c2),
  count(*) FILTER (WHERE found AND starts_with(mid,'S3-') AND c1 IS NOT NULL AND c2 IS NOT NULL), count(*) FILTER (WHERE starts_with(mid,'S3-') AND c1=c2) FROM sj""")
g['country_agreement_sample']={'method':'deterministic: 500000 distinct (s1,mid) pairs with smallest hash(s1,mid)','sampled_pairs':r[0],'pairs_with_both_records_found':r[1],'both_country_nonnull':r[2],'exact_equal':r[3],
  'exact_rate_of_both_nonnull':round(r[3]/r[2],6),'case_trim_insensitive_equal':r[4],'ci_rate':round(r[4]/r[2],6),
  'S2_rate':round(r[6]/r[5],6) if r[5] else None,'S3_rate':round(r[8]/r[7],6) if r[7] else None,
  'top_disagreeing_country_pairs':[list(x) for x in q("SELECT c1,c2,count(*) FROM sj WHERE found AND c1<>c2 GROUP BY 1,2 ORDER BY 3 DESC LIMIT 15")]}
R['ground_truth']=g
# non-Latin names by country (S2/S3)
nbc={}
for t in [x for x in files if not x.endswith('s1')]:
    r=q(rf"""SELECT country, count(*), count(*) FILTER (WHERE {NONLATIN}),
      count(*) FILTER (WHERE {NONLATIN} AND NOT {ANY_LISTED})
      FROM {t} GROUP BY 1 ORDER BY 1""")
    nbc[t]={k:{'rows':n,'nonlatin':x,'nonlatin_rate':round(x/n,6),'nonlatin_other_script':o} for k,n,x,o in r}
R['nonlatin_by_country']=nbc
# scoring script facts (read-only)
vs={'path':VAL,'exists':os.path.isfile(VAL)}
if vs['exists']:
    v=open(VAL,encoding='utf-8').read(); vl=v.lower()
    vs['mentions']={k:(k in vl) for k in ['f0.5','f_0.5','fbeta','f-beta','precision','recall','singleton','ground_truth','ground truth']}
    vs['computes_score']=any(k in vl for k in ['f0.5','fbeta','precision','recall'])
    vs['reads_ground_truth_file']='ground_truth.tsv' in vl
    for k in ['MATCHING_HEADER','CANDIDATE_HEADER','DELIM']:
        m=re.search(rf'^{k}\s*=\s*(.+)$',v,re.M); vs[k]=m.group(1).strip() if m else None
    vs['rules']=[
     'Format validator only: never reads the ground truth and never computes a score',
     'TAB-separated; header exactly source1_entity_id<TAB>matched_entity_ids (case-insensitive after strip)',
     'Exactly one row per test_source1 S1 ID: none missing, none extra, no duplicate rows',
     'Empty matched list = no match (singleton); IDs comma-separated',
     'Only S2-/S3- prefixed IDs; no S1 self-matches; no duplicate IDs within a list; file must be UTF-8',
     '--check-ids (optional) flags IDs not in test S2/S3; docstring: a nonexistent ID only lowers the score, never rejects',
     'candidate_pairs.tsv (header source1_entity_id<TAB>candidate_entity_ids) optional here, same checks; warning (never failure) if matches are not a subset of candidates; docstring says it is expected in the final zip']
R['scoring_script']=vs
R['params']={'data_dir':D,'output_dir':P,'tmp_dir':TMP,'memory_limit':A.memory_limit,'threads':A.threads}
c.close()
if not A.keep_db:
    os.remove(DB); shutil.rmtree(os.path.join(TMP,'spill'),ignore_errors=True)
R['timing_s']=timing; R['runtime_s']=round(time.time()-T0,1)
R['duckdb_version']=duckdb.__version__
json.dump(R,open(os.path.join(P,'profile.json'),'w'),indent=1,ensure_ascii=False)
print('TOTAL',R['runtime_s'])
