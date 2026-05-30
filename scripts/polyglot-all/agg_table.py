import json, glob, os, re, subprocess

LANGS=["ar","cs","de","es"]
METHODS=["mean","tsv","isoc","wudi","actmat","actmat_gd","ties","dare_ties"]
# benchmark -> langs (number-only MGSM dropped; CMGSM = CoT MGSM is the math benchmark)
GROUPS=[("mmlu",["ar","de","es"]),("mrb",["ar","cs","de","es"]),("cmgsm",["de","es"])]
GLABEL={"mmlu":"MMLU","mrb":"MRB","cmgsm":"CMGSM"}
taskcols=[f"{b}_{l}" for b,ls in GROUPS for l in ls]
avgcols=["avg_mmlu","avg_mrb","avg_cmgsm","avg_all"]
HDR={f"{b}_{l}":f"{GLABEL[b]}·{l}" for b,ls in GROUPS for l in ls}
HDR.update({"avg_mmlu":"MMLU","avg_mrb":"MRB","avg_cmgsm":"CMGSM","avg_all":"AVG"})
LOGD="artifacts/logs"

def scv(res,t,ks):
    v=res.get(t)
    if not v: return None
    for k in ks:
        if k in v and isinstance(v[k],(int,float)): return v[k]
    return None
def row_from(p):
    r=json.load(open(p))["results"]
    d={}
    for l in LANGS:
        d[f"mmlu_{l}"]=scv(r,f"global_mmlu_lite:{l}|0",["acc"])
        d[f"mrb_{l}"] =scv(r,f"mrewardbench_mcf:{l}|0",["weighted_acc"])
    return d
def latest(pat):
    js=glob.glob(pat,recursive=True); return sorted(js)[-1] if js else None
def cot_mgsm(src,lang):
    p=latest(f"artifacts/results-polyglot-all-mgsmcot/{src}/mgsm/**/results_*.json")
    if not p: return None
    r=json.load(open(p))["results"]
    return scv(r,f"mgsm_native_cot_{lang}",["exact_match,strict-match","exact_match,flexible-extract"])
def cot_meta(name):
    if name=="base·new": return ("base",{"de","es"})
    if name in ("expert-de","expert-es"): return (name,{name.split("-")[1]})
    if name.startswith("merge-"): return (name,{"de","es"})
    return (None,set())

def active_jobs():
    out={}
    try:
        q=subprocess.run(["squeue","-u",os.environ.get("USER",""),"-h","-o","%i|%j|%t"],
                         capture_output=True,text=True,timeout=15).stdout
    except Exception: return out
    for line in q.splitlines():
        try: jid,name,state=line.split("|")
        except ValueError: continue
        if not name.startswith("polyglot_all"): continue
        log=os.path.join(LOGD,f"{name}_{jid}.out"); txt=open(log).read() if os.path.exists(log) else ""
        m=re.search(r"with '([\w]+)'",txt); e=re.search(r"Expert ([a-z]{2}):",txt)
        key=None
        if "_merge_" in name and m: key=f"merge-{m.group(1)}"
        elif "_expert_" in name and e: key=f"expert-{e.group(1)}"
        if key is None and "_merge_" in name and "_" in jid:
            idx=int(jid.split("_")[1]); key=f"merge-{METHODS[idx]}" if idx<len(METHODS) else None
        if key: out[key]=f"{jid} ({state})"
    return out

rows=[]
def merge_files(files):
    d={c:None for c in taskcols}
    for f in files:
        for k,v in row_from(f).items():
            if v is not None: d[k]=v
    return d if files else None
newf=glob.glob("artifacts/results-polyglot-all/base-Olmo-3-1025-7B/**/results_*.json",recursive=True)
oldf=glob.glob("polyglot-teachers/lighteval-results/allenai___Olmo-3-1025-7B/results_*.json")
rows.append(("base","base·new",merge_files(newf)))
rows.append(("base","base·old",merge_files(oldf)))
for lg in LANGS:
    p=latest(f"artifacts/results-polyglot-all/expert-{lg}/**/results_*.json")
    rows.append(("expert",f"expert-{lg}", row_from(p) if p else None))
for me in METHODS:
    p=latest(f"artifacts/results-polyglot-all/Olmo-3-7b-polyglot-all-{me}/**/results_*.json")
    rows.append(("merge",f"merge-{me}", row_from(p) if p else None))

COT_RUNNING = subprocess.run(["bash","-c","squeue -u $USER -h -o %j 2>/dev/null | grep -c polyglot_all_mgsmcot || true"],
                             capture_output=True,text=True).stdout.strip() not in ("","0")
for g,name,r in rows:
    if r is None: continue
    src,langs=cot_meta(name)
    for lang in ["de","es"]:
        key=f"cmgsm_{lang}"
        if src and lang in langs:
            v=cot_mgsm(src,lang)
            r[key]= v if v is not None else ("PENDING" if COT_RUNNING else None)
        else:
            r[key]=None

def gmean(r,b,ls):
    vals=[r.get(f"{b}_{l}") for l in ls]
    nums=[v for v in vals if isinstance(v,(int,float))]
    if nums: return sum(nums)/len(nums)
    if any(v=="PENDING" for v in vals): return "PENDING"
    return None
for g,name,r in rows:
    if r is None: continue
    r["avg_mmlu"]=gmean(r,"mmlu",["ar","de","es"])
    r["avg_mrb"] =gmean(r,"mrb",["ar","cs","de","es"])
    r["avg_cmgsm"]=gmean(r,"cmgsm",["de","es"])
    parts=[r[k] for k in ("avg_mmlu","avg_mrb","avg_cmgsm") if isinstance(r[k],(int,float))]
    r["avg_all"]=sum(parts)/len(parts) if parts else ("PENDING" if r.get("avg_cmgsm")=="PENDING" else None)

# drop task cols with no data anywhere
taskcols=[c for c in taskcols if any(isinstance(r.get(c),(int,float)) for _,_,r in rows if r)]
allnum=taskcols+avgcols
jobs=active_jobs()
def rank1_2(c):
    vals=sorted({round(r[c],3) for g,_,r in rows if g=="merge" and r and isinstance(r.get(c),(int,float))},reverse=True)
    return (vals[0] if vals else None, vals[1] if len(vals)>1 else None)
RANK={c:rank1_2(c) for c in allnum}
champ=None; best_avg=-1
for g,name,r in rows:
    if g=="merge" and r and isinstance(r.get("avg_all"),(int,float)) and r["avg_all"]>best_avg:
        best_avg=r["avg_all"]; champ=name
def cell(c,v,is_merge):
    if v is None: return "·"
    if v=="PENDING": return "⏳"
    s=f"{v:.3f}"
    if is_merge:
        b1,b2=RANK[c]; vr=round(v,3)
        if b1 is not None and vr==b1: return f"{s}\\*"
        if b2 is not None and vr==b2: return f"{s}\\*\\*"
    return s

hdr_task=" | ".join(HDR[c] for c in taskcols)
hdr_avg=" | ".join(HDR[c] for c in avgcols)
print(f"| model | {hdr_task} | {hdr_avg} | job |")
print("|"+"---|"*(len(allnum)+2))
last=None
for g,name,r in rows:
    if last and g!=last: print("| |"+" |"*(len(allnum))+" |")
    last=g
    jb=jobs.get(name,"")
    if r is None:
        print(f"| `{name}` | "+" | ".join("⏳" for _ in allnum)+f" | {jb or '—'} |"); continue
    label=f"`{name}` 🥇" if name==champ else f"`{name}`"
    print(f"| {label} | "+" | ".join(cell(c,r.get(c),g=='merge') for c in allnum)+f" | {jb} |")
