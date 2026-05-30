import json, glob

LANGS=["ar","cs","de","es"]
METHODS=["mean","tsv","actmat","isoc","wudi","actmat_gd"]
cols=["mmlu_ar","mmlu_de","mmlu_es","mrb_ar","mrb_cs","mrb_de","mrb_es","mgsm_de","mgsm_es"]

def sc(r,t,ks):
    v=r.get(t)
    if not v: return None
    for k in ks:
        if k in v and isinstance(v[k],(int,float)): return v[k]
    return None
def row_from(p):
    r=json.load(open(p))["results"]
    g=lambda l:sc(r,f"global_mmlu_lite:{l}|0",["acc"])
    w=lambda l:sc(r,f"mrewardbench_mcf:{l}|0",["weighted_acc"])
    m=lambda l:sc(r,f"mgsm_custom:{l}|5",["extractive_match"])
    return {"mmlu_ar":g("ar"),"mmlu_de":g("de"),"mmlu_es":g("es"),
            "mrb_ar":w("ar"),"mrb_cs":w("cs"),"mrb_de":w("de"),"mrb_es":w("es"),
            "mgsm_de":m("de"),"mgsm_es":m("es")}
def latest(pat):
    js=glob.glob(pat,recursive=True); return sorted(js)[-1] if js else None

rows=[]
# base
base={c:None for c in cols}
for f in glob.glob("artifacts/results-polyglot-all/base-Olmo-3-1025-7B/**/results_*.json",recursive=True)+glob.glob("polyglot-teachers/lighteval-results/allenai___Olmo-3-1025-7B/results_*.json"):
    for k,v in row_from(f).items():
        if v is not None: base[k]=v
rows.append(("base",base))
# experts (only own-language cols are meaningful)
for lg in LANGS:
    p=latest(f"artifacts/results-polyglot-all/expert-{lg}/**/results_*.json")
    rows.append((f"expert-{lg}", row_from(p) if p else None))
# merges
for me in METHODS:
    p=latest(f"artifacts/results-polyglot-all/Olmo-3-7b-polyglot-all-{me}/**/results_*.json")
    rows.append((f"merge-{me}", row_from(p) if p else None))

W=9
hdr="model".ljust(13)+"".join(c.rjust(W) for c in cols)+"AVG".rjust(W)
print(hdr); print("-"*len(hdr))
for name,r in rows:
    if r is None:
        print(name.ljust(13)+"".join("pending".rjust(W) for _ in cols)+"pending".rjust(W)); continue
    vals=[r[c] for c in cols]; pres=[v for v in vals if v is not None]
    avg=sum(pres)/len(pres) if pres else None
    cells="".join(("—" if v is None else f"{v:.3f}").rjust(W) for v in vals)
    print(name.ljust(13)+cells+(("—" if avg is None else f"{avg:.3f}").rjust(W)))
