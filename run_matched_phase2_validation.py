from __future__ import annotations

import concurrent.futures,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import r2_pipeline as rp
from run_matched_metric_validation import CLEAR as BASE_CLEAR,COMMON,HybridView

ROOT=Path(__file__).resolve().parent;PYTHON=Path(sys.executable)
PROFILES={5:{"R2_MATCHED_JOINT_METRIC_KERNEL_EXTENDED":"1"},7:{"R2_MATCHED_JOINT_METRIC_KERNEL_MAP":"1"}}
CLEAR=set(BASE_CLEAR)|{"R2_MATCHED_PHASE_SAFE","R2_FAST_VALIDATE","R2_SKIP_LATEST_PRED","R2_PAS_POOL_OVERRIDE"}


def predict(g,f):
 out=ROOT/f"matched_phase2_g{g}_fold{f}.npy";labels=np.load(ROOT/f"matched_rect_groups_{f}.npy");expected=int(np.sum(labels==g))
 if out.exists():
  a=np.load(out,mmap_mode="r")
  if a.shape==(expected,256,4,192):print(json.dumps({"group":g,"fold":f,"cached":True}),flush=True);return
 env=os.environ.copy()
 for k in CLEAR:env.pop(k,None)
 env.update(COMMON);env.update(PROFILES[g]);env.update({"R2_MATCHED_PHASE_SAFE":"1","R2_ONLY_GROUP":str(g),"R2_VAL_FILE":f"matched_rect_val_{f}.npy","R2_VAL_GROUP_FILE":f"matched_rect_groups_{f}.npy","R2_SAVE_PRED":out.name,"R2_FAST_VALIDATE":"1","R2_SKIP_LATEST_PRED":"1"})
 subprocess.run([str(PYTHON),"r2_pipeline.py","island-validate"],cwd=ROOT,env=env,text=True,capture_output=True,check=True)
 print(json.dumps({"group":g,"fold":f,"cached":False}),flush=True)


def milestone_path(g,f):
 if g==7:return ROOT/f"matched_map_g7_fold{f}.npy"
 if g==6:return ROOT/f"matched_g6_g6_fold{f}.npy"
 return ROOT/f"matched_extended_g{g}_fold{f}.npy"


def run():
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda x:predict(*x),[(g,f) for f in range(5) for g in PROFILES]))
 _,channel,_=rp.load_data();tg=rp.official_island_labels(np.load(ROOT/"Round2_Test_Pos.npy"));counts=dict(zip(*np.unique(tg,return_counts=True)));records=[]
 for f in range(5):
  val=np.load(ROOT/f"matched_rect_val_{f}.npy");labels=np.load(ROOT/f"matched_rect_groups_{f}.npy");w=np.asarray([counts.get(int(g),0)/max(1,np.sum(labels==g)) for g in labels]);base=np.load(ROOT/f"matched_pred_core_nog10_safe_fold{f}.npy",mmap_mode="r")
  milestone={g:np.load(milestone_path(g,f),mmap_mode="r") for g in (0,1,5,6,7,8)}
  current=dict(milestone);current.update({g:np.load(ROOT/f"matched_phase_g{g}_fold{f}.npy",mmap_mode="r") for g in (3,4,5,6,8,9)})
  phase2=dict(current);phase2.update({g:np.load(ROOT/f"matched_phase2_g{g}_fold{f}.npy",mmap_mode="r") for g in PROFILES})
  m=rp.score_numpy_weighted(HybridView(base,labels,milestone),channel[val],w);c=rp.score_numpy_weighted(HybridView(base,labels,current),channel[val],w);p=rp.score_numpy_weighted(HybridView(base,labels,phase2),channel[val],w)
  row={"fold":f,"milestone":m,"current":c,"phase2":p,"delta_vs_milestone":p["score"]-m["score"],"delta_vs_current":p["score"]-c["score"]};records.append(row);print(json.dumps(row),flush=True)
 summary={"phase2_scores":[x["phase2"]["score"] for x in records],"deltas_vs_milestone":[x["delta_vs_milestone"] for x in records],"deltas_vs_current":[x["delta_vs_current"] for x in records]};summary["mean_delta_vs_milestone"]=float(np.mean(summary["deltas_vs_milestone"]));summary["mean_delta_vs_current"]=float(np.mean(summary["deltas_vs_current"]));summary["min_delta_vs_current"]=float(np.min(summary["deltas_vs_current"]));(ROOT/"matched_phase2_validation.json").write_text(json.dumps({"records":records,"summary":summary},indent=2),encoding="utf-8");print(json.dumps(summary),flush=True)


if __name__=="__main__":run()
