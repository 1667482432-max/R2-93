from __future__ import annotations
import numpy as np,json
from scipy.spatial import cKDTree
import r2_pipeline as rp

def load_vertices():
 lines=(rp.ROOT/'Round2_Map.ply').read_text().splitlines();end=lines.index('end_header');n=int([x for x in lines[:end] if x.startswith('element vertex')][0].split()[-1]);return np.loadtxt(lines[end+1:end+1+n])

def features(points,v):
 xy=points[:,:2];tree=cKDTree(v[:,:2]);out=[]
 for p in xy:
  d,ids=tree.query(p,k=64);near=v[ids];row=[]
  row += list(np.quantile(d,[0,.1,.25,.5,.75,1]));row += list(np.quantile(near[:,2],[0,.25,.5,.75,1]))
  for r in [5,10,20,30,50]:
   ids=tree.query_ball_point(p,r);z=v[ids,2] if ids else np.array([0]);row += [len(ids),z.mean(),z.max(),np.mean(z>2)]
  out.append(row)
 return np.asarray(out)

def run():
 v=load_vertices();p=np.load('Round2_Train_Pos.npy');q=np.load('Round2_Test_Pos.npy');print('vertices',v.shape,v.min(0),v.max(0));f=features(np.r_[p,q],v);np.save('map_features.npy',f);print('features',f.shape,np.nanmin(f,0),np.nanmax(f,0))
if __name__=='__main__':run()
