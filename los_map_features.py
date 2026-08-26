from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
import r2_pipeline as rp

def load_mesh():
 lines=(rp.ROOT/'Round2_Map.ply').read_text().splitlines();e=lines.index('end_header');n=int(next(x for x in lines[:e] if x.startswith('element vertex')).split()[-1]);nf=int(next(x for x in lines[:e] if x.startswith('element face')).split()[-1]);v=np.loadtxt(lines[e+1:e+1+n]);faces=[]
 for x in lines[e+1+n:e+1+n+nf]:
  q=list(map(int,x.split()));faces.append(q[1:1+q[0]])
 return v,faces

def cross(a,b,c,d):return (b[0]-a[0])*(d[1]-c[1])-(b[1]-a[1])*(d[0]-c[0])

def run():
 v,faces=load_mesh();edges=set()
 for f in faces:
  for i in range(len(f)):
   a,b=sorted((f[i],f[(i+1)%len(f)]));
   if np.linalg.norm(v[a,:2]-v[b,:2])>.25:edges.add((a,b))
 seg=np.array([[*v[a,:2],*v[b,:2],max(v[a,2],v[b,2])] for a,b in edges]);pos=np.r_[np.load(rp.ROOT/'Round2_Train_Pos.npy'),np.load(rp.ROOT/'Round2_Test_Pos.npy')];bs=np.array([[-18.413,-65.881],[52.,35.]])
 out=[]
 for pi,p in enumerate(pos):
  row=[]
  for base in bs:
   lo=np.minimum(base,p[:2]);hi=np.maximum(base,p[:2]);m=(seg[:,0]<=hi[0])&(seg[:,2]>=lo[0])&(seg[:,1]<=hi[1])&(seg[:,3]>=lo[1])&(seg[:,4]>=1.5);s=seg[m];hits=[]
   r=p[:2]-base
   for z in s:
    a=z[:2]-base;b=z[2:4]-z[:2];den=r[0]*b[1]-r[1]*b[0]
    if abs(den)<1e-9:continue
    t=(a[0]*b[1]-a[1]*b[0])/den;u=(a[0]*r[1]-a[1]*r[0])/den
    if 0<t<1 and 0<=u<=1:hits.append((t,z[4]))
   # De-duplicate shared triangle edges at essentially identical ray distance.
   hits.sort();uniq=[]
   for h in hits:
    if not uniq or abs(h[0]-uniq[-1][0])>1e-3:uniq.append(h)
   row += [len(uniq),min([h[0] for h in uniq],default=1.),max([h[1] for h in uniq],default=0.),sum(h[1]>3 for h in uniq)]
  out.append(row)
  if (pi+1)%500==0:print(pi+1,flush=True)
 out=np.asarray(out,float);np.save(rp.ROOT/'los_map_features.npy',out);print(out.shape,np.quantile(out,[0,.25,.5,.75,1],axis=0))
if __name__=='__main__':run()
