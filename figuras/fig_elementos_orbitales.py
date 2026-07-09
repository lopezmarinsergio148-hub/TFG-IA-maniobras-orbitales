# Diagrama de los elementos orbitales clásicos (para 2.2), estilo diagrama clásico:
# dos planos sombreados + ángulos grandes (con sus radios) + leyenda.
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

BG="#0a0a1a"
Om,inc,w,nu = np.radians(50), np.radians(38), np.radians(55), np.radians(125)
a,e = 1.0, 0.5
NU="#5df08a"; OM="#36d0e6"; WW="#c8a8ff"; II="#ff9a3c"

def Rz(t): return np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1]])
def Rx(t): return np.array([[1,0,0],[0,np.cos(t),-np.sin(t)],[0,np.sin(t),np.cos(t)]])
R = Rz(Om)@Rx(inc)@Rz(w)
def pf(th):
    r=a*(1-e**2)/(1+e*np.cos(th)); return np.array([r*np.cos(th),r*np.sin(th),0.0])
def Rpf(th, rr=None):
    v=pf(th);
    if rr is not None: v=rr*np.array([np.cos(th),np.sin(th),0.0])
    return R@v

class Arrow3D(FancyArrowPatch):
    def __init__(s,xs,ys,zs,**k): super().__init__((0,0),(0,0),**k); s._v=(xs,ys,zs)
    def do_3d_projection(s,renderer=None):
        x,y,z=proj3d.proj_transform(*s._v,s.axes.M); s.set_positions((x[0],y[0]),(x[1],y[1])); return np.min(z)

fig=plt.figure(figsize=(10.5,8.2)); ax=fig.add_subplot(111,projection="3d")
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

# Planos
ax.add_collection3d(Poly3DCollection([[(-1.8,-1.4,0),(2.0,-1.4,0),(2.0,1.4,0),(-1.8,1.4,0)]],
                    facecolor="#6f7a8a",alpha=0.16,edgecolor="#9aa6b6",lw=0.8))
orbpl=[R@np.array([x,y,0]) for x,y in [(-1.75,-1.1),(0.8,-1.1),(0.8,1.1),(-1.75,1.1)]]
ax.add_collection3d(Poly3DCollection([orbpl],facecolor="#e8c84a",alpha=0.16,edgecolor="#e8c84a",lw=0.8))

# Órbita (tramo sobre el plano de referencia sólido; bajo el plano, punteado)
th=np.linspace(0,2*np.pi,500); orb=np.array([R@pf(t) for t in th]).T
above=orb.copy(); above[:,orb[2]<0]=np.nan
below=orb.copy(); below[:,orb[2]>=0]=np.nan
ax.plot(above[0],above[1],above[2],color="#ff5a4d",lw=2.6,zorder=10)
ax.plot(below[0],below[1],below[2],color="#ff5a4d",lw=1.6,ls=":",alpha=0.5,zorder=6)
ax.text(orb[0][300],orb[1][300],orb[2][300]-0.13,"Órbita",color="#ff5a4d",fontsize=10,zorder=11)

# Cuerpo central + radios que delimitan los ángulos
ax.scatter([0],[0],[0],color="white",s=80,zorder=13)
ax.text(-0.5,-0.05,0.18,"Cuerpo celeste",color="white",fontsize=10,ha="right",zorder=13)
peri=R@pf(0.0); sat=R@pf(nu); nodo=np.array([np.cos(Om),np.sin(Om),0])
node_in_orb=Rpf(-w, rr=1.0)   # dirección al nodo dentro del plano orbital
ax.plot([0,1.15*nodo[0]],[0,1.15*nodo[1]],[0,0],color="#cfd8e3",lw=1.1,zorder=9)         # radio al nodo (ref)
ax.plot([0,0.95*peri[0]/np.linalg.norm(peri)],[0,0.95*peri[1]/np.linalg.norm(peri)],[0,0.95*peri[2]/np.linalg.norm(peri)],color="#ffb14d",lw=1.0,ls=":",alpha=0.8,zorder=9)  # radio al periapsis
ax.plot([0,sat[0]],[0,sat[1]],[0,sat[2]],color=NU,lw=1.0,alpha=0.7,zorder=9)             # radio al satélite
ax.plot([-1.5*nodo[0],1.85*nodo[0]],[-1.5*nodo[1],1.85*nodo[1]],[0,0],color="#cfd8e3",lw=0.9,ls="--",zorder=8)  # línea de nodos

# Dirección de referencia
ax.add_artist(Arrow3D([0,1.95],[0,0],[0,0],color="#ff7a7a",lw=1.8,arrowstyle="-|>",mutation_scale=16,zorder=9))
ax.text(1.35,0,-0.42,"Dirección de referencia\n(punto Aries)",color="#ff7a7a",fontsize=9.5)

# Puntos. NODO ASCENDENTE = punto donde la órbita cruza el plano subiendo (perifocal -w)
asc=R@pf(-w); desc=R@pf(-w+np.pi)
ax.scatter(*asc,color="#ffffff",s=60,zorder=14)
ax.plot([asc[0],asc[0]+0.45],[asc[1],asc[1]],[asc[2],asc[2]+0.30],color="#cfd8e3",lw=0.7,alpha=0.7,zorder=13)
ax.text(asc[0]+0.48,asc[1],asc[2]+0.34,"Nodo ascendente",color="#cfd8e3",fontsize=9,zorder=14)
ax.scatter(*desc,color="#888",s=28,zorder=10); ax.text(desc[0]-0.05,desc[1],desc[2]-0.26,"nodo descendente",color="#888",fontsize=7.5,ha="right",zorder=10)
ax.scatter(*peri,color="#ffb14d",s=42,zorder=13); ax.text(peri[0]-0.05,peri[1],peri[2]+0.22,"periapsis",color="#ffb14d",fontsize=8.5,ha="right",zorder=13)
ax.scatter(*sat,color=NU,s=58,zorder=13); ax.text(sat[0]-0.05,sat[1],sat[2]+0.10,"satélite",color=NU,fontsize=8.5,ha="right",zorder=13)

# Arcos grandes
def arc_xy(a0,a1,rr): ph=np.linspace(a0,a1,60); return np.array([[rr*np.cos(p),rr*np.sin(p),0] for p in ph]).T
def arc_pf(a0,a1,rr): ph=np.linspace(a0,a1,60); return np.array([R@np.array([rr*np.cos(p),rr*np.sin(p),0]) for p in ph]).T
A=arc_xy(0,Om,0.78); ax.plot(A[0],A[1],A[2],color=OM,lw=2.3,zorder=12); ax.text(0.9*np.cos(Om*0.5),0.9*np.sin(Om*0.5),0.06,r"$\Omega$",color=OM,fontsize=17,zorder=13)
B=arc_pf(-w,0,0.78); ax.plot(B[0],B[1],B[2],color=WW,lw=2.3,zorder=12); bm=R@np.array([0.92*np.cos(-w*0.5),0.92*np.sin(-w*0.5),0]); ax.text(bm[0],bm[1],bm[2]+0.06,r"$\omega$",color=WW,fontsize=17,zorder=13)
C=arc_pf(0,nu,0.52); ax.plot(C[0],C[1],C[2],color=NU,lw=2.3,zorder=12); cm=R@np.array([0.62*np.cos(nu*0.5),0.62*np.sin(nu*0.5),0]); ax.text(cm[0],cm[1],cm[2]+0.06,r"$\nu$",color=NU,fontsize=17,zorder=13)
# Inclinación: cuña en el NODO ASCENDENTE, entre el plano de referencia y la órbita que sube
e_eq=np.array([np.sin(Om),-np.cos(Om),0])                     # en el plano de referencia, perp. al nodo
e_orb=R@np.array([np.cos(-w+np.pi/2),np.sin(-w+np.pi/2),0])   # tangente de la órbita en el nodo (sube)
ci=asc; tt=np.linspace(0,1,30)
ax.plot([ci[0],ci[0]+0.42*e_eq[0]],[ci[1],ci[1]+0.42*e_eq[1]],[ci[2],ci[2]+0.42*e_eq[2]],color=II,lw=0.9,alpha=0.85,zorder=12)
ax.plot([ci[0],ci[0]+0.42*e_orb[0]],[ci[1],ci[1]+0.42*e_orb[1]],[ci[2],ci[2]+0.42*e_orb[2]],color=II,lw=0.9,alpha=0.85,zorder=12)
ai=np.array([ci+0.30*((1-t)*e_eq+t*e_orb)/np.linalg.norm((1-t)*e_eq+t*e_orb) for t in tt]).T
ax.plot(ai[0],ai[1],ai[2],color=II,lw=2.3,zorder=13)
mid=ci+0.40*(e_eq+e_orb)/np.linalg.norm(e_eq+e_orb); ax.text(mid[0]+0.04,mid[1],mid[2]+0.05,r"$i$",color=II,fontsize=16,zorder=14)

# Etiquetas de planos
ax.text(-1.6,-1.2,0.0,"Plano de referencia",color="#aab4c2",fontsize=9)
po=R@np.array([-1.6,0.95,0]); ax.text(po[0],po[1],po[2],"Plano orbital",color="#d8bf55",fontsize=9)

# Leyenda
leg=[(r"$\nu$  anomalía verdadera",NU),(r"$\omega$  argumento de periapsis",WW),
     (r"$\Omega$  longitud del nodo ascendente",OM),(r"$i$   inclinación",II)]
for k,(t,c) in enumerate(leg): ax.text2D(0.015,0.97-0.045*k,t,transform=ax.transAxes,color=c,fontsize=11)

ax.set_box_aspect([1,1,0.8]); ax.set_xlim(-1.7,2.05); ax.set_ylim(-1.5,1.6); ax.set_zlim(-0.95,1.0)
ax.set_axis_off(); ax.view_init(elev=13, azim=-52)
out=os.path.join(os.path.dirname(__file__),"..","imagenes")
fig.savefig(os.path.join(out,"fig_elementos_orbitales.pdf"),facecolor=BG,bbox_inches="tight")
fig.savefig(os.path.join(out,"fig_elementos_orbitales.png"),dpi=135,facecolor=BG,bbox_inches="tight")
print("ok")
