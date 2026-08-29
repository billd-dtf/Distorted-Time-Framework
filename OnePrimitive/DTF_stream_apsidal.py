"""
STAGE 2 -- the actual test. The apsidal angle (azimuth swept from pericentre to apocentre)
is THE apsidal-precession observable:
   DTF frozen-inertia (closed, boosted-Kepler)      -> 180 deg
   AQUAL / DM / Milgrom (precessing rosette)         -> < 180 (127 deep-MOND log limit)
The Orphan stream runs from r~15 to r~98 kpc = ~peri to ~apo, so its in-plane azimuthal sweep
between r_min and r_max directly measures the apsidal angle. Compare to 180 (DTF) and to a
directly-integrated AQUAL orbit of the same peri/apo.
"""
import numpy as np, os
TR = os.environ.get("GALSTREAMS_TRACKS")
if not TR:
    try:
        import galstreams
        TR = os.path.join(os.path.dirname(galstreams.__file__), "tracks")
    except ImportError:
        raise SystemExit(
            "Stream tracks not found. This script uses tracks from the galstreams package "
            "(Mateu 2023); they are not redistributed here. Run: pip install galstreams, "
            "or set GALSTREAMS_TRACKS to a directory of stream-track .ecsv files.")
c=2.99792458e8; G=6.674e-11; Msun=1.989e30; kpc=3.0857e19
Mpc=3.0856775815e22; H0=67.4e3/Mpc; a0=c*H0/(2*np.pi); R0=8.122; Mb=6.0e10*Msun
A_G=np.array([[-0.0548755604,-0.8734370902,-0.4838350155],
              [ 0.4941094279,-0.4448296300, 0.7469822445],
              [-0.8676661490,-0.1980763734, 0.4559837762]])
def read_ecsv(path):
    rows=[]; cols=None
    for ln in open(path):
        if ln.startswith('#') or not ln.strip(): continue
        if cols is None: cols=ln.strip().split(','); continue
        rows.append(ln.strip().split(','))
    return {cols[i]:np.array([r[i] for r in rows]) for i in range(len(cols))}
def to_galcen(ra,dec,dist):
    ra=np.radians(ra); dec=np.radians(dec)
    icrs=np.vstack([np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec)])
    xh,yh,zh=(A_G@icrs)*dist
    return np.vstack([xh-R0,yh,zh])          # galactocentric X,Y,Z (kpc)

def a_aqual(r):
    gb=G*Mb/(r*kpc)**2; return np.sqrt((gb**2+gb*np.sqrt(gb**2+4*a0**2))/2)

def aqual_apsidal(rp_kpc,ra_kpc):
    """Integrate an AQUAL orbit with given peri/apo; return azimuth swept peri->apo (deg)."""
    rp,ra=rp_kpc*kpc,ra_kpc*kpc
    # set L,E so that r_peri,r_apo are turning points: v_r=0 there.
    # energy method: 0.5 vt_p^2 + Phi(rp) = 0.5 vt_a^2 + Phi(ra), L=rp*vt_p=ra*vt_a
    # Phi from integrating a_aqual (radial accel = dPhi/dr)
    rg=np.logspace(np.log10(rp*0.5),np.log10(ra*1.5),6000)
    ag=a_aqual(rg/kpc); Phi=np.concatenate([[0],np.cumsum((ag[:-1]+ag[1:])/2*np.diff(rg))])
    Phi_i=lambda r: np.interp(r,rg,Phi)
    # solve for L^2: 0.5 L^2/rp^2 + Phi(rp) = 0.5 L^2/ra^2 + Phi(ra)
    L2=2*(Phi_i(ra)-Phi_i(rp))/(1/rp**2-1/ra**2); L=np.sqrt(L2)
    E=0.5*L2/rp**2+Phi_i(rp)
    # azimuth swept peri->apo: integral of (L/r^2)/v_r dr, v_r=sqrt(2(E-Phi)-L^2/r^2)
    r=np.linspace(rp*1.0001,ra*0.9999,20000)
    vr2=2*(E-Phi_i(r))-L2/r**2; vr2=np.clip(vr2,1e-30,None)
    dphi=(L/r**2)/np.sqrt(vr2)
    return np.degrees(np.trapz(dphi,r))

targets={"Orphan-Chenab":"track.st.Orphan-Chenab.koposov2019.ecsv",
         "Sagittarius":"track.st.Sagittarius.antoja2020.ecsv"}
print(f"a0={a0:.2e}   Mbar={Mb/Msun:.0e} Msun (point-mass approx)\n")
print(f"{'stream':<15}{'peri-apo kpc':>15}{'measured apsidal':>18}{'AQUAL pred':>13}{'DTF pred':>10}")
for name,fn in targets.items():
    d=read_ecsv(os.path.join(TR,fn))
    XYZ=to_galcen(d['ra'].astype(float),d['dec'].astype(float),d['distance'].astype(float))
    r=np.linalg.norm(XYZ,axis=0)
    # orbital plane via SVD (thin stream ~ planar); project, get in-plane azimuth
    Xc=XYZ-XYZ.mean(1,keepdims=True)
    U,S,Vt=np.linalg.svd(Xc@Xc.T); n=U[:,2]              # plane normal = smallest axis
    e1=U[:,0]; e2=np.cross(n,e1)
    u=e1@XYZ; v=e2@XYZ; phi=np.unwrap(np.arctan2(v,u))
    # apsidal angle = |azimuth at r_max - azimuth at r_min|
    i_min=np.argmin(r); i_max=np.argmax(r)
    daps=np.degrees(abs(phi[i_max]-phi[i_min]))
    daps=min(daps,360-daps) if daps>180 else daps
    aq=aqual_apsidal(r.min(),r.max())
    print(f"{name:<15}{r.min():>6.1f} -{r.max():>6.1f}{daps:>17.1f}{aq:>13.1f}{180.0:>10.1f}")
print("""
READ: DTF predicts apsidal angle = 180 deg (closed); AQUAL/DM/Milgrom predict the smaller
integrated value. Whichever the MEASURED sweep sits closer to is favoured.
CAVEATS (why this is proof-of-concept, not a verdict): (1) a stream track != the progenitor
orbit exactly (few-deg offset); (2) point-mass baryons + no halo flattening; (3) distance
errors along the track; (4) endpoints assumed ~peri/apo. A real test marginalizes these.""")
