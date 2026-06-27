"""1D spherical Sn (weighted-diamond, curvilinear angular redistribution) fixed-source
solver, with validation against the analytic point-source-in-uniform-absorber flux
phi(r) = S*exp(-Sigma_t r)/(4 pi r^2). Used for the real-geometry deterministic transport."""
import numpy as np

def solve_sphere(redge, sigt, Svol, mu, w, alpha):
    """One-group 1D spherical Sn fixed-source solve.
    redge: cell edges r_0..r_I (r_0=0 center). sigt[I], Svol[I] = isotropic source/vol.
    mu,w: Gauss-Legendre nodes/weights ([-1,1], sum w=2), ascending. alpha[M+1] curvature edges.
    Returns phi[I] = sum_m w_m psi_{i,m}. Vacuum at outer; symmetry at center.
    Down-scatter only within a group -> single source iteration (no within-group scatter)."""
    I = len(sigt); M = len(mu)
    A = 4*np.pi*redge**2                       # surface area at each edge (A[0]=0)
    V = 4*np.pi/3*(redge[1:]**3 - redge[:-1]**3)
    src = 0.5*Svol*V                           # isotropic angular source per cell
    psi = np.zeros((I, M))                      # cell-avg angular flux
    # --- starting direction mu = -1 (no angular redistribution), sweep inward ---
    psi_edge_m = np.zeros(I+1)                  # angular-edge (m-1/2) cell-avg flux, init mu=-1
    pin = 0.0                                   # vacuum at outer edge (incoming for inward)
    psm = np.zeros(I)
    for i in range(I-1, -1, -1):               # inward: in=outer edge, out=inner edge
        # -(A_{i+1}psi_out_outeredge ...): mu=-1 streaming  -1*(A[i+1]*pin - A[i]*pout)
        # balance: -1*(A[i+1]*pin - A[i]*pout) + sigt*V*psi = src ; psi=0.5(pin+pout)
        # pout = 2 psi - pin
        den = A[i] + sigt[i]*V[i] + 1e-30
        psi_s = (src[i] + 0.5*(A[i+1]+A[i])*pin*0 + A[i+1]*0 + pin*(A[i] *0) )  # placeholder
        # solve: -(A[i+1]pin - A[i](2psi-pin)) + sigt V psi = src
        # = -A[i+1]pin + 2A[i]psi - A[i]pin + sigt V psi = src
        # psi(2A[i] + sigt V) = src + (A[i+1]+A[i]) pin
        psi_s = (src[i] + (A[i+1]+A[i])*pin) / (2*A[i] + sigt[i]*V[i] + 1e-30)
        psm[i] = psi_s; pin = 2*psi_s - pin    # pout becomes next inner cell's incoming
    psi_edge_prev = psm.copy()                 # psi at mu-edge 1/2 (the mu=-1 start)
    # --- ordinate sweep m=0..M-1 (mu ascending: negatives first=inward, then positives=outward) ---
    for m in range(M):
        a_lo = alpha[m]; a_hi = alpha[m+1]
        cur = np.zeros(I)
        if mu[m] < 0:                          # inward sweep
            pin = 0.0
            for i in range(I-1, -1, -1):
                # streaming mu(A[i+1]pin - A[i]pout), pout=2psi-pin
                c_ang = (A[i+1]-A[i])/w[m]
                num = src[i] - mu[m]*(A[i+1]+A[i])*pin + c_ang*(a_hi+a_lo)*psi_edge_prev[i]
                den = -2*mu[m]*A[i] + 2*c_ang*a_hi + sigt[i]*V[i] + 1e-30
                ps = num/den; cur[i] = ps; pin = 2*ps - pin
        else:                                  # outward sweep
            pin = 0.0                          # symmetry at center: incoming = outgoing of mu=-mu; approx 0 net at r=0
            for i in range(I):
                c_ang = (A[i+1]-A[i])/w[m]
                num = src[i] + mu[m]*(A[i+1]+A[i])*pin + c_ang*(a_hi+a_lo)*psi_edge_prev[i]
                den = 2*mu[m]*A[i+1] + 2*c_ang*a_hi + sigt[i]*V[i] + 1e-30
                ps = num/den; cur[i] = ps; pin = 2*ps - pin
        psi[:, m] = cur
        psi_edge_prev = 2*cur - psi_edge_prev  # angular-edge recursion psi_{m+1/2}=2psi_m-psi_{m-1/2}
    return psi @ w                             # scalar flux

if __name__ == "__main__":
    # validation: uniform absorber, point source at center -> phi = S exp(-st r)/(4pi r^2)
    R = 20.0; I = 400; st = 0.1; S = 1.0
    redge = np.linspace(0, R, I+1); rc = 0.5*(redge[1:]+redge[:-1])
    sigt = np.full(I, st); Svol = np.zeros(I)
    V = 4*np.pi/3*(redge[1:]**3-redge[:-1]**3); Svol[0] = S/V[0]   # point source in cell 0
    M = 16; mu, w = np.polynomial.legendre.leggauss(M)
    alpha = np.zeros(M+1)
    for m in range(M): alpha[m+1] = alpha[m] - mu[m]*w[m]
    phi = solve_sphere(redge, sigt, Svol, mu, w, alpha)
    ana = S*np.exp(-st*rc)/(4*np.pi*rc**2)
    msk = (rc > 2) & (rc < 16)
    err = np.abs(phi[msk]-ana[msk])/ana[msk]
    print(f"spherical Sn validation (uniform absorber, point source):")
    print(f"  mean rel err vs analytic exp(-st r)/(4pi r^2): {100*err.mean():.1f}%  max {100*err.max():.1f}%")
    for rr in (4, 8, 12):
        j = np.argmin(np.abs(rc-rr)); print(f"  r={rr}: Sn {phi[j]:.4e}  analytic {ana[j]:.4e}  ratio {phi[j]/ana[j]:.3f}")
