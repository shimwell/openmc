"""1D spherical Sn fixed-source solver. Two schemes:
  scheme='diamond' : weighted-diamond (2nd order, can go negative on resonant problems)
  scheme='step'    : step characteristic (upwind in space AND angle). Since the curvature
                     coefficients alpha_{m+1/2} are >= 0 (tent: 0 -> peak -> 0), step is
                     UNCONDITIONALLY POSITIVE -- robust for resonant/optically-thick cells.
Validated against the analytic point-source-in-uniform-absorber flux S*exp(-st r)/(4 pi r^2)."""
import numpy as np

def solve_sphere(redge, sigt, Svol, mu, w, alpha, scheme='step'):
    I = len(sigt); M = len(mu)
    A = 4*np.pi*redge**2
    V = 4*np.pi/3*(redge[1:]**3 - redge[:-1]**3)
    src = 0.5*Svol*V
    psi = np.zeros((I, M))
    # starting direction mu=-1 (no angular redistribution), inward sweep
    pin = 0.0; psi_edge = np.zeros(I)
    for i in range(I-1, -1, -1):
        if scheme == 'step':                                    # psi_cell = psi_out (inner edge)
            psc = (src[i] + A[i+1]*pin) / (A[i] + sigt[i]*V[i] + 1e-30)
            psi_edge[i] = psc; pin = psc
        else:
            psc = (src[i] + (A[i+1]+A[i])*pin) / (2*A[i] + sigt[i]*V[i] + 1e-30)
            psi_edge[i] = psc; pin = 2*psc - pin
    for m in range(M):
        a_lo = alpha[m]; a_hi = alpha[m+1]; cur = np.zeros(I)
        if mu[m] < 0:                                           # inward: incoming = outer edge
            pin = 0.0
            for i in range(I-1, -1, -1):
                c = (A[i+1]-A[i])/w[m]
                if scheme == 'step':
                    psc = (src[i] - mu[m]*A[i+1]*pin + c*a_lo*psi_edge[i]) / (-mu[m]*A[i] + c*a_hi + sigt[i]*V[i] + 1e-30)
                    cur[i] = psc; pin = psc
                else:
                    psc = (src[i] - mu[m]*(A[i+1]+A[i])*pin + c*(a_hi+a_lo)*psi_edge[i]) / (-2*mu[m]*A[i] + 2*c*a_hi + sigt[i]*V[i] + 1e-30)
                    cur[i] = psc; pin = 2*psc - pin
        else:                                                  # outward: incoming = inner edge
            pin = 0.0
            for i in range(I):
                c = (A[i+1]-A[i])/w[m]
                if scheme == 'step':
                    psc = (src[i] + mu[m]*A[i]*pin + c*a_lo*psi_edge[i]) / (mu[m]*A[i+1] + c*a_hi + sigt[i]*V[i] + 1e-30)
                    cur[i] = psc; pin = psc
                else:
                    psc = (src[i] + mu[m]*(A[i+1]+A[i])*pin + c*(a_hi+a_lo)*psi_edge[i]) / (2*mu[m]*A[i+1] + 2*c*a_hi + sigt[i]*V[i] + 1e-30)
                    cur[i] = psc; pin = 2*psc - pin
        psi[:, m] = cur
        psi_edge = cur if scheme == 'step' else (2*cur - psi_edge)   # angular-edge update
    return psi @ w

if __name__ == "__main__":
    R = 20.0; I = 400; st = 0.1; S = 1.0
    redge = np.linspace(0, R, I+1); rc = 0.5*(redge[1:]+redge[:-1])
    sigt = np.full(I, st); Svol = np.zeros(I)
    V = 4*np.pi/3*(redge[1:]**3-redge[:-1]**3); Svol[0] = S/V[0]
    mu, w = np.polynomial.legendre.leggauss(16); alpha = np.zeros(17)
    for m in range(16): alpha[m+1] = alpha[m] - mu[m]*w[m]
    ana = S*np.exp(-st*rc)/(4*np.pi*rc**2); msk = (rc > 2) & (rc < 16)
    print("spherical Sn validation (point source in uniform absorber):")
    for sch in ('diamond', 'step'):
        phi = solve_sphere(redge, sigt, Svol, mu, w, alpha, scheme=sch)
        err = np.abs(phi[msk]-ana[msk])/ana[msk]
        print(f"  {sch:8}: mean {100*err.mean():.1f}%  max {100*err.max():.1f}%  min phi {phi.min():.2e}")
