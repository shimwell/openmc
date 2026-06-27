"""1D spherical Sn fixed-source solver. Schemes:
  'diamond' : weighted-diamond (2nd order; can go negative on resonant problems)
  'step'    : step characteristic (upwind space+angle); unconditionally positive but diffusive
  'fixup'   : diamond + negative-flux fixup (set negative outgoing edges to 0 and re-solve the
              cell). 2nd-order where positive, local upwind only where needed -> accurate AND
              positive. DEFAULT.
Curvature coeffs alpha_{m+1/2} >= 0 (tent 0->peak->0), which guarantees the fixup terminates
positive. Validated vs analytic point-source-in-absorber S*exp(-st r)/(4 pi r^2)."""
import numpy as np

def _cell(srci, sigtV, amu, Aout, Ain, pin, c, a_hi, a_lo, pang, fixup):
    """Solve one cell for psi_cell and the outgoing spatial (pout) + angular (pmid) edges.
    Balance: amu*(Aout*pout - Ain*pin) + c*(a_hi*pmid - a_lo*pang) + sigtV*psc = srci,
    with diamond closures pout=2psc-pin, pmid=2psc-pang unless an edge is clamped to 0."""
    co = cm = False
    for _ in range(3):
        coef = sigtV + (0.0 if co else 2*amu*Aout) + (0.0 if cm else 2*c*a_hi)
        const = srci + amu*Ain*pin + c*a_lo*pang + (0.0 if co else amu*Aout*pin) + (0.0 if cm else c*a_hi*pang)
        psc = const/(coef + 1e-30)
        pout = 0.0 if co else 2*psc - pin
        pmid = 0.0 if cm else 2*psc - pang
        if not fixup or (pout >= 0 and pmid >= 0):
            break
        if pout < 0: co = True
        if pmid < 0: cm = True
    return psc, pout, pmid

def solve_sphere(redge, sigt, Svol, mu, w, alpha, scheme='fixup'):
    I = len(sigt); M = len(mu)
    A = 4*np.pi*redge**2
    V = 4*np.pi/3*(redge[1:]**3 - redge[:-1]**3)
    src = 0.5*Svol*V
    fixup = (scheme == 'fixup'); step = (scheme == 'step')
    psi = np.zeros((I, M))
    # starting direction mu=-1 (no angular redistribution), inward
    pin = 0.0; psi_edge = np.zeros(I)
    for i in range(I-1, -1, -1):
        if step:
            psc = (src[i] + A[i+1]*pin)/(A[i] + sigt[i]*V[i] + 1e-30); pout = psc
        else:
            psc, pout, _ = _cell(src[i], sigt[i]*V[i], 1.0, A[i], A[i+1], pin, 0.0, 0.0, 0.0, 0.0, fixup)
        psi_edge[i] = psc; pin = pout
    for m in range(M):
        a_lo = alpha[m]; a_hi = alpha[m+1]; cur = np.zeros(I); amu = abs(mu[m])
        if mu[m] < 0:                                              # inward: Aout=A[i], Ain=A[i+1]
            pin = 0.0
            for i in range(I-1, -1, -1):
                c = (A[i+1]-A[i])/w[m]
                if step:
                    psc = (src[i] + amu*A[i+1]*pin + c*a_lo*psi_edge[i])/(amu*A[i] + c*a_hi + sigt[i]*V[i] + 1e-30); pout = psc; cur[i] = psc; pin = pout; psi_edge[i] = psc
                else:
                    psc, pout, pmid = _cell(src[i], sigt[i]*V[i], amu, A[i], A[i+1], pin, c, a_hi, a_lo, psi_edge[i], fixup)
                    cur[i] = psc; pin = pout; psi_edge[i] = pmid
        else:                                                     # outward: Aout=A[i+1], Ain=A[i]
            pin = 0.0
            for i in range(I):
                c = (A[i+1]-A[i])/w[m]
                if step:
                    psc = (src[i] + amu*A[i]*pin + c*a_lo*psi_edge[i])/(amu*A[i+1] + c*a_hi + sigt[i]*V[i] + 1e-30); pout = psc; cur[i] = psc; pin = pout; psi_edge[i] = psc
                else:
                    psc, pout, pmid = _cell(src[i], sigt[i]*V[i], amu, A[i+1], A[i], pin, c, a_hi, a_lo, psi_edge[i], fixup)
                    cur[i] = psc; pin = pout; psi_edge[i] = pmid
        psi[:, m] = cur
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
    for sch in ('diamond', 'step', 'fixup'):
        phi = solve_sphere(redge, sigt, Svol, mu, w, alpha, scheme=sch)
        err = np.abs(phi[msk]-ana[msk])/ana[msk]
        print(f"  {sch:8}: mean {100*err.mean():.2f}%  max {100*err.max():.2f}%  min phi {phi.min():.2e}")
