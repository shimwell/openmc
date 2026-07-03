"""Assemble the labyrinth A/B/C FOM table across seeds."""
import glob
import json

import numpy as np

BASE = '/home/jon/openmc/omega_bench'
DETS = ['D1_leg1_end', 'D2_leg2_mid', 'D3_leg3_start', 'D4_leg3_mid',
        'D5_leg3_end']

rows = []
for f in sorted(glob.glob(f'{BASE}/lab_*/result_*.json')):
    if 'pilot' in f:
        continue
    rows.append(json.load(open(f)))

print(f"{'case':<16}{'seed':>4} {'active_s':>9}", end='')
for d in DETS:
    print(f" {d.split('_')[0]+'_relerr':>10} {d.split('_')[0]+'_fom':>9}",
          end='')
print()
for r in sorted(rows, key=lambda x: (x['case'], x['seed'])):
    print(f"{r['case']:<16}{r['seed']:>4} {r['active_s']:>9.1f}", end='')
    for d in DETS:
        print(f" {r[d]['rel_err']:>10.4f} {r[d]['fom']:>9.2f}", end='')
    print()

CASES = ['analog', 'fw_cadis', 'fw_cadis_omega', 'omega_p3', 'omega_p3_amp',
         'omega_p1_amp', 'ang_p1', 'ang_p3']

print('\nPer-detector FOM geo-means:')
hdr = f"{'detector':<15}"
for c in CASES:
    hdr += f"{c:>15}"
print(hdr)
for d in DETS:
    line = f"{d:<15}"
    for case in CASES:
        foms = [r[d]['fom'] for r in rows if r['case'] == case
                and r[d]['fom'] > 0]
        gm = np.exp(np.mean(np.log(foms))) if foms else 0.0
        line += f"{gm:>15.2f}"
    print(line)

print('\nPaired FOM ratios vs fw_cadis (geo-mean over matched seeds):')
hdr = f"{'detector':<15}"
for c in CASES[2:]:
    hdr += f"{c:>15}"
print(hdr)
for d in DETS:
    line = f"{d:<15}"
    for case in CASES[2:]:
        pairs = []
        for s in sorted({r['seed'] for r in rows}):
            f_ = [r for r in rows
                  if r['case'] == 'fw_cadis' and r['seed'] == s]
            o_ = [r for r in rows if r['case'] == case and r['seed'] == s]
            if f_ and o_ and f_[0][d]['fom'] > 0 and o_[0][d]['fom'] > 0:
                pairs.append(o_[0][d]['fom'] / f_[0][d]['fom'])
        ratio = np.exp(np.mean(np.log(pairs))) if pairs else float('nan')
        line += f"{ratio:>15.2f}"
    print(line)

print('\nDetector mean consistency (all runs):')
for d in DETS:
    for case in CASES:
        sel = [r[d] for r in rows if r['case'] == case]
        if sel:
            ms = np.array([x['mean'] for x in sel])
            print(f"  {d:<15} {case:<15} mean={ms.mean():.4e} "
                  f"(n={len(sel)}, spread={ms.std(ddof=1)/ms.mean()*100 if len(sel)>1 else 0:.1f}%)")
