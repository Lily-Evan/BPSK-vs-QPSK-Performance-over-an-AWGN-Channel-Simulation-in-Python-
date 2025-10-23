#!/usr/bin/env python3
"""
BPSK vs QPSK over AWGN — Simulation in Python (NumPy + Matplotlib)

This script:
  1) Computes BER-vs-Eb/N0 curves for BPSK and QPSK (theory and Monte Carlo).
  2) Plots noisy constellations at a selected Eb/N0.
  3) (Optional) Demonstrates a simple matched-filter receiver with rectangular pulse shaping.

Usage:
  python bpsk_qpsk_awgn_sim.py --ebn0_start 0 --ebn0_stop 12 --ebn0_step 1 --n_bits 200000 --const_ebn0 10 --do_matched_filter 1

Notes:
  - Uses only NumPy and Matplotlib.
  - One figure per plot; no seaborn; no style/colors enforced.
  - Theory: P_b = Q(sqrt(2*Eb/N0)) for both BPSK and QPSK (with Gray coding).
"""

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

def qfunc(x):
    x = np.asarray(x)
    vec = np.vectorize(lambda t: 0.5 * math.erfc(t / math.sqrt(2.0)))
    return vec(x)

def ber_theory_bpsk(EbN0_dB):
    EbN0 = 10**(EbN0_dB/10.0)
    return qfunc(np.sqrt(2*EbN0))

def ber_theory_qpsk(EbN0_dB):
    return ber_theory_bpsk(EbN0_dB)

def simulate_bpsk(EbN0_dB, n_bits=100_000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    bits = rng.integers(0, 2, size=n_bits)
    s = 1 - 2*bits  # 0->+1, 1->-1
    Es = 1.0
    EbN0 = 10**(EbN0_dB/10.0)
    N0 = Es/EbN0
    noise_sigma = np.sqrt(N0/2)
    w = rng.normal(0.0, noise_sigma, size=n_bits)
    r = s + w
    bits_hat = (r < 0).astype(int)
    ber = np.mean(bits != bits_hat)
    return ber, r

def simulate_qpsk(EbN0_dB, n_bits=100_000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n_sym = n_bits // 2
    bits = rng.integers(0, 2, size=(n_sym, 2))
    # Gray map: 00->(+,+), 01->(-,+), 11->(-,-), 10->(+,-)
    I = np.where(bits[:,0]==0, 1.0, -1.0)
    Q = np.where(bits[:,1]==0, 1.0, -1.0)
    s = (I + 1j*Q) / np.sqrt(2)  # Es=1 per symbol
    EbN0 = 10**(EbN0_dB/10.0)
    N0 = 1.0 / (2*EbN0)
    noise_sigma = np.sqrt(N0/2)
    w = rng.normal(0.0, noise_sigma, size=n_sym) + 1j*rng.normal(0.0, noise_sigma, size=n_sym)
    r = s + w
    I_hat = (r.real < 0).astype(int)
    Q_hat = (r.imag < 0).astype(int)
    bits_hat = np.column_stack([I_hat, Q_hat])
    ber = np.mean(bits != bits_hat)
    return ber, r

def simulate_bpsk_matched_filter(EbN0_dB, n_bits=100_000, sps=8, rng=None):
    """Simple rectangular pulse shaping and matched filter demo for BPSK."""
    if rng is None:
        rng = np.random.default_rng()
    bits = rng.integers(0, 2, size=n_bits)
    symbols = 1 - 2*bits  # NRZ +/-1

    # Upsample
    x = np.zeros(n_bits * sps)
    x[::sps] = symbols

    # Tx pulse: rectangular
    h_tx = np.ones(sps)
    tx = np.convolve(x, h_tx, mode='full')

    # Normalize so Eb=1
    Eb_raw = np.mean(symbols**2) * np.sum(h_tx**2)
    scale = 1/np.sqrt(Eb_raw)
    tx = tx * scale

    # AWGN (per-sample variance = N0/2)
    EbN0 = 10**(EbN0_dB/10.0)
    N0 = 1.0 / EbN0
    noise_sigma = np.sqrt(N0/2)
    rng = np.random.default_rng()
    w = rng.normal(0.0, noise_sigma, size=tx.shape)
    rx = tx + w

    # Matched filter
    h_mf = h_tx[::-1] * scale
    y = np.convolve(rx, h_mf, mode='full')

    # Sample at symbol indices
    delay = (len(h_tx)-1) + (len(h_mf)-1)
    sample_idx = np.arange(delay, delay + n_bits*sps, sps)
    y_samp = y[sample_idx]

    bits_hat = (y_samp < 0).astype(int)
    ber = np.mean(bits != bits_hat)
    return ber

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ebn0_start', type=float, default=0.0)
    ap.add_argument('--ebn0_stop', type=float, default=12.0)
    ap.add_argument('--ebn0_step', type=float, default=1.0)
    ap.add_argument('--n_bits', type=int, default=200000)
    ap.add_argument('--const_ebn0', type=float, default=10.0)
    ap.add_argument('--do_matched_filter', type=int, default=1)
    ap.add_argument('--outdir', type=str, default='outputs')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(42)

    EbN0_dB = np.arange(args.ebn0_start, args.ebn0_stop+1e-9, args.ebn0_step)
    bpsk_sim, qpsk_sim = [], []
    for val in EbN0_dB:
        ber_bpsk, _ = simulate_bpsk(val, n_bits=args.n_bits, rng=rng)
        bpsk_sim.append(ber_bpsk)
        ber_qpsk, _ = simulate_qpsk(val, n_bits=args.n_bits, rng=rng)
        qpsk_sim.append(ber_qpsk)

    bpsk_sim = np.array(bpsk_sim)
    qpsk_sim = np.array(qpsk_sim)
    bpsk_th = ber_theory_bpsk(EbN0_dB)
    qpsk_th = ber_theory_qpsk(EbN0_dB)

    # BER plot
    plt.figure()
    plt.semilogy(EbN0_dB, bpsk_th, label='BPSK (theory)')
    plt.semilogy(EbN0_dB, bpsk_sim, 'o', label='BPSK (simulation)')
    plt.semilogy(EbN0_dB, qpsk_th, label='QPSK (theory)')
    plt.semilogy(EbN0_dB, qpsk_sim, 's', label='QPSK (simulation)')
    plt.grid(True, which='both')
    plt.xlabel(r'$E_b/N_0$ (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER of BPSK and QPSK over AWGN')
    plt.legend()
    ber_path = os.path.join(args.outdir, 'ber_bpsk_qpsk_awgn.png')
    plt.savefig(ber_path, bbox_inches='tight')
    plt.close()

    # Constellations
    _, r_bpsk = simulate_bpsk(args.const_ebn0, n_bits=5000, rng=rng)
    plt.figure()
    plt.plot(r_bpsk, np.zeros_like(r_bpsk), '.', alpha=0.5)
    plt.grid(True)
    plt.xlabel('In-phase')
    plt.ylabel('Quadrature')
    plt.title(f'BPSK Constellation (Eb/N0 = {args.const_ebn0:.1f} dB)')
    bpsk_const_path = os.path.join(args.outdir, f'const_bpsk_{int(args.const_ebn0)}dB.png')
    plt.savefig(bpsk_const_path, bbox_inches='tight')
    plt.close()

    _, r_qpsk = simulate_qpsk(args.const_ebn0, n_bits=10000, rng=rng)
    plt.figure()
    plt.plot(r_qpsk.real, r_qpsk.imag, '.', alpha=0.5)
    plt.grid(True)
    plt.xlabel('In-phase')
    plt.ylabel('Quadrature')
    plt.title(f'QPSK Constellation (Eb/N0 = {args.const_ebn0:.1f} dB)')
    qpsk_const_path = os.path.join(args.outdir, f'const_qpsk_{int(args.const_ebn0)}dB.png')
    plt.savefig(qpsk_const_path, bbox_inches='tight')
    plt.close()

    mf_path = None
    if args.do_matched_filter:
        EbN0_dB_mf = np.arange(args.ebn0_start, args.ebn0_stop+1e-9, max(2.0, args.ebn0_step))
        mf_bers = [simulate_bpsk_matched_filter(x, n_bits=60000, sps=8, rng=rng) for x in EbN0_dB_mf]
        plt.figure()
        plt.semilogy(EbN0_dB, bpsk_th, label='BPSK (theory, symbol-rate)')
        plt.semilogy(EbN0_dB_mf, mf_bers, 'o', label='BPSK + Rectangular MF (sim)')
        plt.grid(True, which='both')
        plt.xlabel(r'$E_b/N_0$ (dB)')
        plt.ylabel('Bit Error Rate (BER)')
        plt.title('Effect of Matched Filtering (Rectangular Pulse, SPS=8)')
        plt.legend()
        mf_path = os.path.join(args.outdir, 'ber_bpsk_matched_filter.png')
        plt.savefig(mf_path, bbox_inches='tight')
        plt.close()

    # Save a small JSON with numbers (for reproducibility / grading)
    results = {
        'EbN0_dB': EbN0_dB.tolist(),
        'bpsk_ber_theory': bpsk_th.tolist(),
        'qpsk_ber_theory': qpsk_th.tolist(),
        'bpsk_ber_sim': bpsk_sim.tolist(),
        'qpsk_ber_sim': qpsk_sim.tolist()
    }
    with open(os.path.join(args.outdir, 'ber_data.json'), 'w') as f:
        import json
        json.dump(results, f, indent=2)

    print('Saved:')
    print(' ', ber_path)
    print(' ', bpsk_const_path)
    print(' ', qpsk_const_path)
    if mf_path:
        print(' ', mf_path)

if __name__ == '__main__':
    import os
    main()
