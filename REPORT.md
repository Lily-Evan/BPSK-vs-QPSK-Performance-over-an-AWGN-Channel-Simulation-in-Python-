# BPSK vs QPSK over AWGN — Simulation Report

**Date:** 2025-10-23

## Objective
Compare the bit error rate (BER) performance of BPSK and QPSK over an AWGN channel at equal energy per bit, \(E_b\), and illustrate constellation diagrams under noise.

## Theory
For coherent detection and Gray coding,
\[ P_b^{\text{BPSK}} = Q\!\big(\sqrt{2 E_b/N_0}\big), \quad
   P_b^{\text{QPSK}} = Q\!\big(\sqrt{2 E_b/N_0}\big). \]
Hence, both achieve the same BER vs \(E_b/N_0\). QPSK carries two bits per symbol, so for the same bit rate it halves the symbol rate and bandwidth versus BPSK (with identical pulse shaping).

## Method
- Monte Carlo simulation in Python (NumPy) with hard-decision detection.
- \(E_b/N_0\) sweep: 0–12 dB (1 dB step). Bits per trial: 2e5.
- Constellation snapshots at 10 dB.
- Optional: rectangular pulse shaping and matched filtering for BPSK.

## How to run
```bash
python bpsk_qpsk_awgn_sim.py --ebn0_start 0 --ebn0_stop 12 --ebn0_step 1 --n_bits 200000 --const_ebn0 10 --do_matched_filter 1 --outdir outputs
```

## Outputs
- `outputs/ber_bpsk_qpsk_awgn.png`: BER curves (theory vs simulation) for BPSK and QPSK.
- `outputs/const_bpsk_10dB.png`: BPSK constellation at 10 dB.
- `outputs/const_qpsk_10dB.png`: QPSK constellation at 10 dB.
- `outputs/ber_bpsk_matched_filter.png`: Matched-filter effect (optional).
- `outputs/ber_data.json`: Numerical BER data (theory and simulation).

## Observations
- Simulation aligns with \(Q(\sqrt{2 E_b/N_0})\) for both schemes.
- Constellation clouds contract around ideal points as \(E_b/N_0\) increases.
- With identical \(E_b/N_0\) and target BER, QPSK attains ~2× spectral efficiency relative to BPSK (same pulse shaping).

## Notes
- Extend with RRC pulse shaping (e.g., \(\alpha=0.2\)) for realistic spectra and eye diagrams.
- Add carrier/timing offsets to study practical receiver robustness.
