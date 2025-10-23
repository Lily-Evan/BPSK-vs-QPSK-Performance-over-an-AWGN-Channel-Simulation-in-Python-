
"""
Digital Communication Systems – BER Simulation for BPSK & QPSK over AWGN
Author: Panagiota (template generated)
Requirements: numpy, matplotlib
Outputs:
- ber_results.csv            (BER vs Eb/N0 for BPSK & QPSK – sim & theory)
- ber_plot.png               (Curves: simulated vs theoretical)
- constellation_qpsk_XXdB.png (Example constellation at Eb/N0 = XX dB)
- constellation_bpsk_XXdB.png (Example BPSK scatter at Eb/N0 = XX dB)
Usage:
    python digicom_bpsk_qpsk_awgn.py
Notes:
- Normalized energy per bit Eb = 1.
- For QPSK: Es = k * Eb with k = log2(M) = 2.
- AWGN noise variance: sigma^2 = N0/2 per dimension. With Eb/N0 (linear),
  N0 = Eb / (Eb/N0) = 1/(EbN0_lin), sigma = sqrt(N0/2) for each real dimension.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# ------------------------------- Helpers ---------------------------------

def db2lin(db):
    return 10.0**(db/10.0)

def Q(x):
    """Tail probability of standard normal (works with scalars or numpy arrays)."""
    import numpy as _np, math as _math
    x_arr = _np.asarray(x, dtype=float)
    # vectorized wrapper around math.erf
    vec = _np.vectorize(lambda t: 0.5*(1.0 - _math.erf(t / _np.sqrt(2.0))))
    return vec(x_arr)


def theory_ber_bpsk(EbN0_dB):
    EbN0 = db2lin(EbN0_dB)
    return Q(np.sqrt(2*EbN0))

def theory_ber_qpsk(EbN0_dB):
    # Per-bit error rate is equal to BPSK's in AWGN
    return theory_ber_bpsk(EbN0_dB)

def awgn_noise_sigma(EbN0_dB, M=2):
    """
    Return per-dimension sigma for AWGN given Eb/N0 (in dB) and M-ary modulation.
    Using Eb=1 (normalized), N0 = 1 / (Eb/N0). For complex/baseband constellations:
    - For BPSK (real), sigma = sqrt(N0/2) per real dimension.
    - For QPSK (complex), Es = k*Eb with k=log2(M)=2. The symbol SNR Es/N0 = k*Eb/N0.
      However, the per-dimension sigma derives from N0 (not Es). Each real dim has variance N0/2.
    """
    EbN0_lin = db2lin(EbN0_dB)
    N0 = 1.0 / EbN0_lin
    sigma = np.sqrt(N0/2.0)
    return sigma

# ------------------------------- Modems ----------------------------------

def modulate_bpsk(bits):
    # Map 0 -> +1, 1 -> -1  (or vice versa; consistent demod handles it)
    return 1 - 2*bits.astype(np.int8)  # 0->+1, 1->-1

def demod_bpsk(received):
    # Hard decision: sign
    # Decide bit 0 if y >= 0 else 1
    return (received < 0).astype(np.uint8)

def modulate_qpsk(bits):
    """
    Gray-coded QPSK:
      Bits -> symbol (I + jQ) with unit energy per bit Eb=1 -> Es = 2
      Mapping (b0 b1): 00->(+1,+1), 01->(+1,-1), 11->(-1,-1), 10->(-1,+1)
    """
    assert len(bits) % 2 == 0, "QPSK requires an even number of bits"
    b0 = bits[0::2]
    b1 = bits[1::2]
    I = 1 - 2*b0  # 0->+1, 1->-1
    Q = 1 - 2*b1
    return (I + 1j*Q).astype(np.complex128)

def demod_qpsk(received):
    # Decide on I and Q separately
    b0_hat = (received.real < 0).astype(np.uint8)
    b1_hat = (received.imag < 0).astype(np.uint8)
    # Interleave back
    out = np.empty(b0_hat.size + b1_hat.size, dtype=np.uint8)
    out[0::2] = b0_hat
    out[1::2] = b1_hat
    return out

# ------------------------------- Simulation -------------------------------

def simulate_ber(mod, demod, EbN0_dB_list, n_bits=200_000, M=2, constellation_figure_EbN0_dB=None, constellation_filename_prefix="constellation"):
    ber = []
    for EbN0_dB in EbN0_dB_list:
        bits = np.random.randint(0, 2, size=n_bits, dtype=np.uint8)

        # Modulation
        s = mod(bits)

        # Noise
        sigma = awgn_noise_sigma(EbN0_dB, M=M)
        if np.iscomplexobj(s):
            noise = sigma*(np.random.randn(s.size) + 1j*np.random.randn(s.size))
        else:
            noise = sigma*np.random.randn(s.size)

        y = s + noise

        # Demodulation
        bhat = demod(y)

        # BER
        errors = np.count_nonzero(bits != bhat)
        ber.append(errors / n_bits)

        # Optional constellation snapshot
        if constellation_figure_EbN0_dB is not None and abs(EbN0_dB - constellation_figure_EbN0_dB) < 1e-9:
            # Downsample for plotting
            idx = np.random.choice(len(s), size=min(5000, len(s)), replace=False)
            fig = plt.figure()
            if np.iscomplexobj(y):
                plt.scatter(y[idx].real, y[idx].imag, s=6, alpha=0.5)
                plt.title(f"Constellation at Eb/N0 = {EbN0_dB:.1f} dB")
                plt.xlabel("In-phase (I)")
                plt.ylabel("Quadrature (Q)")
                plt.grid(True, which="both", ls=":")
                fname = f"{constellation_filename_prefix}_{int(EbN0_dB)}dB.png"
                plt.savefig(fname, dpi=160, bbox_inches="tight")
                plt.close(fig)
            else:
                # For BPSK, scatter y versus sample index (or histogram)
                fig = plt.figure()
                plt.scatter(np.arange(len(y[idx])), y[idx], s=6, alpha=0.5)
                plt.title(f"BPSK samples at Eb/N0 = {EbN0_dB:.1f} dB")
                plt.xlabel("Sample index")
                plt.ylabel("Amplitude")
                plt.grid(True, which="both", ls=":")
                fname = f"{constellation_filename_prefix}_{int(EbN0_dB)}dB.png"
                plt.savefig(fname, dpi=160, bbox_inches="tight")
                plt.close(fig)

    return np.array(ber)

def main():
    np.random.seed(12345)

    EbN0_dB_list = np.arange(0, 13, 1)  # 0..12 dB
    n_bits = 200_000

    # --- BPSK ---
    ber_bpsk_sim = simulate_ber(
        mod=modulate_bpsk,
        demod=demod_bpsk,
        EbN0_dB_list=EbN0_dB_list,
        n_bits=n_bits,
        M=2,
        constellation_figure_EbN0_dB=6.0,
        constellation_filename_prefix="constellation_bpsk"
    )
    ber_bpsk_theory = theory_ber_bpsk(EbN0_dB_list)

    # --- QPSK ---
    ber_qpsk_sim = simulate_ber(
        mod=modulate_qpsk,
        demod=demod_qpsk,
        EbN0_dB_list=EbN0_dB_list,
        n_bits=n_bits,
        M=4,
        constellation_figure_EbN0_dB=6.0,
        constellation_filename_prefix="constellation_qpsk"
    )
    ber_qpsk_theory = theory_ber_qpsk(EbN0_dB_list)

    # Save CSV
    with open("ber_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EbN0_dB", "BER_BPSK_sim", "BER_BPSK_theory", "BER_QPSK_sim", "BER_QPSK_theory"])
        for i, eb in enumerate(EbN0_dB_list):
            w.writerow([float(eb), float(ber_bpsk_sim[i]), float(ber_bpsk_theory[i]), float(ber_qpsk_sim[i]), float(ber_qpsk_theory[i])])

    # Plot
    plt.figure()
    plt.semilogy(EbN0_dB_list, ber_bpsk_theory, label="BPSK (theory)")
    plt.semilogy(EbN0_dB_list, ber_bpsk_sim, marker="o", linestyle="none", label="BPSK (sim)")
    plt.semilogy(EbN0_dB_list, ber_qpsk_theory, label="QPSK (theory)")
    plt.semilogy(EbN0_dB_list, ber_qpsk_sim, marker="s", linestyle="none", label="QPSK (sim)")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.title("BER for BPSK & QPSK over AWGN (Simulated vs Theoretical)")
    plt.ylim(1e-6, 1)
    plt.savefig("ber_plot.png", dpi=200, bbox_inches="tight")

    print("Saved: ber_results.csv, ber_plot.png, constellation_*_6dB.png")
    print("Done.")

if __name__ == "__main__":
    main()
