"""Esperimento A: numero di passate P al variare della memoria S.

Protocollo, fissato prima delle esecuzioni:
- N = 10^3, 10^4, 10^5, 10^6, permutazione casuale di 1..N con seed 2026;
- k = ceil(N/2), cioè la mediana;
- S su una griglia geometrica di fattore 2^(1/4), da S_min a N + 2 incluso,
per un totale di 101 configurazioni;
- verifica aggiuntiva per N <= 10^4 con k = 1, ceil(N/2), N e seed 1, 2, 3.

Per ogni esecuzione si registrano P osservato (scansioni del nastro), P_max
dato dalla ricorrenza di n_bar, il numero di passate di campionamento e il
massimo teorico di celle logiche. P può essere minore di P_max solo quando
una PASS termina con Found; altrimenti i due valori devono coincidere.

Le celle logiche sono il limite teorico dell'algoritmo, non la RAM usata
da Python.

Il file contiene anche lo stile comune delle figure, usato pure da
experiment_candidates.py.

Uso, dalla cartella principale del progetto:
    python experiments/experiment_passes.py                      campagna e figura
    python experiments/experiment_passes.py --plot-only          solo figura dai CSV
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from metrics_multipass import run_experiment, theoretical_metrics


# ---------------------------------------------------------------- protocollo

N_VALUES = (1_000, 10_000, 100_000, 1_000_000)
S_MIN = {1_000: 209, 10_000: 445, 100_000: 761, 1_000_000: 1152}
SEED = 2026
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIELDS = [
    "N", "S", "k", "seed", "ordine", "P_max", "P_osservato",
    "passate_campionamento", "celle_logiche_max",
]


# ------------------------------------------------- stile comune delle figure

CM = 1 / 2.54                 # pollici in un centimetro
TEXT_WIDTH = 16.0 * CM        # \textwidth della tesi: A4 con margini di 2,5 cm

# Palette Okabe-Ito, leggibile anche con daltonismo; i marcatori cambiano
# forma, così le serie si distinguono anche stampate in bianco e nero.
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
MARKERS = ["o", "s", "^", "D"]


def set_style() -> None:
    """Imposta lo stile comune: font Times come newtx, assi sobri, PDF."""
    plt.rcParams.update({
        "font.family": "STIXGeneral",     # Times, come newtxtext
        "mathtext.fontset": "stix",       # Times anche nelle formule
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": True,
        "grid.color": "0.88",
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,               # font incorporati nel PDF
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    """Salva un CSV completo, con intestazione, sovrascrivendo il precedente."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    """Legge un CSV come lista di dizionari (i valori restano stringhe)."""
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


# ------------------------------------------------------------------- misure

def memory_grid(N: int) -> list[int]:
    """Valori di S del protocollo: da S_min con fattore 2^(1/4), più N + 2."""
    grid = []
    j = 0
    while True:
        S = round(S_MIN[N] * 2 ** (j / 4))
        if S >= N + 2:
            break
        if not grid or S != grid[-1]:
            grid.append(S)
        j += 1

    grid.append(N + 2)  # con S = N + 2 basta la sola passata finale
    return grid


def measure(N: int, S: int, k: int, seed: int) -> dict:
    """Esegue MP-SELECT su una configurazione e confronta P con P_max.

    run_experiment controlla già che il risultato sia N + 1 - k e che il
    conteggio delle passate sia coerente con le PASS eseguite.
    """
    P_max, cells, _ = theoretical_metrics(N, S)
    _, P, pass_rows = run_experiment(N, S, k, seed)

    # Senza Found il numero di PASS è fissato dalla ricorrenza di n_bar,
    # quindi P deve valere esattamente P_max; Found può solo accorciare.
    found = bool(pass_rows) and pass_rows[-1]["stato"] == "Found"
    if P > P_max or (not found and P != P_max):
        raise AssertionError(f"N={N}, S={S}, k={k}: P={P}, P_max={P_max}")
    if cells > S:
        raise AssertionError(f"N={N}, S={S}: celle teoriche oltre il budget")

    return {
        "N": N, "S": S, "k": k, "seed": seed, "ordine": "casuale",
        "P_max": P_max, "P_osservato": P,
        "passate_campionamento": len(pass_rows),
        "celle_logiche_max": cells,
    }


def run_campaign() -> None:
    """Esegue le 101 misure principali e le 270 verifiche aggiuntive."""
    rows = []
    for N in N_VALUES:
        k = (N + 1) // 2
        grid = memory_grid(N)
        for S in grid:
            rows.append(measure(N, S, k, SEED))
        print(f"N={N}: {len(grid)} valori di S misurati")

    if len(rows) != 101:
        raise AssertionError("La griglia non contiene i 101 punti previsti")
    write_csv(RESULTS_DIR / "p_vs_s.csv", FIELDS, rows)

    # Verifica aggiuntiva: tre ranghi e tre seed, solo per N <= 10^4.
    checks = []
    for N in (1_000, 10_000):
        for S in memory_grid(N):
            for k in (1, (N + 1) // 2, N):
                for seed in (1, 2, 3):
                    checks.append(measure(N, S, k, seed))
    write_csv(RESULTS_DIR / "p_vs_s_validation.csv", FIELDS, checks)

    for name, data in (("misure principali", rows), ("verifiche", checks)):
        equal = sum(r["P_osservato"] == r["P_max"] for r in data)
        print(f"{name}: {len(data)} esecuzioni, P = P_max in {equal}")


# ------------------------------------------- gradini teorici della figura

def p_max(N: int, S: int) -> int | None:
    """P_max dalla ricorrenza, None se con questa memoria non si termina."""
    try:
        return theoretical_metrics(N, S)[0]
    except ValueError:
        return None


def min_memory(N: int, P: int) -> int:
    """Soglia S*(P): il più piccolo S con P_max(N, S) <= P.

    La ricerca binaria presuppone che P_max sia non crescente al crescere
    di S; questa proprietà non è dimostrata qui in generale.
    """
    low, high = S_MIN[N], N + 2
    first = p_max(N, low)
    if first is not None and first <= P:
        return low

    while high - low > 1:
        mid = (low + high) // 2
        p = p_max(N, mid)
        if p is not None and p <= P:
            high = mid
        else:
            low = mid
    return high



# ------------------------------------------------------------------- figura

def plot_figure() -> None:
    """Figura P-S: P_max a gradini dalla ricorrenza e misure da p_vs_s.csv."""
    rows = read_csv(RESULTS_DIR / "p_vs_s.csv")
    set_style()
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 8.0 * CM))

    handles = []
    for N, color, marker in zip(N_VALUES, COLORS, MARKERS):
        # P_max vale P per S*(P) <= S < S*(P - 1): bastano le soglie.
        top = p_max(N, S_MIN[N])
        levels = list(range(top, 0, -1))
        edges = [min_memory(N, P) for P in levels]
        ax.step(edges, levels, where="post", color=color)

        points = sorted((int(r["S"]), int(r["P_osservato"]))
                        for r in rows if int(r["N"]) == N)
        S_meas, P_meas = zip(*points)
        ax.plot(S_meas, P_meas, linestyle="none", marker=marker,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=0.8)

        handles.append(Line2D([], [], color=color, marker=marker,
                            markerfacecolor="white", markeredgecolor=color,
                            label=rf"$N=10^{{{round(math.log10(N))}}}$"))

    ax.set_xscale("log")
    ax.set_xlim(1.5e2, 1.6e6)
    ax.set_ylim(0.5, 9.5)
    ax.set_yticks(range(1, 10))
    ax.grid(False, which="minor")
    ax.set_xlabel(r"Memoria $S$ (celle)")
    ax.set_ylabel(r"Numero di passate $P$")
    ax.legend(handles=handles, loc="upper right")

    path = RESULTS_DIR / "fig_p_vs_s.pdf"
    fig.savefig(path)
    print(f"Figura salvata in: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esperimento A: P in funzione di S")
    parser.add_argument("--plot-only", action="store_true",
                        help="rigenera solo la figura dai CSV esistenti")
    args = parser.parse_args()

    if args.plot_only:
        plot_figure()
    else:
        run_campaign()
        plot_figure()
