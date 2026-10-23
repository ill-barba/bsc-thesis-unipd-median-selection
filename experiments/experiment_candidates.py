"""Esperimento B: candidati effettivi n_p e limite teorico n_bar_p.

Domanda: dopo la passata di campionamento p, quanti candidati restano
davvero (n_p) rispetto al limite garantito n_bar_p = (2 r_p - 1) 2^r_p?

Protocollo di base, fissato prima delle esecuzioni:
- (N, S) = (10^4, 668), (10^4, 1335), (10^5, 1142), (10^5, 2283);
- k = 1, ceil(N/10), ceil(N/2), N;
- input casuali: permutazioni di 1..N con seed da 1 a 30;
- input ordinati: crescente e decrescente, una esecuzione ciascuno
(sono deterministici, il seed non viene usato).
In totale 480 esecuzioni casuali e 32 ordinate.

Estensione in S, aggiunta perché le quattro configurazioni danno per la
mediana soltanto 11 passate misurate e non coprono tutti i valori del limite:
- N = 10^4 e 10^5, k = ceil(N/2), permutazioni casuali con seed da 1 a 30;
- S su una griglia geometrica di fattore 2^(1/8), da S_min fino a N + 1
(contiene la griglia dell'esperimento A, tranne S = N + 2, dove non
c'è nessuna passata di campionamento).
In totale 93 valori di S e 2790 esecuzioni. Valori di S minori di S_min
non sono possibili: l'implementazione non terminerebbe.

n_p si conta sull'input, fuori dall'algoritmo: sono i valori strettamente
compresi tra i filtri restituiti dalla passata p. r_p e n_bar_p vengono
dalla ricorrenza (theoretical_metrics) e non dipendono dai dati: n_p e
n_bar_p descrivono quindi lo stesso momento, la fine della passata p.
Una passata che termina con Found non ha nuovi filtri, quindi non ha n_p;
le esecuzioni che arrivano a ciascuna passata vengono contate a parte.

Uscite in results/:
- candidates_raw.csv, candidates_summary.csv: protocollo di base;
- candidates_sweep_raw.csv, candidates_sweep_summary.csv: estensione in S;
- fig_candidates.pdf: la figura per la tesi (mediana, dati dell'estensione).

Uso, dalla cartella principale del progetto:
    python experiments/experiment_candidates.py              campagne e figura
    python experiments/experiment_candidates.py --plot-only  solo figura dai CSV
"""

import argparse
import math
import statistics

import matplotlib.pyplot as plt

from experiment_passes import (CM, COLORS, MARKERS, RESULTS_DIR, S_MIN,
                            TEXT_WIDTH, read_csv, set_style, write_csv)
from metrics_multipass import run_experiment, theoretical_metrics


# ---------------------------------------------------------------- protocollo

CONFIGS = [(10_000, 668), (10_000, 1335), (100_000, 1142), (100_000, 2283)]
SEEDS = range(1, 31)
RAW_FIELDS = [
    "N", "S", "k", "ordine", "seed", "P", "passata", "stato",
    "n", "n_bar", "rapporto", "f_max_invariato", "f_min_invariato",
]
SUMMARY_FIELDS = [
    "N", "S", "k", "ordine", "passata", "r", "n_bar", "esecuzioni", "found",
    "n_medio", "rapporto_medio", "rapporto_dev_std", "rapporto_min",
    "rapporto_max", "f_max_invariato", "f_min_invariato",
]

# Estensione in S.
SWEEP_N = (10_000, 100_000)


def ranks(N: int) -> list[int]:
    """I quattro ranghi del protocollo: massimo, N/10, mediana, minimo."""
    return [1, math.ceil(N / 10), math.ceil(N / 2), N]


def sweep_grid(N: int) -> list[int]:
    """Valori di S dell'estensione: da S_min con fattore 2^(1/8), sotto N + 2.

    Gli esponenti pari danno esattamente i valori della griglia
    dell'esperimento A, che usa il fattore 2^(1/4).
    """
    grid = []
    j = 0
    while True:
        S = round(S_MIN[N] * 2 ** (j / 8))
        if S >= N + 2:
            return grid
        if not grid or S != grid[-1]:
            grid.append(S)
        j += 1


# ------------------------------------------------------------------- misure

def run_one(N: int, S: int, k: int, ordine: str, seed: int,
            theory: list[dict]) -> list[dict]:
    """Esegue MP-SELECT su un input e restituisce una riga per passata.

    Oltre ai dati di run_experiment segna se f_min è rimasto invariato:
    insieme a f_max_invariato dice se l'intervallo dei candidati è stato
    ristretto da entrambi i lati o da uno solo.
    """
    _, P, pass_rows = run_experiment(N, S, k, seed, ordine)
    rows = []
    previous_f_min = -math.inf

    for row in pass_rows:
        p = row["passata"]
        out = {
            "N": N, "S": S, "k": k, "ordine": ordine,
            "seed": seed if ordine == "casuale" else "",
            "P": P, "passata": p, "stato": row["stato"],
            "n": "", "n_bar": "", "rapporto": "",
            "f_max_invariato": "", "f_min_invariato": "",
        }
        if row["stato"] == "Continue":
            # Il limite della passata deve essere quello della ricorrenza.
            if row["n_bar"] != theory[p - 1]["n_bar_out"]:
                raise AssertionError(f"N={N}, S={S}: n_bar diverso dalla teoria")
            out.update({
                "n": row["n"], "n_bar": row["n_bar"],
                "rapporto": row["n_su_n_bar"],
                "f_max_invariato": row["f_max_invariato"],
                "f_min_invariato": row["f_min"] == previous_f_min,
            })
            previous_f_min = row["f_min"]
        rows.append(out)
    return rows


def summarize(raw: list[dict]) -> list[dict]:
    """Aggrega il rapporto per (N, S, k, ordine, passata).

    Le statistiche usano solo le passate concluse con Continue; il numero
    di esecuzioni arrivate alla passata e quante terminano lì con Found
    sono riportati a parte, così una media non nasconde campioni diversi.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in raw:
        key = (row["N"], row["S"], row["k"], row["ordine"], row["passata"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (N, S, k, ordine, p), rows in groups.items():
        theory = theoretical_metrics(N, S)[2][p - 1]
        cont = [row for row in rows if row["stato"] == "Continue"]
        ratios = [row["rapporto"] for row in cont]
        summary.append({
            "N": N, "S": S, "k": k, "ordine": ordine, "passata": p,
            "r": theory["r"], "n_bar": theory["n_bar_out"],
            "esecuzioni": len(rows), "found": len(rows) - len(cont),
            "n_medio": statistics.mean(row["n"] for row in cont) if cont else "",
            "rapporto_medio": statistics.mean(ratios) if ratios else "",
            "rapporto_dev_std": statistics.stdev(ratios) if len(ratios) > 1 else "",
            "rapporto_min": min(ratios) if ratios else "",
            "rapporto_max": max(ratios) if ratios else "",
            "f_max_invariato": sum(row["f_max_invariato"] for row in cont),
            "f_min_invariato": sum(row["f_min_invariato"] for row in cont),
        })
    return summary


def run_campaign() -> None:
    """Esegue le 512 esecuzioni del protocollo e salva i due CSV."""
    raw = []
    for N, S in CONFIGS:
        theory = theoretical_metrics(N, S)[2]
        for k in ranks(N):
            for seed in SEEDS:
                raw += run_one(N, S, k, "casuale", seed, theory)
            for ordine in ("crescente", "decrescente"):
                raw += run_one(N, S, k, ordine, 0, theory)
        print(f"N={N}, S={S}: completate le esecuzioni dei quattro ranghi")

    summary = summarize(raw)
    write_csv(RESULTS_DIR / "candidates_raw.csv", RAW_FIELDS, raw)
    write_csv(RESULTS_DIR / "candidates_summary.csv", SUMMARY_FIELDS, summary)

    # Riepilogo dei controlli: quante esecuzioni e quante terminano prima.
    for ordine in ("casuale", "crescente", "decrescente"):
        runs = {(r["N"], r["S"], r["k"], r["seed"]) for r in raw
                if r["ordine"] == ordine}
        found = sum(r["stato"] == "Found" for r in raw if r["ordine"] == ordine)
        print(f"input {ordine}: {len(runs)} esecuzioni, {found} con Found")


def run_sweep() -> None:
    """Estensione: mediana, 30 permutazioni per ogni S della griglia fine."""
    raw = []
    for N in SWEEP_N:
        k = math.ceil(N / 2)
        grid = sweep_grid(N)
        for S in grid:
            theory = theoretical_metrics(N, S)[2]
            for seed in SEEDS:
                raw += run_one(N, S, k, "casuale", seed, theory)
        print(f"N={N}: {len(grid)} valori di S, {len(grid) * len(SEEDS)} esecuzioni")

    write_csv(RESULTS_DIR / "candidates_sweep_raw.csv", RAW_FIELDS, raw)
    write_csv(RESULTS_DIR / "candidates_sweep_summary.csv", SUMMARY_FIELDS,
            summarize(raw))
    found = sum(row["stato"] == "Found" for row in raw)
    print(f"estensione: {len(raw)} passate registrate, {found} con Found")


# ------------------------------------------------------------------- figura

# Configurazione del pannello (b): N = 10^5 con la memoria minima, cioè il
# caso con più passate di campionamento (sei, più la passata finale).
EXAMPLE_N = 100_000
EXAMPLE_S = S_MIN[EXAMPLE_N]

# Colori con lo stesso significato nei due pannelli.
BOUND_COLOR = "#9F2020"     # rosso scuro: limite teorico n_bar_p
GUIDE_COLOR = "#124C87"     # blu scuro: riferimenti (n_bar_p / 4 e soglia S - 2)
GUIDE_FRACTION = 4          # retta guida n_p = n_bar_p / 4, staccata dai dati


def plot_against_bound(ax, summary: list[dict]) -> None:
    """Pannello (a): media di n_p contro n_bar_p, una per configurazione.

    Ogni punto è una configurazione (N, S, p) dell'estensione in S. Le rette
    guida sono il limite n_p = n_bar_p e n_p = n_bar_p / 4: in scala
    logaritmica n_bar_p / 2 cade esattamente a metà tra le due, e una retta
    a 1/2 passerebbe sopra i punti, sembrando un'interpolazione.
    """
    # Stessi colori e marcatori usati per N nella figura dell'esperimento A.
    for N, color, marker in zip(SWEEP_N, COLORS[1:3], MARKERS[1:3]):
        rows = [row for row in summary if int(row["N"]) == N]

        # Linea di tendenza leggera: per ogni valore distinto del limite
        # aggrega i punti che hanno la stessa ascissa. Evita collegamenti
        # verticali artificiosi e lascia tutti i dati originali visibili.
        by_bound: dict[int, list[float]] = {}
        for row in rows:
            by_bound.setdefault(int(row["n_bar"]), []).append(float(row["n_medio"]))
        x_trend = sorted(by_bound)
        y_trend = [statistics.mean(by_bound[x]) for x in x_trend]
        ax.plot(x_trend, y_trend, color=color, linestyle="-",
                linewidth=0.65, alpha=0.40, zorder=2,
                label="_nolegend_")

        # I marcatori restano le misure, non vengono sostituiti dalla linea.
        ax.plot([int(row["n_bar"]) for row in rows],
                [float(row["n_medio"]) for row in rows],
                linestyle="none", marker=marker, markerfacecolor="white",
                markeredgecolor=color, zorder=3,
                label=rf"$N=10^{{{round(math.log10(N))}}}$")

    ends = [60, 1.5e5]
    ax.plot(ends, ends, color=BOUND_COLOR, linestyle="--", linewidth=1.0)
    ax.plot(ends, [x / GUIDE_FRACTION for x in ends], color=GUIDE_COLOR,
            linestyle="--", linewidth=1.0)
    ax.text(1.5e3, 3.4e3, r"$n_p=\bar n_p$", color=BOUND_COLOR, fontsize=9,
            ha="right")
    ax.text(1.6e4, 2.0e3, rf"$n_p=\bar n_p/{GUIDE_FRACTION}$",
            color=GUIDE_COLOR, fontsize=9, ha="left")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(60, 1.5e5)
    ax.set_ylim(15, 1.5e5)
    ax.grid(False, which="minor")
    ax.set_xlabel(r"Limite teorico $\bar n_p$")
    ax.set_ylabel(r"Candidati effettivi $n_p$")
    ax.set_title("(a)", loc="left", fontsize=10)
    ax.legend(loc="upper left")


def plot_pass_by_pass(ax, raw: list[dict]) -> None:
    """Pannello (b): candidati passata dopo passata in una configurazione.

    p = 0 è l'input iniziale, in cui tutti gli N elementi sono candidati e
    il limite vale N. Per p >= 1 si disegnano la media di n_p sulle 30
    permutazioni, con barre tra minimo e massimo, e il limite n_bar_p.
    """
    N, S = EXAMPLE_N, EXAMPLE_S
    values: dict[int, list[int]] = {}
    bounds: dict[int, int] = {}
    for row in raw:
        if int(row["N"]) == N and int(row["S"]) == S:
            p = int(row["passata"])
            values.setdefault(p, []).append(int(row["n"]))
            bounds[p] = int(row["n_bar"])
    if any(len(v) != len(SEEDS) for v in values.values()):
        raise AssertionError(f"N={N}, S={S}: mancano esecuzioni in qualche passata")

    passes = [0] + sorted(values)
    bound = [N] + [bounds[p] for p in passes[1:]]
    mean = [N] + [statistics.mean(values[p]) for p in passes[1:]]
    low = [0] + [m - min(values[p]) for p, m in zip(passes[1:], mean[1:])]
    high = [0] + [max(values[p]) - m for p, m in zip(passes[1:], mean[1:])]

    ax.plot(passes, bound, color=BOUND_COLOR, linestyle="--", linewidth=1.0,
            marker="s", markersize=3.3, label=r"limite teorico $\bar n_p$")
    ax.errorbar(passes, mean, yerr=[low, high], color=COLORS[2],
                marker=MARKERS[2], markersize=4, markerfacecolor="white",
                elinewidth=0.8, label=r"candidati effettivi $n_p$")

    # Soglia della passata finale: quando n_bar_p <= S - 2 i candidati
    # rimasti vengono caricati in memoria. Etichetta piccola, a sinistra,
    # dove il pannello è vuoto.
    ax.axhline(S - 2, color=GUIDE_COLOR, linestyle="--", linewidth=0.8)
    ax.text(0.05, (S - 2) * 1.12, rf"$S-2={S - 2}$", color=GUIDE_COLOR,
            fontsize=8)

    ax.set_yscale("log")
    ax.set_xticks(passes)
    ax.set_xlim(-0.4, passes[-1] + 0.4)
    ax.set_ylim(150, 2e5)
    ax.grid(False, which="minor")
    ax.set_xlabel(r"Passata $P$")
    ax.set_ylabel("Numero di candidati")
    exponent = round(math.log10(N))
    ax.set_title(rf"(b)  $N=10^{{{exponent}}}$, $S={S}$", loc="left",
                fontsize=10)
    ax.legend(loc="upper right")


def plot_figure() -> None:
    """Figura B: candidati effettivi e limite teorico, per la mediana.

    Usa i dati dell'estensione in S. In entrambi i pannelli n_p e n_bar_p
    si riferiscono alla stessa passata (run_one lo controlla per ogni riga).
    """
    raw = [row for row in read_csv(RESULTS_DIR / "candidates_sweep_raw.csv")
        if row["stato"] == "Continue"]
    summary = read_csv(RESULTS_DIR / "candidates_sweep_summary.csv")
    if any(float(row["rapporto"]) > 1 for row in raw):
        raise AssertionError("Trovato n_p > n_bar_p: il limite non è rispettato")

    set_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 7.0 * CM),
                                    layout="constrained")
    plot_against_bound(ax_a, summary)
    plot_pass_by_pass(ax_b, raw)

    path = RESULTS_DIR / "fig_candidates.pdf"
    fig.savefig(path)
    print(f"Figura salvata in: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esperimento B: candidati")
    parser.add_argument("--plot-only", action="store_true",
                        help="rigenera solo la figura dai CSV esistenti")
    args = parser.parse_args()

    if not args.plot_only:
        run_campaign()
        run_sweep()
    plot_figure()
