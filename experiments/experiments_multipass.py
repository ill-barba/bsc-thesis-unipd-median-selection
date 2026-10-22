"""Parte sperimentale sull'algoritmo MP-SELECT.

Il file si limita a contare le passate sul nastro (P) e, dopo ogni passata 
di campionamento, a confrontare il numero reale di candidati n con 
il bound teorico n_bar.

NB: le liste usate per le misure non contano nella memoria S dell'algoritmo.
Servono solo a verificare i risultati, e trarne delle conclusioni analitiche.

Per eseguire dalla cartella principale del progetto si consiglia:
    python experiments/experiments_multipass.py
"""

import csv
import io
import random
import sys
from pathlib import Path
from unittest.mock import patch

# Così il file si può lanciare anche da dentro la cartella experiments
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import multipass  # noqa: E402 (deve stare dopo sys.path.insert)


class CountingTape(io.StringIO):
    """Nastro di testo che conta quante volte viene riavvolto con seek(0).

    Ogni seek(0) segna l'inizio di una nuova scansione completa, quindi
    l'inizio di una nuova passata. Dunque il contatore coincide con P. 
    NB: non contiamo le readline, a fine nastro il padding ne fa molte in più 
    e il conto sarebbe falsato.
    """

    def __init__(self, values: list[int]) -> None:
        """Crea il nastro di testo con un intero per riga."""
        super().__init__("\n".join(map(str, values)) + "\n")
        self.passes = 0

    def seek(self, pos: int, whence: int = 0) -> int:
        """Riavvolge il nastro e, se riparte da capo, conta una passata."""
        if pos == 0 and whence == 0:
            self.passes += 1
        return super().seek(pos, whence)


def generate_values(N: int, seed: int) -> list[int]:
    """Genera una permutazione dei numeri da 1 a N, sempre la stessa per un dato seed."""
    if N <= 0:
        raise ValueError("N deve essere positivo")

    values = list(range(1, N + 1))
    random.Random(seed).shuffle(values)
    return values


def run_experiment(N: int, S: int, k: int, seed: int) -> tuple[int, int, list[dict]]:
    """Esegue MP-SELECT e misura P e il rapporto n/n_bar dopo ogni PASS.

    Args:
        N: numero di elementi distinti sul nastro.
        S: celle logiche di memoria disponibili.
        k: rango cercato, in ordine decrescente.
        seed: seme per generare la permutazione dell'input.

    Returns:
        Una terna (risultato, P, righe). Ogni riga corrisponde a una passata
        di campionamento. Se PASS ritorna Found non ci sono nuovi filtri,
        quindi n e n/n_bar sono vuoti.
    """
    values = generate_values(N, seed)
    tape = CountingTape(values)
    rows: list[dict] = []
    original_pass = multipass.single_pass

    def observed_pass(tape, k, S, f_min, f_max, n_bar):
        """Esegue una instantanea di ogni passata e dei relativi risultati."""
        result = original_pass(tape, k, S, f_min, f_max, n_bar)
        row = {
            "passata": len(rows) + 1,
            "stato": result.status,
            "f_min": "",
            "f_max": "",
            "n": "",
            "n_bar": "",
            "n_su_n_bar": "",
            "f_max_invariato": "",
        }

        if result.status == "Continue":
            # n viene misurato da fuori, conto i valori tra i nuovi filtri
            n = sum(result.f_min < x < result.f_max for x in values)
            if n > result.n_bar:
                raise AssertionError("Il numero di candidati supera il limite teorico")

            row.update({
                "f_min": result.f_min,
                "f_max": result.f_max,
                "n": n,
                "n_bar": result.n_bar,
                "n_su_n_bar": n / result.n_bar,
                "f_max_invariato": result.f_max == f_max,
            })

        rows.append(row)
        return result

    try:
        # patch.object sostituisce single_pass solo dentro questo blocco with
        # all'uscita viene rimessa quella originale
        with patch.object(multipass, "single_pass", observed_pass):
            answer = multipass.mp_select(tape, k, S, N)
    finally:
        tape.close()

    # Eseguo re-mapping, una permutazione di 1..N il k-esimo più grande vale N + 1 - k
    expected = N + 1 - k
    if answer != expected:
        raise AssertionError(f"Risultato errato! atteso: {expected} invece si è ottenuto: {answer}")

    P = tape.passes
    # Se l'ultima PASS non ha trovato il risultato, c'è ancora la passata finale
    # in memori, per questo P vale una in più delle righe.
    found = bool(rows and rows[-1]["stato"] == "Found")
    expected_passes = len(rows) if found else len(rows) + 1
    if P != expected_passes:
        raise AssertionError("Conteggio delle passate non coerente")

    return answer, P, rows


def save_csv(path: Path, N: int, S: int, k: int, seed: int,
            answer: int, P: int, rows: list[dict]) -> None:
    """Salva l'esecuzione in un CSV, con una riga per ogni passata PASS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "N", "S", "k", "seed", "risultato", "P",
        "passata", "stato", "f_min", "f_max",
        "n", "n_bar", "n_su_n_bar", "f_max_invariato",
    ]

    # Se tutto il nastro entra subito in memoria PASS non viene mai chiamata,
    # ma scriviamo comunque una riga per registrare l'unica esecuzione (P = 1)
    if not rows:
        rows = [{"passata": 0, "stato": "FinalOnly"}]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "N": N, "S": S, "k": k, "seed": seed,
                "risultato": answer, "P": P, **row,
            })


if __name__ == "__main__":
    # prova
    N = 1000
    S = 418
    k = 400
    seed = 2026

    answer, P, rows = run_experiment(N, S, k, seed)
    csv_path = Path(__file__).resolve().parent / "results" / "demo.csv"
    save_csv(csv_path, N, S, k, seed, answer, P, rows)

    print(f"N={N}, S={S}, k={k}, risultato={answer}, P={P}")
    for row in rows:
        if row["stato"] == "Continue":
            print(
                f"Passata {row['passata']}: "
                f"n={row['n']}, n_bar={row['n_bar']}, "
                f"n/n_bar={row['n_su_n_bar']:.3f}"
            )
    print(f"Risultati salvati in: {csv_path}")