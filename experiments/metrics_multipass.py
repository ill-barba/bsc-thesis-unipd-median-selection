"""Strumenti di misurazione per gli esperimenti su MP-SELECT.

Il file si limita a contare le passate sul nastro (P) e, dopo ogni passata
di campionamento, a confrontare il numero reale di candidati n con
il bound teorico n_bar.

NB: le liste usate per le misure non contano nella memoria S dell'algoritmo.
Servono solo a verificare i risultati, e trarne delle conclusioni analitiche.

Questo file viene importato dagli script degli esperimenti nella cartella
experiments e non va eseguito direttamente.
"""

import csv
import io
import math
import random
import sys
from pathlib import Path
from unittest.mock import patch

# Permette di importare src.multipass dagli script nella cartella experiments
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


def generate_values(N: int, seed: int, ordine: str = "casuale") -> list[int]:
    """Genera i numeri da 1 a N nell'ordine scelto.

    L'ordine casuale è riproducibile usando lo stesso seed. Negli altri
    due casi il seed non viene utilizzato.
    """
    if N <= 0:
        raise ValueError("N deve essere positivo")

    values = list(range(1, N + 1))
    if ordine == "casuale":
        random.Random(seed).shuffle(values)
    elif ordine == "decrescente":
        values.reverse()
    elif ordine != "crescente":
        raise ValueError("Ordine non valido: usare casuale, crescente o decrescente")

    return values


def theoretical_metrics(N: int, S: int) -> tuple[int, int, list[dict]]:
    """Calcola P massimo e celle logiche usando la ricorrenza di n_bar.

    Ripete gli stessi calcoli di s, r e n_bar presenti in single_pass,
    ma senza leggere dati e senza eseguire l'algoritmo.

    Args:
        N: numero di elementi dell'input.
        S: celle logiche disponibili.

    Returns:
        Terna (P_max, celle_max, righe). Le righe descrivono ogni PASS
        teorica e l'ultima passata finale. Per quest'ultima le celle
        indicate sono un limite superiore (n_bar + 2), non una misura
        del numero di candidati realmente caricati.

    Raises:
        ValueError: se N o S non sono validi, il campione non rispetta
            la memoria disponibile oppure n_bar non diminuisce.
    """
    if N <= 0 or S < 3:
        raise ValueError("N oppure memoria S non validi")

    n_bar = N
    rows: list[dict] = []

    while n_bar > S - 2:
        # Stesse formule usate dall'implementazione di PASS.
        s = 2 * math.floor(S / (2 * math.log2(n_bar)))
        if s < 2:
            raise ValueError("Memoria insufficiente per costruire il campione")

        r = ((n_bar + s - 1) // s - 1).bit_length()
        celle = r * s // 2 + s + 2
        if celle > S:
            raise ValueError("Vincolo di memoria S non rispettato")

        new_n_bar = (2 * r - 1) * 2**r
        if new_n_bar >= n_bar:
            raise ValueError("S insufficiente: il limite dei candidati non diminuisce")

        rows.append({
            "passata": len(rows) + 1,
            "fase": "PASS",
            "n_bar_in": n_bar,
            "s": s,
            "r": r,
            "n_bar_out": new_n_bar,
            "celle": celle,
        })
        n_bar = new_n_bar

    # Dopo il campionamento i candidati rimasti entrano tutti in memoria.
    # Qui n_bar + 2 è il massimo teorico per candidati e due filtri.
    rows.append({
        "passata": len(rows) + 1,
        "fase": "Final",
        "n_bar_in": n_bar,
        "s": "",
        "r": "",
        "n_bar_out": n_bar,
        "celle": n_bar + 2,
    })

    P_max = len(rows)
    celle_max = max(row["celle"] for row in rows)
    return P_max, celle_max, rows


def run_experiment(N: int, S: int, k: int, seed: int,
                ordine: str = "casuale") -> tuple[int, int, list[dict]]:
    """Esegue MP-SELECT e misura P e il rapporto n/n_bar dopo ogni PASS.

    Args:
        N: numero di elementi distinti sul nastro.
        S: celle logiche di memoria disponibili.
        k: rango cercato, in ordine decrescente.
        seed: seme per generare la permutazione casuale.
        ordine: casuale, crescente oppure decrescente.

    Returns:
        Una terna (risultato, P, righe). Ogni riga corrisponde a una passata
        di campionamento. Se PASS ritorna Found non ci sono nuovi filtri,
        quindi n e n/n_bar sono vuoti.
    """
    values = generate_values(N, seed, ordine)
    tape = CountingTape(values)
    rows: list[dict] = []
    original_pass = multipass.single_pass

    def observed_pass(tape, k, S, f_min, f_max, n_bar):
        """Esegue una passata e ne registra i risultati."""
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

    # Verifico il risultato: tra 1 e N, il k-esimo più grande vale N + 1 - k
    expected = N + 1 - k
    if answer != expected:
        raise AssertionError(f"Risultato errato! atteso: {expected} invece si è ottenuto: {answer}")

    P = tape.passes
    # Se nessuna PASS ha trovato il risultato, si esegue la passata finale
    # per raccogliere i candidati in memoria: P vale una in più delle righe.
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
