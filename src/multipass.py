"""Simulazione del nastro di input e lettura dei candidati durante una passata.

Precondizione: gli N elementi dell'input sono tutti distinti, come descritto nel
documento. Il codice non può verificarlo in generale perchè controllarlo dentro l'algoritmo
richiederebbe di ricordare gli elementi già letti, cioè O(N) memoria,
contravvenendo al vincolo su S. Il controllo, se serve, va fatto fuori dall'algoritmo.
"""
# Tipo per flussi di input come ad es: file aperti, stdin, ecc.
from typing import TextIO


class CandidateStream:
    """Lettura sequenziale dei candidati durante una passata."""

    def __init__(
        self,
        tape: TextIO,
        f_min: int | float,
        f_max: int | float,
    ) -> None:
        """Inizializza lo stream con il nastro e i filtri correnti.

        Args:
            tape: flusso di testo leggibile sequenzialmente.
            f_min: filtro inferiore corrente.
            f_max: filtro superiore corrente.
        """
        self.tape = tape
        self.f_min = f_min
        self.f_max = f_max
        # servono per il rango dei filtri, numero di elementi letti
        # che sono più grandi di f_max e di f_min
        self.q_max = 0
        self.q_min = 0

    def _read_next(self) -> int | None:
        """Legge il prossimo elemento del nastro e aggiorna i contatori.

        Returns:
            L'elemento letto, oppure None se il nastro è finito.
        """
        while True:
            line = self.tape.readline()
            if line == "":  # se fine file
                return None

            line = line.strip()
            if not line:  # se riga vuota nel file, la salto
                continue

            x = int(line)
            # conta gli elementi maggiori di ciascun filtro, anche se non candidati
            if x > self.f_max:
                self.q_max += 1
            if x > self.f_min:
                self.q_min += 1
            return x

    def next_cand(self) -> int | None:
        """Restituisce il prossimo candidato, oppure None a fine nastro."""
        while True:
            x = self._read_next()
            if x is None:
                return None
            # gli elementi fuori dai filtri vengono letti (e contati da _read_next) ma non restituiti
            if self.f_min < x < self.f_max:
                return x

    def finish_scan(self) -> None:
        """Legge ciò che rimane del nastro, per completare i contatori."""
        # Completa la lettura del nastro se SAMPLE termina in anticipo,
        # garantendo che q_max e q_min siano calcolati su tutti gli N elementi.
        while True:
            x = self._read_next()
            if x is None:
                break


def sample_base(stream: CandidateStream, s: int) -> list[int | float]:
    """Costruisce il campione di livello 0 con s candidati in ordine decrescente.

    Se il nastro finisce prima di aver letto s candidati, il campione viene
    completato con elementi fittizi (padding). Gli elementi di padding valgono 
    tutti f_min e si considerano distinti per posizione occupata.

    Args:
        stream: sequenza da cui leggere i candidati (il cursore è condiviso).
        s: dimensione del campione, che è intero positivo e pari.

    Returns:
        Lista di esattamente s elementi, ordinata in modo decrescente.
    """
    if s <= 0 or s % 2 != 0:
        raise ValueError("s deve essere un intero positivo e pari")

    sample: list[int | float] = []
    for _ in range(s):
        x = stream.next_cand()
        if x is None:
            # Il padding usa f_min, sempre inferiore ai candidati reali
            # NB: un valore fisso (es. 0) potrebbe invece rientrare tra i filtri
            x = stream.f_min
        sample.append(x)

    sample.sort(reverse=True)
    return sample


def thinning(sample: list[int | float]) -> None:
    """Dimezza il campione tenendo le posizioni pari, direttamente nella lista.

    Il campione deve avere dimensione positiva, pari ed essere già ordinato
    in modo decrescente. Al termine conterrà s/2 elementi.

    Args:
        sample: campione di s elementi, modificato in-place.

    Raises:
        ValueError: se la dimensione non è un intero positivo e pari.
    """
    s = len(sample)
    if s <= 0 or s % 2 != 0:
        raise ValueError("Il campione deve avere dimensione positiva e pari")

    # Sposta gli elementi nelle posizioni pari all'inizio della stessa lista,
    # gli elementi sovrascritti non serviranno più
    for j in range(s // 2):
        sample[j] = sample[2*j + 1]

    # Elimina la parte della lista che non serve più
    del sample[s // 2:]



def merge(first_subsample: list[int | float], buffer: list[int | float]) -> None:
    """Fonde due sottocampioni ordinati nel buffer di lavoro, sul posto.

    All'ingresso `first_subsample` contiene il primo sottocampione (h elementi)
    e `buffer` ha 2h celle: nelle quali le prime h contengono il secondo sottocampione,
    le altre h, invece, sono libere. Entrambi i sottocampioni sono ordinati in modo 
    decrescente. Alla fine `buffer` conterrà i 2h elementi fusi, in ordine decrescente,
    mentre `first_subsample` non cambia.

    Args:
        first_subsample: primo sottocampione, ordinato in modo decrescente.
        buffer: buffer di lavoro di 2h celle, già allocato alla chiamata.

    Raises:
        ValueError: se il primo sottocampione è vuoto o il buffer non ha
            esattamente 2h celle.
    """
    h = len(first_subsample)
    if h == 0 or len(buffer) != 2 * h:
        raise ValueError(
            "Il buffer deve avere dimensione doppia rispetto al sottocampione"
        )

    i = h - 1       # ultimo elemento del primo sottocampione ancora da copiare
    j = h - 1       # ultimo elemento del secondo sottocampione (nel buffer)
    k = 2 * h - 1   # prossima cella del buffer in cui scrivere

    while i >= 0 and j >= 0:
        if first_subsample[i] < buffer[j]:
            buffer[k] = first_subsample[i]
            i -= 1
        else:
            buffer[k] = buffer[j]
            j -= 1
        k -= 1

    # Se è finito il secondo sottocampione, restano elementi del primo
    # da copiare. Se è finito il primo, quelli rimasti nel buffer sono
    # già nella posizione corretta
    while i >= 0:
        buffer[k] = first_subsample[i]
        i -= 1
        k -= 1



# test con data/demo.txt
# NB: viene eseguito solo se il file viene lanciato come main,
# quindi ignorato se viene importato.
if __name__ == "__main__":
    with open("data/demo.txt", encoding="utf-8") as tape:
        stream = CandidateStream(tape, f_min=3, f_max=9)

        while True:
            candidate = stream.next_cand()
            if candidate is None:
                break
            print("Candidato:", candidate)

        print("q_max =", stream.q_max)
        print("q_min =", stream.q_min)