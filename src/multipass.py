"""Simulazione del nastro di input e lettura dei candidati durante una passata."""
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