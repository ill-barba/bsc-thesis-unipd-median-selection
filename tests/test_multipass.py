"""Test di supporto per multipass.py.

Questi test sono stati generati con il supporto di un sistema di intelligenza
artificiale. Non rientrano tra gli obiettivi della tesi: non vogliono validare
il codice come si farebbe in un progetto di software engineering, coprendo ogni
caso possibile, ma sono solo uno strumento per controllare che l'implementazione
Python realizzi quanto descritto nella tesi.

La correttezza dell'algoritmo non è affidata ai test. Poggia sulla base teorica
e matematica riportata nella Tesi (invariante, limiti dei campioni,
dimostrazione di correttezza e di terminazione). I test ne verificano soltanto
la traduzione in codice, su esempi ordinari, casi limite e input casuali
riproducibili. Superarli, anche con copertura completa delle istruzioni, non
equivale a una verifica formale.

Si assumono le precondizioni dell'implementazione: interi distinti, un intero
per riga e nastro riavvolgibile. Il budget S indica celle logiche, non RAM
fisica consumata dall'interprete Python.

Per eseguirli, dalla cartella principale del progetto:
    python -m pytest tests/test_multipass.py -v
"""

import io
import math
import random

import pytest

from src.multipass import (
    CandidateStream,
    PassResult,
    merge,
    mp_select,
    pass_scan,
    relative_rank,
    sample,
    sample_base,
    single_pass,
    thinning,
)


def make_tape(values: list[int]) -> io.StringIO:
    """Simula il nastro di input con un intero per riga."""
    return io.StringIO("\n".join(map(str, values)) + "\n")


def reference_sample(values: list[int], level: int, s: int, padding: float) -> list[int | float]:
    """Costruisce il campione atteso con liste indipendenti dai buffer reali."""
    values_iter = iter(values)

    def build(i: int) -> list[int | float]:
        if i == 0:
            return sorted((next(values_iter, padding) for _ in range(s)), reverse=True)
        first = build(i - 1)[1::2]
        second = build(i - 1)[1::2]
        return sorted(first + second, reverse=True)

    return build(level)


# --------------------------------------------------------------
# CandidateStream: lettura, filtri e contatori
# --------------------------------------------------------------


def test_stream_legge_solo_i_candidati_e_conta_tutto():
    stream = CandidateStream(make_tape([9, 4, 1, 8, 6, 3, 10, 2]), 3, 9)
    assert list(iter(stream.next_cand, None)) == [4, 8, 6]
    assert (stream.q_max, stream.q_min) == (1, 5)
    assert stream.next_cand() is None


def test_stream_salta_righe_vuote_e_legge_interi_negativi():
    tape = io.StringIO("\n -5 \n\n 0\n 6 \n")
    stream = CandidateStream(tape, -5, 6)
    assert stream.next_cand() == 0
    assert stream.next_cand() is None
    assert (stream.q_max, stream.q_min) == (0, 2)


def test_stream_esclude_esattamente_i_due_filtri():
    stream = CandidateStream(make_tape([3, 4, 5, 6, 7]), 4, 6)
    assert list(iter(stream.next_cand, None)) == [5]
    assert (stream.q_max, stream.q_min) == (1, 3)


@pytest.mark.parametrize("values", [[], [5], [-10, 0, 10]])
def test_stream_con_filtri_infiniti(values):
    stream = CandidateStream(make_tape(values), -math.inf, math.inf)
    assert list(iter(stream.next_cand, None)) == values
    assert stream.q_max == 0
    assert stream.q_min == len(values)


def test_stream_finish_scan_completa_contatori_senza_restituire_candidati():
    stream = CandidateStream(make_tape([4, 8, 10, 2, 6, 9]), 3, 9)
    assert stream.next_cand() == 4
    stream.finish_scan()
    assert (stream.q_max, stream.q_min) == (1, 5)
    assert stream.next_cand() is None


def test_stream_senza_candidati_conserva_i_contatori():
    stream = CandidateStream(make_tape([1, 2, 8, 9]), 3, 7)
    assert stream.next_cand() is None
    assert (stream.q_max, stream.q_min) == (2, 2)


# --------------------------------------------------------------
# SAMPLE livello 0 e thinning
# --------------------------------------------------------------


@pytest.mark.parametrize(
    "values, s, expected",
    [
        ([3, 9, 1, 7], 4, [9, 7, 3, 1]),
        ([7, 2], 4, [7, 2, -math.inf, -math.inf]),
        ([], 4, [-math.inf] * 4),
        ([-3, -8, -1], 4, [-1, -3, -8, -math.inf]),
        ([9, 4], 2, [9, 4]),
    ],
)
def test_sample_base_ordinamento_e_padding(values, s, expected):
    stream = CandidateStream(make_tape(values), -math.inf, math.inf)
    buffer = [0] * s
    sample_base(stream, s, buffer)
    assert buffer == expected
    assert len(buffer) == s


def test_sample_base_padding_usa_il_filtro_corrente():
    stream = CandidateStream(make_tape([7, 3, 11, 5]), 4, 10)
    buffer = [0] * 4
    sample_base(stream, 4, buffer)
    assert buffer == [7, 5, 4, 4]
    assert (stream.q_max, stream.q_min) == (1, 3)


@pytest.mark.parametrize("s, buffer", [(0, []), (3, [0] * 3), (-2, []), (4, [0] * 2)])
def test_sample_base_dimensioni_non_valide(s, buffer):
    with pytest.raises(ValueError):
        sample_base(CandidateStream(make_tape([1, 2]), -math.inf, math.inf), s, buffer)


@pytest.mark.parametrize(
    "values, n, expected_valid",
    [
        ([9, 8, 7, 6, 5, 4], 6, [8, 6, 4]),
        ([10, 8], 2, [8]),
        ([10, 9, 8, 7, 0, 0], 4, [9, 7]),
        ([6, 5, 4, 3, 2, 1, 0, -1], 8, [5, 3, 1, -1]),
    ],
)
def test_thinning_seleziona_posizioni_pari_senza_accorciare(values, n, expected_valid):
    original_len = len(values)
    assert thinning(values, n) == n // 2
    assert values[: n // 2] == expected_valid
    assert len(values) == original_len


@pytest.mark.parametrize("n", [-2, 0, 1, 3, 8])
def test_thinning_dimensione_non_valida(n):
    with pytest.raises(ValueError):
        thinning([8, 7, 6, 5], n)


# --------------------------------------------------------------
# MERGE: ordinamento, sovrascrittura e memoria riutilizzata
# --------------------------------------------------------------


@pytest.mark.parametrize(
    "first, second, expected",
    [
        ([9, 5], [8, 3], [9, 8, 5, 3]),
        ([9, 8], [4, 3], [9, 8, 4, 3]),
        ([4, 3], [9, 8], [9, 8, 4, 3]),
        ([8], [7], [8, 7]),
        ([7], [8], [8, 7]),
        ([9, 5], [9, 2], [9, 9, 5, 2]),
        ([-3, -10], [-2, -8], [-2, -3, -8, -10]),
        ([4, -math.inf], [2, -math.inf], [4, 2, -math.inf, -math.inf]),
    ],
)
def test_merge_casi_limite(first, second, expected):
    first_before = first.copy()
    buffer = second + [0] * len(second)
    merge(first, buffer)
    assert buffer == expected
    assert first == first_before


def test_merge_casuale_confrontato_con_sort():
    rng = random.Random(19)
    for h in (1, 2, 3, 8, 25):
        for _ in range(25):
            values = rng.sample(range(-1000, 1000), 2 * h)
            first = sorted(values[:h], reverse=True)
            second = sorted(values[h:], reverse=True)
            buffer = second + [0] * h
            merge(first, buffer)
            assert buffer == sorted(values, reverse=True)


@pytest.mark.parametrize("first, buffer", [([], []), ([3, 2], [1, 0]), ([3], [2, 1, 0])])
def test_merge_dimensioni_non_valide(first, buffer):
    with pytest.raises(ValueError):
        merge(first, buffer)


# --------------------------------------------------------------
# SAMPLE ricorsivo: confronto con costruzione di riferimento
# --------------------------------------------------------------


def test_sample_esempio_noto():
    stream = CandidateStream(make_tape([9, 4, 1, 8, 6, 3, 10, 2]), -math.inf, math.inf)
    buffer = [0] * 4
    saved_halves = [[0, 0]]
    sample(1, 4, stream, buffer, saved_halves)
    assert buffer == [8, 6, 2, 1]


@pytest.mark.parametrize("level", [0, 1, 2, 3])
@pytest.mark.parametrize("s", [2, 4, 6, 10])
def test_sample_ricorsivo_campione_riferimento(level, s):
    rng = random.Random(10000 + level * 100 + s)
    capacity = s * 2**level
    for n in {0, 1, capacity // 2, max(0, capacity - 1), capacity, capacity + 5}:
        values = rng.sample(range(-3000, 3000), n)
        buffer = [0] * s
        saved_halves = [[0] * (s // 2) for _ in range(level)]
        stream = CandidateStream(make_tape(values), -math.inf, math.inf)
        sample(level, s, stream, buffer, saved_halves)
        assert buffer == reference_sample(values, level, s, -math.inf)
        assert len(buffer) == s
        assert len(saved_halves) == level
        assert all(len(half) == s // 2 for half in saved_halves)


def test_sample_continua_dallo_stesso_cursore():
    stream = CandidateStream(make_tape([3, 1, 8, 4, 7, 2]), -math.inf, math.inf)
    buffer = [0] * 2
    sample(1, 2, stream, buffer, [[0]])  # Consuma i primi quattro candidati.
    assert buffer == reference_sample([3, 1, 8, 4], 1, 2, -math.inf)
    assert stream.next_cand() == 7
    assert stream.next_cand() == 2


@pytest.mark.parametrize("level, s, buffer, saved", [
    (-1, 4, [0] * 4, []),
    (2, 4, [0] * 4, [[0, 0]]),
    (0, 3, [0] * 3, []),
    (0, 4, [0] * 3, []),
    (1, 4, [0] * 4, [[0]]),
])
def test_sample_parametri_non_validi(level, s, buffer, saved):
    with pytest.raises(ValueError):
        sample(level, s, CandidateStream(make_tape([1, 2, 3]), -math.inf, math.inf), buffer, saved)


# --------------------------------------------------------------
# PASS_SCAN: campione e ranghi con scansione completa
# --------------------------------------------------------------


def test_pass_scan_esempio_noto():
    stream = CandidateStream(make_tape([9, 4, 1, 8, 6, 3, 10, 2]), 3, 9)
    buffer = [0] * 4
    assert pass_scan(stream, 1, 4, buffer, [[0, 0]]) == (2, 6)
    assert buffer == [6, 3, 3, 3]
    assert stream.next_cand() is None


def test_pass_scan_termina_nastro_anche_se_sample_non_lo_usa_tutto():
    values = [4, 6, 8, 7, 3, 10, 2, 9, 1, 5]
    stream = CandidateStream(make_tape(values), 3, 9)
    buffer = [0] * 2
    ranks = pass_scan(stream, 0, 2, buffer, [])
    assert ranks == (2, 8)
    assert buffer == [6, 4]
    assert stream.tape.read() == ""


def test_pass_scan_con_filtri_infiniti():
    stream = CandidateStream(make_tape([7, 4, 1, 9]), -math.inf, math.inf)
    assert pass_scan(stream, 0, 2, [0, 0], []) == (0, 5)


def test_pass_scan_senza_candidati():
    stream = CandidateStream(make_tape([1, 2, 8, 9]), 3, 7)
    buffer = [0] * 2
    assert pass_scan(stream, 0, 2, buffer, []) == (3, 3)
    assert buffer == [3, 3]


# --------------------------------------------------------------
# Rango relativo e risultato strutturato di PASS
# --------------------------------------------------------------


@pytest.mark.parametrize("k, rank_max, r, expected", [
    (8, 0, 2, (8, 0, 2)),
    (20, 0, 2, (20, 3, 5)),
    (1, 0, 0, (1, 1, 1)),
    (1, 0, 3, (1, -2, 1)),
    (8, 4, 2, (4, -1, 1)),
    (13, 4, 2, (9, 1, 3)),
    (12, 4, 2, (8, 0, 2)),
    (13, 0, 2, (13, 2, 4)),
])
def test_relative_rank_posizioni_uno_based(k, rank_max, r, expected):
    assert relative_rank(k, rank_max, r) == expected


@pytest.mark.parametrize("k, rank_max, r", [(0, 0, 1), (5, 5, 2), (2, 4, 1), (5, 0, -1)])
def test_relative_rank_parametri_non_validi(k, rank_max, r):
    with pytest.raises(ValueError):
        relative_rank(k, rank_max, r)


def test_pass_result_campi_nominati():
    result = PassResult("Continue", None, 10, 90, 30)
    assert result.status == "Continue"
    assert result.value is None
    assert (result.f_min, result.f_max, result.n_bar) == (10, 90, 30)


# --------------------------------------------------------------
# SINGLE_PASS: stati Found/Continue, filtri, bound e input invalido
# --------------------------------------------------------------


def test_single_pass_risultato_noto():
    result = single_pass(make_tape(list(range(1, 1001))), 400, 418,
                         -math.inf, math.inf, 1000)
    assert result.status == "Continue"
    assert result.value is None
    assert (result.f_min, result.f_max, result.n_bar) == (577, 737, 288)


@pytest.mark.parametrize("f_min, f_max, k, expected", [
    (-math.inf, 40, 11, 40),  # Il filtro superiore ha rango 11.
    (20, math.inf, 31, 20),  # Il filtro inferiore ha rango 31.
])
def test_single_pass_trova_un_filtro(f_min, f_max, k, expected):
    result = single_pass(make_tape(list(range(1, 51))), k, 25, f_min, f_max, 50)
    assert result.status == "Found"
    assert result.value == expected


def test_single_pass_u_non_positivo_conserva_filtro_superiore():
    values = list(range(1, 1001))
    result = single_pass(make_tape(values), 1, 418, -math.inf, math.inf, 1000)
    assert result.status == "Continue"
    assert result.f_max == math.inf
    assert result.f_min <= max(values)


def test_single_pass_filtri_delimitano_risposta_e_rispettano_bound():
    rng = random.Random(20261022)
    for _ in range(45):
        values = rng.sample(range(-100_000, 100_000), 1000)
        k = rng.randint(1, len(values))
        result = single_pass(make_tape(values), k, 418, -math.inf, math.inf, 1000)
        wanted = sorted(values, reverse=True)[k - 1]
        if result.status == "Found":
            assert result.value == wanted
        else:
            assert result.value is None
            assert result.f_min <= wanted <= result.f_max
            actual_candidates = sum(result.f_min < x < result.f_max for x in values)
            assert actual_candidates <= result.n_bar
            assert result.n_bar < 1000


def test_single_pass_rilegge_nastro_dall_inizio():
    tape = make_tape(list(range(1, 1001)))
    tape.read(100)
    result = single_pass(tape, 400, 418, -math.inf, math.inf, 1000)
    assert (result.f_min, result.f_max, result.n_bar) == (577, 737, 288)


@pytest.mark.parametrize("n_bar, S", [
    (0, 100),  # Limite candidati non valido.
    (1, 100),
    (1000, 3),  # Il campione non entra in memoria.
    (3, 4),  # Il vincolo s + rs/2 + 2 <= S è violato.
])
def test_single_pass_rifiuta_parametri_o_memoria_insufficienti(n_bar, S):
    with pytest.raises(ValueError):
        single_pass(make_tape([3, 2, 1]), 1, S, -math.inf, math.inf, n_bar)


# --------------------------------------------------------------
# MP_SELECT: selezione finale, più passate, confronto con sorted()
# --------------------------------------------------------------


@pytest.mark.parametrize("values", [
    [42],
    [9, 4, 1, 8, 6, 3, 10, 2],
    [-5, -10, -1, -20, 0],
    [1, 2, 3, 4, 5, 6],
    [6, 5, 4, 3, 2, 1],
    [-10**12, 0, 10**12],
    [2, 1],
])
def test_mp_select_tutti_i_ranghi_senza_campionamento(values):
    expected = sorted(values, reverse=True)
    for k, wanted in enumerate(expected, start=1):
        assert mp_select(make_tape(values), k, len(values) + 2, len(values)) == wanted


def test_mp_select_memoria_esattamente_sufficiente_alla_selezione_finale():
    values = [15, -3, 9, 0, 7, 20]
    for k in range(1, len(values) + 1):
        assert mp_select(make_tape(values), k, 8, 6) == sorted(values, reverse=True)[k - 1]


@pytest.mark.parametrize("k", [1, 2, 100, 400, 500, 900, 999, 1000])
def test_mp_select_con_campionamento(k):
    values = list(range(1, 1001))
    assert mp_select(make_tape(values), k, 418, len(values)) == 1001 - k


def test_mp_select_con_due_passate_di_campionamento():
    values = list(range(1, 10001))
    for k in (1, 100, 3333, 5000, 9999, 10000):
        assert mp_select(make_tape(values), k, 733, len(values)) == 10001 - k


def test_mp_select_input_casuali_riproducibili():
    rng = random.Random(2026)
    for _ in range(200):
        n = rng.randint(1, 2500)
        values = rng.sample(range(-1_000_000, 1_000_000), n)
        k = rng.randint(1, n)
        log_n = math.log2(n) if n > 1 else 0
        S = max(3, math.ceil(4 * log_n**2 + 2 * log_n))
        assert mp_select(make_tape(values), k, S, n) == sorted(values, reverse=True)[k - 1]


def test_mp_select_input_casuali_con_filtro_iniziale_e_memoria_variabile():
    rng = random.Random(77)
    for n in (1, 2, 8, 31, 64, 150, 500, 1000):
        values = rng.sample(range(-100000, 100000), n)
        expected = sorted(values, reverse=True)
        threshold = max(3, math.ceil(4 * math.log2(n)**2 + 2 * math.log2(n)))
        for S in (threshold, max(threshold, n + 2)):
            for k in (1, (n + 1) // 2, n):
                assert mp_select(make_tape(values), k, S, n) == expected[k - 1]


def test_mp_select_riavvolge_nastro_anche_nella_fase_finale():
    tape = make_tape([8, 2, 5, 10, -3, 7])
    tape.read()
    assert mp_select(tape, 3, 8, 6) == 7


@pytest.mark.parametrize("values, k, S, N", [
    ([1, 2], 0, 10, 2),
    ([1, 2], 3, 10, 2),
    ([1, 2], -1, 10, 2),
    ([], 1, 10, 0),
    ([1, 2], 1, 2, 2),
    ([1], 1, 0, 1),
])
def test_mp_select_parametri_non_validi(values, k, S, N):
    with pytest.raises(ValueError):
        mp_select(make_tape(values), k, S, N)


def test_mp_select_mancato_progresso_del_limite_teorico():
    with pytest.raises(ValueError, match="limite dei candidati non diminuisce"):
        mp_select(make_tape(list(range(1, 1001))), 400, 208, 1000)


def test_mp_select_memoria_insufficiente_per_costruire_campione():
    with pytest.raises(ValueError, match="Memoria insufficiente"):
        mp_select(make_tape(list(range(1, 1001))), 400, 3, 1000)

# --------------------------------------------------------------
# Casi meno frequenti: padding ricorsivo e rami di terminazione
# --------------------------------------------------------------


def test_sample_ricorsivo_con_padding_su_filtro_finito():
    values = [17, 5, 4, 14, 3, 20, 8]
    stream = CandidateStream(make_tape(values), 4, 18)
    buffer = [0] * 4
    sample(2, 4, stream, buffer, [[0, 0], [0, 0]])
    candidates = [x for x in values if 4 < x < 18]
    assert buffer == reference_sample(candidates, 2, 4, 4)


def test_single_pass_rifiuta_posizioni_filtri_fuori_limite(monkeypatch):
    import src.multipass as multipass

    # Isola il ramo di validazione: simuliamo una posizione v impossibile.
    monkeypatch.setattr(multipass, "relative_rank", lambda k, rank_max, r: (k, 0, 100_000))
    with pytest.raises(ValueError, match="Posizioni dei filtri non valide"):
        single_pass(make_tape(list(range(1, 1001))), 400, 418,
                    -math.inf, math.inf, 1000)


def test_mp_select_trova_un_filtro_alla_seconda_passata(monkeypatch):
    import src.multipass as multipass

    # Questo caso reale produce Found alla seconda chiamata di PASS.
    original_pass = multipass.single_pass
    calls = 0

    def counting_pass(*args):
        nonlocal calls
        calls += 1
        return original_pass(*args)

    monkeypatch.setattr(multipass, "single_pass", counting_pass)
    values = list(range(1, 10_001))
    assert mp_select(make_tape(values), 256, 733, len(values)) == 9745
    assert calls == 2


def test_mp_select_fase_finale_restituisce_il_filtro_inferiore(monkeypatch):
    import src.multipass as multipass

    # Caso reale, senza simulazioni: dopo le passate di campionamento il nuovo
    # filtro inferiore coincide con l'elemento cercato, e la fase finale lo
    # restituisce. Lo spia verifica che nessuna passata abbia dato Found.
    original_pass = multipass.single_pass
    statuses = []

    def spying_pass(*args):
        result = original_pass(*args)
        statuses.append(result.status)
        return result

    monkeypatch.setattr(multipass, "single_pass", spying_pass)
    values = list(range(1, 2001))
    assert mp_select(make_tape(values), 32, 733, len(values)) == 2001 - 32
    assert statuses and set(statuses) == {"Continue"}


def test_mp_select_puo_restituire_filtro_superiore_nella_fase_finale(monkeypatch):
    import src.multipass as multipass

    # Ramo difensivo: dopo la passata l'elemento cercato coincide con il filtro
    # superiore. Con gli input provati non si raggiunge con dati reali, quindi
    # PASS è simulata per eseguire direttamente questo caso.
    monkeypatch.setattr(
        multipass, "single_pass",
        lambda *args: PassResult("Continue", None, -math.inf, 7, 3),
    )
    values = [7, 9, 3, 11, 2, 6]
    assert mp_select(make_tape(values), 3, 5, len(values)) == 7


def test_demo_main_continua_a_funzionare(tmp_path, monkeypatch, capsys):
    import runpy
    from pathlib import Path
    import src.multipass as multipass

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "demo.txt").write_text("9\n4\n1\n8\n6\n3\n10\n2\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(Path(multipass.__file__).resolve()), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "Candidato: 4",
        "Candidato: 8",
        "Candidato: 6",
        "q_max = 1",
        "q_min = 5",
    ]