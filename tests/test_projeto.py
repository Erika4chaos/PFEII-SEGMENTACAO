"""Testes minimos do projeto.

Nao medem qualidade do resultado -- garantem que as regras que o texto do TCC
descreve sao as que o codigo executa, e que o painel abre sem erro. Rodar da
raiz do projeto:

    python -m pytest

Os testes que precisam do UAH-DriveSet bruto (data/raw/uah-driveset/) sao
pulados quando a pasta nao esta presente.
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

import validacao_hardware as vh  # noqa: E402
from gerar_base_sintetica import gerar_base, marginais_geradas  # noqa: E402
from preprocessamento import (  # noqa: E402
    COLUNAS_19, COLUNAS_MODELO, parse_qst3, processar,
)

PROCESSADOS = RAIZ / "data" / "processed"
UAH = RAIZ / "data" / "raw" / "uah-driveset"
tem_uah = pytest.mark.skipif(
    not (UAH.exists() and vh.listar_trajetos(UAH)),
    reason="UAH-DriveSet bruto ausente em data/raw/uah-driveset/",
)
PESOS = {"alto_risco": 0.30, "alta_cobertura": 0.35, "referral": 0.35}


# ---------------------------------------------------------------------------
# Regra de deteccao de evento (espelho do firmware)
# ---------------------------------------------------------------------------

def test_travessias_dentro_do_refratario_contam_um_evento():
    tempo = np.arange(0, 10, 0.1)
    magnitude = np.zeros_like(tempo)
    magnitude[10:15] = 4.0   # 1,0 s a 1,4 s: uma manobra
    magnitude[20:22] = 4.0   # 2,0 s: ainda dentro dos 2 s do refratario
    magnitude[60:62] = 4.0   # 6,0 s: evento novo
    assert vh.contar_eventos_agrupados(magnitude, tempo, 2.9, refratario_s=2.0) == 2


def test_marcador_cai_na_primeira_amostra_de_cada_evento():
    tempo = np.arange(0, 10, 0.1)
    magnitude = np.zeros_like(tempo)
    magnitude[10:15] = 4.0
    magnitude[60:62] = 4.0
    inicio = vh.marcar_inicios_de_evento(magnitude, tempo, 2.9, refratario_s=2.0)
    assert np.flatnonzero(inicio).tolist() == [10, 60]


def test_sinal_abaixo_do_limiar_nao_gera_evento():
    tempo = np.arange(0, 5, 0.1)
    assert vh.contar_eventos_agrupados(np.full_like(tempo, 2.8), tempo, 2.9) == 0


def test_limiar_e_refratario_vem_do_config_do_firmware():
    texto = vh.CONFIG_FIRMWARE.read_text(encoding="utf-8", errors="ignore")
    assert f"#define LIMIAR_MAGNITUDE_MS2 {vh.THRESH_MAG_MS2}f" in texto
    assert vh.REFRATARIO_S == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Trecho inicial sem calibracao de pose
# ---------------------------------------------------------------------------

def _sinal(offsets_por_janela, hz=10.0, janela_s=30.0, ruido=0.02, semente=0):
    """Acelerometro sintetico: um offset em g no eixo y para cada janela."""
    rng = np.random.default_rng(semente)
    n = int(hz * janela_s)
    y = np.concatenate([rng.normal(off, ruido, n) for off in offsets_por_janela])
    tempo = np.arange(len(y)) / hz
    return pd.DataFrame({"timestamp": tempo, "acc_y_kf": y,
                         "acc_z_kf": rng.normal(0, ruido, len(y))})


def test_trajeto_limpo_nao_perde_nada():
    assert vh.duracao_sem_calibracao(_sinal([0.0] * 8)) == 0.0


def test_offset_desde_o_inicio_e_excluido_ate_onde_dura():
    # 0,35 g (~3,4 m/s^2) nas tres primeiras janelas, depois calibrado.
    assert vh.duracao_sem_calibracao(_sinal([0.35] * 3 + [0.0] * 6)) == 90.0


def test_offset_isolado_no_meio_do_trajeto_nao_e_excluido():
    # Pode ser uma curva longa de verdade: so o bloco que comeca na primeira
    # amostra e tratado como falta de calibracao.
    assert vh.duracao_sem_calibracao(_sinal([0.0, 0.0, 0.35, 0.0, 0.0])) == 0.0


def test_recalibracao_em_dois_tempos_entra_no_mesmo_trecho():
    # Uma janela limpa entre duas com offset (30 s <= folga de 60 s).
    assert vh.duracao_sem_calibracao(_sinal([0.35, 0.35, 0.0, 0.35, 0.0, 0.0])) == 120.0


# ---------------------------------------------------------------------------
# Pre-processamento (Etapa 1)
# ---------------------------------------------------------------------------

def test_parse_qst3_soma_valores_e_identifica_o_tipo():
    texto = ("Seguradora Anterior: Alfa | Tipo: Tombamento | Pago: R$ 10.000,00 | "
             "Valor sinistro: R$ 120.500,50. Seguradora Anterior: Beta | "
             "Tipo: Incêndio | Pago: R$ 5.000,00 | Valor sinistro: R$ 80.000,00")
    resultado = parse_qst3(texto)
    assert resultado["valor_pago_historico"] == pytest.approx(15000.0)
    assert resultado["valor_sinistro_historico"] == pytest.approx(200500.50)
    assert resultado["qt_cias_anteriores"] == 2
    # Empate entre dois tipos: vence o primeiro citado (tombamento = 1).
    assert resultado["tipo_sinistro_predominante"] == 1


def test_parse_qst3_de_texto_vago_nao_inventa_valor():
    resultado = parse_qst3("em anexo")
    assert resultado["valor_pago_historico"] == 0
    assert resultado["tipo_sinistro_predominante"] == 0


def test_pipeline_da_etapa_1_entrega_matriz_completa(tmp_path):
    base = gerar_base(150, 7, PESOS, date(2026, 9, 1), 0.04)
    caminho = tmp_path / "base.json"
    caminho.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")
    original, normalizada = processar(caminho)

    assert original["numeroApolice"].is_unique
    assert not original[COLUNAS_19].isna().any().any(), "imputacao deixou lacuna"
    matriz = normalizada[COLUNAS_MODELO]
    assert not matriz.isna().any().any()
    # Tolerancia de ponto flutuante: o Min-Max devolve 1.0000000000000009.
    assert matriz.min().min() >= -1e-9 and matriz.max().max() <= 1 + 1e-9


def test_gerador_e_reprodutivel_e_sem_rotulo_disfarcado():
    a = gerar_base(200, 11, PESOS, date(2026, 9, 1), 0.04)
    b = gerar_base(200, 11, PESOS, date(2026, 9, 1), 0.04)
    assert a == b
    # Nenhum perfil pode ter desconto, agravo ou juros em 0% ou 100% dos casos.
    grande = gerar_base(1500, 3, PESOS, date(2026, 9, 1), 0.0)
    marginais = marginais_geradas(grande)
    assert 0 < marginais["qst1"] < 1 and 0 < marginais["qst2"] < 1


# ---------------------------------------------------------------------------
# Coerencia entre os arquivos que o painel le
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (PROCESSADOS / "grade_calibracao.csv").exists(),
                    reason="rode src/validacao_hardware.py antes")
def test_grade_e_resumo_por_trajeto_contam_os_mesmos_sinalizados():
    grade = pd.read_csv(PROCESSADOS / "grade_calibracao.csv")
    trajetos = pd.read_csv(PROCESSADOS / "uah_trips.csv")
    linha = grade[((grade["limiar_ms2"] - vh.THRESH_MAG_MS2).abs() < 1e-9)
                  & (grade["taxa_min_eventos_min"] == 0)].iloc[0]
    for rotulo in ("normal", "agressiva", "sonolenta"):
        sinalizados = int((trajetos.loc[trajetos["estilo"] == rotulo, "eventos"] > 0).sum())
        assert sinalizados == int(linha[f"{rotulo}_sinalizados"]), rotulo


@pytest.mark.skipif(not (PROCESSADOS / "matriz_original_clusters.csv").exists(),
                    reason="rode src/clustering.py antes")
def test_cada_cluster_cai_em_um_perfil_diferente():
    from dashboard.app import mapear_cluster_para_perfil
    original = pd.read_csv(PROCESSADOS / "matriz_original_clusters.csv")
    mapa = mapear_cluster_para_perfil(original)
    assert sorted(mapa.values()) == [1, 2, 3]


@tem_uah
def test_limiar_do_firmware_e_o_p999_da_conducao_normal():
    magnitudes = vh.coletar_magnitudes(UAH)
    assert vh.derivar_limiar_recalibrado(magnitudes) == pytest.approx(vh.THRESH_MAG_MS2, abs=0.05)


@tem_uah
def test_nenhum_trajeto_valido_comeca_descalibrado():
    for pasta in vh.listar_trajetos(UAH):
        assert vh.duracao_sem_calibracao(vh.carregar_sinal_valido(pasta)) == 0.0, pasta.name


# ---------------------------------------------------------------------------
# O painel abre
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (PROCESSADOS / "matriz_normalizada_clusters.csv").exists(),
                    reason="rode o pipeline antes")
def test_dashboard_renderiza_sem_excecao():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(RAIZ / "dashboard" / "app.py"), default_timeout=180)
    app.run()
    assert not app.exception, [str(e.value) for e in app.exception]
