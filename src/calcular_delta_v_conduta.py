"""
calcular_delta_v_conduta.py

Separa os eventos de ACELERACAO e FRENAGEM ja harmonizados por
src/validacao_hardware.py (fonte unica: Ferreira Jr. et al., 2017) e
CALCULA -- por integracao, nao simulacao -- a variacao de velocidade
(delta-v) de cada evento a partir da propria aceleracao ja medida.
Substitui o script anterior (simular_velocidade_conduta.py), que gerava
duracao e velocidade inicial por regra de negocio; aqui so um numero
(a taxa de amostragem) e citado, nao medido neste arquivo -- tudo o mais
vem do proprio dado real.

Como e calculado, passo a passo, com o que e real e o que e citado:

1. N (numero de amostras da janela) e RECUPERADO do proprio dado real:
   como AccMeanX = AccSumX / N para qualquer eixo, N = AccSumX / AccMeanX.
   Confirmado empiricamente que os tres eixos (X, Y, Z) dao exatamente o
   mesmo N por janela -- nao e uma estimativa, e uma consequencia exata da
   definicao de media. N varia por janela (82 a 239 amostras no dado
   atual) porque as janelas deste dataset tem duracao variavel, casada
   com a duracao real de cada manobra rotulada (confirmado no
   repositorio do dataset: o arquivo de ground truth registra
   inicio/fim por evento, nao janelas de tamanho fixo).

2. A taxa de amostragem do acelerometro (fs) NAO esta no arquivo de
   features ja pre-extraidas -- e o unico numero desta conta que e
   citado, nao medido neste arquivo especifico. O repositorio oficial do
   dataset (github.com/jair-jr/driverBehaviorDataset) documenta que a
   "sampling rate varied between 50 and 200 Hz, depending on the sensor";
   uma publicacao secundaria que analisa este mesmo dataset cita
   especificamente 50 Hz para o canal do acelerometro. Usamos fs=50Hz
   como estimativa central (FS_HZ_CITADO) e reportamos, ao rodar o
   script, a sensibilidade do resultado no limite superior documentado
   (200Hz) -- para deixar essa incerteza visivel, nao escondida.
       duracao_evento_s = n_amostras / fs

3. O vetor de aceleracao MEDIA de cada janela (AccMeanX/Y/Z) inclui a
   gravidade e o offset de montagem do aparelho, que precisam ser
   removidos antes de integrar (o proprio firmware faz essa subtracao de
   calibracao -- Secao 2.8 do escopo tecnico). Esse "baseline parado" e
   estimado a partir das proprias janelas REAIS rotuladas NON_AGGRESSIVE
   da MESMA sessao (GroupID) dos eventos de aceleracao/frenagem -- nao um
   valor assumido de fora do dataset.

4. Pelo teorema do valor medio para integrais, a integral de uma funcao
   sobre um intervalo e igual a sua media multiplicada pela duracao do
   intervalo. Como so temos a MEDIA da aceleracao por janela (nao a
   amostra a amostra), essa e a unica forma de integracao literalmente
   calculavel com os dados disponiveis -- nao uma aproximacao arbitraria:
       aceleracao_dinamica_media = media_janela - baseline_parado
       delta_v (m/s) = |aceleracao_dinamica_media| x duracao_evento_s
       delta_v (km/h) = delta_v (m/s) x 3,6

O QUE ISSO NAO DA: velocidade absoluta (km/h antes/depois do evento).
Integrar aceleracao so recupera uma VARIACAO (delta-v), nunca um valor
absoluto, sem uma condicao de contorno (velocidade inicial conhecida) --
e esta fonte confirmadamente nao tem canal de velocidade/GPS (Part B.5).
Por decisao explicita de Erika (2026-08-30), este script reporta APENAS
delta_v, sem inventar uma velocidade inicial: o resultado e inteiramente
calculado a partir de dado real + uma unica constante citada (fs), sem
nenhuma variavel simulada por regra de negocio.

Fontes:
  - github.com/jair-jr/driverBehaviorDataset (taxa de amostragem; janelas
    de duracao variavel casadas com cada evento rotulado)
  - Ferreira Jr., J. et al. (2017). Driver behavior profiling: An
    investigation with different smartphone sensors and machine
    learning. PLoS ONE, 12(4), e0174959.

Uso:
    python src/calcular_delta_v_conduta.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

IN_PATH = "data/processed/driver_conduct_harmonized.csv"
OUT_PATH = "data/processed/conduta_delta_v.csv"

CATEGORIAS_ALVO = ["ACCELERATION", "BRAKING"]

FS_HZ_CITADO = 50.0  # ver docstring: citado (Ferreira Jr. et al. 2017 / repo), nao medido neste arquivo
FS_HZ_LIMITE_SUPERIOR_DOCUMENTADO = 200.0  # faixa documentada no repositorio oficial (50-200Hz)
MPS2_PARA_KMH_POR_S = 3.6  # 1 m/s^2 sustentado por 1s = 3,6 km/h de delta-v


def recuperar_n_amostras(df: pd.DataFrame) -> pd.Series:
    """N = AccSum / AccMean, exato pela definicao de media -- confirmado
    identico entre os eixos X/Y/Z em todas as janelas do dataset."""
    return (df["AccSumZ"] / df["AccMeanZ"]).round().astype(int)


def calcular_baselines(df: pd.DataFrame, group_ids: list) -> pd.DataFrame:
    """Vetor de aceleracao 'parado' (gravidade + offset de montagem) por
    sessao, estimado a partir das janelas reais NON_AGGRESSIVE dessa
    mesma sessao (GroupID) -- nao um valor assumido de fora do dataset."""
    baselines = df[(df["ValidationRole"] == "baseline") & (df["GroupID"].isin(group_ids))]
    faltando = set(group_ids) - set(baselines["GroupID"].unique())
    if faltando:
        raise ValueError(f"Sem janela NON_AGGRESSIVE real para calcular baseline de: {faltando}")
    return baselines.groupby("GroupID")[["AccMeanX", "AccMeanY", "AccMeanZ"]].mean()


def calcular_delta_v(df: pd.DataFrame, fs_hz: float) -> pd.DataFrame:
    eventos = df[df["EventCategory"].isin(CATEGORIAS_ALVO)].copy()

    eventos["n_amostras"] = recuperar_n_amostras(eventos)
    eventos["duracao_evento_s"] = eventos["n_amostras"] / fs_hz

    baselines = calcular_baselines(df, eventos["GroupID"].unique().tolist())
    dx = eventos["AccMeanX"] - eventos["GroupID"].map(baselines["AccMeanX"])
    dy = eventos["AccMeanY"] - eventos["GroupID"].map(baselines["AccMeanY"])
    dz = eventos["AccMeanZ"] - eventos["GroupID"].map(baselines["AccMeanZ"])
    eventos["aceleracao_dinamica_media_mps2"] = np.sqrt(dx**2 + dy**2 + dz**2)

    sinal = eventos["EventCategory"].map({"ACCELERATION": 1, "BRAKING": -1})
    eventos["delta_v_kmh"] = (
        sinal * eventos["aceleracao_dinamica_media_mps2"] * eventos["duracao_evento_s"] * MPS2_PARA_KMH_POR_S
    )
    return eventos


def montar_saida(eventos: pd.DataFrame) -> pd.DataFrame:
    colunas = [
        "SourceDataset", "GroupID", "EventLabel", "EventCategory", "WindowIndex",
        "n_amostras", "duracao_evento_s", "PeakDynamicAccel_mps2",
        "aceleracao_dinamica_media_mps2", "delta_v_kmh", "HarshPredicted_6mps2",
    ]
    return eventos[colunas]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Separa eventos de aceleracao/frenagem ja harmonizados e CALCULA a "
            "variacao de velocidade (delta-v) por integracao da aceleracao real "
            "(sem simular nenhuma variavel)."
        )
    )
    parser.add_argument("--in", dest="caminho_entrada", default=IN_PATH,
                         help=f"CSV harmonizado de entrada (default: {IN_PATH})")
    parser.add_argument("--out", dest="caminho_saida", default=OUT_PATH,
                         help=f"CSV de saida (default: {OUT_PATH})")
    parser.add_argument("--fs-hz", type=float, default=FS_HZ_CITADO,
                         help=f"taxa de amostragem citada em Hz (default: {FS_HZ_CITADO}, ver docstring)")
    args = parser.parse_args()

    df = pd.read_csv(args.caminho_entrada)
    eventos = calcular_delta_v(df, args.fs_hz)
    saida = montar_saida(eventos)

    out_path = Path(args.caminho_saida)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saida.to_csv(out_path, index=False)

    print(f"{len(saida)} eventos de aceleracao/frenagem separados "
          f"({(saida['EventCategory'] == 'ACCELERATION').sum()} aceleracao, "
          f"{(saida['EventCategory'] == 'BRAKING').sum()} frenagem).")
    print(f"\nfs = {args.fs_hz:.0f} Hz (CITADO -- ver docstring do modulo; repositorio documenta faixa de "
          f"{FS_HZ_CITADO:.0f}-{FS_HZ_LIMITE_SUPERIOR_DOCUMENTADO:.0f} Hz dependendo do sensor).")
    print("n_amostras, duracao_evento_s (via fs) e delta_v_kmh sao CALCULADOS a partir de dado real "
          "(nenhuma variavel simulada por regra de negocio).")
    print(f"\nBase gravada em {out_path} ({len(saida)} linhas, {len(saida.columns)} colunas).")
    print("\nResumo por categoria:")
    print(saida.groupby("EventCategory")[["duracao_evento_s", "delta_v_kmh"]]
          .agg(["mean", "min", "max"]).round(2))

    eventos_alt = calcular_delta_v(df, FS_HZ_LIMITE_SUPERIOR_DOCUMENTADO)
    print(f"\nSensibilidade a fs: no limite superior documentado "
          f"({FS_HZ_LIMITE_SUPERIOR_DOCUMENTADO:.0f} Hz), |delta_v| medio seria "
          f"{eventos_alt['delta_v_kmh'].abs().mean():.2f} km/h (vs "
          f"{saida['delta_v_kmh'].abs().mean():.2f} km/h a {args.fs_hz:.0f} Hz) -- a incerteza da "
          f"taxa de amostragem afeta a duracao, e portanto o delta_v, linearmente.")


if __name__ == "__main__":
    main()
