"""
simular_velocidade_conduta.py

Separa os eventos de ACELERACAO e FRENAGEM ja harmonizados por
src/validacao_hardware.py (fonte unica: Ferreira Jr. et al., 2017) e simula,
por regras de negocio, um perfil de velocidade plausivel para cada evento --
exatamente porque essa fonte NAO tem canal real de GPS/velocidade (ver Part
B.5 do escopo tecnico e a docstring de validacao_hardware.py).

Isto e um exercicio de dado SINTETICO/ilustrativo, no mesmo espirito de
src/gerar_base_sintetica.py (Faker + regras de negocio para a base de
apolices): aqui a "semente" real e a magnitude de aceleracao de pico ja
medida (`PeakDynamicAccel_mps2`), e a duracao do evento e a velocidade
inicial sao simuladas dentro de faixas plausiveis de transito urbano, ja
que nao existem no dado original. A velocidade final e derivada por
cinematica simples (delta-v = a * t) a partir da aceleracao real e da
duracao simulada.

IMPORTANTE -- separacao metodologica deliberada: esta e uma extensao
exploratoria pedida a parte, NAO uma correcao da limitacao documentada em
Part B.5 ("No speed- or GPS-linked hardware validation"). As colunas
geradas aqui carregam o sufixo `_simulada(o)` e o script grava um aviso
junto do CSV de saida. Este dado NAO deve ser lido de volta por
validacao_hardware.py nem usado para recalcular precisao/recall/F1 --
faze-lo apresentaria uma variavel simulada como se fosse validacao real de
velocidade, o que o proprio escopo tecnico proibe explicitamente.

Uso:
    python src/simular_velocidade_conduta.py
"""

import argparse
import random
from pathlib import Path

import pandas as pd

IN_PATH = "data/processed/driver_conduct_harmonized.csv"
OUT_PATH = "data/processed/conduta_velocidade_simulada.csv"

CATEGORIAS_ALVO = ["ACCELERATION", "BRAKING"]

# Faixas assumidas (nao medidas -- ver docstring do modulo). Duracao tipica
# de uma frenagem/aceleracao brusca urbana; velocidade inicial tipica antes
# de cada tipo de evento (frenagem parte de uma velocidade de cruzeiro mais
# alta; aceleracao brusca tende a partir de uma velocidade baixa, ex. saida
# de semaforo).
FAIXAS_DURACAO_S = {"ACCELERATION": (1.0, 3.0), "BRAKING": (0.8, 2.5)}
FAIXAS_VELOCIDADE_INICIAL_KMH = {"ACCELERATION": (0.0, 40.0), "BRAKING": (20.0, 80.0)}
VELOCIDADE_MAXIMA_KMH = 120.0

MPS2_PARA_KMH_POR_S = 3.6  # 1 m/s^2 sustentado por 1s = 3.6 km/h de delta-v


def carregar_eventos_alvo(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    eventos = df[df["EventCategory"].isin(CATEGORIAS_ALVO)].copy()
    return eventos.reset_index(drop=True)


def simular_velocidade(eventos: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    duracoes, vel_iniciais = [], []
    for categoria in eventos["EventCategory"]:
        d_min, d_max = FAIXAS_DURACAO_S[categoria]
        v_min, v_max = FAIXAS_VELOCIDADE_INICIAL_KMH[categoria]
        duracoes.append(rng.uniform(d_min, d_max))
        vel_iniciais.append(rng.uniform(v_min, v_max))

    eventos = eventos.copy()
    eventos["duracao_evento_simulada_s"] = duracoes
    eventos["velocidade_inicial_simulada_kmh"] = vel_iniciais
    eventos["delta_v_simulado_kmh"] = (
        eventos["PeakDynamicAccel_mps2"] * eventos["duracao_evento_simulada_s"] * MPS2_PARA_KMH_POR_S
    )

    sinal = eventos["EventCategory"].map({"ACCELERATION": 1, "BRAKING": -1})
    velocidade_final = eventos["velocidade_inicial_simulada_kmh"] + sinal * eventos["delta_v_simulado_kmh"]
    eventos["velocidade_final_simulada_kmh"] = velocidade_final.clip(lower=0.0, upper=VELOCIDADE_MAXIMA_KMH)

    return eventos


def montar_saida(eventos: pd.DataFrame) -> pd.DataFrame:
    colunas = [
        "SourceDataset", "GroupID", "EventLabel", "EventCategory", "WindowIndex",
        "PeakDynamicAccel_mps2", "HarshPredicted_6mps2",
        "duracao_evento_simulada_s", "velocidade_inicial_simulada_kmh",
        "delta_v_simulado_kmh", "velocidade_final_simulada_kmh",
    ]
    return eventos[colunas]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Separa eventos de aceleracao/frenagem ja harmonizados e simula um "
            "perfil de velocidade plausivel (dado sintetico/ilustrativo -- a fonte "
            "real nao tem canal de velocidade)."
        )
    )
    parser.add_argument("--in", dest="caminho_entrada", default=IN_PATH,
                         help=f"CSV harmonizado de entrada (default: {IN_PATH})")
    parser.add_argument("--out", dest="caminho_saida", default=OUT_PATH,
                         help=f"CSV de saida (default: {OUT_PATH})")
    parser.add_argument("--seed", type=int, default=42, help="semente para reprodutibilidade")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    eventos = carregar_eventos_alvo(args.caminho_entrada)
    print(f"{len(eventos)} eventos de aceleracao/frenagem separados de {args.caminho_entrada} "
          f"({(eventos['EventCategory'] == 'ACCELERATION').sum()} aceleracao, "
          f"{(eventos['EventCategory'] == 'BRAKING').sum()} frenagem).")

    eventos = simular_velocidade(eventos, rng)
    saida = montar_saida(eventos)

    out_path = Path(args.caminho_saida)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saida.to_csv(out_path, index=False)

    print(f"\nAVISO: velocidade_inicial/final e duracao sao SIMULADAS por regras de "
          f"negocio (faixas plausiveis de transito urbano), nao medidas -- a fonte "
          f"Ferreira Jr. et al. (2017) nao tem canal de GPS/velocidade (Part B.5). "
          f"So `PeakDynamicAccel_mps2` e `HarshPredicted_6mps2` vem do dado real.")
    print(f"Base simulada gravada em {out_path} ({len(saida)} linhas, {len(saida.columns)} colunas).")
    print("\nResumo por categoria:")
    print(saida.groupby("EventCategory")[
        ["duracao_evento_simulada_s", "velocidade_inicial_simulada_kmh", "velocidade_final_simulada_kmh"]
    ].mean().round(1))


if __name__ == "__main__":
    main()
