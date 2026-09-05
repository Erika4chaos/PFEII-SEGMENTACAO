"""
validacao_hardware.py

Valida a logica de deteccao de eventos do modulo ESP32 (Secao 2.8): o
limiar de magnitude da aceleracao (~6 m/s^2) e aplicado sobre o
UAH-DriveSet (ROMERA; BERGASA; ARROYO, 2016) e a taxa de eventos por minuto
resultante e confrontada, por ANOVA, com o rotulo comportamental conhecido
de cada trajeto (normal / agressiva / sonolenta).

Layout de colunas e delimitador conferidos contra o leitor de referencia do
proprio autor do dataset:
  https://github.com/Eromera/uah_driveset_reader
  http://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/

RAW_ACCELEROMETERS.txt (espaco como delimitador, sem cabecalho):
  0 timestamp (s desde o inicio do trajeto)
  1 ativo_v50 (1 se velocidade > 50 km/h)
  2-4 acc_x, acc_y, acc_z brutos (Gs)
  5-7 acc_x_kf, acc_y_kf, acc_z_kf filtrados por Kalman (Gs)
  8-10 roll, pitch, yaw (graus)
Por convencao do dataset, Z e Y sao as aceleracoes longitudinal e lateral,
respectivamente; X carrega componente vertical/gravitacional residual e por
isso e excluido do calculo da magnitude de manobra -- a mesma logica que o
firmware aplica ao subtrair a gravidade calibrada antes do limiar (Secao 2.8).

RAW_GPS.txt (espaco como delimitador, sem cabecalho; opcional -- usado apenas
para enriquecer o resumo por trajeto, a deteccao de eventos independe dele):
  0 timestamp (s desde o inicio do trajeto)
  1 velocidade (Km/h)
  2-3 latitude, longitude
  4 altitude
  5-6 precisao vertical/horizontal
  7 curso (graus)
  8 var_curso: variacao do curso entre amostras (indicador de ziguezague)
  9-11 estados internos do dataset (posicao, faixa) -- nao utilizados aqui

O rotulo comportamental (normal/agressiva/sonolenta) nao vem em uma coluna:
esta codificado em algum nivel do nome da pasta do trajeto (ex.:
"D1/20151111125233-24km-D1-AGGRESSIVE-MOTORWAY/"). Por isso a deteccao e
feita por busca de palavra-chave no caminho completo, robusta a variacoes
exatas de nomenclatura entre motoristas/versoes do dataset.

Nota: EVENTS_INERTIAL.txt (eventos ja pre-extraidos pelo algoritmo online do
proprio dataset) nao e usado aqui de proposito -- o objetivo desta validacao
e testar a logica de deteccao do FIRMWARE (limiar de magnitude sobre o sinal
bruto do acelerometro), nao reaproveitar a deteccao de outro algoritmo.

Uso:
    python src/validacao_hardware.py
    python src/validacao_hardware.py --dataset-dir data/raw/uah-driveset --limiar 6.0
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway

G_MS2 = 9.80665  # 1 G em m/s^2
LIMIAR_PADRAO_MS2 = 6.0

ACC_COLUNAS = [
    "timestamp", "ativo_v50",
    "acc_x", "acc_y", "acc_z",
    "acc_x_kf", "acc_y_kf", "acc_z_kf",
    "roll", "pitch", "yaw",
]

GPS_COLUNAS = [
    "timestamp", "velocidade_kmh", "latitude", "longitude", "altitude",
    "precisao_vertical", "precisao_horizontal", "curso", "var_curso",
    "estado_posicao", "estado_lanex", "historico_lanex",
]

PALAVRAS_CHAVE_COMPORTAMENTO = {
    "aggressive": "agressiva",
    "drowsy": "sonolenta",
    "sleepy": "sonolenta",
    "normal": "normal",
}

MOTORISTA_RE = re.compile(r"^D\d+$", re.IGNORECASE)


def detectar_comportamento(caminho_trajeto: Path) -> str:
    caminho_str = str(caminho_trajeto).lower()
    for chave, rotulo in PALAVRAS_CHAVE_COMPORTAMENTO.items():
        if chave in caminho_str:
            return rotulo
    return "desconhecido"


def extrair_motorista(caminho_trajeto: Path) -> str:
    """Identifica o motorista pelo segmento de caminho no padrao 'D<numero>'
    (ex.: D1, D2), usado pelo UAH-DriveSet para agrupar os trajetos de um
    mesmo condutor. Usado apenas para exibicao no dashboard."""
    for parte in caminho_trajeto.parts:
        if MOTORISTA_RE.match(parte):
            return parte.upper()
    return "desconhecido"


def carregar_acelerometro(caminho_arquivo: Path) -> pd.DataFrame:
    return pd.read_csv(caminho_arquivo, sep=r"\s+", header=None, names=ACC_COLUNAS)


def carregar_gps(caminho_arquivo: Path) -> pd.DataFrame:
    return pd.read_csv(caminho_arquivo, sep=r"\s+", header=None, names=GPS_COLUNAS)


def resumir_gps(df_gps: pd.DataFrame) -> dict:
    """Velocidade e variacao de curso (ziguezague) agregadas do trajeto,
    usadas apenas como contexto adicional -- nao alimentam a deteccao de
    eventos do ESP32, que depende somente do acelerometro."""
    return {
        "velocidade_media_kmh": df_gps["velocidade_kmh"].mean(),
        "velocidade_maxima_kmh": df_gps["velocidade_kmh"].max(),
        "var_curso_media_abs": df_gps["var_curso"].abs().mean(),
    }


GPS_AUSENTE = {
    "velocidade_media_kmh": np.nan,
    "velocidade_maxima_kmh": np.nan,
    "var_curso_media_abs": np.nan,
}


def calcular_magnitude_ms2(df_acc: pd.DataFrame) -> pd.Series:
    magnitude_g = np.sqrt(df_acc["acc_y_kf"] ** 2 + df_acc["acc_z_kf"] ** 2)
    return magnitude_g * G_MS2


def detectar_eventos(df_acc: pd.DataFrame, limiar_ms2: float = LIMIAR_PADRAO_MS2) -> pd.DataFrame:
    df = df_acc.copy()
    df["magnitude_ms2"] = calcular_magnitude_ms2(df)
    dt = df["timestamp"].diff().replace(0, np.nan)
    df["jerk_ms3"] = df["magnitude_ms2"].diff() / dt
    df["evento"] = df["magnitude_ms2"] > limiar_ms2
    return df


def listar_trajetos(raiz: Path) -> list:
    return sorted({p.parent for p in raiz.rglob("RAW_ACCELEROMETERS.txt")})


def resumir_trajeto(pasta_trajeto: Path, limiar_ms2: float = LIMIAR_PADRAO_MS2) -> dict:
    df_acc = carregar_acelerometro(pasta_trajeto / "RAW_ACCELEROMETERS.txt")
    df_acc = detectar_eventos(df_acc, limiar_ms2)

    duracao_min = (df_acc["timestamp"].max() - df_acc["timestamp"].min()) / 60.0
    n_eventos = int(df_acc["evento"].sum())

    arquivo_gps = pasta_trajeto / "RAW_GPS.txt"
    if arquivo_gps.exists():
        gps_stats = resumir_gps(carregar_gps(arquivo_gps))
    else:
        gps_stats = GPS_AUSENTE

    resumo = {
        "fonte": "UAH-DriveSet (ROMERA; BERGASA; ARROYO, 2016)",
        "trajeto": pasta_trajeto.name,
        "motorista": extrair_motorista(pasta_trajeto),
        "comportamento": detectar_comportamento(pasta_trajeto),
        "duracao_min": duracao_min,
        "n_amostras": len(df_acc),
        "n_eventos": n_eventos,
        "eventos_por_min": n_eventos / duracao_min if duracao_min > 0 else np.nan,
        "magnitude_media_ms2": df_acc["magnitude_ms2"].mean(),
        "magnitude_maxima_ms2": df_acc["magnitude_ms2"].max(),
    }
    resumo.update(gps_stats)
    return resumo


def processar_dataset(raiz: Path, limiar_ms2: float = LIMIAR_PADRAO_MS2) -> pd.DataFrame:
    trajetos = listar_trajetos(raiz)
    return pd.DataFrame(resumir_trajeto(p, limiar_ms2) for p in trajetos)


def validar_discriminacao(df_resumo: pd.DataFrame, coluna: str):
    """ANOVA one-way comparando a coluna informada entre os rotulos
    comportamentais conhecidos (normal/agressiva/sonolenta)."""
    conhecidos = df_resumo[df_resumo["comportamento"] != "desconhecido"]
    grupos = [g[coluna].dropna().to_numpy() for _, g in conhecidos.groupby("comportamento")]
    grupos = [g for g in grupos if len(g) > 0]
    if len(grupos) < 2:
        return None
    return f_oneway(*grupos)


def main():
    parser = argparse.ArgumentParser(
        description="Valida o limiar de deteccao de eventos do firmware contra o UAH-DriveSet."
    )
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--limiar", type=float, default=LIMIAR_PADRAO_MS2, help="limiar de magnitude em m/s^2")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    raiz_projeto = Path(__file__).resolve().parent.parent
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else raiz_projeto / "data" / "raw" / "uah-driveset"
    saida_dir = Path(args.out_dir) if args.out_dir else raiz_projeto / "data" / "processed"

    if not dataset_dir.exists() or not listar_trajetos(dataset_dir):
        print(
            f"Nenhum trajeto (RAW_ACCELEROMETERS.txt) encontrado em {dataset_dir}.\n"
            "Baixe o UAH-DriveSet (ROMERA; BERGASA; ARROYO, 2016) em:\n"
            "  http://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/\n"
            "e extraia de forma que cada pasta de trajeto contenha RAW_ACCELEROMETERS.txt "
            "(ex.: data/raw/uah-driveset/UAH-DRIVESET-v1/D1/<trajeto>/RAW_ACCELEROMETERS.txt)."
        )
        return

    saida_dir.mkdir(parents=True, exist_ok=True)

    df_resumo = processar_dataset(dataset_dir, args.limiar)

    print(f"Limiar de magnitude adotado: {args.limiar} m/s^2\n")
    print(f"{len(df_resumo)} trajetos processados de {dataset_dir}\n")
    print("Resumo por trajeto:")
    print(df_resumo.to_string(index=False))

    print("\nTaxa de eventos por minuto, agregada por comportamento conhecido:")
    print(df_resumo.groupby("comportamento")["eventos_por_min"].agg(["mean", "median", "std", "count"]))

    colunas_anova = ["eventos_por_min"]
    if df_resumo["velocidade_media_kmh"].notna().any():
        colunas_anova += ["velocidade_media_kmh", "var_curso_media_abs"]

    for coluna in colunas_anova:
        resultado_anova = validar_discriminacao(df_resumo, coluna)
        if resultado_anova is not None:
            print(
                f"\nANOVA one-way ({coluna} ~ comportamento): "
                f"F={resultado_anova.statistic:.4f}, p={resultado_anova.pvalue:.6f}"
            )
        else:
            print(f"\nRotulos comportamentais insuficientes para ANOVA em {coluna} (minimo de 2 grupos conhecidos).")

    caminho_saida = saida_dir / "validacao_hardware_uah_driveset.csv"
    df_resumo.to_csv(caminho_saida, index=False)
    print(f"\nResumo por trajeto salvo em: {caminho_saida}")


if __name__ == "__main__":
    main()
