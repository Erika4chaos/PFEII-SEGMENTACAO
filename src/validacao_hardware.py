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

RAW_ACCELEROMETERS.txt (espaco como delimitador, sem cabecalho) -- layout
CONFERIDO CONTRA OS ARQUIVOS REAIS (2026-09-05), nao apenas contra a
documentacao publica:
  0 timestamp (s desde o inicio do trajeto)
  1 ativo_v50 (1 se velocidade > 50 km/h) -- confirmado: so assume {0, 1}
  2-4 acc_x, acc_y, acc_z brutos (Gs)
  5-7 acc_x_kf, acc_y_kf, acc_z_kf filtrados por Kalman (Gs)
  8-10 roll, pitch, yaw -- em RADIANOS, nao em graus como diz a documentacao
       publica (yaw varre exatamente -pi..+pi; roll tem media ~ -pi/2).
       Nao sao usados em nenhum calculo aqui, mas a unidade estava errada.

Duas correcoes empiricas importantes em relacao ao que se supunha antes:

(1) NENHUM dos seis canais de aceleracao carrega gravidade. A magnitude do
    vetor 3-eixos dos canais "brutos" tem media ~0.05 G (e nao ~1.0 G), e a
    media de cada canal fica proxima de zero. Ou seja, o dataset ja entrega
    aceleracao dinamica no referencial do veiculo, com a gravidade removida
    na origem -- este script NAO precisa subtrair gravidade, e a afirmacao
    anterior de que o eixo X carregaria "componente gravitacional residual"
    estava incorreta.

(2) Os canais _kf sao versoes filtradas (Kalman) dos MESMOS eixos brutos --
    confirmado por correlacao 0.73/0.89/0.93 entre cada par bruto/filtrado e
    desvio-padrao menor no filtrado nos tres eixos. Nao sao um segundo
    conjunto de sensores nem uma etapa de remocao de gravidade.

Por convencao do dataset (ROMERA et al., 2016, Secao III), Z e Y sao as
aceleracoes longitudinal e lateral, respectivamente -- os dois eixos que
importam para frenagem/aceleracao/curva. X (vertical) e excluido da
magnitude de manobra por nao corresponder a nenhuma dessas manobras.

ATENCAO -- diferenca em relacao ao firmware descrito na Secao 2.8: o
firmware completo faz subtracao de gravidade por calibracao estacionaria,
media movel passa-baixa e usa magnitude + jerk. O que e validado aqui e uma
versao simplificada: magnitude de 2 eixos sobre o sinal que o dataset ja
entrega condicionado, SEM criterio de jerk (o jerk e calculado e persistido
para inspecao, mas nao entra na deteccao). Essa diferenca precisa estar
declarada no texto do TCC -- nao e o firmware inteiro que esta validado.

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

Antes de reportar qualquer resultado de nao-discriminacao (taxa de eventos
proxima de zero), este script confere empiricamente a taxa de amostragem, a
faixa fisica do sinal e o efeito do filtro de Kalman -- ver
verificar_calibracao_sinal(). Uma taxa de eventos baixa so e reportada como
achado real depois que essas checagens descartam erro de unidade/amostragem
como causa alternativa.

Tambem persiste uma tabela de confundimento motorista x comportamento x
tipo de via (rodovia/via secundaria): esses tres fatores nao sao
totalmente cruzados neste dataset (nem todo motorista tem trajeto em toda
combinacao), entao nenhuma conclusao por motorista ou por tipo de via deve
ser tirada sem checar essa tabela.

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


def detectar_tipo_via(caminho_trajeto: Path) -> str:
    nome = caminho_trajeto.name.upper()
    if "MOTORWAY" in nome:
        return "Rodovia"
    if "SECONDARY" in nome:
        return "Via secundaria"
    return "Desconhecido"


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
        "tipo_via": detectar_tipo_via(pasta_trajeto),
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


def verificar_calibracao_sinal(raiz: Path) -> list:
    """Tres checagens empiricas do sinal, feitas ANTES de reportar qualquer
    resultado de nao-discriminacao (Secao B.5: nao publicar 'sem
    discriminacao' sem descartar erro de unidade/amostragem como causa
    alternativa).

    Uma checagem de 'gravidade ~1G no sinal bruto' NAO e usada aqui: o
    paper do dataset (ROMERA; BERGASA; ARROYO, 2016) documenta que Y/Z (e,
    por extensao, os canais X/Y/Z_KF) sao a aceleracao longitudinal/lateral
    do veiculo, ja no referencial do carro -- nao uma leitura bruta de
    telefone com a gravidade ainda misturada no eixo. Confirmado
    empiricamente: a magnitude de |acc_x,acc_y,acc_z| em janelas estaveis
    fica em ~0.05G, nao ~1G, o que inicialmente parecia um bug de unidade
    mas e consistente com o dataset ja entregar aceleracao dinamica
    orientada ao veiculo (ver docstring do modulo).

    1) Taxa de amostragem: confere que n_amostras/duracao bate com os
       ~10 Hz documentados (senao, eventos_por_min estaria calculado sobre
       uma duracao errada).
    2) Faixa fisica plausivel: a magnitude dinamica maxima observada nao
       deve exceder um teto fisicamente absurdo para um carro de passeio
       (>4G indicaria unidade trocada, nao conducao real).
    3) O filtro de Kalman de fato reduz ruido: desvio-padrao pos-filtro
       deve ser menor que o do sinal bruto no mesmo eixo -- senao as
       colunas *_kf podem estar trocadas/mal interpretadas.
    """
    taxas_hz, magnitudes_max_g, reducao_ruido = [], [], []
    for pasta in listar_trajetos(raiz):
        df_acc = carregar_acelerometro(pasta / "RAW_ACCELEROMETERS.txt")
        duracao_s = df_acc["timestamp"].max() - df_acc["timestamp"].min()
        if duracao_s > 0:
            taxas_hz.append(len(df_acc) / duracao_s)

        magnitude_g = np.sqrt(df_acc["acc_y_kf"] ** 2 + df_acc["acc_z_kf"] ** 2)
        magnitudes_max_g.append(magnitude_g.max())

        std_bruto = np.sqrt(df_acc["acc_y"].std() ** 2 + df_acc["acc_z"].std() ** 2)
        std_filtrado = np.sqrt(df_acc["acc_y_kf"].std() ** 2 + df_acc["acc_z_kf"].std() ** 2)
        reducao_ruido.append(std_filtrado < std_bruto)

    taxa_media_hz = float(np.mean(taxas_hz))
    magnitude_max_g_geral = float(np.max(magnitudes_max_g))
    pct_kf_reduz_ruido = float(np.mean(reducao_ruido)) * 100

    return [
        {
            "verificacao": "taxa de amostragem do acelerometro (~10 Hz documentado)",
            "valor_esperado": "~10 Hz",
            "valor_observado": f"{taxa_media_hz:.2f} Hz",
            "status": "ok" if 8 <= taxa_media_hz <= 12 else "revisar",
        },
        {
            "verificacao": "magnitude dinamica maxima observada (teto fisico plausivel p/ carro de passeio)",
            "valor_esperado": "< 4 G",
            "valor_observado": f"{magnitude_max_g_geral:.2f} G",
            "status": "ok" if magnitude_max_g_geral < 4 else "revisar",
        },
        {
            "verificacao": "filtro de Kalman reduz ruido em relacao ao sinal bruto (% de trajetos)",
            "valor_esperado": "100%",
            "valor_observado": f"{pct_kf_reduz_ruido:.0f}%",
            "status": "ok" if pct_kf_reduz_ruido >= 90 else "revisar",
        },
    ]


def coletar_magnitudes(raiz: Path) -> pd.DataFrame:
    """Magnitude de manobra amostra a amostra (~10 Hz), de todos os trajetos,
    rotulada por estilo de conducao e tipo de via. E a base do histograma
    didatico da aba 'Validacao de Hardware': ao contrario das metricas por
    trajeto (n=40), aqui ha ~311 mil amostras, o que sustenta um histograma
    de verdade em vez de um grafico de ruido."""
    partes = []
    for pasta in listar_trajetos(raiz):
        df_acc = carregar_acelerometro(pasta / "RAW_ACCELEROMETERS.txt")
        partes.append(pd.DataFrame({
            "comportamento": detectar_comportamento(pasta),
            "tipo_via": detectar_tipo_via(pasta),
            "magnitude_ms2": calcular_magnitude_ms2(df_acc),
        }))
    return pd.concat(partes, ignore_index=True)


def derivar_limiar_recalibrado(magnitudes: pd.DataFrame, percentil: float = 99.9) -> float:
    """Limiar alternativo derivado DOS DADOS, nao da literatura: o percentil
    99,9 da magnitude observada em trajetos NORMAIS. A regra e deliberadamente
    simples de explicar -- 'acima disto e atipico para conducao normal' -- em
    vez de um valor ajustado para maximizar alguma metrica, que seria dificil
    de justificar perante a banca e sujeito a sobreajuste com n=40 trajetos."""
    normais = magnitudes.loc[magnitudes["comportamento"] == "normal", "magnitude_ms2"]
    return round(float(np.percentile(normais, percentil)), 1)


def construir_histograma_magnitude(magnitudes: pd.DataFrame, n_classes: int = 14) -> pd.DataFrame:
    """Bina a magnitude em classes de largura igual, contando amostras por
    (estilo, tipo de via). As bordas das classes sao calculadas UMA vez para
    o conjunto todo e reaproveitadas em todos os grupos -- se cada painel
    tivesse suas proprias classes, a comparacao visual entre eles seria
    invalida. O binning e feito aqui (numpy) e nao no motor do grafico para
    que os limites de cada classe existam como dado, permitindo rotular as
    barras e posicionar a linha do limiar exatamente."""
    bordas = np.linspace(0.0, float(magnitudes["magnitude_ms2"].max()), n_classes + 1)
    linhas = []
    for (comportamento, via), grupo in magnitudes.groupby(["comportamento", "tipo_via"]):
        contagens, _ = np.histogram(grupo["magnitude_ms2"], bins=bordas)
        for i, n in enumerate(contagens):
            linhas.append({
                "comportamento": comportamento, "tipo_via": via, "classe_idx": i,
                "classe_min": bordas[i], "classe_max": bordas[i + 1], "n_amostras": int(n),
            })
    return pd.DataFrame(linhas)


def avaliar_limiares(magnitudes: pd.DataFrame, limiares: list) -> pd.DataFrame:
    """Percentual de amostras acima de cada limiar candidato, por estilo e
    tipo de via -- o numero que a figura exibe ao lado de cada linha de
    limiar."""
    linhas = []
    for (comportamento, via), grupo in magnitudes.groupby(["comportamento", "tipo_via"]):
        for limiar in limiares:
            n_acima = int((grupo["magnitude_ms2"] > limiar).sum())
            linhas.append({
                "comportamento": comportamento, "tipo_via": via, "limiar_ms2": limiar,
                "n_acima": n_acima, "n_total": len(grupo),
                "pct_acima": 100 * n_acima / len(grupo),
            })
    return pd.DataFrame(linhas)


def calcular_confundimento(df_resumo: pd.DataFrame) -> pd.DataFrame:
    """Contagem de trajetos por motorista x comportamento x tipo de via --
    documenta, com numeros, o quanto esses tres fatores estao cruzados
    (nem todo motorista tem trajeto em toda combinacao comportamento/via),
    para que nenhuma conclusao por motorista ou por via seja tirada sem
    essa checagem (Secao B.5)."""
    return (
        df_resumo.groupby(["motorista", "comportamento", "tipo_via"])
        .size()
        .reset_index(name="n_trajetos")
    )


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

    print("=== Verificacao de calibracao do sinal ===")
    checagens = verificar_calibracao_sinal(dataset_dir)
    for c in checagens:
        print(f"- {c['verificacao']}: observado={c['valor_observado']} (esperado {c['valor_esperado']}) "
              f"-- status: {c['status'].upper()}")
    pd.DataFrame(checagens).to_csv(saida_dir / "validacao_hardware_uah_calibracao.csv", index=False)

    df_resumo = processar_dataset(dataset_dir, args.limiar)

    print(f"\nLimiar de magnitude adotado: {args.limiar} m/s^2\n")
    print(f"{len(df_resumo)} trajetos processados de {dataset_dir}\n")
    print("Resumo por trajeto:")
    print(df_resumo.to_string(index=False))

    print("\nTaxa de eventos por minuto, agregada por comportamento conhecido:")
    print(df_resumo.groupby("comportamento")["eventos_por_min"].agg(["mean", "median", "std", "count"]))

    colunas_anova = ["eventos_por_min", "magnitude_maxima_ms2"]
    if df_resumo["velocidade_media_kmh"].notna().any():
        colunas_anova += ["velocidade_media_kmh", "var_curso_media_abs"]

    for coluna in colunas_anova:
        resultado_anova = validar_discriminacao(df_resumo, coluna)
        conhecidos = df_resumo[df_resumo["comportamento"] != "desconhecido"]
        n_por_grupo = conhecidos.groupby("comportamento")[coluna].count().to_dict()
        if resultado_anova is not None:
            print(
                f"\nANOVA one-way ({coluna} ~ comportamento): "
                f"F={resultado_anova.statistic:.4f}, p={resultado_anova.pvalue:.6f}, n={n_por_grupo}"
            )
        else:
            print(f"\nRotulos comportamentais insuficientes para ANOVA em {coluna} (minimo de 2 grupos conhecidos).")

    print("\n=== Confundimento motorista x comportamento x tipo de via ===")
    confundimento = calcular_confundimento(df_resumo)
    print(confundimento.to_string(index=False))
    confundimento.to_csv(saida_dir / "validacao_hardware_uah_confundimento.csv", index=False)

    # -----------------------------------------------------------------------
    # Distribuicao amostra a amostra (base do histograma didatico da aba
    # "Validacao de Hardware") + avaliacao do limiar da literatura contra um
    # limiar recalibrado a partir destes dados.
    # -----------------------------------------------------------------------
    print("\n=== Distribuicao da magnitude amostra a amostra ===")
    magnitudes = coletar_magnitudes(dataset_dir)
    limiar_recalibrado = derivar_limiar_recalibrado(magnitudes)
    print(f"{len(magnitudes):,} amostras a ~10 Hz")
    print(f"Limiar da literatura: {args.limiar} m/s^2 | "
          f"limiar recalibrado (p99.9 da conducao normal): {limiar_recalibrado} m/s^2")

    histograma = construir_histograma_magnitude(magnitudes)
    histograma.to_csv(saida_dir / "validacao_hardware_uah_histograma.csv", index=False)

    limiares = avaliar_limiares(magnitudes, [args.limiar, limiar_recalibrado])
    limiares.to_csv(saida_dir / "validacao_hardware_uah_limiares.csv", index=False)

    print("\n% de amostras acima de cada limiar (todas as vias):")
    for limiar in [args.limiar, limiar_recalibrado]:
        print(f"  limiar {limiar} m/s^2:")
        for comportamento, grupo in magnitudes.groupby("comportamento"):
            n_acima = int((grupo["magnitude_ms2"] > limiar).sum())
            print(f"    {comportamento:10s} {n_acima:6d} de {len(grupo):7,} = "
                  f"{100 * n_acima / len(grupo):.4f}%")
    print(f"\nHistograma salvo em: {saida_dir / 'validacao_hardware_uah_histograma.csv'}")
    print(f"Limiares salvos em:  {saida_dir / 'validacao_hardware_uah_limiares.csv'}")

    caminho_saida = saida_dir / "validacao_hardware_uah_driveset.csv"
    df_resumo.to_csv(caminho_saida, index=False)
    print(f"\nResumo por trajeto salvo em: {caminho_saida}")


if __name__ == "__main__":
    main()
