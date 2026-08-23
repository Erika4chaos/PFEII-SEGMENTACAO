"""
Etapa 3 (Secao 3.5.3): estatisticas descritivas por cluster no espaco
original das 19 variaveis, teste ANOVA one-way para as variaveis continuas e
teste qui-quadrado de independencia para as variaveis binarias/categoricas.

Uso:
    python src/perfilamento.py --in data/processed/matriz_original_clusters.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, f_oneway

from preprocessamento import COLUNAS_19, COLUNAS_BINARIAS, COLUNAS_CONTINUAS

# valor_pago_historico, valor_sinistro_historico e qt_cias_anteriores sao
# tratadas como continuas (ANOVA), conforme explicitado na Secao 3.5.3;
# tipo_sinistro_predominante e categorica nominal (tombamento/incendio/
# terceiros/nenhum), por isso entra no qui-quadrado junto das binarias.
COLUNAS_CONTINUAS_TESTE = COLUNAS_CONTINUAS + [
    "valor_pago_historico", "valor_sinistro_historico", "qt_cias_anteriores",
]
COLUNAS_CATEGORICAS_TESTE = COLUNAS_BINARIAS + ["tipo_sinistro_predominante"]


def carregar_dados(caminho: Path) -> pd.DataFrame:
    return pd.read_csv(caminho)


def estatisticas_descritivas(df: pd.DataFrame) -> pd.DataFrame:
    """Media, mediana e desvio padrao de cada uma das 19 variaveis, por
    cluster, no espaco original (unidades reais)."""
    agregados = df.groupby("cluster")[COLUNAS_19].agg(["mean", "median", "std"])
    agregados = agregados.stack(level=0, future_stack=True).reset_index()
    agregados = agregados.rename(columns={
        "level_1": "variavel", "mean": "media", "median": "mediana", "std": "desvio_padrao",
    })
    return agregados[["cluster", "variavel", "media", "mediana", "desvio_padrao"]]


def teste_anova(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for var in COLUNAS_CONTINUAS_TESTE:
        grupos = [g[var].dropna().to_numpy() for _, g in df.groupby("cluster")]
        estatistica, p_valor = f_oneway(*grupos)
        linhas.append({"variavel": var, "teste": "ANOVA", "estatistica": estatistica, "p_valor": p_valor})
    return pd.DataFrame(linhas)


def teste_qui_quadrado(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for var in COLUNAS_CATEGORICAS_TESTE:
        tabela = pd.crosstab(df["cluster"], df[var])
        estatistica, p_valor, _, _ = chi2_contingency(tabela)
        linhas.append({"variavel": var, "teste": "qui-quadrado", "estatistica": estatistica, "p_valor": p_valor})
    return pd.DataFrame(linhas)


def relatorio_significancia(df: pd.DataFrame) -> pd.DataFrame:
    resultado = pd.concat([teste_anova(df), teste_qui_quadrado(df)], ignore_index=True)
    resultado["significativo_5pct"] = resultado["p_valor"] < 0.05
    return resultado.sort_values("p_valor").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Perfilamento e testes de significancia por cluster (Etapa 3 da metodologia)."
    )
    parser.add_argument("--in", dest="entrada", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    raiz = Path(__file__).resolve().parent.parent
    entrada = Path(args.entrada) if args.entrada else raiz / "data" / "processed" / "matriz_original_clusters.csv"
    saida_dir = Path(args.out_dir) if args.out_dir else raiz / "data" / "processed"
    saida_dir.mkdir(parents=True, exist_ok=True)

    df = carregar_dados(entrada)

    descritivas = estatisticas_descritivas(df)
    significancia = relatorio_significancia(df)

    caminho_descritivas = saida_dir / "perfis_estatisticas_descritivas.csv"
    caminho_significancia = saida_dir / "perfis_testes_significancia.csv"
    descritivas.to_csv(caminho_descritivas, index=False)
    significancia.to_csv(caminho_significancia, index=False)

    tamanhos = df["cluster"].value_counts().sort_index()
    print("Tamanho dos clusters:")
    for cluster, n in tamanhos.items():
        print(f"  cluster {cluster}: {n} registros")

    print("\nVariaveis mais discriminantes (menor p-valor primeiro):")
    print(significancia.to_string(index=False))

    if "_perfil_gerador" in df.columns:
        print("\nValidacao cruzada com o perfil de origem (somente base sintetica):")
        print(pd.crosstab(df["cluster"], df["_perfil_gerador"]))

    print(f"\nEstatisticas descritivas por cluster: {caminho_descritivas}")
    print(f"Testes de significancia: {caminho_significancia}")


if __name__ == "__main__":
    main()
