"""
Etapa 3 (Secao 3.5.3): caracterizacao dos clusters no espaco original das 19
variaveis do Quadro 2.

LEITURA OBRIGATORIA ANTES DE CITAR ESTES NUMEROS NO TEXTO
---------------------------------------------------------
Os clusters foram DEFINIDOS pelas mesmas variaveis que esta etapa compara
entre eles. O K-Means maximiza a separacao nessas variaveis; testar depois se
elas separam os grupos e circular (o chamado *double dipping*). Um p-valor
obtido assim nao e inferencial: ele nao responde "existe diferenca na
populacao?", apenas reflete que o algoritmo fez o que foi programado para
fazer. Na pratica, p-valores da ordem de 1e-100 aparecem naturalmente e nao
significam achado forte.

Por isso esta etapa reporta TAMANHO DE EFEITO, nao significancia:

  eta quadrado (eta2)  -- variaveis continuas: fracao da variancia da variavel
                          explicada pela particao. Referencia de Cohen: 0,01
                          pequeno, 0,06 medio, 0,14 grande.
  V de Cramer          -- variaveis categoricas: forca da associacao entre a
                          variavel e a particao, de 0 a 1.

A leitura correta e DESCRITIVA: "as variaveis X e Y sao as que mais
diferenciam os perfis encontrados", e nao "a diferenca e estatisticamente
significativa". O p-valor continua sendo calculado e gravado por
transparencia, mas nao deve ser usado como evidencia no texto.

Uso:
    python src/perfilamento.py --in data/processed/matriz_original_clusters.csv
"""

import argparse
from pathlib import Path

import numpy as np
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


def _eta_quadrado(grupos: list) -> float:
    """Fracao da variancia total explicada pela particao (SQ_entre / SQ_total).

    E a medida de tamanho de efeito que acompanha a ANOVA de um fator. Ao
    contrario do p-valor, nao cresce indefinidamente com o n: um eta2 de 0,40
    significa que a particao explica 40% da variacao da variavel, seja a
    amostra de 600 ou de 600 mil registros.
    """
    todos = np.concatenate(grupos)
    media_geral = todos.mean()
    sq_total = ((todos - media_geral) ** 2).sum()
    if sq_total == 0:
        return 0.0
    sq_entre = sum(len(g) * (g.mean() - media_geral) ** 2 for g in grupos)
    return float(sq_entre / sq_total)


def teste_anova(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for var in COLUNAS_CONTINUAS_TESTE:
        grupos = [g[var].dropna().to_numpy() for _, g in df.groupby("cluster")]
        grupos = [g for g in grupos if len(g) > 0]
        estatistica, p_valor = f_oneway(*grupos)
        linhas.append({
            "variavel": var,
            "medida": "eta2",
            "tamanho_efeito": _eta_quadrado(grupos),
            "estatistica": estatistica,
            "p_valor": p_valor,
        })
    return pd.DataFrame(linhas)


def _v_de_cramer(qui2: float, tabela: pd.DataFrame) -> float:
    """Forca da associacao entre a variavel categorica e a particao, de 0 a 1.

    Normaliza o qui-quadrado pelo tamanho da amostra e pela dimensao da
    tabela, o que torna o valor comparavel entre variaveis com numeros
    diferentes de categorias.
    """
    n = tabela.to_numpy().sum()
    menor_dimensao = min(tabela.shape) - 1
    if n == 0 or menor_dimensao <= 0:
        return 0.0
    return float(np.sqrt(qui2 / (n * menor_dimensao)))


def teste_qui_quadrado(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for var in COLUNAS_CATEGORICAS_TESTE:
        tabela = pd.crosstab(df["cluster"], df[var])
        estatistica, p_valor, _, _ = chi2_contingency(tabela)
        linhas.append({
            "variavel": var,
            "medida": "V de Cramer",
            "tamanho_efeito": _v_de_cramer(estatistica, tabela),
            "estatistica": estatistica,
            "p_valor": p_valor,
        })
    return pd.DataFrame(linhas)


def _classificar_efeito(linha) -> str:
    """Rotulo qualitativo do tamanho de efeito, pelas referencias usuais."""
    valor = linha["tamanho_efeito"]
    if linha["medida"] == "eta2":
        limites = [(0.14, "grande"), (0.06, "medio"), (0.01, "pequeno")]
    else:  # V de Cramer
        limites = [(0.50, "grande"), (0.30, "medio"), (0.10, "pequeno")]
    for corte, rotulo in limites:
        if valor >= corte:
            return rotulo
    return "desprezivel"


def relatorio_discriminancia(df: pd.DataFrame) -> pd.DataFrame:
    """Ranking DESCRITIVO das variaveis que mais diferenciam os clusters.

    Ordenado por tamanho de efeito, nao por p-valor -- ver a nota sobre
    circularidade no topo deste modulo.
    """
    resultado = pd.concat([teste_anova(df), teste_qui_quadrado(df)], ignore_index=True)
    resultado["efeito"] = resultado.apply(_classificar_efeito, axis=1)
    return resultado.sort_values("tamanho_efeito", ascending=False).reset_index(drop=True)


# Nome antigo mantido para nao quebrar importacoes existentes.
relatorio_significancia = relatorio_discriminancia


def main():
    parser = argparse.ArgumentParser(
        description="Perfilamento e discriminancia por cluster (Etapa 3 da metodologia)."
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
    discriminancia = relatorio_discriminancia(df)

    caminho_descritivas = saida_dir / "perfis_estatisticas_descritivas.csv"
    caminho_significancia = saida_dir / "perfis_testes_significancia.csv"
    descritivas.to_csv(caminho_descritivas, index=False)
    discriminancia.to_csv(caminho_significancia, index=False)

    tamanhos = df["cluster"].value_counts().sort_index()
    print("Tamanho dos clusters:")
    for cluster, n in tamanhos.items():
        print(f"  cluster {cluster}: {n} registros")

    print("\nVariaveis que mais diferenciam os clusters (maior tamanho de efeito):")
    print("NOTA: leitura descritiva. Os clusters foram definidos por estas mesmas")
    print("variaveis, entao p-valores aqui nao sao inferenciais -- ver docstring.")
    print(discriminancia.to_string(index=False))

    if "_perfil_gerador" in df.columns:
        print("\nTeste de sanidade do pipeline (somente base sintetica):")
        print("Cruzamento entre cluster encontrado e perfil usado na geracao.")
        print("NAO e validacao do metodo: mede apenas se o pipeline reencontra")
        print("uma estrutura que sabidamente existe na base.")
        cruz = pd.crosstab(df["cluster"], df["_perfil_gerador"])
        print(cruz)
        acerto = cruz.max(axis=1).sum() / len(df)
        print(f"Concordancia pelo pareamento majoritario: {acerto:.1%}")

    print(f"\nEstatisticas descritivas por cluster: {caminho_descritivas}")
    print(f"Ranking de discriminancia: {caminho_significancia}")


if __name__ == "__main__":
    main()
