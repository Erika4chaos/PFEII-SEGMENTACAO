"""
Estabilidade do resultado da segmentacao: as verificacoes da Secao 4
("Analise de Sensibilidade e Estabilidade") que nao tinham programa proprio.

POR QUE ESTE SCRIPT EXISTE
--------------------------
O texto do TCC reporta quatro numeros de robustez que haviam sido calculados
a parte e nao estavam em nenhum arquivo do repositorio -- quem quisesse
conferi-los nao tinha o que rodar:

  1. Inicializacao do K-Means. Refazendo o ajuste com outras sementes do
     algoritmo, a particao muda? Mede-se pelo indice de Rand ajustado (ARI)
     contra a solucao de referencia (semente 42). ARI = 1 e particao identica.

  2. Calibracao do gerador. As marginais que a base sintetica precisa
     reproduzir (qst1, qst2, ambos e sinistro medio da amostra real, Secao 3.3)
     continuam dentro da faixa gerada quando a semente muda?

  3. Peso de cada bloco de variaveis. Que fracao da variancia total da matriz
     do modelo vem das binarias, das indicadoras de tipo de sinistro e das
     continuas -- ou seja, de onde vem a distancia euclidiana que define os
     grupos.

  4. Escolha de escala (Quadro 11). Trocando o Min-Max por padronizacao z,
     reduzindo o peso das binarias ou tirando as indicadoras, a estrutura de
     tres perfis ainda e recuperada?

A quinta verificacao da mesma secao -- a faixa entre realizacoes da base --
ja tinha programa: src/sensibilidade_sementes.py.

Nada aqui altera o modelo: o K-Means adotado continua sendo o de
src/clustering.py. Este script so mede o quanto o resultado depende de
escolhas que poderiam ter sido outras.

LEITURA DO QUADRO DE ESCALA. A Silhueta NAO e comparavel entre as linhas:
cada configuracao define um espaco metrico proprio. A coluna comparavel e a
concordancia com os perfis de origem -- que, por sua vez, e teste de sanidade
do pipeline sobre base sintetica, nao validacao do metodo em dado real.

Uso:
    python src/estabilidade.py
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clustering import RANDOM_STATE, treinar_kmeans  # noqa: E402
from gerar_base_sintetica import (  # noqa: E402
    MARGINAIS_OBSERVADAS, gerar_base, marginais_geradas,
)
from preprocessamento import (  # noqa: E402
    COLUNAS_BINARIAS, COLUNAS_MODELO, INDICADORAS_SINISTRO,
)
from sensibilidade_sementes import PESOS_PADRAO, concordancia_majoritaria  # noqa: E402

SEMENTES_INICIALIZACAO = [43, 44, 45, 46, 47, 48]
SEMENTES_CALIBRACAO = list(range(42, 52))


def estabilidade_inicializacao(X: np.ndarray, k: int, sementes: list) -> pd.DataFrame:
    """ARI entre a particao de referencia (RANDOM_STATE) e a obtida com cada
    semente de inicializacao, mantendo K-Means++ e n_init=10."""
    referencia = treinar_kmeans(X, k, random_state=RANDOM_STATE).labels_
    linhas = []
    for semente in sementes:
        rotulos = treinar_kmeans(X, k, random_state=semente).labels_
        linhas.append({
            "semente_inicializacao": semente,
            "ari_contra_referencia": float(adjusted_rand_score(referencia, rotulos)),
            "registros_em_grupo_diferente": int(_trocas(referencia, rotulos)),
        })
    return pd.DataFrame(linhas)


def _trocas(referencia: np.ndarray, rotulos: np.ndarray) -> int:
    """Quantos registros mudam de grupo, depois de parear cada grupo novo ao
    grupo de referencia com que mais se sobrepoe (os numeros dos clusters sao
    arbitrarios e podem vir permutados)."""
    tabela = pd.crosstab(pd.Series(rotulos), pd.Series(referencia))
    return int(tabela.to_numpy().sum() - tabela.max(axis=1).sum())


def estabilidade_calibracao(sementes: list, n: int, taxa_ausencia: float,
                            data_base: date) -> pd.DataFrame:
    """Marginais da base gerada para cada semente do gerador -- as mesmas que
    relatorio_calibracao() imprime, via marginais_geradas()."""
    linhas = []
    for semente in sementes:
        base = gerar_base(n, semente, PESOS_PADRAO, data_base, taxa_ausencia)
        geradas = marginais_geradas(base)
        valores = geradas["valores_sinistro"]
        linhas.append({
            "semente_base": semente,
            "qst1_entre_declarantes": geradas["qst1"],
            "qst2_entre_declarantes": geradas["qst2"],
            "ambos_entre_declarantes": geradas["ambos"],
            "sinistro_medio": float(np.mean(valores)) if valores else np.nan,
        })
    return pd.DataFrame(linhas)


def resumir_calibracao(por_semente: pd.DataFrame) -> pd.DataFrame:
    """Faixa e mediana de cada marginal, confrontadas com o valor observado."""
    observado = {
        "qst1_entre_declarantes": MARGINAIS_OBSERVADAS["qst1_entre_declarantes"],
        "qst2_entre_declarantes": MARGINAIS_OBSERVADAS["qst2_entre_declarantes"],
        "ambos_entre_declarantes": MARGINAIS_OBSERVADAS["ambos_entre_declarantes"],
        "sinistro_medio": MARGINAIS_OBSERVADAS["sinistro_medio"],
    }
    linhas = []
    for coluna, valor in observado.items():
        serie = por_semente[coluna]
        mediana = float(serie.median())
        linhas.append({
            "estatistica": coluna,
            "observado": valor,
            "gerado_minimo": float(serie.min()),
            "gerado_mediana": mediana,
            "gerado_maximo": float(serie.max()),
            "observado_dentro_da_faixa": bool(serie.min() <= valor <= serie.max()),
            # Proporcoes: diferenca em pontos percentuais. Valor monetario:
            # diferenca relativa, em %.
            "desvio_da_mediana": (100 * (mediana - valor) if valor <= 1
                                  else 100 * (mediana - valor) / valor),
            "unidade_do_desvio": "p.p." if valor <= 1 else "%",
        })
    return pd.DataFrame(linhas)


def variancia_por_bloco(normalizada: pd.DataFrame) -> pd.DataFrame:
    """Fracao da variancia total da matriz do modelo em cada bloco de
    variaveis. Como a distancia euclidiana ao quadrado soma as diferencas por
    coluna, essa fracao e tambem a participacao esperada de cada bloco na
    distancia entre dois registros quaisquer."""
    variancias = normalizada[COLUNAS_MODELO].var(axis=0, ddof=0)
    blocos = {
        "binarias": COLUNAS_BINARIAS,
        "indicadoras_tipo_sinistro": INDICADORAS_SINISTRO,
        "continuas": [c for c in COLUNAS_MODELO
                      if c not in COLUNAS_BINARIAS and c not in INDICADORAS_SINISTRO],
    }
    total = float(variancias.sum())
    return pd.DataFrame([
        {"bloco": nome, "n_variaveis": len(colunas),
         "fracao_da_variancia": float(variancias[colunas].sum()) / total}
        for nome, colunas in blocos.items()
    ])


def sensibilidade_escala(normalizada: pd.DataFrame, k: int) -> pd.DataFrame:
    """Silhueta e concordancia do K-Means sob quatro configuracoes de escala
    (Quadro 11). Todas partem da matriz que alimenta o modelo adotado, ja com
    log1p nas variaveis de cauda longa e a nominal expandida em indicadoras."""
    X = normalizada[COLUNAS_MODELO].to_numpy(dtype=float)
    perfis = normalizada["_perfil_gerador"]
    colunas = np.array(COLUNAS_MODELO)
    eh_binaria = np.isin(colunas, COLUNAS_BINARIAS)
    eh_indicadora = np.isin(colunas, INDICADORAS_SINISTRO)

    meio_peso = X.copy()
    meio_peso[:, eh_binaria] *= 0.5

    configuracoes = [
        ("Min-Max sobre as variaveis do modelo (adotada)", X),
        # Padronizar a matriz Min-Max ou a matriz antes dele da o mesmo
        # resultado: o escore z e invariante a transformacoes afins por coluna.
        ("Padronizacao por escore z", StandardScaler().fit_transform(X)),
        ("Min-Max com metade do peso nas binarias", meio_peso),
        ("Min-Max sem as indicadoras de tipo de sinistro", X[:, ~eh_indicadora]),
    ]
    linhas = []
    for nome, matriz in configuracoes:
        rotulos = treinar_kmeans(matriz, k).labels_
        linhas.append({
            "configuracao": nome,
            "n_colunas": matriz.shape[1],
            "silhouette": float(silhouette_score(matriz, rotulos)),
            "concordancia": concordancia_majoritaria(pd.Series(rotulos), perfis),
        })
    return pd.DataFrame(linhas)


def main():
    parser = argparse.ArgumentParser(
        description="Verificacoes de estabilidade da segmentacao (Secao 4)."
    )
    parser.add_argument("--in", dest="entrada", type=str, default=None,
                        help="matriz normalizada (default: data/processed/matriz_normalizada.csv)")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--taxa-ausencia", type=float, default=0.04)
    parser.add_argument("--data-base", type=str, default="2026-09-01")
    args = parser.parse_args()

    raiz = Path(__file__).resolve().parent.parent
    entrada = Path(args.entrada) if args.entrada else raiz / "data" / "processed" / "matriz_normalizada.csv"
    saida_dir = Path(args.out_dir) if args.out_dir else raiz / "data" / "processed"
    saida_dir.mkdir(parents=True, exist_ok=True)

    normalizada = pd.read_csv(entrada)
    X = normalizada[COLUNAS_MODELO].to_numpy(dtype=float)

    print("1. Inicializacao do K-Means (ARI contra a semente de referencia "
          f"{RANDOM_STATE})")
    inicializacao = estabilidade_inicializacao(X, args.k, SEMENTES_INICIALIZACAO)
    for linha in inicializacao.itertuples():
        print(f"   semente {linha.semente_inicializacao}: ARI "
              f"{linha.ari_contra_referencia:.4f} "
              f"({linha.registros_em_grupo_diferente} registro(s) em grupo diferente)")
    inicializacao.to_csv(saida_dir / "estabilidade_inicializacao.csv", index=False)

    print(f"\n2. Calibracao do gerador em {len(SEMENTES_CALIBRACAO)} sementes "
          f"({SEMENTES_CALIBRACAO[0]} a {SEMENTES_CALIBRACAO[-1]})")
    por_semente = estabilidade_calibracao(
        SEMENTES_CALIBRACAO, args.n, args.taxa_ausencia,
        date.fromisoformat(args.data_base))
    resumo = resumir_calibracao(por_semente)
    for linha in resumo.itertuples():
        if linha.observado <= 1:
            faixa = (f"{linha.gerado_minimo:.1%} a {linha.gerado_maximo:.1%}, "
                     f"mediana {linha.gerado_mediana:.1%}; observado {linha.observado:.1%}")
        else:
            faixa = (f"R$ {linha.gerado_minimo:,.0f} a R$ {linha.gerado_maximo:,.0f}, "
                     f"mediana R$ {linha.gerado_mediana:,.0f}; observado R$ {linha.observado:,.0f}")
        dentro = "dentro" if linha.observado_dentro_da_faixa else "FORA"
        print(f"   {linha.estatistica:<26} {faixa} -> {dentro} da faixa, mediana a "
              f"{linha.desvio_da_mediana:+.2f} {linha.unidade_do_desvio}")
    por_semente.to_csv(saida_dir / "estabilidade_calibracao_por_semente.csv", index=False)
    resumo.to_csv(saida_dir / "estabilidade_calibracao.csv", index=False)

    print("\n3. Fracao da variancia da matriz do modelo por bloco de variaveis")
    blocos = variancia_por_bloco(normalizada)
    for linha in blocos.itertuples():
        print(f"   {linha.bloco:<28} {linha.n_variaveis:>2} variaveis  "
              f"{linha.fracao_da_variancia:.1%}")
    blocos.to_csv(saida_dir / "estabilidade_variancia_blocos.csv", index=False)

    print("\n4. Sensibilidade da particao a escolha de escala (Quadro 11)")
    print("   A Silhueta nao e comparavel entre linhas; a concordancia e.")
    escala = sensibilidade_escala(normalizada, args.k)
    for linha in escala.itertuples():
        print(f"   {linha.configuracao:<50} Silhueta {linha.silhouette:.4f}  "
              f"concordancia {linha.concordancia:.1%}")
    escala.to_csv(saida_dir / "sensibilidade_escala.csv", index=False)

    print(f"\nArquivos gravados em {saida_dir}")


if __name__ == "__main__":
    main()
