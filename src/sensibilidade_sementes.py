"""
Analise de sensibilidade da segmentacao a semente da base sintetica.

POR QUE ESTE SCRIPT EXISTE
--------------------------
Os resultados da Etapa 2 (Silhouette do modelo final e concordancia com os
perfis usados na geracao) sao medidos sobre UMA realizacao da base sintetica,
gerada com uma semente fixa. Um numero isolado nessa condicao e ambiguo: nao
se distingue estrutura do dado de sorte do sorteio.

Regerando a base com sementes diferentes e repetindo o pipeline inteiro --
geracao, pre-processamento, K-Means, pareamento majoritario --, obtem-se uma
FAIXA. Uma faixa estreita indica que a estrutura recuperada nao depende da
realizacao particular; uma faixa larga indica o contrario, e nesse caso o
numero unico do texto nao deveria ser reportado sem ressalva.

E a faixa, nao o valor de uma semente, que deve ir para o texto do TCC.

O que este script NAO e: validacao externa do metodo. A base continua
sintetica e os rotulos continuam vindo do proprio gerador. Isto mede
estabilidade, nao acerto.

Uso:
    python src/sensibilidade_sementes.py --sementes 42 43 44 45 46
"""

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clustering import treinar_kmeans  # noqa: E402
from gerar_base_sintetica import gerar_base  # noqa: E402
from preprocessamento import COLUNAS_MODELO, processar  # noqa: E402

PESOS_PADRAO = {"alto_risco": 0.30, "alta_cobertura": 0.35, "referral": 0.35}


def concordancia_majoritaria(rotulos: pd.Series, perfis: pd.Series) -> float:
    """Fracao de registros recuperados quando cada cluster e pareado ao perfil
    gerador majoritario dentro dele. E o mesmo criterio do perfilamento."""
    tabela = pd.crosstab(rotulos, perfis)
    return float(tabela.max(axis=1).sum() / tabela.to_numpy().sum())


def rodar_uma_semente(semente: int, n: int, k: int, taxa_ausencia: float,
                      data_base) -> dict:
    base = gerar_base(n, semente, PESOS_PADRAO, data_base, taxa_ausencia)

    with tempfile.TemporaryDirectory() as tmp:
        caminho = Path(tmp) / "base.json"
        caminho.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")
        original, normalizada = processar(caminho)

    X = normalizada[COLUNAS_MODELO].to_numpy()
    modelo = treinar_kmeans(X, k)
    rotulos = pd.Series(modelo.labels_, index=normalizada.index)

    return {
        "semente": semente,
        "silhouette": float(silhouette_score(X, modelo.labels_)),
        "concordancia": concordancia_majoritaria(rotulos, original["_perfil_gerador"]),
    }


def resumir(resultados: list) -> str:
    linhas = ["", f"Sensibilidade a semente da base sintetica (k fixo, n fixo)",
              "-" * 56,
              f"{'Semente':>9}{'Silhouette':>14}{'Concordancia':>15}"]
    for r in resultados:
        linhas.append(f"{r['semente']:>9}{r['silhouette']:>14.4f}"
                      f"{r['concordancia']:>14.1%}")

    sils = [r["silhouette"] for r in resultados]
    concs = [r["concordancia"] for r in resultados]
    linhas += [
        "-" * 56,
        f"{'mediana':>9}{statistics.median(sils):>14.4f}"
        f"{statistics.median(concs):>14.1%}",
        f"{'minimo':>9}{min(sils):>14.4f}{min(concs):>14.1%}",
        f"{'maximo':>9}{max(sils):>14.4f}{max(concs):>14.1%}",
        "",
        "Para o texto do TCC, reportar a MEDIANA e a FAIXA acima, nunca o",
        "valor de uma semente isolada.",
        "",
    ]
    return "\n".join(linhas)


def main():
    parser = argparse.ArgumentParser(
        description="Mede a estabilidade da segmentacao entre realizacoes da base."
    )
    parser.add_argument("--sementes", type=int, nargs="+",
                        default=[42, 43, 44, 45, 46])
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--taxa-ausencia", type=float, default=0.04)
    parser.add_argument("--data-base", type=str, default="2026-09-01")
    parser.add_argument("--out", type=str, default=None,
                        help="CSV de saida (default: data/processed/sensibilidade_sementes.csv)")
    args = parser.parse_args()

    from datetime import date
    data_base = date.fromisoformat(args.data_base)

    resultados = []
    for semente in args.sementes:
        resultado = rodar_uma_semente(semente, args.n, args.k,
                                      args.taxa_ausencia, data_base)
        print(f"  semente {semente}: Silhouette {resultado['silhouette']:.4f}, "
              f"concordancia {resultado['concordancia']:.1%}")
        resultados.append(resultado)

    print(resumir(resultados))

    destino = (Path(args.out) if args.out else
               Path(__file__).resolve().parent.parent / "data" / "processed"
               / "sensibilidade_sementes.csv")
    destino.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(resultados).to_csv(destino, index=False)
    print(f"Resultados por semente: {destino}")


if __name__ == "__main__":
    main()
