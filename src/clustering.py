"""
Etapa 2 (Secao 3.5.2): aplicacao e calibracao do K-Means sobre a matriz
normalizada das 19 variaveis (Quadro 2), com inicializacao K-Means++,
n_init=10, validacao do numero de clusters pelo Metodo do Cotovelo (k=2..8)
e pelo indice de Silhouette, e ajuste do modelo final para k=3 (hipotese dos
tres perfis-alvo definidos na Secao 3.4).

Uso:
    python src/clustering.py --in data/processed/matriz_normalizada.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from preprocessamento import COLUNAS_19

RANDOM_STATE = 42


def carregar_matriz(caminho: Path) -> pd.DataFrame:
    return pd.read_csv(caminho)


def curva_cotovelo(X: np.ndarray, k_min: int = 2, k_max: int = 8) -> dict:
    """Inercia (WCSS) do K-Means para cada k no intervalo [k_min, k_max]."""
    inercias = {}
    for k in range(k_min, k_max + 1):
        modelo = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
        modelo.fit(X)
        inercias[k] = modelo.inertia_
    return inercias


def curva_silhouette(X: np.ndarray, k_min: int = 2, k_max: int = 8) -> dict:
    """Indice de Silhouette medio do K-Means para cada k no intervalo."""
    scores = {}
    for k in range(k_min, k_max + 1):
        modelo = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
        rotulos = modelo.fit_predict(X)
        scores[k] = silhouette_score(X, rotulos)
    return scores


def k_otimo_cotovelo(inercias: dict) -> int:
    """Ponto de inflexao pela distancia perpendicular maxima a reta que liga
    o primeiro e o ultimo ponto da curva de inercia (metodo do cotovelo)."""
    ks = np.array(sorted(inercias))
    valores = np.array([inercias[k] for k in ks], dtype=float)

    p1 = np.array([ks[0], valores[0]], dtype=float)
    p2 = np.array([ks[-1], valores[-1]], dtype=float)
    direcao = p2 - p1
    direcao_normalizada = direcao / np.linalg.norm(direcao)

    distancias = []
    for k, v in zip(ks, valores):
        ponto = np.array([k, v], dtype=float)
        projecao = p1 + np.dot(ponto - p1, direcao_normalizada) * direcao_normalizada
        distancias.append(np.linalg.norm(ponto - projecao))

    return int(ks[int(np.argmax(distancias))])


def k_otimo_silhouette(scores: dict) -> int:
    return max(scores, key=scores.get)


def treinar_kmeans(X: np.ndarray, k: int, n_init: int = 10, random_state: int = RANDOM_STATE) -> KMeans:
    modelo = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=random_state)
    modelo.fit(X)
    return modelo


def main():
    parser = argparse.ArgumentParser(
        description="Aplicacao e calibracao do K-Means (Etapa 2 da metodologia)."
    )
    parser.add_argument("--in", dest="entrada", type=str, default=None)
    parser.add_argument("--in-original", dest="entrada_original", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--k", type=int, default=3, help="numero de clusters do modelo final")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    args = parser.parse_args()

    raiz = Path(__file__).resolve().parent.parent
    entrada = Path(args.entrada) if args.entrada else raiz / "data" / "processed" / "matriz_normalizada.csv"
    entrada_original = (
        Path(args.entrada_original) if args.entrada_original
        else raiz / "data" / "processed" / "matriz_original.csv"
    )
    saida_dir = Path(args.out_dir) if args.out_dir else raiz / "data" / "processed"
    saida_dir.mkdir(parents=True, exist_ok=True)

    df_norm = carregar_matriz(entrada)
    X = df_norm[COLUNAS_19].to_numpy()

    inercias = curva_cotovelo(X, args.k_min, args.k_max)
    silhouettes = curva_silhouette(X, args.k_min, args.k_max)

    k_cotovelo = k_otimo_cotovelo(inercias)
    k_silhouette = k_otimo_silhouette(silhouettes)

    print("Metodo do Cotovelo (inercia por k):")
    for k, v in inercias.items():
        marcador = " <- cotovelo" if k == k_cotovelo else ""
        print(f"  k={k}: {v:.4f}{marcador}")

    print("\nIndice de Silhouette por k:")
    for k, v in silhouettes.items():
        marcador = " <- maximo" if k == k_silhouette else ""
        print(f"  k={k}: {v:.4f}{marcador}")

    print(f"\nk sugerido pelo cotovelo: {k_cotovelo}")
    print(f"k sugerido pelo Silhouette: {k_silhouette}")
    print(f"k adotado para o modelo final: {args.k} (hipotese dos 3 perfis-alvo, Secao 3.4)")

    validacao_df = pd.DataFrame({
        "k": list(inercias.keys()),
        "inercia": list(inercias.values()),
        "silhouette": [silhouettes[k] for k in inercias],
    })
    caminho_validacao = saida_dir / "validacao_k.csv"
    validacao_df.to_csv(caminho_validacao, index=False)

    modelo_final = treinar_kmeans(X, args.k)
    rotulos = modelo_final.labels_
    silhouette_final = silhouette_score(X, rotulos)
    print(f"\nSilhouette do modelo final (k={args.k}): {silhouette_final:.4f}")

    df_norm_saida = df_norm.copy()
    df_norm_saida["cluster"] = rotulos
    caminho_norm_saida = saida_dir / "matriz_normalizada_clusters.csv"
    df_norm_saida.to_csv(caminho_norm_saida, index=False)

    if entrada_original.exists():
        df_orig = pd.read_csv(entrada_original)
        df_orig_saida = df_orig.merge(
            df_norm_saida[["numeroApolice", "cluster"]], on="numeroApolice", how="left"
        )
        caminho_orig_saida = saida_dir / "matriz_original_clusters.csv"
        df_orig_saida.to_csv(caminho_orig_saida, index=False)
        print(f"Matriz original com rotulos: {caminho_orig_saida}")

    print(f"Curva de validacao (cotovelo + silhouette): {caminho_validacao}")
    print(f"Matriz normalizada com rotulos: {caminho_norm_saida}")


if __name__ == "__main__":
    main()
