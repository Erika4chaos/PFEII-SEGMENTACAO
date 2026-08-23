"""
Etapa 4 (Secao 3.5.4): dashboard Streamlit com projecao PCA 2D dos clusters,
tabela de perfis, KPIs por segmento e indices de validacao tecnica
(Silhouette, Davies-Bouldin, Calinski-Harabasz).

Uso:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

RAIZ = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ / "src"))
from preprocessamento import COLUNAS_19  # noqa: E402

DADOS_DIR = RAIZ / "data" / "processed"

TIPO_SINISTRO_LABEL = {0: "nenhum identificado", 1: "tombamento", 2: "incendio", 3: "terceiros"}


def _fmt_brl(valor: float) -> str:
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_data
def carregar_dados():
    normalizada = pd.read_csv(DADOS_DIR / "matriz_normalizada_clusters.csv")
    original = pd.read_csv(DADOS_DIR / "matriz_original_clusters.csv")
    return normalizada, original


@st.cache_data
def carregar_validacao_k():
    caminho = DADOS_DIR / "validacao_k.csv"
    return pd.read_csv(caminho) if caminho.exists() else None


def projetar_pca(normalizada: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    pca = PCA(n_components=2, random_state=42)
    componentes = pca.fit_transform(normalizada[COLUNAS_19])
    projecao = normalizada[["numeroApolice", "cluster"]].copy()
    projecao["PCA1"] = componentes[:, 0]
    projecao["PCA2"] = componentes[:, 1]
    return projecao, float(pca.explained_variance_ratio_.sum())


def nomear_cluster(medias: pd.Series) -> str:
    """Heuristica simples que confronta as medias do cluster no espaco
    original com os tres perfis-alvo da Secao 3.4, para apresentacao em
    linguagem de negocio (sem expor os detalhes matematicos do modelo)."""
    if medias["referral_pendente"] > 0.5:
        return "Perfil 3 - Cotacao em Referral ou Conversao Tardia"
    if medias["qt_coberturas_ativas"] >= 5 and medias["agravo_aplicado"] < 0.2:
        return "Perfil 2 - Segurado de Alta Cobertura e Baixo Custo Relativo"
    return "Perfil 1 - Frota de Alto Risco Operacional"


def calcular_kpis(original: pd.DataFrame) -> pd.DataFrame:
    kpis = original.groupby("cluster").agg(
        premio_por_veiculo_medio=("premio_por_veiculo", "mean"),
        lmi_por_veiculo_medio=("lmi_por_veiculo", "mean"),
        pct_motorista_licenciado=("motorista_licenciado", "mean"),
        valor_pago_historico_medio=("valor_pago_historico", "mean"),
        taxa_referral=("referral_pendente", "mean"),
        tempo_medio_conversao=("tempo_cotacao_emissao", "mean"),
        n_apolices=("numeroApolice", "count"),
    )
    medias_originais = original.groupby("cluster")[COLUNAS_19].mean()
    kpis["nome_perfil"] = medias_originais.apply(nomear_cluster, axis=1)
    return kpis.reset_index()


def calcular_indices_validacao(normalizada: pd.DataFrame) -> dict:
    X = normalizada[COLUNAS_19].to_numpy()
    rotulos = normalizada["cluster"].to_numpy()
    return {
        "Silhouette": silhouette_score(X, rotulos),
        "Davies-Bouldin": davies_bouldin_score(X, rotulos),
        "Calinski-Harabasz": calinski_harabasz_score(X, rotulos),
    }


def main():
    st.set_page_config(page_title="Segmentacao RCT", layout="wide")
    st.title("Segmentacao de Transportadoras - Produto RCT")
    st.caption(
        "Dashboard de visualizacao e validacao dos resultados da segmentacao por K-Means "
        "(Etapa 4 da metodologia)."
    )

    if not (DADOS_DIR / "matriz_normalizada_clusters.csv").exists():
        st.error(
            "Nenhum resultado de clusterizacao encontrado em data/processed/. "
            "Execute src/preprocessamento.py e src/clustering.py antes de abrir o dashboard."
        )
        return

    normalizada, original = carregar_dados()
    kpis = calcular_kpis(original)
    indices = calcular_indices_validacao(normalizada)

    st.header("KPIs por segmento")
    colunas = st.columns(len(kpis))
    for col, (_, linha) in zip(colunas, kpis.iterrows()):
        with col:
            st.subheader(linha["nome_perfil"])
            st.metric("Apolices no cluster", int(linha["n_apolices"]))
            st.metric("Premio medio / veiculo", _fmt_brl(linha["premio_por_veiculo_medio"]))
            st.metric("LMI medio / veiculo", _fmt_brl(linha["lmi_por_veiculo_medio"]))
            st.metric("Frotas c/ motorista licenciado", f"{linha['pct_motorista_licenciado']:.0%}")
            st.metric("Custo historico medio de sinistros", _fmt_brl(linha["valor_pago_historico_medio"]))
            st.metric("Taxa de referral", f"{linha['taxa_referral']:.0%}")
            st.metric("Tempo medio de conversao", f"{linha['tempo_medio_conversao']:.1f} dias")

    st.header("Visualizacao dos clusters (projecao PCA 2D)")
    projecao, variancia_pct = projetar_pca(normalizada)
    projecao = projecao.merge(
        original[["numeroApolice", "classe_risco", "premio_por_veiculo",
                  "valor_pago_historico", "referral_pendente", "tempo_cotacao_emissao"]],
        on="numeroApolice", how="left",
    )
    projecao["cluster"] = projecao["cluster"].astype(str)
    st.caption(f"Variancia explicada pelos dois primeiros componentes: {variancia_pct:.1%}")

    grafico = alt.Chart(projecao).mark_circle(size=70, opacity=0.7).encode(
        x=alt.X("PCA1", title="Componente Principal 1"),
        y=alt.Y("PCA2", title="Componente Principal 2"),
        color=alt.Color("cluster:N", title="Cluster"),
        tooltip=["numeroApolice", "cluster", "classe_risco", "premio_por_veiculo",
                 "valor_pago_historico", "referral_pendente", "tempo_cotacao_emissao"],
    ).interactive().properties(height=450)
    st.altair_chart(grafico, width="stretch")

    st.header("Tabela de perfis")
    variaveis_chave = [
        "classe_risco", "pct_autonomos", "qt_coberturas_ativas", "premio_por_veiculo",
        "lmi_por_veiculo", "valor_pago_historico", "referral_pendente", "tempo_cotacao_emissao",
    ]
    tabela_perfis = original.groupby("cluster")[variaveis_chave].mean().round(2)
    tabela_perfis.insert(0, "nome_perfil", kpis.set_index("cluster")["nome_perfil"])
    st.dataframe(tabela_perfis, width="stretch")

    st.subheader("Distribuicao de tipo de sinistro predominante por cluster")
    distribuicao_tipo = pd.crosstab(original["cluster"], original["tipo_sinistro_predominante"])
    distribuicao_tipo = distribuicao_tipo.rename(columns=TIPO_SINISTRO_LABEL)
    st.dataframe(distribuicao_tipo, width="stretch")

    st.header("Validacao tecnica do modelo")
    col1, col2, col3 = st.columns(3)
    col1.metric("Indice Silhouette", f"{indices['Silhouette']:.4f}")
    col2.metric("Indice Davies-Bouldin", f"{indices['Davies-Bouldin']:.4f}")
    col3.metric("Indice Calinski-Harabasz", f"{indices['Calinski-Harabasz']:.1f}")
    st.caption(
        "Silhouette: quanto mais proximo de 1, melhor. Davies-Bouldin: quanto mais proximo de 0, "
        "melhor. Calinski-Harabasz: quanto maior, melhor (sem limite superior)."
    )

    validacao_k = carregar_validacao_k()
    if validacao_k is not None:
        st.subheader("Metodo do Cotovelo e Indice de Silhouette por k (k=2..8)")
        col_a, col_b = st.columns(2)
        with col_a:
            grafico_cotovelo = alt.Chart(validacao_k).mark_line(point=True).encode(
                x=alt.X("k:O", title="Numero de clusters (k)"),
                y=alt.Y("inercia", title="Inercia (WCSS)"),
            ).properties(height=300)
            st.altair_chart(grafico_cotovelo, width="stretch")
        with col_b:
            grafico_silhouette = alt.Chart(validacao_k).mark_line(point=True, color="orange").encode(
                x=alt.X("k:O", title="Numero de clusters (k)"),
                y=alt.Y("silhouette", title="Indice de Silhouette"),
            ).properties(height=300)
            st.altair_chart(grafico_silhouette, width="stretch")


if __name__ == "__main__":
    main()
