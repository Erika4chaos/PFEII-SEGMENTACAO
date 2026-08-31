"""
Etapa 4 (Secao 3.5.4): dashboard Streamlit com tres secoes, navegadas por
botoes na barra lateral -- Segmentacao (projecao PCA 2D dos clusters,
tabela de perfis, KPIs e indices de validacao tecnica), Validacao de
Hardware (discriminacao do limiar de ~6 m/s^2 do firmware contra o dataset
publico de conducao de Ferreira Jr. et al., 2017) e Coligacao Conceitual
(passeio ilustrativo perfil x evento de hardware, sem juncao real de
dados).

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

CARTEIRA_COMPLETA = "Carteira completa"

NOME_PERFIL = {
    1: "Perfil 1 - Frota de Alto Risco Operacional",
    2: "Perfil 2 - Segurado de Alta Cobertura e Baixo Custo Relativo",
    3: "Perfil 3 - Cotacao em Referral ou Conversao Tardia",
}

# Paleta categorica (magenta/azul/verde) validada com scripts/validate_palette.py
# da skill dataviz para 3 series em grafico de dispersao (checagem --pairs all,
# light e dark): todos os checks passam, com WARN de contraste no slot magenta/
# verde mitigado pela legenda + tooltip + tabela de perfis ja presentes na pagina.
# Cor atribuida por PERFIL (numero de negocio), nao pelo indice arbitrario que o
# K-Means da ao cluster, para que a identidade visual nao mude entre execucoes.
CORES_PERFIL = {1: "#e87ba4", 2: "#2a78d6", 3: "#008300"}
COR_LINHA_DESTAQUE = "#c2185b"

# Citacao academica da fonte publica unica de validacao de hardware (Part
# B.5 do escopo tecnico) -- o valor bruto de SourceDataset em
# driver_conduct_harmonized.csv e um identificador de arquivo, nao a
# citacao a exibir na UI. Um segundo dataset (Yuksel, 2021) foi avaliado e
# deliberadamente descartado (unidades divergentes, rotulos com evidencia
# de serem por sessao de gravacao, nao por janela) -- ver docstring de
# src/validacao_hardware.py; nao reintroduzir sem repetir essa checagem.
FONTE_LABEL = {"jair_jr_driverBehaviorDataset_2016": "Ferreira Jr. et al. (2017)"}
FONTE_CITACAO = FONTE_LABEL["jair_jr_driverBehaviorDataset_2016"]

CATEGORIA_LABEL = {
    "ACCELERATION": "Aceleracao", "BRAKING": "Frenagem", "TURN": "Curva",
    "NON_AGGRESSIVE": "Nao agressivo", "LANE_CHANGE": "Mudanca de faixa",
}


def _fmt_brl(valor: float) -> str:
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _fmt_brl_compacto(valor: float) -> str:
    """Formato curto para caber nos cartoes de KPI estreitos (mockup usa o
    mesmo padrao: 'R$ 1,8 mi', 'R$ 112 mil')."""
    if abs(valor) >= 1_000_000:
        texto = f"R$ {valor / 1_000_000:.1f} mi"
    elif abs(valor) >= 1_000:
        texto = f"R$ {valor / 1_000:.0f} mil"
    else:
        texto = f"R$ {valor:.0f}"
    return texto.replace(".", ",")


@st.cache_data
def carregar_dados():
    normalizada = pd.read_csv(DADOS_DIR / "matriz_normalizada_clusters.csv")
    original = pd.read_csv(DADOS_DIR / "matriz_original_clusters.csv")
    return normalizada, original


@st.cache_data
def carregar_validacao_k():
    caminho = DADOS_DIR / "validacao_k.csv"
    return pd.read_csv(caminho) if caminho.exists() else None


@st.cache_data
def carregar_significancia():
    caminho = DADOS_DIR / "perfis_testes_significancia.csv"
    return pd.read_csv(caminho) if caminho.exists() else None


@st.cache_data
def carregar_conduta_harmonizada():
    """Le a saida ja harmonizada (unidades + rotulos) de
    src/validacao_hardware.py, ja restrita a fonte unica escolhida
    (Ferreira Jr. et al., 2017) -- o dashboard nunca reimplementa a
    verificacao de escala m/s^2 nem o mapeamento de rotulos, apenas
    consome o CSV que o script produz (ver Part B.5/A.7 do escopo
    tecnico)."""
    caminho = DADOS_DIR / "driver_conduct_harmonized.csv"
    if not caminho.exists():
        return None
    df = pd.read_csv(caminho)
    df["FonteLabel"] = df["SourceDataset"].map(FONTE_LABEL).fillna(df["SourceDataset"])
    return df


@st.cache_data
def carregar_conduta_metricas():
    """Matriz de confusao / precisao / recall / F1 (pooled + por categoria
    harmonizada), ja calculada por src/validacao_hardware.py -- reaproveitada
    aqui, nunca recalculada, para nao duplicar a logica de avaliacao entre
    script e dashboard."""
    caminho = DADOS_DIR / "driver_conduct_metrics.csv"
    return pd.read_csv(caminho) if caminho.exists() else None


def projetar_pca(normalizada: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    pca = PCA(n_components=2, random_state=42)
    componentes = pca.fit_transform(normalizada[COLUNAS_19])
    projecao = normalizada[["numeroApolice", "cluster"]].copy()
    projecao["PCA1"] = componentes[:, 0]
    projecao["PCA2"] = componentes[:, 1]
    return projecao, float(pca.explained_variance_ratio_.sum())


def numero_perfil(medias: pd.Series) -> int:
    """Heuristica simples que confronta as medias do cluster no espaco
    original com os tres perfis-alvo da Secao 3.4, para apresentacao em
    linguagem de negocio (sem expor os detalhes matematicos do modelo)."""
    if medias["referral_pendente"] > 0.5:
        return 3
    if medias["qt_coberturas_ativas"] >= 5 and medias["agravo_aplicado"] < 0.2:
        return 2
    return 1


def mapear_cluster_para_perfil(original: pd.DataFrame) -> dict:
    medias = original.groupby("cluster")[COLUNAS_19].mean()
    return medias.apply(numero_perfil, axis=1).to_dict()


def calcular_kpis_por_cluster(original: pd.DataFrame, mapa_perfil: dict) -> pd.DataFrame:
    kpis = original.groupby("cluster").agg(
        premio_por_veiculo_medio=("premio_por_veiculo", "mean"),
        lmi_por_veiculo_medio=("lmi_por_veiculo", "mean"),
        pct_motorista_licenciado=("motorista_licenciado", "mean"),
        valor_pago_historico_medio=("valor_pago_historico", "mean"),
        taxa_referral=("referral_pendente", "mean"),
        tempo_medio_conversao=("tempo_cotacao_emissao", "mean"),
        n_apolices=("numeroApolice", "count"),
    ).reset_index()
    kpis["perfil_numero"] = kpis["cluster"].map(mapa_perfil)
    kpis["nome_perfil"] = kpis["perfil_numero"].map(NOME_PERFIL)
    kpis["cor"] = kpis["perfil_numero"].map(CORES_PERFIL)
    return kpis.sort_values("perfil_numero").reset_index(drop=True)


def calcular_kpis_gerais(original: pd.DataFrame) -> dict:
    return {
        "n_apolices": len(original),
        "premio_por_veiculo_medio": original["premio_por_veiculo"].mean(),
        "lmi_por_veiculo_medio": original["lmi_por_veiculo"].mean(),
        "pct_motorista_licenciado": original["motorista_licenciado"].mean(),
        "valor_pago_historico_medio": original["valor_pago_historico"].mean(),
        "taxa_referral": original["referral_pendente"].mean(),
    }


def calcular_indices_validacao(normalizada: pd.DataFrame) -> dict:
    X = normalizada[COLUNAS_19].to_numpy()
    rotulos = normalizada["cluster"].to_numpy()
    return {
        "Silhouette": silhouette_score(X, rotulos),
        "Davies-Bouldin": davies_bouldin_score(X, rotulos),
        "Calinski-Harabasz": calinski_harabasz_score(X, rotulos),
    }


def aplicar_estilo():
    st.markdown(
        """
        <style>
        h1 {
            color: #c2185b;
            border-bottom: 3px solid #e87ba4;
            padding-bottom: 0.3rem;
        }
        h2 {
            color: #ad1457;
            border-bottom: 2px solid #f5c2d8;
            padding-bottom: 0.2rem;
            margin-top: 1.2rem;
        }
        h3 { color: #ad1457; }
        [data-testid="stMetric"] {
            background-color: #fff0f5;
            border: 1px solid #f5c2d8;
            border-radius: 12px;
            padding: 12px 14px 8px 14px;
        }
        [data-testid="stMetricLabel"] { color: #a8375f; }
        [data-testid="stMetricValue"] { color: #c2185b; font-size: 1.5rem; }

        [data-testid="stSidebar"] {
            background-color: #2e2440;
        }
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span, [data-testid="stSidebar"] div {
            color: #cfc4dc;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #ffffff;
            border: none;
        }
        [data-testid="stSidebar"] button[kind="secondary"] {
            background-color: transparent;
            border-color: transparent;
            color: #cfc4dc;
            text-align: left;
            justify-content: flex-start;
        }
        [data-testid="stSidebar"] button[kind="secondary"]:hover {
            background-color: #5a4a75;
            color: #ffffff;
        }
        .side-badge {
            background: #5a4a75;
            border-radius: 10px;
            padding: 12px;
            font-size: 12px;
            line-height: 1.5;
            color: #cfc4dc;
            margin-top: 12px;
        }
        .side-badge b { color: #e87ba4; }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #f5c2d8 !important;
        }

        .pill-badge {
            display: inline-block;
            background: #f6e6ee;
            color: #9d5577;
            font-size: 12px;
            font-weight: 600;
            padding: 5px 12px;
            border-radius: 999px;
        }
        .insight-item { padding: 8px 0; border-bottom: 1px solid #f5c2d8; font-size: 13px; }
        .insight-item:last-child { border-bottom: none; }
        .insight-dot {
            display: inline-block; width: 8px; height: 8px; border-radius: 50%;
            background: #e87ba4; margin-right: 8px;
        }
        .insight-title { font-weight: 700; color: #44355b; }
        .insight-caption { color: #65596f; font-size: 12px; }

        .tag-pill {
            display: inline-block; padding: 3px 10px; border-radius: 999px;
            font-size: 12px; font-weight: 600; color: #ffffff;
        }
        .no-join-banner {
            background: #fff3cd; border: 1px solid #e8b83c; border-radius: 10px;
            padding: 14px 16px; color: #7a5a00; font-size: 13px; line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


NAV_OPCOES = ["Segmentacao", "Validacao de Hardware", "Coligacao Conceitual"]


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<h2 style="margin-top:0;">Dashboard de <span style="color:#e87ba4;">Risco</span></h2>'
            '<div style="font-size:12px;margin-bottom:14px;">RCT Transportador</div>',
            unsafe_allow_html=True,
        )

        if "view" not in st.session_state:
            st.session_state["view"] = NAV_OPCOES[0]

        for opcao in NAV_OPCOES:
            ativo = st.session_state["view"] == opcao
            if st.button(
                opcao, key=f"nav_{opcao.replace(' ', '_')}",
                width="stretch", type=("primary" if ativo else "secondary"),
            ):
                st.session_state["view"] = opcao

        st.markdown(
            '<div class="side-badge"><b>Etapa 4 da metodologia.</b> '
            'Segmentacao via K-Means (k=3) sobre 19 variaveis derivadas (Quadro 2), '
            'complementada por validacao independente da camada de hardware (Parte B). '
            'Correlacao espacial com criminalidade/vulnerabilidade e alertas de '
            'telemetria por apolice ficam registrados como extensao futura (Secao 3.8).'
            '</div>',
            unsafe_allow_html=True,
        )

    return st.session_state["view"]


def render_linha_kpis(dados: dict):
    colunas = st.columns(6)
    itens = [
        ("Apolices", str(int(dados["n_apolices"]))),
        ("Premio medio / veic.", _fmt_brl_compacto(dados["premio_por_veiculo_medio"])),
        ("LMI medio / veic.", _fmt_brl_compacto(dados["lmi_por_veiculo_medio"])),
        ("Custo hist. medio", _fmt_brl_compacto(dados["valor_pago_historico_medio"])),
        ("Motorista licenciado", f"{dados['pct_motorista_licenciado']:.0%}"),
        ("Taxa de referral", f"{dados['taxa_referral']:.0%}"),
    ]
    for coluna, (rotulo, valor) in zip(colunas, itens):
        coluna.metric(rotulo, valor)


def render_scatter(projecao: pd.DataFrame, perfil_selecionado_numero):
    ordem_nomes = [NOME_PERFIL[n] for n in sorted(projecao["perfil_numero"].unique())]
    ordem_cores = [CORES_PERFIL[n] for n in sorted(projecao["perfil_numero"].unique())]

    if perfil_selecionado_numero is None:
        opacidade = alt.value(0.75)
    else:
        opacidade = alt.condition(
            alt.datum.perfil_numero == perfil_selecionado_numero, alt.value(0.85), alt.value(0.12)
        )

    grafico = alt.Chart(projecao).mark_circle(size=70).encode(
        x=alt.X("PCA1", title="Componente Principal 1"),
        y=alt.Y("PCA2", title="Componente Principal 2"),
        color=alt.Color(
            "nome_perfil:N", title="Perfil",
            scale=alt.Scale(domain=ordem_nomes, range=ordem_cores),
        ),
        opacity=opacidade,
        tooltip=["numeroApolice", "nome_perfil", "classe_risco", "premio_por_veiculo",
                 "valor_pago_historico", "referral_pendente", "tempo_cotacao_emissao"],
    ).interactive().properties(height=420)
    st.altair_chart(grafico, width="stretch")


def render_insights(significancia: pd.DataFrame, top_n: int = 5):
    if significancia is None or significancia.empty:
        st.caption("Rode src/perfilamento.py para gerar os testes de significancia.")
        return
    top = significancia.sort_values("p_valor").head(top_n)
    linhas = []
    for _, linha in top.iterrows():
        titulo = f"<code>{linha['variavel']}</code>"
        legenda = f"{linha['teste']} · p = {linha['p_valor']:.2e}"
        linhas.append(
            f'<div class="insight-item"><span class="insight-dot"></span>'
            f'<span class="insight-title">{titulo}</span><br>'
            f'<span class="insight-caption" style="margin-left:16px;">{legenda}</span></div>'
        )
    st.markdown("".join(linhas), unsafe_allow_html=True)


def render_bar_custo(kpis_cluster: pd.DataFrame):
    grafico = alt.Chart(kpis_cluster).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("nome_perfil:N", title=None, axis=alt.Axis(labelAngle=0, labels=False)),
        y=alt.Y("valor_pago_historico_medio:Q", title="Custo historico medio (R$)"),
        color=alt.Color(
            "nome_perfil:N", title="Perfil",
            scale=alt.Scale(
                domain=[NOME_PERFIL[n] for n in sorted(kpis_cluster["perfil_numero"])],
                range=[CORES_PERFIL[n] for n in sorted(kpis_cluster["perfil_numero"])],
            ),
        ),
        tooltip=["nome_perfil", alt.Tooltip("valor_pago_historico_medio:Q", format=",.2f")],
    ).properties(height=260)
    st.altair_chart(grafico, width="stretch")


def render_tabela_perfis(kpis_cluster: pd.DataFrame):
    tabela = kpis_cluster[[
        "nome_perfil", "n_apolices", "premio_por_veiculo_medio",
        "valor_pago_historico_medio", "taxa_referral",
    ]].copy()
    tabela["premio_por_veiculo_medio"] = tabela["premio_por_veiculo_medio"].map(_fmt_brl)
    tabela["valor_pago_historico_medio"] = tabela["valor_pago_historico_medio"].map(_fmt_brl)
    tabela["taxa_referral"] = tabela["taxa_referral"].map(lambda v: f"{v:.0%}")
    tabela.columns = ["Perfil", "Apolices", "Premio/veic.", "Custo hist.", "Referral"]
    st.dataframe(tabela, width="stretch", hide_index=True)


def render_view_visao_geral(projecao, kpis_cluster, gerais, significancia, perfil_sel_nome):
    perfil_sel_numero = None
    dados_kpi = gerais
    if perfil_sel_nome != CARTEIRA_COMPLETA:
        linha_sel = kpis_cluster[kpis_cluster["nome_perfil"] == perfil_sel_nome].iloc[0]
        perfil_sel_numero = int(linha_sel["perfil_numero"])
        dados_kpi = linha_sel.to_dict()

    st.subheader("KPIs")
    render_linha_kpis(dados_kpi)

    col_scatter, col_insights = st.columns([1.5, 1])
    with col_scatter:
        with st.container(border=True):
            st.markdown("##### Clusters no espaco PCA")
            st.caption("Cada ponto e uma apolice; cores por perfil. Passe o mouse para detalhes.")
            render_scatter(projecao, perfil_sel_numero)
    with col_insights:
        with st.container(border=True):
            st.markdown("##### Variaveis mais discriminantes")
            st.caption("ANOVA (continuas) e qui-quadrado (binarias/categoricas) entre os 3 perfis")
            render_insights(significancia)

    col_bar, col_tabela = st.columns([1, 1])
    with col_bar:
        with st.container(border=True):
            st.markdown("##### Custo historico medio de sinistros por perfil")
            st.caption("valor_pago_historico extraido do campo qst3 (regex)")
            render_bar_custo(kpis_cluster)
    with col_tabela:
        with st.container(border=True):
            st.markdown("##### Perfis da carteira")
            st.caption("Estatisticas descritivas no espaco original (medias)")
            render_tabela_perfis(kpis_cluster)


def render_view_clusters(projecao, variancia_pct):
    st.caption(f"Variancia explicada pelos dois primeiros componentes: {variancia_pct:.1%}")
    with st.container(border=True):
        render_scatter(projecao, None)


def render_view_validacao_indices(normalizada, validacao_k, significancia):
    indices = calcular_indices_validacao(normalizada)
    col1, col2, col3 = st.columns(3)
    col1.metric("Indice Silhouette", f"{indices['Silhouette']:.4f}")
    col2.metric("Indice Davies-Bouldin", f"{indices['Davies-Bouldin']:.4f}")
    col3.metric("Indice Calinski-Harabasz", f"{indices['Calinski-Harabasz']:.1f}")
    st.caption(
        "Silhouette: quanto mais proximo de 1, melhor. Davies-Bouldin: quanto mais proximo de 0, "
        "melhor. Calinski-Harabasz: quanto maior, melhor (sem limite superior)."
    )

    if validacao_k is not None:
        st.markdown("##### Metodo do Cotovelo e Indice de Silhouette por k (k=2..8)")
        col_a, col_b = st.columns(2)
        with col_a:
            grafico_cotovelo = alt.Chart(validacao_k).mark_line(
                point=alt.OverlayMarkDef(color=COR_LINHA_DESTAQUE), color=COR_LINHA_DESTAQUE
            ).encode(
                x=alt.X("k:O", title="Numero de clusters (k)"),
                y=alt.Y("inercia", title="Inercia (WCSS)"),
            ).properties(height=280)
            st.altair_chart(grafico_cotovelo, width="stretch")
        with col_b:
            grafico_silhouette = alt.Chart(validacao_k).mark_line(
                point=alt.OverlayMarkDef(color=COR_LINHA_DESTAQUE), color=COR_LINHA_DESTAQUE
            ).encode(
                x=alt.X("k:O", title="Numero de clusters (k)"),
                y=alt.Y("silhouette", title="Indice de Silhouette"),
            ).properties(height=280)
            st.altair_chart(grafico_silhouette, width="stretch")

    st.markdown("##### Testes de significancia (todas as variaveis)")
    if significancia is not None:
        st.dataframe(
            significancia, width="stretch", hide_index=True,
            column_config={
                "estatistica": st.column_config.NumberColumn(format="%.4f"),
                "p_valor": st.column_config.NumberColumn(format="%.2e"),
            },
        )
    else:
        st.caption("Rode src/perfilamento.py para gerar os testes de significancia.")


def render_view_segmentacao(projecao, variancia_pct, kpis_cluster, gerais, significancia,
                             normalizada, validacao_k):
    opcoes_segmento = [CARTEIRA_COMPLETA] + kpis_cluster["nome_perfil"].tolist()
    segmento_selecionado = st.pills(
        "Filtrar por perfil", opcoes_segmento, default=CARTEIRA_COMPLETA, key="segmento_pills",
    )
    if not segmento_selecionado:
        segmento_selecionado = CARTEIRA_COMPLETA

    aba_geral, aba_clusters, aba_validacao = st.tabs(
        ["Visao geral", "Clusters (PCA)", "Validacao e insights"]
    )
    with aba_geral:
        render_view_visao_geral(projecao, kpis_cluster, gerais, significancia, segmento_selecionado)
    with aba_clusters:
        render_view_clusters(projecao, variancia_pct)
    with aba_validacao:
        render_view_validacao_indices(normalizada, validacao_k, significancia)


# ---------------------------------------------------------------------------
# Aba 2 -- Validacao de Hardware (Part B.5 / A.7)
# ---------------------------------------------------------------------------

def render_metodologia_hardware():
    st.info(
        "**Substituicao de fonte de dados (documentada, nao um desvio silencioso).** "
        "O plano original validava o firmware contra o UAH-DriveSet (ROMERA; BERGASA; "
        "ARROYO, 2016), cujo host oficial e espelhos conhecidos ficaram indisponiveis "
        f"durante a execucao do PFE II. A fonte de substituicao -- {FONTE_CITACAO} -- "
        "fornece janelas estatisticas ja pre-extraidas por evento (media, maximo, "
        "minimo, variancia, assimetria, curtose, desvio padrao por eixo), nao um fluxo "
        "continuo bruto com rotulo de trajeto. Por isso a validacao aqui e de "
        "**discriminacao do tipo de evento a partir de estatisticas de janela**, e nao "
        "de deteccao em fluxo continuo -- o que, na verdade, aproxima-se mais do que o "
        "proprio firmware produz (eventos resumidos, nunca o fluxo bruto)."
    )
    st.caption(
        "Uma segunda fonte (Yuksel, 2021) foi avaliada e descartada deliberadamente: "
        "suas unidades divergiam do documentado (exigindo harmonizacao empirica "
        "fragil) e seus rotulos mostravam estatisticas de pico quase constantes ao "
        "longo de dezenas de janelas consecutivas, sinal de rotulagem por sessao de "
        "gravacao e nao por janela -- uma evidencia mais fraca que a rotulagem de "
        f"{FONTE_CITACAO}, feita por pesquisadores contra video de referencia. "
        "Ver docstring de `src/validacao_hardware.py`."
    )


def render_confusao(metricas: pd.DataFrame):
    st.markdown("##### Matriz de confusao e metricas de discriminacao")
    linha = metricas[metricas["categoria"] == "todas"].iloc[0]
    st.caption(
        f"n = {int(linha['n'])} janelas avaliadas -- verificacao de viabilidade em "
        "amostra pequena, na mesma postura ja adotada pelos testes de bancada e rodagem "
        "no veiculo do autor (Secao 3.3), nao uma validacao estatisticamente "
        "dimensionada."
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("n", int(linha["n"]))
    col2.metric("Precisao", f"{linha['precision']:.1%}")
    col3.metric("Recall", f"{linha['recall']:.1%}")
    col4.metric("F1", f"{linha['f1']:.3f}")
    st.markdown(
        f"VP={int(linha['tp'])} · FN={int(linha['fn'])} · "
        f"VN={int(linha['tn'])} · FP={int(linha['fp'])}"
    )


def render_recall_por_categoria(metricas: pd.DataFrame):
    st.markdown("##### Recall por categoria harmonizada de evento")
    st.caption(
        "ACCELERATION / BRAKING / TURN sao as categorias que o firmware efetivamente "
        "classifica. O n de cada categoria e mostrado junto ao recall -- ate 6 janelas "
        "em algumas classes -- para que a leitura nao pareca mais robusta do que e."
    )
    dados = metricas[metricas["categoria"] != "todas"].copy()
    dados["CategoriaLabel"] = dados["categoria"].map(CATEGORIA_LABEL)
    dados["RotuloEixo"] = dados["CategoriaLabel"] + " (n=" + dados["n"].astype(int).astype(str) + ")"

    grafico = alt.Chart(dados).mark_bar(
        cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color=CORES_PERFIL[1],
    ).encode(
        x=alt.X("RotuloEixo:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("recall:Q", title="Recall", axis=alt.Axis(format="%")),
        tooltip=["CategoriaLabel", "n", alt.Tooltip("recall:Q", format=".1%")],
    ).properties(height=280)
    st.altair_chart(grafico, width="stretch")


def render_resumo_motoristas(conduta: pd.DataFrame):
    st.markdown("##### Resumo por motorista")
    resumo = conduta.groupby("GroupID").agg(
        n_eventos=("WindowIndex", "count"),
        n_categorias=("EventCategory", "nunique"),
        taxa_acima_limiar=("HarshPredicted_6mps2", "mean"),
    ).reset_index().sort_values("GroupID")

    cores = [CORES_PERFIL[1], CORES_PERFIL[2], CORES_PERFIL[3]]
    colunas = st.columns(len(resumo))
    for coluna, (_, linha), cor in zip(colunas, resumo.iterrows(), cores):
        with coluna:
            with st.container(border=True):
                st.markdown(
                    f'<span class="tag-pill" style="background:{cor};">{linha["GroupID"]}</span>',
                    unsafe_allow_html=True,
                )
                st.metric("Eventos", int(linha["n_eventos"]))
                st.metric("Categorias distintas", int(linha["n_categorias"]))
                st.metric("Acima do limiar (~6 m/s^2)", f"{linha['taxa_acima_limiar']:.0%}")


def render_grafico_motoristas(conduta: pd.DataFrame):
    dados = conduta.copy()
    dados["CategoriaLabel"] = dados["EventCategory"].map(CATEGORIA_LABEL).fillna(dados["EventCategory"])
    agregado = dados.groupby(["GroupID", "CategoriaLabel"]).size().reset_index(name="n_eventos")

    grafico = alt.Chart(agregado).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("GroupID:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("n_eventos:Q", title="Numero de eventos"),
        color=alt.Color("CategoriaLabel:N", title="Categoria"),
        xOffset="CategoriaLabel:N",
        tooltip=["GroupID", "CategoriaLabel", "n_eventos"],
    ).properties(height=300)
    st.altair_chart(grafico, width="stretch")


def render_tabela_motoristas(conduta: pd.DataFrame):
    st.markdown("##### Dataset dos motoristas")
    st.caption(
        f"Janelas de evento por sessao de gravacao (motorista), ja harmonizadas "
        f"(unidades e rotulos) por src/validacao_hardware.py. Fonte: {FONTE_CITACAO}."
    )
    tabela = conduta.copy()
    tabela["CategoriaLabel"] = tabela["EventCategory"].map(CATEGORIA_LABEL).fillna(tabela["EventCategory"])
    tabela["AcimaLimiar"] = tabela["HarshPredicted_6mps2"].map({1: "sim", 0: "nao"})
    tabela = tabela[[
        "GroupID", "EventLabel", "CategoriaLabel", "WindowIndex",
        "PeakDynamicAccel_mps2", "AcimaLimiar",
    ]]
    tabela.columns = [
        "Motorista", "Rotulo original", "Categoria", "Janela",
        "Aceleracao de pico (m/s^2)", "Acima do limiar (~6 m/s^2)",
    ]
    st.dataframe(
        tabela, width="stretch", hide_index=True,
        column_config={"Aceleracao de pico (m/s^2)": st.column_config.NumberColumn(format="%.2f")},
    )
    st.caption(
        "**Legenda:** `Motorista` -- identificador anonimo da sessao de gravacao no "
        "dataset publico (nao e uma pessoa identificada). `Categoria` -- tipo de "
        "manobra harmonizado pelo firmware (Aceleracao, Frenagem, Curva, Nao "
        "agressivo, Mudanca de faixa). `Janela` -- indice da janela de tempo rotulada "
        "dentro daquela sessao. `Aceleracao de pico` -- magnitude dinamica maxima "
        "medida naquela janela, em m/s^2, com a gravidade ja removida. `Acima do "
        "limiar` -- se essa magnitude ultrapassa o limiar de deteccao do firmware "
        "(~6 m/s^2)."
    )


def render_view_validacao_hardware(conduta, metricas):
    if conduta is None or metricas is None:
        st.error(
            "Nenhum resultado de validacao de hardware encontrado em data/processed/. "
            "Execute `python src/validacao_hardware.py` antes de abrir esta aba."
        )
        return

    render_metodologia_hardware()

    # Matriz de confusao (render_confusao) desativada por pedido -- por
    # enquanto mostra so o resumo/grafico/tabela dos motoristas abaixo.
    # Funcao mantida para reativacao futura.
    with st.container(border=True):
        render_resumo_motoristas(conduta)
        render_grafico_motoristas(conduta)
        render_tabela_motoristas(conduta)

    with st.container(border=True):
        render_recall_por_categoria(metricas)


# ---------------------------------------------------------------------------
# Aba 3 -- Coligacao Conceitual (Part A.7 / C) -- passeio ilustrativo, sem
# juncao real entre dados sinteticos de apolice e dados publicos de conducao.
# ---------------------------------------------------------------------------

def render_exemplo_perfil(perfil_numero: int, apolice: pd.Series, evento: pd.Series, narrativa: str):
    st.markdown(f"**{NOME_PERFIL[perfil_numero]}**")
    col_apolice, col_evento = st.columns(2)
    with col_apolice:
        with st.container(border=True):
            st.markdown("**Apolice (dado sintetico)**")
            st.caption("Cadastro/cotacao -- lado da segmentacao (Parte A)")
            st.markdown(
                f"- Numero: `{apolice['numeroApolice']}`\n"
                f"- Frota total: {apolice['total_veiculos']:.0f} veiculos "
                f"({apolice['pct_autonomos']:.0%} autonomos/terceiros)\n"
                f"- Classe de risco: {apolice['classe_risco']:.0f}/5\n"
                f"- Coberturas ativas: {apolice['qt_coberturas_ativas']:.0f}\n"
                f"- Referral pendente: {'sim' if apolice['referral_pendente'] else 'nao'}\n"
                f"- Custo historico declarado: {_fmt_brl(apolice['valor_pago_historico'])}"
            )
    with col_evento:
        with st.container(border=True):
            st.markdown(f"**Evento de conducao (dataset publico) · {FONTE_CITACAO}**")
            st.caption("Janela rotulada -- lado da validacao de hardware (Parte B)")
            st.markdown(
                f"- Categoria harmonizada: {CATEGORIA_LABEL.get(evento['EventCategory'], evento['EventCategory'])}\n"
                f"- Rotulo original: {evento['EventLabel']}\n"
                f"- Magnitude dinamica de pico: {evento['PeakDynamicAccel_mps2']:.2f} m/s^2\n"
                f"- Acima do limiar (~6 m/s^2): {'sim' if evento['HarshPredicted_6mps2'] else 'nao'}"
            )
    st.caption(narrativa)
    st.divider()


def render_view_coligacao(original: pd.DataFrame, mapa_perfil: dict, conduta: pd.DataFrame):
    st.markdown(
        '<div class="no-join-banner">'
        '<b>Nao ha juncao real entre os dois conjuntos de dados nesta aba.</b> '
        'A apolice sintetica e o evento de conducao publico abaixo nao compartilham '
        'nenhuma chave/entidade real -- nao existe motorista, veiculo ou vinculo '
        'contratual em comum entre uma apolice gerada sinteticamente para esta PoC e '
        'um motorista de um dataset publico de conducao. O par exibido e puramente '
        'ilustrativo: narra o TIPO de sinal que, em producao, um dispositivo ESP32 '
        'proprio instalado na frota dessa apolice alimentaria de volta na '
        'classificacao (Secao 2.2.5 / 3.4.1) -- nao um resultado estatistico '
        'demonstrado.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("##### Como a coligacao funcionaria (arquitetura conceitual, Secao 2.2.5/3.4)")
    st.caption(
        "Este diagrama descreve o MECANISMO de token/pseudonimizacao do design do "
        "projeto -- nenhuma destas etapas esta implementada neste codigo (nao ha "
        "sistema de tokens, ingestao de telemetria ou reassociacao rodando aqui). "
        "E a arquitetura de producao descrita no texto do TCC, mostrada para "
        "explicar como um vinculo REAL seria possivel sem expor a identidade da "
        "apolice ao dispositivo nem ao pipeline de telemetria (LGPD, minimizacao de "
        "dados -- Parte C)."
    )
    st.graphviz_chart(
        """
        digraph fluxo {
            rankdir=LR;
            bgcolor="transparent";
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin=0.18];
            edge [fontname="Helvetica", fontsize=9, color="#8a7f97", fontcolor="#65596f"];

            perfil [label="Perfil historico\n(cotacao/apolice)\nclassificado Perfil 1/2/3", fillcolor="#e87ba4", fontcolor="white"];
            token [label="Token pseudonimo\nassociado a frota/veiculo\n(nunca o numero da apolice)", fillcolor="#f6e6ee", fontcolor="#44355b"];
            dispositivo [label="Dispositivo ESP32\nna frota\n(so ve o token)", fillcolor="#2a78d6", fontcolor="white"];
            ingestao [label="Ingestao de telemetria\nrecebe token + evento\nresumido (nunca GPS bruto)", fillcolor="#f6e6ee", fontcolor="#44355b"];
            reassociacao [label="Reassociacao\nso a seguradora tem o\nmapa token -> apolice", fillcolor="#f6e6ee", fontcolor="#44355b"];
            classificacao [label="Classificacao\nreforcada ou corrigida\n(decisao com humano no loop)", fillcolor="#008300", fontcolor="white"];

            perfil -> token [label="  emissao do token  "];
            token -> dispositivo [label="  gravado no dispositivo  "];
            dispositivo -> ingestao [label="  eventos + token  "];
            ingestao -> reassociacao [label="  lookup do token  "];
            reassociacao -> classificacao [label="  feedback  "];
            classificacao -> perfil [label="  atualiza o perfil  ", style=dashed, color="#c2185b", fontcolor="#c2185b"];
        }
        """
    )
    st.caption(
        "O passo critico de privacidade e 'Reassociacao': o dispositivo e o pipeline "
        "de ingestao nunca veem o numero da apolice, so o token -- somente a "
        "seguradora, que emitiu o token, consegue voltar do token ate a apolice. "
        "A seta tracejada de volta a 'Classificacao' e o loop de feedback descrito "
        "na Secao 3.4.1: e sempre um sinal de apoio a decisao, nunca uma decisao "
        "automatica (Parte C, LGPD Art. 20)."
    )
    st.divider()

    if conduta is None or conduta.empty:
        st.info("Execute src/validacao_hardware.py para habilitar os exemplos ilustrativos.")
        return

    original = original.copy()
    original["perfil_numero"] = original["cluster"].map(mapa_perfil)

    st.markdown("##### Exemplos ilustrativos por perfil")
    st.caption(
        "Um exemplo por perfil, escolhido para tornar a narrativa concreta -- nao "
        "uma amostra representativa nem um resultado estatistico (ver aviso acima: "
        "nenhuma relacao real entre os dois conjuntos de dados existe nesta PoC)."
    )

    cand1 = original[original["perfil_numero"] == 1]
    if not cand1.empty:
        apolice1 = cand1.sort_values(["valor_pago_historico", "classe_risco"], ascending=False).iloc[0]
        eventos1 = conduta[
            (conduta["ValidationRole"] == "harsh")
            & (conduta["EventCategory"].isin(["TURN", "BRAKING"]))
            & (conduta["HarshPredicted_6mps2"] == 1)
        ]
        if eventos1.empty:
            eventos1 = conduta[conduta["ValidationRole"] == "harsh"]
        evento1 = eventos1.sort_values("PeakDynamicAccel_mps2", ascending=False).iloc[0]
        render_exemplo_perfil(
            1, apolice1, evento1,
            "Narrativa ilustrativa: se o dispositivo ESP32 desta frota (ja "
            "sinalizada como alto risco pela segmentacao historica) registrasse "
            "um evento de conducao com esta magnitude, esse sumario "
            "reforcaria/confirmaria a classificacao existente (Secao 3.4.1)."
        )

    cand2 = original[original["perfil_numero"] == 2]
    if not cand2.empty:
        apolice2 = cand2.sort_values(["qt_coberturas_ativas", "lmi_por_veiculo"], ascending=False).iloc[0]
        eventos2 = conduta[conduta["ValidationRole"] == "baseline"]
        if eventos2.empty:
            eventos2 = conduta
        evento2 = eventos2.sort_values("PeakDynamicAccel_mps2", ascending=True).iloc[0]
        render_exemplo_perfil(
            2, apolice2, evento2,
            "Narrativa ilustrativa: um evento sem sinal de conducao agressiva "
            "reforcaria a leitura de baixo risco ja indicada pela alta cobertura "
            "e baixo custo historico desta apolice -- consistente com a manutencao "
            "de condicoes comerciais favoraveis."
        )

    cand3 = original[original["perfil_numero"] == 3]
    if not cand3.empty:
        apolice3 = cand3.sort_values(["tempo_cotacao_emissao", "referral_pendente"], ascending=False).iloc[0]
        mediana_magnitude = conduta["PeakDynamicAccel_mps2"].median()
        evento3 = conduta.loc[(conduta["PeakDynamicAccel_mps2"] - mediana_magnitude).abs().idxmin()]
        render_exemplo_perfil(
            3, apolice3, evento3,
            "Narrativa ilustrativa: cotacoes em referral ou conversao tardia ainda "
            "nao tem um padrao comportamental estabelecido -- um primeiro evento "
            "real, agressivo ou nao, teria peso desproporcional em reduzir essa "
            "incerteza, ao contrario dos Perfis 1 e 2, onde a evidencia historica ja "
            "aponta uma direcao."
        )


def render_view_exportar():
    arquivos = {
        "Matriz original com rotulos de cluster": "matriz_original_clusters.csv",
        "Matriz normalizada com rotulos de cluster": "matriz_normalizada_clusters.csv",
        "Estatisticas descritivas por cluster": "perfis_estatisticas_descritivas.csv",
        "Testes de significancia por variavel": "perfis_testes_significancia.csv",
        "Curva de validacao de k (cotovelo/silhouette)": "validacao_k.csv",
        "Conduta harmonizada por janela (Ferreira Jr. et al., 2017)": "driver_conduct_harmonized.csv",
        "Metricas de discriminacao (pooled + por categoria)": "driver_conduct_metrics.csv",
        "Confundimento categoria x sessao (GroupID)": "driver_conduct_confound.csv",
    }
    for titulo, nome_arquivo in arquivos.items():
        caminho = DADOS_DIR / nome_arquivo
        if not caminho.exists():
            continue
        with st.container(border=True):
            col_desc, col_botao = st.columns([3, 1])
            col_desc.markdown(f"**{titulo}**  \n`{nome_arquivo}`")
            col_botao.download_button(
                "Baixar CSV", data=caminho.read_bytes(), file_name=nome_arquivo,
                mime="text/csv", key=f"download_{nome_arquivo}", width="stretch",
            )


def main():
    st.set_page_config(page_title="Segmentacao RCT", layout="wide")
    aplicar_estilo()
    view = render_sidebar()

    col_titulo, col_badge = st.columns([4, 1])
    with col_titulo:
        st.title("Segmentacao de Transportadoras - Produto RCT")
        st.caption(
            "Dashboard de visualizacao e validacao dos resultados da segmentacao por "
            "K-Means (Parte A) e da camada de telemetria embarcada (Parte B)."
        )
    with col_badge:
        st.markdown(
            '<div style="text-align:right;padding-top:18px;">'
            '<span class="pill-badge">Dados sinteticos (PoC)</span></div>',
            unsafe_allow_html=True,
        )

    if not (DADOS_DIR / "matriz_normalizada_clusters.csv").exists():
        st.error(
            "Nenhum resultado de clusterizacao encontrado em data/processed/. "
            "Execute src/preprocessamento.py e src/clustering.py antes de abrir o dashboard."
        )
        return

    normalizada, original = carregar_dados()
    mapa_perfil = mapear_cluster_para_perfil(original)
    kpis_cluster = calcular_kpis_por_cluster(original, mapa_perfil)
    gerais = calcular_kpis_gerais(original)
    significancia = carregar_significancia()
    validacao_k = carregar_validacao_k()
    conduta = carregar_conduta_harmonizada()
    metricas = carregar_conduta_metricas()

    projecao, variancia_pct = projetar_pca(normalizada)
    projecao = projecao.merge(
        original[["numeroApolice", "classe_risco", "premio_por_veiculo",
                  "valor_pago_historico", "referral_pendente", "tempo_cotacao_emissao"]],
        on="numeroApolice", how="left",
    )
    projecao["perfil_numero"] = projecao["cluster"].map(mapa_perfil)
    projecao["nome_perfil"] = projecao["perfil_numero"].map(NOME_PERFIL)

    if view == "Segmentacao":
        render_view_segmentacao(
            projecao, variancia_pct, kpis_cluster, gerais, significancia, normalizada, validacao_k
        )
    elif view == "Validacao de Hardware":
        render_view_validacao_hardware(conduta, metricas)
    elif view == "Coligacao Conceitual":
        render_view_coligacao(original, mapa_perfil, conduta)

    with st.expander("Exportar dados"):
        render_view_exportar()

    st.markdown(
        '<div style="text-align:center;color:#65596f;font-size:11px;margin-top:24px;">'
        "Erika Oliveira Silva &middot; Centro Universitario Senac Santo Amaro &middot; "
        "Validacao: Silhouette, Davies-Bouldin e Calinski-Harabasz</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
