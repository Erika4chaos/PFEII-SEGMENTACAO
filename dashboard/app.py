"""
Etapa 4 (Secao 3.5.4): dashboard Streamlit com tres abas -- Segmentacao
(projecao PCA 2D dos clusters, tabela de perfis, KPIs e indices de validacao
tecnica), Validacao de Hardware (discriminacao do limiar de ~6 m/s^2 do
firmware contra o dataset publico de conducao de Ferreira Jr. et al., 2017)
e Coligacao Conceitual (passeio ilustrativo perfil x evento de hardware,
sem juncao real de dados).

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


@st.cache_data
def carregar_conduta_confound():
    """Contagem de janelas por categoria harmonizada x sessao de gravacao
    (GroupID), ja calculada por src/validacao_hardware.py -- evidencia o
    confundimento categoria/sessao (ex.: aceleracao e frenagem so aparecem
    em uma das tres sessoes) sem o dashboard recalcular nada."""
    caminho = DADOS_DIR / "driver_conduct_confound.csv"
    return pd.read_csv(caminho) if caminho.exists() else None


@st.cache_data
def carregar_delta_v():
    """Variacao de velocidade (delta-v) por evento de aceleracao/frenagem,
    CALCULADA por integracao da aceleracao real (numero de amostras da
    janela recuperado do proprio dado, baseline de sessao a partir de
    janelas reais NON_AGGRESSIVE) por src/calcular_delta_v_conduta.py --
    a unica constante nao medida neste arquivo e a taxa de amostragem
    citada (ver docstring daquele script). Nao ha variavel simulada por
    regra de negocio."""
    caminho = DADOS_DIR / "conduta_delta_v.csv"
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


def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<h2 style="margin-top:0;">Dashboard de <span style="color:#e87ba4;">Risco</span></h2>'
            '<div style="font-size:12px;margin-bottom:14px;">RCT Transportador</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="side-badge"><b>Etapa 4 da metodologia.</b> '
            'Segmentacao via K-Means (k=3) sobre 19 variaveis derivadas (Quadro 2), '
            'complementada por validacao independente da camada de hardware (Parte B). '
            'Correlacao espacial com criminalidade/vulnerabilidade e alertas de '
            'telemetria por apolice ficam registrados como extensao futura (Secao 3.8).'
            '</div>',
            unsafe_allow_html=True,
        )


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


def render_confundimento_sessao(confound: pd.DataFrame):
    st.markdown("##### Confundimento entre categoria de evento e sessao de gravacao (GroupID)")
    tabela = confound.pivot(index="EventCategory", columns="GroupID", values="n").fillna(0).astype(int)
    tabela.index = tabela.index.map(lambda c: CATEGORIA_LABEL.get(c, c))
    st.dataframe(tabela, width="stretch")

    so_uma_sessao = confound[confound["n_sessions_com_categoria"] == 1]
    categorias_confundidas = sorted(
        {CATEGORIA_LABEL.get(c, c) for c in so_uma_sessao["EventCategory"].unique()}
    )
    if categorias_confundidas:
        st.warning(
            "**Nota de confundimento (nao ocultada):** neste dataset, "
            f"{', '.join(categorias_confundidas)} aparecem em uma unica sessao de "
            "gravacao (GroupID) cada -- evento e sessao estao parcialmente "
            "confundidos. Nenhuma conclusao por motorista deve ser tirada apenas "
            "desta validacao."
        )


def render_limitacoes_hardware():
    st.markdown("##### Limitacoes desta validacao (Secao 3.8)")
    st.warning(
        "- **Sem validacao de sonolencia**: a fonte de substituicao nao possui rotulo "
        "de conducao sonolenta; esse eixo foi removido do escopo, nao apenas nao "
        "implementado.\n"
        "- **Sem dado de GPS/velocidade, em nenhum momento**: confirmado contra o "
        "repositorio do dataset e o artigo de origem -- apenas acelerometro, "
        "aceleracao linear, giroscopio e magnetometro do smartphone foram registrados; "
        "o velocimetro do veiculo aparece somente no video de referencia usado para "
        "rotulagem manual, nunca como campo de dado. O criterio de excesso de "
        "velocidade do firmware nao pode ser validado com estes dados.\n"
        "- **Mudanca de faixa fora de escopo**: um no de acelerometro + GPS unico nao "
        "e capaz de detectar mudanca de faixa (falta dado de direcao/esterco), entao "
        "essas linhas sao excluidas da avaliacao, nao descartadas silenciosamente.\n"
        "- **Amostra pequena**: n=55 janelas no total, ate 6 na menor categoria -- "
        "leia todo resultado acima como verificacao de viabilidade, nao como validacao "
        "estatisticamente dimensionada.\n"
        "- **Confundimento categoria/sessao**: ver nota acima -- nem toda sessao "
        "(GroupID) contem toda categoria de evento.\n"
        "- **Proxy de pico, nao amostra bruta**: como a fonte so fornece estatisticas "
        "por janela (nao o sinal bruto por amostra), o pico dinamico usado aqui e "
        "aproximado pelo desvio Max/Min em relacao a Media da janela, nao pela deteccao "
        "de pico instantaneo que o firmware faz sobre o sinal continuo."
    )


def render_delta_v(delta_v: pd.DataFrame):
    st.markdown("##### Variacao de velocidade (delta-v) por evento")
    st.caption(
        "Calculado por integracao da propria aceleracao do evento -- nao simulado. "
        "O numero de amostras da janela e a media da aceleracao dinamica (com a "
        "gravidade/offset de montagem removidos via janelas reais de linha de base "
        "da mesma sessao) vem do dado medido; a duracao usa uma taxa de amostragem "
        "citada da literatura (nao medida neste arquivo), com a sensibilidade a essa "
        "taxa reportada ao rodar `src/calcular_delta_v_conduta.py`. Isto e uma "
        "variacao de velocidade, nao uma velocidade absoluta -- a fonte nao tem "
        "canal de velocidade/GPS para fornecer um ponto de partida."
    )

    dados = delta_v.copy()
    dados["CategoriaLabel"] = dados["EventCategory"].map(CATEGORIA_LABEL)
    dados["EventoId"] = (
        dados["CategoriaLabel"] + " #" + (dados.groupby("EventCategory").cumcount() + 1).astype(str)
    )
    ordem_eventos = dados.sort_values(["EventCategory", "EventoId"])["EventoId"].tolist()

    grafico = alt.Chart(dados).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("EventoId:N", sort=ordem_eventos, title=None),
        x=alt.X("delta_v_kmh:Q", title="Delta-v (km/h)"),
        color=alt.Color(
            "CategoriaLabel:N", title="Categoria",
            scale=alt.Scale(domain=["Aceleracao", "Frenagem"], range=[CORES_PERFIL[3], CORES_PERFIL[1]]),
        ),
        tooltip=["EventoId", "n_amostras", alt.Tooltip("duracao_evento_s:Q", format=".2f", title="Duracao (s)"),
                 alt.Tooltip("delta_v_kmh:Q", format=".1f")],
    ).properties(height=320)
    st.altair_chart(grafico, width="stretch")

    tabela = dados[[
        "EventoId", "n_amostras", "duracao_evento_s",
        "aceleracao_dinamica_media_mps2", "delta_v_kmh",
    ]].copy()
    tabela.columns = [
        "Evento", "Amostras (real)", "Duracao (s)",
        "Aceleracao dinamica media (m/s^2, real)", "Delta-v (km/h, calculado)",
    ]
    st.dataframe(
        tabela, width="stretch", hide_index=True,
        column_config={
            "Duracao (s)": st.column_config.NumberColumn(format="%.2f"),
            "Aceleracao dinamica media (m/s^2, real)": st.column_config.NumberColumn(format="%.2f"),
            "Delta-v (km/h, calculado)": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def render_view_validacao_hardware(conduta, metricas, confound, delta_v):
    if conduta is None or metricas is None:
        st.error(
            "Nenhum resultado de validacao de hardware encontrado em data/processed/. "
            "Execute `python src/validacao_hardware.py` antes de abrir esta aba."
        )
        return

    render_metodologia_hardware()

    with st.container(border=True):
        render_confusao(metricas)

    with st.container(border=True):
        render_recall_por_categoria(metricas)

    if confound is not None and not confound.empty:
        with st.container(border=True):
            render_confundimento_sessao(confound)

    render_limitacoes_hardware()

    if delta_v is not None and not delta_v.empty:
        st.divider()
        with st.container(border=True):
            render_delta_v(delta_v)


# ---------------------------------------------------------------------------
# Aba 3 -- Coligacao Conceitual (Part A.7 / C) -- passeio ilustrativo, sem
# juncao real entre dados sinteticos de apolice e dados publicos de conducao.
# ---------------------------------------------------------------------------

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

    if conduta is None or conduta.empty:
        st.info("Execute src/validacao_hardware.py para habilitar o exemplo ilustrativo.")
        return

    original = original.copy()
    original["perfil_numero"] = original["cluster"].map(mapa_perfil)
    candidatos_perfil1 = original[original["perfil_numero"] == 1]
    if candidatos_perfil1.empty:
        st.info("Nenhuma apolice do Perfil 1 encontrada na base atual.")
        return
    apolice = candidatos_perfil1.sort_values(
        ["valor_pago_historico", "classe_risco"], ascending=False
    ).iloc[0]

    eventos_alvo = conduta[
        (conduta["ValidationRole"] == "harsh")
        & (conduta["EventCategory"].isin(["TURN", "BRAKING"]))
        & (conduta["HarshPredicted_6mps2"] == 1)
    ]
    if eventos_alvo.empty:
        eventos_alvo = conduta[conduta["ValidationRole"] == "harsh"]
    evento = eventos_alvo.sort_values("PeakDynamicAccel_mps2", ascending=False).iloc[0]

    st.markdown("##### Exemplo ilustrativo")
    col_apolice, col_evento = st.columns(2)
    with col_apolice:
        with st.container(border=True):
            st.markdown(f"**Apolice (dado sintetico) · {NOME_PERFIL[1]}**")
            st.caption("Cadastro/cotacao -- lado da segmentacao (Parte A)")
            st.markdown(
                f"- Numero: `{apolice['numeroApolice']}`\n"
                f"- Frota total: {apolice['total_veiculos']:.0f} veiculos "
                f"({apolice['pct_autonomos']:.0%} autonomos/terceiros)\n"
                f"- Classe de risco: {apolice['classe_risco']:.0f}/5\n"
                f"- Motorista licenciado: {'sim' if apolice['motorista_licenciado'] else 'nao'}\n"
                f"- Custo historico declarado: {_fmt_brl(apolice['valor_pago_historico'])}"
            )
    with col_evento:
        with st.container(border=True):
            fonte_label = FONTE_LABEL.get(evento["SourceDataset"], evento["SourceDataset"])
            st.markdown(f"**Evento de conducao (dataset publico) · {fonte_label}**")
            st.caption("Janela rotulada -- lado da validacao de hardware (Parte B)")
            st.markdown(
                f"- Categoria harmonizada: {CATEGORIA_LABEL.get(evento['EventCategory'], evento['EventCategory'])}\n"
                f"- Rotulo original: {evento['EventLabel']}\n"
                f"- Magnitude dinamica de pico: {evento['PeakDynamicAccel_mps2']:.2f} m/s^2\n"
                f"- Acima do limiar (~6 m/s^2): {'sim' if evento['HarshPredicted_6mps2'] else 'nao'}"
            )

    st.caption(
        "Narrativa ilustrativa: se o dispositivo ESP32 desta frota (Perfil 1, ja "
        "sinalizada como alto risco pela segmentacao historica) registrasse um evento "
        "de conducao com esta magnitude, esse sumario reforcaria/confirmaria a "
        "classificacao existente -- e o inverso, uma frota com poucos eventos "
        "assim, poderia reduzi-la ao longo do tempo (Secao 3.4.1). Nenhum coeficiente "
        "de correlacao ou KPI estatistico e calculado aqui, pois nenhuma relacao real "
        "entre os dois conjuntos de dados existe nesta PoC."
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
        "Delta-v calculado por evento de aceleracao/frenagem": "conduta_delta_v.csv",
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
    render_sidebar()

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
    confound = carregar_conduta_confound()
    delta_v = carregar_delta_v()

    projecao, variancia_pct = projetar_pca(normalizada)
    projecao = projecao.merge(
        original[["numeroApolice", "classe_risco", "premio_por_veiculo",
                  "valor_pago_historico", "referral_pendente", "tempo_cotacao_emissao"]],
        on="numeroApolice", how="left",
    )
    projecao["perfil_numero"] = projecao["cluster"].map(mapa_perfil)
    projecao["nome_perfil"] = projecao["perfil_numero"].map(NOME_PERFIL)

    aba_segmentacao, aba_hardware, aba_coligacao = st.tabs(
        ["Segmentacao", "Validacao de Hardware", "Coligacao Conceitual"]
    )
    with aba_segmentacao:
        render_view_segmentacao(
            projecao, variancia_pct, kpis_cluster, gerais, significancia, normalizada, validacao_k
        )
    with aba_hardware:
        render_view_validacao_hardware(conduta, metricas, confound, delta_v)
    with aba_coligacao:
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
