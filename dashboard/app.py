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

# Rotulos comportamentais do UAH-DriveSet (src/validacao_hardware.py), usados
# na aba de telemetria. Reaproveita as 3 cores categoricas ja validadas acima
# (mesmos hex de CORES_PERFIL) para manter a identidade visual do dashboard;
# "desconhecido" e um cinza neutro (nao e uma 4a categoria de dados, apenas
# trajetos sem rotulo no caminho) e nao entrou na validacao de pares CVD.
COMPORTAMENTO_LABEL = {
    "normal": "Normal",
    "agressiva": "Agressiva",
    "sonolenta": "Sonolenta",
    "desconhecido": "Nao identificado",
}
CORES_COMPORTAMENTO = {
    "normal": "#2a78d6",
    "agressiva": "#e87ba4",
    "sonolenta": "#008300",
    "desconhecido": "#8a7f97",
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
def carregar_telemetria():
    """Resumo por trajeto do UAH-DriveSet, gerado por
    src/validacao_hardware.py: evidencia comportamental de referencia para a
    camada de hardware enquanto o dispositivo ESP32 proprio nao coletou
    viagens reais (mesmo esquema de evento: contagem, intensidade,
    geolocalizacao)."""
    caminho = DADOS_DIR / "validacao_hardware_uah_driveset.csv"
    return pd.read_csv(caminho) if caminho.exists() else None


@st.cache_data
def carregar_conduta_combinada():
    """Resumo por rotulo de evento de dois datasets publicos de conducao
    (Yuksel & Atmaca, 2020; Jair Jr., 2016), classificados pelo mesmo
    limiar de deteccao do ESP32 em src/validacao_hardware.py -- segunda
    fonte de validacao, independente do UAH-DriveSet."""
    caminho = DADOS_DIR / "validacao_hardware_conduta_combinada_resumo.csv"
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<h2 style="margin-top:0;">Dashboard de <span style="color:#e87ba4;">Risco</span></h2>'
            '<div style="font-size:12px;margin-bottom:14px;">RCT Transportador</div>',
            unsafe_allow_html=True,
        )

        if "view" not in st.session_state:
            st.session_state["view"] = "Visao geral"

        opcoes = [
            "Visao geral", "Clusters (PCA)", "Validacao e insights",
            "Telemetria (UAH-DriveSet)", "Exportar dados",
        ]
        for opcao in opcoes:
            ativo = st.session_state["view"] == opcao
            if st.button(
                opcao, key=f"nav_{opcao.replace(' ', '_')}",
                width="stretch", type=("primary" if ativo else "secondary"),
            ):
                st.session_state["view"] = opcao

        st.markdown(
            '<div class="side-badge"><b>Etapa 4 da metodologia.</b> '
            'Segmentacao via K-Means (k=3) sobre 19 variaveis derivadas (Quadro 2). '
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


def render_view_validacao(normalizada, validacao_k, significancia):
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


def render_kpis_telemetria(telemetria: pd.DataFrame):
    colunas = st.columns(4)
    itens = [
        ("Trajetos analisados", str(len(telemetria))),
        ("Motoristas distintos", str(telemetria["motorista"].nunique())),
        ("Eventos detectados", str(int(telemetria["n_eventos"].sum()))),
        ("Eventos/min (media)", f"{telemetria['eventos_por_min'].mean():.2f}"),
    ]
    for coluna, (rotulo, valor) in zip(colunas, itens):
        coluna.metric(rotulo, valor)


def _ordem_e_cores_comportamento(telemetria: pd.DataFrame) -> tuple[list, list]:
    presentes = [c for c in CORES_COMPORTAMENTO if c in telemetria["comportamento"].unique()]
    ordem = [COMPORTAMENTO_LABEL[c] for c in presentes]
    cores = [CORES_COMPORTAMENTO[c] for c in presentes]
    return ordem, cores


def render_bar_eventos_comportamento(telemetria: pd.DataFrame):
    ordem, cores = _ordem_e_cores_comportamento(telemetria)
    agregado = telemetria.groupby("comportamento")["eventos_por_min"].mean().reset_index()
    agregado["comportamento_label"] = agregado["comportamento"].map(COMPORTAMENTO_LABEL)

    grafico = alt.Chart(agregado).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("comportamento_label:N", title=None, sort=ordem, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("eventos_por_min:Q", title="Eventos por minuto (media)"),
        color=alt.Color(
            "comportamento_label:N", title="Rotulo comportamental",
            scale=alt.Scale(domain=ordem, range=cores),
        ),
        tooltip=["comportamento_label", alt.Tooltip("eventos_por_min:Q", format=".2f")],
    ).properties(height=280)
    st.altair_chart(grafico, width="stretch")


def render_box_magnitude(telemetria: pd.DataFrame):
    ordem, cores = _ordem_e_cores_comportamento(telemetria)
    dados = telemetria.copy()
    dados["comportamento_label"] = dados["comportamento"].map(COMPORTAMENTO_LABEL)

    grafico = alt.Chart(dados).mark_boxplot(size=36).encode(
        x=alt.X("comportamento_label:N", title=None, sort=ordem, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("magnitude_maxima_ms2:Q", title="Magnitude maxima do trajeto (m/s^2)"),
        color=alt.Color(
            "comportamento_label:N",
            scale=alt.Scale(domain=ordem, range=cores), legend=None,
        ),
    ).properties(height=280)
    st.altair_chart(grafico, width="stretch")


def render_tabela_telemetria(telemetria: pd.DataFrame):
    tabela = telemetria[[
        "motorista", "comportamento", "duracao_min", "n_eventos",
        "eventos_por_min", "magnitude_media_ms2", "magnitude_maxima_ms2",
    ]].copy()
    tabela["comportamento"] = tabela["comportamento"].map(COMPORTAMENTO_LABEL)
    tabela["duracao_min"] = tabela["duracao_min"].map(lambda v: f"{v:.1f}")
    tabela["eventos_por_min"] = tabela["eventos_por_min"].map(lambda v: f"{v:.2f}")
    tabela["magnitude_media_ms2"] = tabela["magnitude_media_ms2"].map(lambda v: f"{v:.2f}")
    tabela["magnitude_maxima_ms2"] = tabela["magnitude_maxima_ms2"].map(lambda v: f"{v:.2f}")
    tabela.columns = [
        "Motorista", "Comportamento", "Duracao (min)", "Eventos",
        "Eventos/min", "Magnitude media (m/s^2)", "Magnitude maxima (m/s^2)",
    ]
    st.dataframe(tabela, width="stretch", hide_index=True)


def render_kpis_conduta(conduta: pd.DataFrame):
    total_janelas = int(conduta["n_janelas"].sum())
    total_detectados = int(conduta["n_detectados"].sum())
    taxa_geral = total_detectados / total_janelas if total_janelas else float("nan")
    n_motoristas = conduta.loc[conduta["motorista"] != "desconhecido", "motorista"].nunique()
    colunas = st.columns(4)
    itens = [
        ("Janelas analisadas", str(total_janelas)),
        ("Fontes de dados", str(conduta["fonte"].nunique())),
        ("Motoristas identificados", str(n_motoristas)),
        ("Taxa de deteccao geral", f"{taxa_geral:.0%}"),
    ]
    for coluna, (rotulo, valor) in zip(colunas, itens):
        coluna.metric(rotulo, valor)


def render_bar_taxa_deteccao_conduta(conduta: pd.DataFrame):
    """Taxa de deteccao por rotulo de evento, agregada entre motoristas
    (a mesma janela do arquivo tem uma linha por motorista dentro de cada
    fonte -- ver render_tabela_conduta para a granularidade por motorista)."""
    agregado = conduta.groupby(["fonte", "rotulo_evento"], as_index=False).agg(
        n_janelas=("n_janelas", "sum"), n_detectados=("n_detectados", "sum"),
    )
    agregado["taxa_deteccao"] = agregado["n_detectados"] / agregado["n_janelas"]
    agregado["comportamento"] = agregado["rotulo_evento"].map(
        lambda r: "normal" if r == "Non-aggressive event" else "agressiva"
    )
    agregado["comportamento_label"] = agregado["comportamento"].map(COMPORTAMENTO_LABEL)
    presentes = [c for c in ("agressiva", "normal") if c in agregado["comportamento"].unique()]
    ordem = [COMPORTAMENTO_LABEL[c] for c in presentes]
    cores = [CORES_COMPORTAMENTO[c] for c in presentes]

    grafico = alt.Chart(agregado).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("rotulo_evento:N", title=None, sort="-y", axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("taxa_deteccao:Q", title="Taxa de deteccao", axis=alt.Axis(format="%")),
        color=alt.Color(
            "comportamento_label:N", title="Rotulo comportamental",
            scale=alt.Scale(domain=ordem, range=cores),
        ),
        tooltip=["fonte", "rotulo_evento", alt.Tooltip("taxa_deteccao:Q", format=".0%"), "n_janelas"],
    ).properties(height=280)
    st.altair_chart(grafico, width="stretch")


def render_bar_taxa_deteccao_motorista(conduta: pd.DataFrame):
    dados = conduta[conduta["motorista"] != "desconhecido"].groupby(
        ["fonte", "motorista"], as_index=False
    ).agg(n_janelas=("n_janelas", "sum"), n_detectados=("n_detectados", "sum"))
    dados["taxa_deteccao"] = dados["n_detectados"] / dados["n_janelas"]

    grafico = alt.Chart(dados).mark_bar(
        cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color=CORES_COMPORTAMENTO["agressiva"],
    ).encode(
        x=alt.X("motorista:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("taxa_deteccao:Q", title="Taxa de deteccao", axis=alt.Axis(format="%")),
        tooltip=["fonte", "motorista", alt.Tooltip("taxa_deteccao:Q", format=".0%"), "n_janelas"],
    ).properties(height=280)
    st.altair_chart(grafico, width="stretch")


def render_tabela_conduta(conduta: pd.DataFrame):
    tabela = conduta.copy()
    tabela["taxa_deteccao"] = tabela["taxa_deteccao"].map(lambda v: f"{v:.0%}")
    tabela["magnitude_media_ms2"] = tabela["magnitude_media_ms2"].map(lambda v: f"{v:.2f}")
    tabela["magnitude_maxima_ms2"] = tabela["magnitude_maxima_ms2"].map(lambda v: f"{v:.2f}")
    tabela = tabela[[
        "fonte", "motorista", "rotulo_evento", "n_janelas", "n_detectados",
        "taxa_deteccao", "magnitude_media_ms2", "magnitude_maxima_ms2",
    ]]
    tabela.columns = [
        "Fonte", "Motorista", "Rotulo do evento", "Janelas", "Detectados",
        "Taxa de deteccao", "Magnitude media (m/s^2)", "Magnitude maxima (m/s^2)",
    ]
    st.dataframe(tabela, width="stretch", hide_index=True)


def render_view_telemetria(telemetria, conduta=None):
    st.caption(
        "Validacao dos limiares de deteccao de eventos do dispositivo (Secao B.3/B.5) "
        "sobre trajetos reais do UAH-DriveSet (ROMERA; BERGASA; ARROYO, 2016), com "
        "rotulo comportamental conhecido (normal/agressiva/sonolenta). Funciona como "
        "evidencia comportamental de referencia enquanto o dispositivo ESP32 proprio "
        "ainda nao coletou viagens reais: quando isso acontecer, os eventos resumidos "
        "do dispositivo (contagem, intensidade, geolocalizacao) passam a complementar "
        "esta mesma visao, sob o mesmo esquema de evento."
    )

    if telemetria is None or telemetria.empty:
        st.info(
            "Nenhum resumo de telemetria encontrado em data/processed/. Baixe o "
            "UAH-DriveSet e rode:\n\n"
            "`python src/validacao_hardware.py --dataset-dir data/raw/uah-driveset`"
        )
    else:
        render_kpis_telemetria(telemetria)

        col_bar, col_box = st.columns(2)
        with col_bar:
            with st.container(border=True):
                st.markdown("##### Eventos por minuto, por rotulo comportamental")
                st.caption("Media entre os trajetos de cada rotulo conhecido")
                render_bar_eventos_comportamento(telemetria)
        with col_box:
            with st.container(border=True):
                st.markdown("##### Magnitude maxima por trajeto, por rotulo comportamental")
                st.caption("Distribuicao entre os trajetos de cada rotulo conhecido")
                render_box_magnitude(telemetria)

        with st.container(border=True):
            st.markdown("##### Trajetos analisados")
            render_tabela_telemetria(telemetria)

    st.divider()
    st.markdown("#### Validacao cruzada: datasets publicos de conducao")
    st.caption(
        "Segunda fonte de validacao, independente do UAH-DriveSet: janelas ja "
        "rotuladas por manobra dos datasets de Yuksel & Atmaca (2020) e Jair Jr. "
        "(2016), classificadas pelo mesmo limiar de magnitude do ESP32. Apenas o "
        "Jair Jr. (2016) tem uma classe 'nao agressiva' de controle -- e nele que "
        "o qui-quadrado (impresso ao rodar src/validacao_hardware.py) confirma que "
        "o limiar discrimina evento agressivo de evento normal."
    )
    if conduta is None or conduta.empty:
        st.info(
            "Nenhum resumo encontrado em data/processed/. Rode:\n\n"
            "`python src/validacao_hardware.py --conduta-csv "
            "data/raw/combined_normalized_driver_conduct.csv`"
        )
        return

    render_kpis_conduta(conduta)

    tem_motorista = (conduta["motorista"] != "desconhecido").any()
    if tem_motorista:
        col_evento, col_motorista = st.columns(2)
    else:
        col_evento = st.container()

    with col_evento:
        with st.container(border=True):
            st.markdown("##### Taxa de deteccao por rotulo de evento")
            render_bar_taxa_deteccao_conduta(conduta)
    if tem_motorista:
        with col_motorista:
            with st.container(border=True):
                st.markdown("##### Taxa de deteccao por motorista")
                st.caption("Somente Jair Jr. (2016) identifica o motorista por trajeto")
                render_bar_taxa_deteccao_motorista(conduta)

    with st.container(border=True):
        st.markdown("##### Resumo por motorista e rotulo de evento")
        render_tabela_conduta(conduta)


def render_view_exportar():
    arquivos = {
        "Matriz original com rotulos de cluster": "matriz_original_clusters.csv",
        "Matriz normalizada com rotulos de cluster": "matriz_normalizada_clusters.csv",
        "Estatisticas descritivas por cluster": "perfis_estatisticas_descritivas.csv",
        "Testes de significancia por variavel": "perfis_testes_significancia.csv",
        "Curva de validacao de k (cotovelo/silhouette)": "validacao_k.csv",
        "Telemetria por trajeto (UAH-DriveSet)": "validacao_hardware_uah_driveset.csv",
        "Validacao cruzada por janela (datasets publicos de conducao)": "validacao_hardware_conduta_combinada.csv",
        "Resumo da validacao cruzada por rotulo de evento": "validacao_hardware_conduta_combinada_resumo.csv",
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
            "Dashboard de visualizacao e validacao dos resultados da segmentacao por K-Means "
            "(Etapa 4 da metodologia)."
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

    projecao, variancia_pct = projetar_pca(normalizada)
    projecao = projecao.merge(
        original[["numeroApolice", "classe_risco", "premio_por_veiculo",
                  "valor_pago_historico", "referral_pendente", "tempo_cotacao_emissao"]],
        on="numeroApolice", how="left",
    )
    projecao["perfil_numero"] = projecao["cluster"].map(mapa_perfil)
    projecao["nome_perfil"] = projecao["perfil_numero"].map(NOME_PERFIL)

    if view not in ("Exportar dados", "Telemetria (UAH-DriveSet)"):
        opcoes_segmento = [CARTEIRA_COMPLETA] + kpis_cluster["nome_perfil"].tolist()
        segmento_selecionado = st.pills(
            "Filtrar por perfil", opcoes_segmento, default=CARTEIRA_COMPLETA,
            key="segmento_pills",
        )
        if not segmento_selecionado:
            segmento_selecionado = CARTEIRA_COMPLETA

    if view == "Visao geral":
        render_view_visao_geral(projecao, kpis_cluster, gerais, significancia, segmento_selecionado)
    elif view == "Clusters (PCA)":
        render_view_clusters(projecao, variancia_pct)
    elif view == "Validacao e insights":
        render_view_validacao(normalizada, validacao_k, significancia)
    elif view == "Telemetria (UAH-DriveSet)":
        render_view_telemetria(carregar_telemetria(), carregar_conduta_combinada())
    elif view == "Exportar dados":
        render_view_exportar()

    st.markdown(
        '<div style="text-align:center;color:#65596f;font-size:11px;margin-top:24px;">'
        "Erika Oliveira Silva &middot; Centro Universitario Senac Santo Amaro &middot; "
        "Validacao: Silhouette, Davies-Bouldin e Calinski-Harabasz</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
