"""
Etapa 4 (Secao 3.5.4): dashboard Streamlit com tres secoes, navegadas por
botoes na barra lateral -- Segmentacao (projecao PCA 2D dos clusters,
tabela de perfis, KPIs e indices de validacao tecnica), Validacao de
Hardware (discriminacao do limiar de ~6 m/s^2 do firmware, aplicado ao
UAH-DriveSet, ROMERA; BERGASA; ARROYO, 2016) e Coligacao Conceitual
(passeio ilustrativo perfil x trajeto de hardware, sem juncao real de
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
# src/ para os modulos de pipeline; a raiz para importar `dashboard` como
# pacote (a Aba 2 vive em dashboard/tab_validacao_hardware.py).
sys.path.append(str(RAIZ / "src"))
sys.path.append(str(RAIZ))
from preprocessamento import COLUNAS_19  # noqa: E402
from validacao_hardware import THRESH_MAG_MS2  # noqa: E402
from dashboard.tab_validacao_hardware import render, carregar  # noqa: E402

DADOS_DIR = RAIZ / "data" / "processed"

TIPO_SINISTRO_LABEL = {0: "nenhum identificado", 1: "tombamento", 2: "incendio", 3: "terceiros"}

CARTEIRA_COMPLETA = "Carteira completa"

NOME_PERFIL = {
    1: "Perfil 1 - Frota de Alto Risco Operacional",
    2: "Perfil 2 - Segurado de Alta Cobertura e Baixo Custo Relativo",
    3: "Perfil 3 - Cotacao em Referral ou Conversao Tardia",
}

# Paleta categorica (roxo/azul/verde) validada com scripts/validate_palette.py
# da skill dataviz para 3 series em grafico de dispersao (checagem --pairs all):
# todos os checks passam, pior par #2a78d6 x #8e2f9e com dE 9,6 sob deuteranopia
# e 20,8 na visao normal. O roxo substituiu o rosa #e87ba4 anterior, que passava
# na separacao mas reprovava o contraste contra o fundo claro (2,62:1). Roxo mais
# claro nao serve: qualquer violeta que puxe para o azul cai para dE 4-7 contra o
# azul do Perfil 2 -- foi por isso que este tom puxa para o magenta.
# Cor atribuida por PERFIL (numero de negocio), nao pelo indice arbitrario que o
# K-Means da ao cluster, para que a identidade visual nao mude entre execucoes.
CORES_PERFIL = {1: "#8e2f9e", 2: "#2a78d6", 3: "#008300"}
COR_LINHA_DESTAQUE = "#6d3bc4"

# Citacao academica da fonte de validacao de hardware (Part B.5 do escopo
# tecnico). O UAH-DriveSet e a fonte prevista no plano original; um
# dataset de conducao ja segmentado em janelas (Ferreira Jr. et al., 2017)
# foi usado como substituto temporario enquanto o host/espelhos do
# UAH-DriveSet estiveram indisponiveis -- ver historico em
# src/validacao_hardware.py. Nao reintroduzir a substituicao sem que o
# UAH-DriveSet volte a ficar inacessivel.
FONTE_CITACAO = "UAH-DriveSet (Romera; Bergasa; Arroyo, 2016)"

COMPORTAMENTO_LABEL = {
    "normal": "Normal", "agressiva": "Agressiva", "sonolenta": "Sonolenta",
    "desconhecido": "Desconhecido",
}
# As cores por rotulo comportamental vivem em dashboard/charts_validacao.py
# (ESTILOS/CORES), junto das figuras que as usam. COMPORTAMENTO_LABEL fica aqui
# porque a Aba 3 ainda rotula o trajeto ilustrativo com ele.


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
def carregar_validacao_uah():
    """Le o resumo por trajeto ja calculado por src/validacao_hardware.py
    (deteccao de eventos pelo limiar do firmware + rotulo comportamental
    extraido do nome da pasta) -- o dashboard nunca reimplementa a
    deteccao nem a extracao de rotulo, apenas consome o CSV que o script
    produz (ver Part B.5/A.7 do escopo tecnico)."""
    caminho = DADOS_DIR / "validacao_hardware_uah_driveset.csv"
    if not caminho.exists():
        return None
    df = pd.read_csv(caminho)
    df["ComportamentoLabel"] = df["comportamento"].map(COMPORTAMENTO_LABEL).fillna(df["comportamento"])
    return df


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
        :root {
            --navy: #44355b; --navy2: #5a4a75; --navy3: #2e2440;
            --accent: #7b3fbf; --accent-soft: #ede4f9;
            /* Violeta claro para uso SOBRE a barra lateral escura: o --accent
               e escuro demais contra o roxo do painel e some. */
            --accent-claro: #d9c2f7;
            --bg: #f7f4fc; --card: #ffffff;
            --text: #44355b; --muted: #65596f; --line: #e3dcea;
        }
        [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            background-color: var(--bg);
        }
        h1 {
            color: var(--text);
            border-bottom: 1px solid var(--line);
            padding-bottom: 0.4rem;
            font-weight: 700;
        }
        h2 {
            color: var(--text);
            margin-top: 1.2rem;
            font-weight: 700;
        }
        h3 { color: var(--text); }
        [data-testid="stMetric"] {
            background-color: var(--card);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 14px 16px 10px 16px;
        }
        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-size: 11.5px;
            font-weight: 600;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }
        [data-testid="stMetricValue"] { color: var(--text); font-size: 1.5rem; }

        [data-testid="stSidebar"] {
            background-color: var(--navy3);
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
            background-color: var(--navy2);
            color: #ffffff;
        }
        .side-badge {
            background: var(--navy2);
            border-radius: 10px;
            padding: 12px;
            font-size: 12px;
            line-height: 1.5;
            color: #cfc4dc;
            margin-top: 12px;
        }
        .side-badge b { color: var(--accent-claro); }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--line) !important;
            border-radius: 12px !important;
        }

        .pill-badge {
            display: inline-block;
            background: var(--accent-soft);
            color: #6a3a96;
            font-size: 12px;
            font-weight: 600;
            padding: 5px 12px;
            border-radius: 999px;
        }
        .insight-item { padding: 9px 0; border-bottom: 1px solid var(--line); font-size: 13px; }
        .insight-item:last-child { border-bottom: none; }
        .insight-dot {
            display: inline-block; width: 8px; height: 8px; border-radius: 50%;
            background: var(--accent); margin-right: 8px; margin-top: 5px;
            flex-shrink: 0;
        }
        .insight-title { font-weight: 700; color: var(--text); font-size: 12.5px; }
        .insight-caption { color: var(--muted); font-size: 11.5px; line-height: 1.45; }

        .tag-pill {
            display: inline-block; padding: 3px 10px; border-radius: 999px;
            font-size: 12px; font-weight: 600; color: #ffffff;
        }
        .no-join-banner {
            background: #fff3cd; border: 1px solid #e8b83c; border-radius: 10px;
            padding: 14px 16px; color: #7a5a00; font-size: 13px; line-height: 1.6;
        }

        /* Cartoes de KPI customizados (render_linha_kpis) */
        .kpi-card {
            background: var(--card); border: 1px solid var(--line); border-radius: 12px;
            padding: 14px 16px;
        }
        .kpi-card .lbl {
            font-size: 11.5px; color: var(--muted); font-weight: 600;
            letter-spacing: 0.02em; text-transform: uppercase;
        }
        .kpi-card .val { font-size: 24px; font-weight: 700; margin-top: 5px; color: var(--text); }
        .kpi-card .val small { font-size: 12px; color: var(--muted); font-weight: 500; }

        /* Tabela de perfis customizada (render_tabela_perfis) */
        .tabela-perfis { width: 100%; border-collapse: collapse; font-size: 12.5px; }
        .tabela-perfis th {
            text-align: left; color: var(--muted); font-size: 11px; text-transform: uppercase;
            letter-spacing: 0.03em; padding: 8px 10px; border-bottom: 1.5px solid var(--line);
        }
        .tabela-perfis td { padding: 10px; border-bottom: 1px solid var(--line); color: var(--text); }
        .tabela-perfis tr:last-child td { border-bottom: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )


NAV_OPCOES = ["Segmentacao", "Validacao de Hardware", "Coligacao Conceitual"]


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<h2 style="margin-top:0;">Dashboard de <span style="color:#b07be0;">Risco</span></h2>'
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
    itens = [
        ("Apolices", str(int(dados["n_apolices"])), ""),
        ("Premio medio / veic.", _fmt_brl_compacto(dados["premio_por_veiculo_medio"]), "por veiculo/ano"),
        ("LMI medio / veic.", _fmt_brl_compacto(dados["lmi_por_veiculo_medio"]), "por veiculo"),
        ("Custo hist. medio", _fmt_brl_compacto(dados["valor_pago_historico_medio"]), "por sinistro declarado"),
        ("Motorista licenciado", f"{dados['pct_motorista_licenciado']:.0%}", ""),
        ("Taxa de referral", f"{dados['taxa_referral']:.0%}", "aprovacao especial"),
    ]
    colunas = st.columns(len(itens))
    for coluna, (rotulo, valor, secundario) in zip(colunas, itens):
        with coluna:
            st.markdown(
                f'<div class="kpi-card"><div class="lbl">{rotulo}</div>'
                f'<div class="val">{valor}'
                f'{f" <small>{secundario}</small>" if secundario else ""}</div></div>',
                unsafe_allow_html=True,
            )


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
    linhas_html = []
    for _, linha in kpis_cluster.sort_values("perfil_numero").iterrows():
        cor = CORES_PERFIL[int(linha["perfil_numero"])]
        linhas_html.append(
            "<tr>"
            f'<td><span class="tag-pill" style="background:{cor};">'
            f'Perfil {int(linha["perfil_numero"])}</span></td>'
            f'<td>{int(linha["n_apolices"])}</td>'
            f'<td>{_fmt_brl(linha["premio_por_veiculo_medio"])}</td>'
            f'<td>{_fmt_brl(linha["valor_pago_historico_medio"])}</td>'
            f'<td>{linha["taxa_referral"]:.0%}</td>'
            "</tr>"
        )
    st.markdown(
        '<table class="tabela-perfis"><thead><tr>'
        "<th>Perfil</th><th>Apolices</th><th>Premio/veic.</th>"
        "<th>Custo hist.</th><th>Referral</th>"
        "</tr></thead><tbody>" + "".join(linhas_html) + "</tbody></table>",
        unsafe_allow_html=True,
    )


def render_histograma_variavel(original: pd.DataFrame, mapa_perfil: dict):
    st.markdown("##### Distribuicao das variaveis de entrada")
    st.caption(
        "Como uma das 19 variaveis derivadas (Quadro 2) usadas no K-Means se "
        "distribui na carteira, antes de ver os resultados da clusterizacao abaixo."
    )
    col_var, col_toggle = st.columns([3, 1])
    with col_var:
        indice_padrao = COLUNAS_19.index("premio_por_veiculo") if "premio_por_veiculo" in COLUNAS_19 else 0
        variavel = st.selectbox("Variavel", COLUNAS_19, index=indice_padrao, key="hist_variavel")
    with col_toggle:
        st.markdown("<div style='margin-top:1.7rem;'></div>", unsafe_allow_html=True)
        sobrepor = st.toggle("Sobrepor por perfil", value=False, key="hist_sobrepor")

    dados = original.copy()
    dados["perfil_numero"] = dados["cluster"].map(mapa_perfil)
    dados["nome_perfil"] = dados["perfil_numero"].map(NOME_PERFIL)

    if sobrepor:
        ordem_nomes = [NOME_PERFIL[n] for n in sorted(dados["perfil_numero"].unique())]
        ordem_cores = [CORES_PERFIL[n] for n in sorted(dados["perfil_numero"].unique())]
        grafico = alt.Chart(dados).mark_bar(opacity=0.6).encode(
            x=alt.X(f"{variavel}:Q", bin=alt.Bin(maxbins=30), title=variavel),
            y=alt.Y("count()", title="Numero de apolices", stack=None),
            color=alt.Color(
                "nome_perfil:N", title="Perfil",
                scale=alt.Scale(domain=ordem_nomes, range=ordem_cores),
            ),
            tooltip=["nome_perfil", "count()"],
        ).properties(height=300)
        st.caption(
            "Barras sobrepostas (nao empilhadas), uma cor por perfil -- a mesma "
            "paleta do grafico de clusters no espaco PCA."
        )
    else:
        grafico = alt.Chart(dados).mark_bar(color=COR_LINHA_DESTAQUE).encode(
            x=alt.X(f"{variavel}:Q", bin=alt.Bin(maxbins=30), title=variavel),
            y=alt.Y("count()", title="Numero de apolices"),
            tooltip=["count()"],
        ).properties(height=300)
    st.altair_chart(grafico, width="stretch")


def render_view_visao_geral(projecao, kpis_cluster, gerais, significancia, perfil_sel_nome,
                             original, mapa_perfil):
    perfil_sel_numero = None
    dados_kpi = gerais
    if perfil_sel_nome != CARTEIRA_COMPLETA:
        linha_sel = kpis_cluster[kpis_cluster["nome_perfil"] == perfil_sel_nome].iloc[0]
        perfil_sel_numero = int(linha_sel["perfil_numero"])
        dados_kpi = linha_sel.to_dict()

    st.subheader("KPIs")
    render_linha_kpis(dados_kpi)

    with st.container(border=True):
        render_histograma_variavel(original, mapa_perfil)

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
                             normalizada, validacao_k, original, mapa_perfil):
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
        render_view_visao_geral(
            projecao, kpis_cluster, gerais, significancia, segmento_selecionado, original, mapa_perfil
        )
    with aba_clusters:
        render_view_clusters(projecao, variancia_pct)
    with aba_validacao:
        render_view_validacao_indices(normalizada, validacao_k, significancia)


# ---------------------------------------------------------------------------
# Aba 2 -- Validacao de Hardware (Part B.5 / A.7): reescrita e movida para
# dashboard/tab_validacao_hardware.py + dashboard/charts_validacao.py, que
# consomem data/processed/uah_trips.csv (contrato definido por
# validacao_hardware.montar_trajetos). Ver o dispatch em main().
# ---------------------------------------------------------------------------

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
            st.markdown(f"**Trajeto de conducao (dataset publico) · {FONTE_CITACAO}**")
            st.caption("Trajeto real -- lado da validacao de hardware (Parte B)")
            st.markdown(
                f"- Comportamento rotulado: {COMPORTAMENTO_LABEL.get(evento['comportamento'], evento['comportamento'])}\n"
                f"- Motorista: {evento['motorista']} · Tipo de via: {evento['tipo_via']}\n"
                f"- Eventos por minuto (limiar ~6 m/s^2): {evento['eventos_por_min']:.3f}\n"
                f"- Aceleracao de pico no trajeto: {evento['magnitude_maxima_ms2']:.2f} m/s^2\n"
                f"- Velocidade media: {evento['velocidade_media_kmh']:.0f} km/h"
            )
    st.caption(narrativa)
    st.divider()


def render_view_coligacao(original: pd.DataFrame, mapa_perfil: dict, validacao: pd.DataFrame):
    st.markdown(
        '<div class="no-join-banner">'
        '<b>Nao ha juncao real entre os dois conjuntos de dados nesta aba.</b> '
        'A apolice sintetica e o trajeto de conducao publico abaixo nao compartilham '
        'nenhuma chave/entidade real -- nao existe motorista, veiculo ou vinculo '
        'contratual em comum entre uma apolice gerada sinteticamente para esta PoC e '
        'um motorista de um dataset publico de conducao. O par exibido e puramente '
        'ilustrativo: narra o TIPO de sinal que, em producao, um dispositivo ESP32 '
        'proprio instalado na frota dessa apolice alimentaria de volta na '
        'classificacao (Secao 2.2.5 / 3.4.1) -- nao um resultado estatistico '
        'demonstrado. O UAH-DriveSet entra nesta narrativa apenas como ilustracao do '
        'TIPO de sinal (evento de frenagem/curva/aceleracao) que, em producao, um '
        'dispositivo instalado numa transportadora real geraria -- ele nao e, e nunca '
        'foi, dado coletado de nenhuma apolice deste projeto.'
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

            perfil [label="Perfil historico\n(cotacao/apolice)\nclassificado Perfil 1/2/3", fillcolor="#8e2f9e", fontcolor="white"];
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
            classificacao -> perfil [label="  atualiza o perfil  ", style=dashed, color="#6d3bc4", fontcolor="#6d3bc4"];
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

    if validacao is None or validacao.empty:
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
        trajetos1 = validacao[validacao["comportamento"] == "agressiva"]
        if trajetos1.empty:
            trajetos1 = validacao
        trajeto1 = trajetos1.sort_values(
            ["eventos_por_min", "magnitude_maxima_ms2"], ascending=False
        ).iloc[0]
        render_exemplo_perfil(
            1, apolice1, trajeto1,
            "Narrativa ilustrativa: se o dispositivo ESP32 desta frota (ja "
            "sinalizada como alto risco pela segmentacao historica) registrasse "
            "um trajeto com esta taxa de eventos, esse sumario "
            "reforcaria/confirmaria a classificacao existente (Secao 3.4.1)."
        )

    cand2 = original[original["perfil_numero"] == 2]
    if not cand2.empty:
        apolice2 = cand2.sort_values(["qt_coberturas_ativas", "lmi_por_veiculo"], ascending=False).iloc[0]
        trajetos2 = validacao[validacao["comportamento"] == "normal"]
        if trajetos2.empty:
            trajetos2 = validacao
        trajeto2 = trajetos2.sort_values("magnitude_maxima_ms2", ascending=True).iloc[0]
        render_exemplo_perfil(
            2, apolice2, trajeto2,
            "Narrativa ilustrativa: um trajeto sem sinal de conducao agressiva "
            "reforcaria a leitura de baixo risco ja indicada pela alta cobertura "
            "e baixo custo historico desta apolice -- consistente com a manutencao "
            "de condicoes comerciais favoraveis."
        )

    cand3 = original[original["perfil_numero"] == 3]
    if not cand3.empty:
        apolice3 = cand3.sort_values(["tempo_cotacao_emissao", "referral_pendente"], ascending=False).iloc[0]
        mediana_magnitude = validacao["magnitude_maxima_ms2"].median()
        trajeto3 = validacao.loc[(validacao["magnitude_maxima_ms2"] - mediana_magnitude).abs().idxmin()]
        render_exemplo_perfil(
            3, apolice3, trajeto3,
            "Narrativa ilustrativa: cotacoes em referral ou conversao tardia ainda "
            "nao tem um padrao comportamental estabelecido -- um primeiro trajeto "
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
        "Resumo por trajeto (UAH-DriveSet)": "validacao_hardware_uah_driveset.csv",
        "Checagens de calibracao do sinal (UAH-DriveSet)": "validacao_hardware_uah_calibracao.csv",
        "Confundimento motorista x comportamento x via (UAH-DriveSet)": "validacao_hardware_uah_confundimento.csv",
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
    validacao_uah = carregar_validacao_uah()

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
            projecao, variancia_pct, kpis_cluster, gerais, significancia, normalizada, validacao_k,
            original, mapa_perfil,
        )
    elif view == "Validacao de Hardware":
        render(carregar("data/processed/uah_trips.csv"), limiar=THRESH_MAG_MS2)
    elif view == "Coligacao Conceitual":
        render_view_coligacao(original, mapa_perfil, validacao_uah)

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
