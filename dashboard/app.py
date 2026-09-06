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
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

RAIZ = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ / "src"))
from preprocessamento import COLUNAS_19  # noqa: E402
from histograma_didatico import (  # noqa: E402
    frase_de_leitura, histograma_por_perfil, int_ptbr as _int_ptbr, num_ptbr as _num_ptbr,
)

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
# Cores por rotulo comportamental do trajeto (reaproveita a paleta de perfis
# de negocio: agressiva ~ Perfil 1 de alto risco, normal ~ Perfil 2 de baixo
# risco); sonolenta e desconhecido usam tons neutros pois nao correspondem
# a nenhum perfil de negocio da segmentacao.
COMPORTAMENTO_COR = {
    "agressiva": CORES_PERFIL[1], "normal": CORES_PERFIL[2],
    "sonolenta": "#8a6d3b", "desconhecido": "#9a90a3",
}

# Agregacao simplificada dos tres rotulos em duas categorias -- ver
# render_binario_uah(): normal e o unico rotulo "sem sinal de risco", os
# outros dois viram "Mau comportamento". Cores reaproveitam a mesma
# paleta (verde/rosa) usada para normal/agressiva acima, para leitura
# consistente entre a visao de tres grupos e a visao binaria.
COMPORTAMENTO_BINARIO_LABEL = {
    "normal": "Bom comportamento", "agressiva": "Mau comportamento", "sonolenta": "Mau comportamento",
}
COMPORTAMENTO_BINARIO_COR = {
    "Bom comportamento": CORES_PERFIL[2], "Mau comportamento": CORES_PERFIL[1],
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


@st.cache_data
def carregar_calibracao_uah():
    """Checagens empiricas de calibracao do sinal (taxa de amostragem,
    teto fisico plausivel, reducao de ruido pelo filtro de Kalman), ja
    calculadas por src/validacao_hardware.py::verificar_calibracao_sinal
    -- exibidas aqui para que o resultado de nao-discriminacao da taxa de
    eventos (ver render_anova_uah) nao seja apresentado sem antes descartar
    um problema de unidade/amostragem como causa alternativa (Secao B.5)."""
    caminho = DADOS_DIR / "validacao_hardware_uah_calibracao.csv"
    return pd.read_csv(caminho) if caminho.exists() else None


@st.cache_data
def carregar_confundimento_uah():
    caminho = DADOS_DIR / "validacao_hardware_uah_confundimento.csv"
    return pd.read_csv(caminho) if caminho.exists() else None


@st.cache_data
def carregar_histograma_uah():
    """Classes ja binadas (numpy, em src/validacao_hardware.py) da magnitude
    amostra a amostra, por estilo de conducao e tipo de via. O binning nao e
    feito aqui nem no motor do grafico: as bordas de classe precisam existir
    como dado para que as barras possam ser rotuladas e a linha do limiar
    posicionada no valor exato."""
    caminho = DADOS_DIR / "validacao_hardware_uah_histograma.csv"
    if not caminho.exists():
        return None
    df = pd.read_csv(caminho)
    df["Estilo"] = df["comportamento"].map(COMPORTAMENTO_LABEL).fillna(df["comportamento"])
    return df


@st.cache_data
def carregar_limiares_uah():
    caminho = DADOS_DIR / "validacao_hardware_uah_limiares.csv"
    if not caminho.exists():
        return None
    df = pd.read_csv(caminho)
    df["Estilo"] = df["comportamento"].map(COMPORTAMENTO_LABEL).fillna(df["comportamento"])
    return df


def calcular_anova_uah(validacao: pd.DataFrame, coluna: str):
    """ANOVA one-way (coluna ~ comportamento), mesma logica de
    src/validacao_hardware.py::validar_discriminacao -- recalculada aqui
    apenas para exibicao (poucas linhas, custo desprezivel), nunca para
    decidir o resultado (o CSV consumido ja e o produzido pelo script)."""
    conhecidos = validacao[validacao["comportamento"] != "desconhecido"]
    grupos = [g[coluna].dropna().to_numpy() for _, g in conhecidos.groupby("comportamento")]
    grupos = [g for g in grupos if len(g) > 0]
    if len(grupos) < 2:
        return None
    return f_oneway(*grupos)


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
            --accent: #c97b9e; --accent-soft: #f6e6ee;
            --bg: #f6f4f9; --card: #ffffff;
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
        .side-badge b { color: var(--accent); }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--line) !important;
            border-radius: 12px !important;
        }

        .pill-badge {
            display: inline-block;
            background: var(--accent-soft);
            color: #9d5577;
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
# Aba 2 -- Validacao de Hardware (Part B.5 / A.7)
# ---------------------------------------------------------------------------

def render_metodologia_hardware(validacao: pd.DataFrame):
    n_trajetos = len(validacao)
    n_motoristas = validacao["motorista"].nunique()
    st.info(
        f"O limiar de magnitude de aceleracao do firmware (~6 m/s^2, Secao 2.8) e "
        f"aplicado diretamente ao sinal bruto do acelerometro (10 Hz) de cada um dos "
        f"{n_trajetos} trajetos gravados por {n_motoristas} motoristas do "
        f"**{FONTE_CITACAO}**. A taxa de eventos detectados por minuto e comparada, "
        "por ANOVA, entre os tres rotulos comportamentais conhecidos do dataset "
        "(normal / agressiva / sonolenta), codificados pelos proprios autores no "
        "nome de cada trajeto -- nao inferidos por este projeto."
    )
    st.caption(
        "O algoritmo de deteccao de eventos ja pre-extraido pelo dataset "
        "(`EVENTS_INERTIAL.txt`) nao e usado aqui de proposito: o objetivo desta aba "
        "e validar a logica do FIRMWARE, nao reaproveitar a deteccao de outro "
        "algoritmo. Ver docstring de `src/validacao_hardware.py`."
    )


def render_calibracao_uah(calibracao: pd.DataFrame):
    st.markdown("##### Verificacao de calibracao do sinal")
    st.caption(
        "Checagens empiricas feitas por src/validacao_hardware.py ANTES de reportar "
        "qualquer resultado de nao-discriminacao abaixo -- para descartar erro de "
        "unidade/amostragem como causa alternativa de uma taxa de eventos baixa "
        "(Secao B.5), em vez de assumir que o sinal esta correto."
    )
    if calibracao is None or calibracao.empty:
        st.caption("Execute src/validacao_hardware.py para gerar as checagens de calibracao.")
        return
    colunas = st.columns(len(calibracao))
    for coluna, (_, linha) in zip(colunas, calibracao.iterrows()):
        with coluna:
            icone = "✅" if linha["status"] == "ok" else "⚠️"
            st.markdown(f"{icone} **{linha['valor_observado']}**")
            st.caption(f"{linha['verificacao']} (esperado: {linha['valor_esperado']})")


def render_anova_uah(validacao: pd.DataFrame):
    st.markdown("##### Discriminacao por comportamento (ANOVA one-way)")
    st.caption(
        "**Eventos por minuto** = quantas vezes, em media, o firmware detectaria uma "
        "manobra brusca (frenagem, aceleracao ou curva acima de ~6 m/s^2) a cada "
        "minuto de viagem. **Aceleracao de pico** = a maior magnitude de aceleracao "
        "medida no trajeto, em m/s^2 -- quanto maior, mais brusca foi a manobra mais "
        "forte registrada."
    )
    st.caption(
        "F e p referem-se a diferenca de medias entre normal / agressiva / sonolenta; "
        "n e o numero de trajetos em cada grupo (pequeno e desigual -- ver "
        "confundimento abaixo). p < 0,05 indica que o rotulo comportamental explica "
        "parte da variacao observada na metrica."
    )
    conhecidos = validacao[validacao["comportamento"] != "desconhecido"]
    n_por_grupo = conhecidos.groupby("ComportamentoLabel").size().to_dict()
    legenda_n = " · ".join(f"{rotulo}: n={n}" for rotulo, n in n_por_grupo.items())
    st.caption(f"**{legenda_n}**")

    resultado_eventos = calcular_anova_uah(validacao, "eventos_por_min")
    resultado_pico = calcular_anova_uah(validacao, "magnitude_maxima_ms2")
    resultado_velocidade = calcular_anova_uah(validacao, "velocidade_media_kmh")

    col1, col2, col3 = st.columns(3)
    for coluna, rotulo, resultado in [
        (col1, "Eventos/min ~ comportamento", resultado_eventos),
        (col2, "Aceleracao de pico ~ comportamento", resultado_pico),
        (col3, "Velocidade media ~ comportamento", resultado_velocidade),
    ]:
        with coluna:
            if resultado is not None:
                st.metric(rotulo, f"F={resultado.statistic:.2f}", f"p={resultado.pvalue:.3f}", delta_color="off")
            else:
                st.metric(rotulo, "n/d")

    pico_significativo = resultado_pico is not None and resultado_pico.pvalue < 0.05
    eventos_nao_significativo = resultado_eventos is not None and resultado_eventos.pvalue >= 0.05
    if eventos_nao_significativo and pico_significativo:
        st.warning(
            "**A taxa de eventos por minuto nao discrimina significativamente entre "
            "os rotulos comportamentais (p >= 0,05) -- mas a aceleracao de pico, sim "
            "(p < 0,05).** O sinal de fato registra picos maiores em trajetos "
            "agressivos; o que perde poder discriminativo e o LIMIAR fixo de ~6 m/s^2 "
            "aplicado sobre ele -- picos de conducao agressiva sao breves (fracao de "
            "segundo) e raramente cruzam esse patamar especifico, mesmo quando estao "
            "visivelmente mais altos que o normal. A velocidade media tambem "
            "discrimina significativamente, mas correlaciona com o tipo de via "
            "(ver confundimento abaixo), entao nao deve ser lida como um efeito puro "
            "do rotulo comportamental."
        )
    elif eventos_nao_significativo:
        st.warning(
            "**A taxa de eventos por minuto nao discrimina significativamente entre "
            "os rotulos comportamentais nesta amostra (p >= 0,05).** As checagens de "
            "calibracao acima nao indicam erro de unidade/amostragem -- a leitura mais "
            "provavel e que o limiar fixo de ~6 m/s^2 e raramente cruzado mesmo em "
            "trajetos agressivos, dado o curto tempo de trajeto e a natureza breve "
            "desses picos."
        )


def _contagem_com_evento(dados: pd.DataFrame, coluna_grupo: str, ordem: list) -> pd.DataFrame:
    return (
        dados.groupby(coluna_grupo)
        .apply(lambda g: pd.Series({"com_evento": int((g["n_eventos"] > 0).sum()), "total": len(g)}), include_groups=False)
        .reindex(ordem)
    )


LARGURA_PILHA_MS2 = 0.25   # largura da faixa em que trajetos proximos se empilham


def _dot_plot_empilhado(dados: pd.DataFrame, coluna_grupo: str, ordem: list,
                         cores: list, altura_painel: int = 88) -> alt.Chart:
    """Dot plot com empilhamento DETERMINISTICO: trajetos com pico proximo sao
    agrupados numa faixa e empilhados verticalmente, em vez de deslocados por
    jitter aleatorio. Com jitter a posicao vertical nao significa nada, os
    pontos ainda se sobrepoem e o grafico muda a cada execucao; empilhados, a
    altura da coluna passa a ser a propria contagem daquele intervalo."""
    d = dados.copy()
    d["_faixa"] = (d["magnitude_maxima_ms2"] / LARGURA_PILHA_MS2).round().astype(int)
    d["_pilha"] = d.groupby([coluna_grupo, "_faixa"]).cumcount()
    d["_x"] = d["_faixa"] * LARGURA_PILHA_MS2
    altura_max = int(d["_pilha"].max()) + 1

    pontos = alt.Chart(d).mark_circle(size=95, opacity=0.9).encode(
        x=alt.X("_x:Q", title="Aceleracao de pico do trajeto (m/s²)",
                scale=alt.Scale(domain=[0.5, 6.4], nice=False)),
        y=alt.Y("_pilha:Q", title=None, axis=None,
                scale=alt.Scale(domain=[-0.6, altura_max])),
        color=alt.Color(f"{coluna_grupo}:N", legend=None,
                        scale=alt.Scale(domain=ordem, range=cores)),
        tooltip=["motorista", "trajeto", coluna_grupo, "tipo_via",
                 alt.Tooltip("magnitude_maxima_ms2:Q", title="Pico (m/s²)", format=".2f"),
                 alt.Tooltip("n_eventos:Q", title="Eventos detectados")],
    )
    mediana = alt.Chart(d).mark_rule(strokeWidth=2.5, color=COR_LINHA_DESTAQUE).encode(
        x=alt.X("median(magnitude_maxima_ms2):Q"))
    limiar = alt.Chart(pd.DataFrame({"v": [6.0]})).mark_rule(
        strokeDash=[6, 4], strokeWidth=2, color="#44355b").encode(x=alt.X("v:Q"))

    return (pontos + mediana + limiar).properties(height=altura_painel).facet(
        row=alt.Row(f"{coluna_grupo}:N", sort=ordem, title=None,
                    header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=12,
                                      labelFontWeight="bold", labelColor="#44355b")),
    ).resolve_scale(y="independent")


def render_grafico_eventos_comportamento(validacao: pd.DataFrame):
    st.markdown("##### Aceleracao de pico de cada trajeto, por rotulo comportamental")
    st.caption(
        "**Um ponto por trajeto** (nao um histograma): sao apenas 40 trajetos no "
        "dataset inteiro, e com esse n um histograma agruparia 2 ou 3 trajetos por "
        "classe e viraria ruido. Trajetos de pico parecido ficam empilhados na "
        "vertical, entao a altura de cada coluna e a contagem daquele intervalo."
    )
    dados = validacao[validacao["comportamento"] != "desconhecido"].copy()
    dados["ComportamentoLabel"] = dados["comportamento"].map(COMPORTAMENTO_LABEL)
    ordem = ["Normal", "Agressiva", "Sonolenta"]
    cores = [COMPORTAMENTO_COR["normal"], COMPORTAMENTO_COR["agressiva"],
             COMPORTAMENTO_COR["sonolenta"]]

    col_dots, col_slope = st.columns([1.5, 1])
    with col_dots:
        st.altair_chart(_dot_plot_empilhado(dados, "ComportamentoLabel", ordem, cores))
        st.caption(
            "Linha vertical rosa = mediana do grupo. Tracejada = limiar de 6 m/s²: "
            "nenhum trajeto normal ou agressivo chega la."
        )
    with col_slope:
        render_slope_motorista(dados, ordem, cores)

    render_contagem_limiar(dados, "ComportamentoLabel", ordem)


def render_slope_motorista(dados: pd.DataFrame, ordem: list, cores: list):
    """Mesmo motorista ligado nos tres estilos. O desenho experimental do
    UAH-DriveSet e de medidas repetidas -- os mesmos 6 condutores dirigiram nos
    tres estilos -- e o dot plot ao lado descarta essa informacao ao tratar os
    40 trajetos como independentes. Aqui cada linha e um motorista, o que
    responde diretamente a objecao 'nao e so que alguns motoristas dirigem mais
    forte que outros?': se a subida acontece dentro de cada condutor, a
    diferenca nao vem da composicao dos grupos."""
    st.markdown("**O mesmo motorista, nos tres estilos**")
    por_motorista = dados.groupby(["motorista", "ComportamentoLabel"], as_index=False).agg(
        pico=("magnitude_maxima_ms2", "mean"), n=("trajeto", "size"))

    base = alt.Chart(por_motorista).encode(
        x=alt.X("ComportamentoLabel:N", sort=ordem, title=None,
                axis=alt.Axis(labelAngle=0, labelFontSize=11)),
        y=alt.Y("pico:Q", title="Pico medio dos trajetos (m/s²)",
                scale=alt.Scale(domain=[0, 6.4])),
        detail=alt.Detail("motorista:N"),
    )
    linhas = base.mark_line(strokeWidth=2, color="#8a7f97", opacity=0.75)
    marcas = base.mark_point(size=85, filled=True).encode(
        color=alt.Color("ComportamentoLabel:N", legend=None,
                        scale=alt.Scale(domain=ordem, range=cores)),
        tooltip=["motorista", "ComportamentoLabel",
                 alt.Tooltip("pico:Q", title="Pico medio (m/s²)", format=".2f"),
                 alt.Tooltip("n:Q", title="trajetos")],
    )
    rotulos = alt.Chart(por_motorista[por_motorista["ComportamentoLabel"] == ordem[-1]]).mark_text(
        align="left", dx=10, fontSize=10, color="#65596f",
    ).encode(x=alt.X("ComportamentoLabel:N", sort=ordem), y=alt.Y("pico:Q"),
             text=alt.Text("motorista:N"))
    limiar = alt.Chart(pd.DataFrame({"v": [6.0]})).mark_rule(
        strokeDash=[6, 4], strokeWidth=2, color="#44355b").encode(y=alt.Y("v:Q"))

    st.altair_chart((linhas + marcas + rotulos + limiar).properties(height=300),
                    width="stretch")

    tabela = por_motorista.pivot(index="motorista", columns="ComportamentoLabel", values="pico")
    subiram = int((tabela["Agressiva"] > tabela["Normal"]).sum())
    st.caption(
        f"**{subiram} de {len(tabela)} motoristas** tem pico maior na conducao "
        f"agressiva do que na normal — a diferenca se sustenta dentro de cada "
        f"condutor, nao vem de uns dirigirem mais forte que outros. Em sonolenta o "
        f"padrao ja nao e consistente. Cada ponto e a media dos trajetos daquele "
        f"motorista naquele estilo (2 a 3 trajetos; D6 tem 1 em agressiva)."
    )


def render_contagem_limiar(dados: pd.DataFrame, coluna_grupo: str, ordem: list):
    st.markdown("###### Trajetos que ultrapassaram o limiar do firmware (~6 m/s^2)")
    st.caption(
        "Como o limiar e raramente cruzado (ver aviso acima), a contagem direta de "
        "trajetos com pelo menos um evento detectado e mais facil de ler do que a "
        "taxa de eventos por minuto, que fica perto de zero na quase totalidade dos "
        "casos."
    )
    resumo = _contagem_com_evento(dados, coluna_grupo, ordem)
    colunas = st.columns(len(ordem))
    for coluna, rotulo in zip(colunas, ordem):
        linha = resumo.loc[rotulo]
        with coluna:
            st.metric(
                rotulo, f"{int(linha['com_evento'])} de {int(linha['total'])}",
                "trajetos com evento", delta_color="off",
            )


ORDEM_ESTILO = ["Normal", "Agressiva", "Sonolenta"]
CORES_ESTILO = {
    "Normal": COMPORTAMENTO_COR["normal"],
    "Agressiva": COMPORTAMENTO_COR["agressiva"],
    "Sonolenta": COMPORTAMENTO_COR["sonolenta"],
}
LIMIAR_LITERATURA = 6.0


def _frase_limiares(limiares: pd.DataFrame, via: str) -> str:
    """Leitura automatica da parte que importa da figura: quanto de cada
    estilo fica acima de cada limiar, e a razao entre agressiva e normal."""
    sub = limiares if via == "Todas" else limiares[limiares["tipo_via"] == via]
    agregado = sub.groupby(["limiar_ms2", "Estilo"], as_index=False)[["n_acima", "n_total"]].sum()
    agregado["pct"] = 100 * agregado["n_acima"] / agregado["n_total"]

    partes = []
    for limiar in sorted(agregado["limiar_ms2"].unique(), reverse=True):
        linhas = agregado[agregado["limiar_ms2"] == limiar].set_index("Estilo")
        trechos = [
            f"{est.lower()} {_num_ptbr(linhas.loc[est, 'pct'], 3)}%"
            for est in ORDEM_ESTILO if est in linhas.index
        ]
        frase = f"Acima de {_num_ptbr(limiar)} m/s²: " + ", ".join(trechos) + "."
        if "Agressiva" in linhas.index and "Normal" in linhas.index:
            pct_agr, pct_norm = linhas.loc["Agressiva", "pct"], linhas.loc["Normal", "pct"]
            if pct_norm > 0:
                frase += f" A condução agressiva dispara {pct_agr / pct_norm:.0f}× mais que a normal."
            elif pct_agr == 0:
                frase += (" Nem a condução normal nem a agressiva cruzam esse limiar "
                          "nesta amostra — ou seja, ele não separa os dois estilos.")
            else:
                frase += " A condução normal nunca cruza esse limiar nesta amostra."
        partes.append(frase)
    return " ".join(partes)


def render_histograma_sinal(histograma: pd.DataFrame, limiares: pd.DataFrame,
                             limiar_recalibrado: float):
    st.markdown("##### Distribuicao do sinal do acelerometro e onde o limiar corta")
    st.caption(
        "Cada amostra do acelerometro (10 Hz, todos os trajetos) entra em uma classe "
        "de magnitude. E a figura que mostra, diretamente, quanto do sinal real fica "
        "acima do limiar de deteccao do firmware."
    )

    via = st.radio(
        "Tipo de via", ["Todas", "Rodovia", "Via secundaria"], horizontal=True,
        key="hist_sinal_via",
        help="Trajetos de rodovia sao mais rapidos que os de via secundaria "
             "independentemente do estilo de conducao -- separar por via deixa esse "
             "confundimento visivel.",
    )

    dados = histograma if via == "Todas" else histograma[histograma["tipo_via"] == via]
    classes = dados.groupby(
        ["Estilo", "classe_idx", "classe_min", "classe_max"], as_index=False
    )["n_amostras"].sum()

    grafico = histograma_por_perfil(
        classes, coluna_grupo="Estilo", ordem=ORDEM_ESTILO, cores=CORES_ESTILO,
        rotulo_x="Magnitude da aceleracao (m/s²)", unidade="amostras",
        limiares={
            f"{_num_ptbr(limiar_recalibrado)} m/s² (recalibrado)": limiar_recalibrado,
            f"{_num_ptbr(LIMIAR_LITERATURA)} m/s² (literatura)": LIMIAR_LITERATURA,
        },
        escala_log=True, largura=660, altura=135,
    )
    st.altair_chart(grafico)

    total = int(classes["n_amostras"].sum())
    st.info(
        f"**Leitura:** {frase_de_leitura(classes, unidade='amostras', rotulo_x='magnitude')} "
        f"{_frase_limiares(limiares, via)}"
    )
    st.caption(
        f"Escala logaritmica no eixo Y (a classe mais baixa concentra a maior parte "
        f"das {_int_ptbr(total)} amostras; em escala linear as classes da cauda -- "
        f"justamente onde o limiar corta -- ficariam invisiveis). Classes vazias nao "
        f"aparecem."
    )

    with st.expander("O que este grafico mostra"):
        st.markdown(
            "- **O que e um histograma:** ele agrupa os valores medidos em faixas "
            "(classes) de mesma largura e conta quantas medidas caem em cada faixa. "
            "Aqui cada medida e uma leitura do acelerometro a 10 Hz.\n"
            "- **Eixo Y:** numero de amostras naquela faixa de magnitude -- nao e "
            "tempo nem numero de viagens. Esta em escala logaritmica porque a "
            "primeira classe concentra dezenas de milhares de amostras e as ultimas, "
            "poucas dezenas.\n"
            "- **Por que a linha do limiar e o centro da figura:** o firmware so "
            "registra um evento quando a magnitude ultrapassa o limiar. Tudo que "
            "esta a esquerda da linha e ignorado pelo dispositivo; so o que esta a "
            "direita vira evento transmitido.\n"
            "- **Como comparar os tres paineis:** eles usam as mesmas classes e a "
            "mesma escala. O que muda entre estilos e ate onde a distribuicao se "
            "estende para a direita."
        )


def render_binario_uah(validacao: pd.DataFrame):
    st.markdown("##### Visao simplificada: bom vs. mau comportamento")
    st.caption(
        "**Nao e uma analise nova ou concorrente** -- e a mesma ANOVA de tres grupos "
        "mostrada acima, apenas agregada em duas categorias para leitura rapida: "
        "normal vira 'Bom comportamento'; agressiva e sonolenta viram 'Mau "
        "comportamento'."
    )
    dados = validacao[validacao["comportamento"] != "desconhecido"].copy()
    dados["ComportamentoBinario"] = dados["comportamento"].map(COMPORTAMENTO_BINARIO_LABEL)

    ordem = ["Bom comportamento", "Mau comportamento"]
    resumo = dados.groupby("ComportamentoBinario").agg(
        n=("trajeto", "count"),
        eventos_por_min_medio=("eventos_por_min", "mean"),
        magnitude_maxima_media=("magnitude_maxima_ms2", "mean"),
    ).reindex(ordem)

    colunas = st.columns(2)
    for coluna, rotulo in zip(colunas, ordem):
        linha = resumo.loc[rotulo]
        with coluna:
            with st.container(border=True):
                cor = COMPORTAMENTO_BINARIO_COR[rotulo]
                st.markdown(
                    f'<span class="tag-pill" style="background:{cor};">{rotulo}</span> '
                    f'<span style="color:var(--muted);font-size:12px;">n={int(linha["n"])} trajetos</span>',
                    unsafe_allow_html=True,
                )
                col_a, col_b = st.columns(2)
                col_a.metric("Eventos/min (media)", f"{linha['eventos_por_min_medio']:.3f}")
                col_b.metric("Pico medio (m/s^2)", f"{linha['magnitude_maxima_media']:.2f}")

    st.markdown("###### Aceleracao de pico: bom vs. mau comportamento")
    st.caption(
        "Um ponto por trajeto (mesmo motivo do grafico anterior: n=40 nao sustenta "
        "histograma). Eventos por minuto fica perto de zero nos dois grupos -- a "
        "diferenca aparece na aceleracao de pico."
    )
    cores = [COMPORTAMENTO_BINARIO_COR[o] for o in ordem]
    st.altair_chart(
        _dot_plot_empilhado(dados, "ComportamentoBinario", ordem, cores, altura_painel=105)
    )
    st.caption(
        "Trajetos de pico parecido ficam empilhados; linha rosa = mediana do grupo, "
        "tracejada = limiar de 6 m/s²."
    )

    st.markdown("###### Trajetos que ultrapassaram o limiar do firmware (~6 m/s^2)")
    resumo_evento = _contagem_com_evento(dados, "ComportamentoBinario", ordem)
    colunas = st.columns(2)
    for coluna, rotulo in zip(colunas, ordem):
        linha = resumo_evento.loc[rotulo]
        with coluna:
            st.metric(
                rotulo, f"{int(linha['com_evento'])} de {int(linha['total'])}",
                "trajetos com evento", delta_color="off",
            )


def render_tabela_trajetos(validacao: pd.DataFrame):
    st.markdown("##### Trajetos do UAH-DriveSet")
    st.caption(
        f"Um trajeto por linha, ja resumido por src/validacao_hardware.py. Fonte: "
        f"{FONTE_CITACAO}."
    )
    tabela = validacao.copy()
    tabela = tabela[[
        "motorista", "ComportamentoLabel", "tipo_via", "duracao_min", "n_eventos",
        "eventos_por_min", "magnitude_maxima_ms2", "velocidade_media_kmh",
    ]]
    tabela.columns = [
        "Motorista", "Comportamento", "Tipo de via", "Duracao (min)", "N. eventos",
        "Eventos/min", "Aceleracao de pico (m/s^2)", "Velocidade media (km/h)",
    ]
    st.dataframe(
        tabela, width="stretch", hide_index=True,
        column_config={
            "Duracao (min)": st.column_config.NumberColumn(format="%.1f"),
            "Eventos/min": st.column_config.NumberColumn(format="%.3f"),
            "Aceleracao de pico (m/s^2)": st.column_config.NumberColumn(format="%.2f"),
            "Velocidade media (km/h)": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption(
        "**Legenda:** `Motorista` -- identificador D1-D6 usado pelo UAH-DriveSet "
        "(nao e uma pessoa identificada). `Comportamento` -- rotulo atribuido pelos "
        "autores do dataset ao gravar o trajeto, nao inferido por este projeto. "
        "`Tipo de via` -- rodovia ou via secundaria, extraido do nome do trajeto; "
        "trajetos de rodovia tendem a velocidade media mais alta, um fator "
        "parcialmente confundido com o rotulo comportamental (ver confundimento abaixo)."
    )


def render_confundimento_uah(confundimento: pd.DataFrame):
    st.markdown("##### Confundimento motorista x comportamento x tipo de via")
    if confundimento is None or confundimento.empty:
        st.caption("Execute src/validacao_hardware.py para gerar a tabela de confundimento.")
        return
    st.caption(
        "Numero de trajetos em cada combinacao. Motorista, comportamento e tipo de "
        "via NAO sao totalmente cruzados neste dataset -- celulas ausentes (n=0, nao "
        "listadas abaixo) significam que aquele motorista nao tem trajeto naquela "
        "combinacao, entao nenhuma conclusao por motorista ou por tipo de via deve "
        "ser tirada isoladamente desta validacao."
    )
    tabela = confundimento.rename(columns={
        "motorista": "Motorista", "comportamento": "Comportamento",
        "tipo_via": "Tipo de via", "n_trajetos": "N. trajetos",
    })
    st.dataframe(tabela, width="stretch", hide_index=True)


def render_limitacoes_hardware():
    with st.expander("Limitacoes desta validacao (Secao 3.8)", expanded=False):
        st.markdown(
            "- O rotulo comportamental (normal/agressiva/sonolenta) e atribuido ao "
            "**trajeto inteiro**, por instrucao ao motorista ou pelos autores do "
            "dataset no momento da coleta -- nao e verificado evento a evento por "
            "este projeto.\n"
            "- Em particular, **'sonolenta' e um estilo de conducao instruido durante "
            "a coleta, nao uma medida fisiologica de sonolencia** -- esta validacao "
            "nao demonstra deteccao clinica ou fisiologicamente verificada de "
            "sonolencia, apenas se o sinal cinematico discrimina esse estilo "
            "instruido dos demais.\n"
            "- Motorista, comportamento e tipo de via so sao parcialmente cruzados "
            "(ver tabela de confundimento acima) -- conclusoes por motorista "
            "isolado, ou que atribuam um efeito de velocidade puramente ao "
            "comportamento sem considerar o tipo de via, nao sao sustentadas por "
            "esta amostra.\n"
            "- Amostra pequena e desigual por grupo (n=11 a 17 trajetos por rotulo "
            "comportamental) -- tratada como checagem de viabilidade/discriminacao, "
            "nao como validacao estatisticamente dimensionada, na mesma postura "
            "adotada pelos testes de bancada e rodagem no veiculo do autor "
            "(Secao 3.3).\n"
            "- **O detector e generico**: ele aplica um limiar a magnitude de "
            "aceleracao de dois eixos, sem separar qual eixo disparou. Uma curva "
            "brusca aciona o mesmo criterio que uma frenagem ou uma aceleracao "
            "forte -- o dispositivo registra 'manobra brusca', nao 'frenagem "
            "brusca'.\n"
            "- **O limiar de jerk de 12 m/s³ citado na Secao 2.8 e um valor de "
            "partida, ainda sem base empirica** -- e, nesta validacao, ele nao esta "
            "em uso: o jerk e calculado a partir do sinal, mas a deteccao aqui "
            "depende apenas da magnitude. Portanto o que este resultado valida e a "
            "regra de magnitude, nao o criterio combinado descrito no firmware.\n"
            "- O firmware completo (Secao 2.8) ainda faz subtracao de gravidade por "
            "calibracao estacionaria e media movel passa-baixa. O UAH-DriveSet ja "
            "entrega o sinal com gravidade removida e filtrado por Kalman, entao "
            "essas duas etapas nao sao exercitadas por esta validacao."
        )


def render_view_validacao_hardware(validacao, calibracao, confundimento,
                                    histograma=None, limiares=None):
    st.info(
        "**Por que usamos o UAH-DriveSet aqui?** O UAH-DriveSet e uma base publica de "
        "terceiros, usada exclusivamente para validar se a logica de deteccao do "
        "firmware (limiar de ~6 m/s^2 sobre a magnitude da aceleracao) consegue "
        "discriminar estilos de conducao conhecidos (normal/agressiva/sonolenta). Ele "
        "nao contem, e nunca contera, dados de nenhuma transportadora real deste "
        "projeto. Quando o dispositivo for instalado em campo, os dados de producao "
        "-- eventos detectados nas apolices reais -- e que alimentam o loop de "
        "classificacao descrito na Secao 3.4.1 do TCC. Esta aba valida o metodo; nao "
        "e a fonte dos dados que classificam clientes."
    )

    if validacao is None or validacao.empty:
        st.error(
            "Nenhum resultado de validacao de hardware encontrado em data/processed/. "
            "Execute `python src/validacao_hardware.py` antes de abrir esta aba."
        )
        return

    # O limiar recalibrado nao e escolhido aqui: vem do CSV que
    # src/validacao_hardware.py produziu (p99.9 da conducao normal).
    limiar_recalibrado = None
    if limiares is not None and not limiares.empty:
        candidatos = sorted(set(limiares["limiar_ms2"]) - {LIMIAR_LITERATURA})
        limiar_recalibrado = candidatos[0] if candidatos else None

    render_metodologia_hardware(validacao)

    with st.container(border=True):
        render_calibracao_uah(calibracao)

    if histograma is not None and limiares is not None and limiar_recalibrado is not None:
        with st.container(border=True):
            render_histograma_sinal(histograma, limiares, limiar_recalibrado)

    with st.container(border=True):
        render_anova_uah(validacao)
        render_grafico_eventos_comportamento(validacao)

    with st.container(border=True):
        render_binario_uah(validacao)

    with st.container(border=True):
        render_tabela_trajetos(validacao)
        render_confundimento_uah(confundimento)

    render_limitacoes_hardware()


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
    calibracao_uah = carregar_calibracao_uah()
    confundimento_uah = carregar_confundimento_uah()
    histograma_uah = carregar_histograma_uah()
    limiares_uah = carregar_limiares_uah()

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
        render_view_validacao_hardware(validacao_uah, calibracao_uah, confundimento_uah,
                                        histograma_uah, limiares_uah)
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
