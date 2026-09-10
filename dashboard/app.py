"""
Dashboard Seguros IoT -- plataforma hibrida de validacao de risco
(PFE II, Etapa 4 / Secao 3.5.4).

Tres abas, na estrutura do painel desenhado pela autora:
  * Software (K-Means)     -- Segmentacao da carteira (Parte A)
  * Hardware (IoT Edge)    -- Validacao da assinatura inercial (Parte B)
  * Integracao UBI         -- Coligacao conceitual das duas camadas

Todo numero em tela vem do pipeline real (data/processed/*.csv). O mockup
que serviu de referencia visual trazia valores de exemplo (2.489 apolices,
radar 0-100 arbitrario, "amostragem 10Hz") -- nenhum deles foi copiado: a
taxa real do firmware e 50Hz (firmware/esp32/include/config.h) e todas as
estatisticas saem dos CSVs gerados por src/.

A analise metodologica longa da camada de hardware -- as nove secoes de
dashboard/tab_validacao_hardware.py (regua da literatura, curva de
sensibilidade, varredura de calibracao, efeito pareado por motorista,
tabela de cobertura) -- continua no repositorio e volta a aparecer com uma
chamada a `tab_validacao_hardware.render(df, limiar)`. Ela saiu da tela a
pedido da autora, para que o painel abrisse na leitura curta; os modulos
nao foram removidos.

Uso:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

RAIZ = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ / "src"))
sys.path.append(str(RAIZ))
from preprocessamento import COLUNAS_19  # noqa: E402
from validacao_hardware import THRESH_MAG_MS2  # noqa: E402
from dashboard.tab_validacao_hardware import (  # noqa: E402
    carregar_histograma, carregar_limiar_recalibrado, contagem_por_faixa,
)

DADOS_DIR = RAIZ / "data" / "processed"

# Nome longo (linguagem da metodologia) e nome curto (o que cabe no cartao e
# na tabela). Os dois descrevem o mesmo perfil da Secao 3.4.
NOME_PERFIL = {
    1: "Perfil 1 - Frota de Alto Risco Operacional",
    2: "Perfil 2 - Segurado de Alta Cobertura e Baixo Custo Relativo",
    3: "Perfil 3 - Cotação em Referral ou Conversão Tardia",
}
NOME_CURTO = {
    1: "1 - Alto Risco (Agravo)",
    2: "2 - Baixo Risco (Desconto)",
    3: "3 - Risco Incerto (Referral)",
}

# Paleta de risco (vermelho/verde/ambar), escolhida pela autora no desenho do
# painel: a cor carrega o significado de negocio, o que ajuda a leitura pela
# banca. Substitui a paleta roxo/azul/verde anterior, que era validada para
# daltonismo com scripts/validate_palette.py -- se a acessibilidade voltar a
# pesar mais que a semantica, o mapa antigo era {1:"#8e2f9e", 2:"#2a78d6",
# 3:"#008300"}. A cor e atribuida por PERFIL, nunca pelo indice que o K-Means
# sorteia para o cluster, para a identidade visual nao mudar entre execucoes.
CORES_PERFIL = {1: "#ef4444", 2: "#10b981", 3: "#f59e0b"}
CLASSE_BADGE = {1: "badge-red", 2: "badge-green", 3: "badge-yellow"}

# Acao de hardware prevista por perfil -- regra de negocio da Secao 3.4
# (o dispositivo como agravo, como fidelizacao e como condicao de emissao),
# nao um numero calculado.
ACAO_HARDWARE = {
    1: "Mandatório (renovação)",
    2: "Opcional (fidelização)",
    3: "Condição de emissão",
}

# Cores dos rotulos comportamentais do UAH-DriveSet. Verde/ambar/vermelho
# para casar com as tres caixas de status logo ao lado do grafico.
CORES_COMPORTAMENTO = {
    "normal": ("#10b981", "16,185,129"),
    "sonolenta": ("#f59e0b", "245,158,11"),
    "agressiva": ("#ef4444", "239,68,68"),
}
ORDEM_COMPORTAMENTO = ["normal", "sonolenta", "agressiva"]

FONTE_CITACAO = "UAH-DriveSet (ROMERA; BERGASA; ARROYO, 2016)"

# Rotulo de apresentacao das 19 variaveis do Quadro 2. O nome tecnico continua
# sendo a chave em todo o pipeline; isto e so o texto que a banca le no radar.
ROTULO_VARIAVEL = {
    "total_veiculos": "Tamanho da frota",
    "pct_autonomos": "% de autônomos/terceiros",
    "motorista_licenciado": "Motorista licenciado",
    "classe_risco": "Classe de risco",
    "sinistralidade_rc_declarada": "Sinistralidade RC declarada",
    "sinistralidade_rcfv_declarada": "Sinistralidade RCFV declarada",
    "valor_pago_historico": "Custo histórico pago",
    "valor_sinistro_historico": "Valor histórico de sinistro",
    "qt_cias_anteriores": "Seguradoras anteriores",
    "tipo_sinistro_predominante": "Tipo de sinistro predominante",
    "lmi_por_veiculo": "LMI por veículo",
    "premio_por_veiculo": "Prêmio por veículo",
    "qt_coberturas_ativas": "Coberturas ativas",
    "agravo_aplicado": "Agravo aplicado",
    "desconto_aplicado": "Desconto aplicado",
    "referral_pendente": "Referral pendente",
    "tempo_cotacao_emissao": "Tempo cotação → emissão",
    "parcelas_com_juros": "Parcelas com juros",
    "apolice_anterior": "Apólice anterior",
}


def _fmt_brl_compacto(valor: float) -> str:
    if abs(valor) >= 1_000_000:
        texto = f"R$ {valor / 1_000_000:.1f} mi"
    elif abs(valor) >= 1_000:
        texto = f"R$ {valor / 1_000:.0f} mil"
    else:
        texto = f"R$ {valor:.0f}"
    return texto.replace(".", ",")


def _br(valor: float, casas: int = 1) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def _int_br(valor: int) -> str:
    return f"{int(valor):,}".replace(",", ".")


# ---------------------------------------------------------------------------
# Dados -- tudo lido dos CSVs que src/ produz; o dashboard nunca recalcula
# deteccao de evento, limiar ou rotulo.
# ---------------------------------------------------------------------------

@st.cache_data
def carregar_dados():
    normalizada = pd.read_csv(DADOS_DIR / "matriz_normalizada_clusters.csv")
    original = pd.read_csv(DADOS_DIR / "matriz_original_clusters.csv")
    return normalizada, original


@st.cache_data
def carregar_significancia():
    caminho = DADOS_DIR / "perfis_testes_significancia.csv"
    return pd.read_csv(caminho) if caminho.exists() else None


def projetar_pca(normalizada: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    pca = PCA(n_components=2, random_state=42)
    componentes = pca.fit_transform(normalizada[COLUNAS_19])
    projecao = normalizada[["numeroApolice", "cluster"]].copy()
    projecao["PCA1"] = componentes[:, 0]
    projecao["PCA2"] = componentes[:, 1]
    return projecao, float(pca.explained_variance_ratio_.sum())


def numero_perfil(medias: pd.Series) -> int:
    """Confronta as medias do cluster no espaco original com os tres
    perfis-alvo da Secao 3.4, para apresentacao em linguagem de negocio."""
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
        valor_pago_historico_medio=("valor_pago_historico", "mean"),
        taxa_referral=("referral_pendente", "mean"),
        n_apolices=("numeroApolice", "count"),
    ).reset_index()
    kpis["perfil_numero"] = kpis["cluster"].map(mapa_perfil)
    return kpis.sort_values("perfil_numero").reset_index(drop=True)


def calcular_radar_perfis(normalizada: pd.DataFrame, significancia: pd.DataFrame,
                          mapa_perfil: dict, top_n: int = 6):
    """Media por perfil das `top_n` variaveis mais discriminantes (ranking por
    p_valor dos testes de significancia), na matriz Min-Max 0-1 que alimentou o
    K-Means -- escalada a 0-100 so para leitura no radar."""
    variaveis = significancia.sort_values("p_valor")["variavel"].head(top_n).tolist()
    dados = normalizada.copy()
    dados["perfil_numero"] = dados["cluster"].map(mapa_perfil)
    return variaveis, dados.groupby("perfil_numero")[variaveis].mean() * 100


# ---------------------------------------------------------------------------
# Estilo
# ---------------------------------------------------------------------------

def aplicar_estilo():
    st.markdown(
        """
        <style>
        .stApp { background-color: #faf5ff; }

        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #ede9fe;
            border-radius: 8px 8px 0px 0px;
            color: #6d28d9;
            font-weight: 600;
            padding: 0.5rem 1.5rem;
            border: 1px solid #ddd6fe;
            border-bottom: none;
        }
        .stTabs [aria-selected="true"] {
            background-color: #7c3aed;
            color: white !important;
            border-color: #7c3aed;
        }

        .kpi-card {
            background-color: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            border-left: 5px solid #8b5cf6;
            display: flex;
            align-items: center;
            gap: 15px;
            height: 100%;
        }
        .kpi-icon {
            background-color: #ede9fe;
            color: #7c3aed;
            border-radius: 10px;
            padding: 12px;
            font-size: 22px;
            line-height: 1;
        }
        .kpi-title {
            font-size: 0.72rem; text-transform: uppercase; font-weight: 700;
            color: #6b7280; margin: 0; letter-spacing: 0.05em;
        }
        .kpi-value {
            font-size: 1.4rem; font-weight: 800; color: #1f2937; margin: 5px 0 0 0;
        }
        .kpi-sub { font-size: 0.8rem; font-weight: 400; color: #9ca3af; }

        .hw-card {
            background-color: white; border: 1px solid #e5e7eb; border-radius: 12px;
            padding: 20px; text-align: center; height: 100%;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        .hw-icon { font-size: 30px; margin-bottom: 10px; }
        .hw-title {
            font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: #6b7280;
        }
        .hw-value { font-size: 1.05rem; font-weight: 700; color: #4c1d95; margin-top: 5px; }

        .alert-box {
            padding: 16px; border-radius: 12px; display: flex;
            align-items: flex-start; gap: 12px; margin-bottom: 14px;
        }
        .alert-box h4 { margin: 0 0 4px 0; font-size: 15px; }
        .alert-box p { margin: 0; font-size: 13.5px; line-height: 1.5; }
        .alert-good { background-color: #ecfdf5; border: 1px solid #a7f3d0; }
        .alert-good h4 { color: #065f46; } .alert-good p { color: #047857; }
        .alert-warn { background-color: #fffbeb; border: 1px solid #fde68a; }
        .alert-warn h4 { color: #92400e; } .alert-warn p { color: #b45309; }
        .alert-bad { background-color: #fef2f2; border: 1px solid #fecaca; }
        .alert-bad h4 { color: #991b1b; } .alert-bad p { color: #b91c1c; }

        .styled-table {
            width: 100%; border-collapse: collapse; font-size: 13.5px;
            background-color: white; border-radius: 12px; overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        }
        .styled-table thead tr {
            background-color: #ede9fe; color: #4c1d95; text-align: left;
        }
        .styled-table th, .styled-table td { padding: 12px 15px; }
        .styled-table tbody tr { border-bottom: 1px solid #f3f4f6; }
        .styled-table tbody tr:nth-of-type(even) { background-color: #fafaf9; }
        .badge {
            padding: 4px 10px; border-radius: 999px; font-size: 11px;
            font-weight: 600; white-space: nowrap;
        }
        .badge-red { background-color: #fee2e2; color: #b91c1c; }
        .badge-green { background-color: #d1fae5; color: #047857; }
        .badge-yellow { background-color: #fef3c7; color: #b45309; }
        .badge-violet { background-color: #ede9fe; color: #6d28d9; }

        .secao-titulo {
            color: #4c1d95; font-size: 16px; font-weight: 700; margin-bottom: 4px;
        }
        .secao-sub { color: #6b7280; font-size: 12.5px; margin-bottom: 14px; }

        .nota-integridade {
            background-color: white; border: 1px solid #ddd6fe;
            border-left: 5px solid #7c3aed; border-radius: 12px;
            padding: 14px 18px; color: #4b5563; font-size: 13px; line-height: 1.55;
        }
        .nota-integridade b { color: #4c1d95; }

        .card-texto {
            background-color: white; padding: 26px; border-radius: 12px;
            border: 1px solid #ddd6fe; height: 100%;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _kpi_card(coluna, icone: str, titulo: str, valor: str, sufixo: str = ""):
    complemento = f' <span class="kpi-sub">{sufixo}</span>' if sufixo else ""
    with coluna:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-icon">{icone}</div><div>'
            f'<p class="kpi-title">{titulo}</p>'
            f'<p class="kpi-value">{valor}{complemento}</p>'
            "</div></div>",
            unsafe_allow_html=True,
        )


def _hw_card(coluna, icone: str, titulo: str, valor: str):
    with coluna:
        st.markdown(
            f'<div class="hw-card"><div class="hw-icon">{icone}</div>'
            f'<p class="hw-title">{titulo}</p>'
            f'<p class="hw-value">{valor}</p></div>',
            unsafe_allow_html=True,
        )


def _alert_box(classe: str, icone: str, titulo: str, texto: str) -> str:
    return (
        f'<div class="alert-box {classe}"><div style="font-size:22px;">{icone}</div>'
        f"<div><h4>{titulo}</h4><p>{texto}</p></div></div>"
    )


def _titulo_secao(titulo: str, subtitulo: str):
    st.markdown(
        f'<p class="secao-titulo">{titulo}</p>'
        f'<p class="secao-sub">{subtitulo}</p>',
        unsafe_allow_html=True,
    )


def _layout_plotly(fig: go.Figure, altura: int) -> go.Figure:
    fig.update_layout(
        height=altura,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#4b5563"),
    )
    return fig


# ---------------------------------------------------------------------------
# Aba 1 -- Software (K-Means): segmentacao da carteira (Parte A)
# ---------------------------------------------------------------------------

def render_tab_software(projecao, kpis_cluster, gerais, variancia_pct):
    linha_p1 = kpis_cluster[kpis_cluster["perfil_numero"] == 1]
    custo_p1 = float(linha_p1["valor_pago_historico_medio"].iloc[0]) if not linha_p1.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    _kpi_card(c1, "👥", "Base analisada", _int_br(gerais["n_apolices"]), "apólices")
    _kpi_card(c2, "💰", "Prêmio médio / veículo",
              _fmt_brl_compacto(gerais["premio_por_veiculo_medio"]), "por ano")
    _kpi_card(c3, "⏳", "Taxa de referral", f"{_br(100 * gerais['taxa_referral'])}%")
    _kpi_card(c4, "⚠️", "Custo médio (Perfil 1)", _fmt_brl_compacto(custo_p1), "/sinistro")

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart, col_table = st.columns([1, 2])

    with col_chart:
        _titulo_secao(
            "Projeção PCA (2D)",
            f"Redução das {len(COLUNAS_19)} variáveis derivadas (Quadro 2) para "
            f"visualização espacial dos 3 clusters do K-Means. Os dois componentes "
            f"explicam {_br(100 * variancia_pct)}% da variância. A projeção é apenas "
            f"visualização — nunca entra no treino.",
        )
        fig = px.scatter(
            projecao, x="PCA1", y="PCA2", color="perfil_curto",
            color_discrete_map={NOME_CURTO[n]: CORES_PERFIL[n] for n in NOME_CURTO},
            category_orders={"perfil_curto": [NOME_CURTO[n] for n in (1, 2, 3)]},
            custom_data=["numeroApolice"],
        )
        fig.update_traces(
            marker=dict(size=9, opacity=0.8),
            hovertemplate="Apólice %{customdata[0]}<extra></extra>",
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                        xanchor="center", x=0.5, title=""),
        )
        st.plotly_chart(_layout_plotly(fig, 320), use_container_width=True)

    with col_table:
        _titulo_secao(
            "Segmentação estratégica (k=3)",
            "Médias no espaço original de cada perfil. A coluna de ação via hardware "
            "é a regra de negócio da Seção 3.4 (o dispositivo como agravo, "
            "fidelização ou condição de emissão), não um valor calculado.",
        )
        linhas = ""
        for _, linha in kpis_cluster.iterrows():
            perfil = int(linha["perfil_numero"])
            linhas += (
                "<tr>"
                f'<td style="font-weight:600;color:#1f2937;">{NOME_CURTO[perfil]}</td>'
                f'<td>{_int_br(linha["n_apolices"])} apólices</td>'
                f'<td>{_fmt_brl_compacto(linha["lmi_por_veiculo_medio"])}</td>'
                f'<td>{_fmt_brl_compacto(linha["premio_por_veiculo_medio"])}</td>'
                f'<td style="font-weight:600;color:#b91c1c;">'
                f'{_fmt_brl_compacto(linha["valor_pago_historico_medio"])}</td>'
                f'<td><span class="badge {CLASSE_BADGE[perfil]}">{ACAO_HARDWARE[perfil]}</span></td>'
                "</tr>"
            )
        st.markdown(
            '<table class="styled-table"><thead><tr>'
            "<th>Perfil identificado</th><th>Volumetria</th><th>LMI médio/veíc.</th>"
            "<th>Prêmio médio/veíc.</th><th>Custo hist.</th><th>Ação via hardware</th>"
            "</tr></thead><tbody>" + linhas + "</tbody></table>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Aba 2 -- Hardware (IoT Edge): assinatura inercial medida na borda (Parte B)
# ---------------------------------------------------------------------------

def fig_assinatura_inercial(histograma: pd.DataFrame, limiar: float,
                            limiar_recal: float | None) -> go.Figure:
    """Densidade da magnitude por rotulo comportamental, sobre as 311 mil
    amostras do UAH-DriveSet.

    Eixo Y em log de proposito: a primeira classe concentra a maior parte das
    amostras e, em escala linear, a cauda -- justamente onde os limiares
    cortam -- fica invisivel."""
    fig = go.Figure()
    for comportamento in ORDEM_COMPORTAMENTO:
        sub = histograma[histograma["comportamento"] == comportamento].sort_values("classe_idx")
        if sub.empty:
            continue
        centro = (sub["classe_min"] + sub["classe_max"]) / 2
        cor, rgb = CORES_COMPORTAMENTO[comportamento]
        fig.add_trace(go.Scatter(
            x=centro, y=sub["n_amostras"].clip(lower=1),
            fill="tozeroy", mode="lines", name=comportamento.capitalize(),
            line=dict(color=cor, width=3, shape="spline"),
            fillcolor=f"rgba({rgb},0.18)",
            hovertemplate="%{y:,.0f} amostras perto de %{x:.2f} m/s²<extra>"
                          + comportamento.capitalize() + "</extra>",
        ))

    # Os dois rotulos saem para lados opostos da propria linha: ancorados no
    # topo, eles caiam por cima da legenda e o do firmware era cortado pela
    # borda direita da area de plotagem.
    marcas = [(limiar, "#7c3aed", f"limiar do firmware · {_br(limiar, 0)} m/s²", "top left")]
    if limiar_recal is not None:
        marcas.append((limiar_recal, "#6b7280",
                       f"recalibrado · {_br(limiar_recal)} m/s²", "top right"))
    for valor, cor, rotulo, posicao in marcas:
        fig.add_vline(x=valor, line_dash="dash", line_color=cor, line_width=2,
                      annotation_text=rotulo, annotation_position=posicao,
                      annotation_font=dict(size=11, color=cor))

    fig.update_layout(
        xaxis=dict(showgrid=False, title="Magnitude da aceleração (m/s²)"),
        yaxis=dict(showgrid=True, gridcolor="#e5e7eb", type="log",
                   title="Amostras (escala log)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0, title=""),
    )
    return _layout_plotly(fig, 380)


def render_tab_hardware(limiar: float):
    h1, h2, h3, h4 = st.columns(4)
    _hw_card(h1, "🔧", "Microcontrolador", "ESP32 Dual-Core")
    _hw_card(h2, "📈", "Acelerômetro MPU-6050", "Amostragem 50Hz")
    _hw_card(h3, "📡", "Transmissão (store-and-forward)", "Wi-Fi / MQTT QoS 1")
    _hw_card(h4, "🌱", "Eficiência Green IT", "Filtragem na borda (LittleFS)")

    st.markdown("<br>", unsafe_allow_html=True)

    histograma = carregar_histograma()
    limiar_recal = carregar_limiar_recalibrado(limiar)

    col_area, col_alertas = st.columns([1.5, 1])

    with col_area:
        _titulo_secao(
            "Assinatura inercial (magnitude da aceleração)",
            f"Densidade das 311 mil leituras do acelerômetro no {FONTE_CITACAO}, "
            "em escala log — a primeira classe concentra a maior parte das amostras "
            "e, no eixo linear, a cauda onde os limiares cortam fica invisível. "
            "A cauda da condução agressiva se estende muito além da normal, mas "
            "termina bem antes do limiar programado no firmware.",
        )
        if histograma is None:
            st.info(
                "Rode `python src/validacao_hardware.py` para gerar o histograma "
                "de amostras e esta figura aparece.", icon="ℹ️",
            )
        else:
            st.plotly_chart(
                fig_assinatura_inercial(histograma, limiar, limiar_recal),
                use_container_width=True,
            )

    with col_alertas:
        _titulo_secao(
            "O que o firmware registra, por faixa",
            "Faixas de magnitude do sinal medido — não classificação de segurado: "
            "o detector não decide perfil nem preço.",
        )
        faixas = (contagem_por_faixa(limiar, limiar_recal)
                  if limiar_recal is not None else None)
        if faixas is None or not faixas["n_total"]:
            st.info("Rode `python src/validacao_hardware.py` para gerar as contagens.",
                    icon="ℹ️")
        else:
            n_total = faixas["n_total"]
            st.markdown(
                _alert_box(
                    "alert-good", "✅", f"Sem evento (abaixo de {_br(limiar_recal)} m/s²)",
                    f"{_br(100 * faixas['n_abaixo_recal'] / n_total)}% das amostras "
                    f"({_int_br(faixas['n_abaixo_recal'])} de {_int_br(n_total)}). "
                    "Nenhum dos dois limiares dispara aqui.",
                )
                + _alert_box(
                    "alert-warn", "⚠️",
                    f"Zona intermediária ({_br(limiar_recal)} a {_br(limiar, 0)} m/s²)",
                    f"{_br(100 * faixas['n_entre'] / n_total, 2)}% das amostras "
                    f"({_int_br(faixas['n_entre'])}). Só seria registrada se o "
                    f"THRESH_MAG_MS2 fosse recalibrado para {_br(limiar_recal)}.",
                )
                + _alert_box(
                    "alert-bad", "⛔", f"Evento registrado (acima de {_br(limiar, 0)} m/s²)",
                    f"Apenas {_int_br(faixas['n_acima_vigente'])} amostra(s) em "
                    f"{_int_br(n_total)} cruzam o limiar vigente — a contagem de "
                    "eventos por minuto fica zerada em quase toda a base.",
                ),
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="nota-integridade">'
        "<b>O que esta aba é e o que não é.</b> É uma checagem de discriminação da "
        "regra de detecção que roda no firmware contra um dataset público de "
        "condução, rotulado por trajeto — não é medição de sonolência, não é "
        "precisão por evento e não tem poder estatístico de validação. A análise "
        "metodológica completa (régua da literatura, curva de sensibilidade, "
        "varredura de calibração e efeito pareado por motorista) continua em "
        "<code>dashboard/tab_validacao_hardware.py</code>. "
        f"Dados: {FONTE_CITACAO} — acelerômetro a 10Hz, rótulo por trajeto; os "
        "códigos D1–D6 são do próprio dataset e não identificam pessoas."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Aba 3 -- Integracao UBI: coligacao conceitual das duas camadas
# ---------------------------------------------------------------------------

def fig_radar_perfis(variaveis: list, medias: pd.DataFrame) -> go.Figure:
    eixos = [ROTULO_VARIAVEL.get(v, v) for v in variaveis]
    fig = go.Figure()
    for perfil in (2, 1):
        if perfil not in medias.index:
            continue
        valores = medias.loc[perfil, variaveis].tolist()
        cor = CORES_PERFIL[perfil]
        rgb = {1: "239,68,68", 2: "16,185,129"}[perfil]
        fig.add_trace(go.Scatterpolar(
            r=valores + valores[:1], theta=eixos + eixos[:1],
            fill="toself", name=NOME_CURTO[perfil],
            line=dict(color=cor), fillcolor=f"rgba({rgb},0.35)",
            hovertemplate="%{theta}: %{r:.0f}/100<extra>" + NOME_CURTO[perfil] + "</extra>",
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False,
                            gridcolor="#e5e7eb"),
            angularaxis=dict(gridcolor="#e5e7eb"),
            bgcolor="rgba(0,0,0,0)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
        margin=dict(l=60, r=60, t=30, b=30),
    )
    return _layout_plotly(fig, 400)


def render_tab_hibrido(normalizada, significancia, mapa_perfil):
    col_radar, col_texto = st.columns([1, 1])

    with col_radar:
        _titulo_secao(
            "Raio-X multidimensional do segurado",
            "As 6 variáveis que mais separam os perfis (menor p-valor nos testes de "
            "significância), na média normalizada Min-Max do próprio K-Means, "
            "reescalada a 0–100. São dados declaratórios/históricos da Parte A — "
            "nenhuma leitura de telemetria entra neste radar. O Perfil 2 aparece "
            "quase colado no centro porque pontua baixo justamente nas variáveis "
            "que mais discriminam: isso é o resultado, não falha da figura.",
        )
        if significancia is None or significancia.empty:
            st.info("Rode `python src/perfilamento.py` para gerar os testes de "
                    "significância.", icon="ℹ️")
        else:
            variaveis, medias = calcular_radar_perfis(normalizada, significancia, mapa_perfil)
            st.plotly_chart(fig_radar_perfis(variaveis, medias), use_container_width=True)

    with col_texto:
        st.markdown(
            '<div class="card-texto">'
            '<h4 style="color:#6d28d9;margin-top:0;">⚡ Matriz de transição dinâmica</h4>'
            '<p style="color:#4b5563;font-size:14.5px;line-height:1.6;">'
            "O diferencial previsto para esta arquitetura é transformar a fotografia "
            "estática da cotação em um <b>filme dinâmico</b>: a telemetria gerada na "
            "borda (ESP32) corrigiria as distorções da declaração inicial. As duas "
            "transições abaixo são o desenho da Seção 3.4.1."
            "</p>"
            '<div style="display:flex;flex-direction:column;gap:18px;margin-top:18px;">'
            '<div style="display:flex;gap:14px;align-items:flex-start;">'
            '<div style="background:#d1fae5;padding:9px;border-radius:8px;font-size:18px;">🛡️</div>'
            '<div><strong style="color:#065f46;display:block;margin-bottom:3px;">'
            "Referral → Desconto</strong>"
            '<span style="color:#4b5563;font-size:13.5px;">Cliente sem histórico '
            "(Perfil 3) aceita a instalação do dispositivo como condição de emissão; "
            "a conduta observada sustentaria a migração para o Perfil 2.</span></div></div>"
            '<div style="display:flex;gap:14px;align-items:flex-start;">'
            '<div style="background:#fee2e2;padding:9px;border-radius:8px;font-size:18px;">📉</div>'
            '<div><strong style="color:#991b1b;display:block;margin-bottom:3px;">'
            "Desconto → Agravo</strong>"
            '<span style="color:#4b5563;font-size:13.5px;">Frota cuja má gestão '
            "operacional não aparece na declaração (Perfil 2 aparente): eventos "
            "inerciais recorrentes justificariam a reclassificação para o Perfil 1, "
            "sempre com decisão humana no circuito.</span></div></div>"
            "</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="nota-integridade">'
        "<b>Não existe junção real entre as duas camadas nesta PoC.</b> As apólices "
        "são sintéticas e o dataset de condução é público: não há motorista, veículo "
        "ou vínculo contratual em comum entre eles, e nenhuma correlação foi calculada "
        "entre as duas bases. O radar acima usa exclusivamente dados da Parte A "
        "(segmentação); a matriz de transição descreve a arquitetura prevista no texto "
        "do TCC — token pseudônimo por frota, ingestão que só enxerga o token e "
        "reassociação restrita à seguradora (LGPD, minimização de dados) — e não um "
        "resultado medido. A classificação permanece apoio à decisão, com humano no "
        "circuito (LGPD, Art. 20)."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Dashboard Seguros IoT", page_icon="🛡️",
        layout="wide", initial_sidebar_state="collapsed",
    )
    aplicar_estilo()

    col_titulo, col_badge = st.columns([5, 1])
    with col_titulo:
        st.markdown(
            '<h1 style="color:#4c1d95;font-size:2.1rem;font-weight:800;margin-bottom:0;">'
            "🛡️ Dashboard Seguros IoT</h1>"
            '<p style="color:#6b7280;font-size:1.05rem;margin-top:5px;">'
            "Plataforma híbrida de validação de risco: K-Means (software) + "
            "UBI edge computing (hardware) &middot; Produto RCT Transportador</p>",
            unsafe_allow_html=True,
        )
    with col_badge:
        st.markdown(
            '<div style="text-align:right;padding-top:22px;">'
            '<span class="badge badge-violet">Dados sintéticos (PoC)</span></div>',
            unsafe_allow_html=True,
        )

    if not (DADOS_DIR / "matriz_normalizada_clusters.csv").exists():
        st.error(
            "Nenhum resultado de clusterização encontrado em data/processed/. "
            "Execute src/preprocessamento.py e src/clustering.py antes de abrir o dashboard."
        )
        return

    normalizada, original = carregar_dados()
    mapa_perfil = mapear_cluster_para_perfil(original)
    kpis_cluster = calcular_kpis_por_cluster(original, mapa_perfil)
    significancia = carregar_significancia()
    gerais = {
        "n_apolices": len(original),
        "premio_por_veiculo_medio": original["premio_por_veiculo"].mean(),
        "taxa_referral": original["referral_pendente"].mean(),
    }

    projecao, variancia_pct = projetar_pca(normalizada)
    projecao["perfil_numero"] = projecao["cluster"].map(mapa_perfil)
    projecao["perfil_curto"] = projecao["perfil_numero"].map(NOME_CURTO)

    aba_software, aba_hardware, aba_hibrido = st.tabs([
        "📊 Software (K-Means)",
        "⚙️ Hardware (IoT Edge)",
        "⚡ Integração UBI (Híbrido)",
    ])
    with aba_software:
        render_tab_software(projecao, kpis_cluster, gerais, variancia_pct)
    with aba_hardware:
        render_tab_hardware(THRESH_MAG_MS2)
    with aba_hibrido:
        render_tab_hibrido(normalizada, significancia, mapa_perfil)

    st.markdown(
        '<div style="text-align:center;color:#9ca3af;font-size:11.5px;margin-top:28px;">'
        "Erika Oliveira Silva &middot; Centro Universitário Senac Santo Amaro &middot; "
        "Segmentação validada por Silhouette, Davies-Bouldin e Calinski-Harabasz "
        "(ver src/clustering.py)</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
