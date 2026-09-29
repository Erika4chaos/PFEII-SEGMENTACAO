"""
Dashboard Seguros IoT -- plataforma hibrida de validacao de risco
(PFE II, Etapa 4 / Secao 3.5.4).

Tres abas, na estrutura do painel desenhado pela autora:
  * Software (K-Means)     -- Segmentacao da carteira (Parte A)
  * Hardware (IoT Edge)    -- Validacao da assinatura inercial (Parte B)
  * Integracao UBI         -- Coligacao conceitual das duas camadas

Todo numero em tela vem do pipeline real (data/processed/*.csv). O mockup
que serviu de referencia visual trazia valores de exemplo (2.489 apolices,
escala 0-100 arbitraria, "amostragem 10Hz") -- nenhum deles foi copiado: a
taxa real do firmware e 50Hz (firmware/esp32/include/config.h) e todas as
estatisticas saem dos CSVs gerados por src/.

A analise metodologica longa da camada de hardware -- as nove secoes de
dashboard/tab_validacao_hardware.py (regua da literatura, curva de
sensibilidade, varredura de calibracao, efeito pareado por motorista,
tabela de cobertura) -- continua no repositorio e volta a aparecer com uma
`tab_validacao_hardware.render(df, limiar)` -- passou a ser chamada dentro de
um expander na propria aba de hardware. Recolhida, ela nao compete com a
leitura curta com que o painel abre (motivo pelo qual havia sido retirada) e
deixa de ser codigo morto no repositorio.

Uso:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

RAIZ = Path(__file__).resolve().parent.parent
sys.path.append(str(RAIZ / "src"))
sys.path.append(str(RAIZ))
from preprocessamento import COLUNAS_19, COLUNAS_MODELO  # noqa: E402
from validacao_hardware import THRESH_MAG_MS2  # noqa: E402
from validacao_hardware import ESPECIFICIDADE_MINIMA, _melhor_par  # noqa: E402
from validacao_hardware import (  # noqa: E402
    _perfil_deteccao, _taxa_eventos, listar_trajetos,
)
from dashboard.tab_validacao_hardware import (  # noqa: E402
    carregar, carregar_grade, render as render_validacao_longa,
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

CARTEIRA_COMPLETA = "Carteira completa"

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
# sendo a chave em todo o pipeline; isto e so o texto que a banca le nos graficos.
# Variaveis sem ordem intrinseca: a media delas nao e interpretavel como
# "mais" ou "menos", entao nao entram em figuras de magnitude.
VARIAVEIS_NOMINAIS = {"tipo_sinistro_predominante"}

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
    elif abs(valor) >= 100_000:
        texto = f"R$ {valor / 1_000:.0f} mil"
    elif abs(valor) >= 1_000:
        # Uma casa abaixo de 100 mil: com arredondamento inteiro, premios de
        # R$ 2,4 mil e R$ 2,6 mil apareciam iguais ("R$ 2 mil") na tabela.
        texto = f"R$ {valor / 1_000:.1f} mil"
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
def carregar_validacao_k():
    """Curva de indices internos por k, como src/clustering.py a escreveu."""
    caminho = DADOS_DIR / "validacao_k.csv"
    return pd.read_csv(caminho) if caminho.exists() else None


@st.cache_data
def carregar_sensibilidade():
    """Faixa de Silhouette e concordancia entre realizacoes da base
    sintetica, como src/sensibilidade_sementes.py a escreveu."""
    caminho = DADOS_DIR / "sensibilidade_sementes.csv"
    return pd.read_csv(caminho) if caminho.exists() else None


@st.cache_data
def carregar_significancia():
    caminho = DADOS_DIR / "perfis_testes_significancia.csv"
    return pd.read_csv(caminho) if caminho.exists() else None


@st.cache_data
def carregar_resumo_uah():
    """Resumo por trajeto do UAH-DriveSet, como src/validacao_hardware.py o
    escreveu (uma linha por trajeto, com magnitude, velocidade e variacao de
    curso). E a base do painel de conduta -- Parte B, sem nenhuma relacao com
    as apolices da Parte A."""
    caminho = DADOS_DIR / "validacao_hardware_uah_driveset.csv"
    return pd.read_csv(caminho) if caminho.exists() else None


# Eixos do painel de conduta: nome da coluna -> rotulo. `eventos_por_min` fica
# de fora de proposito -- no limiar vigente ela e zero em quase todos os
# trajetos e entraria como um eixo colado no centro, sem informacao.
METRICAS_CONDUTA = {
    "magnitude_media_ms2": "Magnitude média",
    "magnitude_maxima_ms2": "Pico de magnitude",
    "velocidade_media_kmh": "Velocidade média",
    "velocidade_maxima_kmh": "Velocidade máxima",
    "var_curso_media_abs": "Variação de curso",
}


def calcular_conduta_normalizada(resumo: pd.DataFrame):
    """Media por rotulo comportamental das metricas de conduta, cada uma
    normalizada Min-Max sobre os trajetos disponiveis e reescalada a 0-100.

    Mesma receita do painel de perfis (normalizar, depois agrupar), o que
    mantem as duas figuras lendo na mesma regua -- posicao relativa dentro da
    propria base, nunca comparacao numerica entre as duas bases."""
    dados = resumo.copy()
    for coluna in METRICAS_CONDUTA:
        minimo, maximo = dados[coluna].min(), dados[coluna].max()
        dados[coluna] = (100 * (dados[coluna] - minimo) / (maximo - minimo)
                         if maximo > minimo else 0.0)
    medias = dados.groupby("comportamento")[list(METRICAS_CONDUTA)].mean()
    return list(METRICAS_CONDUTA.values()), medias.rename(columns=METRICAS_CONDUTA)


def projetar_pca(normalizada: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    pca = PCA(n_components=2, random_state=42)
    # Projeta as MESMAS colunas que alimentaram o K-Means. Usar a lista do
    # Quadro 2 aqui mostraria uma projecao de um espaco diferente daquele em
    # que os clusters foram formados.
    componentes = pca.fit_transform(normalizada[COLUNAS_MODELO])
    projecao = normalizada[["numeroApolice", "cluster"]].copy()
    projecao["PCA1"] = componentes[:, 0]
    projecao["PCA2"] = componentes[:, 1]
    return projecao, pca.explained_variance_ratio_[:2].tolist()


def numero_perfil(medias: pd.Series) -> int:
    """Confronta as medias do cluster no espaco original com os tres
    perfis-alvo da Secao 3.4, para apresentacao em linguagem de negocio."""
    if medias["referral_pendente"] > 0.5:
        return 3
    if medias["qt_coberturas_ativas"] >= 5 and medias["agravo_aplicado"] < 0.2:
        return 2
    return 1


def mapear_cluster_para_perfil(original: pd.DataFrame) -> dict:
    """Associa cada cluster a um dos tres perfis-alvo.

    A correspondencia precisa ser biunivoca: se duas particoes caissem no
    mesmo perfil, a tabela de perfis, os KPIs e o filtro lateral passariam a
    somar dois clusters sob um rotulo so, sem nenhum aviso na tela. Como as
    regras de `numero_perfil` sao heuristicas sobre medias, isso pode
    acontecer se a base mudar -- por isso a checagem e explicita.
    """
    medias = original.groupby("cluster")[COLUNAS_19].mean()
    mapa = medias.apply(numero_perfil, axis=1).to_dict()

    perfis = list(mapa.values())
    if len(set(perfis)) != len(perfis):
        duplicados = sorted({p for p in perfis if perfis.count(p) > 1})
        raise ValueError(
            "Mais de um cluster foi classificado no mesmo perfil-alvo "
            f"(perfis duplicados: {duplicados}; mapa: {mapa}). As regras de "
            "numero_perfil() precisam ser revistas para esta base -- sem isso, "
            "a tabela de perfis e os KPIs somariam clusters distintos sob o "
            "mesmo rotulo."
        )
    return mapa


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


def calcular_discriminantes_por_perfil(normalizada: pd.DataFrame, significancia: pd.DataFrame,
                          mapa_perfil: dict, top_n: int = 6):
    """Media por perfil das `top_n` variaveis mais discriminantes, na matriz
    Min-Max 0-1 que alimentou o K-Means, escalada a 0-100 so para leitura.

    O ranking usa TAMANHO DE EFEITO (eta2 / V de Cramer), nao p-valor: como os
    clusters foram definidos por essas mesmas variaveis, o p-valor aqui e
    circular e satura em valores minusculos para quase todas elas, o que o
    torna inutil para ordenar. O tamanho de efeito continua discriminando.
    """
    coluna_ranking = ("tamanho_efeito" if "tamanho_efeito" in significancia.columns
                      else "p_valor")
    ordenada = significancia.sort_values(
        coluna_ranking, ascending=(coluna_ranking == "p_valor"))
    # Variaveis NOMINAIS ficam de fora desta figura. `tipo_sinistro_predominante`
    # e um codigo (1=tombamento, 2=incendio, 3=terceiros): a media desse codigo
    # nao significa nada, e numa barra de 0 a 100 ela seria lida como "mais" ou
    # "menos" de alguma coisa, que e exatamente a leitura errada. O V de Cramer
    # dela continua no ranking de discriminancia do perfilamento, onde a
    # associacao e medida sem pressupor ordem.
    variaveis = [v for v in ordenada["variavel"] if v not in VARIAVEIS_NOMINAIS]
    variaveis = variaveis[:top_n]
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
            /* Os rotulos das abas sao frases, nao palavras soltas: sem estes
               dois, o texto da aba selecionada estoura a pilula e fica
               cortado na borda direita. */
            white-space: nowrap;
            width: auto;
        }
        .stTabs [aria-selected="true"] {
            background-color: #7c3aed;
            color: white !important;
            border-color: #7c3aed;
            /* Repetido aqui porque o Streamlit aplica estilo proprio a aba
               ativa: sem isto o rotulo encosta na borda direita da pilula. */
            padding: 0.5rem 1.5rem;
            white-space: nowrap;
            width: auto;
            flex: none;
        }

        /* O container das abas nao pode comprimir os rotulos. */
        .stTabs [data-baseweb="tab-list"] > * { flex: none; }
        .stTabs [data-baseweb="tab"] > div,
        .stTabs [data-baseweb="tab"] p { white-space: nowrap; }

        /* Filtro por perfil (st.pills) no tom das abas. Os dois seletores
           cobrem as duas formas que o Streamlit ja usou para marcar a pilula
           selecionada (kind="pillsActive" e aria-checked). */
        button[kind="pills"], button[kind="pillsActive"] {
            border-radius: 999px !important;
            border: 1px solid #ddd6fe !important;
            color: #6d28d9 !important;
            background-color: #ffffff !important;
            font-weight: 600;
        }
        button[kind="pillsActive"], button[kind="pills"][aria-checked="true"] {
            background-color: #7c3aed !important;
            border-color: #7c3aed !important;
            color: #ffffff !important;
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
            padding: 20px; text-align: center; height: 100%; min-height: 210px;
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


def _hw_card(coluna, icone: str, titulo: str, valor: str, nota: str = ""):
    complemento = (f'<p class="kpi-sub" style="margin:4px 0 0 0;">{nota}</p>'
                   if nota else "")
    with coluna:
        st.markdown(
            f'<div class="hw-card"><div class="hw-icon">{icone}</div>'
            f'<p class="hw-title">{titulo}</p>'
            f'<p class="hw-value">{valor}</p>{complemento}</div>',
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

def render_escolha_de_k():
    """Declara a divergencia entre os indices internos e o k adotado.

    O rodape do painel afirmava que a segmentacao era "validada por Silhouette,
    Davies-Bouldin e Calinski-Harabasz". Isso e literalmente verdadeiro e
    materialmente enganoso: os tres indices favorecem k=2, e o modelo adota
    k=3. Omitir a divergencia num painel que sera lido pela banca e pela area
    tecnica seria esconder justamente o ponto que exige justificativa -- e que
    o trabalho sabe justificar. Os numeros abaixo sao lidos dos CSVs do
    pipeline, nunca fixos no codigo.
    """
    curva = carregar_validacao_k()
    if curva is None or curva.empty:
        st.info("Rode `python src/clustering.py` para gerar a curva de validação de k.",
                icon="ℹ️")
        return

    k_sil = int(curva.loc[curva["silhouette"].idxmax(), "k"])
    k_db = int(curva.loc[curva["davies_bouldin"].idxmin(), "k"])
    k_ch = int(curva.loc[curva["calinski_harabasz"].idxmax(), "k"])
    linha3 = curva[curva["k"] == 3]
    sil3 = float(linha3["silhouette"].iloc[0]) if not linha3.empty else float("nan")

    indices = sorted({k_sil, k_db, k_ch})
    texto_indices = (f"k = {indices[0]}" if len(indices) == 1
                     else "k = " + ", ".join(str(i) for i in indices))

    _titulo_secao(
        "Os índices apontam k=2; o modelo adota k=3, e isso é uma decisão declarada",
        f"Silhueta, Davies-Bouldin e Calinski-Harabasz favorecem {texto_indices}. "
        f"O Método do Cotovelo aponta k=3, e é esse o valor adotado. A Silhueta da "
        f"solução final é {_br(sil3, 3)}.",
    )
    st.markdown(
        '<div class="nota-integridade">'
        "<b>Por que k=3 mesmo assim.</b> Em k=2 o perfil de <i>referral</i> não vira "
        "grupo próprio: suas apólices são redistribuídas entre os outros dois. Em k=3 "
        "ele reaparece íntegro. Os índices internos medem compacidade geométrica, não "
        "utilidade de negócio — e o perfil de <i>referral</i> se sobrepõe aos demais "
        "justamente por ser o segmento de informação incompleta, que é a característica "
        "que o define. Adotar k=2 seria concluir que o segmento não existe porque é "
        "difícil de separar."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    sens = carregar_sensibilidade()
    if sens is None or sens.empty:
        st.caption("Rode `python src/sensibilidade_sementes.py` para medir a "
                   "estabilidade entre realizações da base.")
        return

    _titulo_secao(
        f"O resultado não depende de uma realização da base ({len(sens)} sementes)",
        "Repetindo o pipeline inteiro — geração, pré-processamento, K-Means e "
        "pareamento — sobre sementes distintas da base sintética. É a faixa, e não o "
        "valor de uma semente, que deve ser citada como resultado.",
    )
    c1, c2 = st.columns(2)
    _kpi_card(c1, "📐", "Silhueta (mediana e faixa)",
              _br(sens["silhouette"].median(), 3),
              f"entre {_br(sens['silhouette'].min(), 3)} e {_br(sens['silhouette'].max(), 3)}")
    _kpi_card(c2, "🎯", "Concordância (mediana e faixa)",
              f"{_br(100 * sens['concordancia'].median())}%",
              f"entre {_br(100 * sens['concordancia'].min())}% e "
              f"{_br(100 * sens['concordancia'].max())}%")
    st.caption("A concordância mede se o pipeline reencontra os perfis usados na "
               "geração da base sintética. É teste de sanidade do pipeline, não "
               "validação do método sobre dados reais.")


def render_tab_software(projecao, kpis_cluster, gerais, variancia_pc):
    variancia_pct = sum(variancia_pc)
    # Filtro por perfil: os quatro KPIs, a nuvem do PCA e a linha destacada da
    # tabela respondem juntos a esta selecao, para analisar um perfil de cada vez.
    perfis_disponiveis = sorted(int(n) for n in kpis_cluster["perfil_numero"])
    opcoes = [CARTEIRA_COMPLETA] + [NOME_CURTO[n] for n in perfis_disponiveis]
    escolha = st.pills("Analisar", opcoes, default=CARTEIRA_COMPLETA,
                       key="filtro_perfil") or CARTEIRA_COMPLETA
    perfil_sel = next((n for n in perfis_disponiveis if NOME_CURTO[n] == escolha), None)

    if perfil_sel is None:
        linha_p1 = kpis_cluster[kpis_cluster["perfil_numero"] == 1]
        n_base, rotulo_base = gerais["n_apolices"], "Base analisada"
        premio = gerais["premio_por_veiculo_medio"]
        referral = gerais["taxa_referral"]
        custo = float(linha_p1["valor_pago_historico_medio"].iloc[0]) if not linha_p1.empty else 0.0
        rotulo_custo = "Custo pago médio (Perfil 1)"
    else:
        linha = kpis_cluster[kpis_cluster["perfil_numero"] == perfil_sel].iloc[0]
        n_base, rotulo_base = linha["n_apolices"], f"Apólices do Perfil {perfil_sel}"
        premio = linha["premio_por_veiculo_medio"]
        referral = linha["taxa_referral"]
        custo = linha["valor_pago_historico_medio"]
        rotulo_custo = f"Custo pago médio (Perfil {perfil_sel})"

    c1, c2, c3, c4 = st.columns(4)
    _kpi_card(c1, "👥", rotulo_base, _int_br(n_base), "apólices")
    _kpi_card(c2, "💰", "Prêmio médio/veículo", _fmt_brl_compacto(premio), "por ano")
    _kpi_card(c3, "⏳", "Taxa de referral", f"{_br(100 * referral)}%")
    # `valor_pago_historico` e a SOMA do que ja foi pago em sinistros da apolice
    # (frota inteira, historico acumulado -- src/gerar_base_sintetica.py), nao o
    # custo de um sinistro. O rotulo antigo "/sinistro" o fazia parecer dezenas
    # de vezes o LMI por veiculo.
    _kpi_card(c4, "⚠️", rotulo_custo, _fmt_brl_compacto(custo), "/apólice, acumulado")

    st.markdown("<br>", unsafe_allow_html=True)

    # A tabela vem primeiro e sozinha na largura total: ela e a resposta de
    # negocio, e quem abre o painel abre por ela. A projecao PCA e metodologia
    # -- mostra COMO os grupos se separam, nao QUAIS sao --, entao desceu para
    # um bloco recolhido logo abaixo, disponivel para quem quiser auditar sem
    # competir com a leitura principal.
    _titulo_secao(
        "Três perfis, três tratamentos comerciais",
        "Médias no espaço original de cada perfil. A coluna de ação via hardware "
        "é a regra de negócio da Seção 3.4 (o dispositivo como agravo, "
        "fidelização ou condição de emissão), não um valor calculado.",
    )
    linhas = ""
    for _, linha in kpis_cluster.iterrows():
        perfil = int(linha["perfil_numero"])
        destaque = (' style="background-color:#f5f3ff;box-shadow:inset 4px 0 0 #7c3aed;"'
                    if perfil == perfil_sel else "")
        linhas += (
            f"<tr{destaque}>"
            f'<td style="font-weight:600;color:#1f2937;">{NOME_CURTO[perfil]}</td>'
            f'<td>{_int_br(linha["n_apolices"])} apólices</td>'
            f'<td>{_fmt_brl_compacto(linha["lmi_por_veiculo_medio"])}</td>'
            f'<td>{_fmt_brl_compacto(linha["premio_por_veiculo_medio"])}</td>'
            f'<td style="font-weight:600;color:#1f2937;">'
            f'{_fmt_brl_compacto(linha["valor_pago_historico_medio"])}</td>'
            f'<td><span class="badge {CLASSE_BADGE[perfil]}">{ACAO_HARDWARE[perfil]}</span></td>'
            "</tr>"
        )
    st.markdown(
        '<table class="styled-table"><thead><tr>'
        "<th>Perfil identificado</th><th>Volumetria</th><th>LMI médio/veíc.</th>"
        "<th>Prêmio médio/veíc.</th><th>Custo pago/apólice*</th><th>Ação via hardware</th>"
        "</tr></thead><tbody>" + linhas + "</tbody></table>"
        '<p class="secao-sub" style="margin-top:8px;">* Soma histórica do que foi '
        "pago em sinistros por apólice (frota inteira, todos os anos declarados) — "
        "não é o custo de um sinistro, por isso pode superar o LMI por veículo.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Como o modelo foi calibrado e o quanto o resultado é estável (metodologia)"):
        render_escolha_de_k()
        st.markdown("<br>", unsafe_allow_html=True)
        _titulo_secao(
            "Os grupos se separam no espaço do modelo",
            f"Redução das {len(COLUNAS_MODELO)} colunas que alimentam o modelo "
            f"(as {len(COLUNAS_19)} do Quadro 2, menos a nominal, mais as 3 indicadoras) "
            f"para visualização espacial dos 3 clusters. Os dois componentes "
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
        # Com um perfil escolhido, os outros dois ficam de fundo em vez de sumir:
        # a posicao relativa das nuvens e metade da leitura da figura.
        if perfil_sel is not None:
            for traco in fig.data:
                traco.marker.opacity = 0.9 if traco.name == NOME_CURTO[perfil_sel] else 0.12
        # Eixos visiveis so com o nome do componente e a variancia que ele
        # explica; os valores dos componentes nao tem unidade e ficam ocultos.
        eixo_pc = dict(showgrid=False, zeroline=False, showticklabels=False,
                       title_font=dict(size=12, color="#6b7280"))
        fig.update_layout(
            xaxis=dict(eixo_pc, title=f"PC1 ({_br(100 * variancia_pc[0])}% da variância)"),
            yaxis=dict(eixo_pc, title=f"PC2 ({_br(100 * variancia_pc[1])}% da variância)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.0,
                        xanchor="left", x=0, title=""),
        )
        fig = _layout_plotly(fig, 360)
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Aba 2 -- Hardware (IoT Edge): assinatura inercial medida na borda (Parte B)
# ---------------------------------------------------------------------------

UAH_DIR = RAIZ / "data" / "raw" / "uah-driveset"


def ponto_operacao_recomendado() -> dict | None:
    """Par (limiar, taxa minima) que a varredura de calibracao recomenda --
    `_melhor_par` sobre a grade que src/validacao_hardware.py gerou, com o
    mesmo piso de especificidade do pipeline. E a UNICA fonte do limiar
    recalibrado nesta aba: o grafico e os cartoes leem daqui, para nao
    voltarem a mostrar dois numeros diferentes (2,9 no grafico, que era o
    p99,9 da conducao normal, e 2,3 nos cartoes)."""
    grade = carregar_grade()
    if grade is None:
        return None
    candidatos = grade[grade["especificidade"] >= ESPECIFICIDADE_MINIMA]
    return _melhor_par(candidatos if not candidatos.empty else grade).to_dict()


@st.cache_data
def carregar_trajetos_classificados(limiar_rec: float, taxa_min: float) -> pd.DataFrame | None:
    """Um registro por trajeto: rotulo, pico, eventos/min no limiar
    recomendado e se a regra o sinaliza.

    A taxa por trajeto nao e reimplementada aqui: sai de `_perfil_deteccao` e
    `_taxa_eventos`, as mesmas funcoes que montam a grade de calibracao, entao
    a contagem de sinalizados bate com a da grade por construcao."""
    if not UAH_DIR.exists():
        return None
    linhas = []
    for pasta in listar_trajetos(UAH_DIR):
        perfil = _perfil_deteccao(pasta)
        taxa = _taxa_eventos(perfil, limiar_rec)
        linhas.append({
            "trajeto": perfil["trajeto"],
            "motorista": perfil["motorista"],
            "comportamento": perfil["comportamento"],
            "pico_ms2": float(perfil["magnitude_ordenada"][-1]),
            "eventos_min": taxa,
            "sinalizado": taxa > taxa_min,
        })
    return pd.DataFrame(linhas)


def fig_pico_por_trajeto(trajetos: pd.DataFrame, limiar: float,
                         limiar_rec: float, taxa_min: float) -> go.Figure:
    """Um ponto por trajeto (n=40), no pico de magnitude, com os dois limiares.

    Substitui a densidade das 311 mil amostras: os cartoes ao lado classificam
    TRAJETOS, e uma figura por amostra nao sustentava essa leitura. Ponto cheio
    = trajeto sinalizado pela regra recomendada; ponto vazado = nao sinalizado.
    O eixo vai alem de 6 m/s² para que o limiar do firmware fique dentro da
    area de plotagem, com a cauda inteira visivel."""
    fig = go.Figure()
    ordem_y = list(reversed(ORDEM_COMPORTAMENTO))  # normal no topo
    rng = np.random.default_rng(7)
    for comportamento in ORDEM_COMPORTAMENTO:
        sub = trajetos[trajetos["comportamento"] == comportamento]
        if sub.empty:
            continue
        cor, _ = CORES_COMPORTAMENTO[comportamento]
        y_base = ordem_y.index(comportamento)
        jitter = rng.uniform(-0.22, 0.22, len(sub))
        for sinalizado, simbolo, legenda in ((True, "circle", "sinalizado"),
                                             (False, "circle-open", "não sinalizado")):
            parte = sub[sub["sinalizado"] == sinalizado]
            if parte.empty:
                continue
            fig.add_trace(go.Scatter(
                x=parte["pico_ms2"], y=y_base + jitter[sub["sinalizado"].to_numpy() == sinalizado],
                mode="markers", showlegend=False,
                marker=dict(symbol=simbolo, size=12, color=cor,
                            line=dict(width=2, color=cor)),
                customdata=np.stack([parte["motorista"], parte["eventos_min"]], axis=-1),
                hovertemplate=(
                    "Motorista %{customdata[0]} · pico %{x:.2f} m/s²<br>"
                    "%{customdata[1]:.2f} eventos/min acima de "
                    f"{_br(limiar_rec)} m/s² → {legenda}"
                    f"<extra>{comportamento.capitalize()}</extra>"
                ),
            ))
        n_sin, n_tot = int(sub["sinalizado"].sum()), len(sub)
        fig.add_annotation(
            x=1.0, xref="paper", xanchor="left", y=y_base, showarrow=False,
            text=f"<b>{n_sin}/{n_tot}</b> sinalizados", font=dict(size=11, color="#4b5563"),
        )

    marcas = [
        (limiar_rec, "#6b7280", f"recalibrado · {_br(limiar_rec)} m/s²"),
        (limiar, "#7c3aed", f"firmware atual · {_br(limiar, 0)} m/s²"),
    ]
    for valor, cor, rotulo in marcas:
        fig.add_vline(x=valor, line_dash="dash", line_color=cor, line_width=2,
                      annotation_text=rotulo, annotation_position="top",
                      annotation_font=dict(size=11, color=cor))

    # Legenda da forma do ponto (a cor ja esta nomeada no eixo Y).
    for simbolo, texto in (("circle", "sinalizado pela regra"),
                           ("circle-open", "não sinalizado")):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=texto,
            marker=dict(symbol=simbolo, size=11, color="#6b7280",
                        line=dict(width=2, color="#6b7280")),
        ))

    x_max = max(7.0, float(trajetos["pico_ms2"].max()) + 0.5)
    fig.update_layout(
        xaxis=dict(range=[0, x_max], showgrid=True, gridcolor="#eef0f3", zeroline=False,
                   title="Pico de magnitude do trajeto (m/s²)  ·  1 g ≈ 9,8 m/s²",
                   dtick=1),
        yaxis=dict(tickvals=list(range(len(ordem_y))),
                   ticktext=[f"<b>{c.capitalize()}</b> (n={int((trajetos['comportamento'] == c).sum())})"
                             for c in ordem_y],
                   range=[-0.6, len(ordem_y) - 0.4], showgrid=False, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="left", x=0, title=""),
    )
    fig = _layout_plotly(fig, 360)
    fig.update_layout(margin=dict(l=0, r=110, t=40, b=0))
    return fig


def _matriz_confusao_html(rec: dict) -> str:
    """Matriz rotulo real x saida da regra (3 x 2), com as contagens da grade."""
    linhas = [
        ("Normal", "normal", rec["n_normal"], rec["normal_sinalizados"]),
        ("Sonolenta", "sonolenta", rec["n_sonolenta"], rec["sonolenta_sinalizados"]),
        ("Agressiva", "agressiva", rec["n_agressiva"], rec["agressiva_sinalizados"]),
    ]
    corpo = ""
    for nome, chave, n, sin in linhas:
        n, sin = int(n), int(sin)
        cor, rgb = CORES_COMPORTAMENTO[chave]
        nao = n - sin
        # Intensidade do fundo proporcional a fracao da linha: a matriz se le
        # por linha (cada rotulo soma 100%).
        def celula(v):
            alfa = 0.08 + 0.55 * (v / n if n else 0)
            return (f'<td style="text-align:center;background:rgba({rgb},{alfa:.2f});'
                    f'font-weight:700;color:#1f2937;">{v}'
                    f'<span style="font-weight:400;color:#6b7280;font-size:11.5px;">'
                    f" &nbsp;({_br(100 * v / n, 0) if n else '–'}%)</span></td>")
        corpo += (f'<tr><td style="font-weight:600;"><span style="color:{cor};">●</span> '
                  f"{nome} <span style='color:#9ca3af;font-weight:400;'>(n={n})</span></td>"
                  f"{celula(nao)}{celula(sin)}</tr>")
    return (
        '<table class="styled-table" style="margin-bottom:14px;"><thead><tr>'
        "<th>Rótulo do dataset</th><th style='text-align:center;'>Não sinalizado</th>"
        "<th style='text-align:center;'>Sinalizado</th></tr></thead><tbody>"
        + corpo + "</tbody></table>"
    )


def render_classificacao_conduta(limiar: float, rec: dict | None):
    """Resultado por TRAJETO, no ponto de operacao que a varredura recomenda.

    A matriz rotulo x saida e a leitura principal; os tres cartoes viraram
    texto de apoio curto, um por linha da matriz."""
    _titulo_secao(
        "Classificação de conduta (o que a regra sinaliza)",
        "Veredito por trajeto, não por amostra: o detector não decide perfil "
        "nem preço — a leitura abaixo é o quanto a regra separa condução "
        "agressiva de condução normal.",
    )
    if rec is None:
        st.info(
            "Rode `python src/validacao_hardware.py` para gerar a grade de "
            "calibração e esta classificação aparece.", icon="ℹ️",
        )
        return

    n_agr, agr_sin = int(rec["n_agressiva"]), int(rec["agressiva_sinalizados"])
    n_nor, nor_sin = int(rec["n_normal"]), int(rec["normal_sinalizados"])
    n_son, son_sin = int(rec["n_sonolenta"]), int(rec["sonolenta_sinalizados"])

    st.markdown(
        _matriz_confusao_html(rec)
        + _alert_box(
            "alert-good", "✅", "Conduta preservada",
            f"{n_nor - nor_sin} dos {n_nor} trajetos normais não disparam "
            f"({_br(100 * rec['especificidade'], 0)}% de especificidade).",
        )
        + _alert_box(
            "alert-warn", "⚠️", "Sonolência → análise humana",
            f"{son_sin} dos {n_son} sonolentos disparam: sem decomposição por eixo, "
            "correção tardia de volante parece frenagem forte. Reavaliação humana, "
            "não agravo automático.",
        )
        + _alert_box(
            "alert-bad", "⛔", "Conduta sinalizada",
            f"{agr_sin} dos {n_agr} agressivos disparam "
            f"({_br(100 * rec['sensibilidade'], 0)}% de sensibilidade). "
            f"Índice de Youden {_br(rec['youden'], 2)} (0 = não separa nada).",
        ),
        unsafe_allow_html=True,
    )

    trajetos = carregar()
    acima_vigente = int((trajetos["pico_ms2"] >= limiar).sum())
    st.markdown(
        '<div class="nota-integridade" style="margin-top:6px;font-size:12.5px;">'
        f"<b>Regra usada:</b> limiar de {_br(rec['limiar_ms2'])} m/s² com mais de "
        f"{_br(rec['taxa_min_eventos_min'])} evento/min — o ponto de operação que a "
        "varredura de calibração recomenda. "
        f"<b>O limiar de {_br(limiar, 0)} m/s² que está hoje no firmware é cruzado "
        f"por apenas {acima_vigente} dos {len(trajetos)} trajetos</b> — com ele a "
        "classificação acima deixa de existir, e é isso que justifica a recalibração."
        "</div>",
        unsafe_allow_html=True,
    )


def render_tab_hardware(limiar: float):
    h1, h2, h3, h4 = st.columns(4)
    _hw_card(h1, "🔧", "Microcontrolador", "ESP32 Dual-Core")
    _hw_card(h2, "📈", "Acelerômetro MPU-6050", "Amostragem 50 Hz",
             "dataset desta aba: 10 Hz")
    _hw_card(h3, "📡", "Transmissão (store-and-forward)", "Wi-Fi / MQTT QoS 1")
    _hw_card(h4, "🌱", "Eficiência Green IT", "Filtragem na borda (LittleFS)")

    st.markdown("<br>", unsafe_allow_html=True)

    rec = ponto_operacao_recomendado()
    col_area, col_alertas = st.columns([1.5, 1])

    trajetos = (carregar_trajetos_classificados(float(rec["limiar_ms2"]),
                                                float(rec["taxa_min_eventos_min"]))
                if rec is not None else None)
    n_traj = len(trajetos) if trajetos is not None else 0

    with col_area:
        _titulo_secao(
            "Pico de aceleração por trajeto",
            f"Cada ponto é um dos {n_traj} trajetos do {FONTE_CITACAO}, na maior magnitude "
            "que ele registrou. Ponto cheio: a regra recomendada sinaliza o trajeto; "
            "vazado: não sinaliza — cruzar o limiar recalibrado uma vez não basta, a regra "
            "pede leituras acima dele com frequência (ver a regra usada, ao lado). "
            "À direita do limiar atual do firmware quase não há pontos: com ele, "
            "praticamente nenhum trajeto dispararia.",
        )
        if trajetos is None or trajetos.empty:
            st.info(
                "Rode `python src/validacao_hardware.py` (com o UAH-DriveSet em "
                "data/raw/uah-driveset/) para esta figura aparecer.", icon="ℹ️",
            )
        else:
            st.plotly_chart(
                fig_pico_por_trajeto(trajetos, limiar, float(rec["limiar_ms2"]),
                                     float(rec["taxa_min_eventos_min"])),
                use_container_width=True,
            )
            st.markdown(
                '<p class="secao-sub">Por que a sonolenta tem picos tão altos quanto a '
                "agressiva: a magnitude sem decomposição por eixo registra a correção "
                "tardia de trajetória (desvio brusco de volante) igual a uma frenagem "
                "forte. Não é erro de pipeline — é limite do sensor usado só por "
                "magnitude.</p>",
                unsafe_allow_html=True,
            )

    with col_alertas:
        render_classificacao_conduta(limiar, rec)

    st.markdown(
        '<div class="nota-integridade">'
        "<b>O que esta aba é e o que não é.</b> É uma checagem de discriminação da "
        "regra de detecção que roda no firmware contra um dataset público de "
        "condução, rotulado por trajeto — não é medição de sonolência, não é "
        "precisão por evento e não tem poder estatístico de validação. A análise "
        "metodológica completa (régua da literatura, curva de sensibilidade, "
        "varredura de calibração e efeito pareado por motorista) está no bloco "
        "recolhido abaixo. "
        f"Dados: {FONTE_CITACAO} — acelerômetro a 10 Hz (o firmware amostra a 50 Hz), "
        "rótulo por trajeto; os códigos D1–D6 são do próprio dataset e não "
        "identificam pessoas."
        "</div>",
        unsafe_allow_html=True,
    )

    # A analise metodologica longa vivia no repositorio sem nenhuma chamada:
    # existia, ninguem via, e o rodape apenas informava o caminho do arquivo.
    # Recolhida num expander ela para de competir com a leitura curta da aba
    # -- que foi o motivo de te-la retirado -- e volta a ser auditavel por quem
    # quiser conferir a calibracao do limiar.
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Análise metodológica completa da calibração do limiar"):
        # carregar() levanta FileNotFoundError quando o resumo do UAH ainda nao
        # foi gerado -- e nesse caso a aba inteira quebrava, nao so o expander.
        try:
            resumo_trajetos = carregar()
        except (FileNotFoundError, OSError):
            resumo_trajetos = None
        if resumo_trajetos is None or resumo_trajetos.empty:
            st.info("Rode `python src/validacao_hardware.py` para gerar o resumo "
                    "por trajeto do UAH-DriveSet.", icon="ℹ️")
        else:
            render_validacao_longa(resumo_trajetos, limiar)


# ---------------------------------------------------------------------------
# Aba 3 -- Integracao UBI: coligacao conceitual das duas camadas
# ---------------------------------------------------------------------------

# Barras horizontais em vez de radar. O radar e ruim exatamente para o que
# estas figuras precisam fazer: comparar magnitudes. A area de um poligono
# cresce com o quadrado do raio, o que exagera diferencas grandes e esconde as
# pequenas; a ordem dos eixos e arbitraria e muda a forma sem mudar o dado; e
# a leitura depende de comparar distancias radiais em angulos diferentes, que
# e a tarefa perceptual em que o olho humano vai pior. Barras ordenadas por
# magnitude resolvem os tres problemas: escala linear, ordem informativa
# (do mais discriminante para o menos) e comparacao ao longo de um eixo comum.
def fig_barras_perfis(variaveis: list, medias: pd.DataFrame) -> go.Figure:
    """Media normalizada de cada perfil nas variaveis mais discriminantes,
    em barras agrupadas e ordenadas pela ordem de discriminancia recebida."""
    # Invertido porque o Plotly desenha o eixo Y de baixo para cima: assim a
    # variavel mais discriminante fica no TOPO, que e onde o olho comeca.
    eixos = [ROTULO_VARIAVEL.get(v, v) for v in variaveis][::-1]
    fig = go.Figure()
    for perfil in (1, 2, 3):
        if perfil not in medias.index:
            continue
        valores = medias.loc[perfil, variaveis].tolist()[::-1]
        fig.add_trace(go.Bar(
            y=eixos, x=valores, name=NOME_CURTO[perfil], orientation="h",
            marker=dict(color=CORES_PERFIL[perfil]),
            hovertemplate="%{y}: %{x:.0f}/100<extra>" + NOME_CURTO[perfil] + "</extra>",
        ))
    fig.update_layout(
        barmode="group",
        bargap=0.28,
        # Folga a direita: com range exatamente [0, 100] a barra do valor
        # maximo encosta na borda e parece cortada.
        xaxis=dict(range=[0, 104], title="", gridcolor="#e5e7eb",
                   zeroline=False),
        yaxis=dict(title="", automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22,
                    xanchor="center", x=0.5, title=""),
        margin=dict(l=10, r=20, t=20, b=20),
    )
    return _layout_plotly(fig, 420)


def fig_barras_conduta(eixos: list, medias: pd.DataFrame) -> go.Figure:
    """Mesma leitura, do lado da telemetria: media de cada rotulo
    comportamental por metrica, na regua 0-100 da propria base."""
    rotulos = list(eixos)[::-1]
    fig = go.Figure()
    for comportamento in ORDEM_COMPORTAMENTO:
        if comportamento not in medias.index:
            continue
        valores = medias.loc[comportamento, eixos].tolist()[::-1]
        cor, _ = CORES_COMPORTAMENTO[comportamento]
        fig.add_trace(go.Bar(
            y=rotulos, x=valores, name=comportamento.capitalize(), orientation="h",
            marker=dict(color=cor),
            hovertemplate="%{y}: %{x:.0f}/100<extra>"
                          + comportamento.capitalize() + "</extra>",
        ))
    fig.update_layout(
        barmode="group",
        bargap=0.28,
        xaxis=dict(range=[0, 104], title="", gridcolor="#e5e7eb", zeroline=False),
        yaxis=dict(title="", automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22,
                    xanchor="center", x=0.5, title=""),
        margin=dict(l=10, r=20, t=20, b=20),
    )
    return _layout_plotly(fig, 420)


def _diagrama_transicoes_html() -> str:
    """Os tres perfis em fila, com as duas transicoes da Secao 3.4.1 como
    setas. Substitui os dois cartoes soltos: o fluxo Referral -> Desconto ->
    Agravo passa a ser visto como caminho entre perfis, nao como lista."""
    def no(perfil, cor_fundo, cor_borda):
        return (f'<div style="flex:1;min-width:150px;background:{cor_fundo};'
                f'border:2px solid {cor_borda};border-radius:12px;padding:14px;'
                f'text-align:center;"><div style="font-weight:800;color:#1f2937;">'
                f'{NOME_CURTO[perfil]}</div><div style="font-size:12px;color:#4b5563;'
                f'margin-top:4px;">{ACAO_HARDWARE[perfil]}</div></div>')

    def seta(titulo, texto, cor):
        return (f'<div style="flex:1.3;min-width:170px;text-align:center;padding:0 6px;">'
                f'<div style="font-size:12.5px;font-weight:700;color:{cor};">{titulo}</div>'
                f'<div style="font-size:26px;line-height:1;color:{cor};">⟶</div>'
                f'<div style="font-size:12px;color:#4b5563;line-height:1.45;">{texto}</div></div>')

    return (
        '<div class="card-texto">'
        '<h4 style="color:#6d28d9;margin-top:0;">⚡ Transições previstas entre perfis</h4>'
        '<p style="color:#4b5563;font-size:14px;line-height:1.6;">'
        "O diferencial previsto para esta arquitetura é transformar a fotografia "
        "estática da cotação em um <b>filme dinâmico</b>: a telemetria gerada na "
        "borda (ESP32) corrigiria as distorções da declaração inicial. As setas são "
        "o desenho da Seção 3.4.1 — arquitetura prevista, não resultado medido.</p>"
        '<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-top:14px;">'
        + no(3, "#fffbeb", "#f59e0b")
        + seta("Conduta observada boa",
               "cliente sem histórico aceita o dispositivo como condição de emissão "
               "e migra para desconto", "#047857")
        + no(2, "#ecfdf5", "#10b981")
        + seta("Eventos inerciais recorrentes",
               "má gestão operacional que a declaração escondia; reclassificação "
               "sempre com decisão humana", "#b91c1c")
        + no(1, "#fef2f2", "#ef4444")
        + "</div></div>"
    )


def render_tab_hibrido(normalizada, significancia, mapa_perfil):
    col_perfis, col_conduta = st.columns([1, 1])

    with col_perfis:
        _titulo_secao(
            "A composição da frota separa mais que o histórico financeiro",
            "As 6 variáveis com maior tamanho de efeito (η² para contínuas, V de "
            "Cramér para categóricas), da mais discriminante para a menos, na média "
            "normalizada Min-Max do próprio K-Means reescalada a 0–100. São dados "
            "declaratórios e históricos da Parte A — nenhuma leitura de telemetria "
            "entra aqui. A ordenação é o ranking: a barra de cima é a variável que "
            "mais separa a carteira.",
        )
        if significancia is None or significancia.empty:
            st.info("Rode `python src/perfilamento.py` para gerar os testes de "
                    "significância.", icon="ℹ️")
        else:
            variaveis, medias = calcular_discriminantes_por_perfil(normalizada, significancia, mapa_perfil)
            st.plotly_chart(fig_barras_perfis(variaveis, medias), use_container_width=True)

    with col_conduta:
        resumo = carregar_resumo_uah()
        n_trajetos = 0 if resumo is None else len(resumo)
        texto_trajetos = f"{n_trajetos} trajetos reais" if n_trajetos else "trajetos reais"
        _titulo_secao(
            f"Assinatura de conduta · {FONTE_CITACAO}",
            "A mesma leitura, do outro lado da arquitetura: cinco métricas "
            f"medidas em {texto_trajetos}, normalizadas na mesma régua 0–100. É esta "
            "a evidência comportamental que, em produção, um dispositivo próprio "
            "traria de volta — aqui ela vem de um dataset público, de motoristas "
            "que não têm apólice nenhuma neste projeto.",
        )
        if resumo is None or resumo.empty:
            st.info("Rode `python src/validacao_hardware.py` para gerar o resumo "
                    "por trajeto do UAH-DriveSet.", icon="ℹ️")
        else:
            eixos, medias_conduta = calcular_conduta_normalizada(resumo)
            st.plotly_chart(fig_barras_conduta(eixos, medias_conduta),
                            use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(_diagrama_transicoes_html(), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="nota-integridade">'
        "<b>Por que são duas figuras, e não uma só.</b> Um gráfico único, com "
        "variáveis declaratórias e métricas de telemetria lado a lado, afirmaria que aquele "
        "segurado tem aquela condução medida — e esse vínculo não existe nesta PoC. "
        "As apólices são sintéticas (geradas para este trabalho) e os trajetos são de "
        "um dataset público de motoristas que não têm apólice nenhuma aqui: não há "
        "chave, motorista, veículo ou vínculo contratual em comum, e nenhuma "
        "correlação foi calculada entre as duas bases. Cada figura é normalizada "
        "dentro da própria base, então a leitura válida é o contraste entre as barras "
        "de uma mesma figura, nunca a comparação numérica de uma contra a outra. "
        "As transições acima descrevem a arquitetura prevista no texto do TCC "
        "— token pseudônimo por frota, ingestão que só enxerga o token e reassociação "
        "restrita à seguradora (LGPD, minimização de dados) — e não um resultado "
        "medido. A classificação permanece apoio à decisão, com humano no circuito "
        "(LGPD, Art. 20)."
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

    # Ideia central (Knaflic, "Storytelling with Data"): uma unica frase
    # declarativa antes das abas, dizendo o que o painel AFIRMA -- e nao o que
    # ele contem. Sem ela, quem abre o painel encontra tres abas nomeadas por
    # componente tecnico e precisa montar sozinho a conclusao.
    st.markdown(
        '<div style="background:#f5f3ff;border-left:4px solid #7c3aed;'
        'border-radius:8px;padding:16px 20px;margin:6px 0 22px 0;">'
        '<p style="color:#4c1d95;font-size:15.5px;line-height:1.55;margin:0;">'
        "<b>A carteira não é homogênea, e o dado para separá-la já existe na "
        "cotação.</b> Três perfis com exposição e comportamento de compra distintos "
        "são recuperáveis apenas do que a corretora já declara — e a telemetria de "
        "borda é o caminho para corrigir, ao longo da vigência, o que a declaração "
        "inicial não captura."
        "</p></div>",
        unsafe_allow_html=True,
    )

    # As abas sao nomeadas pela AFIRMACAO que cada uma sustenta, e nao pelo
    # componente tecnico que a produz. Nomes como "Software (K-Means)" descrevem
    # a implementacao e obrigam quem abre o painel a montar sozinho a conclusao;
    # nomes como "A carteira tem tres perfis" ja entregam a leitura e deixam a
    # figura no papel de evidencia (KNAFLIC, 2015).
    aba_software, aba_hardware, aba_hibrido = st.tabs([
        "📊 A carteira tem três perfis",
        "⚙️ A conduta é mensurável na borda",
        "⚡ Onde as duas camadas se encontram",
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
        "k=3 adotado com os três índices internos reportados e a divergência declarada "
        "(ver src/clustering.py e a aba de metodologia)</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
