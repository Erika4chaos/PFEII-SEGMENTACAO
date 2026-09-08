"""Figuras da aba "Validação de Hardware" (PFE II — Parte B.5).

Todas as figuras consomem o MESMO DataFrame por trajeto, com as colunas:

    trajeto_id    str   identificador do trajeto
    motorista     str   D1..D6 (código anônimo do próprio UAH-DriveSet)
    estilo        str   'normal' | 'agressiva' | 'sonolenta'
    via           str   'rodovia' | 'secundaria'
    duracao_min   float duração do trajeto em minutos
    pico_ms2      float maior magnitude de aceleração do trajeto (m/s²)
    eventos       int   passagens acima do limiar vigente
    eventos_min   float eventos / duracao_min

É a saída de validacao_hardware.validate_against_uah_driveset().

Regras de codificação seguidas aqui (para não repetir os erros da versão antiga):
  * três cores fixas, sempre na mesma ordem, nunca recicladas;
  * mediana em tinta neutra — nunca na cor de uma das categorias;
  * n visível em todo agrupamento;
  * um eixo por figura, nunca dois eixos y;
  * ordenação por valor, não pela ordem alfabética do rótulo.

O limiar do firmware NÃO tem valor padrão em nenhuma função deste módulo: ele
é sempre recebido de quem chama, e a única definição do número vive em
``THRESH_MAG_MS2`` (src/validacao_hardware.py).
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paleta — validada para daltonismo nas três primeiras posições.
# --------------------------------------------------------------------------
ESTILOS = ["normal", "agressiva", "sonolenta"]
CORES = ["#2a78d6", "#eb6834", "#1baf7a"]

TINTA = "#12181c"        # mediana, linhas de referência do leitor
TINTA2 = "#4b555b"
EIXO = "#b3bec2"
GRADE = "#e6ebec"
DESTAQUE = "#8a4b12"     # limiar vigente do firmware
BANDA_OK = "#e2f0e2"     # faixa recomendada pela literatura

G = 9.80665

# Largura fixa das quatro figuras, em px. Elas NAO sao esticadas ate a largura
# do container: com autosize "fit", a Figura 1 (dois paineis concatenados, com
# rotulos de eixo de larguras diferentes) passava a ser recortada na direita e
# perdia a marca de 8 m/s². Largura fixa e igual nas quatro tambem mantem o
# mesmo pixel por m/s² de uma figura para a outra.
LARGURA_FIGURA = 900
# Figura 2: altura util de cada linha de rotulo, mais a folga que o eixo x, o
# titulo dele e o padding consomem. O Streamlit pede ao Vega um autosize que
# INCLUI essa folga na altura declarada, entao declarar so ALTURA_LINHA * n
# faz a area de plotagem encolher e as linhas colarem umas nas outras.
ALTURA_LINHA = 56
FOLGA_EIXO = 64

# Limiares de referência da literatura, em m/s² (ver docstring do módulo tab_*).
LIT = {
    "pesado_lo": 0.20 * G,   # ~1,96
    "pesado_hi": 0.23 * G,   # ~2,26
    "passeio_lo": 0.25 * G,  # ~2,45
    "passeio_hi": 0.78 * G,  # ~7,65
    "tipico": 0.27 * G,      # ~2,65
}

ESCALA_COR = alt.Scale(domain=ESTILOS, range=CORES)
_BASE = {"font": "IBM Plex Sans, system-ui, sans-serif"}


def _tema(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            labelFont=_BASE["font"], titleFont=_BASE["font"],
            labelColor=TINTA2, titleColor=TINTA2,
            domainColor=EIXO, tickColor=EIXO, gridColor=GRADE,
            labelFontSize=12, titleFontSize=12, titlePadding=12,
        )
        .configure_legend(
            labelFont=_BASE["font"], titleFont=_BASE["font"],
            labelColor=TINTA2, titleColor=TINTA2, orient="top",
            direction="horizontal", title=None, symbolType="circle",
        )
        .configure_text(font=_BASE["font"])
    )


def _ordem_por_mediana(df: pd.DataFrame) -> list[str]:
    m = df.groupby("estilo")["pico_ms2"].median().sort_values()
    return list(m.index)


def _rotulo_n(df: pd.DataFrame) -> dict[str, str]:
    n = df.groupby("estilo").size()
    return {e: f"{e}  (n = {int(n.get(e, 0))})" for e in ESTILOS}


# ==========================================================================
# FIGURA 1 — o limiar do firmware contra a régua da literatura
# ==========================================================================
def fig1_escala_do_limiar(df: pd.DataFrame, limiar: float,
                          largura: int = LARGURA_FIGURA) -> alt.LayerChart:
    """Onde o limiar programado cai em relação às faixas usadas na literatura.

    É a primeira figura da aba de propósito: sem ela, um limiar que ninguém
    cruza parece defeito de pipeline em vez de escolha de projeto.
    """
    xmax = max(8.0, float(df["pico_ms2"].max()) + 0.5, limiar + 1.0)
    escala_x = alt.Scale(domain=[0, xmax], nice=False)
    # O eixo aparece uma vez so, no painel de baixo: dois painéis empilhados com
    # eixo próprio repetiam o mesmo título duas vezes.
    eixo_x = alt.Axis(title="Magnitude da aceleração (m/s²)", grid=False, tickCount=9)
    rot = _rotulo_n(df)

    faixas = pd.DataFrame([
        {"lo": LIT["pesado_lo"], "hi": LIT["pesado_hi"],
         "y": "Caminhão pesado", "leg": "0,20–0,23 g", "meio": None},
        {"lo": LIT["passeio_lo"], "hi": LIT["passeio_hi"],
         "y": "Carro de passeio (DOT)", "leg": "0,25–0,78 g",
         "meio": (LIT["passeio_lo"] + LIT["passeio_hi"]) / 2},
    ])
    ordem_faixa = list(faixas["y"])

    banda = alt.Chart(faixas).mark_bar(height=16, cornerRadius=3,
                                       color="#e2e8ea", stroke="#c8d2d5").encode(
        x=alt.X("lo:Q", scale=escala_x, axis=None),
        x2="hi:Q",
        y=alt.Y("y:N", title=None, sort=ordem_faixa,
                axis=alt.Axis(labelFontWeight="bold", labelLimit=220,
                              domain=False, ticks=False)),
    )
    # A faixa de pesado é estreita demais para conter o texto: o rótulo dela vai
    # para a esquerda, onde não há nada. A de passeio é larga e recebe o texto
    # dentro. Antes os dois iam à direita da faixa, e o de pesado caía em cima
    # da linha do valor típico.
    rot_estreita = alt.Chart(faixas[faixas["meio"].isna()]).mark_text(
        align="right", dx=-9, fontSize=11.5, color=TINTA2).encode(
        x=alt.X("lo:Q", scale=escala_x, axis=None),
        y=alt.Y("y:N", sort=ordem_faixa), text="leg:N")
    rot_larga = alt.Chart(faixas[faixas["meio"].notna()]).mark_text(
        fontSize=11.5, color=TINTA2).encode(
        x=alt.X("meio:Q", scale=escala_x, axis=None),
        y=alt.Y("y:N", sort=ordem_faixa), text="leg:N")

    # As duas linhas de referência ficam no painel de baixo, juntas, com os
    # rótulos no topo: são marcas do mesmo tipo e não competem com as barras.
    referencias = pd.DataFrame([
        {"v": LIT["tipico"], "cor": TINTA2,
         "t": f"mais usado na literatura · {LIT['tipico'] / G:.2f} g".replace(".", ",")},
        {"v": limiar, "cor": DESTAQUE,
         "t": f"limiar do firmware · {limiar:.0f} m/s² = {limiar / G:.2f} g".replace(".", ",")},
    ])

    pontos = alt.Chart(df.assign(faixa=df["estilo"].map(rot))).mark_circle(
        size=76, opacity=1, stroke="white", strokeWidth=1.4).encode(
        x=alt.X("pico_ms2:Q", scale=escala_x, axis=eixo_x),
        y=alt.Y("faixa:N", title=None, sort=[rot[e] for e in ESTILOS],
                axis=alt.Axis(domain=False, ticks=False, labelFontWeight="bold",
                              labelLimit=220)),
        color=alt.Color("estilo:N", scale=ESCALA_COR,
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("trajeto_id:N", title="Trajeto"),
                 alt.Tooltip("motorista:N", title="Motorista"),
                 alt.Tooltip("estilo:N", title="Rótulo"),
                 alt.Tooltip("via:N", title="Via"),
                 alt.Tooltip("pico_ms2:Q", title="Pico (m/s²)", format=".2f")],
    )
    regua = alt.Chart(referencias).mark_rule(strokeWidth=2.5).encode(
        x=alt.X("v:Q", scale=escala_x, axis=eixo_x),
        # A linha arranca abaixo do próprio rótulo: centrada e de altura cheia,
        # ela riscava o texto no meio.
        y=alt.value(22), y2=alt.value(150),
        color=alt.Color("cor:N", scale=None, legend=None))
    rot_regua = alt.Chart(referencias).mark_text(
        align="center", fontSize=11.5, fontWeight="bold", baseline="top").encode(
        x=alt.X("v:Q", scale=escala_x, axis=eixo_x), y=alt.value(2),
        text="t:N", color=alt.Color("cor:N", scale=None, legend=None))

    topo = (banda + rot_estreita + rot_larga).properties(height=58, width=largura)
    base = (pontos + regua + rot_regua).properties(height=150, width=largura)
    return _tema(alt.vconcat(topo, base, spacing=4).resolve_scale(x="shared"))


# ==========================================================================
# FIGURA 2 — pico por rótulo comportamental
# ==========================================================================
def fig2_pico_por_estilo(df: pd.DataFrame, limiar: float,
                         largura: int = LARGURA_FIGURA) -> alt.LayerChart:
    """Uma barra por rótulo, com os trajetos como pontos discretos acima dela.

    Substitui o dot plot que ocupava este lugar. Quarenta pontos em três
    linhas exigiam que o leitor estimasse o centro de cada nuvem antes de
    concluir qualquer coisa; a barra já entrega a comparação pronta, e o
    comprimento dela até a marca do limiar é a segunda leitura da figura.

    Os pontos continuam na figura, mas em cinza e acima da barra, nunca atrás
    dela: barra sobre pontos esconderia metade de cada grupo, e nenhum trajeto
    pode sumir de uma figura que declara n.

    Inverte de propósito a convenção das outras figuras deste módulo, onde a
    mediana é neutra e as categorias coloridas: aqui a barra É a categoria, e
    quem precisa ficar em tinta neutra são os trajetos individuais.
    """
    ordem = _ordem_por_mediana(df)
    rot = _rotulo_n(df)
    ordem_rot = [rot[e] for e in ordem]
    d = df.assign(faixa=df["estilo"].map(rot))

    med = (df.groupby("estilo", as_index=False)["pico_ms2"].median()
           .rename(columns={"pico_ms2": "mediana"}))
    med["faixa"] = med["estilo"].map(rot)
    med["texto"] = med["mediana"].map(lambda v: f"{v:.1f} m/s²".replace(".", ","))

    # Escala presa ao zero e estendida ate depois do limiar: a barra so pode
    # ser lida como comprimento se comecar no zero, e a distancia ate o limiar
    # e a propria conclusao da figura.
    escala_x = alt.Scale(domain=[0, limiar + 0.6], nice=False)
    eixo_x = alt.Axis(title="Pico de aceleração do trajeto (m/s²)", tickCount=8)
    eixo_y = alt.Y("faixa:N", title=None, sort=ordem_rot,
                   axis=alt.Axis(domain=False, ticks=False, labelFontWeight="bold",
                                 labelLimit=220))

    barra = alt.Chart(med).mark_bar(height=14, cornerRadius=2, opacity=0.92).encode(
        x=alt.X("mediana:Q", scale=escala_x, axis=eixo_x),
        y=eixo_y,
        color=alt.Color("estilo:N", scale=ESCALA_COR, legend=None),
        tooltip=[alt.Tooltip("estilo:N", title="Rótulo"),
                 alt.Tooltip("mediana:Q", title="Mediana do pico (m/s²)", format=".2f")],
    )
    # O valor sobe acima da ponta da barra: na mesma linha ele caía justamente
    # onde ficam os primeiros trajetos à direita da mediana. Dentro da barra
    # também não serve -- branco sobre o laranja e o verde não tem contraste.
    valor = alt.Chart(med).mark_text(align="left", dx=6, dy=-14, fontSize=13,
                                     fontWeight="bold", color=TINTA).encode(
        x=alt.X("mediana:Q", scale=escala_x, axis=eixo_x), y=eixo_y, text="texto:N")

    # Os trajetos ficam na MESMA linha da barra, não deslocados acima dela: um
    # yOffset em pixel não sobrevive ao redimensionamento do Streamlit -- o
    # deslocamento efetivo cresce, e os pontos de um rótulo acabam pousando na
    # barra do rótulo de cima. Contorno branco para o ponto se ler tanto sobre a
    # barra colorida quanto sobre o fundo claro à direita dela.
    pontos = alt.Chart(d).mark_circle(size=46, opacity=0.9, color=TINTA2,
                                      stroke="white", strokeWidth=1.2).encode(
        x=alt.X("pico_ms2:Q", scale=escala_x, axis=eixo_x),
        y=eixo_y,
        tooltip=[alt.Tooltip("trajeto_id:N", title="Trajeto"),
                 alt.Tooltip("motorista:N", title="Motorista"),
                 alt.Tooltip("via:N", title="Via"),
                 alt.Tooltip("duracao_min:Q", title="Duração (min)", format=".1f"),
                 alt.Tooltip("pico_ms2:Q", title="Pico (m/s²)", format=".2f")],
    )

    # A régua ocupa a altura cheia da área de plotagem, sem y/y2 em pixel: sob
    # o autosize "fit" do Streamlit, extremos fixados em pixel somam-se ao eixo,
    # estouram a altura declarada e o Vega encolhe a figura inteira.
    regua = alt.Chart(pd.DataFrame({"v": [limiar]})).mark_rule(
        color=DESTAQUE, strokeWidth=2.5).encode(
        x=alt.X("v:Q", scale=escala_x, axis=eixo_x))
    # O rótulo da régua vai para a folga reservada acima da área de plotagem,
    # onde não cruza a própria linha nem a fileira de pontos.
    rot_regua = alt.Chart(pd.DataFrame({
        "v": [limiar], "t": [f"limiar do firmware · {limiar:.0f} m/s²".replace(".", ",")],
    })).mark_text(align="right", dx=-8, dy=-9, baseline="bottom", fontSize=11.5,
                  fontWeight="bold", color=DESTAQUE).encode(
        x=alt.X("v:Q", scale=escala_x, axis=eixo_x), y=alt.value(0), text="t:N")

    return _tema((barra + pontos + valor + regua + rot_regua).properties(
        width=largura, height=ALTURA_LINHA * len(ordem_rot) + FOLGA_EIXO,
        padding={"top": 26, "left": 5, "right": 14, "bottom": 5}))


# ==========================================================================
# FIGURA 3 — curva de sensibilidade do limiar (a decisão de projeto)
# ==========================================================================
def fig3_sensibilidade(df: pd.DataFrame, limiar: float,
                       largura: int = LARGURA_FIGURA) -> alt.LayerChart:
    """Fração de trajetos que dispararia ao menos um evento, por limiar.

    Matematicamente é a função de sobrevivência empírica do pico por rótulo:
    'trajetos com pelo menos um evento em T' == 'trajetos cujo pico >= T'.
    É a figura que transforma o limiar de constante herdada em decisão
    justificada — e a que substitui o gráfico de linhas cruzadas antigo.
    """
    grade = np.round(np.arange(1.0, 8.01, 0.1), 2)
    linhas = []
    for e in ESTILOS:
        picos = df.loc[df["estilo"] == e, "pico_ms2"].to_numpy()
        if picos.size == 0:
            continue
        for t in grade:
            linhas.append({"estilo": e, "limiar": float(t),
                           "pct": float((picos >= t).mean() * 100.0),
                           "n": int(picos.size)})
    curva = pd.DataFrame(linhas)
    rot = _rotulo_n(df)
    curva["faixa"] = curva["estilo"].map(rot)

    # Todas as camadas declaram a MESMA escala de x. Sem isso o retângulo da
    # faixa recomendada criava a própria escala, com zero=True, e o eixo do
    # gráfico inteiro voltava a começar em 0.
    escala_x = alt.Scale(domain=[1, 8], nice=False)
    eixo_x = alt.Axis(title="Limiar de magnitude programado no firmware (m/s²)",
                      tickCount=15)
    faixa_lo, faixa_hi = LIT["pesado_lo"], LIT["passeio_lo"] + 0.55

    zona = alt.Chart(pd.DataFrame({"lo": [faixa_lo], "hi": [faixa_hi]})).mark_rect(
        color=BANDA_OK).encode(
        x=alt.X("lo:Q", scale=escala_x, axis=eixo_x), x2=alt.X2("hi:Q"))
    rot_zona = alt.Chart(pd.DataFrame({
        "m": [(faixa_lo + faixa_hi) / 2], "t": ["faixa da literatura"],
    })).mark_text(align="center", baseline="middle", fontSize=11.5,
                  color="#3d7a4a").encode(
        # Rodapé da zona (no topo o rótulo encostava na curva de sonolenta, que
        # fica em 100% ao longo de toda a faixa), ancorado na ESCALA e não em
        # pixels: sob o autosize "fit" do Streamlit, um texto fixado perto do
        # rodapé soma-se ao eixo x, estoura a altura declarada e o Vega encolhe
        # a área de plotagem até tudo caber -- a figura vinha achatada.
        x=alt.X("m:Q", scale=escala_x, axis=eixo_x), y=alt.datum(6), text="t:N")

    linha = alt.Chart(curva).mark_line(strokeWidth=2.5, interpolate="step-after").encode(
        x=alt.X("limiar:Q", scale=escala_x, axis=eixo_x),
        y=alt.Y("pct:Q", title="% de trajetos com ao menos 1 evento",
                scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("faixa:N", legend=alt.Legend(title=None),
                        scale=alt.Scale(domain=[rot[e] for e in ESTILOS], range=CORES)),
        tooltip=[alt.Tooltip("faixa:N", title="Rótulo"),
                 alt.Tooltip("limiar:Q", title="Limiar (m/s²)", format=".1f"),
                 alt.Tooltip("pct:Q", title="% de trajetos", format=".0f")],
    )
    atual = alt.Chart(pd.DataFrame({"v": [limiar]})).mark_rule(
        color=DESTAQUE, strokeWidth=2.5).encode(
        x=alt.X("v:Q", scale=escala_x, axis=eixo_x))
    rot_atual = alt.Chart(pd.DataFrame({
        "v": [limiar], "t": [f"limiar atual · {limiar:.0f} m/s²".replace(".", ",")],
    })).mark_text(align="right", dx=-8, baseline="middle", fontSize=11.5,
                  fontWeight="bold", color=DESTAQUE).encode(
        x=alt.X("v:Q", scale=escala_x, axis=eixo_x), y=alt.datum(96), text="t:N")

    return _tema((zona + rot_zona + linha + atual + rot_atual)
                 .properties(width=largura, height=260))


# ==========================================================================
# FIGURA 4 — efeito pareado dentro do mesmo motorista
# ==========================================================================
def fig4_delta_por_motorista(df: pd.DataFrame, largura: int = LARGURA_FIGURA) -> alt.LayerChart:
    """Pico médio na agressiva menos o pico médio na normal, por condutor.

    Substitui o gráfico de três colunas com linhas se cruzando. A afirmação
    'todos os motoristas sobem' vira 'todas as hastes à direita do zero'.
    Sonolenta fica fora de propósito: o efeito não é consistente entre
    condutores, e a figura honesta é dizer isso no texto.
    """
    piv = (df[df["estilo"].isin(["normal", "agressiva"])]
           .groupby(["motorista", "estilo"], as_index=False)
           .agg(pico=("pico_ms2", "mean"), n=("pico_ms2", "size"))
           .pivot(index="motorista", columns="estilo", values=["pico", "n"]))
    if piv.empty:
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()

    d = pd.DataFrame({
        "motorista": piv.index,
        "delta": piv[("pico", "agressiva")] - piv[("pico", "normal")],
        "n": piv[("n", "agressiva")].fillna(0) + piv[("n", "normal")].fillna(0),
    }).dropna(subset=["delta"]).sort_values("delta")
    d["texto"] = d["delta"].map(lambda v: f"{v:+.1f} m/s²".replace(".", ","))
    # n de cada haste no próprio eixo: cada barra resume dois grupos de trajetos
    # do mesmo condutor, e D6 tem bem menos que os outros.
    d["faixa"] = [f"{m}  (n = {int(k)})" for m, k in zip(d["motorista"], d["n"])]
    ordem = list(d["faixa"])

    # Folga à direita para o valor escrito na ponta: sem ela o rótulo da maior
    # haste ficava cortado na borda do gráfico.
    escala_x = alt.Scale(domain=[0, float(d["delta"].max()) * 1.22], nice=False)
    eixo_x = alt.Axis(title="Pico médio na agressiva menos o pico médio na normal (m/s²)",
                      tickCount=8)
    eixo_y = alt.Y("faixa:N", title=None, sort=ordem,
                   axis=alt.Axis(domain=False, ticks=False, labelFontWeight="bold",
                                 labelLimit=220))

    haste = alt.Chart(d).mark_rule(strokeWidth=2.5, color=CORES[1]).encode(
        x=alt.X("zero:Q", scale=escala_x, axis=eixo_x),
        x2=alt.X2("delta:Q"), y=eixo_y,
    ).transform_calculate(zero="0")
    ponta = alt.Chart(d).mark_circle(size=110, color=CORES[1], stroke="white",
                                     strokeWidth=1.5).encode(
        x=alt.X("delta:Q", scale=escala_x, axis=eixo_x), y=eixo_y,
        tooltip=[alt.Tooltip("motorista:N", title="Motorista"),
                 alt.Tooltip("delta:Q", title="Δ pico (m/s²)", format="+.2f"),
                 alt.Tooltip("n:Q", title="trajetos usados")],
    )
    valor = alt.Chart(d).mark_text(align="left", dx=12, fontSize=12.5,
                                   color=TINTA2).encode(
        x=alt.X("delta:Q", scale=escala_x, axis=eixo_x), y=eixo_y, text="texto:N")
    eixo_zero = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(
        color=EIXO, strokeWidth=1.5).encode(x=alt.X("v:Q", scale=escala_x, axis=eixo_x))

    return _tema((eixo_zero + haste + ponta + valor)
                 .properties(width=largura, height=28 * len(ordem) + 30))


# ==========================================================================
# FIGURA 5 — o que muda ao recalibrar (a figura de defesa)
# ==========================================================================
# Rampa sequencial de um tom so, validada com scripts/validate_palette.py
# --ordinal: luminosidade monotona, degraus visiveis e a classe mais clara
# ainda separada do fundo (2,19:1). Usa o roxo da identidade, distinto das
# tres cores categoricas de estilo, porque codifica MAGNITUDE, nao categoria.
RAMPA_ROXO = ["#bfa0e2", "#a67ed3", "#8c5cc0", "#7140a6", "#552a85", "#3a1760"]

COR_VIGENTE = "#8a4b12"      # mesmo tom do limiar atual nas outras figuras
COR_RECOMENDADO = "#1baf7a"


def fig5_antes_depois(vigente: dict, recomendado: dict,
                      largura: int = LARGURA_FIGURA) -> alt.LayerChart:
    """As duas metricas que a banca vai perguntar, antes e depois.

    Uma linha por metrica, dois pontos ligados: o valor com o limiar que esta
    no firmware hoje e o valor com o par recomendado. E a figura mais direta
    da aba, e por isso vem ANTES da grade completa -- a grade e a conferencia,
    esta e a afirmacao.

    So duas metricas de proposito. Sensibilidade e especificidade respondem
    "pega quem dirige mal?" e "incomoda quem dirige bem?", que sao as duas
    perguntas de produto; Youden e um resumo das duas e fica como numero no
    texto, nao como terceira barra.
    """
    linhas = []
    for rotulo, ponto in (("Limiar de hoje", vigente), ("Recomendado", recomendado)):
        # int() explicito: quando o ponto vem de uma linha da grade, o pandas
        # entrega as contagens como float e o rotulo sairia "9.0 de 11.0".
        linhas.append({
            "metrica": "Agressivos", "cenario": rotulo,
            "valor": 100 * ponto["sensibilidade"],
            "detalhe": f"{int(ponto['agressiva_sinalizados'])} de "
                       f"{int(ponto['n_agressiva'])}",
        })
        linhas.append({
            "metrica": "Normais", "cenario": rotulo,
            "valor": 100 * ponto["especificidade"],
            "detalhe": f"{int(ponto['n_normal']) - int(ponto['normal_sinalizados'])} de "
                       f"{int(ponto['n_normal'])}",
        })
    d = pd.DataFrame(linhas)
    d["texto"] = [f"{v:.0f}%  ({t})" for v, t in zip(d["valor"], d["detalhe"])]
    # Rotulos curtos e o sentido no titulo do eixo. Nomes longos no eixo y sao
    # medidos com a fonte declarada no tema e desenhados com a que a maquina
    # tem: "Agressivos detectados" vinha sem o ultimo caractere. O enquadramento
    # tambem fica melhor -- uma metrica so ("acertou?"), lida em dois grupos,
    # em vez de sensibilidade e especificidade como conceitos separados.
    ordem_metrica = ["Agressivos", "Normais"]
    escala_cor = alt.Scale(domain=["Limiar de hoje", "Recomendado"],
                           range=[COR_VIGENTE, COR_RECOMENDADO])

    escala_x = alt.Scale(domain=[0, 118], nice=False)
    eixo_x = alt.Axis(title="% do grupo que o detector trata corretamente",
                      values=[0, 25, 50, 75, 100])
    eixo_y = alt.Y("metrica:N", title=None, sort=ordem_metrica,
                   axis=alt.Axis(domain=False, ticks=False, labelFontWeight="bold",
                                 labelLimit=320))

    faixa = (d.pivot(index="metrica", columns="cenario", values="valor")
             .reset_index().rename(columns={"Limiar de hoje": "antes", "Recomendado": "depois"}))
    ligacao = alt.Chart(faixa).mark_rule(strokeWidth=2, color="#c3bccd").encode(
        x=alt.X("antes:Q", scale=escala_x, axis=eixo_x), x2=alt.X2("depois:Q"), y=eixo_y)

    pontos = alt.Chart(d).mark_point(size=190, filled=True, opacity=1,
                                     stroke="white", strokeWidth=1.5).encode(
        x=alt.X("valor:Q", scale=escala_x, axis=eixo_x), y=eixo_y,
        color=alt.Color("cenario:N", scale=escala_cor,
                        legend=alt.Legend(orient="top", direction="horizontal",
                                          title=None, labelFontSize=12, symbolSize=140)),
        shape=alt.Shape("cenario:N", legend=None,
                        scale=alt.Scale(domain=["Limiar de hoje", "Recomendado"],
                                        range=["square", "circle"])),
        tooltip=[alt.Tooltip("cenario:N", title="Cenário"),
                 alt.Tooltip("metrica:N", title="Métrica"),
                 alt.Tooltip("valor:Q", title="%", format=".1f"),
                 alt.Tooltip("detalhe:N", title="Trajetos")],
    )
    # Uma camada de texto por cenario, cada uma com seu proprio deslocamento
    # vertical: em "Normais preservados" os dois valores ficam a 12 pontos de
    # distancia e os rotulos se sobreporiam na mesma linha. (yOffset nao aceita
    # condicao no Vega-Lite, dai serem duas camadas em vez de uma.)
    rotulos = [
        alt.Chart(d[d["cenario"] == cenario]).mark_text(
            align="left", dx=14, dy=deslocamento, fontSize=11.5, fontWeight="bold",
            color=cor,
        ).encode(x=alt.X("valor:Q", scale=escala_x, axis=eixo_x), y=eixo_y, text="texto:N")
        for cenario, deslocamento, cor in (
            ("Limiar de hoje", -13, COR_VIGENTE),
            ("Recomendado", 13, COR_RECOMENDADO),
        )
    ]

    # A altura declarada e o TOTAL quando o Streamlit ajusta o grafico ao
    # container (autosize "fit"): legenda e eixo saem dela, nao se somam a ela.
    # Com 62px por linha sobravam 32px para as duas faixas e os rotulos
    # deslocados se encavalavam. A folga de 110px cobre legenda mais eixo.
    return _tema(alt.layer(ligacao, pontos, *rotulos).properties(
        width=largura, height=70 * len(ordem_metrica) + 110,
        padding={"left": 26, "top": 4, "right": 14, "bottom": 4}))


# ==========================================================================
# FIGURA 6 — a grade de calibracao inteira
# ==========================================================================
def fig6_grade_calibracao(grade: pd.DataFrame, melhor: dict,
                          largura: int = LARGURA_FIGURA) -> alt.LayerChart:
    """Youden para cada par (limiar, taxa minima) da varredura.

    A figura 5 afirma; esta mostra de onde a afirmacao saiu, e que o ponto
    escolhido nao e um pico isolado de ruido, e sim o centro de uma regiao
    inteira de pares que funcionam. E o que responde "por que 2,3 e nao 2,4?"
    -- a resposta honesta e "tanto faz, e a figura deixa isso visivel".

    Youden = sensibilidade + especificidade - 1. Zero significa um detector
    que nao distingue os dois grupos; 1 seria separacao perfeita.
    """
    melhor_ponto = pd.DataFrame([{
        "limiar_ms2": melhor["limiar_ms2"],
        "taxa_min_eventos_min": melhor["taxa_min_eventos_min"],
    }])

    celulas = alt.Chart(grade).mark_rect().encode(
        x=alt.X("limiar_ms2:O",
                title="Limiar de magnitude T (m/s²)",
                axis=alt.Axis(labelAngle=0, labelOverlap=True, format=".1f")),
        y=alt.Y("taxa_min_eventos_min:O",
                title="Taxa mínima N (eventos/min)",
                sort="descending", axis=alt.Axis(format=".1f")),
        color=alt.Color("youden:Q", title="Youden",
                        scale=alt.Scale(range=RAMPA_ROXO, type="quantize", nice=False),
                        legend=alt.Legend(orient="right", gradientLength=170,
                                          format=".2f")),
        tooltip=[alt.Tooltip("limiar_ms2:Q", title="Limiar T (m/s²)", format=".1f"),
                 alt.Tooltip("taxa_min_eventos_min:Q", title="Taxa mín. N", format=".1f"),
                 alt.Tooltip("youden:Q", title="Youden", format=".3f"),
                 alt.Tooltip("sensibilidade:Q", title="Sensibilidade", format=".1%"),
                 alt.Tooltip("especificidade:Q", title="Especificidade", format=".1%")],
    )
    marca = alt.Chart(melhor_ponto).mark_point(
        size=150, shape="circle", filled=False, stroke="white", strokeWidth=2.5,
    ).encode(x=alt.X("limiar_ms2:O"), y=alt.Y("taxa_min_eventos_min:O", sort="descending"))

    return _tema((celulas + marca).properties(width=largura, height=330))


# ==========================================================================
# Tabela de cobertura — expõe o confundidor motorista × estilo × via
# ==========================================================================
def tabela_cobertura(df: pd.DataFrame) -> pd.DataFrame:
    """Trajetos por motorista e rótulo, com a divisão rodovia/secundária.

    A especificação exige n visível e o confundidor declarado; uma tabela
    resolve os dois de uma vez e mostra quais células estão vazias.
    """
    linhas = []
    for m in sorted(df["motorista"].unique()):
        linha = {"Motorista": m}
        for e in ESTILOS:
            sub = df[(df["motorista"] == m) & (df["estilo"] == e)]
            r = int((sub["via"] == "rodovia").sum())
            s = int((sub["via"] == "secundaria").sum())
            linha[e.capitalize()] = "—" if len(sub) == 0 else f"{len(sub)}  ({r}R / {s}S)"
        linhas.append(linha)
    return pd.DataFrame(linhas)
