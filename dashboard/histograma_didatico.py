"""
histograma_didatico.py

Histogramas autoexplicativos em Altair, pensados para figura de defesa: cada
classe rotulada com seu intervalo, contagem escrita sobre a barra, classe
modal destacada e linhas de limiar posicionadas exatamente.

Tres armadilhas que este modulo resolve (e que fazem esse grafico falhar
silenciosamente quando se usa px.histogram / st.bar_chart / alt.X(bin=True)):

1. O binning e feito no numpy, nao no motor do grafico. Quando as classes sao
   calculadas dentro do JavaScript, o Python nunca ve os limites delas -- fica
   impossivel rotular cada barra, escrever a contagem em cima, destacar a moda
   ou posicionar a linha do limiar no lugar certo. Aqui as classes chegam como
   dado (ver binar()).

2. y2=alt.datum(0) e obrigatorio. Com x + x2 + y quantitativo, o Vega-Lite NAO
   assume y2 = 0: a barra sai como um tracinho fino na altura da contagem, sem
   preenchimento e sem nenhum erro no console.

3. Rotulos de classe via labelExpr + labelOverlap=False. Os ticks ficam no
   centro de cada classe (idx + 0.5) e o labelExpr traduz essa posicao no texto
   do intervalo. Sem labelOverlap=False o Vega esconde rotulos alternados.

As barras sao desenhadas no espaco de INDICE da classe (x=idx, x2=idx+1), nao
no espaco do valor: assim toda classe tem a mesma largura na tela, como num
histograma de livro-texto, e o eixo pode ser rotulado por intervalo. Limiares
sao convertidos para esse mesmo espaco por posicao_no_eixo().
"""

import altair as alt
import numpy as np
import pandas as pd

COR_PADRAO = "#8e2f9e"
COR_MODA = "#6d3bc4"
COR_LIMIAR = "#44355b"

# Base da escala logaritmica. log(0) nao existe, entao em escala log a barra
# nao pode descer ate zero: ela desce ate este piso, que precisa ser o MESMO
# valor no domainMin da escala e no y2 das barras -- se os dois divergirem, a
# barra flutua ou some.
PISO_LOG = 0.7


def num_ptbr(valor: float, casas: int = 1) -> str:
    """Numero em formato pt-BR: milhar com ponto, decimal com virgula."""
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def int_ptbr(valor: int) -> str:
    """Inteiro em formato pt-BR: milhar com ponto."""
    return f"{int(valor):,}".replace(",", ".")


_num_ptbr, _int_ptbr = num_ptbr, int_ptbr   # nomes internos historicos


def binar(valores, n_classes: int = 20, minimo: float = 0.0, maximo: float = None) -> pd.DataFrame:
    """np.histogram -> tabela de classes com limites explicitos.

    As bordas saem daqui, e nao do grafico, justamente para que o Python
    conheca cada classe (ver docstring do modulo)."""
    valores = np.asarray(valores, dtype=float)
    valores = valores[~np.isnan(valores)]
    if maximo is None:
        maximo = float(valores.max()) if len(valores) else 1.0
    bordas = np.linspace(minimo, maximo, n_classes + 1)
    contagens, _ = np.histogram(valores, bins=bordas)
    return pd.DataFrame({
        "classe_idx": np.arange(n_classes),
        "classe_min": bordas[:-1],
        "classe_max": bordas[1:],
        "n_amostras": contagens.astype(int),
    })


def posicao_no_eixo(valor: float, classes: pd.DataFrame) -> float:
    """Converte um valor na unidade original (ex.: 6 m/s^2) para a posicao
    correspondente no eixo de indices de classe, para que a linha do limiar
    caia no ponto exato -- inclusive no meio de uma classe."""
    largura = float(classes["classe_max"].iloc[0] - classes["classe_min"].iloc[0])
    minimo = float(classes["classe_min"].iloc[0])
    return (valor - minimo) / largura


def _eixo_de_classes(classes: pd.DataFrame, rotulo_x: str, casas: int = 1,
                      mostrar_rotulos: bool = True, campo: str = "classe_idx") -> alt.X:
    """Ticks no centro de cada classe; labelExpr reconstroi o texto do
    intervalo a partir da posicao. Decimal com virgula (pt-BR).

    O eixo e construido no espaco de indice de classe, com dominio fixo
    [0, n_classes] -- o mesmo dominio precisa valer para a camada das barras E
    para a camada das linhas de limiar, senao o Vega alinha as duas em escalas
    diferentes e a linha cai no lugar errado."""
    largura = float(classes["classe_max"].iloc[0] - classes["classe_min"].iloc[0])
    minimo = float(classes["classe_min"].iloc[0])
    fmt = f"'.{casas}f'"
    ini = f"({minimo} + floor(datum.value) * {largura})"
    fim = f"({minimo} + (floor(datum.value) + 1) * {largura})"
    label_expr = (
        f"replace(format({ini}, {fmt}), '.', ',') + '–' + "
        f"replace(format({fim}, {fmt}), '.', ',')"
    )
    n = int(classes["classe_idx"].max()) + 1
    return alt.X(
        f"{campo}:Q",
        title=rotulo_x,
        scale=alt.Scale(domain=[0, n], nice=False),
        axis=alt.Axis(
            values=[i + 0.5 for i in range(n)],
            labelExpr=label_expr,
            labelOverlap=False,          # sem isto o Vega esconde rotulos alternados
            labelAngle=-45,
            labelFontSize=9,
            labels=mostrar_rotulos,
            tickSize=0,
            grid=False,
        ),
    )


def histograma_explicado(classes: pd.DataFrame, *, rotulo_x: str, unidade: str = "amostras",
                          cor: str = COR_PADRAO, limiares: dict = None, altura: int = 260,
                          largura: int = 720, escala_log: bool = False,
                          destacar_moda: bool = True, casas_classe: int = 1,
                          escala_y: alt.Scale = None, mostrar_rotulos_x: bool = True,
                          mostrar_texto_limiar: bool = True,
                          mostrar_contagem: bool = True) -> alt.LayerChart:
    """Um histograma didatico. `limiares` e um dict {rotulo: valor}, desenhado
    como linha tracejada vertical na posicao exata do valor (mesmo no meio de
    uma classe)."""
    dados = classes.copy()
    if escala_log:
        # Classe vazia nao tem representacao em escala log (log(0) nao existe);
        # a ausencia da barra ja comunica "nenhuma amostra nesta classe".
        dados = dados[dados["n_amostras"] > 0].copy()
    dados["classe_fim"] = dados["classe_idx"] + 1        # x2 como coluna real, nao transform
    dados["is_moda"] = dados["n_amostras"] == dados["n_amostras"].max()
    dados["rotulo_classe"] = [
        f"{_num_ptbr(a, casas_classe)} a {_num_ptbr(b, casas_classe)}"
        for a, b in zip(dados["classe_min"], dados["classe_max"])
    ]
    dados["rotulo_n"] = dados["n_amostras"].map(_int_ptbr)

    if escala_y is None:
        escala_y = (alt.Scale(type="log", domainMin=PISO_LOG) if escala_log
                    else alt.Scale(type="linear"))
    titulo_y = f"Numero de {unidade}" + (" — escala log" if escala_log else "")

    cor_encode = (
        alt.condition(alt.datum.is_moda, alt.value(COR_MODA), alt.value(cor))
        if destacar_moda else alt.value(cor)
    )

    eixo_x = _eixo_de_classes(classes, rotulo_x, casas_classe, mostrar_rotulos_x)

    # y2 e obrigatorio nos DOIS casos (armadilha 2 no topo do modulo): sem ele a
    # barra vira um tracinho fino na altura da contagem, sem preenchimento e sem
    # erro no console. O que muda e a base: 0 na escala linear, PISO_LOG na log
    # (porque log(0) nao existe).
    encodings = dict(
        x=eixo_x,
        x2=alt.X2("classe_fim:Q"),
        y=alt.Y("n_amostras:Q", title=titulo_y, scale=escala_y),
        color=cor_encode,
        tooltip=[
            alt.Tooltip("rotulo_classe:N", title=f"Classe ({rotulo_x})"),
            alt.Tooltip("rotulo_n:N", title="Amostras"),
        ],
    )
    encodings["y2"] = alt.datum(PISO_LOG if escala_log else 0)

    barras = alt.Chart(dados).mark_bar(stroke="white", strokeWidth=0.4).encode(**encodings)
    camadas = [barras]

    # Contagem escrita sobre cada barra -- so e possivel porque os limites das
    # classes existem no Python (armadilha 1). Suprimida automaticamente quando
    # ha classes demais para caber sem colidir.
    if mostrar_contagem and len(dados) <= 14:
        camadas.append(
            alt.Chart(dados).mark_text(dy=-5, fontSize=8, color="#65596f").encode(
                x=_eixo_de_classes(classes, rotulo_x, casas_classe, mostrar_rotulos_x, "centro"),
                y=alt.Y("n_amostras:Q", scale=escala_y),
                text=alt.Text("rotulo_n:N"),
            ).transform_calculate(centro="datum.classe_idx + 0.5")
        )

    if limiares:
        marcas = pd.DataFrame([
            {"pos": posicao_no_eixo(v, classes), "rotulo": r, "valor": v}
            for r, v in limiares.items()
        ])
        # As camadas de limiar repetem o MESMO eixo/dominio das barras: sem isso
        # o Vega cria uma segunda escala e a linha cai fora do lugar.
        camadas.append(
            alt.Chart(marcas).mark_rule(strokeDash=[6, 4], strokeWidth=2, color=COR_LIMIAR)
            .encode(x=_eixo_de_classes(classes, rotulo_x, casas_classe, mostrar_rotulos_x, "pos"))
        )
        if mostrar_texto_limiar:
            camadas.append(
                alt.Chart(marcas).mark_text(
                    align="right", dx=-4, dy=8, fontSize=10, fontWeight="bold",
                    color=COR_LIMIAR, angle=0,
                ).encode(
                    x=_eixo_de_classes(classes, rotulo_x, casas_classe, mostrar_rotulos_x, "pos"),
                    y=alt.value(6),
                    text=alt.Text("rotulo:N"),
                )
            )

    return alt.layer(*camadas).properties(width=largura, height=altura)


def histograma_por_perfil(classes: pd.DataFrame, *, coluna_grupo: str, ordem: list,
                           cores: dict, rotulo_x: str, unidade: str = "amostras",
                           limiares: dict = None, altura: int = 150, largura: int = 720,
                           escala_log: bool = False, casas_classe: int = 1,
                           mostrar_contagem: bool = True) -> alt.VConcatChart:
    """Um painel por grupo, empilhados verticalmente, com as MESMAS classes e a
    MESMA escala no eixo Y -- se cada painel tivesse sua propria escala, a
    comparacao visual entre eles seria invalida."""
    maximo_y = int(classes.groupby(coluna_grupo)["n_amostras"].max().max())
    # Escala Y unica, calculada uma vez e imposta a todos os paineis: se cada
    # painel escolhesse a propria escala, a comparacao visual entre eles seria
    # invalida (barra alta num painel poderia valer menos que barra baixa noutro).
    escala_y = (alt.Scale(type="log", domainMin=PISO_LOG, domainMax=maximo_y * 2.0)
                if escala_log else alt.Scale(type="linear", domain=[0, maximo_y * 1.15]))

    paineis = []
    for i, grupo in enumerate(ordem):
        sub = classes[classes[coluna_grupo] == grupo].copy()
        if sub.empty:
            continue
        ultimo = i == len(ordem) - 1
        n_grupo = int(sub["n_amostras"].sum())
        painel = histograma_explicado(
            sub, rotulo_x=(rotulo_x if ultimo else " "), unidade=unidade,
            cor=cores.get(grupo, COR_PADRAO), limiares=limiares, altura=altura,
            largura=largura, escala_log=escala_log, destacar_moda=False,
            casas_classe=casas_classe, escala_y=escala_y, mostrar_contagem=mostrar_contagem,
            mostrar_rotulos_x=ultimo,          # rotulos de classe so no painel de baixo
            mostrar_texto_limiar=(i == 0),     # texto do limiar so no painel de cima
        ).properties(title=alt.TitleParams(
            f"{grupo}  (n = {_int_ptbr(n_grupo)} {unidade})",
            anchor="start", fontSize=12, color="#44355b",
        ))
        paineis.append(painel)
    return alt.vconcat(*paineis, spacing=6).resolve_scale(x="shared", y="shared")


def frase_de_leitura(classes: pd.DataFrame, *, unidade: str, rotulo_x: str,
                      casas_classe: int = 1) -> str:
    """Leitura automatica do histograma, no mesmo espirito da figura.

    Aceita tanto a tabela de um grupo so quanto a de varios grupos empilhados
    (um bloco de classes por grupo): as contagens sao somadas por classe antes
    de contar as classes e achar a moda, senao o texto reportaria o numero de
    LINHAS como se fosse o numero de classes, e tomaria como moda a maior barra
    de um unico grupo."""
    por_classe = classes.groupby(
        ["classe_idx", "classe_min", "classe_max"], as_index=False
    )["n_amostras"].sum()
    total = int(por_classe["n_amostras"].sum())
    moda = por_classe.loc[por_classe["n_amostras"].idxmax()]
    pct_moda = 100 * moda["n_amostras"] / total if total else 0
    classes = por_classe
    return (
        f"O grafico agrupa {_int_ptbr(total)} {unidade} em {len(classes)} classes de "
        f"{rotulo_x.lower()}. A classe mais frequente e "
        f"{_num_ptbr(moda['classe_min'], casas_classe)} a "
        f"{_num_ptbr(moda['classe_max'], casas_classe)}, com "
        f"{_int_ptbr(moda['n_amostras'])} {unidade} "
        f"({_num_ptbr(pct_moda)}% do total)."
    )
