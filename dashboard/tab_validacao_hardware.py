"""Aba 2 do dashboard — Validação de Hardware (PFE II, Parte B.5).

Ordem de leitura (é o ponto da reescrita — a versão antiga mostrava o
resultado antes de dizer o que estava sendo testado):

    01  a resposta, em linguagem de negócio, antes de qualquer gráfico
    02  glossário dos oito termos que as figuras usam
    03  onde o limiar do firmware cai na régua da literatura
    04  distribuição do pico por rótulo comportamental
    05  curva de sensibilidade — que limiar escolher
    06  o que muda se o limiar for recalibrado — a figura de defesa
    07  a varredura (limiar x taxa mínima) inteira, para conferência
    08  efeito pareado dentro do mesmo motorista
    09  cobertura da base e o que este resultado não é

Referências das faixas da Figura 1 (todas em desaceleração brusca):
  * padrões DOT: 0,25–0,78 g para veículos de passeio; ~0,20 g para pesados;
  * Desai et al. e Mahmud & Day: ~0,27 g;
  * Kamla et al. (frota mista, Reino Unido): ~0,23 g pesados, ~0,45 g leves.
O limiar do firmware (LIMIAR_MAGNITUDE_MS2 no config.h, hoje 2,9 m/s² ≈ 0,30 g,
o p99,9 da condução normal do UAH-DriveSet) chega aqui como parâmetro. O limiar
anterior (6 m/s² = 0,61 g, Seção 2.8; MASELLO et al., 2025; BRÜHWILER et al.,
2022) aparece só como referência histórica.

Uso:
    from validacao_hardware import THRESH_MAG_MS2
    from dashboard.tab_validacao_hardware import render, carregar
    render(carregar("data/processed/uah_trips.csv"), limiar=THRESH_MAG_MS2)

O limiar é sempre recebido como parâmetro: não existe valor padrão aqui nem
em charts_validacao, para que THRESH_MAG_MS2 continue sendo a única definição
do número no projeto.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from .charts_validacao import (
    G, LIT, ESTILOS, CORES, LARGURA_FIGURA,
    fig1_escala_do_limiar, fig3_sensibilidade,
    fig4_delta_por_motorista, fig5_antes_depois, fig6_grade_calibracao,
    tabela_cobertura,
)
from .histograma_didatico import histograma_por_perfil, int_ptbr

COLUNAS = ["trajeto_id", "motorista", "estilo", "via",
           "duracao_min", "pico_ms2", "eventos", "eventos_min"]

CAMINHO_GRADE = "data/processed/grade_calibracao.csv"
CAMINHO_HISTOGRAMA = "data/processed/validacao_hardware_uah_histograma.csv"
CAMINHO_LIMIARES = "data/processed/validacao_hardware_uah_limiares.csv"

RAIZ_PROJETO = Path(__file__).resolve().parent.parent

# O piso de especificidade e a regra de desempate da varredura sao IMPORTADOS
# do pipeline, nunca reescritos aqui: se o dashboard aplicasse a propria regra,
# poderia recomendar um par diferente do que src/validacao_hardware.py imprime,
# e as duas respostas conviveriam sem ninguem notar.
if str(RAIZ_PROJETO / "src") not in sys.path:
    sys.path.append(str(RAIZ_PROJETO / "src"))
from validacao_hardware import ESPECIFICIDADE_MINIMA, _melhor_par  # noqa: E402
from validacao_hardware import (  # noqa: E402
    PISO_MAGNITUDE_JERK_MS2, THRESH_MAG_ANTERIOR_MS2,
)


def carregar_grade(caminho: str = CAMINHO_GRADE) -> pd.DataFrame | None:
    """Grade da varredura de calibracao, ou None se ela ainda nao foi gerada.

    Ausencia nao e erro: a grade sai de uma passada sobre o sinal bruto do
    UAH-DriveSet, que nem toda copia do projeto tem em disco."""
    alvo = Path(caminho)
    if not alvo.is_absolute():
        alvo = RAIZ_PROJETO / alvo
    return pd.read_csv(alvo) if alvo.exists() else None


def carregar_histograma(caminho: str = CAMINHO_HISTOGRAMA) -> pd.DataFrame | None:
    """Classes do histograma de amostras (leituras do acelerometro a
    10Hz), uma linha por comportamento x tipo_via x classe -- agregada aqui
    somando os dois tipos de via, porque esta figura compara comportamento,
    nao via. Vem de src/validacao_hardware.py; None se ainda nao foi gerada."""
    alvo = Path(caminho)
    if not alvo.is_absolute():
        alvo = RAIZ_PROJETO / alvo
    if not alvo.exists():
        return None
    df = pd.read_csv(alvo)
    return df.groupby(["comportamento", "classe_idx"], as_index=False).agg(
        classe_min=("classe_min", "first"), classe_max=("classe_max", "first"),
        n_amostras=("n_amostras", "sum"),
    )


def carregar_limiar_recalibrado(limiar_vigente: float,
                                caminho: str = CAMINHO_LIMIARES) -> float | None:
    """O outro limiar avaliado por src/validacao_hardware.py (p99,9 da
    condução normal) -- lido do CSV que o proprio pipeline gerou, nunca
    recalculado aqui, para nao divergir do numero que ele imprime."""
    alvo = Path(caminho)
    if not alvo.is_absolute():
        alvo = RAIZ_PROJETO / alvo
    if not alvo.exists():
        return None
    df = pd.read_csv(alvo)
    outros = df.loc[df["limiar_ms2"] != limiar_vigente, "limiar_ms2"]
    return float(outros.iloc[0]) if not outros.empty else None


def contagem_por_faixa(limiar_vigente: float, limiar_recalibrado: float,
                       caminho: str = CAMINHO_LIMIARES) -> dict | None:
    """Quantas amostras caem abaixo do recalibrado, entre os dois
    limiares, e acima do vigente -- somado dos n_acima/n_total que
    src/validacao_hardware.py ja calculou por comportamento x via (nunca
    reimplementado aqui a partir do sinal bruto)."""
    alvo = Path(caminho)
    if not alvo.is_absolute():
        alvo = RAIZ_PROJETO / alvo
    if not alvo.exists():
        return None
    df = pd.read_csv(alvo)
    n_total = int(df.loc[df["limiar_ms2"] == limiar_vigente, "n_total"].sum())
    n_acima_vigente = int(df.loc[df["limiar_ms2"] == limiar_vigente, "n_acima"].sum())
    n_acima_recal = int(df.loc[df["limiar_ms2"] == limiar_recalibrado, "n_acima"].sum())
    n_entre = n_acima_recal - n_acima_vigente
    return {
        "n_total": n_total, "n_acima_vigente": n_acima_vigente, "n_entre": n_entre,
        "n_abaixo_recal": n_total - n_acima_recal,
    }


def ponto_operacao(df: pd.DataFrame, sinalizado: pd.Series,
                   limiar: float, taxa_minima: float) -> dict:
    """Mesmas metricas da grade, para um par que nao esta nela.

    Serve ao limiar do firmware com N = 0 (qualquer evento sinaliza), a partir
    da coluna `eventos` que o proprio pipeline gerou naquele limiar."""
    agressiva = df["estilo"] == "agressiva"
    normal = df["estilo"] == "normal"
    n_agressiva, n_normal = int(agressiva.sum()), int(normal.sum())
    agressiva_sinalizados = int((sinalizado & agressiva).sum())
    normal_sinalizados = int((sinalizado & normal).sum())
    sensibilidade = agressiva_sinalizados / n_agressiva if n_agressiva else float("nan")
    especificidade = (n_normal - normal_sinalizados) / n_normal if n_normal else float("nan")
    return {
        "limiar_ms2": limiar, "taxa_min_eventos_min": taxa_minima,
        "n_agressiva": n_agressiva, "agressiva_sinalizados": agressiva_sinalizados,
        "sensibilidade": sensibilidade,
        "n_normal": n_normal, "normal_sinalizados": normal_sinalizados,
        "especificidade": especificidade,
        "youden": sensibilidade + especificidade - 1,
    }


def carregar(caminho: str = "data/processed/uah_trips.csv") -> pd.DataFrame:
    # Caminho relativo e resolvido contra a raiz do projeto, nao contra o
    # diretorio de onde o streamlit foi chamado.
    alvo = Path(caminho)
    if not alvo.is_absolute():
        alvo = RAIZ_PROJETO / alvo
    df = pd.read_csv(alvo)
    faltando = [c for c in COLUNAS if c not in df.columns]
    if faltando:
        raise ValueError(
            f"{alvo} não tem as colunas {faltando}. "
            "Rode `python src/validacao_hardware.py` primeiro "
            "(gera o CSV via processar_dataset + montar_trajetos)."
        )
    df["estilo"] = df["estilo"].str.lower().replace(
        {"aggressive": "agressiva", "drowsy": "sonolenta", "normal": "normal"})
    return df


def _br(v: float, casas: int = 1) -> str:
    return f"{v:.{casas}f}".replace(".", ",")


def _cartao(coluna, pergunta: str, veredito: str, cor: str, detalhe: str) -> None:
    with coluna:
        st.markdown(
            f"<div style='font-size:13.5px;color:#7d888e;min-height:3.4em'>{pergunta}</div>"
            f"<div style='font-size:25px;font-weight:600;line-height:1.16;"
            f"margin:6px 0 4px;color:{cor}'>{veredito}</div>"
            f"<div style='font-size:14px;color:#4b555b'>{detalhe}</div>",
            unsafe_allow_html=True,
        )


def render(df: pd.DataFrame, limiar: float) -> None:
    n_total = len(df)
    med = df.groupby("estilo")["pico_ms2"].median()
    acima = int((df["pico_ms2"] >= limiar).sum())
    pct_zero = 100.0 * (1 - acima / n_total) if n_total else 0.0

    st.info(
        "**O que esta aba testa.** Se a regra de detecção que roda no firmware produz "
        "mais eventos em trajetos que o próprio dataset chama de agressivos. "
        "Não é medição de sonolência, não é precisão por evento, e não é validação "
        "com poder estatístico — é uma checagem de discriminação, na mesma postura "
        "que a Seção 3.3 adota para os testes de bancada.",
        icon="🧭",
    )

    # ---------------------------------------------------------------- 01
    st.subheader("01 · A resposta, antes do gráfico")
    c1, c2, c3 = st.columns(3)
    _cartao(
        c1, "A regra do firmware separa condução agressiva de condução normal?",
        "Sim, no pico", "#0ca30c",
        f"Mediana do pico sobe de {_br(med.get('normal', float('nan')))} para "
        f"{_br(med.get('agressiva', float('nan')))} m/s². n = {n_total} trajetos.",
    )
    _cartao(
        c2, "Com o limiar do firmware, o dispositivo registraria esses eventos?",
        "Quase nunca" if acima <= n_total * 0.1 else "Em parte", "#eb6834",
        f"{acima} de {n_total} trajetos chegam a {_br(limiar)} m/s². "
        f"A métrica de produto — eventos/min — fica em zero em {_br(pct_zero, 0)}% da base. "
        f"No limiar anterior ({_br(THRESH_MAG_ANTERIOR_MS2, 0)} m/s²), "
        f"{int((df['pico_ms2'] >= THRESH_MAG_ANTERIOR_MS2).sum())} de {n_total}.",
    )
    _cartao(
        c3, "Isso já serve para alimentar precificação por uso?",
        "Ainda não", "#eb6834",
        "A base é de carros de passeio, com 40 trajetos e 6 motoristas. Falta a "
        "coleta com o dispositivo instalado em caminhão, a frota que a apólice "
        f"cobre — cuja literatura trabalha em {_br(LIT['pesado_lo'])}–"
        f"{_br(LIT['pesado_hi'])} m/s².",
    )

    # ---------------------------------------------------------------- 02
    st.subheader("02 · O que cada palavra significa aqui")
    termos = [
        ("trajeto", "Um percurso completo de um motorista, do início ao fim da gravação. "
                    f"A base tem {n_total}. É a unidade de análise: cada ponto das figuras "
                    "é um trajeto, nunca uma leitura do sensor."),
        ("rótulo comportamental", "normal, agressiva ou sonolenta. Foi a *instrução dada ao "
                                  "motorista* antes de sair e vale para o trajeto inteiro — "
                                  "não é medição, não é diagnóstico, não muda ao longo do percurso."),
        ("magnitude da aceleração", "‖(y,z)‖ dos canais já filtrados por Kalman que o próprio "
                                    "dataset entrega — longitudinal e lateral, os dois eixos de "
                                    "manobra. Sem subtração de gravidade: os canais do UAH-DriveSet "
                                    "já vêm sem ela. Sem eixo: uma curva brusca entra na conta igual "
                                    "a uma frenagem. O firmware embarcado calcula os três eixos e "
                                    "subtrai a gravidade por calibração — esta validação cobre uma "
                                    "versão simplificada, como declarado no módulo."),
        ("pico do trajeto", "O maior valor dessa magnitude no trajeto inteiro. Um número por "
                            "trajeto. É a régua das Figuras 1, 2 e 4."),
        ("limiar", f"O valor de magnitude a partir do qual o firmware chama a leitura de "
                   f"evento. Hoje: {_br(limiar)} m/s² (antes, {_br(THRESH_MAG_ANTERIOR_MS2, 0)}). "
                   "Não é constante da física — é "
                   "escolha de projeto, e é ela que está em teste aqui."),
        ("evento", f"No firmware: magnitude acima de {_br(limiar)} m/s², **ou** variação "
                   "brusca dessa magnitude (jerk) acompanhada de pelo menos "
                   f"{_br(PISO_MAGNITUDE_JERK_MS2, 2)} m/s² de "
                   "aceleração real — o jerk é critério auxiliar e não dispara sozinho, "
                   "justamente para que um solavanco isolado não vire evento de condução. "
                   "Nesta validação conta-se apenas o critério de magnitude. É a única "
                   "coisa que o dispositivo grava e transmite; o sinal bruto nunca sai da borda."),
        ("eventos/min", "Eventos detectados dividido pela duração do trajeto. É a métrica de "
                        "produto — a que iria para a precificação."),
        ("g", "1 g = 9,81 m/s². A literatura de telemática fala em g; o firmware fala em m/s². "
              "As duas escalas aparecem juntas na Figura 1."),
    ]
    ca, cb = st.columns(2)
    for i, (termo, definicao) in enumerate(termos):
        with (ca if i % 2 == 0 else cb):
            st.markdown(f"**`{termo}`** — {definicao}")

    # ---------------------------------------------------------------- 03
    st.subheader("03 · Onde o limiar caiu na régua da literatura")
    st.altair_chart(fig1_escala_do_limiar(df, limiar), width="content")
    st.caption(
        f"**Figura 1.** O limiar do firmware, {_br(limiar)} m/s², equivale a "
        f"{_br(limiar / G, 2)} g. Para caminhões pesados — a frota que esta apólice cobre — "
        "os estudos de telemática trabalham entre 0,20 e 0,23 g; as revisões de veículos de "
        "passeio usam de 0,25 a 0,78 g, com a maioria dos trabalhos perto de 0,27 g. O limiar "
        "atual cai logo acima dessa referência de passeio e ainda acima da de veículo pesado. "
        f"O anterior, {_br(THRESH_MAG_ANTERIOR_MS2, 0)} m/s² "
        f"({_br(THRESH_MAG_ANTERIOR_MS2 / G, 2)} g), ficava no topo da amplitude e cerca de "
        "três vezes acima da referência de pesado — o que explicava a contagem zerada."
    )

    # ---------------------------------------------------------------- 04
    st.subheader("04 · O sinal existe — o limiar anterior é que não o alcançava")
    histograma = carregar_histograma()
    limiar_ant = THRESH_MAG_ANTERIOR_MS2
    if histograma is None:
        st.info(
            "Rode `python src/validacao_hardware.py` para gerar "
            f"`{CAMINHO_HISTOGRAMA}` e esta figura aparece.", icon="ℹ️",
        )
    else:
        limiares_fig = {
            f"firmware (p99,9 normal) · {_br(limiar)} m/s²": limiar,
            f"limiar anterior · {_br(limiar_ant, 0)} m/s²": limiar_ant,
        }
        st.altair_chart(
            histograma_por_perfil(
                histograma, coluna_grupo="comportamento", ordem=ESTILOS,
                cores=dict(zip(ESTILOS, CORES)),
                rotulo_x="Magnitude da aceleração (m/s²)", unidade="amostras",
                limiares=limiares_fig, altura=120, largura=LARGURA_FIGURA,
                escala_log=True, casas_classe=2, mostrar_contagem=False,
            ),
            width="content",
        )
        n_amostras_total = int(histograma["n_amostras"].sum())
        st.caption(
            f"**Figura 2.** As {int_ptbr(n_amostras_total)} leituras do acelerômetro a 10Hz "
            "em todo o UAH-DriveSet — não mais um ponto por trajeto, um ponto por amostra —, "
            "em escala log porque a primeira classe concentra a maioria delas. O sinal existe: "
            "a cauda da distribuição agressiva se estende muito mais para a direita que a "
            "normal. O limiar anterior ficava depois de onde as três curvas já esvaziaram; "
            "o do firmware corta justamente essa cauda."
        )

        if True:
            st.markdown(
                "###### O que cada faixa de magnitude dispara hoje "
                "(faixas do sinal — sem ligação com o perfil do segurado)"
            )
            faixas = contagem_por_faixa(limiar_ant, limiar)
            if faixas is not None and faixas["n_total"]:
                n_total = faixas["n_total"]
                c1, c2, c3 = st.columns(3)
                _cartao(
                    c1, f"Abaixo de {_br(limiar)} m/s² (limiar do firmware)",
                    "Nenhum limiar dispara", "#0ca30c",
                    f"{_br(100 * faixas['n_abaixo_recal'] / n_total, 1)}% das amostras "
                    f"({int_ptbr(faixas['n_abaixo_recal'])} de {int_ptbr(n_total)}).",
                )
                _cartao(
                    c2, f"Entre {_br(limiar)} e {_br(limiar_ant, 0)} m/s²",
                    "Só dispara no firmware atual", "#eb6834",
                    f"{_br(100 * faixas['n_entre'] / n_total, 2)}% das amostras "
                    f"({int_ptbr(faixas['n_entre'])}) — é o que a troca de "
                    f"{_br(limiar_ant, 0)} para {_br(limiar)} m/s² passou a detectar.",
                )
                _cartao(
                    c3, f"Acima de {_br(limiar_ant, 0)} m/s² (limiar anterior)",
                    "Dispara nos dois", "#8a4b12",
                    f"Só {int_ptbr(faixas['n_acima_vigente'])} amostra(s) em "
                    f"{int_ptbr(n_total)} cruzam o limiar anterior — a Figura 2 acima "
                    "mostra a que rótulo elas pertencem.",
                )

    # ---------------------------------------------------------------- 05
    st.subheader("05 · Que limiar você deveria escolher")
    st.altair_chart(fig3_sensibilidade(df, limiar), width="content")
    faixa_lo, faixa_hi = LIT["pesado_lo"], LIT["passeio_lo"] + 0.55
    dentro = df[(df["pico_ms2"] >= faixa_lo)]
    st.caption(
        "**Figura 3.** Cada curva responde: programando o firmware com este limiar, que fração "
        "dos trajetos daquele rótulo geraria pelo menos um evento? Um bom limiar é onde as "
        "curvas se afastam — agressiva alta, normal perto do chão. No limiar anterior "
        f"({_br(THRESH_MAG_ANTERIOR_MS2, 0)} m/s²) as três já colapsaram em zero e o detector "
        "deixava de discriminar. Na faixa da literatura "
        f"({_br(faixa_lo)}–{_br(faixa_hi)} m/s²), "
        f"{int((dentro['estilo'] == 'agressiva').sum())} dos "
        f"{int((df['estilo'] == 'agressiva').sum())} trajetos agressivos disparam contra "
        f"{int((dentro['estilo'] == 'normal').sum())} dos "
        f"{int((df['estilo'] == 'normal').sum())} normais. É esta figura que justifica o limiar "
        "atual do firmware — com a curva na tese, a escolha deixa de ser arbitrária."
    )

    # ---------------------------------------------------------------- 06
    st.subheader("06 · O firmware atual contra o melhor par da varredura")
    grade = carregar_grade()
    if grade is None:
        st.info(
            "Rode `python src/validacao_hardware.py` para gerar "
            f"`{CAMINHO_GRADE}` e esta seção aparece.", icon="ℹ️",
        )
    else:
        vigente = ponto_operacao(df, df["eventos_min"] > 0, limiar, 0.0)
        candidatos = grade[grade["especificidade"] >= ESPECIFICIDADE_MINIMA]
        recomendado = _melhor_par(candidatos if not candidatos.empty else grade).to_dict()
        melhor_j = _melhor_par(grade).to_dict()

        st.altair_chart(fig5_antes_depois(vigente, recomendado), width="content")
        st.caption(
            f"**Figura 5.** À esquerda de cada linha, o que o firmware entrega hoje "
            f"(T = {_br(limiar)} m/s², qualquer evento sinaliza); à direita, o melhor par "
            f"da varredura (T = {_br(recomendado['limiar_ms2'])} m/s², "
            f"N = {_br(recomendado['taxa_min_eventos_min'])} eventos/min). A detecção de "
            f"condução agressiva iria de {_br(100 * vigente['sensibilidade'], 0)}% para "
            f"{_br(100 * recomendado['sensibilidade'], 0)}%, com "
            f"{int(vigente['normal_sinalizados'])} → {int(recomendado['normal_sinalizados'])} "
            "trajeto(s) normal(is) sinalizado(s). Ressalva: o par da direita foi escolhido "
            "e medido nos mesmos 40 trajetos; a validação deixando um motorista de fora "
            "(nota \"Regra do firmware\", acima) mostra quanto disso se mantém."
        )

        c1, c2, c3 = st.columns(3)
        _cartao(
            c1, "Quanto o detector discrimina hoje?",
            f"Youden {_br(vigente['youden'], 3)}", "#eb6834",
            f"Com o limiar do firmware ({_br(limiar)} m/s²). Zero é o valor de um "
            "detector que não separa os dois grupos — era onde o limiar anterior "
            f"({_br(THRESH_MAG_ANTERIOR_MS2, 0)} m/s²) ficava: nunca disparava.",
        )
        _cartao(
            c2, "E com o melhor par da varredura?",
            f"Youden {_br(recomendado['youden'], 3)}", "#0ca30c",
            f"Especificidade de {_br(100 * recomendado['especificidade'], 0)}%, acima do "
            f"piso de {_br(100 * ESPECIFICIDADE_MINIMA, 0)}% que a varredura impôs — "
            "falso positivo custa mais que falso negativo numa regra que vira preço.",
        )
        _cartao(
            c3, "E se otimizar só o Youden?",
            f"Youden {_br(melhor_j['youden'], 3)}", "#4b555b",
            f"T = {_br(melhor_j['limiar_ms2'])} m/s², N = "
            f"{_br(melhor_j['taxa_min_eventos_min'])}. Ganha "
            f"{_br(100 * (melhor_j['sensibilidade'] - recomendado['sensibilidade']), 0)} ponto(s) "
            "de sensibilidade e perde especificidade — não é a recomendação, é o teto.",
        )

        # ------------------------------------------------------------ 07
        st.subheader("07 · A varredura inteira, para conferência")
        st.altair_chart(fig6_grade_calibracao(grade, recomendado), width="content")
        n_pares = len(grade)
        st.caption(
            f"**Figura 6.** Os {n_pares} pares testados: cada célula é um firmware "
            f"possível, e a cor é o quanto aquele firmware separaria condução agressiva "
            f"de normal. O círculo branco é o par recomendado. O ponto importante para a "
            f"leitura não é o pico e sim a **região**: há uma faixa larga de limiares "
            f"entre {_br(grade['limiar_ms2'].min())} e "
            f"{_br(grade[grade['youden'] >= 0.6]['limiar_ms2'].max())} m/s² que funciona "
            f"quase igual, então a escolha não depende de acertar a segunda casa decimal. "
            f"O limiar do firmware ({_br(limiar)} m/s²) é uma das colunas da figura; o "
            f"anterior ({_br(THRESH_MAG_ANTERIOR_MS2, 0)} m/s²) fica fora da faixa varrida, "
            "à direita de todas elas."
        )
        st.caption(
            "A varredura usa apenas trajetos normais e agressivos. O rótulo sonolenta "
            "ficaria em cima de duas coisas ao mesmo tempo — ver a ressalva no fim da aba — "
            "e por isso entra na grade só como coluna de inspeção, nunca no cálculo do "
            "Youden. Grade completa e regra de desempate em `src/validacao_hardware.py`."
        )

    # ---------------------------------------------------------------- 08
    st.subheader("08 · O efeito se sustenta dentro do mesmo motorista")
    st.altair_chart(fig4_delta_por_motorista(df), width="content")
    st.caption(
        "**Figura 4.** Comparação pareada dentro de cada condutor, o que elimina a hipótese de "
        "que a diferença vem de uns dirigirem mais forte que outros. Sonolenta fica fora de "
        "propósito: o efeito não é consistente entre motoristas, e misturar os três estilos no "
        "mesmo gráfico foi o que tornou o painel anterior ilegível."
    )

    # ---------------------------------------------------------------- 09
    st.subheader("09 · Onde a base é fina")
    st.dataframe(tabela_cobertura(df), width="stretch", hide_index=True)
    st.caption(
        "Trajetos por motorista e rótulo; entre parênteses, a divisão entre rodovia (R) e via "
        "secundária (S). A célula que fica com 0 de um dos tipos não sustenta conclusão sobre via."
    )

    st.warning(
        "**O que este resultado não é**\n\n"
        "- **Não é detecção de sonolência validada.** O rótulo sonolenta é um estilo de condução "
        "instruído no momento da coleta, não uma medida fisiológica.\n"
        "- **Não é precisão por evento.** O UAH-DriveSet rotula o trajeto inteiro, não cada "
        "manobra; a matriz de confusão da aba é por trajeto — por evento ela não existe.\n"
        "- **Não separa frenagem de curva.** A detecção usa a magnitude sem decomposição por "
        "eixo; a distinção frenagem × aceleração só acontece depois, pelo sinal do eixo X "
        "calibrado, e depende da montagem física do sensor.\n"
        "- **Estilo, motorista e tipo de via estão só parcialmente cruzados.** Rodovia roda mais "
        "rápido que via secundária independentemente do estilo.\n"
        f"- **n pequeno e desigual.** {n_total} trajetos, "
        f"{df['motorista'].nunique()} motoristas.",
        icon="⚠️",
    )

    st.caption(
        "Dados: UAH-DriveSet — ROMERA, E.; BERGASA, L. M.; ARROYO, R. *Need data for driver "
        "behaviour analysis? Presenting the public UAH-DriveSet.* IEEE ITSC, 2016. "
        "Acelerômetro a 10 Hz, GPS a 1 Hz, rótulo por trajeto. Os códigos D1–D6 são do próprio "
        "dataset, não identificam pessoas."
    )
