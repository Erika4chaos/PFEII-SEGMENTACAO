"""Aba 2 do dashboard — Validação de Hardware (PFE II, Parte B.5).

Ordem de leitura (é o ponto da reescrita — a versão antiga mostrava o
resultado antes de dizer o que estava sendo testado):

    01  a resposta, em linguagem de negócio, antes de qualquer gráfico
    02  glossário dos oito termos que as figuras usam
    03  onde o limiar do firmware cai na régua da literatura
    04  distribuição do pico por rótulo comportamental
    05  curva de sensibilidade — que limiar escolher
    06  efeito pareado dentro do mesmo motorista
    07  cobertura da base e o que este resultado não é

Referências das faixas da Figura 1 (todas em desaceleração brusca):
  * padrões DOT: 0,25–0,78 g para veículos de passeio; ~0,20 g para pesados;
  * Desai et al. e Mahmud & Day: ~0,27 g;
  * Kamla et al. (frota mista, Reino Unido): ~0,23 g pesados, ~0,45 g leves.
O limiar do projeto (6 m/s² = 0,61 g) vem da Seção 2.8 do relatório
(MASELLO et al., 2025; BRÜHWILER et al., 2022).

Uso:
    from validacao_hardware import THRESH_MAG_MS2
    from dashboard.tab_validacao_hardware import render, carregar
    render(carregar("data/processed/uah_trips.csv"), limiar=THRESH_MAG_MS2)

O limiar é sempre recebido como parâmetro: não existe valor padrão aqui nem
em charts_validacao, para que THRESH_MAG_MS2 continue sendo a única definição
do número no projeto.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from .charts_validacao import (
    G, LIT, ESTILOS,
    fig1_escala_do_limiar, fig2_pico_por_estilo, fig3_sensibilidade,
    fig4_delta_por_motorista, tabela_cobertura,
)

COLUNAS = ["trajeto_id", "motorista", "estilo", "via",
           "duracao_min", "pico_ms2", "eventos", "eventos_min"]


RAIZ_PROJETO = Path(__file__).resolve().parent.parent


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
            "Rode validacao_hardware.validate_against_uah_driveset() primeiro."
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
        c2, "Com o limiar atual, o dispositivo registraria esses eventos?",
        "Quase nunca" if acima <= n_total * 0.1 else "Em parte", "#eb6834",
        f"{acima} de {n_total} trajetos chegam a {_br(limiar, 0)} m/s². "
        f"A métrica de produto — eventos/min — fica em zero em {_br(pct_zero, 0)}% da base.",
    )
    _cartao(
        c3, "Isso já serve para alimentar precificação por uso?",
        "Ainda não", "#eb6834",
        f"Antes é preciso recalibrar o limiar para a faixa da literatura de "
        f"veículos pesados ({_br(LIT['pesado_lo'])}–{_br(LIT['pesado_hi'])} m/s²) "
        "e refazer a contagem.",
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
        ("magnitude da aceleração", "‖(x,y,z)‖ do acelerômetro depois do filtro passa-baixa e da "
                                    "subtração da gravidade. Sem eixo: uma curva brusca entra na "
                                    "conta igual a uma frenagem."),
        ("pico do trajeto", "O maior valor dessa magnitude no trajeto inteiro. Um número por "
                            "trajeto. É a régua das Figuras 1, 2 e 4."),
        ("limiar", f"O valor a partir do qual o firmware chama a leitura de evento. Hoje: "
                   f"{_br(limiar, 0)} m/s². Não é constante da física — é escolha de projeto, "
                   "e é ela que está em teste aqui."),
        ("evento", "Uma passagem acima do limiar. É a única coisa que o dispositivo grava e "
                   "transmite; o sinal bruto nunca sai da borda."),
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
        f"**Figura 1.** {_br(limiar, 0)} m/s² equivalem a {_br(limiar / G, 2)} g. "
        "Para caminhões pesados — a frota que esta apólice cobre — os estudos de telemática "
        "trabalham entre 0,20 e 0,23 g; as revisões de veículos de passeio usam de 0,25 a "
        "0,78 g, com a maioria dos trabalhos perto de 0,27 g. O limiar do projeto está no topo "
        "dessa amplitude e cerca de três vezes acima da referência de veículo pesado — o que "
        "explica, sozinho, por que a contagem de eventos zera."
    )

    # ---------------------------------------------------------------- 04
    st.subheader("04 · O sinal existe — o limiar é que não o alcança")
    st.altair_chart(fig2_pico_por_estilo(df), width="content")
    st.caption(
        "**Figura 2.** Um ponto por trajeto; o traço escuro é a mediana do grupo, com o valor "
        "escrito e o n no rótulo da linha. As linhas estão ordenadas da menor para a maior "
        "mediana. Sonolenta acima de agressiva não é erro de pipeline: condução sonolenta "
        "produz correções tardias de trajetória — desvios bruscos de volante e frenagens de "
        "recuperação — que a magnitude sem decomposição por eixo registra igual a uma frenagem forte."
    )

    # ---------------------------------------------------------------- 05
    st.subheader("05 · Que limiar você deveria escolher")
    st.altair_chart(fig3_sensibilidade(df, limiar), width="content")
    faixa_lo, faixa_hi = LIT["pesado_lo"], LIT["passeio_lo"] + 0.55
    dentro = df[(df["pico_ms2"] >= faixa_lo)]
    st.caption(
        "**Figura 3.** Cada curva responde: programando o firmware com este limiar, que fração "
        "dos trajetos daquele rótulo geraria pelo menos um evento? Um bom limiar é onde as "
        "curvas se afastam — agressiva alta, normal perto do chão. No limiar atual as três já "
        "colapsaram em zero e o detector deixa de discriminar. Na faixa da literatura "
        f"({_br(faixa_lo)}–{_br(faixa_hi)} m/s²), "
        f"{int((dentro['estilo'] == 'agressiva').sum())} dos "
        f"{int((df['estilo'] == 'agressiva').sum())} trajetos agressivos disparam contra "
        f"{int((dentro['estilo'] == 'normal').sum())} dos "
        f"{int((df['estilo'] == 'normal').sum())} normais. É esta figura que justifica um valor "
        "novo em `THRESH_MAG_MS2` — com a curva na tese, a escolha deixa de ser arbitrária."
    )

    # ---------------------------------------------------------------- 06
    st.subheader("06 · O efeito se sustenta dentro do mesmo motorista")
    st.altair_chart(fig4_delta_por_motorista(df), width="content")
    st.caption(
        "**Figura 4.** Comparação pareada dentro de cada condutor, o que elimina a hipótese de "
        "que a diferença vem de uns dirigirem mais forte que outros. Sonolenta fica fora de "
        "propósito: o efeito não é consistente entre motoristas, e misturar os três estilos no "
        "mesmo gráfico foi o que tornou o painel anterior ilegível."
    )

    # ---------------------------------------------------------------- 07
    st.subheader("07 · Onde a base é fina")
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
        "manobra; não existe matriz de confusão possível.\n"
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
