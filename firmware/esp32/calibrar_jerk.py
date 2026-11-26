#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibracao do piso de jerk a partir do log serial da bancada.

Contexto (config.h): o criterio de jerk e AUXILIAR. Ele so dispara se, junto,
a magnitude real passar de PISO_MAGNITUDE_JERK_MS2. Sem esse piso, qualquer
tranco de mao, buraco ou modulo batendo no suporte viraria "evento de conducao
agressiva" e contaminaria a taxa de eventos/min -- a variavel da analise.

O valor atual (1,45 m/s^2) e PROVISORIO: segue a regra "metade do limiar
principal", mas a medicao que o justificava (9 trancos, 21/09) foi feita na
grandeza ANTIGA (|a|-g escalar) e nao transfere para a subtracao vetorial atual.
Este script reconfirma o piso na grandeza nova.

COMO USAR
---------
1. Grave o firmware (env:esp32dev) e salve o monitor serial num arquivo:
       pio device monitor -f log2file > log_bancada.txt
2. Faca APENAS trancos de mao na placa (NAO incline) -- o objetivo e medir o
   maximo de magnitude que um tranco de mao produz, para por o piso ACIMA dele.
   Faca pelo menos ~10 trancos variados.
3. Rode:
       python calibrar_jerk.py log_bancada.txt

O script le as linhas de evento e de [diag] do log, resume a magnitude e o jerk
dos trancos e sugere um PISO_MAGNITUDE_JERK_MS2 que exclua os trancos de mao,
mantendo-se abaixo do limiar principal (frenagem real de veiculo passa; ruido de
bancada, nao).

Sem dependencias externas (stdlib). Python 3.8+.
"""

import re
import sys
from statistics import mean

# Formatos impressos por src/main.cpp
RE_EVENTO = re.compile(
    r"Evento #(\d+):\s*magnitude=([-\d.]+)\s*m/s2\s*jerk=([-\d.]+)\s*m/s3\s*\(por (\w+)\)"
)
RE_DIAG = re.compile(
    r"\[diag\]\s*pico magnitude=([-\d.]+)\s*m/s2\s*pico \|jerk\|=([-\d.]+)\s*m/s3\s*eventos=(\d+)\s*limiar=([-\d.]+)"
)
RE_LIMIAR_CFG = 2.9  # LIMIAR_MAGNITUDE_MS2 (lido tambem do log, se presente)


def percentil(valores, p):
    """Percentil simples (interpolacao linear), sem numpy."""
    if not valores:
        return None
    xs = sorted(valores)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def main(caminho):
    eventos = []          # (id, magnitude, jerk, motivo)
    picos_magnitude = []  # pico de magnitude por janela [diag]
    picos_jerk = []       # pico |jerk| por janela [diag]
    limiar = RE_LIMIAR_CFG

    with open(caminho, "r", encoding="utf-8", errors="replace") as f:
        for linha in f:
            m = RE_EVENTO.search(linha)
            if m:
                eventos.append(
                    (int(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4))
                )
                continue
            d = RE_DIAG.search(linha)
            if d:
                picos_magnitude.append(float(d.group(1)))
                picos_jerk.append(float(d.group(2)))
                limiar = float(d.group(4))

    print("=" * 64)
    print("CALIBRACAO DO PISO DE JERK — resumo do log")
    print("=" * 64)
    print(f"Arquivo: {caminho}")
    print(f"Limiar principal (LIMIAR_MAGNITUDE_MS2): {limiar:.2f} m/s^2")
    print()

    # --- Magnitude dos trancos (o que define o piso) ---
    # Picos de janela > ~0,2 descartam as janelas de repouso; o que sobra e o
    # que um tranco de mao realmente produziu de aceleracao dinamica filtrada.
    picos_ativos = [v for v in picos_magnitude if v > 0.2]
    print(f"Janelas [diag] lidas: {len(picos_magnitude)} "
          f"({len(picos_ativos)} com atividade > 0,2 m/s^2)")
    print(f"Eventos registrados:  {len(eventos)}")
    print()

    if not picos_ativos and not eventos:
        print("!! Nenhum tranco encontrado no log. Garanta que o monitor estava")
        print("   salvando em arquivo e que houve trancos de mao na placa.")
        return 1

    mags_evento = [e[1] for e in eventos]
    base_magnitude = picos_ativos + mags_evento

    mx = max(base_magnitude)
    p95 = percentil(base_magnitude, 95)
    p99 = percentil(base_magnitude, 99)
    print("Magnitude dos trancos de mao (m/s^2):")
    print(f"  media {mean(base_magnitude):.2f} | p95 {p95:.2f} | "
          f"p99 {p99:.2f} | MAXIMO {mx:.2f}")
    print()

    if picos_jerk:
        pj_ativos = [v for v in picos_jerk if v > 1.0]
        if pj_ativos:
            print("Jerk dos trancos (m/s^3):")
            print(f"  media {mean(pj_ativos):.1f} | "
                  f"p95 {percentil(pj_ativos, 95):.1f} | MAXIMO {max(pj_ativos):.1f}")
            print(f"  (LIMIAR_JERK_MS3 atual = 20,0; trancos de mao o ultrapassam "
                  f"com folga, como esperado)")
            print()

    # --- Recomendacao do piso ---
    # O piso precisa ficar ACIMA do maximo de magnitude de um tranco de mao
    # (para o ruido de bancada nao virar evento) e ABAIXO do limiar principal
    # (para a frenagem real de veiculo ainda passar pelo caminho do jerk).
    margem = 0.15 * mx
    piso_sugerido = round(mx + margem, 2)
    meio_limiar = round(limiar / 2.0, 2)

    print("-" * 64)
    print("RECOMENDACAO")
    print("-" * 64)
    print(f"Maximo de magnitude de tranco de mao: {mx:.2f} m/s^2")
    print(f"Piso sugerido (max + 15% de margem):  {piso_sugerido:.2f} m/s^2")
    print(f"Metade do limiar principal:           {meio_limiar:.2f} m/s^2")
    print()

    if mx >= limiar:
        print("ATENCAO: algum tranco de mao cruzou o LIMIAR PRINCIPAL de magnitude.")
        print("  Provavel inclinacao acidental da placa durante o tranco. Refaca a")
        print("  coleta mantendo a placa nivelada (so trancos, sem inclinar).")
    elif piso_sugerido <= meio_limiar:
        print(f"-> Pode manter PISO_MAGNITUDE_JERK_MS2 = {meio_limiar:.2f} ")
        print(f"   (regra 'metade do limiar'): ele ja fica acima dos trancos de mao")
        print(f"   ({mx:.2f}) e a frenagem real de veiculo passa.")
    else:
        print(f"-> Suba PISO_MAGNITUDE_JERK_MS2 para ~{piso_sugerido:.2f} ")
        print(f"   em config.h: os trancos de mao chegam a {mx:.2f} m/s^2 e")
        print(f"   passariam pelo piso atual de metade do limiar ({meio_limiar:.2f}).")
    print()
    print("Depois de ajustar config.h, rode `python src/validacao_hardware.py`")
    print("para manter firmware, validacao e dashboard no mesmo parametro.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("uso: python calibrar_jerk.py <log_bancada.txt>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
