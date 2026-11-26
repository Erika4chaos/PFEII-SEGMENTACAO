# Calibração do piso de jerk na grandeza nova

Fecha o item deixado em aberto no `config.h`: o `PISO_MAGNITUDE_JERK_MS2 = 1.45`
é **provisório**. A medição que o justificava (9 trancos, 21/09) foi feita na
grandeza antiga (|a|−g escalar) e não transfere para a subtração vetorial atual.

## Por que o piso existe

O jerk sozinho dispara com qualquer transiente rápido — buraco, módulo batendo
no suporte, tranco de mão — sem aceleração real por trás. Nenhum deles é
condução agressiva, mas todos entrariam na fila como evento e contaminariam a
**taxa de eventos/min**, que é a variável da análise comportamental. O piso
exige, junto do jerk, um mínimo de aceleração real.

## Regra de decisão

O piso tem de ficar **acima** do máximo de magnitude que um tranco de mão produz
(para ruído de bancada não virar evento) e **abaixo** do limiar principal de
2,9 m/s² (para a frenagem real de veículo ainda passar pelo caminho do jerk).

## Procedimento

1. Grave o firmware (`env:esp32dev`) e salve o serial em arquivo:
   ```
   pio device monitor -f log2file > log_bancada.txt
   ```
2. Faça **só trancos de mão** na placa — **sem incliná-la** (inclinar gera
   magnitude sustentada e sobe o número artificialmente). Faça ≥ 10 trancos
   variados.
3. Rode o parser:
   ```
   python calibrar_jerk.py log_bancada.txt
   ```
4. Ele lê as linhas `Evento #...` e `[diag]` do log, resume a magnitude e o
   jerk dos trancos e sugere o `PISO_MAGNITUDE_JERK_MS2`:
   - se os trancos ficam abaixo de 1,45 → mantém a regra "metade do limiar";
   - se passam de 1,45 → sugere subir o piso para o máximo + margem.
5. Ajuste o valor em `config.h` e rode `python src/validacao_hardware.py` para
   manter firmware, validação e dashboard no **mesmo** parâmetro (o script de
   validação lê o limiar do `config.h`).

## O que registrar para o TCC

- O máximo de magnitude de tranco de mão medido (número do serial).
- O valor final do piso e a justificativa (acima do ruído de bancada, abaixo da
  frenagem real).
- Que o piso **não** foi ajustado para maximizar métrica — é uma regra física
  simples de defender, coerente com o critério do limiar principal.
