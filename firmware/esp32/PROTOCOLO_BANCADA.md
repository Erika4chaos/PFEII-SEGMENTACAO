# Protocolo de validação em bancada — nó de telemetria RCT

Roteiro para exercitar o firmware modular (`[env:esp32dev]`) no hardware real,
registrar os resultados de forma defensável para o TCC e **validar o
store-and-forward** (persistência após queda de energia). Última revisão:
28/10/2026.

> **Qual firmware está sob teste.** Este protocolo testa o firmware de
> produção — os módulos em `src/` (sensor, GPS, fila, sincronização, OLED),
> ambiente `[env:esp32dev]`. Ele **não** contém o botão de pânico nem os 3 LEDs
> de status: esses existem só no sketch unificado do Wokwi e ainda não foram
> portados para o `main.cpp` (item em aberto). Nos casos abaixo, pânico e LEDs
> ficam fora de escopo; o que o firmware modular aciona é o **OLED** e a
> **fila em LittleFS**.

---

## 0. Pré-requisitos (bloqueio conhecido: solda)

O diagnóstico de 19–20/09 provou que o contato intermitente das barras de pino
**não soldadas** derruba o I²C e, pior, pode gerar transientes que o firmware
classifica como evento — dado contaminado disfarçado de metodologia. Então,
antes de qualquer caso:

1. **Soldar** as barras de pino do **MPU-6050** e do **GPS**; conferir o
   **OLED** (costuma vir soldado).
2. Energizar e medir **3,3 V** no pino 3V3 (multímetro) — mas lembrar: *na
   protoboard, confie no comportamento no serial, não na continuidade do
   multímetro* (a ponta de prova faz a pressão que o jumper apoiado não faz).
3. VS Code / PlatformIO: ambiente **`esp32dev`** selecionado na barra inferior.
4. Gravar e abrir o monitor:
   ```
   pio run -t upload
   pio device monitor      # 115200 baud
   ```
5. Deixar o monitor **salvando em arquivo** (necessário para a calibração do
   jerk — ver `CALIBRACAO_JERK.md`). No menu do monitor serial, ou:
   ```
   pio device monitor -f log2file > log_bancada_AAAA-MM-DD.txt
   ```

---

## 1. Como ler a saída serial

O firmware imprime três coisas úteis:

| Linha no serial | Significado |
|---|---|
| `Calibracao concluida (gravidade = X.XXX g). Deteccao de eventos ativa.` | Calibração OK, offset plausível |
| `ALERTA: calibracao suspeita -- norma ... = X.XXX g` | Offset ruim (módulo solto ou placa em movimento) — **reinicie parado** |
| `Evento #N: magnitude=... jerk=... (por magnitude\|jerk)` | Evento detectado e **gravado na fila** (id N persistido) |
| `[diag] pico magnitude=... pico \|jerk\|=... eventos=N limiar=2.9` | Resumo a cada 5 s: pico da janela e total de eventos |

No OLED: `TAXA /min`, `EVENTOS`, `ULTIMO`, `FILA` (pendentes não enviados),
`sat`/`km/h`.

---

## 2. Casos de teste

Marque cada caso na planilha da Seção 4.

### Caso 1 — Boot e barramento I²C
- **Ação:** apertar EN (reset) com a placa parada.
- **Esperado:** sem `Falha ao iniciar o MPU-6050`; o OLED mostra `CALIBRANDO...`.
- *(Se quiser a varredura explícita de endereços 0x68/0x3C, trocar para o
  ambiente `[env:scanner_i2c]`, gravar, conferir e voltar para `esp32dev`.)*

### Caso 2 — Calibração de gravidade (sanidade)
- **Ação:** manter a placa **imóvel e nivelada** durante os ~4 s de calibração.
- **Esperado:** `Calibracao concluida (gravidade = X.XXX g)` com **X entre 0,80
  e 1,20** (faixa `CALIB_NORMA_*_G`). Anotar o valor.
- **Falha proposital (opcional, rende parágrafo de banca):** reiniciar movendo
  a placa de leve → deve sair o `ALERTA: calibracao suspeita`. Comprova a
  checagem de sanidade.

### Caso 3 — Repouso (sem falso positivo)
- **Ação:** deixar a placa parada por **60 s** após a calibração.
- **Esperado:** **nenhum** `Evento #`. No `[diag]`, `pico magnitude` próximo de
  0 e `eventos` constante.
- *Este é o caso que o contato ruim quebrava: se aparecerem eventos idênticos
  com `jerk=0.00`, a calibração pegou o módulo solto — resolder/reassentar.*

### Caso 4 — Evento por inclinação (gatilho confiável de bancada)
> A magnitude de 2,9 m/s² **não é alcançável com a mão** — a média móvel
> absorve o impulso de ~100 ms de um tranco. Para gerar eventos de forma
> **repetível** na bancada, use a **inclinação**: depois de calibrar na
> horizontal, incline a placa ~30–45° e segure. A gravidade projetada muda, a
> subtração vetorial passa a ver aceleração dinâmica sustentada e cruza o
> limiar com folga.
- **Ação:** inclinar a placa e segurar ~1 s; repetir 5 vezes, respeitando ~2 s
  entre uma e outra (período refratário).
- **Esperado:** um `Evento #N` por inclinação, `magnitude` acima de 2,9,
  `(por magnitude)`. Os ids crescem **sem repetir**.

### Caso 5 — Evento por jerk (tranco manual)
- **Ação:** dar trancos secos na placa (sem incliná-la).
- **Esperado:** pode sair `Evento #N ... (por jerk)` quando `|jerk| > 20` **e**
  `magnitude > 1,45`. Anotar no serial o `pico |jerk|` e o `pico magnitude` de
  cada tranco — **esses números alimentam a calibração do jerk** (ver
  `CALIBRACAO_JERK.md`). Faça **pelo menos 10 trancos** aqui.

### Caso 6 — GPS a céu aberto
- **Ação:** levar a placa à janela/céu aberto; aguardar o primeiro fix (pode
  levar minutos).
- **Esperado:** no OLED, `-- sat` → `N sat` e `km/h` em vez de `sem fix`. O
  firmware funciona sem fix (usa tempo relativo), então a ausência de sinal não
  trava os outros casos.

### Caso 7 — Store-and-forward: persistência após queda de energia  ⭐
> É o caso que valida a correção de 28/10 (contador de eventos persistido).
> Objetivo: provar que, ao **desconectar e reconectar a fonte**, o dispositivo
> **retoma a contagem de onde parou** — sem reiniciar os ids em 1 e sem perder
> os eventos ainda não enviados.

1. **Gerar eventos:** a partir do repouso, provoque **3 eventos** por
   inclinação (Caso 4). Anote o id do último: `Evento #A` (ex.: `#3`). No OLED,
   `EVENTOS 3` e `FILA 3` (3 pendentes, pois não há rede/broker na bancada).
2. **Cortar energia:** **desconecte a fonte/USB** (não use o EN — tem que ser
   corte real de alimentação). Aguarde ~5 s.
3. **Religar:** reconecte. Mantenha a placa **parada** durante a calibração.
4. **Conferir a retomada:**
   - **Repouso limpo:** nenhum evento espúrio ao religar (Caso 3).
   - **Fila preservada:** o OLED deve voltar a mostrar `FILA 3` — os 3 eventos
     continuam no LittleFS (não foram enviados, logo não sumiram).
   - **Contagem contínua:** provoque **mais 1 evento** por inclinação. Deve sair
     **`Evento #4`** (A+1) — **não** `Evento #1`. É a prova de que o contador
     sobreviveu ao corte. No OLED, `EVENTOS 4`, `FILA 4`.
5. **Registrar:** anote os ids antes (3) e depois (4) na planilha. Se o id
   voltasse a 1, a correção teria falhado — reportar.

> **Por que isso importa (para o texto):** o id é a chave de deduplicação na
> nuvem (`dispositivo + id`). Se reiniciasse em 1 a cada religamento, a nuvem
> trataria eventos novos como duplicatas e os **descartaria** — perda
> silenciosa de dado. O contador agora é gravado em flash **antes** do evento,
> então, no pior caso de um corte entre as duas escritas, perde-se no máximo um
> *número* de id (uma lacuna inofensiva), nunca um id repetido.

### Caso 8 — (Opcional) Drenagem por MQTT
Só se houver broker MQTT e Wi-Fi na bancada. Preencher `WIFI_CONFIAVEL_*` e
`MQTT_BROKER_*` no `config.h`, subir um broker local (ex.: Mosquitto) e assinar
`rct/telemetria/eventos`. Esperado: a `FILA` zera conforme o cursor avança; ao
cortar a rede no meio, retoma do mesmo ponto na próxima janela (QoS 1, sessão
persistente). Sem broker, pular — o Caso 7 já valida a persistência local.

---

## 3. Critérios de aprovação

- Caso 2: norma de gravidade entre 0,80 e 1,20 g.
- Caso 3: 0 eventos em 60 s de repouso.
- Caso 4: 1 evento por inclinação, ids monotônicos.
- Caso 7: após religar, `FILA` preservada **e** próximo id = anterior + 1.

---

## 4. Planilha de registro

| Caso | Data/hora | Resultado esperado | Observado | OK? |
|---|---|---|---|---|
| 1 Boot/I²C | | MPU inicia, OLED calibrando | | |
| 2 Calibração | | gravidade ___ g (0,8–1,2) | | |
| 3 Repouso 60 s | | 0 eventos | | |
| 4 Inclinação ×5 | | 5 eventos, ids ___→___ | | |
| 5 Trancos ×10 | | picos de jerk/magnitude anotados | | |
| 6 GPS | | ___ satélites, fix em ___ min | | |
| 7a S&F antes | | EVENTOS ___, FILA ___ | | |
| 7b S&F religado | | FILA preservada = ___ | | |
| 7c S&F contagem | | próximo id = ___ (= anterior+1) | | |
| 8 MQTT (opc.) | | fila drena, cursor avança | | |

**Versão do firmware testada (git):** `________` &nbsp;&nbsp; **Placa:** `________`
