# Validação de hardware — nó de telemetria ESP32

Registro do diagnóstico feito em 19–20/09/2026.

## Situação atual

**Tudo validado, menos um detalhe mecânico.** O firmware compila, grava e roda;
a fiação está correta; o barramento I2C funciona; o MPU-6050 e o OLED
respondem; a calibração de gravidade funciona; a detecção de eventos dispara e
grava na fila.

O único bloqueio restante: **as barras de pinos dos módulos não estão
soldadas**. Os módulos ficam apoiados nos pinos, e o contato abre com qualquer
movimento.

### O que falta fazer

1. Soldar as barras de pinos do **MPU-6050**, do **GPS** e conferir o **OLED**
   (este aparentemente já veio soldado).
2. Reconectar, apertar EN e deixar a placa **parada** durante os ~4 s de
   calibração, até aparecer "Calibracao concluida. Deteccao de eventos ativa."
3. Verificar o repouso: nenhum evento deve aparecer com a placa imóvel.
4. Dar um tranco firme: deve sair `Evento #1` com magnitude variável e jerk
   diferente de zero.
5. Testar o GPS a céu aberto (ver seção própria abaixo).

## Por que soldar não é opcional

O teste final deu a prova empírica: ao sacudir a placa para testar o
acelerômetro, o MPU se desconectou e o log voltou a encher de
`i2cWriteReadNonStop returned Error -1`.

Ou seja: **o sensor não sobrevive ao próprio trabalho.** A função dele é sentir
aceleração, e o ato de sentir aceleração desfaz a ligação. Como o nó é
embarcado em veículo, sob vibração contínua, a falha deixa de ser risco e passa
a ser certeza.

Há um agravante metodológico, e é o que mais importa para o TCC: contato
intermitente num acelerômetro não apenas interrompe a leitura, como pode gerar
transientes que o firmware classifica como frenagem brusca. O resultado seriam
eventos falsos na fila — dados contaminados sustentando a análise, e um
problema de solda disfarçado de problema de metodologia.

## Linha do tempo do diagnóstico

Fica registrada porque cada etapa eliminou uma hipótese com evidência.

### 1. Fiação (multímetro, continuidade)

Alimentação, GPS, MPU-6050 e OLED conferidos ponto a ponto. Sem curto entre VCC
e GND. **3,32 V** medidos no pino 3V3 com a placa energizada. Tudo aprovado.

### 2. Dois bugs de compilação

O firmware nunca tinha sido compilado — o próprio README avisava isso em
"Limitações conhecidas". Dois erros reais, ambos corrigidos:

**`include/sensores.h`** usava `JANELA_MEDIA_MOVEL` na declaração do buffer sem
incluir o `config.h`, onde a macro é definida:

```
include/sensores.h:29:30: error: 'JANELA_MEDIA_MOVEL' was not declared in this scope
```

O `main.cpp` escapava por acaso, porque inclui `config.h` antes de `sensores.h`;
o `sensores.cpp` inclui `sensores.h` primeiro e quebrava. Corrigido com um
`#include "config.h"` no próprio header.

**`include/gps_leitor.h`** — os getters do TinyGPS++ (`lat()`, `lng()`,
`date.year()`, `time.hour()`…) não são `const`, porque limpam a flag interna de
"updated". Como `latitude()`, `longitude()` e `epocaAtual()` são métodos
`const`, dava erro em 8 linhas:

```
error: passing 'const TinyGPSLocation' as 'this' argument discards qualifiers
```

Corrigido com `mutable TinyGPSPlus _gps;`, que preserva a API pública `const` e
é a solução idiomática para o caso.

Depois disso: RAM 14,1%, Flash 63,8%.

### 3. Porta serial inexistente

A gravação falhava com `Could not open COM4 ... semaphore timeout (121)`. A
lista de portas mostrava apenas COM3 e COM4, **ambas Bluetooth** — o ESP32 não
aparecia, e o PlatformIO tentava gravar numa porta Bluetooth por ser a única
existente.

O Windows tocava o som de dispositivo conectado ao plugar a placa, o que provou
que o cabo tinha fios de dados e que a placa era enumerada no nível USB.
Faltava o driver do conversor USB-serial. Instalado o **CP210x da Silicon
Labs**, a placa passou a aparecer como `COM5 — Silicon Labs CP210x USB to UART
Bridge`, VID:PID 10C4:EA60.

### 4. Gravação impossível dentro da protoboard

Mesmo com a porta certa, o upload falhava em `Failed to connect to ESP32: No
serial data received`. Com o **ESP32 fora da protoboard**, gravou de primeira,
inclusive com auto-reset por RTS funcionando.

Ao reencaixar a placa com cuidado, a gravação passou a funcionar também
montada. Era assentamento.

### 5. Barramento I2C mudo — a parte difícil

Sintoma: `i2cWriteReadNonStop returned Error -1` em loop, nenhum dispositivo
respondendo.

O que **não** era, cada um eliminado com evidência:

| Hipótese | Como foi eliminada |
|---|---|
| Falta de alimentação | 3,33 V medidos nos três módulos; LED do MPU aceso |
| Terra separado | Continuidade GND(ESP32) ↔ GND(MPU) confirmada |
| Fios em GPIO errado | Varredura automática de 342 pares de pinos, nada |
| SDA e SCL trocados | O par (22,21) estava incluído na varredura |
| Falta de pull-up | Linhas em repouso medidas a 3,28 V e 3,30 V |
| Pinos do ESP32 danificados | Teste em dreno aberto: o ESP32 puxa e solta as duas linhas |
| EN preso em nível alto | Continuidade EN ↔ 3V3 e EN ↔ GND: sem curto |

A pista decisiva foi o **comportamento intermitente**: os resultados mudavam de
ciclo para ciclo sem ninguém mexer. Ora o SDA falhava, ora o SCL, ora ambos
passavam. Isso só acontece com mau contato.

E havia uma armadilha nos testes: **o multímetro mente nesse caso.** A ponta de
prova faz a pressão que o jumper apoiado não faz, então a continuidade apita
enquanto o circuito real não fecha. Foi por isso que tudo "passava" e nada
funcionava.

A confirmação veio do LED do MPU, que **piscava ao mexer na placa** em vez de
ficar aceso — alimentação intermitente. Causa: barra de pinos sem solda.

### 6. Validação dos módulos

Com os contatos reassentados, os dois apareceram (em ciclos diferentes, nunca
juntos — mais um sintoma do contato precário):

```
>> 0x68 respondeu  <- MPU-6050 (AD0 em nivel baixo)
>> 0x3C respondeu  <- display OLED SSD1306
```

**Nenhum módulo tem defeito.**

### 7. Firmware rodando

```
No de telemetria RCT -- inicializando
Calibrando o offset de gravidade -- mantenha o veiculo parado...
Calibracao concluida. Deteccao de eventos ativa.
```

Sem "Falha ao iniciar o MPU-6050" — o sensor inicializou.

Na primeira tentativa saíram eventos falsos, todos idênticos:

```
Evento #22: magnitude=10.38 m/s2 jerk=0.00 m/s3
```

10,38 ÷ 9,80665 = **1,06 g**, exatamente a gravidade. Isso só ocorre com
`_offsetGravidade_g` igual a zero: o módulo estava desencaixado durante os 4 s
de calibração, o firmware tirou a média de zeros, e passou a interpretar a
gravidade como tranco permanente.

Com o módulo encaixado e a placa parada no boot, a calibração saiu correta e o
repouso ficou **silencioso**, sem nenhum evento — o comportamento esperado.

## Melhoria sugerida no firmware (rende parágrafo em banca)

O firmware confia na calibração cegamente. Uma falha de sensor durante aqueles
4 segundos contamina o limiar em silêncio, e o sistema segue gerando eventos
falsos com aparência de normalidade — como aconteceu aqui.

Uma verificação de sanidade ao fim de `SensorAceleracao::calibrar()` resolveria:
o offset precisa ficar próximo de 1 g (algo como 0,8 a 1,2). Fora dessa faixa,
avisar pela serial e repetir a calibração em vez de seguir adiante. É barato de
implementar e cobre uma falha silenciosa que compromete os dados.

## Sobre o GPS

O GPS **não foi validado** — ele não aparece no scanner I2C porque não é um
dispositivo I2C: o NEO-6M fala por **serial (UART)**, nos GPIOs 16 e 17. Quem o
exercita é o firmware de telemetria.

Ao testar, lembrar que o NEO-6M precisa de **céu aberto ou janela** e que o
primeiro fix pode levar minutos. O firmware funciona sem fix (usa tempo
relativo em `epocaAtual()`), então a ausência de sinal não trava o teste do
acelerômetro.

## Sobre o OLED

O firmware **não usa o OLED**: não há biblioteca de display no código e a
tabela de ligações do README lista apenas MPU-6050 e GPS. Ele está no
barramento sem ninguém falar com ele, e por isso permanece apagado — o que é o
comportamento esperado, não defeito. Vale decidir se ele entra no escopo ou se
sai da montagem.

## Ferramentas de diagnóstico deixadas no projeto

`src/i2c_scanner.cpp` e o ambiente `[env:scanner_i2c]` no `platformio.ini`. Não
fazem parte do firmware: o `build_src_filter` do `[env:esp32dev]` exclui o
scanner, e o do `[env:scanner_i2c]` compila só ele.

O scanner roda em loop e imprime a cada 8 segundos:

- **Etapa 0** — teste elétrico das linhas: confere pull-up e se o ESP32
  consegue puxar e soltar SDA e SCL (usa dreno aberto e lê o nível real).
- **Etapa 1** — varredura de endereços nos pinos configurados.
- **Etapa 2** — se nada responder, procura os módulos em 342 pares de GPIO.

Como imprime continuamente, dá para apertar um fio e ver o resultado mudar na
tela — foi assim que o mau contato apareceu. **Na protoboard, confie no
comportamento, não na continuidade.**

Para usar: trocar o ambiente para `env:scanner_i2c` na barra inferior do VS
Code, gravar, abrir o monitor. Para voltar ao firmware, trocar para
`env:esp32dev`. Quando não forem mais necessários, dá para apagar
`src/i2c_scanner.cpp`, o bloco `[env:scanner_i2c]` e a linha `build_src_filter`
do `[env:esp32dev]`.

## Lembrete

`WIFI_CONFIAVEL_SENHA` e `MQTT_BROKER_HOST` no `config.h` ainda são
placeholders. Não atrapalha o teste de bancada — os eventos continuam sendo
detectados e enfileirados no LittleFS mesmo sem rede.
