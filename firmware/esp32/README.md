# Firmware do no de telemetria (ESP32 + GPS + MPU-6050)

Implementa a Secao B do escopo: amostragem/filtragem do acelerometro,
deteccao de eventos de conducao brusca por limiar de magnitude/jerk, fila
persistente em LittleFS e sincronizacao oportunista via MQTT (QoS 1, sessao
persistente) com uma rede Wi-Fi confiavel pre-cadastrada.

## Ligacoes (bill of materials da Secao B.1)

| Modulo | Pino do modulo | Pino do ESP32 |
|---|---|---|
| MPU-6050 / GY-521 | VCC | 3V3 |
| MPU-6050 / GY-521 | GND | GND |
| MPU-6050 / GY-521 | SDA | GPIO 21 |
| MPU-6050 / GY-521 | SCL | GPIO 22 |
| GPS NEO-6M | VCC | 5V (ou 3V3, conforme o modulo) |
| GPS NEO-6M | GND | GND |
| GPS NEO-6M | TX | GPIO 16 (RX2) |
| GPS NEO-6M | RX | GPIO 17 (TX2) |
| OLED SSD1306 128x64 | VCC | 3V3 |
| OLED SSD1306 128x64 | GND | GND |
| OLED SSD1306 128x64 | SDA | GPIO 21 (compartilhado com o MPU) |
| OLED SSD1306 128x64 | SCL | GPIO 22 (compartilhado com o MPU) |

Os pinos e limiares estao centralizados em [`include/config.h`](include/config.h).

## Build e flash (PlatformIO)

```
cd firmware/esp32
pio run -t upload
pio device monitor
```

Antes de gravar em um dispositivo real, edite em `config.h`:

- `WIFI_CONFIAVEL_SSID` / `WIFI_CONFIAVEL_SENHA`: rede da garagem/base da
  transportadora (nao commitar credenciais reais).
- `MQTT_BROKER_HOST` / `MQTT_BROKER_PORTA` / `MQTT_TOPICO_EVENTOS`: broker
  de destino.
- `LIMIAR_MAGNITUDE_MS2`: mantido em 6.0 m/s^2 para ficar comparavel ao
  limiar padrao usado em `src/validacao_hardware.py` contra o UAH-DriveSet.

## Regra de deteccao de evento

Um evento e registrado quando, fora do periodo refratario de 2 s:

```
magnitude > 6,0 m/s^2
  OU
(|jerk| > 20 m/s^3  E  magnitude > 3,0 m/s^2)
```

O jerk e criterio **auxiliar**, nao alternativo. Ele existe para capturar a
manobra cujo pico a media movel de 100 ms suavizou abaixo do limiar principal,
e por isso exige um piso de aceleracao real junto. Sem esse piso, qualquer
transiente rapido sem aceleracao por tras -- um buraco na pista, o modulo
batendo no suporte, um chacoalhao manual na bancada -- entraria na fila como
evento de conducao agressiva.

O valor do piso vem de medicao: em 9 trancos manuais na protoboard montada
(21/09/2026), o jerk alcancou de 20,9 a 81,3 m/s^3, enquanto a magnitude
filtrada nunca passou de 1,90 m/s^2. Com o piso em 3,0 m/s^2, metade do limiar
principal, nenhum desses trancos e registrado.

**Consequencia para o teste de bancada:** o caminho da magnitude nao e
alcancavel com a mao -- a media movel absorve quase todo o impulso de ~100 ms
de um tranco manual, enquanto uma frenagem de veiculo sustenta a aceleracao por
perto de um segundo e sobrevive ao filtro. A validacao em bancada exercita a
cadeia completa (sensor, filtro, fila, sincronizacao, painel), mas o limiar de
6,0 m/s^2 so pode ser exercitado em veiculo.

## Calibracao

Na inicializacao (`setup()`), o firmware roda uma rotina de calibracao de
alguns segundos (`AMOSTRAS_CALIBRACAO` em `config.h`): o veiculo deve estar
parado nesse intervalo. A media da magnitude do vetor de aceleracao nesse
periodo vira o offset gravitacional subtraido de toda leitura subsequente
(ver `SensorAceleracao::calibrar()`), sem qualquer fusao com o giroscopio.

## Estrutura

- `src/sensores.*` -- leitura do MPU-6050, media movel, calibracao de
  gravidade, calculo de magnitude dinamica e jerk.
- `src/gps_leitor.*` -- parsing NMEA via TinyGPS++, latitude/longitude e
  timestamp do evento.
- `src/fila_eventos.*` -- fila binaria em LittleFS com cursor persistido
  (store-and-forward, sobrevive a queda de energia/conectividade).
- `src/sincronizacao.*` -- conexao oportunista a rede confiavel e drenagem
  da fila via MQTT QoS 1 / sessao persistente, com deduplicacao no lado da
  nuvem por um id unico (dispositivo + contador do evento).
- `src/main.cpp` -- loop principal: amostra, detecta, enfileira e tenta
  sincronizar periodicamente.

## Limitacoes conhecidas

- Nao houve acesso a hardware fisico neste ambiente para compilar/flashear
  e validar em bancada -- revise as APIs das bibliotecas (`TinyGPSPlus`,
  `MPU6050_light`, `256dpi/MQTT`) contra as versoes efetivamente instaladas
  antes do primeiro upload.
- Bateria/consumo de energia, testes com multiplos motoristas e triagem de
  listas restritivas seguem fora de escopo (ja definidos como trabalhos
  futuros no TCC).
