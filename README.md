# PFE II — Segmentação e Classificação de Clientes no Seguro de Transportes

Sistema híbrido que combina **aprendizado de máquina** (segmentação de clientes a
partir de cotações de seguro) com um **nó de telemetria IoT de borda** (ESP32 +
acelerômetro + GPS) para enriquecer a análise de risco na contratação do seguro
de Responsabilidade Civil do Transportador (RCT).

Projeto de Fim de Estudo II — Engenharia da Computação, Centro Universitário
Senac. O repositório reúne o pipeline de dados, o dashboard de resultados, o
firmware do dispositivo e os scripts de validação.

---

## Visão geral

O trabalho tem duas frentes que se encontram na análise de risco:

1. **Segmentação de clientes (software).** A partir de cotações de RCT, um
   pipeline em Python extrai 19 variáveis, normaliza a matriz e aplica K-Means
   (k = 3) para separar perfis-alvo de transportadoras. Os resultados são
   validados por índices internos e por análises de estabilidade.

2. **Nó de telemetria de borda (hardware).** Um dispositivo ESP32 lê o
   acelerômetro (MPU-6050) e o GPS (NEO-6M), detecta eventos de condução brusca
   por limiar de magnitude/jerk, grava-os em fila persistente e os sincroniza
   por MQTT. O limiar de detecção é derivado e validado sobre o UAH-DriveSet,
   garantindo que firmware e validação usem exatamente o mesmo critério.

---

## Estrutura do repositório

```
src/                 Pipeline de segmentação (Python)
  gerar_base_sintetica.py   Gera a base sintética de cotações RCT (3 perfis)
  preprocessamento.py       Etapa 1 — parsing, 19 variáveis, normalização Min-Max
  clustering.py             Etapa 2 — K-Means (k=3) + validação do número de clusters
  perfilamento.py           Etapa 3 — caracterização dos clusters
  sensibilidade_sementes.py Sensibilidade da segmentação à semente da base
  estabilidade.py           Estabilidade da calibração/inicialização
  validacao_hardware.py     Valida o limiar do firmware sobre o UAH-DriveSet (ANOVA)

dashboard/           Dashboard de resultados (Streamlit)
  app.py                    Aplicação principal
  charts_validacao.py       Gráficos da validação de hardware
  tab_validacao_hardware.py Aba de validação de hardware
  histograma_didatico.py    Histograma explicativo

firmware/esp32/      Firmware do nó de telemetria (PlatformIO) — ver README próprio
  src/ include/ platformio.ini
  PROTOCOLO_BANCADA.md, CALIBRACAO_JERK.md, PROXIMOS_PASSOS.md

data/
  raw/                      Entradas (cotações sintéticas; subconjunto do UAH-DriveSet)
  processed/                Saídas do pipeline (matrizes, clusters, validações)

tests/               Testes (pytest)
```

---

## Como executar

### 1. Pipeline de segmentação

Requer Python 3.11+.

```bash
pip install -r requirements.txt

# (opcional) regerar a base sintética de cotações
python src/gerar_base_sintetica.py

# Etapa 1 — pré-processamento e normalização
python src/preprocessamento.py --in data/raw/cotacoes_sinteticas.json

# Etapa 2 — K-Means e validação do número de clusters
python src/clustering.py --in data/processed/matriz_normalizada.csv

# Etapa 3 — caracterização dos clusters
python src/perfilamento.py
```

### 2. Dashboard

```bash
streamlit run dashboard/app.py
```

### 3. Validação de hardware (limiar do firmware sobre o UAH-DriveSet)

```bash
python src/validacao_hardware.py
```

O script lê o limiar diretamente de `firmware/esp32/include/config.h`, de modo
que firmware e validação nunca divergem.

### 4. Firmware (ESP32)

```bash
cd firmware/esp32
pio run -t upload
pio device monitor
```

Detalhes de ligação, pinagem, regra de detecção de evento e calibração estão em
[`firmware/esp32/README.md`](firmware/esp32/README.md).

### 5. Testes

```bash
pip install -r requirements-dev.txt
pytest
```

---

## Dados

- **Cotações RCT:** base **sintética** gerada por `src/gerar_base_sintetica.py`,
  replicando a estrutura do objeto de cotação do produto. Não há dados reais de
  clientes no repositório.
- **UAH-DriveSet** (ROMERA; BERGASA; ARROYO, 2016): apenas o subconjunto de
  arquivos de texto (acelerômetro/GPS/eventos) usado pela validação está
  versionado em `data/raw/uah-driveset/`. O arquivo original completo (~3,5 GB,
  com vídeos) não é incluído.

---

## Observações

- Credenciais de Wi-Fi e broker MQTT **não** são versionadas; configure-as em
  `firmware/esp32/include/config.h` antes de gravar em um dispositivo real.
- As versões em `requirements.txt` são os mínimos exigidos pelo código; para
  reprodução exata, congele com `pip freeze`.
