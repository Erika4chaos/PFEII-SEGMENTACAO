// No de telemetria embarcada RCT (Secao B): ESP32 + GPS NEO-6M + MPU-6050.
// Amostra o acelerometro, condiciona o sinal (media movel + calibracao de
// gravidade), detecta eventos de conducao brusca por limiar de magnitude e
// jerk, enfileira o resumo do evento em LittleFS e sincroniza de forma
// oportunista via MQTT (QoS 1) sempre que a rede confiavel esta ao alcance.
//
// Fora de escopo por decisao de projeto (ja justificada no TCC): fusao com
// giroscopio, conectividade celular, cartao SD externo e transmissao do
// fluxo continuo de sensores.

#include <Arduino.h>

#include "config.h"
#include "display_status.h"
#include "fila_eventos.h"
#include "gps_leitor.h"
#include "sensores.h"
#include "sincronizacao.h"

static SensorAceleracao sensor;
static GpsLeitor gps;
static FilaEventos fila;
static Sincronizador sincronizador;
static DisplayStatus painel;

static uint32_t contadorEventos = 0;
static unsigned long ultimoEventoRegistrado_ms = 0;
static unsigned long ultimaTentativaSync_ms = 0;
static unsigned long instanteAtivacao_ms = 0;  // marco zero da taxa de eventos
static float ultimaMagnitude_ms2 = 0.0f;

void setup() {
    Serial.begin(115200);
    delay(200);
    Serial.println(F("No de telemetria RCT -- inicializando"));

    if (!fila.iniciar()) {
        Serial.println(F("Aviso: fila de eventos indisponivel (LittleFS)."));
    }

    gps.iniciar();

    if (!sensor.iniciar()) {
        Serial.println(F("Falha ao iniciar o MPU-6050. Verifique a fiacao I2C (SDA/SCL)."));
    }

    painel.iniciar();
    painel.mostrarCalibrando();

    Serial.println(F("Calibrando o offset de gravidade -- mantenha o veiculo parado..."));
    sensor.calibrar();
    if (sensor.calibracaoConfiavel()) {
        Serial.printf("Calibracao concluida (gravidade = %.3f g). "
                      "Deteccao de eventos ativa.\n",
                      sensor.normaGravidadeCalibrada_g());
    } else {
        // A mensagem detalhada ja saiu de dentro de calibrar(). Aqui o aviso
        // fica repetido de proposito, para nao passar despercebido no meio do
        // log de inicializacao.
        Serial.println(F("Deteccao ativa, mas a calibracao NAO e confiavel -- "
                         "reinicie com a placa parada antes de coletar."));
    }

    sincronizador.iniciar();
    instanteAtivacao_ms = millis();
}

void loop() {
    gps.atualizar();

    if (sensor.atualizar()) {
        AmostraFiltrada amostra = sensor.ultimaAmostra();

        bool ultrapassouMagnitude = amostra.magnitude_ms2 > LIMIAR_MAGNITUDE_MS2;

        // O jerk e criterio AUXILIAR, nao alternativo: ele existe para pegar a
        // manobra cujo pico a media movel suavizou abaixo do limiar principal,
        // e nao para disparar sozinho em qualquer transiente. Por isso exige,
        // junto, um piso de aceleracao real -- ver PISO_MAGNITUDE_JERK_MS2 no
        // config.h, com a medicao de bancada que justifica o valor.
        bool ultrapassouJerk = fabsf(amostra.jerk_ms3) > LIMIAR_JERK_MS3
                               && amostra.magnitude_ms2 > PISO_MAGNITUDE_JERK_MS2;
        bool foraDoPeriodoRefratario =
            (amostra.timestamp_ms - ultimoEventoRegistrado_ms) > INTERVALO_MIN_ENTRE_EVENTOS_MS;

        if ((ultrapassouMagnitude || ultrapassouJerk) && foraDoPeriodoRefratario) {
            EventoTelemetria evento;
            evento.id = ++contadorEventos;
            evento.timestamp_epoch = gps.epocaAtual();
            evento.magnitude_ms2 = amostra.magnitude_ms2;
            evento.jerk_ms3 = amostra.jerk_ms3;
            evento.latitude = gps.localizacaoValida() ? gps.latitude() : 0.0f;
            evento.longitude = gps.localizacaoValida() ? gps.longitude() : 0.0f;

            if (fila.salvar(evento)) {
                Serial.printf(
                    "Evento #%u: magnitude=%.2f m/s2 jerk=%.2f m/s3 (por %s)\n",
                    evento.id, evento.magnitude_ms2, evento.jerk_ms3,
                    ultrapassouMagnitude ? "magnitude" : "jerk");
            } else {
                Serial.println(F("Fila cheia: evento descartado (aumente MAX_EVENTOS_FILA ou sincronize)."));
            }
            ultimaMagnitude_ms2 = evento.magnitude_ms2;
            ultimoEventoRegistrado_ms = amostra.timestamp_ms;
        }
    }

    // Resumo periodico: magnitude maxima da janela, independentemente de o
    // limiar ter sido cruzado. E o que permite caracterizar a bancada e
    // levantar a distribuicao real do veiculo sem depender de evento algum.
#if INTERVALO_RESUMO_DIAGNOSTICO_MS > 0
    static float picoJanela_ms2 = 0.0f;
    static float picoJerkJanela_ms3 = 0.0f;
    static unsigned long proximoResumo_ms = INTERVALO_RESUMO_DIAGNOSTICO_MS;
    {
        AmostraFiltrada ultima = sensor.ultimaAmostra();
        if (ultima.magnitude_ms2 > picoJanela_ms2) picoJanela_ms2 = ultima.magnitude_ms2;
        if (fabsf(ultima.jerk_ms3) > picoJerkJanela_ms3) picoJerkJanela_ms3 = fabsf(ultima.jerk_ms3);
    }
    if (millis() >= proximoResumo_ms) {
        proximoResumo_ms = millis() + INTERVALO_RESUMO_DIAGNOSTICO_MS;
        Serial.printf("[diag] pico magnitude=%.2f m/s2  pico |jerk|=%.1f m/s3  "
                      "eventos=%u  limiar=%.1f\n",
                      picoJanela_ms2, picoJerkJanela_ms3, contadorEventos,
                      (float)LIMIAR_MAGNITUDE_MS2);
        picoJanela_ms2 = 0.0f;
        picoJerkJanela_ms3 = 0.0f;
    }
#endif

    // O painel so e consultado a 2 Hz: fila.quantidade() abre um arquivo no
    // LittleFS a cada chamada, e no ritmo do loop isso seria I/O de flash sem
    // necessidade.
    static unsigned long proximoPainel_ms = 0;
    if (painel.ativo() && millis() >= proximoPainel_ms) {
        proximoPainel_ms = millis() + 500;
        size_t total = fila.quantidade();
        size_t cursor = fila.cursor();
        painel.atualizar(contadorEventos, ultimaMagnitude_ms2,
                         (total > cursor) ? (total - cursor) : 0,
                         gps.localizacaoValida(), gps.satelites(),
                         gps.velocidadeKmh(), millis() - instanteAtivacao_ms);
    }

    if (millis() - ultimaTentativaSync_ms > WIFI_TENTATIVA_INTERVALO_MS) {
        ultimaTentativaSync_ms = millis();
        if (fila.quantidade() > fila.cursor()) {
            sincronizador.tentarDrenarFila(fila);
        }
    }
}
