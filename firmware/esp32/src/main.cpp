// No de telemetria embarcada RCT (Secao B): ESP32 + GPS NEO-6M + MPU-6050.
// Amostra o acelerometro, condiciona o sinal (media movel + calibracao de
// gravidade), detecta eventos de conducao brusca por limiar de magnitude e
// jerk, enfileira o resumo do evento em LittleFS e sincroniza de forma
// oportunista via MQTT (QoS 1) sempre que a rede confiavel esta ao alcance.
// Sinaliza o estado em tres LEDs (fix de GPS, evento, calibracao) e aceita um
// acionamento manual de SOS pelo botao de panico.
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

static unsigned long ultimoEventoRegistrado_ms = 0;
static unsigned long ultimaTentativaSync_ms = 0;
static unsigned long instanteAtivacao_ms = 0;  // marco zero da taxa de eventos
static float ultimaMagnitude_ms2 = 0.0f;

// Estado dos LEDs de status e do botao de panico.
static unsigned long ledEventoAte_ms = 0;        // ate quando o LED de evento fica aceso
static bool botaoEstavel = HIGH, botaoLeituraAnt = HIGH;
static unsigned long botaoMudanca_ms = 0;

// Acende o LED de evento e agenda seu desligamento (pulso curto por evento).
static void acionarLedEvento() {
    digitalWrite(PINO_LED_EVENTO, HIGH);
    ledEventoAte_ms = millis() + LED_EVENTO_PULSO_MS;
}

// Le o botao de panico com debounce e, na borda de pressao, enfileira um evento
// SOS. O SOS e marcado por magnitude e jerk iguais a zero -- combinacao que a
// deteccao de conducao nunca produz (ela exige magnitude acima do piso) --, de
// modo que a nuvem o distingue de uma manobra sem alterar o formato de 24 bytes.
// A posicao vem do GPS, quando ha fix.
static void lerBotaoPanico() {
    bool leitura = digitalRead(PINO_BOTAO_PANICO);
    if (leitura != botaoLeituraAnt) {
        botaoMudanca_ms = millis();
        botaoLeituraAnt = leitura;
    }
    if ((millis() - botaoMudanca_ms) > BOTAO_DEBOUNCE_MS && leitura != botaoEstavel) {
        botaoEstavel = leitura;
        if (botaoEstavel == LOW) {   // LOW = pressionado (entrada em pull-up)
            EventoTelemetria sos;
            sos.timestamp_epoch = gps.epocaAtual();
            sos.magnitude_ms2 = 0.0f;   // 0/0 marca um acionamento SOS, nao uma manobra
            sos.jerk_ms3 = 0.0f;
            sos.latitude  = gps.localizacaoValida() ? gps.latitude()  : 0.0f;
            sos.longitude = gps.localizacaoValida() ? gps.longitude() : 0.0f;
            if (fila.salvar(sos)) {
                Serial.printf("Evento SOS #%u (botao de panico) lat=%.6f lon=%.6f\n",
                              sos.id, sos.latitude, sos.longitude);
                ultimaMagnitude_ms2 = 0.0f;
                acionarLedEvento();
            } else {
                Serial.println(F("Fila cheia: SOS descartado (aumente MAX_EVENTOS_FILA ou sincronize)."));
            }
        }
    }
}

// Comandos de manutencao pelo monitor serial (115200). NAO interfere na
// deteccao: so le linhas do serial e age sobre a fila.
//   ZERAR -> apaga fila/cursor/contador: volta ao estado de primeiro uso.
//   DUMP  -> imprime todos os eventos da fila em CSV, para analise em casa
//            quando nao houve broker no campo (store-and-forward offline).
// A posicao/timestamp completos so existem no registro de 24 bytes da flash;
// por isso o DUMP le a fila, e nao o log de deteccao (que omite lat/lon).
static void processarComandoSerial() {
    static char buffer[16];
    static uint8_t n = 0;
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            if (n == 0) continue;  // ignora linha vazia / CRLF
            buffer[n] = '\0';
            n = 0;
            if (strcasecmp(buffer, "ZERAR") == 0) {
                fila.zerar();
                instanteAtivacao_ms = millis();     // reinicia a base da taxa/min
                ultimoEventoRegistrado_ms = 0;
                Serial.println(F("OK: fila, cursor e contador ZERADOS. "
                                 "Primeiro uso -- proximo evento sera #1."));
            } else if (strcasecmp(buffer, "DUMP") == 0) {
                size_t total = fila.quantidade();
                size_t cur = fila.cursor();
                Serial.println(F("===DUMP INICIO==="));
                Serial.println(F("id,ts_epoch,magnitude_ms2,jerk_ms3,lat,lon,enviado"));
                for (size_t i = 0; i < total; i++) {
                    EventoTelemetria e;
                    if (!fila.lerNaPosicao(i, e)) continue;
                    Serial.printf("%u,%u,%.2f,%.2f,%.6f,%.6f,%d\n",
                                  e.id, e.timestamp_epoch, e.magnitude_ms2,
                                  e.jerk_ms3, e.latitude, e.longitude,
                                  (i < cur) ? 1 : 0);
                }
                Serial.printf("===DUMP FIM=== total=%u contador=%u cursor=%u\n",
                              (unsigned)total, fila.totalEventosGerados(),
                              (unsigned)cur);
            } else {
                Serial.printf("Comando desconhecido: '%s' (use ZERAR ou DUMP)\n", buffer);
            }
        } else if (n < sizeof(buffer) - 1) {
            buffer[n++] = c;
        } else {
            n = 0;  // estouro do buffer: descarta a linha
        }
    }
}

// Pisca o LED amarelo (FIX, GPIO19) durante a calibracao, como sinal de
// "calibrando -- segure parado". Chamado por calibrar() a cada amostra; alterna
// ~a cada 100 ms (5 amostras x INTERVALO_AMOSTRA_MS). Terminada a calibracao, o
// loop devolve este LED a sua funcao normal (aceso quando ha fix de GPS).
static void piscarCalibracao(int amostra) {
    static bool estado = false;
    if (amostra % 5 == 0) {
        estado = !estado;
        digitalWrite(PINO_LED_FIX, estado ? HIGH : LOW);
    }
}

void setup() {
    Serial.begin(115200);
    delay(200);
    Serial.println(F("No de telemetria RCT -- inicializando"));

    pinMode(PINO_LED_FIX, OUTPUT);
    pinMode(PINO_LED_EVENTO, OUTPUT);
    pinMode(PINO_LED_CALIB, OUTPUT);
    pinMode(PINO_BOTAO_PANICO, INPUT_PULLUP);
    digitalWrite(PINO_LED_FIX, LOW);
    digitalWrite(PINO_LED_EVENTO, LOW);
    digitalWrite(PINO_LED_CALIB, LOW);

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
    sensor.calibrar(piscarCalibracao);
    digitalWrite(PINO_LED_FIX, LOW);  // fim do pisca; o loop reassume este LED conforme o fix de GPS
    if (sensor.calibracaoConfiavel()) {
        digitalWrite(PINO_LED_CALIB, HIGH);  // "pode andar": calibracao confiavel
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
    processarComandoSerial();  // ZERAR / DUMP pelo monitor serial
    gps.atualizar();

    // Botao de panico (acionamento manual do motorista) e LEDs de status.
    lerBotaoPanico();
    digitalWrite(PINO_LED_FIX, gps.localizacaoValida() ? HIGH : LOW);
    if (ledEventoAte_ms != 0 && millis() > ledEventoAte_ms) {
        digitalWrite(PINO_LED_EVENTO, LOW);
        ledEventoAte_ms = 0;
    }

    if (sensor.atualizar()) {
        AmostraFiltrada amostra = sensor.ultimaAmostra();

        // Persistencia (ver AMOSTRAS_MIN_EVENTO no config.h): o evento por
        // magnitude so conta se o limiar for cruzado por amostras CONSECUTIVAS,
        // o que descarta o pico isolado de buraco/tranco sem mexer no limiar.
        static uint8_t amostrasAcimaLimiar = 0;
        if (amostra.magnitude_ms2 > LIMIAR_MAGNITUDE_MS2) {
            if (amostrasAcimaLimiar < 255) amostrasAcimaLimiar++;
        } else {
            amostrasAcimaLimiar = 0;
        }
        bool magnitudeSustentada = amostrasAcimaLimiar >= AMOSTRAS_MIN_EVENTO;

        // O jerk e criterio AUXILIAR, nao alternativo: ele existe para pegar a
        // manobra cujo pico a media movel suavizou abaixo do limiar principal,
        // e nao para disparar sozinho em qualquer transiente. Por isso exige,
        // junto, um piso de aceleracao real -- ver PISO_MAGNITUDE_JERK_MS2 no
        // config.h, com a medicao de bancada que justifica o valor.
        bool ultrapassouJerk = fabsf(amostra.jerk_ms3) > LIMIAR_JERK_MS3
                               && amostra.magnitude_ms2 > PISO_MAGNITUDE_JERK_MS2;
        bool foraDoPeriodoRefratario =
            (amostra.timestamp_ms - ultimoEventoRegistrado_ms) > INTERVALO_MIN_ENTRE_EVENTOS_MS;

        if ((magnitudeSustentada || ultrapassouJerk) && foraDoPeriodoRefratario) {
            EventoTelemetria evento;
            // O id e atribuido dentro de fila.salvar(), a partir do contador
            // persistido em flash, e so e consumido se o evento for de fato
            // gravado -- por isso a numeracao continua de onde parou ao religar.
            evento.timestamp_epoch = gps.epocaAtual();
            evento.magnitude_ms2 = amostra.magnitude_ms2;
            evento.jerk_ms3 = amostra.jerk_ms3;
            evento.latitude = gps.localizacaoValida() ? gps.latitude() : 0.0f;
            evento.longitude = gps.localizacaoValida() ? gps.longitude() : 0.0f;

            if (fila.salvar(evento)) {
                Serial.printf(
                    "Evento #%u: magnitude=%.2f m/s2 jerk=%.2f m/s3 (por %s)\n",
                    evento.id, evento.magnitude_ms2, evento.jerk_ms3,
                    magnitudeSustentada ? "magnitude" : "jerk");
                acionarLedEvento();
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
                      picoJanela_ms2, picoJerkJanela_ms3, fila.totalEventosGerados(),
                      (float)LIMIAR_MAGNITUDE_MS2);
        // Status do GPS no serial (o fix so aparecia no OLED). Util para
        // acompanhar a aquisicao de fix em ceu aberto e diagnosticar antena.
        if (gps.localizacaoValida()) {
            Serial.printf("[gps] FIX  sats=%u  lat=%.6f lon=%.6f  vel=%.1f km/h  chars=%u ok=%u\n",
                          gps.satelites(), gps.latitude(), gps.longitude(),
                          gps.velocidadeKmh(), gps.caracteresRecebidos(),
                          gps.sentencasValidas());
        } else {
            Serial.printf("[gps] SEM FIX  sats=%u  chars=%u ok=%u "
                          "(chars=0 -> checar fiacao RX/TX e alimentacao do GPS)\n",
                          gps.satelites(), gps.caracteresRecebidos(),
                          gps.sentencasValidas());
        }
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
        painel.atualizar(fila.totalEventosGerados(), ultimaMagnitude_ms2,
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
