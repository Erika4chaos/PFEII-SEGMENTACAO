#include "sincronizacao.h"

#include <MQTT.h>
#include <WiFi.h>

#include "config.h"

static WiFiClient redeWifi;
static MQTTClient clienteMqtt(256);

void Sincronizador::iniciar() {
    WiFi.mode(WIFI_STA);
    WiFi.disconnect(true);
    clienteMqtt.begin(MQTT_BROKER_HOST, MQTT_BROKER_PORTA, redeWifi);
    // cleanSession=false mantem a sessao persistente entre reconexoes, para
    // que o QoS 1 (ao menos uma vez) sobreviva a interrupcoes (Secao B.4).
    clienteMqtt.setOptions(MQTT_KEEPALIVE_S, false, MQTT_TIMEOUT_MS);
}

String Sincronizador::idClienteDispositivo() const {
    uint64_t chipId = ESP.getEfuseMac();
    char buffer[17];
    snprintf(buffer, sizeof(buffer), "%016llX", chipId);
    return "rct-" + String(buffer);
}

bool Sincronizador::conectarWifiConfiavel(unsigned long tempoLimite_ms) {
    if (WiFi.status() == WL_CONNECTED) {
        return true;
    }

    WiFi.begin(WIFI_CONFIAVEL_SSID, WIFI_CONFIAVEL_SENHA);
    unsigned long inicio = millis();
    while (WiFi.status() != WL_CONNECTED && (millis() - inicio) < tempoLimite_ms) {
        delay(200);
    }
    return WiFi.status() == WL_CONNECTED;
}

bool Sincronizador::conectarMqtt() {
    if (clienteMqtt.connected()) {
        return true;
    }
    return clienteMqtt.connect(idClienteDispositivo().c_str());
}

String Sincronizador::montarPayload(const EventoTelemetria &evento) const {
    // Somente os indicadores resumidos do evento (contagem/intensidade/
    // localizacao pontual) -- nunca o fluxo continuo de geolocalizacao
    // (Secao B.3/C). O id combina o dispositivo e o contador local para
    // permitir deduplicacao no lado da nuvem.
    String idUnico = idClienteDispositivo() + "-" + String(evento.id);
    String payload = "{";
    payload += "\"id\":\"" + idUnico + "\",";
    payload += "\"ts\":" + String(evento.timestamp_epoch) + ",";
    payload += "\"magnitude_ms2\":" + String(evento.magnitude_ms2, 2) + ",";
    payload += "\"jerk_ms3\":" + String(evento.jerk_ms3, 2) + ",";
    payload += "\"lat\":" + String(evento.latitude, 6) + ",";
    payload += "\"lon\":" + String(evento.longitude, 6);
    payload += "}";
    return payload;
}

void Sincronizador::tentarDrenarFila(FilaEventos &fila) {
    if (!conectarWifiConfiavel(WIFI_TIMEOUT_CONEXAO_MS)) {
        return;  // fora de alcance da rede confiavel: tenta de novo na proxima janela
    }
    if (!conectarMqtt()) {
        WiFi.disconnect(true);
        return;
    }

    size_t total = fila.quantidade();
    size_t indice = fila.cursor();

    while (indice < total) {
        EventoTelemetria evento;
        if (!fila.lerNaPosicao(indice, evento)) {
            break;
        }

        bool publicado = clienteMqtt.publish(MQTT_TOPICO_EVENTOS, montarPayload(evento), false, MQTT_QOS);
        clienteMqtt.loop();

        if (!publicado) {
            break;  // interrompe aqui; retoma deste indice na proxima janela de conectividade
        }

        indice++;
        fila.avancarCursor(indice);
    }

    fila.limparSeDrenada();
    clienteMqtt.disconnect();
    WiFi.disconnect(true);
}
