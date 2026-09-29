#pragma once

#include <Arduino.h>
#include "fila_eventos.h"

// Sincronizacao oportunista (Secao B.4): so tenta conectar ao Wi-Fi
// confiavel pre-cadastrado (garagem/base) e, se conseguir, drena a fila via
// MQTT QoS 1 com sessao persistente. Fora do alcance, nao faz nada -- o
// dispositivo continua detectando e enfileirando eventos normalmente.
class Sincronizador {
public:
    void iniciar();
    void tentarDrenarFila(FilaEventos &fila);

private:
    bool conectarWifiConfiavel(unsigned long tempoLimite_ms);
    bool conectarMqtt();
    String montarPayload(const EventoTelemetria &evento) const;
    String idClienteDispositivo() const;
};
