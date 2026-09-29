#pragma once

#include <Arduino.h>

// Registro resumido de um evento de conducao brusca -- e tudo o que sai do
// dispositivo. Nunca o stream continuo de sensores (Secao B.3/C).
#pragma pack(push, 1)
struct EventoTelemetria {
    uint32_t id;               // contador monotonico, unico por dispositivo
    uint32_t timestamp_epoch;  // segundos desde epoch (UTC), ou relativo sem fix de GPS
    float magnitude_ms2;
    float jerk_ms3;
    float latitude;
    float longitude;
};
#pragma pack(pop)

// Fila persistente em LittleFS (Secao B.4): sobrevive a perda de
// conectividade e a desligamentos do veiculo. Um cursor, tambem persistido,
// marca ate onde a fila ja foi confirmada como enviada, para que uma
// transmissao interrompida retome do ponto onde parou.
class FilaEventos {
public:
    bool iniciar();
    bool salvar(const EventoTelemetria &evento);

    size_t quantidade() const;
    bool lerNaPosicao(size_t indice, EventoTelemetria &evento) const;

    size_t cursor() const { return _cursor; }
    void avancarCursor(size_t novoCursor);
    void limparSeDrenada();

private:
    void carregarCursor();
    void salvarCursor();

    size_t _cursor = 0;
};
