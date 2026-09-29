#include "fila_eventos.h"

#include <LittleFS.h>

#include "config.h"

bool FilaEventos::iniciar() {
    if (!LittleFS.begin(true)) {
        Serial.println(F("Falha ao montar o LittleFS."));
        return false;
    }
    carregarCursor();
    return true;
}

void FilaEventos::carregarCursor() {
    _cursor = 0;
    File f = LittleFS.open(CAMINHO_CURSOR_FILA, "r");
    if (f) {
        f.read(reinterpret_cast<uint8_t *>(&_cursor), sizeof(_cursor));
        f.close();
    }
}

void FilaEventos::salvarCursor() {
    File f = LittleFS.open(CAMINHO_CURSOR_FILA, "w");
    if (f) {
        f.write(reinterpret_cast<const uint8_t *>(&_cursor), sizeof(_cursor));
        f.close();
    }
}

bool FilaEventos::salvar(const EventoTelemetria &evento) {
    if (quantidade() >= MAX_EVENTOS_FILA) {
        return false;  // fila cheia: o chamador decide como reportar/descartar
    }
    File f = LittleFS.open(CAMINHO_FILA_EVENTOS, "a");
    if (!f) {
        return false;
    }
    size_t escritos = f.write(reinterpret_cast<const uint8_t *>(&evento), sizeof(EventoTelemetria));
    f.close();
    return escritos == sizeof(EventoTelemetria);
}

size_t FilaEventos::quantidade() const {
    File f = LittleFS.open(CAMINHO_FILA_EVENTOS, "r");
    if (!f) {
        return 0;
    }
    size_t total = f.size() / sizeof(EventoTelemetria);
    f.close();
    return total;
}

bool FilaEventos::lerNaPosicao(size_t indice, EventoTelemetria &evento) const {
    File f = LittleFS.open(CAMINHO_FILA_EVENTOS, "r");
    if (!f) {
        return false;
    }
    bool ok = f.seek(indice * sizeof(EventoTelemetria));
    if (ok) {
        size_t lidos = f.read(reinterpret_cast<uint8_t *>(&evento), sizeof(EventoTelemetria));
        ok = lidos == sizeof(EventoTelemetria);
    }
    f.close();
    return ok;
}

void FilaEventos::avancarCursor(size_t novoCursor) {
    _cursor = novoCursor;
    salvarCursor();
}

void FilaEventos::limparSeDrenada() {
    size_t total = quantidade();
    if (total > 0 && _cursor >= total) {
        LittleFS.remove(CAMINHO_FILA_EVENTOS);
        _cursor = 0;
        salvarCursor();
    }
}
