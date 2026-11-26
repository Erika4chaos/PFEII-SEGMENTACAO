#include "fila_eventos.h"

#include <LittleFS.h>

#include "config.h"

bool FilaEventos::iniciar() {
    if (!LittleFS.begin(true)) {
        Serial.println(F("Falha ao montar o LittleFS."));
        return false;
    }
    carregarCursor();
    carregarContador();
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

void FilaEventos::carregarContador() {
    _contadorEventos = 0;
    File f = LittleFS.open(CAMINHO_CONTADOR_EVENTOS, "r");
    if (f) {
        f.read(reinterpret_cast<uint8_t *>(&_contadorEventos), sizeof(_contadorEventos));
        f.close();
    }
}

void FilaEventos::salvarContador() {
    File f = LittleFS.open(CAMINHO_CONTADOR_EVENTOS, "w");
    if (f) {
        f.write(reinterpret_cast<const uint8_t *>(&_contadorEventos), sizeof(_contadorEventos));
        f.close();
    }
}

bool FilaEventos::salvar(EventoTelemetria &evento) {
    if (quantidade() >= MAX_EVENTOS_FILA) {
        return false;  // fila cheia: o chamador decide como reportar/descartar
    }

    // Reserva e PERSISTE o proximo id ANTES de gravar o evento. A ordem e
    // deliberada: se faltar energia entre as duas escritas, perde-se no maximo
    // um NUMERO de id (uma lacuna na sequencia, inofensiva para a deduplicacao
    // na nuvem), e nunca um id repetido -- que faria a nuvem tratar um evento
    // novo como duplicata e descarta-lo. E o que mantem a contagem continua
    // mesmo que o dispositivo seja desligado e religado (store-and-forward).
    uint32_t idReservado = _contadorEventos + 1;
    _contadorEventos = idReservado;
    salvarContador();
    evento.id = idReservado;

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

void FilaEventos::zerar() {
    // Remove os tres arquivos de estado (fila, cursor e contador). Ausentes,
    // iniciar() os recarrega como 0 no proximo boot -- e aqui ja zeramos o
    // estado em memoria para o efeito valer na mesma sessao, sem reiniciar.
    LittleFS.remove(CAMINHO_FILA_EVENTOS);
    LittleFS.remove(CAMINHO_CURSOR_FILA);
    LittleFS.remove(CAMINHO_CONTADOR_EVENTOS);
    _cursor = 0;
    _contadorEventos = 0;
}
