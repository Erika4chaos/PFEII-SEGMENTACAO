#include "gps_leitor.h"

#include <time.h>

#include "config.h"

void GpsLeitor::iniciar() {
    _serial.begin(GPS_BAUD, SERIAL_8N1, PINO_GPS_RX, PINO_GPS_TX);
}

void GpsLeitor::atualizar() {
    while (_serial.available() > 0) {
        _gps.encode(_serial.read());
    }
}

bool GpsLeitor::localizacaoValida() const {
    return _gps.location.isValid() && _gps.location.age() < 5000;
}

float GpsLeitor::latitude() const {
    return _gps.location.lat();
}

float GpsLeitor::longitude() const {
    return _gps.location.lng();
}

uint32_t GpsLeitor::satelites() const {
    return _gps.satellites.isValid() ? _gps.satellites.value() : 0;
}

float GpsLeitor::velocidadeKmh() const {
    return _gps.speed.isValid() ? _gps.speed.kmph() : 0.0f;
}

uint32_t GpsLeitor::epocaAtual() const {
    if (!_gps.date.isValid() || !_gps.time.isValid()) {
        // Sem fix de data/hora ainda (ex.: logo apos ligar): usa o tempo
        // relativo desde a ligacao. Aceitavel para a PoC -- a ordem de
        // chegada na fila ja preserva a sequencia dos eventos.
        return millis() / 1000;
    }

    struct tm tempo = {};
    tempo.tm_year = _gps.date.year() - 1900;
    tempo.tm_mon = _gps.date.month() - 1;
    tempo.tm_mday = _gps.date.day();
    tempo.tm_hour = _gps.time.hour();
    tempo.tm_min = _gps.time.minute();
    tempo.tm_sec = _gps.time.second();
    // NMEA fornece hora em UTC; mktime assume o fuso local configurado no
    // ambiente C, que no ESP32 (Arduino core) e UTC por padrao se
    // configTime()/setenv(TZ) nao forem chamados.
    return static_cast<uint32_t>(mktime(&tempo));
}
