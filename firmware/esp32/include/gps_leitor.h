#pragma once

#include <Arduino.h>
#include <HardwareSerial.h>
#include <TinyGPSPlus.h>

// Envelope fino sobre o TinyGPS++ (NMEA) na UART2 do ESP32. Fornece apenas
// o que o registro de evento precisa: posicao e um carimbo de tempo --
// nunca o fluxo continuo de geolocalizacao (Secao B.3/C).
class GpsLeitor {
public:
    void iniciar();
    void atualizar();  // chamar a cada iteracao do loop para drenar a serial

    bool localizacaoValida() const;
    float latitude() const;
    float longitude() const;
    uint32_t satelites() const;        // qualidade do sinal, nao posicao
    float velocidadeKmh() const;       // usada na analise de comportamento
    uint32_t epocaAtual() const;  // segundos desde epoch (UTC) se houver fix; senao, tempo relativo

    // Diagnostico de recepcao (nao e posicao): distingue "sem dado chegando"
    // (fiacao RX/TX ou alimentacao) de "dado chega mas sem satelites" (antena
    // ou ceu obstruido). Inline: so le contadores const do TinyGPS++.
    uint32_t caracteresRecebidos() const { return _gps.charsProcessed(); }
    uint32_t sentencasValidas() const { return _gps.passedChecksum(); }

private:
    HardwareSerial _serial{2};
    // getters do TinyGPS++ (lat/lng/date/time) nao sao const -- eles limpam a
    // flag de "updated" --, entao o objeto precisa ser mutable para que os
    // metodos de leitura acima possam continuar const.
    mutable TinyGPSPlus _gps;
};
