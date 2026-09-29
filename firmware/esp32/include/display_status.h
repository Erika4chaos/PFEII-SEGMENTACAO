#pragma once

#include <Arduino.h>

// Painel de bancada e demonstracao em OLED SSD1306 128x64 (I2C, mesmo
// barramento do MPU-6050).
//
// Mostra apenas INDICADORES RESUMIDOS -- taxa de eventos, contagem,
// intensidade do ultimo evento, profundidade da fila, saude do sinal de GPS e
// velocidade. Nunca latitude/longitude continuas, em coerencia com a decisao
// de escopo da Secao B.3/C: o dispositivo trata posicao como dado pontual de
// evento, nao como fluxo de geolocalizacao.
//
// A taxa de eventos por minuto e a variavel central da analise (ver
// src/validacao_hardware.py, que a confronta por ANOVA com o rotulo
// comportamental do trajeto), por isso ocupa a primeira linha do corpo.
class DisplayStatus {
public:
    bool iniciar();
    bool ativo() const { return _ativo; }

    // Tela intermediaria, exibida durante a calibracao de gravidade.
    void mostrarCalibrando();

    // Chamar a cada iteracao do loop: a propria classe limita a taxa de
    // redesenho, para nao competir com a amostragem de 50 Hz do acelerometro.
    void atualizar(uint32_t totalEventos,
                   float ultimaMagnitude_ms2,
                   size_t pendentesNaFila,
                   bool gpsValido,
                   uint32_t satelites,
                   float velocidade_kmh,
                   unsigned long tempoAtivo_ms);

private:
    void desenhar(uint32_t totalEventos, float ultimaMagnitude_ms2,
                  size_t pendentesNaFila, bool gpsValido, uint32_t satelites,
                  float velocidade_kmh, float taxaPorMinuto);

    bool _ativo = false;
    unsigned long _proximoDesenho_ms = 0;
};
