#include "display_status.h"

#include <U8g2lib.h>
#include <Wire.h>

#include "config.h"

// 128x64 e o tamanho usual do modulo de 0,96". Para o de 0,91" (128x32),
// trocar por U8G2_SSD1306_128X32_UNIVISION_F_HW_I2C.
static U8G2_SSD1306_128X64_NONAME_F_HW_I2C tela(U8G2_R0, U8X8_PIN_NONE);

// Redesenho a 2 Hz. O buffer cheio do SSD1306 tem 1 KB; a 400 kHz cada envio
// ocupa o barramento por ~23 ms, entao redesenhar mais rapido roubaria tempo
// da amostragem do acelerometro sem ganho de leitura para quem olha a tela.
#define INTERVALO_DESENHO_MS 500

static void escreverADireita(const char *texto, int y) {
    int largura = tela.getStrWidth(texto);
    tela.drawStr(128 - largura, y, texto);
}

bool DisplayStatus::iniciar() {
    _ativo = tela.begin();
    if (!_ativo) {
        Serial.println(F("Display OLED nao respondeu. Seguindo sem painel."));
        return false;
    }
    // 400 kHz encurta o envio do buffer; MPU-6050 e SSD1306 suportam.
    tela.setBusClock(400000);
    tela.setFont(u8g2_font_6x12_tf);
    return true;
}

void DisplayStatus::mostrarCalibrando() {
    if (!_ativo) return;
    tela.clearBuffer();
    tela.setFont(u8g2_font_6x12_tf);
    tela.drawStr(0, 10, "RCT-01");
    tela.drawHLine(0, 13, 128);
    tela.drawStr(0, 34, "CALIBRANDO...");
    tela.drawStr(0, 48, "mantenha parado");
    tela.sendBuffer();
}

void DisplayStatus::atualizar(uint32_t totalEventos, float ultimaMagnitude_ms2,
                              size_t pendentesNaFila, bool gpsValido,
                              uint32_t satelites, float velocidade_kmh,
                              unsigned long tempoAtivo_ms) {
    if (!_ativo) return;

    unsigned long agora = millis();
    if (agora < _proximoDesenho_ms) return;
    _proximoDesenho_ms = agora + INTERVALO_DESENHO_MS;

    float minutos = tempoAtivo_ms / 60000.0f;
    float taxa = (minutos > 0.05f) ? (totalEventos / minutos) : 0.0f;

    desenhar(totalEventos, ultimaMagnitude_ms2, pendentesNaFila, gpsValido,
             satelites, velocidade_kmh, taxa);
}

void DisplayStatus::desenhar(uint32_t totalEventos, float ultimaMagnitude_ms2,
                             size_t pendentesNaFila, bool gpsValido,
                             uint32_t satelites, float velocidade_kmh,
                             float taxaPorMinuto) {
    char linha[24];

    tela.clearBuffer();
    tela.setFont(u8g2_font_6x12_tf);

    // Cabecalho: identificacao e saude do sinal de GPS (nunca a posicao).
    tela.drawStr(0, 10, "RCT-01");
    if (satelites > 0) {
        snprintf(linha, sizeof(linha), "%lu sat", (unsigned long)satelites);
    } else {
        snprintf(linha, sizeof(linha), "-- sat");
    }
    escreverADireita(linha, 10);
    tela.drawHLine(0, 13, 128);

    // Corpo: a taxa primeiro, por ser a variavel da analise.
    snprintf(linha, sizeof(linha), "TAXA    %.1f /min", taxaPorMinuto);
    tela.drawStr(0, 27, linha);

    snprintf(linha, sizeof(linha), "EVENTOS %lu", (unsigned long)totalEventos);
    tela.drawStr(0, 39, linha);

    if (totalEventos > 0) {
        snprintf(linha, sizeof(linha), "ULTIMO  %.1f m/s2", ultimaMagnitude_ms2);
    } else {
        snprintf(linha, sizeof(linha), "ULTIMO  --");
    }
    tela.drawStr(0, 51, linha);

    // Rodape: fila do store-and-forward e velocidade (esta sim entra na
    // analise; ver a coluna de velocidade do RAW_GPS e o marcador de
    // trecho acima de 50 km/h no UAH-DriveSet).
    tela.drawHLine(0, 54, 128);
    snprintf(linha, sizeof(linha), "FILA %u", (unsigned)pendentesNaFila);
    tela.drawStr(0, 64, linha);

    if (gpsValido) {
        snprintf(linha, sizeof(linha), "%.0f km/h", velocidade_kmh);
    } else {
        snprintf(linha, sizeof(linha), "sem fix");
    }
    escreverADireita(linha, 64);

    tela.sendBuffer();
}
