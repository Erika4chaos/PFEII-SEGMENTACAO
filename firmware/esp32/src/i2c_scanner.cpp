// Scanner I2C -- ferramenta TEMPORARIA de diagnostico (nao faz parte do
// firmware). Compilado apenas pelo ambiente [env:scanner_i2c].
//
// Faz duas coisas:
//   1. Varre todos os enderecos no barramento configurado (PINO_SDA/PINO_SCL).
//   2. Se nao achar nada, testa dezenas de OUTROS pares de GPIO, procurando os
//      modulos em pinos diferentes dos esperados. Se os fios estiverem em
//      GPIOs errados, isso encontra e diz exatamente quais sao.

#include <Arduino.h>
#include <Wire.h>
#include <U8g2lib.h>

#include "config.h"

// GPIOs utilizaveis e expostos na DevKit. Fora da lista de proposito:
//   6-11  -> ligados a flash interna, nao mexer
//   1, 3  -> UART0, e por onde esta saindo este texto
//   34-39 -> somente entrada, nao servem para I2C
static const uint8_t CANDIDATOS[] = {
    2, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 32, 33};
static const size_t N_CANDIDATOS = sizeof(CANDIDATOS) / sizeof(CANDIDATOS[0]);

// Enderecos que interessam na varredura de pinos (varrer 112 enderecos em cada
// um dos ~340 pares demoraria demais).
// Display SSD1306 128x64 por I2C (o tamanho mais comum, 0,96"). Se a imagem
// aparecer duplicada ou espremida, o modulo e 128x32 e basta trocar o
// construtor por U8G2_SSD1306_128X32_UNIVISION_F_HW_I2C.
static U8G2_SSD1306_128X64_NONAME_F_HW_I2C tela(U8G2_R0, U8X8_PIN_NONE);
static bool telaIniciada = false;
static bool viuMpu = false;
static bool viuOled = false;

static const uint8_t ALVOS[] = {0x68, 0x69, 0x3C, 0x3D};
static const size_t N_ALVOS = sizeof(ALVOS) / sizeof(ALVOS[0]);

static void descrever(uint8_t endereco) {
    if (endereco == 0x68) Serial.print(F("  <- MPU-6050 (AD0 em nivel baixo)"));
    else if (endereco == 0x69) Serial.print(F("  <- MPU-6050 (AD0 em nivel ALTO)"));
    else if (endereco == 0x3C || endereco == 0x3D) Serial.print(F("  <- display OLED SSD1306"));
}

static bool respondeu(uint8_t endereco) {
    Wire.beginTransmission(endereco);
    return Wire.endTransmission() == 0;
}

// --- Etapa 0: as linhas respondem ao controle do ESP32? ---------------------
// Em repouso a linha deve estar ALTA (puxada pelos resistores dos modulos).
// Em dreno aberto, o ESP32 puxa para BAIXO e le de volta o nivel REAL da
// linha -- se nao descer, ou o pino esta danificado, ou algo segura a linha.
static bool testarLinha(const char *nome, uint8_t pino) {
    pinMode(pino, INPUT);
    delay(5);
    int repouso = digitalRead(pino);

    pinMode(pino, OUTPUT_OPEN_DRAIN);
    digitalWrite(pino, LOW);
    delay(5);
    int puxada = digitalRead(pino);

    digitalWrite(pino, HIGH);
    delay(5);
    int solta = digitalRead(pino);

    pinMode(pino, INPUT);

    bool ok = (repouso == HIGH) && (puxada == LOW) && (solta == HIGH);
    Serial.printf("  %s (GPIO%d): repouso=%s  ao puxar=%s  ao soltar=%s  -> %s\n",
                  nome, pino,
                  repouso ? "ALTO" : "BAIXO",
                  puxada ? "ALTO" : "BAIXO",
                  solta ? "ALTO" : "BAIXO",
                  ok ? "OK" : "PROBLEMA");
    return ok;
}

static bool testarLinhas() {
    Serial.println();
    Serial.println(F("--- Etapa 0: controle eletrico das linhas ---"));
    bool a = testarLinha("SDA", PINO_SDA);
    bool b = testarLinha("SCL", PINO_SCL);
    if (a && b) {
        Serial.println(F("  Linhas saudaveis: pull-up presente e ESP32 consegue"));
        Serial.println(F("  puxar e soltar as duas."));
    } else {
        Serial.println(F("  Linha com problema. Leitura:"));
        Serial.println(F("   repouso=BAIXO   -> sem pull-up, ou linha em curto com GND"));
        Serial.println(F("   ao puxar=ALTO   -> pino nao consegue puxar (danificado?)"));
        Serial.println(F("                      ou linha em curto com 3V3"));
        Serial.println(F("   ao soltar=BAIXO -> algo esta segurando a linha em baixo"));
    }
    return a && b;
}

// --- Etapa 1: varredura completa nos pinos configurados ---------------------
static int varrerBarramentoConfigurado() {
    Serial.println();
    Serial.println(F("--- Etapa 1: barramento configurado ---"));
    Serial.printf("SDA=GPIO%d, SCL=GPIO%d\n", PINO_SDA, PINO_SCL);

    Wire.end();
    delay(10);
    Wire.begin(PINO_SDA, PINO_SCL, 100000);
    delay(10);

    int achados = 0;
    for (uint8_t endereco = 0x08; endereco <= 0x77; endereco++) {
        if (respondeu(endereco)) {
            if (endereco == 0x68 || endereco == 0x69) viuMpu = true;
            if (endereco == 0x3C || endereco == 0x3D) viuOled = true;
            Serial.printf("  >> 0x%02X respondeu", endereco);
            descrever(endereco);
            Serial.println();
            achados++;
        }
        delay(2);
    }

    if (achados == 0) {
        Serial.println(F("  Nada respondeu nestes pinos."));
    } else {
        Serial.printf("  %d dispositivo(s) encontrado(s). Barramento OK.\n", achados);
    }
    return achados;
}

// --- Etapa 2: procurar os modulos em outros pares de GPIO -------------------
static int procurarEmOutrosPinos() {
    Serial.println();
    Serial.println(F("--- Etapa 2: procurando os modulos em outros GPIOs ---"));
    Serial.printf("Testando %d x %d pares de pinos. Aguarde...\n",
                  (int)N_CANDIDATOS, (int)(N_CANDIDATOS - 1));

    int achados = 0;
    for (size_t i = 0; i < N_CANDIDATOS; i++) {
        for (size_t j = 0; j < N_CANDIDATOS; j++) {
            if (i == j) continue;
            uint8_t sda = CANDIDATOS[i];
            uint8_t scl = CANDIDATOS[j];

            Wire.end();
            delay(3);
            if (!Wire.begin(sda, scl, 100000)) continue;
            delay(3);

            for (size_t k = 0; k < N_ALVOS; k++) {
                if (respondeu(ALVOS[k])) {
                    Serial.printf("  >> ACHOU 0x%02X com SDA=GPIO%d, SCL=GPIO%d",
                                  ALVOS[k], sda, scl);
                    descrever(ALVOS[k]);
                    Serial.println();
                    achados++;
                }
            }
        }
    }

    if (achados == 0) {
        Serial.println(F("  Nenhum modulo respondeu em NENHUM par de pinos testado."));
    }
    return achados;
}

// Escreve na tela o estado do barramento. Serve para confirmar visualmente
// que o OLED funciona antes de soldar o restante.
static void mostrarNaTela(uint32_t ciclo) {
    if (!telaIniciada) {
        telaIniciada = tela.begin();
        if (!telaIniciada) return;
        tela.setBusClock(100000);
    }

    char linha[26];
    tela.clearBuffer();

    tela.setFont(u8g2_font_6x12_tf);
    tela.drawStr(0, 10, "TESTE OLED  --  RCT");
    tela.drawHLine(0, 13, 128);

    snprintf(linha, sizeof(linha), "MPU-6050 0x68 : %s", viuMpu ? "OK" : "--");
    tela.drawStr(0, 28, linha);

    snprintf(linha, sizeof(linha), "OLED     0x3C : %s", viuOled ? "OK" : "--");
    tela.drawStr(0, 41, linha);

    tela.drawHLine(0, 47, 128);
    snprintf(linha, sizeof(linha), "ciclo %lu", (unsigned long)ciclo);
    tela.drawStr(0, 60, linha);

    tela.sendBuffer();
}

static void imprimirVeredito(int etapa1, int etapa2, bool sweepFeito) {
    Serial.println();
    Serial.println(F("---------------- VEREDITO ----------------"));
    if (etapa1 > 0) {
        Serial.println(F("Barramento OK nos pinos configurados."));
        Serial.println(F("Se o MPU nao apareceu, o problema e no modulo ou na"));
        Serial.println(F("fiacao especifica dele, nao no barramento."));
    } else if (etapa2 > 0) {
        Serial.println(F("OS MODULOS ESTAO EM PINOS DIFERENTES DOS ESPERADOS."));
        Serial.println(F("Ver na etapa 2 quais GPIOs funcionaram: ou os fios"));
        Serial.println(F("estao nos pinos errados, ou a placa esta encaixada"));
        Serial.println(F("deslocada na protoboard."));
    } else if (sweepFeito) {
        Serial.println(F("Nenhum modulo respondeu em nenhum par de pinos."));
        Serial.println(F("A alimentacao ja foi medida e esta correta, entao os"));
        Serial.println(F("suspeitos sao: mau contato dos fios de dados na"));
        Serial.println(F("protoboard, ou SDA/SCL sem chegar nos modulos."));
    } else {
        Serial.println(F("Nada nos pinos configurados."));
    }
    Serial.println(F("=========================================="));
}

void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println();
    Serial.println(F("Scanner I2C iniciado. Relatorio a cada 8 segundos."));
}

void loop() {
    static bool sweepFeito = false;

    Serial.println();
    Serial.println(F("============== SCANNER I2C =============="));

    testarLinhas();

    int etapa1 = varrerBarramentoConfigurado();

    static uint32_t ciclo = 0;
    mostrarNaTela(++ciclo);

    int etapa2 = 0;
    if (etapa1 == 0 && !sweepFeito) {
        etapa2 = procurarEmOutrosPinos();
        sweepFeito = true;
    }

    imprimirVeredito(etapa1, etapa2, sweepFeito);
    delay(8000);
}
