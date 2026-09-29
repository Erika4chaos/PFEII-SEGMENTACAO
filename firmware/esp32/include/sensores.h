#pragma once

#include <Arduino.h>

#include "config.h"  // JANELA_MEDIA_MOVEL usado na declaracao dos buffers abaixo

// Amostra ja condicionada: magnitude dinamica (gravidade ja subtraida, via
// calibracao) e jerk (derivada da magnitude), prontas para o limiar de
// deteccao de eventos (Secao B.3).
struct AmostraFiltrada {
    float magnitude_ms2 = 0.0f;
    float jerk_ms3 = 0.0f;
    unsigned long timestamp_ms = 0;
};

// Le o MPU-6050, aplica media movel EIXO A EIXO, subtrai o vetor de gravidade
// estimado na calibracao e devolve a norma do que sobra (Secao B.2). Nao le
// nem funde o giroscopio -- simplificacao deliberada do escopo.
//
// POR QUE EIXO A EIXO, e nao pela norma escalar. A versao anterior calculava
// |(ax,ay,az)| e subtraia um offset escalar. Isso parece equivalente e nao e:
// com a gravidade em um eixo e a aceleracao do veiculo perpendicular a ela,
// a norma cresce pela soma dos QUADRADOS, de modo que
//
//     |a| - g = sqrt(g^2 + a_h^2) - g
//
// que para a_h pequeno em relacao a g vale aproximadamente a_h^2 / (2g). Uma
// frenagem firme de 4,0 m/s^2 aparecia como 0,78 m/s^2; uma frenagem no limite
// de aderencia do pneu, 8,0 m/s^2, aparecia como 2,85. Para a leitura chegar a
// 6,0 m/s^2 seriam necessarios 12,4 m/s^2 reais -- 1,26 g, fisicamente
// inalcancavel por um veiculo de via. O limiar nao estava alto: a grandeza
// medida e que comprimia a faixa de interesse para perto de zero.
//
// Subtraindo o vetor, |a - g_vec| devolve a aceleracao dinamica na unidade
// real: 4,0 m/s^2 de frenagem leem 4,0 m/s^2. E tambem o que torna a leitura
// comparavel a do UAH-DriveSet, que o dataset ja entrega sem gravidade.
class SensorAceleracao {
public:
    bool iniciar();
    void calibrar();
    bool atualizar();  // retorna true quando uma nova amostra filtrada fica pronta
    AmostraFiltrada ultimaAmostra() const { return _ultima; }

    // Verdadeiro quando a calibracao terminou com um vetor de gravidade de
    // norma plausivel. Falso significa que o offset foi estimado com o modulo
    // desconectado ou com a placa em movimento -- caso em que toda leitura
    // seguinte vira evento. Na bancada isso aparece no serial; em campo, sem
    // ninguem olhando, passaria silencioso.
    bool calibracaoConfiavel() const { return _calibracaoConfiavel; }
    float normaGravidadeCalibrada_g() const { return _normaGravidade_g; }

private:
    void lerEixosBrutos_g(float &ax, float &ay, float &az);
    void aplicarMediaMovel(float ax, float ay, float az,
                           float &fx, float &fy, float &fz);

    float _bufferX_g[JANELA_MEDIA_MOVEL] = {0};
    float _bufferY_g[JANELA_MEDIA_MOVEL] = {0};
    float _bufferZ_g[JANELA_MEDIA_MOVEL] = {0};
    int _indiceBuffer = 0;
    int _amostrasNoBuffer = 0;

    // Vetor de gravidade estimado na calibracao, eixo a eixo (em g).
    float _offsetX_g = 0.0f;
    float _offsetY_g = 0.0f;
    float _offsetZ_g = 1.0f;
    float _normaGravidade_g = 1.0f;
    bool _calibracaoConfiavel = false;

    float _magnitudeAnterior_ms2 = 0.0f;
    unsigned long _tsAmostraAnterior_ms = 0;
    unsigned long _proximaAmostra_ms = 0;
    bool _primeiraAmostra = true;

    AmostraFiltrada _ultima;
};
