#include "sensores.h"

#include <MPU6050_light.h>
#include <Wire.h>

#include "config.h"

static MPU6050 mpu(Wire);

bool SensorAceleracao::iniciar() {
    Wire.begin(PINO_SDA, PINO_SCL);
    return mpu.begin() == 0;
}

void SensorAceleracao::lerEixosBrutos_g(float &ax, float &ay, float &az) {
    mpu.update();
    ax = mpu.getAccX();
    ay = mpu.getAccY();
    az = mpu.getAccZ();
}

void SensorAceleracao::aplicarMediaMovel(float ax, float ay, float az,
                                         float &fx, float &fy, float &fz) {
    _bufferX_g[_indiceBuffer] = ax;
    _bufferY_g[_indiceBuffer] = ay;
    _bufferZ_g[_indiceBuffer] = az;
    _indiceBuffer = (_indiceBuffer + 1) % JANELA_MEDIA_MOVEL;
    if (_amostrasNoBuffer < JANELA_MEDIA_MOVEL) {
        _amostrasNoBuffer++;
    }

    float somaX = 0.0f, somaY = 0.0f, somaZ = 0.0f;
    for (int i = 0; i < _amostrasNoBuffer; i++) {
        somaX += _bufferX_g[i];
        somaY += _bufferY_g[i];
        somaZ += _bufferZ_g[i];
    }
    fx = somaX / _amostrasNoBuffer;
    fy = somaY / _amostrasNoBuffer;
    fz = somaZ / _amostrasNoBuffer;
}

void SensorAceleracao::calibrar(void (*aoAmostrar)(int)) {
    // Calibracao unica na instalacao (Secao B.2): estima o VETOR de gravidade
    // -- a orientacao em que o dispositivo ficou montado -- pela media de cada
    // eixo com o veiculo parado. Substitui a fusao com giroscopio, que este
    // firmware nao implementa por escolha de projeto.
    float somaX = 0.0f, somaY = 0.0f, somaZ = 0.0f;
    for (int i = 0; i < AMOSTRAS_CALIBRACAO; i++) {
        float ax, ay, az, fx, fy, fz;
        lerEixosBrutos_g(ax, ay, az);
        aplicarMediaMovel(ax, ay, az, fx, fy, fz);
        somaX += fx;
        somaY += fy;
        somaZ += fz;
        if (aoAmostrar) aoAmostrar(i);  // feedback visual (ex.: piscar LED) durante a calibracao
        delay(INTERVALO_AMOSTRA_MS);
    }
    _offsetX_g = somaX / AMOSTRAS_CALIBRACAO;
    _offsetY_g = somaY / AMOSTRAS_CALIBRACAO;
    _offsetZ_g = somaZ / AMOSTRAS_CALIBRACAO;
    _normaGravidade_g = sqrtf(_offsetX_g * _offsetX_g +
                              _offsetY_g * _offsetY_g +
                              _offsetZ_g * _offsetZ_g);

    // Checagem de sanidade. Parado, a norma do vetor medido tem de valer
    // aproximadamente 1 g. Fora da faixa significa modulo desconectado
    // (norma ~0) ou placa em movimento durante a calibracao -- nos dois casos
    // o offset sai errado e TODA leitura seguinte vira evento. Sem este
    // aviso, a falha e silenciosa fora da bancada e contamina a viagem
    // inteira.
    _calibracaoConfiavel = (_normaGravidade_g > CALIB_NORMA_MIN_G &&
                            _normaGravidade_g < CALIB_NORMA_MAX_G);
    if (!_calibracaoConfiavel) {
        Serial.printf(
            "ALERTA: calibracao suspeita -- norma do vetor de gravidade = "
            "%.3f g (esperado entre %.1f e %.1f). Verifique se o modulo esta "
            "conectado e a placa imovel, e reinicie antes de coletar.\n",
            _normaGravidade_g, CALIB_NORMA_MIN_G, CALIB_NORMA_MAX_G);
    }

    // Reinicia o filtro e o historico de jerk para o inicio da operacao normal.
    _amostrasNoBuffer = 0;
    _indiceBuffer = 0;
    _primeiraAmostra = true;
    _proximaAmostra_ms = millis();
}

bool SensorAceleracao::atualizar() {
    unsigned long agora = millis();
    if (agora < _proximaAmostra_ms) {
        return false;
    }
    _proximaAmostra_ms = agora + INTERVALO_AMOSTRA_MS;

    float ax, ay, az, fx, fy, fz;
    lerEixosBrutos_g(ax, ay, az);
    aplicarMediaMovel(ax, ay, az, fx, fy, fz);

    // Subtracao VETORIAL da gravidade: o que sobra e a aceleracao dinamica na
    // unidade real, e nao uma diferenca de normas comprimida (ver sensores.h).
    float dx = fx - _offsetX_g;
    float dy = fy - _offsetY_g;
    float dz = fz - _offsetZ_g;
    float magnitude_dinamica_ms2 = sqrtf(dx * dx + dy * dy + dz * dz) * G_MS2;

    float jerk_ms3 = 0.0f;
    if (!_primeiraAmostra) {
        float dt_s = (agora - _tsAmostraAnterior_ms) / 1000.0f;
        if (dt_s > 0.0f) {
            jerk_ms3 = (magnitude_dinamica_ms2 - _magnitudeAnterior_ms2) / dt_s;
        }
    }

    _ultima.magnitude_ms2 = magnitude_dinamica_ms2;
    _ultima.jerk_ms3 = jerk_ms3;
    _ultima.timestamp_ms = agora;

    _magnitudeAnterior_ms2 = magnitude_dinamica_ms2;
    _tsAmostraAnterior_ms = agora;
    _primeiraAmostra = false;

    return true;
}
