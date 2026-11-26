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
// conectividade e a desligamentos do veiculo. Dois ponteiros, tambem
// persistidos em flash, garantem a continuidade apos uma queda de energia:
//  - o CURSOR marca ate onde a fila ja foi confirmada como enviada, para que
//    uma transmissao interrompida retome do ponto onde parou;
//  - o CONTADOR de eventos guarda o ultimo id atribuido, para que a numeracao
//    prossiga de onde parou ao religar -- sem reiniciar em 1 e sem repetir um
//    id ja usado (um id repetido faria a nuvem descartar um evento novo como
//    se fosse duplicata).
class FilaEventos {
public:
    bool iniciar();

    // Atribui a 'evento.id' o proximo id monotonico e grava o evento na fila.
    // O id so e consumido quando o evento e de fato gravado; o contador que o
    // gera e persistido ANTES da gravacao, de modo que a numeracao continue
    // apos uma queda de energia sem jamais repetir um id (ver o .cpp).
    bool salvar(EventoTelemetria &evento);

    size_t quantidade() const;
    bool lerNaPosicao(size_t indice, EventoTelemetria &evento) const;

    size_t cursor() const { return _cursor; }
    void avancarCursor(size_t novoCursor);
    void limparSeDrenada();

    // Apaga fila, cursor e contador e zera o estado em memoria: volta ao
    // estado de PRIMEIRO USO (proximo evento sera #1). Usado pelo comando
    // serial "ZERAR" antes de uma coleta de campo limpa. Diferente de
    // limparSeDrenada(), que so remove a fila ja drenada e NUNCA zera o
    // contador de ids.
    void zerar();

    // Total de eventos ja gerados pelo dispositivo (id do ultimo evento).
    // Persistido: nao zera ao reiniciar nem ao drenar a fila.
    uint32_t totalEventosGerados() const { return _contadorEventos; }

private:
    void carregarCursor();
    void salvarCursor();
    void carregarContador();
    void salvarContador();

    size_t _cursor = 0;
    uint32_t _contadorEventos = 0;  // espelhado em CAMINHO_CONTADOR_EVENTOS
};
