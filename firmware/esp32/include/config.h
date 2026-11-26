#pragma once

// Parametros do no de telemetria (Secao B). Este arquivo e a fonte UNICA do
// limiar de magnitude e do periodo refratario: src/validacao_hardware.py le
// os dois daqui, para que firmware e validacao contra o UAH-DriveSet nao
// voltem a divergir.

// --- Fisica ---
#define G_MS2 9.80665f  // 1 G em m/s^2

// --- Pinagem I2C (MPU-6050 / GY-521) ---
#define PINO_SDA 21
#define PINO_SCL 22

// --- Pinagem serial (GPS NEO-6M na UART2 do ESP32) ---
#define PINO_GPS_RX 16  // ESP32 RX2 <- TX do modulo GPS
#define PINO_GPS_TX 17  // ESP32 TX2 -> RX do modulo GPS
#define GPS_BAUD 9600

// --- LEDs de status e botao de panico (Secao B.1) ---
// Os 3 LEDs sao instrumentacao: tornam o estado do dispositivo headless visivel
// na bancada e na demonstracao, sem depender do monitor serial. O LED CALIB tem
// peso operacional (indica "pode andar" apos a calibracao). O botao de panico e
// um acionamento manual do motorista (contexto de seguranca/roubo de carga).
// Pinos realocados em 31/10/2026: os GPIO 25/26/27 ficaram inacessiveis na
// montagem da protoboard (lado do ESP32 ocupado). Movidos para GPIO livres de
// saida, sem funcao de strapping no boot (evitados D2/D15).
#define PINO_LED_FIX       19   // aceso quando o GPS tem fix valido   (era 26)
#define PINO_LED_EVENTO     4   // pisca a cada evento registrado      (era 25)
#define PINO_LED_CALIB     18   // aceso apos uma calibracao confiavel (era 27)
#define PINO_BOTAO_PANICO  32   // botao SOS -> GND (INPUT_PULLUP; 100 nF de debounce no hardware)
#define LED_EVENTO_PULSO_MS 500 // duracao do pisca do LED de evento
#define BOTAO_DEBOUNCE_MS   50  // janela de debounce do botao (complementa o capacitor)

// --- Amostragem e filtragem (Secao B.2) ---
// 50 Hz e bem acima do dobro da frequencia relevante de transientes de
// frenagem/aceleracao brusca (poucos Hz), atendendo Nyquist com folga.
#define TAXA_AMOSTRAGEM_HZ 50
#define INTERVALO_AMOSTRA_MS (1000 / TAXA_AMOSTRAGEM_HZ)
#define JANELA_MEDIA_MOVEL 5      // amostras no filtro passa-baixa de media movel
#define AMOSTRAS_CALIBRACAO 200   // ~4 s parado, na instalacao, para estimar g

// --- Deteccao de eventos (Secao B.3) ---
//
// ATENCAO AO HISTORICO. Ate 23/09/2026 o limiar era 6,0 m/s^2, escolhido para
// coincidir com o limiar padrao de src/validacao_hardware.py e com a
// literatura citada no TCC. O valor nunca foi alcancavel: a magnitude era
// calculada como |(ax,ay,az)| - g (diferenca de NORMAS), que comprime a
// aceleracao horizontal para perto de zero -- uma frenagem de 4,0 m/s^2
// aparecia como 0,78, e ler 6,0 exigiria 12,4 m/s^2 reais, ou 1,26 g.
// Coerente com isso, apenas 2 dos 311.385 registros do UAH-DriveSet cruzam
// 6,0, e nenhum deles em trajeto agressivo.
//
// A correcao foi na GRANDEZA, nao no limiar: sensores.cpp passou a subtrair o
// vetor de gravidade eixo a eixo, e a magnitude voltou a ser aceleracao
// dinamica em unidade real (ver o comentario extenso em sensores.h).
//
// Com a grandeza corrigida, o limiar adotado e 2,9 m/s^2 -- cerca de 0,30 g,
// faixa usual de "frenagem brusca" na literatura. O valor nao foi escolhido a
// mao: e o percentil 99,9 da magnitude em trajetos NORMAIS do UAH-DriveSet,
// derivado por derivar_limiar_recalibrado() em src/validacao_hardware.py. A
// regra "acima disto e atipico para conducao normal" e deliberadamente
// simples de defender, em vez de um valor ajustado para maximizar metrica,
// que seria sujeito a sobreajuste com n = 40 trajetos.
//
// ALINHADO em 29/09/2026: src/validacao_hardware.py passou a LER este valor
// (e INTERVALO_MIN_ENTRE_EVENTOS_MS) deste arquivo, em vez de manter uma copia
// propria. Firmware, validacao e dashboard usam o mesmo limiar; para muda-lo,
// edite so aqui e rode `python src/validacao_hardware.py` de novo.
#define LIMIAR_MAGNITUDE_MS2 2.9f
#define LIMIAR_JERK_MS3 30.0f               // limiar auxiliar sobre a derivada da magnitude (31/10: 20->30 p/ reduzir disparo em manuseio de bancada)

// Piso de magnitude exigido JUNTO do criterio de jerk.
//
// Por que existe. O jerk sozinho dispara com qualquer transiente rapido, ainda
// que sem aceleracao real por tras: um buraco na pista, o modulo batendo no
// suporte, ou um chacoalhao de mao na protoboard. Nenhuma dessas coisas e
// conducao agressiva, mas todas entrariam na fila como evento e contaminariam
// a taxa de eventos por minuto, que e a variavel da analise comportamental.
//
// VALOR PROVISORIO. O piso de 3,0 anterior vinha de uma medicao de bancada
// (21/09/2026, 9 trancos manuais: jerk de 20,9 a 81,3 m/s^3 com magnitude
// filtrada nunca acima de 1,90 m/s^2). Aquela medicao foi feita na grandeza
// ANTIGA e nao transfere para a nova -- na subtracao vetorial um tranco de mao
// le bem mais alto. O valor abaixo mantem a regra "metade do limiar
// principal", mas precisa ser reconfirmado na bancada antes da coleta que for
// para o texto: basta repetir os trancos manuais e ler a magnitude maxima no
// serial (ver INTERVALO_RESUMO_DIAGNOSTICO_MS).
// RECONFIRMADO em 31/10/2026 na bancada: com a magnitude ja corrigida, o
// manuseio da placa gerava eventos pelo jerk com magnitude de apenas ~1,6-1,8
// m/s^2. Subir o piso de 1,45 para 2,0 corta esses falsos positivos sem mexer
// no limiar principal (2,9) -- este piso e o jerk NAO sao lidos por
// src/validacao_hardware.py, entao a validacao UAH continua inalterada.
#define PISO_MAGNITUDE_JERK_MS2 2.0f
#define INTERVALO_MIN_ENTRE_EVENTOS_MS 2000 // periodo refratario: evita contar a mesma manobra varias vezes

// --- Persistencia do evento de magnitude (Secao B.3; correcao 11/11/2026) ---
// Antes, UMA unica amostra acima do limiar ja virava evento. A 50 Hz isso faz
// um impacto de buraco/junta ou um tranco da protoboard (pico de 1-2 amostras,
// 20-40 ms, lido em 3,1-3,5 m/s2 no teste de rua) contar como conducao brusca,
// inflando a taxa de eventos/min -- a variavel da analise comportamental. Uma
// frenagem/aceleracao real se sustenta por centenas de ms, entao exigir que o
// limiar seja cruzado por N amostras CONSECUTIVAS separa manobra de impacto.
// Nao altera o limiar (2,9) nem e lido por src/validacao_hardware.py: a
// validacao contra o UAH-DriveSet continua inalterada. O criterio de jerk
// permanece sem persistencia (ele existe para pegar o pico curto que a media
// movel suavizou). Reconfirmar N com o resumo [diag] do serial do trajeto.
#define AMOSTRAS_MIN_EVENTO 5   // ~100 ms (5 x 20 ms) consecutivos acima do limiar

// --- Sanidade da calibracao (Secao B.2) ---
// Parado, a norma do vetor de aceleracao medido vale ~1 g. Fora desta faixa, a
// calibracao pegou o modulo desconectado (norma ~0) ou a placa em movimento, e
// o offset sai errado -- caso em que toda leitura seguinte vira evento. Em
// bancada isso aparece no serial; em campo passaria silencioso.
#define CALIB_NORMA_MIN_G 0.80f
#define CALIB_NORMA_MAX_G 1.20f

// --- Resumo periodico de diagnostico ---
// Imprime no serial, a cada intervalo, a magnitude maxima observada na janela.
// Serve para caracterizar a bancada (qual magnitude um tranco de mao produz de
// fato) e para levantar a distribuicao real do veiculo no trajeto de campo,
// sem depender de o limiar ter sido cruzado. Definir como 0 desativa.
#define INTERVALO_RESUMO_DIAGNOSTICO_MS 5000

// --- Fila local / store-and-forward (Secao B.4) ---
#define CAMINHO_FILA_EVENTOS "/fila_eventos.bin"
#define CAMINHO_CURSOR_FILA "/fila_cursor.bin"
#define CAMINHO_CONTADOR_EVENTOS "/fila_contador.bin"  // ultimo id atribuido, persistido: a contagem continua de onde parou ao religar
#define MAX_EVENTOS_FILA 512  // poucos KB por viagem (24 bytes/evento) -- cabe folgado na flash interna

// --- Rede confiavel (ex.: garagem/base de operacoes da transportadora) ---
// Substituir pelos dados reais do ponto de acesso confiavel antes de gravar
// no dispositivo; nao commitar credenciais reais neste arquivo.
#define WIFI_CONFIAVEL_SSID "GARAGEM_TRANSPORTADORA"
#define WIFI_CONFIAVEL_SENHA "troque-esta-senha"
#define WIFI_TENTATIVA_INTERVALO_MS 30000
#define WIFI_TIMEOUT_CONEXAO_MS 8000

// --- MQTT (QoS 1, sessao persistente -- Secao B.4) ---
#define MQTT_BROKER_HOST "broker.transportadora.local"
#define MQTT_BROKER_PORTA 1883
#define MQTT_TOPICO_EVENTOS "rct/telemetria/eventos"
#define MQTT_QOS 1
#define MQTT_KEEPALIVE_S 15
#define MQTT_TIMEOUT_MS 5000
