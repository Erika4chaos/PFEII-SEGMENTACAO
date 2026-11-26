# PROPOSTA de 07/10/2026 — validação do hardware, dashboard e reprodutibilidade

> **Situação: NÃO APLICADA.** Estas alterações chegaram a ser gravadas na pasta do
> projeto em 07/10 e foram revertidas no mesmo dia, a pedido da autora, que vai
> conferir se elas condizem com o projeto antes de qualquer mudança no dashboard.
> O código, o dashboard, o firmware e os arquivos de `data/processed/` que já
> existiam estão como antes. Onde o texto abaixo diz "passou a", "agora" ou
> "corrigido", leia "passaria a" — é a descrição do que está proposto.
>
> Sobraram na pasta, sem efeito sobre o dashboard atual, os arquivos novos da
> proposta: `src/estabilidade.py`, `tests/test_projeto.py`, `pytest.ini`,
> `requirements-dev.txt` e, em `data/processed/`, os `estabilidade_*.csv`,
> `sensibilidade_escala.csv` e `validacao_hardware_uah_qualidade.csv`. O teste em
> `tests/` foi escrito para o código proposto e falha no código atual.

## 1. Achado: dois trajetos do UAH-DriveSet começam com o sensor descalibrado

Ao desenhar o sinal de um trajeto de perto, pela primeira vez, apareceu algo
que nenhuma figura agregada mostrava. Em dois dos 40 trajetos a magnitude fica
presa entre 2,6 e 4,6 m/s² por minutos seguidos e depois cai, de uma amostra
para a outra, para perto de zero:

| Trajeto | Trecho afetado | Offset mediano | Eventos antes → depois | Pico antes → depois |
|---|---|---|---|---|
| D3 · agressiva · rodovia | 4,5 min iniciais | 3,5 m/s² | 117 → 1 | 5,58 → 2,95 m/s² |
| D5 · sonolenta · rodovia | 9,0 min iniciais | 4,0 m/s² | 236 → 0 | 6,05 → 1,60 m/s² |

Não é manobra. Um veículo a 130 km/h em rodovia não sustenta 0,35 g por quatro
minutos. O próprio arquivo explica: a coluna `pitch` vale −0,25 a −0,40 rad
durante o trecho e volta a ~0 no mesmo instante em que o offset some, e o
offset do eixo lateral acompanha sen(pitch) × g. É a gravidade projetada pela
inclinação do celular antes de a pose ser calibrada.

O peso disso na base inteira era grande:

- 6.917 das 7.266 amostras acima de 2,9 m/s² (95%) estavam nesses dois trechos;
- as 2 únicas amostras acima de 6,0 m/s² também;
- 352 dos 353 eventos desses dois trajetos.

### Correção aplicada

`src/validacao_hardware.py` passou a excluir o trecho inicial sem calibração,
por critério declarado em `duracao_sem_calibracao()`:

- janelas de 30 s; a janela tem viés se a norma da mediana de (y, z) passa de
  1,0 m/s²;
- exclui-se do início do trajeto até o fim da última janela com viés da cadeia
  que começa na primeira janela, tolerando até 60 s limpos entre duas janelas
  com viés;
- uma janela com viés isolada no meio do trajeto **não** é excluída (pode ser
  uma curva longa).

O corte não depende de ajuste fino: nas 25 janelas dos dois blocos o offset
fica entre 2,6 e 4,6 m/s²; nas outras 1.024 janelas da base a mediana é 0,05 e
o percentil 99 é 0,30.

Tudo o que o pipeline lê passa por `carregar_sinal_valido()`. O que foi
excluído fica em `data/processed/validacao_hardware_uah_qualidade.csv`. Para
reproduzir os números antigos: `python src/validacao_hardware.py
--manter-trecho-sem-calibracao` (conferido: devolve os CSVs anteriores, byte a
byte).

### O que NÃO mudou

- **O limiar de 2,9 m/s².** É o p99,9 da condução normal, e nenhum trajeto
  normal foi afetado.
- **A matriz por trajeto para agressiva e normal:** 7 de 11 e 2 de 17
  sinalizados. Youden do firmware continua 0,519.
- **O melhor par da varredura** (1,5 m/s², 0,3 evento/min, Youden 0,882) e as
  seis rodadas da validação deixando um motorista de fora (Quadro 15).
- **Velocidade média e variação de curso** (vêm do GPS).

### Números do Produto 5.2 que mudam

| Onde | Antes | Agora |
|---|---|---|
| Seção 3.3 — minutos e amostras | 524 min, 311.385 amostras | 510,5 min, 303.332 amostras válidas |
| 4.5.2 — amostras acima de 6,0 m/s² | 2 (ambas em sonolenta) | nenhuma |
| 4.5.2 — magnitude dinâmica máxima | 0,62 g | 0,57 g |
| 4.5.2 — "6,0 m/s² acima do pico de 39 dos 40 trajetos" | 39 de 40 | todos os 40 |
| 4.5.2 — % de amostras acima de 2,9: agressiva em rodovia | 4,54% | 0,29% |
| 4.5.2 — % de amostras acima de 2,9: sonolenta em rodovia | 7,93% | 0,08% |
| 4.5.2 — % acima de 2,9: normal em rodovia / secundária | 0,22% / 0 | sem mudança |
| Quadro 13 — agressiva: pico mediano | 3,03 | 2,97 |
| Quadro 13 — agressiva: eventos/min (mediana / média) | 0,07 / 0,91 | 0,07 / 0,16 |
| Quadro 13 — sonolenta: pico mediano | 3,54 | 3,33 |
| Quadro 13 — sonolenta: eventos/min (mediana / média) | 0,08 / 1,16 | 0,06 / 0,08 |
| Quadro 13 — sonolenta: trajetos sem evento | 4 | 5 |
| Quadro 14 — pico: F e p | 9,41; 0,0005 | 7,30; 0,0010 |
| Quadro 14 — eventos/min: F e p | 0,87; 0,38 | 1,82; 0,10 |
| 4.5.3 — medianas do pico no texto | 1,66 / 3,03 / 3,54 | 1,66 / 2,97 / 3,33 |
| 4.5.3 — diferença por motorista (Figura 10) | 0,66 a 2,74 (D3 = 2,01) | 0,66 a 2,74 (D3 = 0,70) |
| Matriz por trajeto — sonolenta sinalizada | 8 de 12 | 7 de 12 |
| Figuras 9 e 10 | — | precisam ser regeradas (dois pontos e a barra do D3 mudam) |

Duas frases do texto deixam de valer como estão:

- *"sua distribuição concentra-se em poucos trajetos com muitos eventos"* — os
  "muitos eventos" eram os dois trechos descalibrados. O trajeto com mais
  eventos agora tem 12. A conclusão (a taxa por trajeto é esparsa demais para
  precificação) continua de pé: a taxa é zero em 24 dos 40 trajetos e o teste
  de permutação dá p = 0,10.
- Seção 3.3, *"os canais de aceleração já são entregues sem a componente
  gravitacional"* — vale para o sinal calibrado. Cabe uma frase declarando a
  exceção e a exclusão; é um ponto a favor do trabalho, não contra.

Com a correção, a média de eventos/min passa a ter a ordem esperada
(agressiva 0,16 > sonolenta 0,08 > normal 0,04), o que antes era invertido
pelo artefato.

## 2. Dashboard

**Novo: o detector em um trajeto.** Seletor de trajeto e limiar ajustável. A
linha é a magnitude; cada triângulo, um evento; o trecho excluído aparece em
cinza. Ao lado, a contagem do trajeto e a matriz rótulo × sinalizado da base
inteira naquele limiar. É um fragmento: mover o controle redesenha só o bloco.
O controle é exploratório e não altera nada fora dele.

**Novo: nota de qualidade do dado.** Quantos trajetos e amostras foram lidos,
a taxa efetiva, e quais trechos ficaram de fora, com o motivo.

**Correções:**

- A ressalva final dizia "311385 trajetos, 6 motoristas": a variável do número
  de trajetos era sobrescrita pelo total de amostras.
- Figura 1: os rótulos das duas linhas de referência eram escritos um sobre o
  outro.
- As figuras apareciam na ordem 1, 2, 3, 5, 6, 4. Agora 1 a 6, na ordem da
  tela, com o glossário ajustado.
- Cartão "E se otimizar só o Youden?": dizia "ganha 0 ponto(s) de sensibilidade
  e perde especificidade" quando o par era o próprio recomendado.
- Legenda da figura antes/depois: "com 2 → 2 trajeto(s) normal(is)".
- Legenda da curva de sensibilidade: atribuía à faixa inteira da literatura a
  contagem do piso dela. Agora mostra os dois extremos.
- Legenda do histograma: "a cauda agressiva se estende muito mais" era
  sustentada pelos trechos descalibrados. Agora cita a fração medida por rótulo.
- "40 trajetos", "6 motoristas" e "6 m/s²" escritos à mão passaram a vir dos
  dados; textos que supunham contagem maior que zero foram ajustados.
- A figura do pico por trajeto sumia, com o texto "0 trajetos", sem a pasta
  bruta. Agora cai para `uah_trips.csv`.
- Aba 3: "Seção 3.4.1" virou "coligações com o hardware da Seção 3.4", como no
  Produto 5.2 revisado.

## 3. Software

- **`src/estabilidade.py` (novo).** O texto reportava quatro verificações que
  não tinham programa: ARI entre inicializações, calibração do gerador em dez
  sementes, peso de cada bloco de variáveis e o Quadro 11. O script as
  reproduz — os números batem com os do Produto 5.2, sem exceção — e grava
  cinco CSVs. Fecha a pendência das "dez sementes".
- **Regra de evento em um só lugar.** `marcar_inicios_de_evento()` passou a
  ser a única implementação; a contagem do pipeline e os marcadores do visor
  usam a mesma função.
- **`gerar_base_sintetica.marginais_geradas()`** separada do relatório, para a
  checagem entre sementes medir as mesmas grandezas. A base gerada é idêntica.
- **Testes (`tests/`, 17 casos)** e `requirements-dev.txt`.
- **`README.md`** com estrutura, ordem de execução e a tabela de que programa
  sustenta cada número do texto.
- Comentário do `config.h` que citava "2 dos 311.385 registros" atualizado.
  Nenhuma constante do firmware mudou.

## Arquivos

```
alterados   src/validacao_hardware.py  src/gerar_base_sintetica.py
            dashboard/app.py  dashboard/tab_validacao_hardware.py  dashboard/charts_validacao.py
            firmware/esp32/include/config.h (só comentário)  README.md
novos       src/estabilidade.py  tests/test_projeto.py  pytest.ini  requirements-dev.txt
            ALTERACOES_07-10.md
regerados   data/processed/uah_trips.csv, grade_calibracao.csv e os validacao_hardware_uah_*.csv
novos       data/processed/validacao_hardware_uah_qualidade.csv, estabilidade_*.csv,
            sensibilidade_escala.csv
```
