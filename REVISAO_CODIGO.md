# Revisão do código — o que corrigir antes de entregar

Revisão feita em 20/09/2026, cobrindo `src/`, `dashboard/` e o firmware. O
pipeline Python foi executado de ponta a ponta (`gerar_base_sintetica` →
`preprocessamento` → `clustering` → `perfilamento`) e os módulos do dashboard
foram importados e renderizados.

**Resultado geral: tudo executa sem erro.** Nenhum problema de execução. O que
está listado abaixo é metodológico e de coerência entre código e texto — que é
justamente o que pega em banca.

Os itens estão ordenados por risco, não por esforço.

---

## Crítico — pode derrubar na defesa

### 1. Os testes de significância por cluster são circulares

`src/perfilamento.py:43-64`

O `teste_anova()` compara, entre clusters, as mesmas variáveis que o K-Means
usou para *criar* os clusters (`COLUNAS_CONTINUAS_TESTE` é
`COLUNAS_CONTINUAS` mais três colunas). O algoritmo maximiza a separação nessas
variáveis e o teste depois "confirma" que elas separam. Isso se chama *double
dipping*, e o p-valor resultante não é inferencial.

Na execução de teste, **as 19 variáveis deram p < 1e-30** — `pct_autonomos`
saiu em p = 1,4e-179. Um p-valor desses não é um achado forte; é o sintoma da
circularidade.

**Correção:** reclassificar a tabela como **descritiva**, trocando p-valor por
tamanho de efeito (η² para contínuas, V de Cramér para categóricas). Remover ou
renomear a coluna `significativo_5pct`. E declarar no texto que p-valores
pós-clustering não têm interpretação inferencial. A tabela continua útil — ela
descreve os perfis —, só não pode ser apresentada como teste de hipótese.

### 2. A base sintética torna o resultado tautológico

`src/gerar_base_sintetica.py:128-156`

Quatro variáveis são praticamente o rótulo disfarçado:

| Variável | alto_risco | alta_cobertura | referral |
|---|---|---|---|
| `valorAgravo` | > 0 (60%) | **sempre 0** | > 0 (30%) |
| `valorDesconto` | **sempre 0** | > 0 (75%) | **sempre 0** |
| `vlJuros` | > 0 (50%) | **sempre 0** | > 0 (70%) |

`valorDesconto > 0` implica `alta_cobertura` com certeza absoluta. O K-Means não
descobriu perfis latentes: ele releu a regra do gerador. A crosstab de
`perfilamento.py:78` acertou **598 de 600** contra `_perfil_gerador`.

Isso também explica uma tensão que a banca pode notar: o Silhouette medido foi
**0,32**, que pela régua de Kaufman & Rousseeuw é "estrutura fraca" — convivendo
com 99,7% de recuperação do rótulo. Os dois números juntos denunciam o gerador.

**Correção:** sobrepor as distribuições. Dar agravo, desconto e juros a *todos*
os perfis, com probabilidades e faixas diferentes em vez de zero absoluto. E
apresentar a crosstab como **teste de sanidade do pipeline**, nunca como
validação do método de segmentação.

### 3. O rodapé do dashboard afirma algo que o código não faz

`dashboard/app.py:913`

> "Segmentação validada por Silhouette, Davies-Bouldin e Calinski-Harabasz
> (ver src/clustering.py)"

`clustering.py` calcula **apenas inércia e Silhouette** — conferido, não há
`davies_bouldin_score` nem `calinski_harabasz_score` em lugar nenhum do
repositório. O texto convida a banca a abrir o arquivo e não encontrar.

**Correção:** ou acrescentar as duas métricas em `curva_silhouette()` (são duas
linhas de import e duas de cálculo), ou apagar os dois nomes do rodapé. A
primeira opção é melhor e custa dez minutos.

### 4. Ponto de operação escolhido e reportado na mesma amostra

`src/validacao_hardware.py:415-487`, exibido em
`dashboard/tab_validacao_hardware.py:346-397`

`varrer_calibracao` testa 546 pares (limiar, N) nos mesmos 40 trajetos, e
`_melhor_par` reporta sensibilidade, especificidade e Youden do vencedor. Isso é
o máximo de 546 estimativas ruidosas na mesma amostra — otimista por construção.
O mesmo vício está em `derivar_limiar_recalibrado` (`:322`), que deriva o
percentil 99,9 dos trajetos normais e avalia no mesmo conjunto normal.

**Correção:** *leave-one-driver-out*. Escolher o par em 5 motoristas e medir no
6º, rodando as 6 combinações. Reportar a faixa entre folds, não um número único.
É a correção mais trabalhosa desta lista, mas é a que mais protege o resultado.

### 5. ANOVA com trajetos não independentes

`src/validacao_hardware.py:553-561`

Os 40 trajetos vêm de apenas 6 motoristas — trajetos do mesmo motorista não são
observações independentes, e a ANOVA assume que são. Também não há checagem de
normalidade nem de homocedasticidade, e `eventos_por_min` é ~0 em quase toda a
base (distribuição degenerada). São ainda 4 ANOVAs sem correção para múltiplas
comparações.

**Correção:** teste de permutação estratificado por motorista, ou modelo misto
com motorista como efeito aleatório. A boa notícia: a **Figura 4 (delta pareado
intra-motorista)** já resolve o problema conceitualmente. Promova-a a análise
principal e rebaixe a ANOVA a contexto — é uma mudança de ênfase no texto, não
de código.

### 6. Diferença de taxa de amostragem não declarada

`src/validacao_hardware.py:196`

A contagem é feita **por amostra isolada** que cruza o limiar, sobre dados a
10 Hz. O firmware amostra a **50 Hz** e tem período refratário de 2 s. A métrica
`eventos_por_min` escala com a taxa de amostragem: o mesmo sinal geraria uma
contagem muito diferente no dispositivo real.

**Correção:** agrupar cruzamentos consecutivos em um único evento (janela
refratária, espelhando o firmware) e declarar a diferença de taxa junto da
ressalva de 2 eixos e ausência de jerk que o módulo **já declara honestamente**
na docstring. Essa ressalva existente é um ponto forte do trabalho — vale
completá-la.

### 7. Glossário do dashboard contradiz o código

`dashboard/tab_validacao_hardware.py:236-238`

Define magnitude como "‖(x,y,z)‖ depois do filtro passa-baixa e da subtração da
gravidade". Mas `calcular_magnitude_ms2` (`validacao_hardware.py:186-188`) usa
**dois eixos** (`acc_y_kf`, `acc_z_kf`), sem passa-baixa próprio e **sem
subtração de gravidade** — como a própria docstring do módulo admite.

O texto na tela desfaz a ressalva honesta que está no código. **Correção:**
"‖(y,z)‖ do sinal já filtrado por Kalman entregue pelo dataset".

---

## Importante — corrigir antes de entregar

**Variável nominal tratada como numérica no K-Means.**
`tipo_sinistro_predominante` entra como 0/1/2/3 em `preprocessamento.py:20-28,51`
(distância euclidiana sobre código nominal não significa nada), mas é tratada
como categórica no qui-quadrado de `perfilamento.py:34` — incoerência interna.
Usar one-hot, ou tirar do K-Means.

**Min-Max esmaga as variáveis de cauda longa.** `preprocessamento.py:188-192`:
`valor_pago_historico` tem mediana 0 e máximo R$ 2,1 milhões; normalizada, o
terceiro quartil fica em 0,16 — a variável quase não pesa. Aplicar `log1p` antes
do Min-Max, ou usar RobustScaler, e justificar a escolha no texto.

**`requirements.txt` sem nenhuma versão fixada.** O código exige `pandas >= 2.1`
(`future_stack` em `perfilamento.py:36`) e Streamlit recente (`st.pills` em
`app.py:429`). Numa máquina com versões antigas, quebra. Pinar tudo e registrar
a versão do Python.

**Rótulo comportamental buscado no caminho inteiro.**
`validacao_hardware.py:132-134`: se qualquer pasta acima do trajeto contiver a
palavra "normal", todos os trajetos abaixo são rotulados assim. Restringir a
`caminho_trajeto.name`.

**Números fixos no texto do dashboard.** "311 mil amostras" e "40 trajetos"
estão escritos à mão em `app.py:533,666,785` e
`tab_validacao_hardware.py:79,112`. Se você reprocessar com outro recorte, a
tela mente. Derivar do DataFrame — a Figura 2 (`:294`) já faz isso certo.

**`tratar_ausentes` roda antes de `remover_duplicados`.**
`preprocessamento.py:204-205`: medianas e modas da imputação saem contaminadas
pelas duplicatas. Basta inverter a ordem.

**`date.today()` dentro do gerador.** `gerar_base_sintetica.py:160`: com a mesma
seed, as datas mudam a cada execução — a base não é reprodutível. Receber
`--data-base` como parâmetro.

**Mensagem de erro cita função que não existe.**
`validacao_hardware.validate_against_uah_driveset()` em
`tab_validacao_hardware.py:167` e `charts_validacao.py:14`. As funções reais são
`processar_dataset` e `montar_trajetos`.

**`numero_perfil` pode mapear dois clusters ao mesmo perfil** sem aviso
(`app.py:202-212`), duplicando silenciosamente tabela, KPIs e filtro. Um
`assert` de bijeção resolve.

---

## Se sobrar tempo

Nenhum teste automatizado no repositório. Um `pytest` mínimo que importe cada
módulo do dashboard e renderize com dados sintéticos evita descobrir uma quebra
na véspera da entrega.

O `README.md` da raiz tem duas linhas e não descreve a ordem de execução dos
scripts. Para quem avalia, isso é o primeiro contato com o trabalho.

`parse_qst3` (`preprocessamento.py:96`) resolve empates silenciosamente pela
ordem do dicionário, favorecendo "tombamento", e conta substrings soltas.

A função `render()` da aba longa não é chamada por `app.py` — está documentada
mas não é usada.

---

## Privacidade — sem divergência

Verificado especificamente: `latitude` e `longitude` são lidas em `GPS_COLUNAS`
(`validacao_hardware.py:118`), mas **nunca agregadas, salvas nem plotadas**.
`resumir_gps` (`:163-172`) devolve apenas velocidade média/máxima e variação de
curso. Não há mapa nem coordenada em nenhum CSV de `data/processed/` nem em
nenhuma figura do dashboard.

O pipeline está **coerente com a decisão de escopo da Seção B.3/C**, que o
firmware repete em três arquivos. Sugestão apenas cosmética: descartar as duas
colunas já no `read_csv`, para que a coerência fique explícita no código e não
só no comportamento — isso vira uma frase fácil de defender.

---

## Ordem sugerida de ataque

Pensando na entrega de novembro, e considerando que os itens 1, 2 e 3 são de
texto e poucas linhas:

1. Itens 3, 7 e os "números fixos" — são correções de texto, saem numa tarde e
   eliminam contradições diretas entre tela e código.
2. Itens 1 e 2 — mudam como você **apresenta** os resultados, não como os
   calcula. Fazer cedo, porque afetam a redação do TCC inteiro.
3. Item 5 — trocar a ênfase para a Figura 4 é reescrita, não recálculo.
4. Item 6 — agrupar eventos consecutivos, algumas dezenas de linhas.
5. Item 4 — leave-one-driver-out é o mais trabalhoso. Reservar um fim de semana.
6. Bloco "Importante" — cada um é rápido isoladamente.
