# Alterações de 21/09/2026 — dashboard e base sintética

Quatro arquivos mudaram. Três já existiam; um é novo.

```
src/gerar_base_sintetica.py   alterado
src/preprocessamento.py       alterado
src/sensibilidade_sementes.py NOVO
dashboard/app.py              alterado
```

Para aplicar, rode o pipeline na ordem de sempre:

```
python src/gerar_base_sintetica.py
python src/preprocessamento.py
python src/clustering.py
python src/perfilamento.py
python src/sensibilidade_sementes.py
```

---

## 1. A base sintética agora é calibrada contra o dado real

Antes, as probabilidades de qst1 e qst2 por perfil eram escolhidas a mão. O
problema não era a plausibilidade delas — era não haver como verificar se a
base reproduzia o que você observou nas 100 cotações reais.

As probabilidades foram reajustadas por busca numérica para reproduzir as três
marginais da Seção 3.3, sob duas restrições: nenhuma probabilidade pode ser 0
ou 1 (senão o vazamento de rótulo volta), e a ordenação qualitativa da Seção
3.4 tem de ser preservada.

| Estatística | Observado | Gerado |
|---|---|---|
| qst1 entre declarantes | 46,0% | 44,2% |
| qst2 entre declarantes | 73,0% | 73,7% |
| ambos entre declarantes | 19,0% | 17,9% |
| sinistro — mínimo | R$ 44.672 | R$ 44.760 |
| sinistro — máximo | R$ 855.906 | R$ 842.095 |
| sinistro — média | R$ 299.294 | R$ 301.314 |

O gerador imprime essa tabela ao final de cada execução. A aderência deixou de
ser afirmação e passou a ser saída verificável.

Sobre os valores de sinistro: a distribuição uniforme anterior tinha média
R$ 450.289, cinquenta por cento acima da média real. Uma média bem abaixo do
ponto médio da faixa indica cauda direita longa — muitos sinistros moderados e
poucos catastróficos. Entrou uma lognormal truncada à faixa observada, com
parâmetros ajustados para reproduzir exatamente aquela média.

## 2. A base agora tem lacunas, e a imputação finalmente roda

Cinco campos passam a vir nulos com probabilidade configurável (`--taxa-ausencia`,
padrão 4%): classe de risco, motorista licenciado, qst1, qst2 e data de emissão.

O motivo é direto: sem nenhuma ausência, todo o tratamento de valores ausentes
descrito na Seção 3.5.1 existia no código e nunca executava. A aderência dele ao
dado real nunca tinha sido testada.

Junto veio uma correção material no `preprocessamento.py`. Os campos booleanos
eram convertidos com `int(bool(...))`, o que transforma nulo em zero
silenciosamente — um quesito **não respondido** virava um "não". Isso enviesaria
para baixo justamente as duas variáveis de sinistralidade declarada. Agora o
nulo é preservado, chega como NaN e é imputado pela moda, com critério explícito.

## 3. Análise de sensibilidade por semente (arquivo novo)

`src/sensibilidade_sementes.py` roda o pipeline inteiro sobre várias sementes e
reporta a faixa. Resultado com cinco sementes:

| | Silhouette | Concordância |
|---|---|---|
| mediana | 0,2120 | 96,7% |
| mínimo | 0,1962 | 95,0% |
| máximo | 0,2151 | 98,3% |

**É essa faixa que deve ir para o texto, não o valor de uma semente.** Um número
isolado não distingue estrutura do dado de sorte do sorteio. Uma faixa estreita
como essa mostra que a estrutura recuperada não depende da realização particular.

## 4. Dashboard

**Os dois radares viraram barras horizontais ordenadas.** O radar é ruim
exatamente na tarefa que essas figuras precisam cumprir: comparar magnitudes. A
área do polígono cresce com o quadrado do raio, o que exagera diferenças grandes
e esconde as pequenas; a ordem dos eixos é arbitrária e muda a forma sem mudar o
dado; e a leitura exige comparar distâncias radiais em ângulos diferentes. As
barras resolvem os três: escala linear, ordem informativa (da variável mais
discriminante para a menos) e comparação ao longo de um eixo comum.

**`tipo_sinistro_predominante` saiu da figura.** É um código nominal
(1=tombamento, 2=incêndio, 3=terceiros). A média desse código não significa
nada, e numa barra de 0 a 100 seria lida como "mais" ou "menos" de alguma coisa.
O V de Cramér dela continua no ranking de discriminância, onde a associação é
medida sem pressupor ordem.

**O subtítulo mentia.** Dizia "menor p-valor nos testes de significância"
enquanto o código já ordenava por tamanho de efeito. Essa correção tinha sido
feita ontem mas não chegou ao disco — o OneDrive reverteu o arquivo.

**A tabela de segmentação passou a abrir a aba**, em largura total, e a projeção
PCA desceu para um bloco recolhido. A tabela é a resposta de negócio; o PCA é
metodologia — mostra *como* os grupos se separam, não *quais* são.

**Entrou a ideia central acima das abas**, na linha do que o Knaflic recomenda:
uma frase declarativa dizendo o que o painel afirma, e não o que ele contém.
Sem ela, quem abre encontra três abas nomeadas por componente técnico e precisa
montar a conclusão sozinho.

---

## Números que mudaram

A recalibração alterou os resultados. Com a semente 42:

| Métrica | Antes | Agora |
|---|---|---|
| Concordância | 97,0% | 98,3% |
| Silhouette (k=3) | 0,227 | 0,2151 |

Os três índices internos continuam favorecendo k=2 — essa divergência ainda
precisa ser declarada e justificada no texto.
