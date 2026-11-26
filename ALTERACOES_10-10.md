# Alterações de 10/10/2026 — segmentação, dashboard e Produto 5.2

> **Situação: APLICADAS, com autorização da autora.** Elas vêm da revisão dos cálculos registrada no documento do projeto "Parte B.11". Os originais foram guardados em `backup_10-10/` e, para voltar atrás, basta copiá-los de volta.

**Nenhum número do K-Means mudou:**
- a base sintética continua idêntica byte a byte;
- todos os CSVs de `data/processed/` continuam como estavam, e nenhum foi regerado.

## Arquivos

```
alterado  src/gerar_base_sintetica.py         (só reorganização; a saída é a mesma)
alterado  dashboard/app.py                    (aba 1, metodologia e aba 3)
novo      backup_10-10/gerar_base_sintetica.py  (original)
novo      backup_10-10/app.py                   (original)
novo      ../TCC/PRODUTO 5.2 - revisado 10-10.docx  (cópia com controle de alterações;
                                                    o "PRODUTO 5.2 - revisado.docx" ficou intocado)
novo      ALTERACOES_10-10.md
```

## 1. Gerador: `marginais_geradas()` de volta

O cálculo das marginais (qst1, qst2, as duas juntas e os valores de sinistro do perfil de alto risco) saiu de dentro de `relatorio_calibracao()` e foi para uma função própria. O relatório passa a usar essa função.

O que foi conferido:
- a base gerada, `cotacoes_sinteticas.json`, é **idêntica byte a byte** à da pasta;
- o relatório impresso também é idêntico;
- `src/estabilidade.py` volta a rodar e reproduz os cinco CSVs com diferença 0,0: `estabilidade_*.csv` e `sensibilidade_escala.csv`.

**Testes (`python -m pytest`):** 10 passam e 2 são pulados, porque falta o UAH-DriveSet bruto. Os **5 que falham** testam funções da proposta de hardware de 07/10 (`marcar_inicios_de_evento`, `duracao_sem_calibracao`), que continua não aplicada. Eles vão falhar até você decidir sobre essa proposta. Todos os testes da segmentação e o teste "o painel abre sem erro" passam.

## 2. Dashboard (`dashboard/app.py`)

### Aba 1 — A carteira tem três perfis
- **Cartões:** os quatro passam a ler sempre a mesma coluna, a carteira inteira ou o perfil filtrado. Antes, em "Carteira completa", o cartão de custo mostrava o Perfil 1.
- **Cartão de custo:** virou **"Com sinistro pago"**, com o percentual de apólices e a mediana entre elas. A média do valor pago engana, porque a mediana é zero nos Perfis 2 e 3.
- **Prêmio:** saiu o sufixo "por ano", porque a base não define a duração da vigência.
- **Tabela de perfis:** foi transposta e ficou igual ao **Quadro 9**, com os mesmos números, a mesma ordem e a coluna da carteira.
  - As linhas estão agrupadas em frota, contrato, sinistralidade e processo.
  - Entraram motorista licenciado, classe de risco, coberturas, agravo, desconto, qst1 e qst2, referral, dias até a emissão, juros e apólice anterior.
  - O valor pago aparece em três linhas: % com valor pago, mediana entre essas apólices e média de todas.
  - O perfil escolhido no filtro fica destacado.
- **Nova figura:** tipo de sinistro predominante por perfil, em barras 100% empilhadas.
  - A paleta foi validada para daltonismo e não usa as cores de perfil.
  - O percentual aparece escrito nos segmentos largos.

### Metodologia (bloco recolhido)
- **Tooltip do PCA:** além do número, mostra o perfil, a frota, o % de autônomos, se há referral, os dias até a emissão e o valor pago. Antes mostrava só o número da apólice.
- **Novo bloco de silhueta por perfil:** 0,196, 0,293 e 0,137, com 16 apólices negativas. A conta é feita só com numpy e foi conferida contra o scikit-learn (diferença de 1,6 × 10⁻¹⁵).
- **Nova nota na concordância:** algumas variáveis contínuas ocupam faixas exclusivas de um perfil no gerador (Seção 4.1).

### Aba 3 — Onde as duas camadas se encontram
- **Subtítulo:** passa a dizer que o tipo de sinistro, 5º no ranking, fica fora da figura por ser nominal. Antes dizia "as 6 de maior efeito".
- **Título:** passa a ser "Na base sintética, a composição da frota separa mais que o histórico financeiro".

### Rótulo
- "% de autônomos/terceiros" passou a "% de autônomos". A variável é autônomos ÷ total.

**O que foi testado:**
- o painel renderiza sem exceção (AppTest), inclusive depois de filtrar por perfil;
- os valores da tela foram conferidos contra o Quadro 9;
- o visual foi conferido no navegador.

## 3. Produto 5.2 — `TCC/PRODUTO 5.2 - revisado 10-10.docx`

É uma cópia com **controle de alterações** (autor "Claude"), para você aceitar ou recusar cada mudança no Word, em Revisão → Aceitar/Rejeitar. Fora desses trechos, o texto é idêntico ao original, e o sumário e as equações ficaram intocados.

| Onde | Mudança |
|---|---|
| 3.4.1 | "veículos autônomos e terceiros" → "motoristas autônomos na frota" |
| 3.4.3 (2 trechos) | "início de vigência" / `dadosApolice.inicioVigencia` → "emissão" / `produto.dtEmissao` |
| 3.5.3 | a análise descritiva é feita na matriz no espaço original, e não na normalizada |
| Quadro 5 | nova linha `estabilidade.py` |
| 3.9 | declarada a limitação das faixas exclusivas no gerador (tempo, classe de risco, tipos de sinistro, seguradoras e valores) |
| 4.1 (2 frases) | a sobreposição vale para as binárias (agravo, desconto, juros e referral), com remissão à 3.9 |
| Quadro 9 e Quadro 10 | "Proporção de autônomos e terceiros" → "Proporção de autônomos" |
| 4.3, Perfil 1 | "autônomos ou terceiros" → "de motoristas autônomos"; "duas ordens de grandeza" → "63 vezes o do Perfil 3 e 213 vezes o do Perfil 2" |
| 4.3, observações 1 e 2 | qualificadas como resultado da base sintética, a confirmar na carteira real |
| 4.6 (3 trechos) | "majoritariamente terceirizada" → "com predominância de autônomos"; ressalva das faixas exclusivas; "vazamento nas variáveis binárias" |
| Conclusão (3 trechos) | achados como hipóteses a confirmar; vazamento "nas variáveis binárias de tratamento tarifário e de processo"; limitação das faixas exclusivas |

## Observado e NÃO alterado
- **Seção 3.4.2:** "franquia mínima" faz parte da definição de negócio do perfil, mas não é variável do modelo.
- **Seção 3.7.3:** o texto diz que a aba de segmentação traz "o gráfico de discriminância", mas ele fica na aba 3 do painel.
- **Aba 3 do painel:** as setas citam "Seção 3.4.1" para as transições entre perfis. A correção já estava na proposta de 07/10, que continua não aplicada.
- **Amostra das 100 cotações reais** (46%, 73%, 19%, R$ 299.294): continua sem arquivo de origem nas pastas.
