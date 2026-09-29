"""
Gera uma base sintetica de cotacoes do produto de Responsabilidade Civil do
Transportador (RCT), replicando a estrutura do objeto JSON de cotacao e do
registro de sinistralidade declarada (qst1/qst2/qst3) descritos na Secao 3.3
do TCC, com mistura latente dos tres perfis-alvo definidos na Secao 3.4:

  Perfil 1 - Frota de Alto Risco Operacional
  Perfil 2 - Segurado de Alta Cobertura e Baixo Custo Relativo
  Perfil 3 - Cotacao em Referral ou Conversao Tardia

A saida e um unico arquivo JSON (lista de objetos), um objeto por apolice,
gravado em data/raw/. Esta base alimenta o pipeline de pre-processamento
(src/preprocessamento.py), que calcula as 19 variaveis derivadas do Quadro 2.
"""

import argparse
import json
import random
import re
import unicodedata
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

fake = Faker("pt_BR")

PERFIS = ["alto_risco", "alta_cobertura", "referral"]

ATIVIDADES = [
    "Transporte rodoviario de cargas em geral",
    "Transporte de cargas frigorificadas",
    "Transporte de graneis solidos",
    "Transporte de produtos perigosos",
    "Transporte de mudancas",
    "Transporte de veiculos",
    "Transporte de bebidas",
    "Transporte de eletroeletronicos",
]

SEGURADORAS_ANTERIORES = [
    "Porto Seguro", "Bradesco Seguros", "SulAmerica", "Tokio Marine",
    "Allianz", "HDI Seguros", "Itau Seguros", "Mapfre", "Zurich", "Sompo",
]

TIPOS_SINISTRO = ["tombamento", "incendio", "terceiros"]

COBERTURAS_OPCIONAIS = [
    "rcfv", "operacoesAmplas", "prestacaoServicos", "danos",
    "perdas", "custos", "despesas", "brigada",
]


def _sem_acento(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def _fmt_brl(valor: float) -> str:
    return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _data_aleatoria(inicio: date, fim: date, rng: random.Random) -> date:
    delta_dias = (fim - inicio).days
    return inicio + timedelta(days=rng.randint(0, max(delta_dias, 0)))


def sortear_perfil(rng: random.Random, pesos: dict) -> str:
    return rng.choices(PERFIS, weights=[pesos[p] for p in PERFIS], k=1)[0]


def gerar_frota(perfil: str, rng: random.Random) -> dict:
    if perfil == "alto_risco":
        total = rng.randint(25, 150)
        pct_autonomos = rng.uniform(0.4, 0.9)
        motorista_licenciado = rng.random() < 0.2
    elif perfil == "alta_cobertura":
        total = rng.randint(5, 60)
        pct_autonomos = rng.uniform(0.0, 0.3)
        motorista_licenciado = rng.random() < 0.85
    else:  # referral
        total = rng.randint(3, 40)
        pct_autonomos = rng.uniform(0.1, 0.5)
        motorista_licenciado = rng.random() < 0.5

    autonomos = round(total * pct_autonomos)
    restante = total - autonomos
    terceiros = rng.randint(0, restante)
    proprios = restante - terceiros

    return {
        "proprios": proprios,
        "autonomos": autonomos,
        "terceiros": terceiros,
        "motoristaLicenciado": motorista_licenciado,
    }


def gerar_atividade(perfil: str, rng: random.Random) -> dict:
    if perfil == "alto_risco":
        classe_risco = rng.randint(4, 5)
    elif perfil == "alta_cobertura":
        classe_risco = rng.randint(1, 3)
    else:
        classe_risco = rng.randint(2, 4)

    return {
        "descricao": rng.choice(ATIVIDADES),
        "classeRisco": classe_risco,
    }


def gerar_coberturas(perfil: str, rng: random.Random) -> dict:
    if perfil == "alta_cobertura":
        prob_ativa = 0.85
    elif perfil == "alto_risco":
        prob_ativa = 0.35
    else:
        prob_ativa = 0.5

    coberturas = {"rc": True}
    for nome in COBERTURAS_OPCIONAIS:
        coberturas[nome] = rng.random() < prob_ativa
    return coberturas


# ---------------------------------------------------------------------------
# Calibracao contra a amostra exploratoria real (Secao 3.3)
# ---------------------------------------------------------------------------
#
# A analise exploratoria de 100 cotacoes REAIS com sinistralidade declarada
# positiva (Secao 3.3 do TCC) observou, dentro desse subconjunto:
#
#   46% declararam reclamacao de RC (qst1)
#   73% declararam reclamacao/indenizacao de danos a frota (qst2/RCFV)
#   19% declararam as duas condicoes simultaneamente
#
# e valores de sinistro indenizado por seguradoras anteriores entre
# R$ 44.672 e R$ 855.906, com media de R$ 299.294 por evento.
#
# As probabilidades abaixo NAO foram escolhidas a mao: foram ajustadas
# numericamente para que a base sintetica reproduza essas tres marginais,
# sob duas restricoes que preservam a validade metodologica do experimento:
#
#   (a) nenhuma probabilidade e 0 ou 1 -- os perfis diferem em PROPENSAO,
#       nao em possibilidade (ver PROPENSAO_VALORES abaixo);
#   (b) a ordenacao qualitativa da Secao 3.4 e mantida:
#       alto_risco > referral > alta_cobertura nos dois campos.
#
# O comando imprime ao final a comparacao entre o observado e o gerado, para
# que a aderencia seja verificavel e nao precise ser aceita sob palavra.
MARGINAIS_OBSERVADAS = {
    "qst1_entre_declarantes": 0.46,
    "qst2_entre_declarantes": 0.73,
    "ambos_entre_declarantes": 0.19,
    "sinistro_min": 44672.0,
    "sinistro_max": 855906.0,
    "sinistro_medio": 299294.0,
}

PROPENSAO_SINISTRALIDADE = {
    #                  qst1  qst2
    "alto_risco":     (0.39, 0.51),
    "alta_cobertura": (0.08, 0.24),
    "referral":       (0.22, 0.34),
}

# Distribuicao dos valores de sinistro indenizado. Uma uniforme entre o minimo
# e o maximo observados teria media R$ 450.289 -- 50% acima da media real, que
# e de R$ 299.294. A media bem abaixo do ponto medio da faixa indica cauda
# direita longa: muitos sinistros moderados e poucos catastroficos. Os
# parametros abaixo sao de uma lognormal truncada a faixa observada, ajustada
# por busca numerica para reproduzir exatamente aquela media.
SINISTRO_LOGNORMAL_MU = 12.4984
SINISTRO_LOGNORMAL_SIGMA = 0.8


def sortear_valor_sinistro(rng: random.Random) -> float:
    """Amostra da lognormal truncada a faixa observada, por rejeicao."""
    minimo = MARGINAIS_OBSERVADAS["sinistro_min"]
    maximo = MARGINAIS_OBSERVADAS["sinistro_max"]
    for _ in range(200):
        valor = rng.lognormvariate(SINISTRO_LOGNORMAL_MU, SINISTRO_LOGNORMAL_SIGMA)
        if minimo <= valor <= maximo:
            return valor
    return rng.uniform(minimo, maximo)


# Probabilidade de cada perfil apresentar agravo, desconto e juros.
#
# IMPORTANTE (metodologia): nenhuma destas probabilidades pode ser 0 ou 1.
# Em uma versao anterior deste gerador, "valorDesconto > 0" ocorria
# exclusivamente no perfil alta_cobertura, e "valorAgravo > 0" nunca ocorria
# nele. Isso transformava tres das 19 variaveis derivadas em rotulo
# disfarcado: o K-Means nao descobria perfis latentes, apenas relia a regra do
# gerador, recuperando ~99,7% dos rotulos. As distribuicoes agora se sobrepoem
# -- os perfis diferem em PROPENSAO, nao em possibilidade --, que e o que se
# espera de dados reais e o que torna a segmentacao um problema de fato.
PROPENSAO_VALORES = {
    #                  agravo  desconto  juros
    "alto_risco":     (0.60,   0.15,     0.50),
    "alta_cobertura": (0.15,   0.75,     0.20),
    "referral":       (0.30,   0.25,     0.70),
}


def gerar_valores(perfil: str, total_veiculos: int, rng: random.Random) -> dict:
    total_veiculos = max(total_veiculos, 1)

    if perfil == "alto_risco":
        premio_por_veiculo = rng.uniform(1800, 4500)
        lmi_por_veiculo = rng.uniform(15000, 40000)
    elif perfil == "alta_cobertura":
        premio_por_veiculo = rng.uniform(900, 2200)
        lmi_por_veiculo = rng.uniform(30000, 80000)
    else:  # referral
        premio_por_veiculo = rng.uniform(1200, 3000)
        lmi_por_veiculo = rng.uniform(20000, 50000)

    p_agravo, p_desconto, p_juros = PROPENSAO_VALORES[perfil]
    valor_agravo = rng.uniform(300, 5000) if rng.random() < p_agravo else 0.0
    valor_desconto = rng.uniform(200, 2000) if rng.random() < p_desconto else 0.0
    vl_juros = rng.uniform(50, 500) if rng.random() < p_juros else 0.0

    return {
        "premioLiquido": round(premio_por_veiculo * total_veiculos, 2),
        "danosMateriais": round(lmi_por_veiculo * total_veiculos, 2),
        "valorAgravo": round(valor_agravo, 2),
        "valorDesconto": round(valor_desconto, 2),
        "vlJuros": round(vl_juros, 2),
    }


def gerar_datas(perfil: str, rng: random.Random, hoje: date) -> dict:
    """`hoje` e recebido como parametro (e nao lido de date.today()) para que a
    base seja reprodutivel: com a mesma seed e a mesma data-base, o arquivo
    gerado e byte a byte identico."""
    lancamento_produto = hoje - timedelta(days=730)
    dt_criacao = _data_aleatoria(lancamento_produto, hoje - timedelta(days=1), rng)

    if perfil == "referral":
        tempo_emissao = rng.randint(15, 90)
    else:
        tempo_emissao = rng.randint(1, 10)

    dt_emissao = dt_criacao + timedelta(days=tempo_emissao)
    inicio_vigencia = dt_emissao + timedelta(days=rng.randint(0, 3))

    return {
        "dtCriacao": dt_criacao.isoformat(),
        "dtEmissao": dt_emissao.isoformat(),
        "inicioVigencia": inicio_vigencia.isoformat(),
    }


def gerar_referral_e_apolice_anterior(perfil: str, rng: random.Random) -> dict:
    # Tambem aqui as distribuicoes se sobrepoem: um referral pendente e mais
    # PROVAVEL no perfil de referral, nao exclusivo dele. Antes, "referral
    # preenchido" identificava o perfil com certeza absoluta.
    prob_referral = {"referral": 0.75, "alto_risco": 0.15, "alta_cobertura": 0.08}[perfil]
    prob_anterior = {"referral": 0.25, "alto_risco": 0.40, "alta_cobertura": 0.70}[perfil]

    if rng.random() < prob_referral:
        referral = rng.choice([
            "Analise de comite - risco acima do apetite padrao",
            "Excecao tarifaria solicitada pelo corretor",
            "Pendencia documental para aprovacao",
            "Revisao de condicoes comerciais",
        ])
    else:
        referral = ""

    apolice_anterior = (
        f"RCT-{rng.randint(2024, 2025)}-{rng.randint(1, 99999):06d}"
        if rng.random() < prob_anterior else None
    )

    return {"referral": referral, "apoliceAnterior": apolice_anterior}


def gerar_qst3(perfil: str, rng: random.Random) -> tuple[str, float, float]:
    """Retorna (texto_qst3, soma_valor_pago, soma_valor_sinistro) para
    conferencia opcional do gerador; o parsing real e feito depois pelo
    preprocessamento.py via regex sobre o texto."""

    if perfil == "alto_risco":
        qt_cias = rng.randint(1, 3)
        blocos = []
        soma_pago, soma_sinistro = 0.0, 0.0
        for _ in range(qt_cias):
            valor_sinistro = sortear_valor_sinistro(rng)
            valor_pago = valor_sinistro * rng.uniform(0.7, 1.0)
            soma_pago += valor_pago
            soma_sinistro += valor_sinistro
            blocos.append(
                f"Seguradora Anterior: {rng.choice(SEGURADORAS_ANTERIORES)}. "
                f"Vigencia: {rng.randint(2022, 2025)}. "
                f"Pago: R$ {_fmt_brl(valor_pago)}. "
                f"Valor sinistro: R$ {_fmt_brl(valor_sinistro)}. "
                f"Tipo: {rng.choice(TIPOS_SINISTRO)}."
            )
        return " ".join(blocos), round(soma_pago, 2), round(soma_sinistro, 2)

    if perfil == "alta_cobertura":
        if rng.random() < 0.85:
            return "Sem sinistros declarados nos ultimos 5 anos.", 0.0, 0.0
        valor_sinistro = rng.uniform(5000, 30000)
        valor_pago = valor_sinistro * rng.uniform(0.8, 1.0)
        texto = (
            f"Seguradora Anterior: {rng.choice(SEGURADORAS_ANTERIORES)}. "
            f"Vigencia: {rng.randint(2023, 2025)}. "
            f"Pago: R$ {_fmt_brl(valor_pago)}. "
            f"Valor sinistro: R$ {_fmt_brl(valor_sinistro)}. "
            f"Tipo: terceiros."
        )
        return texto, round(valor_pago, 2), round(valor_sinistro, 2)

    # referral: predominantemente vago/nao estruturado
    if rng.random() < 0.75:
        return rng.choice(["Em anexo.", "Documentacao a ser enviada.", ""]), None, None
    valor_sinistro = rng.uniform(8000, 60000)
    valor_pago = valor_sinistro * rng.uniform(0.6, 1.0)
    texto = (
        f"Seguradora Anterior: {rng.choice(SEGURADORAS_ANTERIORES)}. "
        f"Pago: R$ {_fmt_brl(valor_pago)}. "
        f"Valor sinistro: R$ {_fmt_brl(valor_sinistro)}."
    )
    return texto, round(valor_pago, 2), round(valor_sinistro, 2)


def gerar_sinistralidade(perfil: str, rng: random.Random) -> dict:
    """qst1/qst2 saem de PROPENSAO_SINISTRALIDADE, ajustada para reproduzir as
    marginais da amostra exploratoria real (ver MARGINAIS_OBSERVADAS)."""
    p_qst1, p_qst2 = PROPENSAO_SINISTRALIDADE[perfil]
    qst1 = rng.random() < p_qst1
    qst2 = rng.random() < p_qst2

    qst3, _, _ = gerar_qst3(perfil, rng)

    return {"qst1": qst1, "qst2": qst2, "qst3": qst3}


def gerar_compliance(rng: random.Random) -> dict:
    return {"isHit": rng.random() < 0.02, "pep": rng.random() < 0.015}


# ---------------------------------------------------------------------------
# Ausencias realistas
# ---------------------------------------------------------------------------
#
# O dado real de cotacao e sujo: o corretor deixa a classe de risco em branco,
# nao responde os quesitos de sinistralidade, o campo de motorista licenciado
# volta nulo do formulario. Uma base sintetica sem nenhuma ausencia faz com
# que todo o tratamento de valores ausentes do pre-processamento (Secao 3.5.1)
# nunca seja exercitado -- o codigo existe e nunca roda, e a aderencia dele ao
# dado real nunca e testada. As ausencias abaixo cobrem os quatro campos em
# que a falta e mais comum no formulario real.
#
# A distincao importa: um quesito NAO RESPONDIDO nao e a mesma coisa que um
# quesito respondido "nao". O primeiro vira nulo e passa pela imputacao pela
# moda; o segundo e um false legitimo.
CAMPOS_COM_AUSENCIA = ("classeRisco", "motoristaLicenciado", "qst1", "qst2", "dtEmissao")


def aplicar_ausencias(registro: dict, taxa: float, rng: random.Random) -> dict:
    """Apaga campos isolados do registro com probabilidade `taxa` cada,
    simulando preenchimento incompleto do formulario de cotacao."""
    if taxa <= 0:
        return registro
    if rng.random() < taxa:
        registro["atividade"]["classeRisco"] = None
    if rng.random() < taxa:
        registro["frota"]["motoristaLicenciado"] = None
    if rng.random() < taxa:
        registro["sinistralidade"]["qst1"] = None
    if rng.random() < taxa:
        registro["sinistralidade"]["qst2"] = None
    if rng.random() < taxa:
        registro["produto"]["dtEmissao"] = None
    return registro


def gerar_registro(indice: int, rng: random.Random, pesos: dict, hoje: date,
                   taxa_ausencia: float = 0.0) -> dict:
    perfil = sortear_perfil(rng, pesos)

    frota = gerar_frota(perfil, rng)
    total_veiculos = frota["proprios"] + frota["autonomos"] + frota["terceiros"]

    datas = gerar_datas(perfil, rng, hoje)
    referral_info = gerar_referral_e_apolice_anterior(perfil, rng)

    registro = {
        "numeroApolice": f"RCT-{datas['dtEmissao'][:4]}-{indice:06d}",
        "produto": {
            "nome": "Responsabilidade Civil do Transportador",
            "dtCriacao": datas["dtCriacao"],
            "dtEmissao": datas["dtEmissao"],
        },
        "corretor": {
            "nome": _sem_acento(fake.company()),
            "susep": f"{rng.randint(10000, 99999)}.{rng.randint(1000, 9999)}",
        },
        "frota": frota,
        "atividade": gerar_atividade(perfil, rng),
        "coberturas": gerar_coberturas(perfil, rng),
        "valores": gerar_valores(perfil, total_veiculos, rng),
        "dadosApolice": {
            "inicioVigencia": datas["inicioVigencia"],
            "apoliceAnterior": referral_info["apoliceAnterior"],
        },
        "referral": referral_info["referral"],
        "compliance": gerar_compliance(rng),
        "sinistralidade": gerar_sinistralidade(perfil, rng),
        "_perfil_gerador": perfil,
    }
    return aplicar_ausencias(registro, taxa_ausencia, rng)


def gerar_base(n: int, seed: int, pesos: dict, hoje: date | None = None,
               taxa_ausencia: float = 0.04) -> list:
    rng = random.Random(seed)
    Faker.seed(seed)
    if hoje is None:
        hoje = date.today()
    return [gerar_registro(i + 1, rng, pesos, hoje, taxa_ausencia)
            for i in range(n)]


def relatorio_calibracao(base: list) -> str:
    """Compara a base gerada com as marginais da amostra exploratoria real.

    Existe para que a aderencia seja VERIFICAVEL: sem esta saida, a afirmacao
    de que a base sintetica reproduz as estatisticas observadas teria de ser
    aceita sob palavra. Imprime tambem a taxa de ausencia efetiva, que e o que
    exercita o tratamento de valores ausentes do pre-processamento.
    """
    declarantes = [r for r in base
                   if r["sinistralidade"]["qst1"] or r["sinistralidade"]["qst2"]]
    n = len(declarantes)
    if n == 0:
        return "Nenhum registro com sinistralidade declarada positiva."

    q1 = sum(1 for r in declarantes if r["sinistralidade"]["qst1"]) / n
    q2 = sum(1 for r in declarantes if r["sinistralidade"]["qst2"]) / n
    ambos = sum(1 for r in declarantes
                if r["sinistralidade"]["qst1"] and r["sinistralidade"]["qst2"]) / n

    # A faixa observada (R$ 44.672 a R$ 855.906) veio dos registros REAIS com
    # historico financeiro detalhado, que sao os de sinistralidade relevante --
    # nao das cotacoes com um arranhao de terceiros. A comparacao justa e,
    # portanto, contra o perfil de alto risco, que e o analogo sintetico
    # daquele subconjunto. Os sinistros pequenos dos outros dois perfis
    # continuam existindo na base; apenas nao entram nesta linha.
    valores = []
    for r in base:
        if r.get("_perfil_gerador") != "alto_risco":
            continue
        for trecho in re.findall(r"Valor sinistro: R\$ ([\d\.]+,\d{2})",
                                 r["sinistralidade"]["qst3"] or ""):
            valores.append(float(trecho.replace(".", "").replace(",", ".")))

    obs = MARGINAIS_OBSERVADAS
    linhas = [
        "",
        "Calibracao contra a amostra exploratoria real (Secao 3.3)",
        "-" * 68,
        f"{'Estatistica':<40}{'Observado':>13}{'Gerado':>15}",
        f"{'qst1 entre declarantes':<40}"
        f"{obs['qst1_entre_declarantes']:>12.1%}{q1:>15.1%}",
        f"{'qst2 entre declarantes':<40}"
        f"{obs['qst2_entre_declarantes']:>12.1%}{q2:>15.1%}",
        f"{'ambos entre declarantes':<40}"
        f"{obs['ambos_entre_declarantes']:>12.1%}{ambos:>15.1%}",
    ]
    if valores:
        linhas += [
            f"{'sinistro (perfil alto risco) - minimo':<40}"
            f"{obs['sinistro_min']:>12,.0f}{min(valores):>15,.0f}",
            f"{'sinistro (perfil alto risco) - maximo':<40}"
            f"{obs['sinistro_max']:>12,.0f}{max(valores):>15,.0f}",
            f"{'sinistro (perfil alto risco) - media':<40}"
            f"{obs['sinistro_medio']:>12,.0f}{sum(valores)/len(valores):>15,.0f}",
        ]

    ausentes = {
        "classeRisco": sum(1 for r in base if r["atividade"]["classeRisco"] is None),
        "motoristaLicenciado": sum(1 for r in base
                                   if r["frota"]["motoristaLicenciado"] is None),
        "qst1": sum(1 for r in base if r["sinistralidade"]["qst1"] is None),
        "qst2": sum(1 for r in base if r["sinistralidade"]["qst2"] is None),
        "dtEmissao": sum(1 for r in base if r["produto"]["dtEmissao"] is None),
    }
    total = len(base)
    linhas += ["", "Ausencias geradas (exercitam a imputacao do pre-processamento)"]
    linhas += [f"  {campo:<24}{qtd:>5} registros ({qtd / total:.1%})"
               for campo, qtd in ausentes.items()]
    linhas.append("")
    return "\n".join(linhas)


def main():
    parser = argparse.ArgumentParser(
        description="Gera base sintetica de cotacoes RCT para o pipeline de segmentacao."
    )
    parser.add_argument("--n", type=int, default=600, help="numero de registros a gerar")
    parser.add_argument("--seed", type=int, default=42, help="semente para reprodutibilidade")
    parser.add_argument(
        "--out", type=str, default=None,
        help="caminho de saida (default: data/raw/cotacoes_sinteticas.json)"
    )
    parser.add_argument("--peso-alto-risco", type=float, default=0.30)
    parser.add_argument("--peso-alta-cobertura", type=float, default=0.35)
    parser.add_argument("--peso-referral", type=float, default=0.35)
    parser.add_argument(
        "--taxa-ausencia", type=float, default=0.04,
        help="probabilidade de cada campo sujeito a ausencia vir nulo "
             "(default: 0.04). Use 0 para gerar uma base sem lacunas."
    )
    parser.add_argument(
        "--data-base", type=str, default="2026-09-01",
        help="data de referencia das vigencias (AAAA-MM-DD). Fixa por padrao "
             "para que a base seja reprodutivel; use 'hoje' para a data atual."
    )
    args = parser.parse_args()

    pesos = {
        "alto_risco": args.peso_alto_risco,
        "alta_cobertura": args.peso_alta_cobertura,
        "referral": args.peso_referral,
    }

    if args.data_base.strip().lower() == "hoje":
        data_base = date.today()
    else:
        data_base = date.fromisoformat(args.data_base)

    base = gerar_base(args.n, args.seed, pesos, data_base, args.taxa_ausencia)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "cotacoes_sinteticas.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=2)

    print(f"{len(base)} registros gravados em {out_path} (data-base {data_base.isoformat()})")
    print(relatorio_calibracao(base))


if __name__ == "__main__":
    main()
