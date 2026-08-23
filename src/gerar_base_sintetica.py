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


def gerar_valores(perfil: str, total_veiculos: int, rng: random.Random) -> dict:
    total_veiculos = max(total_veiculos, 1)

    if perfil == "alto_risco":
        premio_por_veiculo = rng.uniform(1800, 4500)
        lmi_por_veiculo = rng.uniform(15000, 40000)
        valor_agravo = rng.uniform(500, 5000) if rng.random() < 0.6 else 0.0
        valor_desconto = 0.0
        vl_juros = rng.uniform(50, 400) if rng.random() < 0.5 else 0.0
    elif perfil == "alta_cobertura":
        premio_por_veiculo = rng.uniform(900, 2200)
        lmi_por_veiculo = rng.uniform(30000, 80000)
        valor_agravo = 0.0
        valor_desconto = rng.uniform(200, 2000) if rng.random() < 0.75 else 0.0
        vl_juros = 0.0
    else:  # referral
        premio_por_veiculo = rng.uniform(1200, 3000)
        lmi_por_veiculo = rng.uniform(20000, 50000)
        valor_agravo = rng.uniform(0, 1500) if rng.random() < 0.3 else 0.0
        valor_desconto = 0.0
        vl_juros = rng.uniform(50, 500) if rng.random() < 0.7 else 0.0

    return {
        "premioLiquido": round(premio_por_veiculo * total_veiculos, 2),
        "danosMateriais": round(lmi_por_veiculo * total_veiculos, 2),
        "valorAgravo": round(valor_agravo, 2),
        "valorDesconto": round(valor_desconto, 2),
        "vlJuros": round(vl_juros, 2),
    }


def gerar_datas(perfil: str, rng: random.Random) -> dict:
    hoje = date.today()
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
    if perfil == "referral":
        referral = rng.choice([
            "Analise de comite - risco acima do apetite padrao",
            "Excecao tarifaria solicitada pelo corretor",
            "Pendencia documental para aprovacao",
        ])
        apolice_anterior = None
    else:
        referral = "" if rng.random() < 0.9 else "Revisao de condicoes comerciais"
        tem_anterior = rng.random() < (0.7 if perfil == "alta_cobertura" else 0.4)
        apolice_anterior = (
            f"RCT-{rng.randint(2024, 2025)}-{rng.randint(1, 99999):06d}"
            if tem_anterior else None
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
            valor_sinistro = rng.uniform(44672, 855906)
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
    if perfil == "alto_risco":
        qst1 = rng.random() < 0.7
        qst2 = rng.random() < 0.85
    elif perfil == "alta_cobertura":
        qst1 = rng.random() < 0.1
        qst2 = rng.random() < 0.15
    else:
        qst1 = rng.random() < 0.35
        qst2 = rng.random() < 0.4

    qst3, _, _ = gerar_qst3(perfil, rng)

    return {"qst1": qst1, "qst2": qst2, "qst3": qst3}


def gerar_compliance(rng: random.Random) -> dict:
    return {"isHit": rng.random() < 0.02, "pep": rng.random() < 0.015}


def gerar_registro(indice: int, rng: random.Random, pesos: dict) -> dict:
    perfil = sortear_perfil(rng, pesos)

    frota = gerar_frota(perfil, rng)
    total_veiculos = frota["proprios"] + frota["autonomos"] + frota["terceiros"]

    datas = gerar_datas(perfil, rng)
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
    return registro


def gerar_base(n: int, seed: int, pesos: dict) -> list:
    rng = random.Random(seed)
    Faker.seed(seed)
    return [gerar_registro(i + 1, rng, pesos) for i in range(n)]


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
    args = parser.parse_args()

    pesos = {
        "alto_risco": args.peso_alto_risco,
        "alta_cobertura": args.peso_alta_cobertura,
        "referral": args.peso_referral,
    }

    base = gerar_base(args.n, args.seed, pesos)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "cotacoes_sinteticas.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=2)

    print(f"{len(base)} registros gravados em {out_path}")


if __name__ == "__main__":
    main()
