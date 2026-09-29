"""
Etapa 1 (Secao 3.5.1): parsing dos objetos JSON de cotacao RCT, extracao de
qst1/qst2 e parsing de qst3 por expressoes regulares, calculo das 19
variaveis derivadas (Quadro 2), tratamento de valores ausentes, remocao de
duplicados por numeroApolice e normalizacao Min-Max.

Uso:
    python src/preprocessamento.py --in data/raw/cotacoes_sinteticas.json
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

COLUNAS_19 = [
    "total_veiculos", "pct_autonomos", "motorista_licenciado", "classe_risco",
    "sinistralidade_rc_declarada", "sinistralidade_rcfv_declarada",
    "valor_pago_historico", "valor_sinistro_historico", "qt_cias_anteriores",
    "tipo_sinistro_predominante", "lmi_por_veiculo", "premio_por_veiculo",
    "qt_coberturas_ativas", "agravo_aplicado", "desconto_aplicado",
    "referral_pendente", "tempo_cotacao_emissao", "parcelas_com_juros",
    "apolice_anterior",
]

COLUNAS_BINARIAS = [
    "motorista_licenciado", "sinistralidade_rc_declarada",
    "sinistralidade_rcfv_declarada", "agravo_aplicado", "desconto_aplicado",
    "referral_pendente", "parcelas_com_juros", "apolice_anterior",
]

COLUNAS_QST3 = [
    "valor_pago_historico", "valor_sinistro_historico",
    "qt_cias_anteriores", "tipo_sinistro_predominante",
]

COLUNAS_CONTINUAS = [
    "total_veiculos", "pct_autonomos", "classe_risco", "lmi_por_veiculo",
    "premio_por_veiculo", "qt_coberturas_ativas", "tempo_cotacao_emissao",
]

COBERTURAS_OPCIONAIS = [
    "rcfv", "operacoesAmplas", "prestacaoServicos", "danos",
    "perdas", "custos", "despesas", "brigada",
]

TIPOS_SINISTRO_COD = {"tombamento": 1, "incendio": 2, "terceiros": 3}

# Variaveis monetarias/contagens de cauda longa. Aplicar Min-Max direto nelas
# comprime quase toda a massa perto de zero (valor_pago_historico tem mediana 0
# e maximo na casa dos milhoes: normalizada, o 3o quartil fica em ~0,16 e a
# variavel praticamente nao pesa na distancia euclidiana). log1p antes do
# Min-Max preserva a ordem e devolve resolucao a faixa onde estao os dados.
COLUNAS_LOG = [
    "valor_pago_historico", "valor_sinistro_historico",
    "lmi_por_veiculo", "premio_por_veiculo", "total_veiculos",
]

# tipo_sinistro_predominante e NOMINAL (0=nenhum, 1=tombamento, 2=incendio,
# 3=terceiros). Distancia euclidiana sobre esse codigo nao tem significado --
# "terceiros" nao esta tres vezes mais longe de "nenhum" que "tombamento".
# Por isso a matriz que alimenta o K-Means usa variaveis indicadoras no lugar
# do codigo, enquanto a matriz original preserva as 19 variaveis do Quadro 2
# como descritas no texto.
COLUNA_NOMINAL = "tipo_sinistro_predominante"
INDICADORAS_SINISTRO = [
    "sinistro_tombamento", "sinistro_incendio", "sinistro_terceiros",
]

# Colunas efetivamente usadas pelo modelo (19 - 1 nominal + 3 indicadoras).
COLUNAS_MODELO = [c for c in COLUNAS_19 if c != COLUNA_NOMINAL] + INDICADORAS_SINISTRO

PAGO_RE = re.compile(r"Pago:\s*R\$\s*(\d{1,3}(?:\.\d{3})*,\d{2})", re.IGNORECASE)
SINISTRO_RE = re.compile(r"Valor sinistro:\s*R\$\s*(\d{1,3}(?:\.\d{3})*,\d{2})", re.IGNORECASE)
SEGURADORA_RE = re.compile(r"Seguradora Anterior:", re.IGNORECASE)
TIPO_RE = re.compile(r"Tipo:\s*([A-Za-zÀ-ÿ]+)", re.IGNORECASE)


def _sem_acento(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def _brl_para_float(valor_str: str) -> float:
    return float(valor_str.replace(".", "").replace(",", "."))


def _get(d: dict, *chave, default=None):
    atual = d
    for k in chave:
        if not isinstance(atual, dict) or k not in atual:
            return default
        atual = atual[k]
    return atual


def _num0(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def parse_qst3(texto: str) -> dict:
    """Extrai, via regex, os valores financeiros, a contagem de seguradoras
    citadas e o tipo de sinistro predominante do campo textual livre qst3."""
    texto = texto or ""

    valores_pagos = [_brl_para_float(v) for v in PAGO_RE.findall(texto)]
    valores_sinistro = [_brl_para_float(v) for v in SINISTRO_RE.findall(texto)]
    qt_cias = len(SEGURADORA_RE.findall(texto))

    texto_normalizado = _sem_acento(texto).lower()
    # Conta apenas ocorrencias no campo "Tipo:", e nao qualquer aparicao solta
    # da palavra no texto livre.
    tipos_citados = [_sem_acento(m).lower() for m in TIPO_RE.findall(texto)]
    contagem_tipos = {t: tipos_citados.count(t) for t in TIPOS_SINISTRO_COD}
    maximo = max(contagem_tipos.values())
    if maximo == 0:
        tipo_cod = 0
    else:
        empatados = [t for t, c in contagem_tipos.items() if c == maximo]
        # Empate resolvido pelo PRIMEIRO tipo citado no texto, criterio
        # explicito e reproduzivel (antes vencia a ordem do dicionario).
        escolhido = next(t for t in tipos_citados if t in empatados)
        tipo_cod = TIPOS_SINISTRO_COD[escolhido]

    return {
        "valor_pago_historico": sum(valores_pagos),
        "valor_sinistro_historico": sum(valores_sinistro),
        "qt_cias_anteriores": qt_cias,
        "tipo_sinistro_predominante": tipo_cod,
    }


def _booleano_ou_nulo(valor):
    """Converte um campo booleano do formulario em 0/1, preservando o NULO.

    A distincao e material: um quesito NAO RESPONDIDO pela corretora nao
    significa "nao". Coagir o nulo a zero registraria ausencia de sinistro
    onde ha apenas ausencia de resposta, enviesando para baixo justamente as
    duas variaveis de sinistralidade declarada. Devolvendo None, o campo chega
    como NaN e e imputado pela moda em tratar_ausentes(), com o criterio
    explicito e auditavel.
    """
    if valor is None:
        return None
    return int(bool(valor))


def extrair_variaveis(registro: dict) -> dict:
    """Calcula as 19 variaveis derivadas (Quadro 2) a partir de um registro
    bruto de cotacao (objeto JSON + sinistralidade declarada)."""

    proprios = _num0(_get(registro, "frota", "proprios"))
    autonomos = _num0(_get(registro, "frota", "autonomos"))
    terceiros = _num0(_get(registro, "frota", "terceiros"))
    total_veiculos = proprios + autonomos + terceiros
    pct_autonomos = autonomos / total_veiculos if total_veiculos > 0 else None

    premio_liquido = _num0(_get(registro, "valores", "premioLiquido"))
    danos_materiais = _num0(_get(registro, "valores", "danosMateriais"))
    lmi_por_veiculo = danos_materiais / total_veiculos if total_veiculos > 0 else None
    premio_por_veiculo = premio_liquido / total_veiculos if total_veiculos > 0 else None

    coberturas = registro.get("coberturas") or {}
    qt_coberturas_ativas = sum(1 for c in COBERTURAS_OPCIONAIS if coberturas.get(c))

    qst3_dados = parse_qst3(_get(registro, "sinistralidade", "qst3"))

    dt_criacao = _get(registro, "produto", "dtCriacao")
    dt_emissao = _get(registro, "produto", "dtEmissao")
    tempo_cotacao_emissao = None
    if dt_criacao and dt_emissao:
        tempo_cotacao_emissao = (pd.Timestamp(dt_emissao) - pd.Timestamp(dt_criacao)).days

    variaveis = {
        "numeroApolice": registro.get("numeroApolice"),
        "total_veiculos": total_veiculos,
        "pct_autonomos": pct_autonomos,
        "motorista_licenciado": _booleano_ou_nulo(
            _get(registro, "frota", "motoristaLicenciado")),
        "classe_risco": _get(registro, "atividade", "classeRisco"),
        "sinistralidade_rc_declarada": _booleano_ou_nulo(
            _get(registro, "sinistralidade", "qst1")),
        "sinistralidade_rcfv_declarada": _booleano_ou_nulo(
            _get(registro, "sinistralidade", "qst2")),
        "lmi_por_veiculo": lmi_por_veiculo,
        "premio_por_veiculo": premio_por_veiculo,
        "qt_coberturas_ativas": qt_coberturas_ativas,
        "agravo_aplicado": int(_num0(_get(registro, "valores", "valorAgravo")) > 0),
        "desconto_aplicado": int(_num0(_get(registro, "valores", "valorDesconto")) > 0),
        "referral_pendente": int(len(registro.get("referral") or "") > 0),
        "tempo_cotacao_emissao": tempo_cotacao_emissao,
        "parcelas_com_juros": int(_num0(_get(registro, "valores", "vlJuros")) > 0),
        "apolice_anterior": int(_get(registro, "dadosApolice", "apoliceAnterior") is not None),
    }
    variaveis.update(qst3_dados)

    if "_perfil_gerador" in registro:
        variaveis["_perfil_gerador"] = registro["_perfil_gerador"]

    return variaveis


def _coagir_numerico(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in COLUNAS_19:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def tratar_ausentes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in COLUNAS_QST3:
        df[col] = df[col].fillna(0)

    for col in COLUNAS_BINARIAS:
        if df[col].isna().any():
            moda = df[col].mode(dropna=True)
            df[col] = df[col].fillna(moda.iloc[0] if not moda.empty else 0)

    for col in COLUNAS_CONTINUAS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def remover_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset="numeroApolice", keep="first").reset_index(drop=True)


def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara a matriz que alimenta o K-Means.

    Tres etapas, nesta ordem: log1p nas variaveis de cauda longa, expansao da
    variavel nominal em indicadoras e Min-Max em tudo. As colunas originais do
    Quadro 2 sao mantidas no resultado (para inspecao), mas quem entra no
    modelo e `COLUNAS_MODELO`.
    """
    df_norm = df.copy()

    for col in COLUNAS_LOG:
        df_norm[col] = np.log1p(df_norm[col].clip(lower=0))

    for nome, codigo in zip(INDICADORAS_SINISTRO, (1, 2, 3)):
        df_norm[nome] = (df[COLUNA_NOMINAL] == codigo).astype(int)

    scaler = MinMaxScaler()
    df_norm[COLUNAS_MODELO] = scaler.fit_transform(df_norm[COLUNAS_MODELO])
    return df_norm


def processar(caminho_entrada: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa o pipeline completo da Etapa 1 e retorna (df_original,
    df_normalizado), ambos indexados por numeroApolice."""

    with open(caminho_entrada, encoding="utf-8") as f:
        registros = json.load(f)

    df = pd.DataFrame(extrair_variaveis(r) for r in registros)
    df = _coagir_numerico(df)
    # Duplicados saem ANTES da imputacao: medianas e modas calculadas com
    # registros repetidos ficam enviesadas em direcao ao que foi duplicado.
    df = remover_duplicados(df)
    df = tratar_ausentes(df)

    colunas_finais = ["numeroApolice"] + COLUNAS_19
    if "_perfil_gerador" in df.columns:
        colunas_finais.append("_perfil_gerador")
    df = df[colunas_finais]

    df_normalizado = normalizar(df)
    df_normalizado = df_normalizado[colunas_finais + INDICADORAS_SINISTRO]

    return df, df_normalizado


def main():
    parser = argparse.ArgumentParser(
        description="Pre-processa a base de cotacoes RCT (Etapa 1 da metodologia)."
    )
    parser.add_argument("--in", dest="entrada", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    raiz = Path(__file__).resolve().parent.parent
    entrada = Path(args.entrada) if args.entrada else raiz / "data" / "raw" / "cotacoes_sinteticas.json"
    saida_dir = Path(args.out_dir) if args.out_dir else raiz / "data" / "processed"
    saida_dir.mkdir(parents=True, exist_ok=True)

    df_original, df_normalizado = processar(entrada)

    caminho_original = saida_dir / "matriz_original.csv"
    caminho_normalizada = saida_dir / "matriz_normalizada.csv"
    df_original.to_csv(caminho_original, index=False)
    df_normalizado.to_csv(caminho_normalizada, index=False)

    print(f"{len(df_original)} registros processados (apos remocao de duplicados).")
    print(f"Matriz original: {caminho_original}")
    print(f"Matriz normalizada: {caminho_normalizada}")


if __name__ == "__main__":
    main()
