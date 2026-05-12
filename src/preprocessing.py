from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "fundamentos_matematicas",
    "algebra_lineal",
    "calculo_diferencial",
    "calculo_integral",
]

PROGRAMMING_COLS = [
    "taller_programacion",
    "desarrollo_software_1",
    "desarrollo_software_2",
]

ID_COL = "id_estudiante"
TARGET_COL = "riesgo_programacion"
AVG_PROGRAMMING_COL = "promedio_programacion"


@dataclass
class PreprocessResult:
    data: pd.DataFrame
    dropped_rows: int
    clipped_cells: int


def validate_required_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    required = [ID_COL, *FEATURE_COLS, *PROGRAMMING_COLS]
    missing = [col for col in required if col not in df.columns]
    return (len(missing) == 0, missing)


def add_target_from_programming_average(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    work_df = df.copy()
    work_df[AVG_PROGRAMMING_COL] = work_df[PROGRAMMING_COLS].mean(axis=1)
    work_df[TARGET_COL] = (work_df[AVG_PROGRAMMING_COL] < threshold).astype(int)
    return work_df


def clean_dataset(df: pd.DataFrame) -> PreprocessResult:
    work_df = df.copy()
    numeric_cols = [*FEATURE_COLS, *PROGRAMMING_COLS]

    for col in numeric_cols:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    before = len(work_df)
    work_df = work_df.dropna(subset=numeric_cols)
    dropped_rows = before - len(work_df)

    clipped_cells = 0
    for col in numeric_cols:
        original = work_df[col].copy()
        work_df[col] = work_df[col].clip(0.0, 5.0)
        clipped_cells += int((original != work_df[col]).sum())

    return PreprocessResult(data=work_df, dropped_rows=dropped_rows, clipped_cells=clipped_cells)


def pearson_interpretation(value: float) -> str:
    abs_value = abs(value)
    if abs_value < 0.2:
        return "Muy debil"
    if abs_value < 0.4:
        return "Debil"
    if abs_value < 0.6:
        return "Moderada"
    if abs_value < 0.8:
        return "Fuerte"
    return "Muy fuerte"


def build_demo_dataset(seed: int = 42, n: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    fundamentos = np.clip(rng.normal(3.0, 0.9, n), 0, 5)
    algebra = np.clip(rng.normal(3.1, 0.8, n), 0, 5)
    diferencial = np.clip(rng.normal(2.9, 0.9, n), 0, 5)
    integral = np.clip(rng.normal(2.8, 1.0, n), 0, 5)

    base = 0.25 * fundamentos + 0.25 * algebra + 0.25 * diferencial + 0.25 * integral
    noise_1 = rng.normal(0.0, 0.45, n)
    noise_2 = rng.normal(0.0, 0.50, n)
    noise_3 = rng.normal(0.0, 0.55, n)

    taller = np.clip(base + noise_1, 0, 5)
    ds1 = np.clip(base + noise_2, 0, 5)
    ds2 = np.clip(base + noise_3, 0, 5)

    return pd.DataFrame(
        {
            ID_COL: np.arange(1, n + 1),
            "fundamentos_matematicas": np.round(fundamentos, 2),
            "algebra_lineal": np.round(algebra, 2),
            "calculo_diferencial": np.round(diferencial, 2),
            "calculo_integral": np.round(integral, 2),
            "taller_programacion": np.round(taller, 2),
            "desarrollo_software_1": np.round(ds1, 2),
            "desarrollo_software_2": np.round(ds2, 2),
        }
    )
