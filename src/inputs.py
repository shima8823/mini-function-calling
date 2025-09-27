from __future__ import annotations

from pathlib import Path
from typing import Dict, List


from pydantic import BaseModel, Field
import json


class FunctionDefinition(BaseModel):
    """関数定義のデータモデル"""

    fn_name: str = Field(..., description="関数名")
    args_names: List[str] = Field(..., description="引数名のリスト")
    args_types: Dict[str, str] = Field(..., description="引数名→型")
    return_type: str = Field(..., description="戻り値の型")


class PromptItem(BaseModel):
    """プロンプト（1件）のデータモデル"""

    prompt: str = Field(..., description="自然言語の質問")


def load_function_definitions(path: str | Path) -> List[FunctionDefinition]:
    """functions_definition.json を読み込み、モデルへ変換して返す"""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [FunctionDefinition(**item) for item in data]


def load_prompts(path: str | Path) -> List[PromptItem]:
    """function_calling_tests.json を読み込み、モデルへ変換して返す"""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [PromptItem(**item) for item in data]
