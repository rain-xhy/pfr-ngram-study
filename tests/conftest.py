"""让测试能用 `from src import data` 导入项目代码。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
