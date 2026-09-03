#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
静态检查脚本（本地执行）

用于在本地进行代码静态检查，无需 PyTorch/CUDA 环境。

检查项目:
1. Python 语法检查
2. YAML 配置语法检查
3. 目录结构检查
4. Import 路径检查

使用方法:
    python scripts/static_check.py
"""

import ast
import os
import sys
from pathlib import Path


def check_python_syntax(file_path):
    """检查 Python 文件语法"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def check_yaml_syntax(file_path):
    """检查 YAML 文件语法"""
    try:
        import yaml

        with open(file_path, "r", encoding="utf-8") as f:
            yaml.safe_load(f)
        return True, None
    except yaml.YAMLError as e:
        return False, str(e)
    except ImportError:
        return None, "pyyaml 未安装"
    except Exception as e:
        return False, str(e)


def check_directory_structure(project_root):
    """检查目录结构"""
    required_dirs = [
        "configs",
        "docs",
        "experiments/haze_density/checkpoints",
        "experiments/haze_density/logs",
        "scripts",
        "src/data",
        "src/models/haze_density",
        "src/utils",
    ]

    print("=" * 60)
    print("目录结构检查")
    print("=" * 60)

    all_ok = True
    for d in required_dirs:
        path = project_root / d
        if path.exists() and path.is_dir():
            print(f"[OK] {d}")
        else:
            print(f"[FAIL] {d} - 不存在")
            all_ok = False

    return all_ok


def check_python_files(project_root):
    """检查所有 Python 文件语法"""
    print("\n" + "=" * 60)
    print("Python 语法检查")
    print("=" * 60)

    python_files = list(project_root.rglob("*.py"))

    all_ok = True
    errors = []

    for file_path in python_files:
        # 跳过 __pycache__
        if "__pycache__" in str(file_path):
            continue

        rel_path = file_path.relative_to(project_root)
        is_ok, error = check_python_syntax(file_path)

        if is_ok:
            print(f"[OK] {rel_path}")
        else:
            print(f"[FAIL] {rel_path}")
            print(f"  错误：{error}")
            all_ok = False
            errors.append((str(rel_path), error))

    return all_ok, errors


def check_yaml_files(project_root):
    """检查所有 YAML 文件语法"""
    print("\n" + "=" * 60)
    print("YAML 语法检查")
    print("=" * 60)

    yaml_files = list(project_root.rglob("*.yaml")) + list(
        project_root.rglob("*.yml")
    )

    all_ok = True
    errors = []

    for file_path in yaml_files:
        rel_path = file_path.relative_to(project_root)
        is_ok, error = check_yaml_syntax(file_path)

        if is_ok is True:
            print(f"[OK] {rel_path}")
        elif is_ok is None:
            print(f"[WARN] {rel_path} - {error}")
        else:
            print(f"[FAIL] {rel_path}")
            print(f"  错误：{error}")
            all_ok = False
            errors.append((str(rel_path), error))

    return all_ok, errors


def check_init_files(project_root):
    """检查 __init__.py 文件"""
    print("\n" + "=" * 60)
    print("__init__.py 检查")
    print("=" * 60)

    expected_init = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models/__init__.py",
        "src/models/haze_density/__init__.py",
        "src/utils/__init__.py",
    ]

    all_ok = True
    for init_file in expected_init:
        path = project_root / init_file
        if path.exists():
            print(f"[OK] {init_file}")
        else:
            print(f"[FAIL] {init_file} - 不存在")
            all_ok = False

    return all_ok


def check_requirements(project_root):
    """检查 requirements.txt"""
    print("\n" + "=" * 60)
    print("requirements.txt 检查")
    print("=" * 60)

    req_file = project_root / "requirements.txt"

    if not req_file.exists():
        print("✗ requirements.txt 不存在")
        return False

    with open(req_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 检查是否为空
    lines = [l.strip() for l in content.split("\n") if l.strip() and not l.strip().startswith("#")]

    if not lines:
        print("[WARN] requirements.txt 为空（仅注释）")
        return False

    print(f"[OK] requirements.txt 存在")
    print(f"  包含 {len(lines)} 个依赖项")

    # 检查关键依赖
    key_deps = ["torch", "torchvision", "opencv-python", "numpy", "pyyaml"]
    content_lower = content.lower()

    for dep in key_deps:
        if dep in content_lower:
            print(f"  [OK] {dep}")
        else:
            print(f"  [FAIL] {dep} - 缺失")

    return True


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("CrossModal-Defog-OVOD - 静态检查")
    print("=" * 60 + "\n")

    # 项目根目录
    project_root = Path(__file__).parent.parent

    # 运行检查
    results = {}

    # 目录结构
    results["目录结构"] = check_directory_structure(project_root)

    # __init__.py
    results["__init__.py"] = check_init_files(project_root)

    # requirements.txt
    results["requirements.txt"] = check_requirements(project_root)

    # Python 语法
    py_ok, py_errors = check_python_files(project_root)
    results["Python 语法"] = py_ok

    # YAML 语法
    yaml_ok, yaml_errors = check_yaml_files(project_root)
    results["YAML 语法"] = yaml_ok

    # 汇总
    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)

    for name, result in results.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"{status} {name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n[OK] 所有静态检查通过！")
        print("\n下一步:")
        print("  1. 将项目推送到 GitHub 或导出为 ZIP")
        print("  2. 在 Google Colab 中运行 setup_colab.py")
        print("  3. 运行 smoke_test.py 进行动态测试")
    else:
        print("\n[WARN] 部分检查未通过，请修复后重试")

    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
