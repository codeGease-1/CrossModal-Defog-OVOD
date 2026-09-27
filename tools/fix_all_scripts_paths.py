#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量修复 scripts 目录下所有 Python 脚本的路径设置问题

问题：原脚本使用 Path(__file__).parent.parent 但位于 scripts/train/ 等子目录下
修复：根据脚本所在深度动态计算正确的 project_root

作者：遥感智研助手
日期：2026-09-27
"""

import re
from pathlib import Path

def fix_script_path(script_path: Path) -> bool:
    """修复单个脚本的路径设置"""

    content = script_path.read_text(encoding='utf-8')

    # 检查是否需要修复（已经修复的文件跳过）
    if "parent.parent.parent" in content:
        return False

    # 检查是否有路径设置代码
    if "Path(__file__)" not in content and "sys.path" not in content:
        return False

    # 检查目录深度
    parts = script_path.relative_to(script_path.parent.parent.parent).parts
    depth = len(parts)  # scripts/train/ = 2, scripts/ = 1

    # 计算需要向上几层才能到达 src 的父目录
    # scripts/ -> parent.parent = 项目根 (2 层)
    # scripts/train/ -> parent.parent.parent = 项目根 (3 层)
    # scripts/utils/ -> parent.parent.parent = 项目根 (3 层)

    parent_count = len(script_path.parts) - len(script_path.parent.parent.parent.parts) + 2
    if depth == 1:  # scripts/ 直接子目录
        parent_needed = "parent.parent"
    else:  # scripts/子目录/
        parent_needed = "parent.parent.parent"

    # Pattern to match the path setup code
    old_pattern = r"project_root = Path\(__file__\)\.parent\.parent(?:\.parent)?"
    new_code = f"project_root = Path(__file__}.{parent_needed}  # {script_path.relative_to(script_path.parent.parent.parent)}"

    new_content = re.sub(old_pattern, new_code, content)

    if new_content != content:
        script_path.write_text(new_content, encoding='utf-8')
        return True
    return False


def main():
    scripts_dir = Path(__file__).parent.parent / "scripts"

    if not scripts_dir.exists():
        print(f"[ERROR] scripts directory not found: {scripts_dir}")
        return

    fixed = []
    skipped = []
    no_change = []

    for script_file in scripts_dir.rglob("*.py"):
        rel_path = script_file.relative_to(scripts_dir)
        try:
            if fix_script_path(script_file):
                fixed.append(rel_path)
                print(f"[FIX] {rel_path}")
            else:
                no_change.append(rel_path)
        except Exception as e:
            skipped.append((rel_path, str(e)))
            print(f"[SKIP] {rel_path}: {e}")

    print("\n" + "="*50)
    print("Summary:")
    print(f"  Fixed: {len(fixed)}")
    print(f"  No change needed: {len(no_change)}")
    print(f"  Skipped (errors): {len(skipped)}")
    print("="*50)

    if fixed:
        print("\nFixed files:")
        for f in fixed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()