"""
雾密度网络.ipynb 更新脚本

功能:
1. 批量替换路径（适配新的 scripts 目录结构）
2. 标记可删除的 cell（在 metadata 中添加 deletion_hint）
3. 输出更新报告

作者：遥感智研助手
日期：2026-09-27
"""

import json
import re
from typing import Tuple, List

def update_notebook(input_path: str, output_path: str) -> dict:
    """更新 notebook 路径并标记可删除的 cell"""

    with open(input_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 路径替换映射表 (旧路径 -> 新路径)
    path_replacements = {
        "scripts/train_haze_density.py": "scripts/train/train_haze_density.py",
        "scripts/train_overfit_8.py": "scripts/train/train_overfit_8.py",
        "scripts/evaluate_haze_density.py": "scripts/evaluate/evaluate_haze_density.py",
        "scripts/audit_prediction_distribution.py": "scripts/evaluate/audit_prediction_distribution.py",
        "scripts/test_physical_prior.py": "scripts/test/test_physical_prior.py",
        "scripts/visualize_physical_prior.py": "scripts/data/visualize_physical_prior.py",
        "scripts/test_core_modules.py": "scripts/test/test_core_modules.py",
        "scripts/inspect_dataset.py": "scripts/data/inspect_dataset.py",
        "scripts/test_model.py": "scripts/test/test_model.py",
        "scripts/generate_rshazeplus_split.py": "scripts/data/generate_rshazeplus_split.py",
        "scripts/verify_rshazeplus_split.py": "scripts/data/verify_rshazeplus_split.py",
        "scripts/visualize_rshazeplus_dataset.py": "scripts/data/visualize_rshazeplus_dataset.py",
        "scripts/test_rshazeplus_physical_prior.py": "scripts/test/test_rshazeplus_physical_prior.py",
        "scripts/visualize_physical_prior_on_rshazeplus.py": "scripts/data/visualize_physical_prior_on_rshazeplus.py",
        "scripts/smoke_test.py": "scripts/utils/smoke_test.py",
        "scripts/fix_decoder_relu.py": "scripts/utils/fix_decoder_relu.py",
        "scripts/test_density_concat_integration.py": "scripts/test/test_density_concat_integration.py",
        "scripts/test_density_guidance.py": "scripts/test/test_density_guidance.py",
        "scripts/test_stage_6_3b.py": "scripts/test/test_stage_6_3b.py",
        "scripts/test_rshazeplus_dataset.py": "scripts/test/test_rshazeplus_dataset.py"
    }

    # 可删除 cell 的关键词匹配
    deletion_patterns = {
        "gradscaler_warning": [
            r"FutureWarning.*GradScaler",
            r"torch\.cuda\.amp\.GradScaler.*deprecated"
        ],
        "autocast_warning": [
            r"FutureWarning.*autocast",
            r"torch\.cuda\.amp\.autocast.*deprecated"
        ],
        "fix_decoder_relu": [
            r"fix_decoder_relu\.py",
            r"Decoder.*RELU.*fix"
        ],
        "old_test_model": [
            r"scripts/test_model\.py[^/]"
        ],
        "smoke_test_old": [
            r"scripts/smoke_test\.py[^/]"
        ],
        "git_checkout revert": [
            r"git checkout -- scripts/"
        ]
    }

    stats = {
        "total_cells": len(nb['cells']),
        "path_replacements_made": 0,
        "cells_marked_for_deletion": 0,
        "deletion_reasons": {},
        "cells_to_delete": []
    }

    for idx, cell in enumerate(nb['cells']):
        source = cell.get('source', '')
        if isinstance(source, list):
            source_text = ''.join(source)
        else:
            source_text = source

        cell_metadata = cell.get('metadata', {})
        deletion_hints = []

        # 1. 路径替换
        original_source = source
        if isinstance(source, list):
            new_source = []
            for line in source:
                old_line = line
                for old_path, new_path in path_replacements.items():
                    if old_path in line:
                        line = line.replace(old_path, new_path)
                        stats['path_replacements_made'] += 1
                new_source.append(line)
            cell['source'] = new_source
            source_text = ''.join(new_source)  # 更新用于后续检查
        elif isinstance(source, str):
            new_source = source
            for old_path, new_path in path_replacements.items():
                if old_path in source:
                    source = source.replace(old_path, new_path)
                    stats['path_replacements_made'] += 1
            cell['source'] = source
            source_text = source

        # 2. 检测可删除的 cell
        for reason, patterns in deletion_patterns.items():
            for pattern in patterns:
                if re.search(pattern, source_text, re.IGNORECASE):
                    deletion_hints.append(reason)
                    if reason not in stats['deletion_reasons']:
                        stats['deletion_reasons'][reason] = []
                    stats['deletion_reasons'][reason].append(idx)
                    break  # 一个 cell 一个原因只计数一次

        if deletion_hints:
            cell_metadata['deletion_hint'] = {
                'recommended': True,
                'reasons': deletion_hints,
                'note': 'This cell contains outdated debugging info or already-fixed issues.'
            }
            cell['metadata'] = cell_metadata
            stats['cells_marked_for_deletion'] += 1
            stats['cells_to_delete'].append({
                'index': idx,
                'reasons': deletion_hints,
                'preview': source_text[:100].replace('\n', ' ')
            })

    # 保存更新后的 notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    return stats

def print_report(stats: dict):
    """打印更新报告"""
    print("\n" + "="*60)
    print("Notebook Update Report")
    print("="*60)
    print(f"Total cells: {stats['total_cells']}")
    print(f"Path replacements made: {stats['path_replacements_made']}")
    print(f"Cells marked for deletion: {stats['cells_marked_for_deletion']}")

    if stats['deletion_reasons']:
        print("\nDeletable cell categories:")
        print("-"*40)
        for reason, cell_indices in stats['deletion_reasons'].items():
            reason_desc = {
                "gradscaler_warning": "GradScaler deprecation warnings",
                "autocast_warning": "autocast deprecation warnings",
                "fix_decoder_relu": "Decoder RELU fix (completed)",
                "old_test_model": "Old test_model.py calls",
                "smoke_test_old": "Old smoke_test.py calls",
                "git_checkout revert": "git checkout revert operations"
            }.get(reason, reason)
            print(f"  {reason_desc}: {len(cell_indices)} cells")
            if cell_indices:
                print(f"    Location: cell #{cell_indices}")

    if stats['cells_to_delete']:
        print("\nCells recommended for deletion:")
        print("-"*40)
        for item in stats['cells_to_delete'][:10]:
            print(f"  Cell #{item['index']}: {', '.join(item['reasons'])}")
            print(f"    Preview: {item['preview']}...")

    print("\n" + "="*60)

def delete_marked_cells(nb: dict) -> int:
    """删除所有带有 deletion_hint metadata 的 cell"""
    original_len = len(nb['cells'])
    nb['cells'] = [cell for cell in nb['cells']
                   if 'deletion_hint' not in cell.get('metadata', {})]
    return original_len - len(nb['cells'])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Update notebook paths')
    parser.add_argument('--input', '-i',
                       default=r"C:\Users\35017\AppData\Roaming\CherryStudio\Data\Files\01a0e1f6-30d3-70a0-a97d-ca57d7a10291.ipynb",
                       help='Input notebook path')
    parser.add_argument('--output', '-o',
                       default=r"D:\Desktop\project\CrossModal-Defog-OVOD\docs\雾密度网络_更新后.ipynb",
                       help='Output notebook path')
    parser.add_argument('--delete-marked', '-d', action='store_true',
                       help='Auto-delete cells marked for deletion')

    args = parser.parse_args()

    try:
        stats = update_notebook(args.input, args.output)
        print_report(stats)

        if args.delete_marked:
            # Read the updated notebook
            with open(args.output, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            deleted_count = delete_marked_cells(nb)
            # Save again
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(nb, f, ensure_ascii=False, indent=1)
            print(f"\n[OK] Deleted {deleted_count} marked cells.")

        print(f"\n[OK] Notebook updated: {args.output}")
    except FileNotFoundError:
        print(f"[ERROR] File not found: {args.input}")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()