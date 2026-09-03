#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fix Decoder ReLU → Sigmoid Issue

问题：
    decoder.py 中存在 ReLU → Sigmoid 结构，导致输出范围 [0.5, 1)

修复：
    移除最后的 ReLU，改为 Identity

使用方法:
    python scripts/fix_decoder_relu.py
"""

import sys
from pathlib import Path

def fix_decoder():
    """修复 decoder.py"""
    decoder_file = Path('src/models/haze_density/decoder.py')

    if not decoder_file.exists():
        print(f"Error: {decoder_file} not found")
        return False

    print(f"Reading {decoder_file}...")
    content = decoder_file.read_text(encoding='utf-8')

    # 备份
    backup_file = Path('src/models/haze_density/decoder.py.bak')
    backup_file.write_text(content, encoding='utf-8')
    print(f"Backup saved: {backup_file}")

    # 修复 1: 移除 relu3 的 ReLU 定义
    old_relu3_def = 'self.relu3 = nn.ReLU(inplace=True)'
    new_relu3_def = 'self.relu3 = nn.Identity()  # Fixed: removed ReLU before Sigmoid'

    if old_relu3_def in content:
        content = content.replace(old_relu3_def, new_relu3_def)
        print("Fixed: ReLU definition changed to Identity")
    else:
        print("Warning: ReLU definition not found")

    # 修复 2: 移除 relu3 的调用（或者保留但改为 Identity）
    # 由于已经改为 Identity，调用可以保留

    # 写入修复后的文件
    decoder_file.write_text(content, encoding='utf-8')
    print(f"\nFixed {decoder_file}")

    # 显示修复后的相关代码
    print("\nFixed code section:")
    print("-" * 60)
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'self.relu3' in line or 'self.conv2' in line or 'self.norm3' in line:
            # 显示上下文
            start = max(0, i - 1)
            end = min(len(lines), i + 2)
            for j in range(start, end):
                marker = " >>> " if j == i else "     "
                print(f"{marker}{j+1:4d}: {lines[j]}")
            print()

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Fix Decoder ReLU → Sigmoid Issue")
    print("=" * 60)

    success = fix_decoder()

    if success:
        print("\n[OK] Fix applied successfully")
        print("\nNext steps:")
        print("1. Verify the fix by checking decoder.py")
        print("2. Re-run 5-epoch smoke training")
        print("3. Check if prediction range is now [0, 1]")
    else:
        print("\n[FAIL] Fix failed")
        sys.exit(1)
