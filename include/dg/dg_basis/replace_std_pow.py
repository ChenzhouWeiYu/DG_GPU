#!/usr/bin/env python3
import re
import sys
import os

def replace_pow_with_ipow(content: str) -> str:
    # 匹配 std::pow( expr , N )，其中 N 是整数（允许空格）
    # 注意：C++ 中可能有嵌套括号，但简单场景下假设 expr 不含逗号（如 x, y+z, (x-1) 等可接受）
    # 更健壮的做法需解析 AST，但此处用正则+简单启发式足够
    pattern = r'std::pow\s*\(\s*([^,()]+(?:\([^)]*\))?)\s*,\s*(\d+)\s*\)'

    def replacer(match):
        expr = match.group(1).strip()
        exp = match.group(2)
        return f'ipow<{exp}>({expr})'

    # 反复替换直到无变化（处理嵌套？其实一般不需要）
    new_content, count = re.subn(pattern, replacer, content)
    return new_content

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 replace_std_pow.py <input_file.cpp>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.isfile(input_file):
        print(f"Error: File {input_file} not found.")
        sys.exit(1)

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = replace_pow_with_ipow(content)

    # 写回文件（或输出到 stdout）
    with open(input_file, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"Successfully replaced std::pow with ipow in {input_file}")

if __name__ == '__main__':
    main()