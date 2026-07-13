import re

with open('src/hal/cpu.rs', 'r') as f:
    content = f.read()

# Pattern 1: gemv
pattern1 = re.compile(
    r'#\[cfg\(target_arch = "x86_64"\)\]\s*unsafe\s*\{\s*(crate::compute::avx2::process_row_avx2_gemv[^;]*;)\s*\}\s*'
    r'#\[cfg\(target_arch = "aarch64"\)\]\s*unsafe\s*\{\s*(crate::compute::neon::process_row_neon_gemv[^;]*;)\s*\}(?:\s*// coverage:ignore-line)?\s*'
    r'#\[cfg\(not\(any\(target_arch = "x86_64", target_arch = "aarch64"\)\)\)\]\s*unsafe\s*\{\s*(crate::compute::scalar::process_row_scalar_gemv[^;]*;)\s*\}'
)

# Pattern 2: gemm
pattern2 = re.compile(
    r'#\[cfg\(target_arch = "x86_64"\)\]\s*unsafe\s*\{\s*(crate::compute::avx2::process_row_avx2_gemm[^;]*;?)\s*\}\s*;?\s*'
    r'#\[cfg\(target_arch = "aarch64"\)\]\s*unsafe\s*\{\s*(crate::compute::neon::process_row_neon_gemm[^;]*;?)\s*\}\s*;?\s*'
    r'#\[cfg\(not\(any\(target_arch = "x86_64", target_arch = "aarch64"\)\)\)\]\s*unsafe\s*\{\s*(crate::compute::scalar::process_row_scalar_gemm[^;]*;?)\s*\}\s*;?'
)

def repl1(m):
    return (f'unsafe {{\n    cfg_select! {{\n'
            f'        target_arch = "x86_64" => {m.group(1).rstrip(";")},\n'
            f'        target_arch = "aarch64" => {m.group(2).rstrip(";")},\n'
            f'        _ => {m.group(3).rstrip(";")},\n'
            f'    }}\n}}')

def repl2(m):
    return (f'unsafe {{\n    cfg_select! {{\n'
            f'        target_arch = "x86_64" => {m.group(1).rstrip(";")},\n'
            f'        target_arch = "aarch64" => {m.group(2).rstrip(";")},\n'
            f'        _ => {m.group(3).rstrip(";")},\n'
            f'    }}\n}}')

content = pattern1.sub(repl1, content)
content = pattern2.sub(repl2, content)

with open('src/hal/cpu.rs', 'w') as f:
    f.write(content)
