from pathlib import Path
from PIL import Image

# 输入路径
base = Path("/Users/yueyiwang/refusal-robustness-scaling/figs")

# 三张图的对应关系：源文件 -> 目标文件
mapping = {
    "fig_rrr_vs_steps.png": "fig_baseline_rrr.pdf",
    "fig_rd_vs_steps.png":  "fig_rrr_collapse.pdf",
    "fig_ce_vs_steps.png":  "fig_ce_before_after.pdf",
}

for src_name, dst_name in mapping.items():
    src = base / src_name
    dst = base / dst_name
    img = Image.open(src)
    # 转 PDF，保持分辨率
    img.save(dst, "PDF", resolution=300.0)
    print(f"Saved {dst}")
