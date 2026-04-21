#!/bin/bash

# 使用临时文件避免数据丢失
INPUT_FILE="cross_freq.txt"
TMP_FILE="cross_freq_tmp.txt"

awk '
/iter:/ {
    # 查找 "iter:" 后的数值
    for(i=1; i<=NF; i++) {
        if($i == "iter:") {
            iter = $(i+1);
            gsub(/,/, "", iter);
            break
        }
    }
}
/Validation CECVal/ {
    # $8 对应 psnr 的数值 (27.1362)
    # $11 对应 ssim 的数值 (0.9562)
    psnr = $8
    ssim = $11
    # 格式化输出
    printf "iter: %-6s # psnr: %-8s # ssim: %-8s\n", iter, psnr, ssim
}
' "$INPUT_FILE" > "$TMP_FILE"

# 检查临时文件是否有内容，防止误删
if [ -s "$TMP_FILE" ]; then
    mv "$TMP_FILE" "$INPUT_FILE"
    echo -e "\n✅ 处理完成！已保存至 $INPUT_FILE"
else
    echo -e "\n❌ 处理失败：未提取到有效数据，原始文件已保留。"
    rm "$TMP_FILE"
fi
