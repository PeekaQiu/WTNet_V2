import os
import re
import json
import random
import math

# 配置文件名映射
FILES = {
    'Baseline': 'cec_baseline.txt',
    'ECM': 'cec_ecm.txt',
    'Cross Frequency': 'cec_cross_frequency.txt'
}


def generate_mock_data():
    """如果文件不存在，生成模拟数据以便演示"""
    print("未检测到全部数据文件，正在生成符合特征的模拟数据...")
    for name, filename in FILES.items():
        with open(filename, 'w', encoding='utf-8') as f:
            # 模拟：初期陡增，后期平缓，且各场景差距很小
            offset = {'Baseline': 0, 'ECM': 0.15, 'Cross Frequency': 0.35}[name]
            for i in range(1, 101):
                iter_val = i * 10
                # 用对数函数模拟陡增后平缓的趋势
                psnr = 15 + 5 * math.log10(iter_val) + offset + random.uniform(-0.02, 0.02)
                ssim = 0.7 + 0.1 * math.log10(iter_val) + (offset * 0.05) + random.uniform(-0.002, 0.002)

                # 保证ssim不超过1
                ssim = min(0.9999, ssim)
                f.write(f"iter: {iter_val:<6} # psnr: {psnr:.4f}  # ssim: {ssim:.4f}\n")


def parse_data():
    """解析文本文件数据"""
    pattern = re.compile(r"iter:\s*(\d+)\s*#\s*psnr:\s*([\d.]+)\s*#\s*ssim:\s*([\d.]+)")

    psnr_data = {}
    ssim_data = {}

    for label, filename in FILES.items():
        if not os.path.exists(filename):
            generate_mock_data()

        psnr_list = []
        ssim_list = []

        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    iter_val = int(match.group(1))
                    psnr = float(match.group(2))
                    ssim = float(match.group(3))

                    # Echarts 需要的二维数组格式 [x, y]
                    psnr_list.append([iter_val, psnr])
                    ssim_list.append([iter_val, ssim])

        psnr_data[label] = psnr_list
        ssim_data[label] = ssim_list

    return psnr_data, ssim_data


def generate_html(psnr_data, ssim_data, output_file='visualization.html'):
    """生成前卫美观的HTML文件"""

    # 转换为JSON字符串供JS使用
    psnr_json = json.dumps(psnr_data)
    ssim_json = json.dumps(ssim_data)

    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Performance Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
    <style>
        :root {{
            --bg-gradient: linear-gradient(135deg, #0b0f19 0%, #1a2235 100%);
            --card-bg: rgba(26, 34, 53, 0.6);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-main: #e2e8f0;
        }}

        body, html {{
            margin: 0;
            padding: 0;
            width: 100%;
            min-height: 100vh;
            background: var(--bg-gradient);
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            color: var(--text-main);
            display: flex;
            flex-direction: column;
            align-items: center;
        }}

        .header {{
            text-align: center;
            margin: 40px 0 20px 0;
            text-transform: uppercase;
            letter-spacing: 3px;
        }}

        .header h1 {{
            font-size: 28px;
            font-weight: 300;
            margin: 0;
            background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .header p {{
            font-size: 12px;
            color: #8b9bb4;
            margin-top: 8px;
        }}

        .container {{
            width: 90%;
            max-width: 1400px;
            display: flex;
            flex-direction: column;
            gap: 40px;
            padding-bottom: 50px;
        }}

        .chart-card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }}

        .chart {{
            width: 100%;
            height: 500px;
        }}
    </style>
</head>
<body>

    <div class="header">
        <h1>Performance Metrics Analysis</h1>
        <p>Dynamic Scale • Scroll to Zoom • Hover for Details</p>
    </div>

    <div class="container">
        <div class="chart-card">
            <div id="psnrChart" class="chart"></div>
        </div>
        <div class="chart-card">
            <div id="ssimChart" class="chart"></div>
        </div>
    </div>

    <script>
        const psnrData = {psnr_json};
        const ssimData = {ssim_json};

        // 霓虹配色方案
        const colors = ['#00f2fe', '#f093fb', '#fce38a'];
        const seriesNames = ['Baseline', 'ECM', 'Cross Frequency'];

        // 生成配置项的工厂函数
        function getChartOption(title, yAxisName, dataDict) {{
            const seriesList = seriesNames.map((name, index) => ({{
                name: name,
                type: 'line',
                smooth: true,
                symbol: 'circle',
                symbolSize: 6,
                showSymbol: false,
                data: dataDict[name],
                lineStyle: {{
                    width: 3,
                    shadowColor: colors[index],
                    shadowBlur: 10,
                    shadowOffsetY: 2
                }},
                itemStyle: {{ color: colors[index] }},
                areaStyle: {{
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        {{ offset: 0, color: colors[index] + '40' }}, // 25% opacity
                        {{ offset: 1, color: colors[index] + '00' }}  // 0% opacity
                    ])
                }}
            }}));

            return {{
                title: {{
                    text: title,
                    left: '20',
                    top: '10',
                    textStyle: {{ color: '#fff', fontSize: 18, fontWeight: '400', letterSpacing: 1 }}
                }},
                tooltip: {{
                    trigger: 'axis',
                    axisPointer: {{
                        type: 'cross',
                        label: {{ backgroundColor: '#1a2235' }},
                        lineStyle: {{ color: 'rgba(255, 255, 255, 0.2)', type: 'dashed' }}
                    }},
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    textStyle: {{ color: '#e2e8f0' }},
                    padding: 15,
                    borderRadius: 8,
                    formatter: function (params) {{
                        let result = `<div style="font-weight:bold;margin-bottom:8px;border-bottom:1px solid rgba(255,255,255,0.1);padding-bottom:5px;">Iter: ${{params[0].value[0]}}</div>`;
                        // 按数值降序排序提示框内容，方便直接看出谁高谁低
                        params.sort((a, b) => b.value[1] - a.value[1]);
                        params.forEach(param => {{
                            result += `<div style="display:flex;justify-content:space-between;align-items:center;margin:4px 0;">
                                        <span>${{param.marker}} ${{param.seriesName}}</span>
                                        <span style="font-weight:bold;margin-left:20px;color:${{param.color}}">${{param.value[1].toFixed(4)}}</span>
                                       </div>`;
                        }});
                        return result;
                    }}
                }},
                legend: {{
                    data: seriesNames,
                    top: '15',
                    right: '30',
                    textStyle: {{ color: '#a0aec0' }},
                    icon: 'roundRect'
                }},
                grid: {{
                    left: '3%',
                    right: '4%',
                    bottom: '15%', // 为缩放条留出空间
                    top: '15%',
                    containLabel: true
                }},
                // X轴使用 value，支持非均匀间隔数据，配合 DataZoom 完美解决初期密集问题
                xAxis: {{
                    type: 'value',
                    name: 'Iteration',
                    nameTextStyle: {{ color: '#a0aec0', padding: [0, 0, 0, 10] }},
                    splitLine: {{ show: false }},
                    axisLabel: {{ color: '#a0aec0' }},
                    axisLine: {{ lineStyle: {{ color: 'rgba(255,255,255,0.1)' }} }},
                    min: 'dataMin'
                }},
                // Y轴动态缩放 (scale: true)，脱离0基准线，极大放大细微差异
                yAxis: {{
                    type: 'value',
                    name: yAxisName,
                    scale: true, 
                    nameTextStyle: {{ color: '#a0aec0', padding: [0, 0, 0, 0] }},
                    splitLine: {{ lineStyle: {{ color: 'rgba(255,255,255,0.05)', type: 'dashed' }} }},
                    axisLabel: {{ color: '#a0aec0' }}
                }},
                dataZoom: [
                    {{
                        type: 'inside', // 支持鼠标滚轮缩放和平移
                        start: 0,
                        end: 100
                    }},
                    {{
                        type: 'slider', // 底部拖拽缩放条
                        start: 0,
                        end: 100,
                        height: 20,
                        bottom: 10,
                        borderColor: 'rgba(255,255,255,0.1)',
                        textStyle: {{ color: '#a0aec0' }},
                        fillerColor: 'rgba(0, 242, 254, 0.2)',
                        handleStyle: {{ color: '#00f2fe' }}
                    }}
                ],
                series: seriesList
            }};
        }}

        // 初始化图表
        const psnrChart = echarts.init(document.getElementById('psnrChart'));
        const ssimChart = echarts.init(document.getElementById('ssimChart'));

        psnrChart.setOption(getChartOption('PSNR Over Iterations', 'PSNR (dB)', psnrData));
        ssimChart.setOption(getChartOption('SSIM Over Iterations', 'SSIM', ssimData));

        // 窗口大小变化时自适应
        window.addEventListener('resize', () => {{
            psnrChart.resize();
            ssimChart.resize();
        }});
    </script>
</body>
</html>
    """

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    print(f"✅ 生成成功！请在浏览器中打开: {os.path.abspath(output_file)}")


if __name__ == '__main__':
    p_data, s_data = parse_data()
    generate_html(p_data, s_data)
