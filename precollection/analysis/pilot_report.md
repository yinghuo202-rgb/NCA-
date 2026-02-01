# 预收集报告（pilot）

## 分层核验（expected vs actual）

| series | creator_id | bvid | pub_date | duration_sec | page_duration | expected_stage | actual_stage | stage_match | needs_review | subtitle_status | subtitle_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fanshi_newanime | Creator_F | BV1QY4y1V7RY | 2022-06-02 | 670 | 670 | S0 | S0 | True | False | OK | asr_faster_whisper |
| fanshi_newanime | Creator_F | BV1rW4y18743 | 2022-08-27 | 740 | 740 | S0 | S0 | True | False | OK | asr_faster_whisper |
| fanshi_newanime | Creator_F | BV1Py4y1Z7p6 | 2023-02-25 | 860 | 860 | S1 | S1 | True | False | OK | asr_faster_whisper |
| fanshi_newanime | Creator_F | BV17z4y1M76T | 2023-08-28 | 965 | 965 | S1 | S1 | True | False | OK | asr_faster_whisper |
| fanshi_newanime | Creator_F | BV1Yi421v7uE | 2024-06-02 | 1085 | 1085 | S2 | S2 | True | False | OK | asr_faster_whisper |
| fanshi_newanime | Creator_F | BV1yUp4eeEQ2 | 2024-09-07 | 863 | 863 | S2 | S2 | True | False | OK | asr_faster_whisper |
| shuianxiaoxi | Creator_M | BV15v411T7ym | 2021-08-15 | 1075 | 1075 | S0 | S0 | True | False | OK | asr_faster_whisper |
| shuianxiaoxi | Creator_M | BV19W4y1h78S | 2022-08-14 | 867 | 867 | S0 | S0 | True | False | OK | asr_faster_whisper |
| shuianxiaoxi | Creator_M | BV1ED4y1n7MD | 2023-01-22 | 4814 | 4814 | S1 | S1 | True | False | OK | asr_faster_whisper |
| shuianxiaoxi | Creator_M | BV1Tm4y1m7ty | 2023-04-09 | 1599 | 1599 | S1 | S1 | True | False | OK | asr_faster_whisper |
| shuianxiaoxi | Creator_M | BV1Qb4y1M7TB | 2023-11-19 | 1186 | 1186 | S2 | S2 | True | False | OK | asr_faster_whisper |
| shuianxiaoxi | Creator_M | BV1tWvDBkEbb | 2026-01-01 | 2067 | 2067 | S2 | S2 | True | False | OK | asr_faster_whisper |


## 字幕下载状态汇总

| status | count |
| --- | --- |
| OK | 12 |


按 series 细分：

| series | subtitle_status | count |
| --- | --- | --- |
| fanshi_newanime | OK | 6 |
| shuianxiaoxi | OK | 6 |


下载日志（download.jsonl）状态统计：

| status | count |
| --- | --- |
| OK | 12 |


## 字幕来源统计（web_cc / ai_player）

- 策略：优先网页 CC（`x/player/wbi/v2`）；AI 字幕（`x/player/v2`）存在串台/错配风险，默认不启用；为满足“12 条 BV 全部有字幕且一一对应”，提供本地 ASR（faster-whisper）兜底（`scripts/02_download_subtitles.py --asr-mode fallback/force`）。

| subtitle_source | count |
| --- | --- |
| asr_faster_whisper | 12 |


## 指标均值（series × stage）

| series | actual_stage | chars | tokens | sentences | mean_sentence_chars | punct_per_1k_tokens | connectors_per_1k_tokens | frame_markers_per_1k_tokens | modality_per_1k_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fanshi_newanime | S0 | 3671.0 | 3642.5 | 6.0 | 1125.3 | 50.014 | 2.881 | 0.0 | 1.384 |
| fanshi_newanime | S1 | 4627.0 | 4549.5 | 6.5 | 713.416 | 44.004 | 4.699 | 0.0 | 1.531 |
| fanshi_newanime | S2 | 4588.0 | 4528.0 | 3.5 | 2991.75 | 28.763 | 6.099 | 0.0 | 2.802 |
| shuianxiaoxi | S0 | 5485.0 | 5472.0 | 1.0 | 5485.0 | 0.338 | 10.186 | 0.0 | 4.563 |
| shuianxiaoxi | S1 | 19044.0 | 18893.5 | 1.5 | 16562.5 | 0.38 | 10.168 | 0.087 | 6.874 |
| shuianxiaoxi | S2 | 9439.5 | 9384.5 | 85.0 | 5865.275 | 42.753 | 11.101 | 0.07 | 7.008 |


## 可视化

![download_status](figures/download_status.png)

![connectors_by_stage](figures/connectors_by_stage.png)


## 风险提示（预收集必须记录）

- 字幕可得率：12/12（缺失：0）

- ASR 风险：本地语音识别可能引入分段/错字偏差，正式研究需在报告中明确标注 subtitle_source，并做敏感性分析。

- subtitle_type 分布：

| subtitle_type | count |
| --- | --- |
| asr | 12 |

- 分层不匹配（needs_review=true）：0/12（本版本不替换样本，仅标记）

- 长视频长度效应：正式研究建议考虑固定窗口（前 N 分钟/前 N 字）或加入长度控制变量。

- 字幕类型偏差：若自动字幕/上传字幕比例随时间变化，可能造成“语言漂移”假象；需在正式研究中控制或分层。
