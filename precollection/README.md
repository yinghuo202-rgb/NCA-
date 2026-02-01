# 生成式AI扩散与中文视频字幕“语言漂移”预收集（v1.0）

工程根目录：`precollection/`

## 快速开始

1.（可选但强烈建议）准备 B 站 cookies：将浏览器中的 cookies 导出为简单 JSON，保存到 `secrets/cookies.json`。

示例见：`secrets/cookies.example.json`

2. 运行全流程（00→06）：

```powershell
cd precollection
python scripts/run_all.py
```

单步运行也可以：

```powershell
python scripts/01_build_manifest.py
python scripts/02_download_subtitles.py
python scripts/03_parse_subtitles.py
python scripts/04_clean_text.py
python scripts/05_compute_features.py
python scripts/06_generate_report.py
```

## 输出位置（关键）

- `data/raw/meta/videos_manifest.csv`
- `data/raw/subtitles/{creator_id}/{bvid}.bcc.json`
- `data/processed/text/{creator_id}/raw_{bvid}.txt`
- `data/processed/text/{creator_id}/clean_{bvid}.txt`
- `data/processed/qc/{bvid}_qc.json`
- `analysis/features.csv`
- `analysis/pilot_report.md`
- `logs/download.jsonl`

## 说明

- 本版本不做 ASR；字幕缺失会标记为 `NO_SUBTITLE`。
- 若 `expected_stage` 与 `actual_stage` 不一致，只标记 `needs_review=true`，不自动替换样本。
- 字幕下载：默认仅下载网页端可见字幕（`x/player/wbi/v2`）；如需尝试 legacy AI 字幕可运行 `python scripts/02_download_subtitles.py --allow-player-v2`（可能串台/错配，需人工核验）。
