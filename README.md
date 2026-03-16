# NCA / 中文视频话语风格漂移项目

本项目围绕“生成式 AI 扩散背景下的中文视频话语风格漂移”展开，当前仓库主要保存数据采集、字幕预处理、特征工程、量化建模与结果导出的工程代码。  
现阶段的正式量化分析已经从早期 pilot / breakpoint-centered 逻辑，重构为以创作者为单位的纵向面板分析。

## 当前正式口径

- 唯一正式主样本：`2412` 条视频 / `87` 位创作者
- 平衡稳健性样本：`1260` 条视频 / `42` 位创作者
- 当前正式样本来源：`ASR only`
- 旧的 `75 / 5` pilot 设计已废止，不能再作为正式分析口径
- 旧的 `990 / 81` 仅作为历史写作快照保留，不再作为正式样本层

当前正式结论基调也已经固定：

- `modest`
- `selective`
- `within-creator`
- `heterogeneous`
- `gradual drift`

这意味着目前最稳妥的量化表述是：存在有限、选择性的创作者内风格漂移，但没有足够证据支持“尖锐断点”或“平台整体突变”的强说法。

## 当前权威 run

当前正式分析以以下 run 为唯一权威基线：

- `run_id = 20260131_234759`
- 入口脚本：`precollection/scripts/10_formal_analysis_rebuild.py`

本轮正式重跑的总览文件是：

- `precollection/runs/20260131_234759/outputs/formal_analysis_report.md`

它已经替代旧的 `analysis_report.md` 作为当前正式量化结果入口。

## 目前工程完成到哪里

当前工程已经可以稳定完成以下流程：

1. 创作者解析与视频索引
2. 三阶段抽样与 manifest 构建
3. 字幕 / ASR 下载与解析
4. 基于 BCC timing chunks 的分段修复
5. 特征重建与主/次/附录三级指标划分
6. 创作者固定效应面板模型
7. random-slope mixed model 异质性分析
8. 断点稳健性检查
9. balanced / extreme-length / metadata-controlled 稳健性重跑
10. blind-review-safe 图表与表格导出

换句话说，目前项目已经不缺“可运行的工程主链”，当前重点更多在于如何基于现有正式口径继续优化分析设计和论文表达。

## 仓库结构

- `inputs/`：论文与课程作业相关外部材料
- `precollection/config/`：断点与运行配置
- `precollection/resources/`：停用词、功能词、模板词、连接词等资源表
- `precollection/data/`：索引、中间数据、原始字幕等数据目录
- `precollection/scripts/`：采集、清洗、分析和导出脚本
- `precollection/runs/`：每轮正式运行的归档目录

关键脚本包括：

- `precollection/scripts/01a_build_video_index.py`
- `precollection/scripts/01b_select_samples.py`
- `precollection/scripts/02_download_subtitles.py`
- `precollection/scripts/03_parse_subtitles.py`
- `precollection/scripts/04_clean_text.py`
- `precollection/scripts/07_v1_analysis.py`
- `precollection/scripts/09_full_analysis.py`
- `precollection/scripts/10_formal_analysis_rebuild.py`

其中，当前正式结果应优先以 `10_formal_analysis_rebuild.py` 的输出为准。

## 当前正式分析的核心结果

### 样本与预处理

- 正式主样本为 `2412 / 87`
- 平衡样本为 `1260 / 42`
- 三阶段完整覆盖创作者为 `76`
- 正式样本全部来自 ASR
- 结构指标已不再依赖原始 ASR 标点断句，而是使用修复后的 segment 逻辑

### 指标系统

主指标只保留四类：

- `connective_density`
- `template_density`
- `mattr`
- `structure_composite`

次级指标包括：

- `heading_like_density`
- `definitional_marker_density`
- `inferential_marker_density`
- `liability_shield_density`
- `segment_count_per_1000_tokens`
- `func_ratio`
- `stop_ratio`
- `mean_word_len`
- `style_index_pc1`

附录或 drop 的指标包括：

- `avg_tokens_per_segment`
- 原始标点相关结构指标
- `formulaic_transition_density`
- `connectives_total`

其中 `connectives_total` 已明确从 headline 结果中剔除，所有主 count-like 指标都已做长度修正。

### 当前最稳妥的量化解释

- 主面板模型没有出现 FDR 显著的 headline 主效应
- mixed model 仍显示明显创作者异质性
- 断点分析弱于 panel-drift 证据
- metadata-controlled 版本与主样本结果保持一致
- 整体更支持 gradual drift，而不是 sharp break

## 关键输出文件

建议优先查看以下结果文件：

- `precollection/runs/20260131_234759/outputs/formal_analysis_report.md`
- `precollection/runs/20260131_234759/outputs/sample_flow.md`
- `precollection/runs/20260131_234759/outputs/preprocessing_log.md`
- `precollection/runs/20260131_234759/outputs/feature_spec.md`
- `precollection/runs/20260131_234759/outputs/main_results_table.csv`
- `precollection/runs/20260131_234759/outputs/heterogeneity_results.csv`
- `precollection/runs/20260131_234759/outputs/metadata_controlled_results.csv`
- `precollection/runs/20260131_234759/outputs/robustness_summary.md`
- `precollection/runs/20260131_234759/outputs/results_for_abstract.md`
- `precollection/runs/20260131_234759/outputs/old_vs_new_spec.md`

## Blind Review 导出

当前仓库已经提供两套输出模式：

- internal working outputs
- blind review outputs

blind-review-safe 导出位于：

- `precollection/runs/20260131_234759/outputs/blind_review_figure_set/`
- `precollection/runs/20260131_234759/outputs/blind_review_tables/`
- `precollection/runs/20260131_234759/outputs/blind_review_export_readme.md`

blind review 版本会：

- 用 `Creator_001` 这类中性 ID 替代创作者标签
- 去掉 figure 中不必要的创作者名称
- 保留正式样本口径与量化结论
- 避免暴露本地路径、用户名和机器信息

## 如何重跑当前正式分析

在 `precollection/` 目录下执行：

```powershell
python .\scripts\10_formal_analysis_rebuild.py --run-id 20260131_234759
```

这会基于当前正式 manifest 和正式 pipeline，重新生成：

- 样本流与口径文件
- 预处理 / segmentation QC
- 特征与相关矩阵
- 面板模型与 mixed model 结果
- 断点稳健性结果
- balanced / extreme-length / metadata-controlled 稳健性结果
- blind-review-safe 表格和图表

## 当前项目状态一句话总结

项目已经完成从数据采集到正式量化分析的工程闭环，当前正式结果支持“有限、选择性、创作者内、异质性的 gradual drift”，而不支持把项目表述为一个强断点、强因果或平台整体突变设计。
