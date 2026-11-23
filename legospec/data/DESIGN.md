# Data

负责原始对话预处理、目标隐藏态离线采集，以及训练缓存的读写和审计。
本目录当前不再分子目录；模型执行交给 modeling/target，训练批次规划交给 training/data。

## 对话与训练数据

- template.py：聊天模板及注册，定义消息如何转换为模型输入。
- parse.py：通用、Harmony、Thinking 等对话解析。
- preprocessing.py：文本/视觉对话 tokenization、loss mask 和训练数据集构建。
- json_training.py：流式读取并投影 JSONL 字段，构建训练用 Dataset。
- utils.py：padding collator、视觉数据 collator、DP dataloader 和 Harmony 内容处理。
- __init__.py：对外导出数据接口。

## 隐藏态采集

- capture_io.py：采集 CLI、层号解析、记录整理、输入标识和原子 JSON 写入。
- capture_engine.py：构造采集任务，调用统一 target 入口；以可复用 pinned buffer 和异步 pipeline 完成 GPU→CPU 传输及写入调度。
- capture_metrics.py：采集耗时、吞吐、显存统计和缓存落盘操作。

采集器提出层号、上下文和批量容量需求；HF/SGLang 执行、模型缩放及 final norm 语义由 modeling/target 负责。

## Raw-tar 缓存

- hidden_tar_raw.py：单条样本的张量编码和解码。
- hidden_tar_stream.py：流式 tar 写入、SHA256、异步持久化提交和最终 manifest。
- hidden_cache.py：manifest/样本读取与检查、训练 Dataset、shard sampler 和 collator；需要重建 final norm 时调用模型 adapter。
- hidden_audit.py：缓存身份、shard 校验及审计报告。

缓存保存指定层的原始 decoder 输出，final norm 后的 teacher 特征单独表示。
读写统一使用当前 raw-tar 格式；采集、传输和存储不实现模型专属计算。
