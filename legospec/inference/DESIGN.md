# inference

generate.py 共用 block/hidden-feedback 生成循环及请求状态；verify.py 使用标准 p/q 接受和正 p-q 残差。EOS/budget censoring 单独记账。采样数学复用 Proposal，模型不导入训练。

当前文件：`generate.py`, `verify.py`

修改要求：共享组件不反向导入具体算法；输入在边界校验；不为单次转发增加 wrapper。数学与缓存语义变化必须有前向、梯度或状态回归测试。
