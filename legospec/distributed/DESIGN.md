# distributed

groups 建立角色与通信组；collectives 实现集合通信；transport 拥有目标特征、训练 packet schema 与有界传输。不包含独立算法 packet。

当前文件：`collectives.py`, `groups.py`, `transport.py`

修改要求：共享组件不反向导入具体算法；输入在边界校验；不为单次转发增加 wrapper。数学与缓存语义变化必须有前向、梯度或状态回归测试。
