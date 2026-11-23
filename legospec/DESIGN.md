# legospec

四轴组件位于 modeling/draft/composable；共享 training、data、distributed、inference、target。公开入口只导出 LegoSpecModel/LegoSpecConfig。

当前文件：`args.py`

修改要求：共享组件不反向导入具体算法；输入在边界校验；不为单次转发增加 wrapper。数学与缓存语义变化必须有前向、梯度或状态回归测试。
