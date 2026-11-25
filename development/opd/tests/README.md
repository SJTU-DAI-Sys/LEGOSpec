# 暂缓 OPD 的测试快照

test_anchor_opd.py 为移出的专属测试；其他文件为包含 OPD 用例的混合测试原件。
普通 E2E-TV 数学、因果 attention、target feature、packed training 和
hidden-feedback 测试仍保留在 tests/unit。此处不加入默认测试发现。

快照引用移出前的正式接口，只用于未来恢复开发时对照，不宣称当前可运行。
对应代码及集成快照在 development/opd/。
