# 已移除的独立 block 分支测试

test_gdn_block_attention.py 保存迁移后的旧分支测试参考。它使用可选 block attention、
独立 Q/K/V/O 及无 block 的旧配方；这些结构已从当前主项目删除，不进入默认回归。
原始迁移前测试完整保存在 tests/draft_boundaries_20261123/before.tgz 中。

当前 intrinsic GDN/KDA 使用共享 Q/K/V/O 和逐头 gate。
其数值、状态、缓存、提交隔离和导出由 tests/unit/test_gdn_shared_block_attention.py、
test_attention_taxonomy.py 以及 test_gdn.py 覆盖。
