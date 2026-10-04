"""engine —— 通用「懂你引擎」。

铁律:本包**绝不** import aria.scenes 下任何东西。
引擎只认 interfaces/ 的协议;场景由外部(app / 测试)注入。
tests/test_pluggability.py 会机械校验这条边界。
"""
