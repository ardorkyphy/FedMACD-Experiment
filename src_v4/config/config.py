# -*- coding: utf-8 -*-
"""v4 全局运行配置（精简版：只留关键项与流程必需项）。

Configuration stays compact:
只写流程需要、会直接影响实验行为/结果的超参；其余默认值内聚在代码。
"""
import os
from typing import Dict, List

# ================= 版本 =================
# Artifact version prefix used by results, logs, intermediate outputs, and baseline data.
# 统一引用本常量作为前缀，版本间产物互相隔离、不污染；改版本号即新一版口径。
VERSION: str = "v4_36"

# ================= 路径 =================
PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_ROOT: str = os.path.join(PROJECT_ROOT, "src_v4", "experiment", "results")
LOGS_ROOT: str = os.path.join(RESULTS_ROOT, "logs")
OUTPUT_ROOT: str = os.path.join(RESULTS_ROOT, "v4")

# ================= LLM (role-specific overrides supported) =================
# 机制：每个 LLM 调用都带一个 role，配置解析 = default 为基底 + roles[role] 覆盖同名键
# （LLM._resolve(role)，可覆盖 model/base_url/api_key_env/temperature/max_tokens）。
# Empty role overrides use the default provider and temperature.
# 到时候要不同温度/异构，在对应角色里写覆盖字段即可，例如：
#   "debater_a": {"temperature": 0.3},             # 正方辩手：稍高温度（观点多样性）
#   "debater_b": {"temperature": 0.3},             # 反方辩手
#   "judge": {"model": "<更强推理模型>", "temperature": 0.0},   # 裁判要确定性
#   "client_existence": {"model": "<另一端点/模型>", "temperature": 0.1},  # 客户端异构
LLM: Dict = {
    "default": {
        "provider": "openai",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "api_key_env": "DEEPSEEK_API_KEY",
        "temperature": 0.0,
        "max_tokens": 10240,
    },
    # Select a registered provider with FEDMACD_PROVIDER; DeepSeek is the default.
    "active_provider": os.environ.get("FEDMACD_PROVIDER", "deepseek"),
    "providers": {
        "deepseek": {
            "provider": "openai",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
            "api_key_env": "DEEPSEEK_API_KEY",
            "temperature": 0.0,
            "max_tokens": 204800,
        },
        "qwen": {
            "provider": "openai",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen3.8-flash",
            "api_key_env": "QWEN_API_KEY",
            "temperature": 0.0,
            "max_tokens": 10240,
        },
        "zhipu": {
            "provider": "openai",
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "model": "glm-5.3-flash",
            "api_key_env": "ZHIPU_API_KEY",
            "temperature": 0.0,
            "max_tokens": 10240,
            "reasoning_effort": "high",  # GLM requires enabled reasoning; high balances cost and depth.
            # glm-5.3-flash: thinking.type 仅支持 enabled（强制思考、默认值），传 disabled 会报错；
            # 思维链在独立 reasoning_content 字段返回，content 不受污染，故不发 thinking 字段（=默认 enabled）。
        },
    },
    "roles": {
        "select": {},             # 客户端：本地 LLM 选算法
        "client_existence": {},   # 客户端：逐边存在合理性分析（答案+分数+一句话理由）
        "client_direction": {},   # 客户端：逐边方向性分析（答案+分数+一句话理由）
        "debater_a": {},          # 服务端：存在性/方向辩论 正方（领证据包）
        "debater_b": {},          # 服务端：辩论 反方
        "judge": {},              # 服务端：裁判（存在性/方向共用角色，模板分设）
    },
}

LLM_CALL: Dict = {
    "max_retries": 3,
    "timeout_seconds": 300,
    "cache_enabled": True,        # 响应缓存（内存；截断/坏 JSON 不入缓存）
    "log_calls": True,
    "prompt_log_dir": "",         # Full prompt/response JSONL directory; empty disables logging.
}

# Client-side edge analysis switch used by the component ablation: False keeps only statistical evidence,
# 跳过存在性/方向性 LLM 逐边分析（含 absence_supplement 补采的 LLM 部分）；不影响
# 辩论/裁判等其它 LLM 通道。默认 True（与主线口径一致）。
CLIENT_ANALYZE: Dict = {"enabled": True}

# Statistical-evidence input switch: False removes values and field descriptions only at rendering time,
# 统计数值与统计字段解释从 LLM 输入移除（保留骨架/支持率/算法方向/语义/辩论与裁判结构提示），
# 统计计算与内部裁决不变；默认 True 与主线一致。
STATISTICAL_EVIDENCE: Dict = {"enabled": True}

# ================= 协议 =================
# 两阶段阶段清单（配置化：新增阶段 = 注册 stage + 在此加名字）
PHASE1_STAGES: List[str] = [
    "align", "local_discovery", "existence_evidence", "upload",
    "aggregate", "absence_supplement", "debate", "judge",
]
PHASE2_STAGES: List[str] = [
    "share_candidate", "direction_evidence", "upload",
    "aggregate", "debate", "judge", "assemble",
]
ALIGN_MODE: str = "identity"      # Implemented variable alignment mode: identity.
DEBATE_MAX_ROUNDS: int = 2        # 服务端辩论轮数上限（辩手 A/B 交替）

# ================= 统计与裁决 =================
ALPHA: float = 0.05               # 独立性检验显著性（统计层存在性/方向性）

MAX_COND: int = 2                 # 条件独立性检验共同邻居数上限

DEBATE_BATCH_EDGES: int = 20      # 辩手批大小（一批多条边一次 LLM 调用，防注意力稀释；v4.28 分批辩论）

# ================= 算法 =================
ALGORITHM_SELECTION: str = "llm"  # 客户端选算法：llm（默认，LLM 单次推理）/ manual
ALGORITHM_OVERRIDE: str = ""      # 实验强制指定算法名（manual 用）
ALGORITHM_PARAMS: Dict = {"fedecd_local": {"keep_ratio": 0.5}}  # 对齐 v3：多数投票语义（v3 同款配置，并集语义会注入假边）

# ================= 数据 =================
DATA: Dict = {
    "name": "asia",
    "format": "",                 # bif / csv / npy（空=按扩展名推断）
    "discrete": True,
    "sample_size": 1000,          # 总样本量（0=全量哨兵）
    "num_clients": 4,
    "split": "",                  # 同质=随机均分 / heterogeneous=按数据类型构造分布漂移
    "seed": 0,                    # 分割种子
    "sampling_seed": 0,           # BIF 前向采样种子（0=各折共用同一子集）
    "variables": [],              # 变量子集（空=全量）
    "graph_path": "",             # 真图路径（相对 PROJECT_ROOT；BIF 自带则留空）
    "semantics": {                # {变量: 一句话描述}——v4 语义通道（必须填！）
        # 默认数据集 asia（医学示例：肺结核诊断）
        "asia": "是否近期去过亚洲（旅行史，结核病暴露风险）",
        "tub": "是否患有肺结核",
        "smoke": "是否吸烟",
        "lung": "是否患有肺癌",
        "bronc": "是否患有支气管炎",
        "either": "是否患有肺结核或肺癌中的至少一种",
        "xray": "胸部 X 光检查结果是否异常",
        "dysp": "是否出现呼吸困难",
    },
}
HETEROGENEITY_STRENGTH: float = 0.7   # 异质构造强度（split=heterogeneous 时用）

# ================= 执行与输出 =================
EXECUTION: Dict = {"parallel": True, "max_workers": 4}
SAVE_ARTIFACTS: bool = True       # 每折审计落盘
RESULT_FORMAT: str = "csv"
LOG_LEVEL: str = "INFO"
SEED: int = 0

