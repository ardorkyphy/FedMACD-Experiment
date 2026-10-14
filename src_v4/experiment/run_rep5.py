# -*- coding: utf-8 -*-
"""FedMACD 论文实验入口：10 数据集 × 2 场景 × 3 个划分种子。

用法:
  python src_v4/experiment/run_rep5.py                      # 完整队列
  python src_v4/experiment/run_rep5.py --dataset asia      # 只跑 asia（6 折）
  python src_v4/experiment/run_rep5.py --smoke             # 开发队列: 每个数据集/场景仅 seed0（20 折）
采样口径：forward_sample(N, seed=DATA.sampling_seed)；分割 DATA.seed=k 独立。
结果目录可由调用方配置；正式复现实验建议使用 scripts/run_v436_main.py。
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_V4_ROOT = os.path.join(PROJECT_ROOT, "src_v4")
for _p in (PROJECT_ROOT, SRC_V4_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run import build_config, main  # noqa: E402

# (数据集, 全量样本量)：BIF=bnlearn 官方行数；外部=官方文件全量行数
DATASETS = [
    ("child", 10000), ("alarm", 10000),
    ("earthquake", 10000), ("survey", 10000), ("asia", 10000),
    ("10nodes", 5000), ("ER20", 256),
    ("Auto_MPG", 392), ("DWD_climate", 349), ("Sachs", 7466),
]
EXTERNAL_DATA = {
    "10nodes": {"format": "csv", "discrete": False},
    "ER20": {"format": "csv", "discrete": False},
    "Auto_MPG": {"format": "csv", "discrete": False},
    "DWD_climate": {"format": "csv", "discrete": False},
    "Sachs": {"format": "csv", "discrete": False},
}
SCENARIOS = [("homogeneous", ""), ("heterogeneous", "heterogeneous")]
SEEDS = [0, 1, 2]
NUM_CLIENTS = 4
STRENGTH = 0.7

# 各数据集变量语义（v4 语义通道：客户端 LLM 存在性/方向性分析 + 服务端辩论/裁判都依赖它）。
# 缺省数据集（如外部连续数据）留空 → 语义仲裁降级为纯统计。
DATASET_SEMANTICS = {
    "asia": {
        "asia": "是否近期去过亚洲（旅行史，结核病暴露风险）",
        "tub": "是否患有肺结核",
        "smoke": "是否吸烟",
        "lung": "是否患有肺癌",
        "bronc": "是否患有支气管炎",
        "either": "是否患有肺结核或肺癌中的至少一种",
        "xray": "胸部 X 光检查结果是否异常",
        "dysp": "是否出现呼吸困难",
    },
    "child": {
        "BirthAsphyxia": "是否出生窒息",
        "HypDistrib": "缺氧分布（均匀/不均匀）",
        "HypoxiaInO2": "缺氧程度（轻度/中度/重度）",
        "CO2": "血二氧化碳水平（高/正常/低）",
        "ChestXray": "胸部 X 光结果（正常/少血/充血/毛玻璃/不对称斑片）",
        "Grunting": "是否出现呻吟（呼吸窘迫体征）",
        "LVHreport": "左心室肥大报告（是/否）",
        "LowerBodyO2": "下半身血氧水平",
        "RUQO2": "右上腹血氧水平",
        "CO2Report": "二氧化碳监测报告",
        "XrayReport": "X 光报告（正常/少血/充血/毛玻璃/不对称斑片）",
        "Disease": "新生儿疾病诊断",
        "GruntingReport": "呻吟观察报告（有/无）",
        "Age": "患儿日龄分组",
        "LVH": "左心室是否肥大",
        "DuctFlow": "动脉导管血流方向",
        "CardiacMixing": "心脏血液混合程度",
        "LungParench": "肺实质状态",
        "LungFlow": "肺血流（正常/减少/增多）",
        "Sick": "患儿是否病重（是/否）",
    },
    "alarm": {
        "HISTORY": "是否记录有心肌梗死病史",
        "CVP": "中心静脉压",
        "PCWP": "肺毛细血管楔压",
        "HYPOVOLEMIA": "是否低血容量",
        "LVEDVOLUME": "左室舒张末期容积",
        "LVFAILURE": "是否左心衰竭",
        "STROKEVOLUME": "每搏输出量",
        "ERRLOWOUTPUT": "心输出量过低错误报警",
        "HRBP": "血压监测仪测得的心率",
        "HREKG": "心电图测得的心率",
        "ERRCAUTER": "电刀干扰错误",
        "HRSAT": "血氧监测仪测得的心率",
        "INSUFFANESTH": "是否麻醉不足",
        "ANAPHYLAXIS": "是否过敏反应",
        "TPR": "总外周阻力",
        "EXPCO2": "呼气末二氧化碳",
        "KINKEDTUBE": "气管导管是否折曲",
        "MINVOL": "分钟通气量",
        "FIO2": "吸入氧浓度",
        "PVSAT": "脉搏血氧饱和度（吸入氧浓度与肺泡通气量的监测结果）",
        "SAO2": "动脉血氧饱和度",
        "PAP": "肺动脉压",
        "PULMEMBOLUS": "是否肺栓塞",
        "SHUNT": "肺内分流程度（正常/高）",
        "INTUBATION": "气管插管状态（正常/误插入食管/单侧插管，属机械状态而非治疗决策）",
        "PRESS": "气道压",
        "DISCONNECT": "管路是否断开",
        "MINVOLSET": "设定分钟通气量",
        "VENTMACH": "呼吸机工作状态（零/低/正常/高）",
        "VENTTUBE": "呼吸机管路通气状态（零/低/正常/高）",
        "VENTLUNG": "肺顺应性状态（零/低/正常/高）",
        "VENTALV": "肺泡通气量",
        "ARTCO2": "动脉血二氧化碳",
        "CATECHOL": "儿茶酚胺水平（正常/高）",
        "HR": "心率",
        "CO": "心输出量",
        "BP": "血压",
    },
    "earthquake": {
        "Burglary": "是否发生入室盗窃",
        "Earthquake": "是否发生地震",
        "Alarm": "家庭警报是否响起",
        "JohnCalls": "约翰是否打电话",
        "MaryCalls": "玛丽是否打电话",
    },
    "survey": {
        "A": "年龄段",
        "S": "性别",
        "E": "受教育程度",
        "O": "职业类型",
        "R": "居住地区域",
        "T": "日常出行交通方式",
    },
    # DWD and Sachs have readable domain variables. The synthetic 10nodes and
    # ER20 variables are anonymous X0..X19 identifiers and therefore remain empty.
    "DWD_climate": {
        "Altitude": "气象站的海拔高度（米）",
        "Temperature": "年均气温（摄氏度）",
        "Precipitation": "年降水量（毫米）",
        "Longitude": "气象站的经度（东经度数）",
        "Sunshine": "年日照时数（小时）",
        "Latitude": "气象站的纬度（北纬度数）",
    },
    "Sachs": {
        "raf": "Raf 蛋白激酶的活性水平（MAPK 信号通路成员）",
        "mek": "MEK 蛋白激酶的活性水平（MAPK 信号通路成员）",
        "plc": "PLCγ 磷脂酶 C 的活性水平",
        "pip2": "磷脂酰肌醇二磷酸 PIP2 的水平（细胞膜磷脂）",
        "pip3": "磷脂酰肌醇三磷酸 PIP3 的水平（细胞膜磷脂）",
        "erk": "ERK 蛋白激酶的活性水平（MAPK 信号通路成员）",
        "akt": "Akt 蛋白激酶的活性水平",
        "pka": "PKA 蛋白激酶 A 的活性水平",
        "pkc": "PKC 蛋白激酶 C 的活性水平",
        "p38": "p38 蛋白激酶的活性水平（MAPK 家族成员）",
        "jnk": "JNK 蛋白激酶的活性水平（MAPK 家族成员）",
    },
    "Auto_MPG": {
        "Displacement": "发动机排量（立方英寸）",
        "Mpg": "每加仑行驶英里数（燃油经济性指标）",
        "Horsepower": "发动机最大马力（输出功率）",
        "Weight": "整车重量（磅）",
        "Acceleration": "0-60mph 加速时间（秒）",
    },
}


def already_done(out_root: str, name: str, split: str, seed: int, sample_size: int,
                 num_clients: int = NUM_CLIENTS) -> bool:
    """判重必须含 scenario（v4.19 修：同质先跑会把异质同 seed 折误判完成跳过）。"""
    csv_path = os.path.join(out_root, "results_v4.csv")
    if not os.path.exists(csv_path):
        return False
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return False
        scenario = split or "homogeneous"
        mask = ((df["dataset"] == name) & (df["scenario"] == scenario)
                & (df["seed"] == seed)
                & (df.get("num_clients", 0) == num_clients)
                & (df.get("sample_size", 0) == sample_size))
        return bool(mask.any())
    except Exception as exc:
        print(f"  [warn] already_done 读取失败（视为未完成，可重跑）: {exc}")
        return False


def run_one(name: str, split: str, seed: int, out_root: str, strength: float,
            sample_size: int, num_clients: int = NUM_CLIENTS) -> dict:
    cfg = build_config()
    cfg["DATA"].update(name=name, sample_size=sample_size, num_clients=num_clients,
                       seed=seed, split=split,
                       semantics=DATASET_SEMANTICS.get(name, {}))
    if name in EXTERNAL_DATA:
        cfg["DATA"].update(EXTERNAL_DATA[name])
    if split == "heterogeneous":
        cfg["DATA"]["heterogeneity_strength"] = strength
        cfg["HETEROGENEITY_STRENGTH"] = strength
    cfg["OUTPUT_ROOT"] = out_root
    print(f"\n===== 开始实验: {name} split={split or 'homogeneous'} seed={seed} "
          f"n={sample_size} clients={num_clients} =====")
    return main(cfg)


def main_loop(smoke: bool = False, only_scenario: str = "", only_dataset: str = "") -> None:
    out_root = os.path.join(SRC_V4_ROOT, "experiment", "results", "v4")
    os.makedirs(out_root, exist_ok=True)
    count = 0
    for scenario, split in SCENARIOS:
        if only_scenario and scenario != only_scenario:
            continue
        for name, sample_size in DATASETS:
            if only_dataset and name != only_dataset:
                continue
            for seed in (SEEDS if not smoke else [0]):
                if already_done(out_root, name, split, seed, sample_size):
                    print(f"  跳过（已完成）: {name} {scenario} seed={seed}")
                    continue
                run_one(name, split, seed, out_root, STRENGTH, sample_size)
                count += 1
    print(f"\n===== 队列完成：新跑 {count} 折 =====")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="开发队列：20折，不是单折烟测")
    ap.add_argument("--dataset", default="", help="只跑指定数据集")
    ap.add_argument("--scenario", default="", choices=["", "homogeneous", "heterogeneous"])
    args = ap.parse_args()
    main_loop(smoke=args.smoke, only_scenario=args.scenario, only_dataset=args.dataset)
