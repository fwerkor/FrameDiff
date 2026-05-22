import json
import math
from datetime import datetime
from pathlib import Path
from queue import Queue
from random import randint

import yaml


def first_missing_positive(nums):
    n = len(nums)
    for i in range(n):
        while 0 <= nums[i] < n and nums[nums[i]] != nums[i]:
            nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
    for i in range(n):
        if nums[i] != i:
            return i
    return n


def probability(rate):
    if rate == 0:
        return False
    if rate == 1:
        return True
    if rate == 0.2:
        return randint(0, 4) == 0
    if rate == 0.5:
        return randint(0, 1) == 0
    if rate == 0.3:
        return randint(0, 9) <= 2
    sample = int(1 / rate)
    return randint(0, sample) == 1


def cal_nodes_nums(g):
    return len({hash(node) for node in g.nodes.values()})


def topo_sort(g):
    r"""
    judge whether graph g is dag, and return the topo seq of g
    :param g: graph
    :return: is_topo_sort, the seq of topo_sort
    """
    print("come into topo_sort")
    seq = []
    in_degree = {k: 0 for k, _ in g.nodes.items()}
    for node_ in g.nodes.values():
        for to_ in node_.to_nodes:
            in_degree[to_] += 1
    q = Queue()
    for key, degree in in_degree.items():
        if degree == 0:
            q.put(key)
    while not q.empty():
        x = q.get()
        seq.append(x)
        for target in g.nodes[x].to_nodes:
            in_degree[target] -= 1
            if in_degree[target] == 0:
                q.put(target)
    print("current topo queue is ")
    print(seq)
    return len(seq) == len(g.nodes), seq


def get_element_nums(shape):
    nums = 1
    for i in shape:
        nums *= i
    return nums


def get_element_det(shape1, shape2):
    r"""
    :param shape1:
    :param shape2:
    :return: the diff(det) between shape1 and shape2
    """
    return get_element_nums(shape1) - get_element_nums(shape2)


def get_random_node2(nums):
    r"""select two elements from nums:list"""
    a = nums[randint(0, len(nums) - 1)]
    b = nums[randint(0, len(nums) - 1)]
    while a == b:
        b = nums[randint(0, len(nums) - 1)]
    return a, b


def get_random_num(nums):
    r"""select one element from nums:list"""
    return nums[randint(0, len(nums) - 1)]


def print_seperate_line(op="=", length=60):
    print(op * length)


def gen_timestamp_h():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_dict_into_json(data: dict, path: str):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=4)


def compare_dicts(before, after):
    created = {}
    deleted = []
    modified = {}

    before_keys = set(before.keys())
    after_keys = set(after.keys())

    for key in after_keys - before_keys:
        created[key] = after[key]

    for key in before_keys - after_keys:
        deleted.append(key)

    for key in before_keys & after_keys:
        if before[key] != after[key]:
            modified[key] = {"from": before[key], "to": after[key]}

    return {"created": created, "deleted": deleted, "modified": modified}


def print_diff(diff):
    print("\nModified:")
    has_any = False
    if diff["modified"]:
        has_any = True
        for key, change in diff["modified"].items():
            print(f"  ~ {key}: {change['from']} -> {change['to']}")
    if diff["created"]:
        has_any = True
        for key, value in diff["created"].items():
            print(f"  ~ {key}: (created) -> {value}")
    if not has_any:
        print("  (no modified fields)")


def normalize_transformer_config_dict(config_dict: dict) -> dict:
    """Normalize transformer config fields after mutation."""
    if not config_dict:
        return config_dict

    hidden = int(config_dict.get("hidden_size", 0) or 0)
    heads = int(config_dict.get("num_attention_heads", 0) or 0)
    num_q = int(config_dict.get("num_query_groups", 0) or 0)
    tp = int(config_dict.get("tensor_model_parallel_size", 1) or 1)
    cp = int(config_dict.get("context_parallel_size", 1) or 1)

    if heads <= 0:
        heads = 1
        config_dict["num_attention_heads"] = heads
    if num_q <= 0:
        num_q = 1
        config_dict["num_query_groups"] = num_q
    if num_q > heads:
        config_dict["num_query_groups"] = heads
        num_q = heads

    if hidden > 0 and hidden % heads != 0:
        hidden = heads * math.ceil(hidden / heads)
        config_dict["hidden_size"] = hidden

    if heads % num_q != 0:
        new_groups = math.gcd(heads, num_q) or 1
        if new_groups != num_q:
            config_dict["num_query_groups"] = new_groups
            num_q = new_groups

    parallel_divisor = max(1, tp * cp)
    if heads % parallel_divisor != 0:
        heads = parallel_divisor * math.ceil(heads / parallel_divisor)
        config_dict["num_attention_heads"] = heads
        if num_q > heads:
            config_dict["num_query_groups"] = heads
            num_q = heads
        if heads % num_q != 0:
            new_groups = math.gcd(heads, num_q) or 1
            if new_groups != num_q:
                config_dict["num_query_groups"] = new_groups
                num_q = new_groups
        if hidden > 0 and hidden % heads != 0:
            hidden = heads * math.ceil(hidden / heads)
            config_dict["hidden_size"] = hidden

    if hidden > 0 and heads > 0:
        kv_channels = hidden // heads
        config_dict["kv_channels"] = kv_channels
        config_dict["hidden_size_per_attention_head"] = kv_channels

    return config_dict


def get_card_num(file_path: str) -> int:
    card_num = 8
    parallel_list = ["data_parallel", "model_parallel", "pipeline_stage"]
    try:
        with Path(file_path).open("r", encoding="utf-8") as handle:
            yamlfile = yaml.safe_load(handle)
        for key in parallel_list:
            if key not in yamlfile:
                continue
            if key == "data_parallel":
                value = yamlfile[key]
                if isinstance(value, str) and "&" in value:
                    value = value.split()[0]
                card_num *= int(value)
            else:
                card_num *= int(yamlfile[key])
    except Exception as exc:
        print(f"read yaml failed, using default card_num={card_num}, error: {exc}")
    return card_num
