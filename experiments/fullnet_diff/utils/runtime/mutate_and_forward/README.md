## RUN mutate_and_forward.py

| 参数                 | 类型      | 是否必须 | 默认值   | 说明                      |
| ------------------ | ------- | ---- | ----- | ----------------------- |
| `-c`, `--configs`  | `str`   | 是    | 无     | 指定配置文件夹的路径。             |
| `-n`, `--node-num` | `int`   | 否    | `2`   | 节点数量，用于控制生成模型结构中的节点个数。 |
| `-r`, `--round`    | `int`   | 否    | `10`  | 变异操作的轮数。       |
| `--rate`           | `float` | 否    | `0.3` | 每轮变异操作的变异率。             |
| `-m`, `--module`   | `str`   | 否    | 无     | 目标模块名称，仅对指定单个模块进行操作时使用，避免随机选择。   |
| `-nm`, `--no-mutating`   |  `bool`  | 否    | False     | 是否禁用变异   |


Run single module mutating
```python
python new_mutate_and_forward.py -c ../../../../../frame_diff_common/model_configs -r 100 --rate 0.2 -m ../../../../../frame_diff_common/model_configs/chatglm_decoder.yaml
```

Run multi modules mutating
```python
python new_mutate_and_forward.py -c ../../../../../frame_diff_common/model_configs -r 100 --rate 0.2 -n 4
```

Run single module without mutating
```python
python new_mutate_and_forward.py -c ../../../../../frame_diff_common/model_configs -r 100 --rate 0.2 -m ../../../../../frame_diff_common/model_configs/baichuan.yaml -nm
```


## An example generated JSON file structure

```json
{
    "0": {
        "mutated": false, 
        "before": {
            "TransformerConfig": {
               ...
            },
            ...
        },
        "after": {}
    },
    "1": {
        "mutated": true,
        "before": {
            "TransformerConfig": {
               ...
            },
            ...
        },
        "after": {
            "TransformerConfig": {
               ...
            },
            ...
        },
        "diff": {
            "created": {},
            "deleted": [],
            "modified": {
                "layernorm_epsilon": {
                    "from": 1e-06,
                    "to": 1e-05
                },
                "normalization": {
                    "from": "RMSNorm",
                    "to": "LayerNorm"
                }
            }
        }
    },
    "success": true
}
```


Run single module mutating
```python
python withnum_new_mutate_and_forward.py -c ../../../../../frame_diff_common/model_configs -r 100 --mutnm 2 -m ../../../../../frame_diff_common/model_configs/chatglm_decoder.yaml 
```

Run multi modules mutating
```python
python withnum_new_mutate_and_forward.py -c ../../../../../frame_diff_common/model_configs -r 100 --mutnm 2 -n 4
```
