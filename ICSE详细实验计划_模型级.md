不再从模型的类似于qwen2.yaml这样的原始文件开始处理了，而是完全取消变异这一步，直接读取mutating.json + mutated_config.yaml（位于../mutated_config/qwen2下面）。我只指定模型（已qwen2为例），程序就一个一个跑它下面的所有变体（每个变体我都要拿到pta-baseline、msa-baseline、pta-preturb、msa-preturb的所有完整各项数据，请注意归档到output/模型名/变体名/训练名，例如output/qwen2/ancestor/pta-baseline），但是只有ancestor跑prepare步骤，后面的变体全都是沿用ancestor留下的共享权重。

接下来要做的，就是等模型变异后数据收集齐之后，一个一个模型跑一遍fullnet就行！数据全会在output里面命名好