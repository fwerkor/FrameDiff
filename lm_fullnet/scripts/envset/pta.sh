

# 额外PYTHONPATH设置，需要预先配置环境变量$PTAPATH

export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${PTAPATH}/MindSpeed-LLM

export HCCL_DETERMINISTIC=true 
export ASCEND_LAUNCH_BLOCKING=1 
export NCCL_DETERMINISTIC=1
