

# 额外PYTHONPATH设置，需要预先配置环境变量$MSAPATH

export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
MSA_ROOT="${MSAPATH}"
if [ -d "${MSAPATH}/MindSpeed-Core-MS" ]; then
  MSA_ROOT="${MSAPATH}/MindSpeed-Core-MS"
fi
export PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${MSA_ROOT}/MindSpeed-LLM:${MSA_ROOT}/MSAdapter:${MSA_ROOT}/MSAdapter/msa_thirdparty:${MSA_ROOT}/MindSpeed:${MSA_ROOT}/Megatron-LM
