

# 额外PYTHONPATH设置，需要预先配置环境变量$MSAPATH

export ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}

# After CANN run-package upgrades/rollbacks, MindSpore may need the NNRT
# runtime root to resolve Ascend runtime symbols, while OPP/opp_kernel still
# come from the full toolkit installation.
if [ -d /usr/local/Ascend/nnrt/latest/lib64 ]; then
  export ASCEND_HOME_PATH=/usr/local/Ascend/nnrt/latest
  export ASCEND_AICPU_PATH=/usr/local/Ascend/nnrt/latest
  export LD_LIBRARY_PATH=/usr/local/Ascend/nnrt/latest/lib64:${LD_LIBRARY_PATH}
fi
if [ -d /usr/local/Ascend/ascend-toolkit/latest/opp ]; then
  export ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
fi
if [ -d /usr/local/Ascend/ascend-toolkit/latest/opp_kernel ]; then
  export ASCEND_OPP_KERNEL_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp_kernel
fi

MSA_ROOT="${MSAPATH}"
if [ -d "${MSAPATH}/MindSpeed-Core-MS" ]; then
  MSA_ROOT="${MSAPATH}/MindSpeed-Core-MS"
fi
export PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:${MSA_ROOT}/MindSpeed-LLM:${MSA_ROOT}/MSAdapter:${MSA_ROOT}/MSAdapter/msa_thirdparty:${MSA_ROOT}/MindSpeed:${MSA_ROOT}/Megatron-LM
