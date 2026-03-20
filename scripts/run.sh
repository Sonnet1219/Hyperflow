cd /mnt/data/lyzhuang/hyperflow

# musique
# SPACY_MODEL="en_core_web_trf"
# EMBEDDING_MODEL="model/all-mpnet-base-v2"
# DATASET="musique"
# LLM_MODEL="gpt-4o-mini"
# MAX_WORKERS=16
# PASSAGE_RATIO=2.0
# DIFFUSION_ALPHA=0.85
# DIFFUSION_MAX_ITER=10
# FLOW_DAMPING=0.5
# ACTIVATION_RATIO=0.05

# 2wikimultihop
# SPACY_MODEL="en_core_web_trf"
# EMBEDDING_MODEL="model/all-mpnet-base-v2"
# DATASET="2wikimultihop"
# LLM_MODEL="gpt-4o-mini"
# MAX_WORKERS=16
# PASSAGE_RATIO=0.05
# DIFFUSION_ALPHA=0.85
# DIFFUSION_MAX_ITER=10
# FLOW_DAMPING=0.5
# ACTIVATION_RATIO=0.05

# hotpotqa
# SPACY_MODEL="en_core_web_trf"
# EMBEDDING_MODEL="model/all-mpnet-base-v2"
# DATASET="hotpotqa"
# LLM_MODEL="gpt-4o-mini"
# MAX_WORKERS=16
# PASSAGE_RATIO=0.05
# DIFFUSION_ALPHA=0.85
# DIFFUSION_MAX_ITER=10
# FLOW_DAMPING=0.5
# ACTIVATION_RATIO=0.05



python run.py \
    --spacy_model ${SPACY_MODEL} \
    --embedding_model ${EMBEDDING_MODEL} \
    --dataset_name ${DATASET} \
    --llm_model ${LLM_MODEL} \
    --max_workers ${MAX_WORKERS} \
    --passage_ratio ${PASSAGE_RATIO} \
    --diffusion_alpha ${DIFFUSION_ALPHA} \
    --diffusion_max_iter ${DIFFUSION_MAX_ITER} \
    --flow_damping ${FLOW_DAMPING} \
    --activation_ratio ${ACTIVATION_RATIO}
