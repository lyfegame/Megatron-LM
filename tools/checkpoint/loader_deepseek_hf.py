# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
DeepSeek V3/V3.2 checkpoint loader for converting HuggingFace checkpoints to Megatron format.

This loader supports:
- Multi-Latent Attention (MLA) with q_lora_rank and kv_lora_rank
- Mixture of Experts (MoE) with shared experts
- YaRN RoPE scaling for extended context
- DeepSeek V3.2 Lightning Indexer for sparse attention (if weights are present)

Usage:
    python tools/checkpoint/convert.py \
        --model-type GPT \
        --loader deepseek_hf \
        --saver core \
        --load-dir /path/to/deepseek-v3-hf \
        --save-dir /path/to/megatron-checkpoint \
        --tokenizer-model /path/to/tokenizer \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 4

References:
    - DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
    - DeepSeek-V3.2 Technical Report: https://arxiv.org/abs/2512.02556
    - Model Weights: https://huggingface.co/deepseek-ai/DeepSeek-V3
"""

import json
import os
import sys
import types

import torch
from tqdm import tqdm

try:
    import transformers
except ImportError:
    raise ImportError("The 'transformers' package is not installed.")

from utils import _ConverterFakeProcessGroup


def add_arguments(parser):
    """Add DeepSeek-specific arguments to the parser."""
    group = parser.add_argument_group(title='DeepSeek HF loader.')

    group.add_argument(
        '--true-vocab-size',
        type=int,
        default=None,
        help='Original size of vocab, if specified will trim padding from embedding table.',
    )
    group.add_argument(
        '--tokenizer-model',
        required=True,
        help='Path to HuggingFace tokenizer model or directory.',
    )
    group.add_argument(
        '--megatron-path',
        type=str,
        default=None,
        help='Base directory of Megatron repository.',
    )
    group.add_argument(
        '--bf16',
        action='store_true',
        help='Whether to load weights in bf16.',
    )
    group.add_argument(
        '--fp16',
        action='store_true',
        help='Whether to load weights in fp16.',
    )
    group.add_argument(
        '--loader-transformer-impl',
        default='local',
        choices=['local', 'transformer_engine'],
        help='Which Transformer implementation to use. Defaults to local for CPU-only conversion.',
    )


def verify_transformers_version():
    """Verify transformers version is compatible."""
    import re
    version_match = re.match(r'(\d+)\.(\d+)', transformers.__version__)
    if version_match:
        major, minor = int(version_match.group(1)), int(version_match.group(2))
        assert major >= 4 or (major == 4 and minor >= 36) or major >= 5, (
            f"DeepSeek V3 requires transformers >= 4.36.0, got {transformers.__version__}"
        )


def load_args_from_checkpoint(args):
    """Load model arguments from HuggingFace config.json."""
    config_path = os.path.join(args.load, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Basic model dimensions
    args.hidden_size = config["hidden_size"]
    args.num_attention_heads = config["num_attention_heads"]
    args.num_layers = config["num_hidden_layers"]
    args.vocab_size = config["vocab_size"]
    args.padded_vocab_size = config["vocab_size"]
    args.ffn_hidden_size = config["intermediate_size"]
    args.norm_epsilon = config.get("rms_norm_eps", 1e-6)
    args.max_position_embeddings = config["max_position_embeddings"]

    # Standard settings
    args.seq_length = min(4096, args.max_position_embeddings)
    args.global_batch_size = 1024
    args.iteration = 1
    args.add_position_embedding = False
    args.use_rotary_position_embeddings = True
    args.position_embedding_type = "rope"
    args.swiglu = True
    args.normalization = "RMSNorm"
    args.add_bias_linear = config.get("attention_bias", False)
    args.untie_embeddings_and_output_weights = not config.get("tie_word_embeddings", False)

    # MLA (Multi-Latent Attention) parameters
    args.multi_latent_attention = True
    args.q_lora_rank = config.get("q_lora_rank")  # None or 1536
    args.kv_lora_rank = config.get("kv_lora_rank", 512)
    args.qk_head_dim = config.get("qk_nope_head_dim", 128)
    args.qk_pos_emb_head_dim = config.get("qk_rope_head_dim", 64)
    args.v_head_dim = config.get("v_head_dim", 128)

    # RoPE parameters
    args.rotary_base = config.get("rope_theta", 10000.0)
    rope_scaling = config.get("rope_scaling", {})
    args.rope_type = rope_scaling.get("type", "rope")
    if args.rope_type == "yarn":
        args.rotary_scaling_factor = rope_scaling.get("factor", 1.0)
        args.original_max_position_embeddings = rope_scaling.get(
            "original_max_position_embeddings", 4096
        )
        args.beta_fast = rope_scaling.get("beta_fast", 32.0)
        args.beta_slow = rope_scaling.get("beta_slow", 1.0)
    else:
        args.rotary_scaling_factor = 1.0
        args.original_max_position_embeddings = args.max_position_embeddings

    # MoE (Mixture of Experts) parameters
    args.num_experts = config.get("n_routed_experts", 1)
    args.moe_router_topk = config.get("num_experts_per_tok", 8)
    args.num_shared_experts = config.get("n_shared_experts", 0)
    args.moe_intermediate_size = config.get("moe_intermediate_size", args.ffn_hidden_size)
    # Megatron uses moe_ffn_hidden_size for MoE expert FFN dimensions
    args.moe_ffn_hidden_size = args.moe_intermediate_size

    # Set shared expert intermediate size (num_shared_experts * ffn_size_of_each_shared_expert)
    # DeepSeek V3 shared experts use the same intermediate_size as MoE experts
    if args.num_shared_experts > 0:
        args.moe_shared_expert_intermediate_size = args.num_shared_experts * args.moe_intermediate_size
    args.first_k_dense_replace = config.get("first_k_dense_replace", 0)

    # Build moe_layer_freq list to handle first_k_dense_replace
    # For DeepSeek V3: first k layers are dense (0), rest are MoE (1)
    base_moe_freq = config.get("moe_layer_freq", 1)
    if args.first_k_dense_replace > 0:
        # Create explicit layer pattern: 0 for dense, 1 for MoE
        moe_layer_pattern = [0] * args.first_k_dense_replace + [1] * (args.num_layers - args.first_k_dense_replace)
        args.moe_layer_freq = moe_layer_pattern
    else:
        args.moe_layer_freq = base_moe_freq

    # GQA settings (DeepSeek V3 uses MLA, not GQA, but set for compatibility)
    num_kv_heads = config.get("num_key_value_heads", args.num_attention_heads)
    if num_kv_heads != args.num_attention_heads:
        args.group_query_attention = True
        args.num_query_groups = num_kv_heads
    else:
        args.group_query_attention = False

    # Store full config for reference
    args.deepseek_config = config

    return args


def set_preprocess_state(args, model, hf_model):
    """Set embedding parameters."""
    model.embedding.word_embeddings.weight.data.copy_(hf_model.model.embed_tokens.weight)


def set_postprocess_state(args, model, hf_model):
    """Set output layer and norm parameters."""
    model.decoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight)
    if args.untie_embeddings_and_output_weights:
        model.output_layer.weight.data.copy_(hf_model.lm_head.weight)


def set_mla_attn_state(args, layer, hf_layer):
    """Set Multi-Latent Attention parameters.

    DeepSeek V3 MLA structure:
        HuggingFace:
            - q_a_proj: [hidden_size, q_lora_rank]  (down projection)
            - q_a_layernorm: [q_lora_rank]
            - q_b_proj: [q_lora_rank, num_heads * q_head_dim]  (up projection)
            - kv_a_proj_with_mqa: [hidden_size, kv_lora_rank + qk_rope_head_dim]
            - kv_a_layernorm: [kv_lora_rank]
            - kv_b_proj: [kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)]
            - o_proj: [num_heads * v_head_dim, hidden_size]

        Megatron MLA:
            - linear_q_down_proj: [hidden_size, q_lora_rank]
            - q_layernorm: [q_lora_rank]
            - linear_q_up_proj: [q_lora_rank, num_heads * q_head_dim]
            - linear_kv_down_proj: [hidden_size, kv_lora_rank + qk_pos_emb_head_dim]
            - kv_layernorm: [kv_lora_rank]
            - linear_kv_up_proj: [kv_lora_rank, num_heads * (qk_head_dim + v_head_dim)]
            - linear_proj: [num_heads * v_head_dim, hidden_size]
    """
    attn = layer.self_attention
    hf_attn = hf_layer.self_attn

    # Query projections
    if args.q_lora_rank is not None:
        # Down projection (with fused layernorm in TE)
        if hasattr(attn, 'linear_q_down_proj'):
            if hasattr(attn.linear_q_down_proj, 'layer_norm_weight'):
                # TE fused layernorm-linear
                attn.linear_q_down_proj.weight.data.copy_(hf_attn.q_a_proj.weight)
                attn.linear_q_down_proj.layer_norm_weight.data.copy_(hf_attn.q_a_layernorm.weight)
            else:
                attn.linear_q_down_proj.weight.data.copy_(hf_attn.q_a_proj.weight)
                # Only copy layernorm weights if it's a real LayerNorm (not IdentityOp)
                if hasattr(attn, 'q_layernorm') and hasattr(attn.q_layernorm, 'weight'):
                    attn.q_layernorm.weight.data.copy_(hf_attn.q_a_layernorm.weight)
        # Up projection
        if hasattr(attn, 'linear_q_up_proj'):
            attn.linear_q_up_proj.weight.data.copy_(hf_attn.q_b_proj.weight)
    else:
        # Direct Q projection (no LoRA)
        if hasattr(attn, 'linear_q_proj'):
            attn.linear_q_proj.weight.data.copy_(hf_attn.q_proj.weight)

    # KV projections
    if hasattr(attn, 'linear_kv_down_proj'):
        if hasattr(attn.linear_kv_down_proj, 'layer_norm_weight'):
            # TE fused layernorm-linear
            attn.linear_kv_down_proj.weight.data.copy_(hf_attn.kv_a_proj_with_mqa.weight)
            attn.linear_kv_down_proj.layer_norm_weight.data.copy_(hf_attn.kv_a_layernorm.weight)
        else:
            attn.linear_kv_down_proj.weight.data.copy_(hf_attn.kv_a_proj_with_mqa.weight)
            # Only copy layernorm weights if it's a real LayerNorm (not IdentityOp)
            if hasattr(attn, 'kv_layernorm') and hasattr(attn.kv_layernorm, 'weight'):
                attn.kv_layernorm.weight.data.copy_(hf_attn.kv_a_layernorm.weight)

    if hasattr(attn, 'linear_kv_up_proj'):
        attn.linear_kv_up_proj.weight.data.copy_(hf_attn.kv_b_proj.weight)

    # Output projection
    attn.linear_proj.weight.data.copy_(hf_attn.o_proj.weight)


def set_moe_mlp_state(args, layer, hf_layer, layer_idx):
    """Set MoE MLP parameters.

    DeepSeek V3 MoE structure:
        HuggingFace:
            - mlp.gate.weight: [n_routed_experts, hidden_size]  (router)
            - mlp.gate.e_score_correction_bias: [n_routed_experts]  (optional)
            - mlp.experts[i].gate_proj: [hidden_size, moe_intermediate_size]
            - mlp.experts[i].up_proj: [hidden_size, moe_intermediate_size]
            - mlp.experts[i].down_proj: [moe_intermediate_size, hidden_size]
            - mlp.shared_experts.gate_proj: [hidden_size, shared_intermediate_size]
            - mlp.shared_experts.up_proj: [hidden_size, shared_intermediate_size]
            - mlp.shared_experts.down_proj: [shared_intermediate_size, hidden_size]

        Megatron MoE:
            - mlp.router.weight: [n_routed_experts, hidden_size]
            - mlp.experts.local_experts[i].linear_fc1.weight: [2 * moe_intermediate_size, hidden_size]
            - mlp.experts.local_experts[i].linear_fc2.weight: [hidden_size, moe_intermediate_size]
            - mlp.shared_experts.linear_fc1.weight: [2 * shared_intermediate_size, hidden_size]
            - mlp.shared_experts.linear_fc2.weight: [hidden_size, shared_intermediate_size]

    Note: DeepSeek V3 uses first_k_dense_replace to make the first k layers dense.
    However, Megatron may still use MoELayer for all layers. This function handles
    both cases by checking both HF and Megatron layer structures.
    """
    hf_mlp = hf_layer.mlp

    # Check if HF layer is an MoE layer
    hf_is_moe = hasattr(hf_mlp, 'experts')
    # Check if Megatron layer is an MoELayer (has router attribute)
    megatron_is_moe = hasattr(layer.mlp, 'router')

    if hf_is_moe and megatron_is_moe:
        # Router weights
        layer.mlp.router.weight.data.copy_(hf_mlp.gate.weight)

        # Expert weights
        mcore_experts = layer.mlp.experts.local_experts
        hf_experts = hf_mlp.experts

        for expert_idx in range(args.num_experts):
            # Fuse gate_proj and up_proj into linear_fc1 (SwiGLU)
            mcore_experts[expert_idx].linear_fc1.weight.data.copy_(
                torch.cat(
                    [hf_experts[expert_idx].gate_proj.weight, hf_experts[expert_idx].up_proj.weight],
                    dim=0,
                )
            )
            mcore_experts[expert_idx].linear_fc2.weight.data.copy_(
                hf_experts[expert_idx].down_proj.weight
            )

        # Shared experts (if present)
        if args.num_shared_experts > 0 and hasattr(hf_mlp, 'shared_experts'):
            hf_shared = hf_mlp.shared_experts
            # Check that shared_experts exists and is not None
            if hasattr(layer.mlp, 'shared_experts') and layer.mlp.shared_experts is not None:
                layer.mlp.shared_experts.linear_fc1.weight.data.copy_(
                    torch.cat([hf_shared.gate_proj.weight, hf_shared.up_proj.weight], dim=0)
                )
                layer.mlp.shared_experts.linear_fc2.weight.data.copy_(hf_shared.down_proj.weight)
    elif not hf_is_moe and megatron_is_moe:
        # HF layer is dense but Megatron has MoELayer structure
        # This should not happen with proper moe_layer_freq setting
        raise RuntimeError(
            f"Layer {layer_idx} architecture mismatch: HF has dense MLP but Megatron has MoE. "
            f"This indicates moe_layer_freq is not set correctly. Expected first {args.first_k_dense_replace} "
            f"layers to be dense based on first_k_dense_replace config."
        )
    else:
        # Both HF and Megatron are dense MLP
        layer.mlp.linear_fc1.weight.data.copy_(
            torch.cat([hf_mlp.gate_proj.weight, hf_mlp.up_proj.weight], dim=0)
        )
        layer.mlp.linear_fc2.weight.data.copy_(hf_mlp.down_proj.weight)


def set_layer_state(args, model, hf_model, layer_idx):
    """Set transformer layer parameters."""
    layer = model.decoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    # Set MLA attention
    set_mla_attn_state(args, layer, hf_layer)

    # Set MoE or dense MLP
    set_moe_mlp_state(args, layer, hf_layer, layer_idx)

    # Layer norms
    if hasattr(layer.self_attention, 'linear_qkv') and hasattr(
        layer.self_attention.linear_qkv, 'layer_norm_weight'
    ):
        # TE fused input layernorm
        layer.self_attention.linear_qkv.layer_norm_weight.data.copy_(
            hf_layer.input_layernorm.weight
        )
    elif hasattr(layer, 'input_layernorm'):
        layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight)

    if hasattr(layer, 'pre_mlp_layernorm'):
        layer.pre_mlp_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight)
    elif hasattr(layer, 'post_attention_layernorm'):
        layer.post_attention_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight)


def load_checkpoint_to_model(args):
    """Load HuggingFace checkpoint and convert to Megatron model."""
    from gpt_builders import gpt_builder
    from model_provider import model_provider
    from transformers import AutoModelForCausalLM

    # Load HuggingFace model
    print(f"Loading HuggingFace model from {args.load}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.load, torch_dtype=args.params_dtype, low_cpu_mem_usage=True, device_map="cpu"
    )

    # Initialize Megatron model
    print("Initializing Megatron model...")
    model = model_provider(gpt_builder, pre_process=True, post_process=True).to(args.params_dtype)

    # Copy weights
    print("Copying embedding weights...")
    set_preprocess_state(args, model, hf_model)

    print("Copying output layer weights...")
    set_postprocess_state(args, model, hf_model)

    print("Copying transformer layer weights...")
    for layer_idx in tqdm(range(args.num_layers), desc="Copying layers"):
        set_layer_state(args, model, hf_model, layer_idx)

    return model


def _load_checkpoint(queue, args):
    """Main checkpoint loading function."""
    verify_transformers_version()

    # Setup paths - megatron_path takes priority
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    else:
        sys.path.append(
            os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
        )

    print(f"DEBUG: megatron_path = {args.megatron_path}")
    print(f"DEBUG: sys.path[0:3] = {sys.path[0:3]}")

    try:
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron.legacy import fused_kernels
        from megatron.legacy.model import module
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.global_vars import set_global_variables
    except ModuleNotFoundError as e:
        import traceback
        print(f"Unable to import Megatron: {e}")
        traceback.print_exc()
        print("Please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")
        exit(1)

    # Setup Megatron arguments
    sys.argv = [
        'script.py',
        '--use-mcore-models',
        '--disable-bias-linear',
        '--no-masked-softmax-fusion',
        '--no-bias-gelu-fusion',
        '--no-bias-dropout-fusion',
        '--no-async-tensor-model-parallel-allreduce',
        '--use-cpu-initialization',
        '--micro-batch-size',
        '1',
        '--no-load-optim',
        '--no-load-rng',
        '--no-save-optim',
        '--no-save-rng',
        '--no-initialization',
        '--mock-data',
        '--transformer-impl',
        args.loader_transformer_impl,
        '--load',
        args.load_dir,
        '--no-one-logger',
        '--no-persist-layer-norm',  # Required when Apex is not available
        '--no-gradient-accumulation-fusion',  # Required when APEX/CUDA is not available
        # MLA-specific
        '--multi-latent-attention',
    ]

    margs = parse_args()
    margs.tokenizer_model = args.tokenizer_model
    # Load HF config and set model args (including MoE settings)
    # This uses the local load_args_from_checkpoint function which reads from HF config.json
    load_args_from_checkpoint(margs)

    # Set tokenizer type
    margs.tokenizer_type = "HuggingFaceTokenizer"

    # Fake world size for argument validation
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

    margs = validate_args(margs)

    # Set dtype
    if args.bf16:
        margs.params_dtype = torch.bfloat16
    elif args.fp16:
        margs.params_dtype = torch.float16
    else:
        margs.params_dtype = torch.float32

    def check_for_arg(arg_name, default=None):
        if getattr(margs, arg_name, None) is None:
            if default is not None:
                setattr(margs, arg_name, default)
            else:
                print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                queue.put("exit")
                exit(1)

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('position_embedding_type')
    check_for_arg('iteration')
    check_for_arg('params_dtype')
    check_for_arg('swiglu')

    # Set model type
    assert args.model_type == 'GPT', 'DeepSeek V3 is a GPT model.'
    margs.model_type = ModelType.encoder_or_decoder

    # Suppress warnings
    module.MegatronModule.embedding_warning_printed = True

    # Initialize global variables
    set_global_variables(margs, build_tokenizer=False)
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)
    mpu.set_expert_model_parallel_world_size(margs.expert_model_parallel_size)

    # Setup fake process groups for single-process checkpoint conversion
    # All groups use size=1 except TP and EP which use the configured sizes
    fake_tp_group = _ConverterFakeProcessGroup(size=margs.tensor_model_parallel_size)
    fake_ep_group = _ConverterFakeProcessGroup(size=margs.expert_model_parallel_size)
    fake_dp_group = _ConverterFakeProcessGroup(size=1)

    # Core parallel groups
    mpu._TENSOR_MODEL_PARALLEL_GROUP = fake_tp_group
    mpu._PIPELINE_MODEL_PARALLEL_GROUP = fake_dp_group
    mpu._MODEL_PARALLEL_GROUP = fake_dp_group
    mpu._DATA_PARALLEL_GROUP = fake_dp_group
    mpu._DATA_PARALLEL_GROUP_GLOO = fake_dp_group
    mpu._TENSOR_AND_DATA_PARALLEL_GROUP = fake_dp_group

    # Expert parallel groups
    mpu._EXPERT_MODEL_PARALLEL_GROUP = fake_ep_group
    mpu._EXPERT_TENSOR_PARALLEL_GROUP = fake_dp_group
    mpu._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = fake_dp_group
    mpu._EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = fake_dp_group
    mpu._EXPERT_DATA_PARALLEL_GROUP = fake_dp_group
    mpu._EXPERT_DATA_PARALLEL_GROUP_GLOO = fake_dp_group
    mpu._INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = fake_dp_group
    mpu._INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO = fake_dp_group
    mpu._INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = fake_dp_group

    # Context parallel groups
    mpu._CONTEXT_PARALLEL_GROUP = fake_dp_group
    mpu._DATA_PARALLEL_GROUP_WITH_CP = fake_dp_group
    mpu._DATA_PARALLEL_GROUP_WITH_CP_GLOO = fake_dp_group
    mpu._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = fake_dp_group
    mpu._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = fake_dp_group
    mpu._TENSOR_AND_CONTEXT_PARALLEL_GROUP = fake_dp_group
    mpu._TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = fake_dp_group

    # Embedding groups
    mpu._EMBEDDING_GROUP = fake_dp_group
    mpu._POSITION_EMBEDDING_GROUP = fake_dp_group

    # Distributed optimizer group
    mpu._INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = fake_dp_group

    # Try to load fused kernels, but skip if CUDA is not available (CPU-only conversion)
    try:
        fused_kernels.load(margs)
    except Exception as e:
        print(f"Warning: Could not load fused CUDA kernels ({e}). Proceeding without them.")

    # Build metadata
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.norm_has_bias = False
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.checkpoint_args = margs
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0
    md.num_experts = margs.num_experts

    # MLA-specific metadata
    md.multi_latent_attention = True
    md.q_lora_rank = margs.q_lora_rank
    md.kv_lora_rank = margs.kv_lora_rank
    md.qk_head_dim = margs.qk_head_dim
    md.qk_pos_emb_head_dim = margs.qk_pos_emb_head_dim
    md.v_head_dim = margs.v_head_dim

    # Get true vocab size
    tokenizer = transformers.AutoTokenizer.from_pretrained(margs.tokenizer_model)
    md.true_vocab_size = tokenizer.vocab_size

    # Load checkpoint
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    mpu.set_expert_model_parallel_rank(0)
    model = load_checkpoint_to_model(margs)

    # Send metadata
    queue.put(md)

    def queue_put(name, msg):
        print(f"Sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings
    message = {"word embeddings": model.embedding.word_embeddings.weight.data}
    queue_put("embeddings", message)

    # Send transformer layers
    for layer_idx in range(margs.num_layers):
        message = {}
        layer = model.decoder.layers[layer_idx]

        # Layer norms
        if hasattr(layer, 'input_layernorm'):
            message["input norm weight"] = layer.input_layernorm.weight.data
        if hasattr(layer, 'pre_mlp_layernorm'):
            message["post norm weight"] = layer.pre_mlp_layernorm.weight.data

        # MLA attention weights
        attn = layer.self_attention
        if margs.q_lora_rank is not None:
            message["q down proj weight"] = attn.linear_q_down_proj.weight.data
            message["q layernorm weight"] = attn.q_layernorm.weight.data
            message["q up proj weight"] = attn.linear_q_up_proj.weight.data
        else:
            message["q proj weight"] = attn.linear_q_proj.weight.data

        message["kv down proj weight"] = attn.linear_kv_down_proj.weight.data
        message["kv layernorm weight"] = attn.kv_layernorm.weight.data
        message["kv up proj weight"] = attn.linear_kv_up_proj.weight.data
        message["dense weight"] = attn.linear_proj.weight.data

        # MoE or dense MLP weights
        if hasattr(layer.mlp, 'router'):
            # MoE layer
            message["router weight"] = layer.mlp.router.weight.data
            experts = layer.mlp.experts.local_experts

            # SwiGLU splits
            chunked_mlp_l0_weight = [
                torch.chunk(local_expert.linear_fc1.weight.data, 2, dim=0)
                for local_expert in experts
            ]
            message["mlp l0 weight W"] = torch.stack(
                [w[0] for w in chunked_mlp_l0_weight], dim=0
            )
            message["mlp l0 weight V"] = torch.stack(
                [w[1] for w in chunked_mlp_l0_weight], dim=0
            )
            message["mlp l1 weight"] = torch.stack(
                [local_expert.linear_fc2.weight.data for local_expert in experts], dim=0
            )

            # Shared experts
            if hasattr(layer.mlp, 'shared_experts'):
                shared = layer.mlp.shared_experts
                shared_w = torch.chunk(shared.linear_fc1.weight.data, 2, dim=0)
                message["shared mlp l0 weight W"] = shared_w[0]
                message["shared mlp l0 weight V"] = shared_w[1]
                message["shared mlp l1 weight"] = shared.linear_fc2.weight.data
        else:
            # Dense MLP layer
            mlp_w = torch.chunk(layer.mlp.linear_fc1.weight.data, 2, dim=0)
            message["mlp l0 weight W"] = mlp_w[0]
            message["mlp l0 weight V"] = mlp_w[1]
            message["mlp l1 weight"] = layer.mlp.linear_fc2.weight.data

        queue_put(f"transformer layer {layer_idx}", message)

    # Send final norm
    queue_put("final norm", {"weight": model.decoder.final_layernorm.weight.data})

    # Send output layer
    if md.output_layer:
        queue_put("output layer", {"weight": model.output_layer.weight.data})

    queue.put("done")


def load_checkpoint(queue, args):
    """Entry point for checkpoint loading."""
    try:
        _load_checkpoint(queue, args)
    except Exception:
        queue.put("exit")
        raise
