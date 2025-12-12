# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import gc
import os
import sys
import torch
from functools import partial
from importlib.metadata import version
from packaging.version import Version as PkgVersion

from schema_core import get_model_schema
from saver_base import MegatronCheckpointSaverBase
from utils import chunk_weight, chunk_bias


def add_arguments(parser):
    group = parser.add_argument_group(title='M-Core saver')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')

    group.add_argument('--target-tensor-parallel-size', type=int,
                       help='Target tensor model parallel size, defaults to the tensor parallel size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-pipeline-parallel-size', type=int,
                       help='Target tensor model parallel size, default to the pipeline parall size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-expert-parallel-size', type=int, default=1,
                       help='Target expert model parallel size, default to 1')
    group.add_argument('--saver-transformer-impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')
    group.add_argument('--sequential-save', action='store_true', default=False,
                       help='Process and save one TP shard at a time to reduce memory usage. '
                            'Required for large models (>100B) with high TP on memory-constrained systems.')


class MegatronCheckpointSaverLLM(MegatronCheckpointSaverBase):
    def import_model_provider(self):
        try:
            from megatron.core.enums import ModelType
        except ModuleNotFoundError as e:
            print(f"Unable to import required Megatron modules: {e}")
            sys.exit(1)

        if self.md.model_type == 'GPT':
            from model_provider import model_provider
            from gpt_builders import gpt_builder
            self.model_provider = partial(model_provider, gpt_builder)
            self.margs.model_type = ModelType.encoder_or_decoder
        elif self.md.model_type == 'BERT':
            from pretrain_bert import model_provider
            self.model_provider = model_provider
            self.margs.model_type = ModelType.encoder_or_decoder
        else:
            raise Exception(f'unrecognized model type: {self.args.model_type}')

    def receive_model(self):
        # Model schema.
        schema = get_model_schema(
            self.md.model_type,
            self.margs.transformer_impl,
            self.margs.num_experts,
            self.margs.expert_model_parallel_size,
        )
        self.receive_lm(schema)

    def receive_and_store_model_data(self):
        """
        Receive all model data from loader and store it without building models.

        This stores:
        - self.stored_embeddings: dict with 'pos' and 'word' embeddings
        - self.stored_layers: list of dicts, one per layer with all weight tensors
        - self.stored_final: dict with final norm, output layer, etc.
        """
        try:
            from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
        except ModuleNotFoundError as e:
            print(f"Unable to import required Megatron modules: {e}")
            sys.exit(1)

        # Get schema for layer count
        self.schema = get_model_schema(
            self.md.model_type,
            self.margs.transformer_impl,
            self.margs.num_experts,
            self.margs.expert_model_parallel_size,
        )

        # Embeddings
        print("Receiving embeddings...")
        embeddings_msg = self.queue_get("embeddings")
        pos_embed = None
        if self.md.position_embedding_type == 'learned_absolute':
            pos_embed = embeddings_msg.pop("position embeddings")
        orig_word_embed = embeddings_msg.pop("word embeddings")
        self.check_message(embeddings_msg)

        # Pad word embeddings
        if self.md.true_vocab_size is not None:
            orig_vocab_size = orig_word_embed.shape[0]
            self.margs.padded_vocab_size = _vocab_size_with_padding(self.md.true_vocab_size, self.margs)
            if orig_vocab_size > self.margs.padded_vocab_size:
                full_word_embed = orig_word_embed[0:self.margs.padded_vocab_size, :]
            elif orig_vocab_size < self.margs.padded_vocab_size:
                padding_size = self.margs.padded_vocab_size - orig_vocab_size
                full_word_embed = torch.cat((
                    orig_word_embed,
                    orig_word_embed[-1].unsqueeze(0).expand(padding_size, -1)))
            else:
                full_word_embed = orig_word_embed
        else:
            self.margs.padded_vocab_size = orig_word_embed.shape[0]
            full_word_embed = orig_word_embed

        self.stored_embeddings = {
            'pos': pos_embed,
            'word': full_word_embed,
        }
        del orig_word_embed

        # Transformer layers - store raw data
        self.stored_layers = []
        # We need to know how many layers per PP stage. Build one temp model to find out.
        print("Building temporary model to determine layer count...")
        temp_model = self.model_provider(True, True).to(self.md.params_dtype)
        num_layers_per_pp = self.schema.get_num_layers(temp_model)
        del temp_model
        gc.collect()

        total_layers = self.args.target_pipeline_parallel_size * num_layers_per_pp
        print(f"Receiving {total_layers} transformer layers...")

        for layer_idx in range(total_layers):
            msg = self.queue_get(f"transformer layer {layer_idx}")

            layer_data = {
                'input_norm_weight': msg.pop("input norm weight"),
                'post_norm_weight': msg.pop("post norm weight"),
            }
            if self.md.norm_has_bias:
                layer_data['input_norm_bias'] = msg.pop("input norm bias")
                layer_data['post_norm_bias'] = msg.pop("post norm bias")

            layer_data['qkv_weight'] = msg.pop("qkv weight")
            layer_data['dense_weight'] = msg.pop("dense weight")
            layer_data['mlp_l1_weight'] = msg.pop("mlp l1 weight")

            if self.margs.num_experts:
                layer_data['router_weight'] = msg.pop("router weight")

            if self.md.swiglu:
                layer_data['mlp_l0_weight_W'] = msg.pop("mlp l0 weight W")
                layer_data['mlp_l0_weight_V'] = msg.pop("mlp l0 weight V")
            else:
                layer_data['mlp_l0_weight'] = msg.pop("mlp l0 weight")

            if self.md.qkv_bias:
                layer_data['qkv_bias'] = msg.pop("qkv bias")
            if self.md.linear_bias:
                layer_data['dense_bias'] = msg.pop("dense bias")
                layer_data['mlp_l1_bias'] = msg.pop("mlp l1 bias")
                if self.md.swiglu:
                    layer_data['mlp_l0_bias_W'] = msg.pop("mlp l0 bias W")
                    layer_data['mlp_l0_bias_V'] = msg.pop("mlp l0 bias V")
                else:
                    layer_data['mlp_l0_bias'] = msg.pop("mlp l0 bias")

            self.stored_layers.append(layer_data)
            self.check_message(msg)

            if (layer_idx + 1) % 10 == 0:
                print(f"  Received layer {layer_idx + 1}/{total_layers}")

        # Final norm and output layer
        print("Receiving final norm and output layer...")
        self.stored_final = {}

        msg = self.queue_get("final norm")
        self.stored_final['norm_weight'] = msg.pop("weight")
        if self.md.norm_has_bias:
            self.stored_final['norm_bias'] = msg.pop("bias")
        self.check_message(msg)

        if self.md.output_layer:
            msg = self.queue_get("output layer")
            output_weight = msg.pop("weight")
            # Pad output layer same as embeddings
            if self.md.true_vocab_size is not None:
                orig_vocab_size = output_weight.shape[0]
                if orig_vocab_size > self.margs.padded_vocab_size:
                    output_weight = output_weight[0:self.margs.padded_vocab_size, :]
                elif orig_vocab_size < self.margs.padded_vocab_size:
                    padding_size = self.margs.padded_vocab_size - orig_vocab_size
                    output_weight = torch.cat((
                        output_weight,
                        output_weight[-1].unsqueeze(0).expand(padding_size, -1)))
            self.stored_final['output_weight'] = output_weight
            self.check_message(msg)

        # Handle remaining messages (pooler, lm_head, binary_head, done)
        msg = self.queue_get()
        if msg != "done" and isinstance(msg, dict) and msg.get("name") == "pooler":
            print("Received pooler")
            self.stored_final['pooler_weight'] = msg.pop("weight")
            self.stored_final['pooler_bias'] = msg.pop("bias")
            self.check_message(msg)
            msg = self.queue_get()

        if msg != "done" and isinstance(msg, dict) and msg.get("name") == "lm head":
            print("Received lm head")
            self.stored_final['lm_head_dense_weight'] = msg.pop("dense weight")
            self.stored_final['lm_head_dense_bias'] = msg.pop("dense bias")
            self.stored_final['lm_head_norm_weight'] = msg.pop("norm weight")
            if self.md.norm_has_bias:
                self.stored_final['lm_head_norm_bias'] = msg.pop("norm bias")
            self.check_message(msg)
            msg = self.queue_get()

        if msg != "done" and isinstance(msg, dict) and msg.get("name") == "binary head":
            print("Received binary head")
            self.stored_final['binary_head_weight'] = msg.pop("weight")
            self.stored_final['binary_head_bias'] = msg.pop("bias")
            self.check_message(msg)
            msg = self.queue_get()

        if msg != "done":
            print(f"WARNING: Unexpected message after final components: {msg}")

        print(f"All model data received. Stored {len(self.stored_layers)} layers.")

    def save_models_sequentially(self):
        """
        Build, populate, and save models one TP shard at a time.
        """
        try:
            from megatron.training.checkpointing import save_checkpoint
            from megatron.core import mpu
        except ModuleNotFoundError as e:
            print(f"Unable to import required Megatron modules: {e}")
            sys.exit(1)

        tp_size = self.args.target_tensor_parallel_size
        ep_size = self.args.target_expert_parallel_size
        pp_size = self.args.target_pipeline_parallel_size

        # Pre-split embeddings (they're needed for every TP rank)
        word_embed_chunks = torch.chunk(self.stored_embeddings['word'], tp_size, dim=0)
        output_weight_chunks = None
        if 'output_weight' in self.stored_final:
            output_weight_chunks = torch.chunk(self.stored_final['output_weight'], tp_size, dim=0)

        print(f"Saving checkpoints sequentially: PP={pp_size}, EP={ep_size}, TP={tp_size}")

        for pp_rank in range(pp_size):
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            pre_process = (pp_rank == 0)
            post_process = (pp_rank == pp_size - 1)

            # Figure out which layers belong to this PP rank
            layers_per_pp = len(self.stored_layers) // pp_size
            layer_start = pp_rank * layers_per_pp
            layer_end = layer_start + layers_per_pp

            for ep_rank in range(ep_size):
                for tp_rank in range(tp_size):
                    print(f"Processing PP={pp_rank}, EP={ep_rank}, TP={tp_rank}...")

                    # Update ranks for model building
                    mpu.set_tensor_model_parallel_rank(tp_rank)
                    mpu.set_expert_model_parallel_rank(ep_rank)
                    self.fake_tp_group.set_rank(tp_rank)
                    self.fake_ep_group.set_rank(ep_rank)

                    # Build model for this shard
                    model = self.model_provider(pre_process, post_process).to(self.md.params_dtype)

                    # Set embeddings (only on first PP rank)
                    if pre_process:
                        self.schema.set("embeddings", model, {
                            "pos": self.stored_embeddings['pos'],
                            "word": word_embed_chunks[tp_rank],
                        })

                    # Set transformer layers
                    local_layer_idx = 0
                    for global_layer_idx in range(layer_start, layer_end):
                        layer_data = self.stored_layers[global_layer_idx]

                        # Chunk weights for this TP/EP rank
                        qkv_weight = chunk_weight(layer_data['qkv_weight'], "column", tp_size)[tp_rank]
                        dense_weight = chunk_weight(layer_data['dense_weight'], "row", tp_size)[tp_rank]
                        mlp_l1_weight = chunk_weight(layer_data['mlp_l1_weight'], "row", tp_size, ep_size)

                        if self.md.swiglu:
                            mlp_l0_W = chunk_weight(layer_data['mlp_l0_weight_W'], "column", tp_size, ep_size)
                            mlp_l0_V = chunk_weight(layer_data['mlp_l0_weight_V'], "column", tp_size, ep_size)
                            if self.margs.num_experts:
                                mlp_l0_weight = torch.cat((mlp_l0_W[ep_rank][tp_rank], mlp_l0_V[ep_rank][tp_rank]), dim=-2)
                            else:
                                mlp_l0_weight = torch.cat((mlp_l0_W[tp_rank], mlp_l0_V[tp_rank]), dim=-2)
                        else:
                            mlp_l0_weight = chunk_weight(layer_data['mlp_l0_weight'], "column", tp_size, ep_size)
                            if self.margs.num_experts:
                                mlp_l0_weight = mlp_l0_weight[ep_rank][tp_rank]
                            else:
                                mlp_l0_weight = mlp_l0_weight[tp_rank]

                        params_dict = {
                            "self_attn_norm_weight": layer_data['input_norm_weight'],
                            "self_attn_qkv_weight": qkv_weight,
                            "self_attn_proj_weight": dense_weight,
                            "mlp_norm_weight": layer_data['post_norm_weight'],
                        }

                        if self.margs.num_experts:
                            params_dict["mlp_fc1_weight"] = mlp_l0_weight
                            params_dict["mlp_fc2_weight"] = mlp_l1_weight[ep_rank][tp_rank]
                            params_dict["router_weight"] = layer_data['router_weight']
                        else:
                            params_dict["mlp_fc1_weight"] = mlp_l0_weight
                            params_dict["mlp_fc2_weight"] = mlp_l1_weight[tp_rank]

                        # Biases
                        params_dict["self_attn_norm_bias"] = layer_data.get('input_norm_bias')
                        params_dict["mlp_norm_bias"] = layer_data.get('post_norm_bias')

                        if self.md.qkv_bias:
                            qkv_bias = chunk_bias(layer_data['qkv_bias'], 'column', tp_size)[tp_rank]
                            params_dict["self_attn_qkv_bias"] = qkv_bias

                        if self.md.linear_bias:
                            params_dict["self_attn_proj_bias"] = layer_data['dense_bias']
                            if self.md.swiglu:
                                mlp_l0_bias_W = chunk_bias(layer_data['mlp_l0_bias_W'], 'column', tp_size, ep_size)
                                mlp_l0_bias_V = chunk_bias(layer_data['mlp_l0_bias_V'], 'column', tp_size, ep_size)
                                if self.margs.num_experts:
                                    mlp_l0_bias = torch.cat((mlp_l0_bias_W[ep_rank][tp_rank], mlp_l0_bias_V[ep_rank][tp_rank]), dim=-1)
                                else:
                                    mlp_l0_bias = torch.cat((mlp_l0_bias_W[tp_rank], mlp_l0_bias_V[tp_rank]), dim=-1)
                            else:
                                mlp_l0_bias = chunk_bias(layer_data['mlp_l0_bias'], 'column', tp_size, ep_size)
                                if self.margs.num_experts:
                                    mlp_l0_bias = mlp_l0_bias[ep_rank][tp_rank]
                                else:
                                    mlp_l0_bias = mlp_l0_bias[tp_rank]
                            params_dict["mlp_fc1_bias"] = mlp_l0_bias

                            mlp_l1_bias = chunk_bias(layer_data['mlp_l1_bias'], 'row', tp_size, ep_size)
                            if self.margs.num_experts:
                                params_dict["mlp_fc2_bias"] = mlp_l1_bias[ep_rank]
                            else:
                                params_dict["mlp_fc2_bias"] = mlp_l1_bias

                        self.schema.set_layer(model, local_layer_idx, params_dict)
                        local_layer_idx += 1

                    # Set final norm and output layer (only on last PP rank)
                    if post_process:
                        self.schema.set("final_norm", model, {
                            "weight": self.stored_final['norm_weight'],
                            "bias": self.stored_final.get('norm_bias'),
                        })

                        if output_weight_chunks is not None:
                            self.schema.set("output_layer", model, {
                                "weight": output_weight_chunks[tp_rank],
                            })
                        elif not self.md.output_layer and pp_rank != 0:
                            # Copy embeddings to output layer
                            self.schema.set("output_layer", model, {
                                "weight": word_embed_chunks[tp_rank],
                            })

                        # Pooler
                        if 'pooler_weight' in self.stored_final:
                            self.schema.set("pooler", model, {
                                "weight": self.stored_final['pooler_weight'],
                                "bias": self.stored_final['pooler_bias'],
                            })

                        # LM head
                        if 'lm_head_dense_weight' in self.stored_final:
                            self.schema.set("lm_head", model, {
                                "dense_weight": self.stored_final['lm_head_dense_weight'],
                                "dense_bias": self.stored_final['lm_head_dense_bias'],
                                "norm_weight": self.stored_final['lm_head_norm_weight'],
                                "norm_bias": self.stored_final.get('lm_head_norm_bias'),
                            })

                        # Binary head
                        if 'binary_head_weight' in self.stored_final:
                            self.schema.set("binary_head", model, {
                                "weight": self.stored_final['binary_head_weight'],
                                "bias": self.stored_final['binary_head_bias'],
                            })

                    # Save checkpoint
                    print(f"  Saving checkpoint for PP={pp_rank}, EP={ep_rank}, TP={tp_rank}...")
                    save_checkpoint(
                        self.md.iteration, [model], None, None,
                        num_floating_point_operations_so_far=0,
                        pipeline_rank=pp_rank,
                        pipeline_parallel=pp_size > 1,
                        expert_rank=ep_rank,
                        expert_parallel=ep_size > 1,
                        tensor_rank=tp_rank
                    )

                    # Release model to free memory
                    del model
                    gc.collect()
                    print(f"  Checkpoint saved and model released.")

        print("All checkpoints saved successfully.")

def save_checkpoint(queue, args):
    """
    Required top-level function that creates the saver and calls its .save().

    If --sequential-save is specified, uses save_sequential() which processes
    one TP shard at a time to reduce memory usage for large models.
    """
    saver = MegatronCheckpointSaverLLM(args, queue)
    try:
        if getattr(args, 'sequential_save', False):
            print("Using sequential save mode (memory-efficient)")
            saver.save_sequential()
        else:
            saver.save()
    except Exception as e:
        raise e
