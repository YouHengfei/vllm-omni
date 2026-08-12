# SPDX-License-Identifier: Apache-2.0
"""Test-only worker diagnostics and fault injection for VibeVoice."""

from __future__ import annotations

import gc
import json
import os
import weakref
from pathlib import Path
from typing import Any


def _runner_snapshot(worker: Any) -> dict[str, Any]:
    runner = worker.model_runner
    model = runner.get_model()
    stateful = model._stateful

    request_states: dict[str, dict[str, Any]] = {}
    for request_id, state in stateful._states.items():
        request_states[request_id] = {
            "in_audio_segment": bool(state.in_audio_segment),
            "audio_token_count": int(state.audio_token_count),
            "has_acoustic_cache": state.acoustic_cache is not None,
            "has_semantic_cache": state.semantic_cache is not None,
            "waveform_chunk_count": len(state.waveform_chunks_cpu),
            "waveform_samples": sum(int(chunk.numel()) for chunk in state.waveform_chunks_cpu),
        }

    named_branches: dict[str, dict[str, Any]] = {}
    for name, branch in runner.named_kv_branches.items():
        named_branches[name] = {
            "closed": bool(branch._closed),
            "entered": bool(branch._entered),
            "num_blocks": int(branch.num_blocks),
            "num_free_blocks": int(branch.num_free_blocks),
            "requests": {
                request_id: {
                    "num_tokens": int(state.num_tokens),
                    "block_ids": list(state.block_ids),
                }
                for request_id, state in branch._states.items()
            },
        }

    return {
        "rank": int(worker.rank),
        "request_states": request_states,
        "deferred_cleanup_ids": sorted(stateful.deferred_cleanup_ids),
        "named_branches": named_branches,
        "runner_request_ids": sorted(runner.requests),
        "model_intermediate_request_ids": sorted(runner.model_intermediate_buffer),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True))
    os.replace(temporary, path)


class VibeVoiceWorkerExtensionForTest:
    """Expose request-local state and inject synchronized test instrumentation.

    This class is loaded only through tests' ``worker_extension_cls``. It adds
    no production worker API and mutates runtime objects only inside the test
    worker processes.
    """

    def vibevoice_test_runtime_state(self) -> dict[str, Any]:
        return _runner_snapshot(self)

    def vibevoice_test_arm_generation_trace(
        self,
        seed: int,
    ) -> dict[str, Any]:
        """Seed this TP rank and capture full cached audio-token transitions."""
        import torch

        runner = self.model_runner
        model = runner.get_model()
        kernel = model.model
        if hasattr(self, "_vibevoice_generation_trace"):
            raise RuntimeError("VibeVoice generation trace was armed twice")

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        original_audio_encode = model.model.audio_tower.encode
        encoded_reference_latents: list[Any] = []

        def encode_with_deterministic_mode(
            input_values: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            kwargs["sample"] = False
            encoded = original_audio_encode(input_values, *args, **kwargs)
            encoded_reference_latents.append(encoded.latents.detach().cpu())
            return encoded

        model.model.audio_tower.encode = encode_with_deterministic_mode
        model._vibevoice_test_encoded_reference_latents = encoded_reference_latents
        trace: list[dict[str, Any]] = []
        original_sample = kernel.sample_audio_latent
        original_decode = kernel.decode_audio_token
        original_negative_forward = model._negative_kv_branch.forward_step
        negative_inputs_for_trace: list[Any] = []

        def traced_negative_forward(
            request_ids: list[str],
            input_embeddings: list[Any],
        ) -> list[Any]:
            conditions = original_negative_forward(
                request_ids,
                input_embeddings,
            )
            if len(request_ids) != 1 or len(input_embeddings) != 1:
                raise RuntimeError("VibeVoice golden trace requires one active negative request")
            negative_inputs_for_trace.append(input_embeddings[0].detach().cpu())
            return conditions

        model._negative_kv_branch.forward_step = traced_negative_forward

        def traced_sample(
            positive_condition: Any,
            negative_condition: Any,
            noise: Any,
            **kwargs: Any,
        ) -> Any:
            step = len(trace)
            deterministic_noise = (
                torch.linspace(
                    -1.0,
                    1.0,
                    128,
                    dtype=torch.float32,
                    device=noise.device,
                )
                .reshape(2, 64)
                .add_(step * 0.03125)
                .to(noise)
            )
            if tuple(noise.shape) != tuple(deterministic_noise.shape):
                raise RuntimeError(
                    f"VibeVoice golden trace expected a single-request noise draw, got {tuple(noise.shape)}"
                )
            latent = original_sample(
                positive_condition,
                negative_condition,
                deterministic_noise,
                **kwargs,
            )
            negative_input = negative_inputs_for_trace.pop(0) if negative_inputs_for_trace else None
            if negative_input is None:
                raise RuntimeError("VibeVoice golden trace is missing the negative input")
            trace.append(
                {
                    "negative_input_embedding": negative_input,
                    "positive_condition": positive_condition.detach().cpu(),
                    "negative_condition": negative_condition.detach().cpu(),
                    "noise": deterministic_noise.detach().cpu(),
                    "audio_latent": latent.detach().cpu(),
                }
            )
            return latent

        def traced_decode(audio_latent: Any, **kwargs: Any) -> Any:
            decoded = original_decode(audio_latent, **kwargs)
            if len(trace) < 1:
                raise RuntimeError("VibeVoice decode ran before traced diffusion")
            trace[-1].update(
                {
                    "audio": decoded.audio.detach().cpu(),
                    "semantic_latent": decoded.semantic_latent.detach().cpu(),
                    "next_embedding": decoded.next_embedding.detach().cpu(),
                }
            )
            return decoded

        kernel.sample_audio_latent = traced_sample
        kernel.decode_audio_token = traced_decode
        self._vibevoice_generation_trace = trace
        self._vibevoice_generation_trace_originals = (
            model,
            original_audio_encode,
            kernel,
            original_sample,
            original_decode,
            original_negative_forward,
        )
        return {
            "rank": int(self.rank),
            "armed": True,
            "seed": int(torch.initial_seed()),
            "encoded_reference_latents": encoded_reference_latents,
        }

    def vibevoice_test_write_generation_trace(
        self,
        output_dir: str,
    ) -> dict[str, Any]:
        """Persist one rank's trace without routing tensors through RPC."""
        import torch

        trace = getattr(self, "_vibevoice_generation_trace", None)
        originals = getattr(
            self,
            "_vibevoice_generation_trace_originals",
            None,
        )
        if trace is None or originals is None:
            raise RuntimeError("VibeVoice generation trace is not armed")
        (
            model,
            original_audio_encode,
            kernel,
            original_sample,
            original_decode,
            original_negative_forward,
        ) = originals
        model.model.audio_tower.encode = original_audio_encode
        kernel.sample_audio_latent = original_sample
        kernel.decode_audio_token = original_decode
        model._negative_kv_branch.forward_step = original_negative_forward
        path = Path(output_dir) / f"omni-rank-{self.rank}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        trace_payload = {
            "steps": trace,
            "encoded_reference_latents": list(model._vibevoice_test_encoded_reference_latents),
        }
        torch.save(trace_payload, temporary)
        os.replace(temporary, path)
        del model._vibevoice_test_encoded_reference_latents
        del self._vibevoice_generation_trace
        del self._vibevoice_generation_trace_originals
        return {
            "rank": int(self.rank),
            "path": str(path),
            "num_steps": len(trace),
        }

    def vibevoice_test_arm_negative_fault(
        self,
        fail_on_call: int,
        shutdown_report_dir: str,
    ) -> dict[str, Any]:
        if fail_on_call < 1:
            raise ValueError("fail_on_call must be positive")

        runner = self.model_runner
        model = runner.get_model()
        negative_branch = model._negative_kv_branch
        if negative_branch is None:
            raise RuntimeError("VibeVoice negative branch is not bound")

        language_model = negative_branch.language_model
        original_forward = language_model.forward
        call_count = 0

        def fail_once(*args: Any, **kwargs: Any):
            nonlocal call_count
            call_count += 1
            if call_count == fail_on_call:
                raise RuntimeError("injected VibeVoice negative forward failure")
            return original_forward(*args, **kwargs)

        language_model.forward = fail_once

        original_shutdown = runner.shutdown
        model_ref = weakref.ref(model)
        branch = runner.named_kv_branches["negative"]
        report_path = Path(shutdown_report_dir) / f"rank-{self.rank}.json"

        def shutdown_with_report() -> None:
            pre_shutdown = _runner_snapshot(self)
            original_shutdown()
            gc.collect()
            _write_json_atomic(
                report_path,
                {
                    "rank": int(self.rank),
                    "pre_shutdown": pre_shutdown,
                    "post_shutdown": {
                        "model_collected": model_ref() is None,
                        "runner_named_branches_empty": not runner.named_kv_branches,
                        "branch_closed": bool(branch._closed),
                        "branch_entered": bool(branch._entered),
                        "branch_request_ids": sorted(branch._states),
                        "num_blocks": int(branch.num_blocks),
                        "num_free_blocks": int(branch.num_free_blocks),
                        "kv_caches_empty": not branch.kv_caches,
                        "raw_caches_empty": not branch._raw_caches,
                    },
                },
            )

        runner.shutdown = shutdown_with_report
        return {
            "rank": int(self.rank),
            "armed": True,
            "fail_on_call": int(fail_on_call),
            "report_path": str(report_path),
        }


__all__ = ["VibeVoiceWorkerExtensionForTest"]
