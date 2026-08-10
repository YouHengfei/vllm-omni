"""Contracts between vLLM multimodal merging and Omni preprocess hooks."""

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_omni.worker.gpu_model_runner as runner_module
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Buffer:
    def __init__(self, tensor: torch.Tensor):
        self.gpu = tensor


class _CpuBuffer:
    def __init__(self, values: list[int]):
        self.cpu = np.asarray(values, dtype=np.int32)


class _CombinedPathModel:
    has_preprocess = True
    requires_raw_input_tokens = False

    def __init__(self, events: list[str], expected_mm_embedding: torch.Tensor):
        self.events = events
        self.expected_mm_embedding = expected_mm_embedding
        self.preprocess_input_embeds: torch.Tensor | None = None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        *,
        multimodal_embeddings: list[torch.Tensor],
        is_multimodal: torch.Tensor,
    ) -> torch.Tensor:
        self.events.append("merge")
        token_embeddings = torch.stack(
            (input_ids.to(torch.float32), -input_ids.to(torch.float32)), dim=-1
        )
        assert len(multimodal_embeddings) == 1
        token_embeddings[is_multimodal] = multimodal_embeddings[0]
        return token_embeddings

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info: object,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        self.events.append("preprocess")
        assert info["_omni_is_prefill"] is True
        assert input_embeds is not None
        self.preprocess_input_embeds = input_embeds.clone()
        torch.testing.assert_close(input_embeds[1], self.expected_mm_embedding)
        return input_ids, input_embeds + 1000, {"preprocess_was_called": True}


@contextmanager
def _encoder_connector_context(*args, **kwargs):
    yield None


def test_multimodal_embeddings_are_merged_before_omni_preprocess(monkeypatch):
    """The combined SupportsMultiModal + has_preprocess path is compositional."""
    monkeypatch.setattr(
        runner_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True),
    )

    events: list[str] = []
    mm_embedding = torch.tensor([91.0, 92.0])
    model = _CombinedPathModel(events, mm_embedding)

    runner = object.__new__(OmniGPUModelRunner)
    runner.model = model
    runner.model_config = SimpleNamespace(is_encoder_decoder=False, async_chunk=True)
    runner.vllm_config = SimpleNamespace(model_config=runner.model_config)
    runner.supports_mm_inputs = True
    runner.enable_prompt_embeds = False
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.has_talker_mtp = False
    runner.encoder_cache = {}

    input_ids = torch.tensor([10, 11, 12], dtype=torch.long)
    runner.input_ids = _Buffer(input_ids.clone())
    runner.inputs_embeds = _Buffer(torch.zeros((3, 2), dtype=torch.float32))
    runner.positions = torch.arange(3, dtype=torch.long)
    runner.query_start_loc = _CpuBuffer([0, 3])
    runner.input_batch = SimpleNamespace(
        req_ids=["request-0"],
        req_id_to_index={"request-0": 0},
        num_computed_tokens_cpu=np.asarray([0], dtype=np.int32),
    )
    runner.requests = {
        "request-0": SimpleNamespace(prompt_token_ids=input_ids.tolist()),
    }
    runner.model_intermediate_buffer = {}

    runner.maybe_get_ec_connector_output = _encoder_connector_context

    def execute_encoder(scheduler_output):
        events.append("encoder")
        return [mm_embedding]

    def gather_embeddings(scheduler_output):
        events.append("gather")
        return [mm_embedding], torch.tensor([False, True, False])

    runner._execute_mm_encoder = execute_encoder
    runner._gather_mm_embeddings = gather_embeddings
    runner._init_model_kwargs = lambda: {}
    runner._extract_mm_kwargs = lambda scheduler_output: {}
    runner._collect_additional_information_for_prefill = lambda counts: None
    runner._update_additional_information = lambda scheduler_output: None
    runner._maybe_run_batch_preprocess = lambda req_ids, device: None
    runner._maybe_attach_mimo_audio_req_infos = lambda req_state, infos, req_id: infos

    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=3,
        num_scheduled_tokens={"request-0": 3},
        scheduled_encoder_inputs={"request-0": [0]},
    )

    output_ids, output_embeds, *_ = OmniGPUModelRunner._preprocess(
        runner,
        scheduler_output,
        num_input_tokens=3,
    )

    assert events == ["encoder", "gather", "merge", "preprocess"]
    assert model.preprocess_input_embeds is not None
    torch.testing.assert_close(
        model.preprocess_input_embeds,
        torch.tensor([[10.0, -10.0], [91.0, 92.0], [12.0, -12.0]]),
    )
    torch.testing.assert_close(
        output_embeds,
        model.preprocess_input_embeds + 1000,
    )
    torch.testing.assert_close(output_ids, input_ids)
    assert runner.model_intermediate_buffer["request-0"]["preprocess_was_called"] is True
