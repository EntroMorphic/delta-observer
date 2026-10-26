import unittest

import torch

from research.lfm.observer_d_lm import ObserverDLM, ObserverDLMConfig


class ObserverDLMTests(unittest.TestCase):
    @staticmethod
    def tiny_model(repetitions=3):
        return ObserverDLM(ObserverDLMConfig(
            vocab_size=101,
            width=24,
            repetitions=repetitions,
            attention_every=1,
            attention_heads=4,
            expansion=2,
            max_context=32,
        )).eval()

    def test_language_head_is_tied(self):
        model = self.tiny_model()
        self.assertEqual(model.output_weight.data_ptr(), model.token_embedding.weight.data_ptr())

    def test_shapes_and_shared_depth_cell(self):
        model = self.tiny_model(repetitions=4)
        output = model(torch.randint(0, 101, (2, 7)), return_method_states=True)
        self.assertEqual(tuple(output["logits"].shape), (2, 7, 101))
        self.assertEqual(len(output["method_states"]), 4)
        method_parameter_names = [
            name for name, _ in model.named_parameters() if name.startswith("method_cell")
        ]
        self.assertFalse(any("layers" in name for name in method_parameter_names))

    def test_explicit_output_bias(self):
        config = self.tiny_model().config
        model = ObserverDLM(ObserverDLMConfig(**{
            **config.__dict__, "output_bias": True,
        })).eval()
        self.assertEqual(model.lm_head_bias.numel(), config.vocab_size)
        self.assertEqual(tuple(model(torch.ones(1, 3, dtype=torch.long))["logits"].shape), (1, 3, 101))

    def test_low_rank_output_head_is_untied(self):
        config = self.tiny_model().config
        model = ObserverDLM(ObserverDLMConfig(**{
            **config.__dict__, "output_head_rank": 8,
        })).eval()
        self.assertEqual(tuple(model.output_projection.weight.shape), (8, 24))
        self.assertEqual(tuple(model.output_embedding.weight.shape), (101, 8))
        self.assertEqual(tuple(model(torch.ones(1, 3, dtype=torch.long))["logits"].shape), (1, 3, 101))
        with self.assertRaises(RuntimeError):
            _ = model.output_weight

    def test_residual_scales_are_applied(self):
        config = self.tiny_model().config
        model = ObserverDLM(ObserverDLMConfig(**{
            **config.__dict__,
            "attention_residual_scale": 0.0,
            "ffn_residual_scale": 0.0,
        })).eval()
        states = torch.randn(2, 5, config.width)
        causal_mask = torch.ones(5, 5, dtype=torch.bool).triu(1)
        with torch.no_grad():
            attended = model.method_cell.global_attention(states, causal_mask)
            fed_forward = model.method_cell.feed_forward(states)
        self.assertTrue(torch.equal(attended, states))
        self.assertTrue(torch.equal(fed_forward, states))

    def test_zero_initialized_depth_adapters_preserve_output(self):
        torch.manual_seed(43)
        base = self.tiny_model()
        config = ObserverDLMConfig(**{
            **base.config.__dict__, "depth_adapter_rank": 8,
        })
        adapted = ObserverDLM(config).eval()
        incompatible = adapted.load_state_dict(base.state_dict(), strict=False)
        self.assertTrue(all(key.startswith("depth_adapters.") for key in incompatible.missing_keys))
        self.assertFalse(incompatible.unexpected_keys)
        input_ids = torch.randint(0, 101, (2, 7))
        with torch.no_grad():
            baseline = base(input_ids)["logits"]
            candidate = adapted(input_ids)["logits"]
        self.assertTrue(torch.equal(baseline, candidate))

    def test_factorized_interfaces_and_multiple_cells(self):
        config = ObserverDLMConfig(
            vocab_size=101,
            width=24,
            repetitions=4,
            attention_every=1,
            attention_heads=4,
            expansion=2,
            max_context=32,
            input_embedding_rank=8,
            output_head_rank=8,
            method_cell_count=3,
        )
        model = ObserverDLM(config).eval()
        self.assertEqual(tuple(model.token_embedding.weight.shape), (101, 8))
        self.assertEqual(tuple(model.input_projection.weight.shape), (24, 8))
        self.assertEqual(len(model.additional_method_cells), 2)
        self.assertIs(model.method_cell_for_depth(0), model.method_cell_for_depth(3))
        self.assertIsNot(model.method_cell_for_depth(0), model.method_cell_for_depth(1))
        output = model(torch.randint(0, 101, (2, 7)))
        self.assertEqual(tuple(output["logits"].shape), (2, 7, 101))

    def test_factorized_output_components_can_be_frozen_independently(self):
        model = ObserverDLM(ObserverDLMConfig(
            vocab_size=101,
            width=24,
            repetitions=2,
            attention_every=1,
            attention_heads=4,
            expansion=2,
            max_context=32,
            input_embedding_rank=8,
            output_head_rank=8,
        ))
        model.output_embedding.weight.requires_grad_(False)
        self.assertFalse(model.output_embedding.weight.requires_grad)
        self.assertTrue(model.output_projection.weight.requires_grad)
        model.output_projection.weight.requires_grad_(False)
        self.assertFalse(model.output_projection.weight.requires_grad)

    def test_future_tokens_do_not_change_past_logits(self):
        torch.manual_seed(23)
        model = self.tiny_model()
        first = torch.tensor([[1, 2, 3, 4, 5]])
        second = torch.tensor([[1, 2, 3, 90, 91]])
        with torch.no_grad():
            first_logits = model(first)["logits"][:, :3]
            second_logits = model(second)["logits"][:, :3]
        self.assertTrue(torch.allclose(first_logits, second_logits, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
