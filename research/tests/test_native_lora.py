import unittest
import torch
from torch import nn

from research.lfm.native_lora import (
    LoRALinear, adapter_state, install_lora, load_adapter, supervised_chat,
)


class Tokenizer:
    eos_token_id = 7
    all_special_ids = [1, 7]

    def apply_chat_template(self, *args, **kwargs):
        return "prefix"

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [1, 8, 9] if text == "prefix" else [10, 11]


class NativeLoRATests(unittest.TestCase):
    def test_zero_init_and_frozen_base_and_reload(self):
        base = nn.Linear(4, 3)
        inputs = torch.randn(2, 4)
        expected = base(inputs).detach()
        model = LoRALinear(base, 2, 4)
        self.assertTrue(torch.equal(model(inputs), expected))
        before = base.weight.detach().clone()
        optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        model(inputs).square().mean().backward()
        self.assertIsNone(base.weight.grad)
        optim.step()
        self.assertTrue(torch.equal(base.weight, before))
        saved = adapter_state(model)
        output = model(inputs).detach()
        with torch.no_grad():
            model.lora_b.zero_()
        load_adapter(model, saved)
        self.assertTrue(torch.equal(model(inputs), output))
        with self.assertRaises(ValueError):
            load_adapter(model, {})

    def test_loss_mask_has_full_answer_and_eos_without_duplicate_bos(self):
        row = supervised_chat(Tokenizer(), "prompt", "response", 6)
        self.assertEqual(row["input_ids"], [1, 8, 9, 10, 11, 7])
        self.assertEqual(row["labels"], [-100, -100, -100, 10, 11, 7])
        self.assertEqual(row["supervised_tokens"], 3)
        with self.assertRaises(ValueError):
            supervised_chat(Tokenizer(), "prompt", "response", 5)

    def test_native_targets_exclude_head_and_ffn_and_reject_double_install(self):
        model = nn.Module()
        model.self_attn = nn.Module()  # root is intentionally not a native layer path
        model.model = nn.Module()
        model.model.layers = nn.ModuleList([nn.Module()])
        layer = model.model.layers[0]
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(4, 4)
        layer.feed_forward = nn.Module()
        layer.feed_forward.w1 = nn.Linear(4, 8)
        model.lm_head = nn.Linear(4, 10)
        targets = install_lora(model, 2, 4)
        self.assertEqual(targets, ["model.layers.0.self_attn.q_proj"])
        self.assertFalse(model.lm_head.weight.requires_grad)
        self.assertFalse(layer.feed_forward.w1.weight.requires_grad)
        with self.assertRaises(ValueError):
            install_lora(model)

    def test_adapter_load_rejects_nonfinite_without_partial_mutation(self):
        model = LoRALinear(nn.Linear(4, 3), 2, 4)
        state = adapter_state(model)
        before = adapter_state(model)
        state["lora_a"].fill_(1)
        state["lora_b"].fill_(float("nan"))
        with self.assertRaises(ValueError):
            load_adapter(model, state)
        for name, value in adapter_state(model).items():
            self.assertTrue(torch.equal(value, before[name]))
