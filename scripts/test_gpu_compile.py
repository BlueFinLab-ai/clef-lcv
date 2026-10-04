"""Cross-compile representative FLA kernels; never launch target GPU code.

This developer check needs a visible CUDA GPU to allocate small tensor buffers.
It overrides the Triton compilation target and forces warmup-only kernel calls.
It proves compilation, not numerical correctness, autotuning or performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.runtime.autotuner import Autotuner
from triton.runtime.jit import JITFunction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", type=int, choices=[80, 89, 90, 120], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Maximum opt-in shared memory per block, rather than the SM total.
    limits = {80: 166912, 89: 101376, 90: 232448, 120: 101376}
    target = GPUTarget("cuda", args.arch, 32)
    records = {}
    original_run = JITFunction.run

    def compile_only(kernel_fn, *call_args, **kwargs):
        kwargs["warmup"] = True
        kernel = original_run(kernel_fn, *call_args, **kwargs)
        if kernel is not None:
            shared = kernel.metadata.shared
            if shared > limits[args.arch]:
                raise triton.runtime.errors.OutOfResources(shared, limits[args.arch], "shared memory")
            records[(kernel_fn.__name__, kernel.hash)] = {
                "kernel": kernel_fn.__name__, "shared_bytes": shared,
                "target": str(kernel.metadata.target),
                "cubin_sha256": hashlib.sha256(kernel.asm["cubin"]).hexdigest(),
            }
        return kernel

    def compile_config(tuner, *call_args, **kwargs):
        tuner.nargs = dict(zip(tuner.arg_names, call_args))
        # No target hardware exists: find a legal configuration without timing.
        failures = []
        for config in tuner.prune_configs(kwargs):
            try:
                result = tuner.fn.run(*call_args, **kwargs, **config.all_kwargs())
                tuner.best_config = config
                tuner.nargs = None
                return result
            except triton.runtime.errors.OutOfResources as exc:
                failures.append(str(exc))
        raise RuntimeError(f"No legal offline configuration: {failures}")

    active = triton.runtime.driver.active
    original_props = active.utils.get_device_properties

    def props(index):
        values = original_props(index).copy()
        values["max_shared_mem"] = limits[args.arch]
        return values

    with patch.object(active, "get_current_target", return_value=target), \
            patch.object(active.utils, "get_device_properties", side_effect=props), \
            patch.object(torch.cuda, "get_device_capability", return_value=(args.arch // 10, args.arch % 10)), \
            patch.object(torch.cuda, "get_device_name", return_value=f"offline SM{args.arch}"), \
            patch.object(JITFunction, "run", compile_only), \
            patch.object(Autotuner, "run", compile_config):
        # Import after the overrides so FLA selects the target's device policy.
        from fla.modules import FusedRMSNormGated
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
        with torch.inference_mode():
            for dtype in [torch.float16, torch.bfloat16]:
                for heads in [32, 48]:  # Flash / Full value-head counts after Q/K repetition.
                    for length in [128, 577]:
                        q, k, v = [torch.zeros(1, length, heads, 128, device="cuda", dtype=dtype)
                                   for _ in range(3)]
                        g = -torch.ones(1, length, heads, device="cuda")
                        beta = torch.ones_like(g).to(dtype)
                        common = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)
                        _, state = chunk_gated_delta_rule(q, k, v, g, beta, **common)
                        chunk_gated_delta_rule(q, k, v, g, beta, initial_state=state, **common)
                        fused_recurrent_gated_delta_rule(
                            q[:, -1:], k[:, -1:], v[:, -1:], g=g[:, -1:], beta=beta[:, -1:],
                            initial_state=state, **common,
                        )
                        norm = FusedRMSNormGated(128).cuda().to(dtype)
                        norm(q, q)
        required = {"fused_recurrent_gated_delta_rule_fwd_kernel", "layer_norm_gated_fwd_kernel",
                    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", "chunk_fwd_kernel_o"}
        assert required <= {row["kernel"] for row in records.values()}
        assert all(row["target"] == str(target) for row in records.values())
    result = {"architecture": args.arch, "compile_only": True, "execution_tested": False,
              "model_value_heads": [32, 48], "dtypes": ["float16", "bfloat16"],
              "specializations": len(records), "kernels": list(records.values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "kernels"}), flush=True)


if __name__ == "__main__":
    main()
