import torch, json
import fnmatch
import os
from ..core import ModelConfig, load_state_dict
from ..utils.controlnet import ControlNetInput
from peft import LoraConfig, inject_adapter_in_model


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        if isinstance(target_modules, list) and len(target_modules) == 1:
            target_modules = target_modules[0]
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict
    
    
    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        if data is None:
            return data
        elif isinstance(data, torch.Tensor):
            data = data.to(device)
            if torch_float_dtype is not None and data.dtype in [torch.float, torch.float16, torch.bfloat16]:
                data = data.to(torch_float_dtype)
            return data
        elif isinstance(data, tuple):
            data = tuple(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
            return data
        elif isinstance(data, list):
            data = list(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
            return data
        elif isinstance(data, dict):
            data = {i: self.transfer_data_to_device(data[i], device, torch_float_dtype) for i in data}
            return data
        else:
            return data
    
    def parse_vram_config(self, fp8=False, offload=False, device="cpu"):
        if fp8:
            return {
                "offload_dtype": torch.float8_e4m3fn,
                "offload_device": device,
                "onload_dtype": torch.float8_e4m3fn,
                "onload_device": device,
                "preparing_dtype": torch.float8_e4m3fn,
                "preparing_device": device,
                "computation_dtype": torch.bfloat16,
                "computation_device": device,
            }
        elif offload:
            return {
                "offload_dtype": "disk",
                "offload_device": "disk",
                "onload_dtype": "disk",
                "onload_device": "disk",
                "preparing_dtype": torch.bfloat16,
                "preparing_device": device,
                "computation_dtype": torch.bfloat16,
                "computation_device": device,
                "clear_parameters": True,
            }
        else:
            return {}
    
    def parse_model_configs(self, model_paths, model_id_with_origin_paths, fp8_models=None, offload_models=None, device="cpu"):
        fp8_models = [] if fp8_models is None else fp8_models.split(",")
        offload_models = [] if offload_models is None else offload_models.split(",")
        model_configs = []
        if model_paths is not None:
            print("[调试] 正在处理 --model_paths 模式")
            model_paths = json.loads(model_paths)
            for path in model_paths:
                # 检查 offload
                matched_offload_pattern = None
                for pattern in offload_models:
                    if self._is_path_list_from_offload_pattern(path, pattern):
                        matched_offload_pattern = pattern
                        break
                should_offload = matched_offload_pattern is not None

                # 检查 fp8
                matched_fp8_pattern = None
                for pattern in fp8_models:
                    if self._is_path_list_from_offload_pattern(path, pattern):
                        matched_fp8_pattern = pattern
                        break
                should_fp8 = matched_fp8_pattern is not None

                # 调试输出
                print(f"  → 路径model_paths: {repr(path)}")
                print(f"  → 匹配的 offload pattern: {repr(matched_offload_pattern)}")
                print(f"  → 匹配的 fp8 pattern: {repr(matched_fp8_pattern)}")
                print(f"      是否启用 offload？{should_offload}")
                print(f"      是否启用 fp8？{should_fp8}")

                vram_config = self.parse_vram_config(
                    fp8=should_fp8,
                    offload=should_offload,
                    device=device
                )
                model_configs.append(ModelConfig(path=path, **vram_config))
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            for model_id_with_origin_path in model_id_with_origin_paths:
                model_id, origin_file_pattern = model_id_with_origin_path.split(":")
                # 注意：这里匹配的是完整的 "model_id:pattern" 字符串
                should_offload = model_id_with_origin_path in offload_models
                should_fp8 = model_id_with_origin_path in fp8_models

                # 调试输出（与上一分支对齐）
                print(f"  → model_id_with_origin_path: {repr(model_id_with_origin_path)}")
                print(f"  → 匹配的 offload pattern: {repr(model_id_with_origin_path if should_offload else None)}")
                print(f"  → 匹配的 fp8 pattern: {repr(model_id_with_origin_path if should_fp8 else None)}")
                print(f"      是否启用 offload？{should_offload}")
                print(f"      是否启用 fp8？{should_fp8}")

                vram_config = self.parse_vram_config(
                    fp8=should_fp8,
                    offload=should_offload,
                    device=device
                )
                model_configs.append(ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern, **vram_config))
        return model_configs
    
    
    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        task="sft",
    ):
        # Scheduler
        pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Preset LoRA
        if preset_lora_path is not None:
            pipe.load_lora(getattr(pipe, preset_lora_model), preset_lora_path)
        
        # FP8
        # FP8 relies on a model-specific memory management scheme.
        # It is delegated to the subclass.
        
        # Add LoRA to the base models
        if lora_base_model is not None and not task.endswith(":data_process"):
            if (not hasattr(pipe, lora_base_model)) or getattr(pipe, lora_base_model) is None:
                print(f"No {lora_base_model} models in the pipeline. We cannot patch LoRA on the model. If this occurs during the data processing stage, it is normal.")
                return
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(pipe, lora_base_model, model)


    def split_pipeline_units(self, task, pipe, trainable_models=None, lora_base_model=None):
        models_require_backward = []
        if trainable_models is not None:
            models_require_backward += trainable_models.split(",")
        if lora_base_model is not None:
            models_require_backward += [lora_base_model]
        if task.endswith(":data_process"):
            _, pipe.units = pipe.split_pipeline_units(models_require_backward)
        elif task.endswith(":train"):
            pipe.units, _ = pipe.split_pipeline_units(models_require_backward)
        return pipe
    
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        controlnet_keys_map = (
            ("blockwise_controlnet_", "blockwise_controlnet_inputs",),
            ("controlnet_", "controlnet_inputs"),
        )
        controlnet_inputs = {}
        for extra_input in extra_inputs:
            for prefix, name in controlnet_keys_map:
                if extra_input.startswith(prefix):
                    if name not in controlnet_inputs:
                        controlnet_inputs[name] = {}
                    controlnet_inputs[name][extra_input.replace(prefix, "")] = data[extra_input]
                    break
            else:
                inputs_shared[extra_input] = data[extra_input]
        for name, params in controlnet_inputs.items():
            inputs_shared[name] = [ControlNetInput(**params)]
        return inputs_shared



    def _is_path_list_from_offload_pattern(self, path_or_list, offload_pattern):
        if ":" not in offload_pattern:
            return False       
        
        base_repo, pattern_part = offload_pattern.split(":", 1)
        
        # 自动将 .pth 模式转为 .safetensors（兼容用户习惯）
        if pattern_part.endswith(".pth"):
            pattern_part = pattern_part[:-4] + ".safetensors"
        
        file_paths = [path_or_list] if isinstance(path_or_list, str) else path_or_list

        for fp in file_paths:
            filename = os.path.basename(fp.replace("\\", "/"))
            
            # 如果 pattern_part 包含 /，说明是相对路径模式，尝试匹配完整路径（但允许任意父目录）
            if "/" in pattern_part:
                # 提取 clean 路径（去掉 ./）
                clean_fp = fp.replace("\\", "/").lstrip("./")
                # 尝试：是否以 pattern_part 结尾？
                if clean_fp.endswith(pattern_part):
                    return True
                # 或者用 fnmatch 匹配整个路径（谨慎）
                if fnmatch.fnmatch(clean_fp, f"*{pattern_part}"):
                    return True
            else:
                # 只匹配文件名
                if fnmatch.fnmatch(filename, pattern_part):
                    return True
                    
        return False