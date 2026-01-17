import os
import subprocess
import sys
import json


def main():
    print("=" * 60)
    print("🚀 启动 accelerate 训练（通过 Python 构造命令）")
    print(f"📁 当前工作目录: {os.getcwd()}")
    print(f"🐍 Python: {sys.executable}")
    print("=" * 60)

    # ✅ 将 model_paths_list 定义在函数内部（无缩进错误）
    model_paths_list = [
        [
            "./models/Wan-AI/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
            "./models/Wan-AI/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors",
            "./models/Wan-AI/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors",
            "./models/Wan-AI/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors",
            "./models/Wan-AI/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors",
            "./models/Wan-AI/Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors"
        ],
        [
            "./models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
        ],
        "./models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors"
    ]
    # ✅ 转为紧凑 JSON 字符串（无空格、无多余逗号）
    model_paths_json = json.dumps(model_paths_list, separators=(',', ':'))

    # ========== 构造 accelerate 命令 ==========
    cmd1 = [
        "accelerate", "launch",
        "--config_file", r"C:\Users\yi\.cache\huggingface\accelerate\default_config.yaml",
        "examples/wanvideo/model_training/train.py",  # 真正的训练脚本

        # 数据集配置
        "--dataset_base_path",          "data/example_video_dataset",                     # 数据集的根目录。
        "--dataset_metadata_path",      "data/example_video_dataset/metadata.csv",       # 数据集的元数据文件路径。
        "--dataset_repeat",             "1",                                              # 每个 epoch 中数据集重复的次数。
        "--dataset_num_workers",        "0",                                              # 每个 Dataloader 的进程数量。
        # "--data_file_keys",           "video",                                          # 元数据中需要加载的字段名称，通常是图像或视频文件的路径，以 , 分隔。

        # 模型加载配置
        #"--model_paths",                model_paths_json,                                 # 本地模型要加载的模型路径。JSON 格式。
        "--model_id_with_origin_paths",
        "Wan-AI/Wan2.2-T2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,"
        "Wan-AI/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,"
        "Wan-AI/Wan2.2-T2V-A14B:Wan2.1_VAE.pth",                                        # 从 Hugging Face 获取模型，带原始路径的模型 ID，例如 "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors"。用逗号分隔。
        # "--extra_inputs",             "",                                               # 模型 Pipeline 所需的额外输入参数，例如训练图像编辑模型 Qwen-Image-Edit 时需要额外参数 edit_image，以 , 分隔。
        "--offload_models",             "Wan-AI/Wan2.2-T2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors",  # 带卸载的模型，以逗号分隔。仅用于分割训练。

        # 训练基础配置
        "--learning_rate",              "1e-4",                                           # 学习率。
        "--num_epochs",                 "5",                                              # 轮数（Epoch）。
        # "--trainable_models",         "dit",                                            # 可训练的模型，例如 dit、vae、text_encoder。
        # "--find_unused_parameters",                                                   # DDP 训练中是否存在未使用的参数，少数模型包含不参与梯度计算的冗余参数，需开启这一设置避免在多 GPU 训练中报错。
        # "--weight_decay",             "0.01",                                           # 权重衰减大小，详见 https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        "--task",                       "sft:data_process",                               # 训练任务，默认为 sft，部分模型支持更多训练模式，请参考每个特定模型的文档。

        # 输出配置
        "--output_path",                "./models/train/Wan2.2-T2V-A14B_high_noise_lora_cache",  # 模型保存路径。
        "--remove_prefix_in_ckpt",      "pipe.dit.",                                      # 在模型文件的 state dict 中移除前缀。
        # "--save_steps",               "1",                                              # 保存模型的训练步数间隔，若此参数留空，则每个 epoch 保存一次。

        # LoRA 配置
        "--lora_base_model",            "dit",                                            # LoRA 添加到哪个模型上。
        "--lora_target_modules",        "q,k,v,o,ffn.0,ffn.2",                            # LoRA 添加到哪些层上。
        "--lora_rank",                  "32",                                             # LoRA 的秩（Rank）。
        # "--lora_checkpoint",          "",                                               # LoRA 检查点的路径。如果提供此路径，LoRA 将从此检查点加载。
        # "--preset_lora_path",         "",                                               # 预置 LoRA 检查点路径，如果提供此路径，这一 LoRA 将会以融入基础模型的形式加载。此参数用于 LoRA 差分训练。
        # "--preset_lora_model",        "",                                               # 预置 LoRA 融入的模型，例如 dit。

        # 梯度配置
        "--use_gradient_checkpointing",                                                 # 是否启用 gradient checkpointing 用时间换显存。
        "--use_gradient_checkpointing_offload",                                         # 是否将 gradient checkpointing 卸载到内存中 用时间换显存。
        "--gradient_accumulation_steps", "4",                                           # 梯度累积步数 用时间换显存。

        # 分辨率 & 帧数
        "--height",                     "480",                                            # 图像或视频的高度。将 height 和 width 留空以启用动态分辨率。
        "--width",                      "832",                                            # 图像或视频的宽度。将 height 和 width 留空以启用动态分辨率。
        # "--max_pixels",               "1048576",                                        # 图像或视频帧的最大像素面积，当启用动态分辨率时，分辨率大于这个数值的图片都会被缩小，分辨率小于这个数值的图片保持不变。
        "--num_frames",                 "29",                                             # 使用视频的多少帧用于训练。

        # 噪声时间步边界
        "--max_timestep_boundary",      "0.417",                                          # 训练时采样噪声范围的下限（按比例计算，范围通常在 0 到 1 之间）。
        "--min_timestep_boundary",      "0",                                              # 训练时采样噪声范围的上限（按比例计算，范围通常在 0 到 1 之间）。
    ]
    print("🔧 构造的第一阶段命令:")
    print(" ".join(cmd1))

    # ========== 构造 accelerate 命令 ==========
    cmd2 = [
        "accelerate", "launch",
        "--config_file", r"C:\Users\yi\.cache\huggingface\accelerate\default_config.yaml",
        "examples/wanvideo/model_training/train.py",  # 真正的训练脚本

        # 数据集配置
        "--dataset_base_path",          "./models/train/Wan2.2-T2V-A14B_high_noise_lora_cache",  # 数据集的根目录。
        # "--dataset_metadata_path",    "data/example_video_dataset/metadata.csv",       # 数据集的元数据文件路径。
        "--dataset_repeat",             "100",                                            # 每个 epoch 中数据集重复的次数。
        "--dataset_num_workers",        "0",                                              # 每个 Dataloader 的进程数量。
        # "--data_file_keys",           "video",                                          # 元数据中需要加载的字段名称，通常是图像或视频文件的路径，以 , 分隔。

        # 模型加载配置
        #"--model_paths",                model_paths_json,                                 # 本地模型要加载的模型路径。JSON 格式。
        "--model_id_with_origin_paths",
        "Wan-AI/Wan2.2-T2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,"
        "Wan-AI/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,"
        "Wan-AI/Wan2.2-T2V-A14B:Wan2.1_VAE.pth",                                        # 从 Hugging Face 获取模型，带原始路径的模型 ID，例如 "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors"。用逗号分隔。
        #"--extra_inputs",             "",                                               # 模型 Pipeline 所需的额外输入参数，例如训练图像编辑模型 Qwen-Image-Edit 时需要额外参数 edit_image，以 , 分隔。
        "--offload_models",             "Wan-AI/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-T2V-A14B:Wan2.1_VAE.pth",  # 带卸载的模型，以逗号分隔。仅用于分割训练。

        # 训练基础配置
        "--learning_rate",              "1e-4",                                           # 学习率。
        "--num_epochs",                 "5",                                              # 轮数（Epoch）。
        # "--trainable_models",         "dit",                                            # 可训练的模型，例如 dit、vae、text_encoder。
        # "--find_unused_parameters",                                                   # DDP 训练中是否存在未使用的参数，少数模型包含不参与梯度计算的冗余参数，需开启这一设置避免在多 GPU 训练中报错。
        # "--weight_decay",             "0.01",                                           # 权重衰减大小，详见 https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        "--task",                       "sft:train",                                      # 训练任务，默认为 sft，部分模型支持更多训练模式，请参考每个特定模型的文档。

        # 输出配置
        "--output_path",                "./models/train/Wan2.2-T2V-A14B_high_noise_lora", # 模型保存路径。
        "--remove_prefix_in_ckpt",      "pipe.dit.",                                      # 在模型文件的 state dict 中移除前缀。
        # "--save_steps",               "1",                                              # 保存模型的训练步数间隔，若此参数留空，则每个 epoch 保存一次。

        # LoRA 配置
        "--lora_base_model",            "dit",                                            # LoRA 添加到哪个模型上。
        "--lora_target_modules",        "q,k,v,o,ffn.0,ffn.2",                            # LoRA 添加到哪些层上。
        "--lora_rank",                  "32",                                             # LoRA 的秩（Rank）。
        # "--lora_checkpoint",          "",                                               # LoRA 检查点的路径。如果提供此路径，LoRA 将从此检查点加载。
        # "--preset_lora_path",         "",                                               # 预置 LoRA 检查点路径，如果提供此路径，这一 LoRA 将会以融入基础模型的形式加载。此参数用于 LoRA 差分训练。
        # "--preset_lora_model",        "",                                               # 预置 LoRA 融入的模型，例如 dit。

        # 梯度配置
        "--use_gradient_checkpointing",                                                 # 是否启用 gradient checkpointing 用时间换显存。
        "--use_gradient_checkpointing_offload",                                         # 是否将 gradient checkpointing 卸载到内存中 用时间换显存。
        "--gradient_accumulation_steps", "1",                                           # 梯度累积步数 用时间换显存。

        # 分辨率 & 帧数
        "--height",                     "480",                                            # 图像或视频的高度。将 height 和 width 留空以启用动态分辨率。
        "--width",                      "832",                                            # 图像或视频的宽度。将 height 和 width 留空以启用动态分辨率。
        # "--max_pixels",               "1048576",                                        # 图像或视频帧的最大像素面积，当启用动态分辨率时，分辨率大于这个数值的图片都会被缩小，分辨率小于这个数值的图片保持不变。
        "--num_frames",                 "29",                                             # 使用视频的多少帧用于训练。

        # 噪声时间步边界
        "--max_timestep_boundary",      "0.417",                                          # 训练时采样噪声范围的下限（按比例计算，范围通常在 0 到 1 之间）。
        "--min_timestep_boundary",      "0",                                              # 训练时采样噪声范围的上限（按比例计算，范围通常在 0 到 1 之间）。
    ]

    print("🔧 构造的第二阶段命令:")
    print(" ".join(cmd2))

    # ========== 执行命令 ==========
    try:
        print("\n⏳ 开始执行第一阶段...\n")
        result = subprocess.run(cmd1, check=True)
        print("✅ 第一阶段成功结束！")
        print("\n⏳ 开始执行第二阶段...\n")
        result = subprocess.run(cmd2, check=True)
        print("✅ 第二阶段成功结束！")

    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败！退出码: {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("❌ 错误：未找到 'accelerate' 命令，请确保已安装 accelerate 并在虚拟环境中！")
        sys.exit(1)


if __name__ == "__main__":
    main()
