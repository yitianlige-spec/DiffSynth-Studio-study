@echo off
REM ==============================
REM 双击启动 WanVideo LoRA 训练
REM 实际命令在 train_lora.py 中定义（带完整注释）
REM ==============================

cd /d G:\H\DiffSynth-Studio_env
if not exist Scripts\activate (
    echo ? 错误：虚拟环境不存在！
    pause
    exit /b 1
)
call Scripts\activate

cd /d G:\H\DiffSynth-Studio
if not exist examples\wanvideo\model_training\train_lora.py (
    echo ? 错误：找不到 train_lora.py！
    pause
    exit /b 1
)

echo ? 已激活虚拟环境，正在启动训练...
python examples\wanvideo\model_training\train_lora.py

if %errorlevel% neq 0 (
    echo ? 训练失败！
    pause
) else (
    echo ? 训练成功完成！
)
pause