"""
FunASR 实时语音转写

支持两种模式：
1. 麦克风转写（默认）- 说话时实时出文字
2. 系统音频转写 - 转写电脑播放的声音（直播/视频）

Usage:
    # 麦克风模式
    python funasr_streaming.py

    # 系统音频模式（需要安装虚拟声卡或使用 WASAPI loopback）
    python funasr_streaming.py --mode system

    # 指定模型
    python funasr_streaming.py --model sensevoice

    # 列出可用音频设备
    python funasr_streaming.py --list-devices
"""

import argparse
import sys
import numpy as np
import sounddevice as sd
from funasr import AutoModel

CHUNK_MS = 600
SAMPLE_RATE = 16000


def list_devices():
    """列出所有音频设备"""
    print("可用音频设备:\n")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        kind = "🎤" if d["max_input_channels"] > 0 else "🔊"
        loopback = " [Loopback]" if "loopback" in d["name"].lower() else ""
        default = " (默认)" if i == sd.default.device[0] else ""
        print(f"  [{i:2d}] {kind} {d['name']}{loopback}{default}")
        print(f"       输入通道: {d['max_input_channels']}, 输出通道: {d['max_output_channels']}")
    print()


def find_loopback_device():
    """查找 WASAPI loopback 设备"""
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if "loopback" in d["name"].lower() and d["max_input_channels"] > 0:
            return i
    return None


def main():
    parser = argparse.ArgumentParser(description="FunASR 实时语音转写")
    parser.add_argument("--device", default="cuda", help="cuda, cpu, mps")
    parser.add_argument("--model", default="paraformer-zh-streaming", help="流式模型")
    parser.add_argument("--mode", default="mic", choices=["mic", "system"], help="mic=麦克风, system=系统音频")
    parser.add_argument("--audio-device", type=int, default=None, help="指定音频设备编号")
    parser.add_argument("--list-devices", action="store_true", help="列出可用音频设备")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    # 选择音频设备
    if args.audio_device is not None:
        device_index = args.audio_device
    elif args.mode == "system":
        device_index = find_loopback_device()
        if device_index is None:
            print("❌ 未找到 WASAPI Loopback 设备")
            print("请确保:")
            print("  1. Windows 10/11")
            print("  2. 音频设备支持 WASAPI Loopback")
            print("  3. 运行: python funasr_streaming.py --list-devices 查看设备列表")
            sys.exit(1)
    else:
        device_index = None  # 使用默认麦克风

    device_info = sd.query_devices(device_index, "input") if device_index else sd.query_devices(kind="input")
    print(f"录音设备: {device_info['name']}")
    print(f"采样率: {device_info['default_samplerate']}Hz")

    print(f"\n加载模型: {args.model} ({args.device})...")
    model = AutoModel(model=args.model, device=args.device)
    print("模型加载完成！\n")

    chunk_size = [0, 10, 5]
    chunk_stride = int(SAMPLE_RATE * CHUNK_MS / 1000)
    cache = {}

    mode_label = "🔊 系统音频" if args.mode == "system" else "🎤 麦克风"
    print(f"{mode_label} 转写中... (Ctrl+C 停止)\n")
    print("-" * 50)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=chunk_stride,
        device=device_index,
    ) as stream:
        while True:
            data, _ = stream.read(chunk_stride)
            chunk = data[:, 0]

            res = model.generate(
                input=chunk,
                cache=cache,
                is_final=False,
                chunk_size=chunk_size,
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1,
            )

            if res and res[0].get("text"):
                print(res[0]["text"], end="", flush=True)

    # Flush
    res = model.generate(
        input=np.array([], dtype=np.float32),
        cache=cache,
        is_final=True,
        chunk_size=chunk_size,
        encoder_chunk_look_back=4,
        decoder_chunk_look_back=1,
    )
    if res and res[0].get("text"):
        print(res[0]["text"])

    print("\n\n停止转写。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n停止转写。")
