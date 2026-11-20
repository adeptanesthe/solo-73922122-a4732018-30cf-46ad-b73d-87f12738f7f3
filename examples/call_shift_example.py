"""班次用电核算调用示例：请求目标设备的用电超额概率。

运行前提：服务已通过 docker compose up 启动（或本地 uvicorn 运行在 8000 端口）。
用法：  python examples/call_shift_example.py
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

URL = "http://127.0.0.1:8000/shift/energy-audit"


def main() -> None:
    payload = json.loads(
        (Path(__file__).parent / "shift_request.json").read_text(encoding="utf-8")
    )
    resp = httpx.post(URL, json=payload, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()

    print(f"窗口长度: {data['window_minutes']} 分钟，缺测: {data['missing_minutes']} 分钟")
    print(f"核算对象: {data['target_name']} (device{data['target_device']})")
    print(f"电量上限: {data['energy_limit_kwh']} 千瓦时")
    print(f"超额概率: {data['exceedance_probability']:.4f}")
    print(f"期望用电: {data['expected_energy_kwh']} 千瓦时"
          f"（理论范围 {data['min_possible_energy_kwh']} ~ {data['max_possible_energy_kwh']}）")
    print(f"期望开机: {data['expected_on_minutes']} 分钟")


if __name__ == "__main__":
    main()
