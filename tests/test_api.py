"""FastAPI 接口层自测（含参数校验与端到端调用）。"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _payload(**overrides):
    base = {
        "power": [0.0] * 10 + [30.0] * 10 + [10.0] * 10,
        "noise_std": 0.8,
        "device1": {
            "name": "蒸饭柜",
            "on_power": 10.0,
            "off_power": 0.0,
            "min_on_minutes": 3,
            "min_off_minutes": 3,
        },
        "device2": {
            "name": "抽风机",
            "on_power": 20.0,
            "off_power": 0.0,
            "min_on_minutes": 3,
            "min_off_minutes": 3,
        },
        "confidence_threshold": 0.9,
    }
    base.update(overrides)
    return base


class TestApi(unittest.TestCase):
    def test_health(self):
        r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_infer_end_to_end(self):
        r = client.post("/infer", json=_payload())
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["window_minutes"], 30)
        self.assertEqual(data["missing_minutes"], 0)
        self.assertEqual(len(data["devices"]), 2)
        for dev in data["devices"]:
            self.assertEqual(len(dev["posterior_on"]), 30)
            covered = sum(i["duration_minutes"] for i in dev["intervals"])
            self.assertEqual(covered, 30)
            self.assertEqual(
                dev["running_minutes"] + dev["stopped_minutes"] + dev["uncertain_minutes"],
                30,
            )
            for iv in dev["intervals"]:
                self.assertIn(iv["state"], ("running", "stopped", "uncertain"))
                self.assertGreaterEqual(iv["confidence"], 0.0)

    def test_missing_values_accepted(self):
        power = [0.0] * 10 + [None] * 5 + [30.0] * 15
        r = client.post("/infer", json=_payload(power=power))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["missing_minutes"], 5)

    def test_threshold_must_be_above_half(self):
        r = client.post("/infer", json=_payload(confidence_threshold=0.5))
        self.assertEqual(r.status_code, 422)
        r = client.post("/infer", json=_payload(confidence_threshold=0.49))
        self.assertEqual(r.status_code, 422)

    def test_noise_std_positive(self):
        r = client.post("/infer", json=_payload(noise_std=0))
        self.assertEqual(r.status_code, 422)

    def test_minutes_range_validation(self):
        bad_device = {
            "name": "坏设备",
            "on_power": 1.0,
            "off_power": 0.0,
            "min_on_minutes": 0,
            "min_off_minutes": 31,
        }
        payload = _payload()
        payload["device1"] = bad_device
        r = client.post("/infer", json=payload)
        self.assertEqual(r.status_code, 422)

    def test_window_too_long(self):
        r = client.post("/infer", json=_payload(power=[1.0] * 1441))
        self.assertEqual(r.status_code, 422)

    def test_duplicate_names_rejected(self):
        payload = _payload()
        payload["device2"]["name"] = payload["device1"]["name"]
        r = client.post("/infer", json=payload)
        self.assertEqual(r.status_code, 422)

    def test_openapi_docs_available(self):
        r = client.get("/openapi.json")
        self.assertEqual(r.status_code, 200)
        self.assertIn("/infer", r.json()["paths"])
        self.assertIn("/shift/energy-audit", r.json()["paths"])


def _shift_payload(**overrides):
    base = {
        # 60 分钟班次：前 20 分钟仅设备1开机（10 kW），其余时间全部停机
        "power": [10.0] * 20 + [0.0] * 40,
        "noise_std": 0.5,
        "device1": {
            "name": "蒸饭柜",
            "on_power": 10.0,
            "standby_power": 0.0,
            "min_duration_minutes": 3,
        },
        "device2": {
            "name": "抽风机",
            "on_power": 20.0,
            "standby_power": 0.0,
            "min_duration_minutes": 3,
        },
        "target_device": 1,
        "energy_limit_kwh": 2.0,
    }
    base.update(overrides)
    return base


class TestShiftEnergyApi(unittest.TestCase):
    def test_end_to_end(self):
        r = client.post("/shift/energy-audit", json=_shift_payload())
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["window_minutes"], 60)
        self.assertEqual(data["missing_minutes"], 0)
        self.assertEqual(data["target_device"], 1)
        self.assertEqual(data["target_name"], "蒸饭柜")
        self.assertEqual(data["energy_limit_kwh"], 2.0)
        self.assertGreaterEqual(data["exceedance_probability"], 0.0)
        self.assertLessEqual(data["exceedance_probability"], 1.0)
        self.assertGreaterEqual(data["expected_energy_kwh"], data["min_possible_energy_kwh"])
        self.assertLessEqual(data["expected_energy_kwh"], data["max_possible_energy_kwh"])
        self.assertGreaterEqual(data["expected_on_minutes"], 0.0)
        self.assertLessEqual(data["expected_on_minutes"], 60.0)

    def test_clear_signal_threshold_behavior(self):
        # 设备1清晰开机 20 分钟（10 kW → 约 3.33 kWh）：
        # 上限 2.0 kWh 时超额概率应接近 1，上限 5.0 kWh 时应接近 0。
        r = client.post("/shift/energy-audit", json=_shift_payload())
        self.assertGreater(r.json()["exceedance_probability"], 0.999)
        r = client.post(
            "/shift/energy-audit", json=_shift_payload(energy_limit_kwh=5.0)
        )
        self.assertLess(r.json()["exceedance_probability"], 0.001)
        # 上限恰好等于真实电量（10*20/60）时不算超额
        r = client.post(
            "/shift/energy-audit",
            json=_shift_payload(energy_limit_kwh=10.0 * 20 / 60.0),
        )
        self.assertLess(r.json()["exceedance_probability"], 0.001)

    def test_target_device_2(self):
        # 设备2全程停机，上限 0.05 kWh：待机 0，几乎不可能超额
        r = client.post(
            "/shift/energy-audit",
            json=_shift_payload(target_device=2, energy_limit_kwh=0.05),
        )
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["target_name"], "抽风机")
        self.assertLess(data["exceedance_probability"], 0.001)

    def test_missing_minutes_accepted_and_counted(self):
        power = [10.0] * 20 + [None] * 5 + [0.0] * 35
        r = client.post("/shift/energy-audit", json=_shift_payload(power=power))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["missing_minutes"], 5)

    def test_window_over_120_rejected(self):
        r = client.post(
            "/shift/energy-audit", json=_shift_payload(power=[1.0] * 121)
        )
        self.assertEqual(r.status_code, 422)

    def test_min_duration_bounds(self):
        for bad in (0, 31):
            payload = _shift_payload()
            payload["device1"]["min_duration_minutes"] = bad
            r = client.post("/shift/energy-audit", json=payload)
            self.assertEqual(r.status_code, 422, f"min_duration={bad}")

    def test_target_device_bounds(self):
        for bad in (0, 3):
            r = client.post(
                "/shift/energy-audit", json=_shift_payload(target_device=bad)
            )
            self.assertEqual(r.status_code, 422, f"target_device={bad}")

    def test_negative_limit_rejected(self):
        r = client.post(
            "/shift/energy-audit", json=_shift_payload(energy_limit_kwh=-1.0)
        )
        self.assertEqual(r.status_code, 422)

    def test_noise_std_positive(self):
        r = client.post("/shift/energy-audit", json=_shift_payload(noise_std=0))
        self.assertEqual(r.status_code, 422)

    def test_duplicate_names_rejected(self):
        payload = _shift_payload()
        payload["device2"]["name"] = payload["device1"]["name"]
        r = client.post("/shift/energy-audit", json=payload)
        self.assertEqual(r.status_code, 422)


if __name__ == "__main__":
    unittest.main()
