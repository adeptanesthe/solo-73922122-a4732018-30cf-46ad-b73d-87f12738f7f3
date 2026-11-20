"""请求/响应数据模型与输入校验。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MIN_WINDOW = 1
MAX_WINDOW = 1440
SHIFT_MAX_WINDOW = 120
MIN_DURATION = 1
MAX_DURATION = 30


class DeviceSpec(BaseModel):
    """单台设备的标定参数。"""

    name: str = Field(..., min_length=1, max_length=64, description="设备名称")
    on_power: float = Field(..., description="开机时该设备贡献的功率")
    off_power: float = Field(..., description="停机时该设备贡献的功率（待机功率）")
    min_on_minutes: int = Field(
        ..., ge=MIN_DURATION, le=MAX_DURATION, description="最短开机分钟数（1-30）"
    )
    min_off_minutes: int = Field(
        ..., ge=MIN_DURATION, le=MAX_DURATION, description="最短停机分钟数（1-30）"
    )

    @field_validator("on_power", "off_power")
    @classmethod
    def _finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("功率必须为有限数值")
        return v


class InferenceRequest(BaseModel):
    """推断接口请求体。"""

    power: list[float | None] = Field(
        ...,
        description=(
            "逐分钟总功率读数，长度 1-1440（至多一天）；"
            "null 表示该分钟缺测，缺测仍计入开停时长"
        ),
    )
    noise_std: float = Field(..., gt=0, description="读数噪声标准差（正值）")
    device1: DeviceSpec
    device2: DeviceSpec
    confidence_threshold: float = Field(
        0.95, gt=0.5, lt=1.0, description="置信度阈值，必须大于 0.5"
    )
    switch_probability: float = Field(
        0.1,
        gt=0.0,
        lt=1.0,
        description="单台设备单分钟状态翻转的先验概率（0-1）",
    )

    @field_validator("power")
    @classmethod
    def _check_power_window(cls, v: list[float | None]) -> list[float | None]:
        if not (MIN_WINDOW <= len(v) <= MAX_WINDOW):
            raise ValueError("power 长度必须在 1 到 1440 分钟之间（至多一天）")
        for x in v:
            if x is not None and (x != x or x in (float("inf"), float("-inf"))):
                raise ValueError("功率读数必须为有限数值或 null")
        return v

    @model_validator(mode="after")
    def _check_names(self) -> "InferenceRequest":
        if self.device1.name == self.device2.name:
            raise ValueError("两台设备的 name 不能相同")
        return self


class Interval(BaseModel):
    state: Literal["running", "stopped", "uncertain"]
    start_minute: int = Field(..., description="区间起始分钟（含，0 基）")
    end_minute: int = Field(..., description="区间结束分钟（不含，0 基）")
    duration_minutes: int
    missing_minutes: int = Field(..., description="区间内缺测分钟数")
    confidence: float = Field(
        ..., description="区间置信度（取区间内逐分钟置信度的最小值）"
    )
    truncated: bool = Field(..., description="区间是否在观测窗口首/尾被截断")
    truncated_edges: list[Literal["start", "end"]]


class DeviceResult(BaseModel):
    name: str
    posterior_on: list[float] = Field(
        ..., description="逐分钟处于开机状态的后验概率"
    )
    intervals: list[Interval]
    running_minutes: int
    stopped_minutes: int
    uncertain_minutes: int


class InferenceResponse(BaseModel):
    window_minutes: int
    missing_minutes: int
    confidence_threshold: float
    log_likelihood: float
    devices: list[DeviceResult]


class ShiftDeviceSpec(BaseModel):
    """班次核算中单台设备的标定参数（开/停共用同一最短时长）。"""

    name: str = Field(..., min_length=1, max_length=64, description="设备名称")
    on_power: float = Field(..., description="开机功率（千瓦）")
    standby_power: float = Field(..., description="待机（停机）功率（千瓦）")
    min_duration_minutes: int = Field(
        ...,
        ge=MIN_DURATION,
        le=MAX_DURATION,
        description="最短开/停分钟数（1-30），开机与停机共用",
    )

    @field_validator("on_power", "standby_power")
    @classmethod
    def _finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("功率必须为有限数值")
        return v


class ShiftEnergyRequest(BaseModel):
    """班次用电核算请求体。"""

    power: list[float | None] = Field(
        ...,
        description=(
            "逐分钟总功率读数（千瓦），长度 1-120；"
            "null 表示该分钟缺测，缺测分钟照常计入电量"
        ),
    )
    noise_std: float = Field(..., gt=0, description="读数噪声标准差（正值）")
    device1: ShiftDeviceSpec
    device2: ShiftDeviceSpec
    target_device: int = Field(
        ..., ge=1, le=2, description="核算对象：1 = device1，2 = device2"
    )
    energy_limit_kwh: float = Field(
        ...,
        ge=0.0,
        description="电量上限（千瓦时）；恰好达到上限不算超额",
    )
    switch_probability: float = Field(
        0.1,
        gt=0.0,
        lt=1.0,
        description="单台设备单分钟状态翻转的先验概率（0-1）",
    )

    @field_validator("power")
    @classmethod
    def _check_power_window(cls, v: list[float | None]) -> list[float | None]:
        if not (MIN_WINDOW <= len(v) <= SHIFT_MAX_WINDOW):
            raise ValueError("power 长度必须在 1 到 120 分钟之间（一个班次）")
        for x in v:
            if x is not None and (x != x or x in (float("inf"), float("-inf"))):
                raise ValueError("功率读数必须为有限数值或 null")
        return v

    @field_validator("energy_limit_kwh")
    @classmethod
    def _finite_limit(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("电量上限必须为有限数值")
        return v

    @model_validator(mode="after")
    def _check_names(self) -> "ShiftEnergyRequest":
        if self.device1.name == self.device2.name:
            raise ValueError("两台设备的 name 不能相同")
        return self


class ShiftEnergyResponse(BaseModel):
    window_minutes: int = Field(..., description="核算窗口长度（分钟）")
    missing_minutes: int = Field(..., description="缺测分钟数（照常计入电量）")
    target_device: int = Field(..., description="核算对象：1 = device1，2 = device2")
    target_name: str = Field(..., description="核算对象设备名称")
    energy_limit_kwh: float = Field(..., description="电量上限（千瓦时）")
    exceedance_probability: float = Field(
        ...,
        description=(
            "目标设备班次用电超过上限的后验概率；按所有合法联合开停路径的"
            "后验权重汇总，开机与待机功率都计入电量，恰好达到上限不算超额"
        ),
    )
    expected_energy_kwh: float = Field(..., description="目标设备用电量的后验期望（千瓦时）")
    expected_on_minutes: float = Field(..., description="目标设备开机分钟数的后验期望")
    min_possible_energy_kwh: float = Field(..., description="窗口内理论最小电量（千瓦时）")
    max_possible_energy_kwh: float = Field(..., description="窗口内理论最大电量（千瓦时）")
    log_likelihood: float = Field(..., description="观测序列在该模型下的对数似然")
