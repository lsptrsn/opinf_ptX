"""Physical system constraints and bounds."""
from dataclasses import dataclass

@dataclass
class PhysicalConstraints:
    """Physical constraints for the reactor system.

    Attributes:
        T_hot_max: Maximum allowable reactor temperature [K]
        u_min: Minimum coolant inlet temperature [K]
        u_max: Maximum coolant inlet temperature [K]
        ramp_up_K_per_min: Maximum heating rate [K/min]
        ramp_down_K_per_min: Maximum cooling rate [K/min]
    """
    T_hot_max: float = 700.0

    # We set both to 10.0 for now to ensure feasibility (symmetric behavior)
    ramp_up_K_per_min: float = 8.0 # 8
    ramp_down_K_per_min: float = 5.0 #5

    u_min: float = 200 + 273.15
    u_max: float = 400 + 273.15

    @property
    def ramp_up_K_per_s(self) -> float:
        """Heating ramp rate in K/s."""
        return self.ramp_up_K_per_min / 60.0

    @property
    def ramp_down_K_per_s(self) -> float:
        """Cooling ramp rate in K/s."""
        return self.ramp_down_K_per_min / 60.0

    def validate(self):
        """Validate constraint values."""
        assert self.u_min < self.u_max, "u_min must be less than u_max"
        assert self.T_hot_max > 0, "T_hot_max must be positive"
        assert self.ramp_up_K_per_min > 0, "ramp_up must be positive"
        assert self.ramp_down_K_per_min > 0, "ramp_down must be positive"
