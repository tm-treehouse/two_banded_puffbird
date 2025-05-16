#!/usr/bin/env python3
"""
Data models for options analysis.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class OptionType(Enum):
    """Enumeration of option types"""

    CALL = "call"
    PUT = "put"


@dataclass
class OptionContract:
    """Class representing an option contract with its key metrics"""

    ticker: str
    strike: float
    expiration: str  # ISO date string
    option_type: OptionType
    midpoint: float
    premium_collected: float  # premium in dollars
    capital_required: float
    return_on_capital: float  # as percentage
    return_on_capital_per_annum: float  # as percentage
    implied_volatility: float  # as percentage
    risk_adjusted_score: float
    delta: float  # Greek delta
    delta_x_iv: float  # delta * implied volatility
    days_to_exp: int
    composite_score: Optional[float] = None

    @classmethod
    def from_dataframe_row(cls, row, option_type):
        """Create an OptionContract instance from a DataFrame row"""
        expiry_date = (
            row["expiration"]
            if isinstance(row["expiration"], str)
            else row["expiration"].strftime("%Y-%m-%d")
        )
        return cls(
            ticker=row["ticker"],
            strike=float(row["strike"]),
            expiration=expiry_date,
            option_type=option_type,
            midpoint=float(row["midpoint"]),
            premium_collected=float(row["premium_collected"]),
            capital_required=float(row["capital_required"]),
            return_on_capital=float(row["return_on_capital_%"]),
            return_on_capital_per_annum=float(row["return_on_capital_per_anum_%"]),
            implied_volatility=float(row["implied_volatility"]),
            risk_adjusted_score=float(row["risk_adjusted_score"]),
            delta=float(row["delta"]),
            delta_x_iv=float(row["delta_x_iv"]),
            days_to_exp=int(row["days_to_exp"]),
            composite_score=float(row.get("composite_score", 0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the option contract to a dictionary"""
        return {
            "ticker": self.ticker,
            "strike": self.strike,
            "expiration": self.expiration,
            "option_type": self.option_type.value,
            "midpoint": self.midpoint,
            "premium_collected": self.premium_collected,
            "capital_required": self.capital_required,
            "return_on_capital_%": self.return_on_capital,
            "return_on_capital_per_anum_%": self.return_on_capital_per_annum,
            "implied_volatility": self.implied_volatility,
            "risk_adjusted_score": self.risk_adjusted_score,
            "delta": self.delta,
            "delta_x_iv": self.delta_x_iv,
            "days_to_exp": self.days_to_exp,
            "composite_score": self.composite_score if self.composite_score is not None else 0,
        }
