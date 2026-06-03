"""
Asian Option Box-Muller Pricer

Python implementation of Asian option pricing using
Box-Muller Gaussian random number generation.

"""

import math
import random
from typing import Tuple


class AsianOptionBoxMullerPricer:
    """
    Asian option pricing engine using a Box-Muller Gaussian generator.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def get_gaussian_by_box_muller(self) -> float:
        """
        Generate a standard normal random variable using the Box-Muller method.
        """
        while True:
            x = 2.0 * self.rng.random() - 1.0
            y = 2.0 * self.rng.random() - 1.0
            radius_squared = x * x + y * y

            if radius_squared != 0.0 and radius_squared <= 1.0:
                break

        scale = math.sqrt(
            -2.0 * math.log(radius_squared) / radius_squared
        )

        return x * scale

    def simulate_asian_option_price(
        self,
        stock_price: float,
        strike_price: float,
        maturity: float,
        risk_free_rate: float,
        volatility: float,
        simulations: int
    ) -> Tuple[float, float]:
        """
        Estimate Asian call and put prices using a lognormal average-price approximation.
        """
        payoff_sum_call = 0.0
        payoff_sum_put = 0.0

        drift = (
            risk_free_rate
            - 0.5 * volatility * volatility
        ) * maturity

        diffusion_scale = volatility * math.sqrt(maturity / 3.0)
        discount_factor = math.exp(-risk_free_rate * maturity)

        for _ in range(simulations):
            gaussian = self.get_gaussian_by_box_muller()
            z = drift + diffusion_scale * gaussian

            average_stock_price = math.exp(
                math.log(stock_price) + z
            )

            payoff_sum_call += discount_factor * max(
                average_stock_price - strike_price,
                0.0
            )

            payoff_sum_put += discount_factor * max(
                strike_price - average_stock_price,
                0.0
            )

        call_price = payoff_sum_call / simulations
        put_price = payoff_sum_put / simulations

        return call_price, put_price


def main() -> None:
    print("=" * 72)
    print("Asian Option Box-Muller Pricer")
    print("=" * 72)

    stock_price = 100.0
    strike_price = 100.0
    maturity = 1.0
    risk_free_rate = 0.05
    volatility = 0.2
    simulations = 10000

    pricer = AsianOptionBoxMullerPricer(seed=42)

    call_price, put_price = pricer.simulate_asian_option_price(
        stock_price,
        strike_price,
        maturity,
        risk_free_rate,
        volatility,
        simulations
    )

    print()
    print("Pricing Results")
    print("-" * 72)
    print(f"Estimated Asian Call Option Price: {call_price:.6f}")
    print(f"Estimated Asian Put Option Price: {put_price:.6f}")


if __name__ == "__main__":
    main()
