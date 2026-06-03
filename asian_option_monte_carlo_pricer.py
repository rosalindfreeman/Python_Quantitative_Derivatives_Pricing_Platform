"""
Asian Option Monte Carlo Pricer

Python implementation of Asian option pricing using Monte Carlo simulation.

"""

import math
import random
import statistics
from typing import Tuple


class AsianOptionMonteCarloPricer:
    """
    Monte Carlo pricing engine for Asian call and put options.

    The payoff depends on the average stock price across the simulated path.
    """

    def __init__(
        self,
        initial_stock_price: float,
        strike_price: float,
        maturity: float,
        risk_free_rate: float,
        volatility: float,
        simulations: int,
        steps: int,
        seed: int | None = None
    ) -> None:
        self.initial_stock_price = initial_stock_price
        self.strike_price = strike_price
        self.maturity = maturity
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.simulations = simulations
        self.steps = steps
        self.rng = random.Random(seed)

    def simulate_average_price_path(self) -> float:
        """
        Simulate one stock price path and return the arithmetic average price.
        """
        dt = self.maturity / self.steps
        stock_price = self.initial_stock_price
        stock_price_sum = 0.0

        for _ in range(self.steps):
            z = self.rng.gauss(0.0, 1.0)

            stock_price *= math.exp(
                (
                    self.risk_free_rate
                    - 0.5 * self.volatility * self.volatility
                ) * dt
                + self.volatility * math.sqrt(dt) * z
            )

            stock_price_sum += stock_price

        return stock_price_sum / self.steps

    def calculate_prices_and_errors(self) -> Tuple[float, float, float, float]:
        """
        Calculate Asian call and put prices with standard errors.
        """
        call_payoffs = []
        put_payoffs = []

        discount_factor = math.exp(-self.risk_free_rate * self.maturity)

        for _ in range(self.simulations):
            average_price = self.simulate_average_price_path()

            call_payoff = max(average_price - self.strike_price, 0.0)
            put_payoff = max(self.strike_price - average_price, 0.0)

            call_payoffs.append(call_payoff * discount_factor)
            put_payoffs.append(put_payoff * discount_factor)

        call_price = statistics.mean(call_payoffs)
        put_price = statistics.mean(put_payoffs)

        if self.simulations > 1:
            call_error = statistics.stdev(call_payoffs) / math.sqrt(self.simulations)
            put_error = statistics.stdev(put_payoffs) / math.sqrt(self.simulations)
        else:
            call_error = 0.0
            put_error = 0.0

        return call_price, put_price, call_error, put_error


def get_float(prompt: str, default: float) -> float:
    value = input(f"{prompt} [{default}]: ").strip()
    return default if value == "" else float(value)


def get_int(prompt: str, default: int) -> int:
    value = input(f"{prompt} [{default}]: ").strip()
    return default if value == "" else int(value)


def main() -> None:
    print("=" * 72)
    print("Asian Option Monte Carlo Pricer")
    print("=" * 72)
    print("Press Enter to use the default value.")
    print()

    stock_price = get_float("Initial stock price S", 100.0)
    strike_price = get_float("Strike price K", 100.0)
    maturity = get_float("Time to maturity T in years", 1.0)
    risk_free_rate = get_float("Risk-free rate r", 0.05)
    volatility = get_float("Volatility sigma", 0.2)
    simulations = get_int("Number of Monte Carlo simulations", 10000)
    steps = get_int("Number of time steps per simulation", 252)

    pricer = AsianOptionMonteCarloPricer(
        stock_price,
        strike_price,
        maturity,
        risk_free_rate,
        volatility,
        simulations,
        steps,
        seed=42
    )

    call_price, put_price, call_error, put_error = pricer.calculate_prices_and_errors()

    print()
    print("Pricing Results")
    print("-" * 72)
    print(f"Asian Call Option Price: {call_price:.6f}")
    print(f"Asian Put Option Price: {put_price:.6f}")
    print(f"Standard Error for Call: {call_error:.6f}")
    print(f"Standard Error for Put: {put_error:.6f}")


if __name__ == "__main__":
    main()
