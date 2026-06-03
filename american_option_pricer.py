"""
American Option Pricer

Professional Python implementation of:
- Black-Scholes European call pricing
- American put pricing using finite difference methods
- American call pricing using finite difference methods
- Delta and initial hedging cash calculation

No external packages are required.
"""

import math
from typing import List, Tuple


class AmericanOptionPricer:
    """
    Pricing engine for European and American options.

    This class contains analytical and numerical pricing methods used in
    quantitative finance and derivatives modelling.
    """

    @staticmethod
    def normal_cdf(x: float) -> float:
        """
        Standard normal cumulative distribution function.
        """
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def black_scholes_call_price(
        stock_price: float,
        strike_price: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float
    ) -> float:
        """
        Calculate the Black-Scholes analytical price of a European call option.
        """
        d1 = (
            math.log(stock_price / strike_price)
            + (risk_free_rate + 0.5 * volatility * volatility) * time_to_maturity
        ) / (volatility * math.sqrt(time_to_maturity))

        d2 = d1 - volatility * math.sqrt(time_to_maturity)

        call_price = (
            stock_price * AmericanOptionPricer.normal_cdf(d1)
            - strike_price
            * math.exp(-risk_free_rate * time_to_maturity)
            * AmericanOptionPricer.normal_cdf(d2)
        )

        return call_price

    @staticmethod
    def finite_difference_american_put(
        stock_price: float,
        strike_price: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
        price_grid_points: int,
        time_grid_points: int
    ) -> List[float]:
        """
        Price an American put option using a finite difference method.
        """
        dt = time_to_maturity / time_grid_points
        dx = volatility * math.sqrt(3.0 * dt)

        pu = 0.5 * dt * (
            volatility * volatility / (dx * dx)
            + (risk_free_rate - 0.5 * volatility * volatility) / dx
        )

        pm = 1.0 - dt * (
            volatility * volatility / (dx * dx)
            + risk_free_rate
        )

        pd = 0.5 * dt * (
            volatility * volatility / (dx * dx)
            - (risk_free_rate - 0.5 * volatility * volatility) / dx
        )

        max_index = 2 * price_grid_points
        prices = [0.0 for _ in range(max_index + 1)]

        for j in range(max_index + 1):
            stock = j * dx
            prices[j] = max(strike_price - stock, 0.0)

        for _ in range(time_grid_points - 1, -1, -1):
            old_prices = prices.copy()

            for j in range(1, max_index):
                continuation_value = (
                    pu * old_prices[j + 1]
                    + pm * old_prices[j]
                    + pd * old_prices[j - 1]
                )

                exercise_value = max(strike_price - j * dx, 0.0)
                prices[j] = max(continuation_value, exercise_value)

            prices[0] = strike_price
            prices[max_index] = 0.0

        return prices

    @staticmethod
    def finite_difference_american_call(
        stock_price: float,
        strike_price: float,
        time_to_maturity: float,
        risk_free_rate: float,
        volatility: float,
        price_grid_points: int,
        time_grid_points: int
    ) -> float:
        """
        Price an American call option using a finite difference method.
        """
        dt = time_to_maturity / time_grid_points
        dx = volatility * math.sqrt(3.0 * dt)

        pu = 0.5 * dt * (
            volatility * volatility / (dx * dx)
            + (risk_free_rate - 0.5 * volatility * volatility) / dx
        )

        pm = 1.0 - dt * (
            volatility * volatility / (dx * dx)
            + risk_free_rate
        )

        pd = 0.5 * dt * (
            volatility * volatility / (dx * dx)
            - (risk_free_rate - 0.5 * volatility * volatility) / dx
        )

        max_index = 2 * price_grid_points
        prices = [0.0 for _ in range(max_index + 1)]

        for j in range(max_index + 1):
            stock = j * dx
            prices[j] = max(stock - strike_price, 0.0)

        for _ in range(time_grid_points - 1, -1, -1):
            old_prices = prices.copy()

            for j in range(1, max_index):
                continuation_value = (
                    pu * old_prices[j + 1]
                    + pm * old_prices[j]
                    + pd * old_prices[j - 1]
                )

                exercise_value = max(j * dx - strike_price, 0.0)
                prices[j] = max(continuation_value, exercise_value)

            prices[0] = 0.0
            prices[max_index] = max_index * dx - strike_price

        x = stock_price / dx
        i = int(x)

        if i < 0:
            return prices[0]

        if i >= max_index:
            return prices[max_index]

        return prices[i] + (prices[i + 1] - prices[i]) * (x - i)

    @staticmethod
    def finite_difference_put_delta_and_cash(
        max_stock_price: float,
        strike_price: float,
        risk_free_rate: float,
        volatility: float,
        time_to_maturity: float,
        time_steps: int,
        stock_steps: int
    ) -> Tuple[float, float, float]:
        """
        Price an American put option and calculate delta and initial cash investment.
        """
        dt = time_to_maturity / time_steps
        d_stock = max_stock_price / stock_steps

        option_values = [0.0 for _ in range(stock_steps + 1)]
        new_values = [0.0 for _ in range(stock_steps + 1)]
        stock_grid = [i * d_stock for i in range(stock_steps + 1)]

        for i in range(stock_steps + 1):
            option_values[i] = max(strike_price - stock_grid[i], 0.0)

        for _ in range(time_steps - 1, -1, -1):
            for i in range(1, stock_steps):
                delta = (
                    option_values[i + 1]
                    - option_values[i - 1]
                ) / (2.0 * d_stock)

                gamma = (
                    option_values[i + 1]
                    - 2.0 * option_values[i]
                    + option_values[i - 1]
                ) / (d_stock * d_stock)

                theta = (
                    -0.5 * volatility * volatility * i * i * gamma
                    - risk_free_rate * i * delta
                    + risk_free_rate * option_values[i]
                )

                new_values[i] = max(
                    option_values[i] + dt * theta,
                    strike_price - i * d_stock
                )

            new_values[0] = strike_price
            new_values[stock_steps] = 0.0
            option_values = new_values.copy()

        strike_index = round(strike_price / d_stock)
        strike_index = max(1, min(strike_index, stock_steps - 1))

        delta_at_strike = (
            option_values[strike_index + 1]
            - option_values[strike_index - 1]
        ) / (2.0 * d_stock)

        initial_cash = (
            option_values[strike_index]
            - delta_at_strike * stock_grid[strike_index]
        )

        return option_values[strike_index], delta_at_strike, initial_cash


def get_float(prompt: str, default: float) -> float:
    value = input(f"{prompt} [{default}]: ").strip()
    return default if value == "" else float(value)


def get_int(prompt: str, default: int) -> int:
    value = input(f"{prompt} [{default}]: ").strip()
    return default if value == "" else int(value)


def main() -> None:
    print("=" * 72)
    print("American Option Pricer")
    print("=" * 72)

    stock_price = 100.0
    strike_price = 100.0
    time_to_maturity = 1.0
    risk_free_rate = 0.05
    volatility = 0.2
    price_grid_points = 100
    time_grid_points = 100

    put_prices = AmericanOptionPricer.finite_difference_american_put(
        stock_price,
        strike_price,
        time_to_maturity,
        risk_free_rate,
        volatility,
        price_grid_points,
        time_grid_points
    )

    european_call = AmericanOptionPricer.black_scholes_call_price(
        stock_price,
        strike_price,
        time_to_maturity,
        risk_free_rate,
        volatility
    )

    american_call = AmericanOptionPricer.finite_difference_american_call(
        stock_price,
        strike_price,
        time_to_maturity,
        risk_free_rate,
        volatility,
        price_grid_points,
        time_grid_points
    )

    print()
    print("Default Example")
    print("-" * 72)
    print(f"American Put Price at S = {stock_price}: {put_prices[price_grid_points]:.6f}")
    print(f"European Call Price using Black-Scholes: {european_call:.6f}")
    print(f"American Call Price using Finite Difference: {american_call:.6f}")

    print()
    print("Enter your own model parameters, or press Enter to use defaults.")
    print()

    max_stock_price = get_float("Maximum stock price S_max", 200.0)
    strike_price = get_float("Strike price K", 100.0)
    risk_free_rate = get_float("Risk-free rate r", 0.05)
    volatility = get_float("Volatility sigma", 0.2)
    time_to_maturity = get_float("Time to maturity T in years", 1.0)
    time_steps = get_int("Number of time steps", 100)
    stock_steps = get_int("Number of stock price steps", 100)

    option_price, delta_x, cash_y = AmericanOptionPricer.finite_difference_put_delta_and_cash(
        max_stock_price,
        strike_price,
        risk_free_rate,
        volatility,
        time_to_maturity,
        time_steps,
        stock_steps
    )

    print()
    print("Custom Parameter Results")
    print("-" * 72)
    print(f"American Put Option Price at S = K: {option_price:.6f}")
    print(f"Delta X at S = K: {delta_x:.6f}")
    print(f"Initial Cash Investment Y at S = K: {cash_y:.6f}")


if __name__ == "__main__":
    main()
