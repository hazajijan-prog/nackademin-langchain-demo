from fastmcp import FastMCP
from typing import Annotated
from pydantic import Field
from config.custom_logging_config import RequestLoggingMiddleware
from config.logging_config import configure_logging

configure_logging()

mcp = FastMCP("Calculator Server")
mcp.add_middleware(RequestLoggingMiddleware())


@mcp.tool()
def add_numbers(
    a: Annotated[float, Field(description="First number to add")],
    b: Annotated[float, Field(description="Second number to add")],
) -> float:
    """Adds two numbers together and returns the result."""
    return a + b


@mcp.tool()
def divide_numbers(
    a: Annotated[float, Field(description="Numerator (number to be divided)")],
    b: Annotated[float, Field(description="Denominator (number to divide by)")],
) -> float:
    """Divides two numbers together and returns the result."""
    return a / b

@mcp.tool()
def subtract_numbers(
    a: Annotated[float, Field(description="Frist number (number to subtract from)")],
    b: Annotated[float, Field(description="Second number (number to subtract)")],
) -> float:
    """Subtracts the second number from the first and returns the result."""
    return a - b

@mcp.tool()
def multiply_numbers(
    a: Annotated[float, Field(description="Frist number (number to multiply)")],
    b: Annotated[float, Field(description="Second number (number to multiply)")],
) -> float: 
    """Multiplies two numbers together and returns the result."""
    return a * b 

@mcp.tool()
def power_numbers(
    a: Annotated[float, Field(description="Base number")],
    b: Annotated[float, Field(description= "Exponent (power to rise tha base)")],
) -> float:
    """Raises the first number to the power of tha second and returns the result"""
    return a ** b

@mcp.tool()
def sqrt_number(
    x: Annotated[float, Field(description="Number to calculate the square root of")],
) -> float | str:
    """Calculates the square root of a number and returns the result."""
    
    if x < 0:
        return "Cannot calculate square root of a negative number"

    return x ** 0.5




if __name__ == "__main__":
    import asyncio

    asyncio.run(
        mcp.run_http_async(
            host="0.0.0.0",
            port=8001,
            log_level="warning",
        )
    )
