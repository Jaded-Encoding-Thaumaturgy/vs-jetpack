def poly3(x: float, c0: float, c1: float, c2: float, c3: float) -> float:
    return c0 + x * (c1 + x * (c2 + x * c3))
