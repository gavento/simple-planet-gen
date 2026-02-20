"""Vectorized 3D Perlin noise and spherical fractal Brownian motion."""

from __future__ import annotations

import numpy as np


def _fade(t):
    """Perlin improved fade curve: 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def _grad(hash_val, x, y, z):
    """Compute gradient dot product using Perlin's hash trick."""
    h = hash_val & 15
    u = np.where(h < 8, x, y)
    v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
    return np.where(h & 1, -u, u) + np.where(h & 2, -v, v)


class PerlinNoise3D:
    """Vectorized 3D Perlin gradient noise.

    Evaluates noise on numpy arrays of arbitrary shape.
    Output range approximately [-1, 1].
    """

    def __init__(self, seed: int = 0):
        rng = np.random.RandomState(seed)
        p = rng.permutation(256).astype(np.int32)
        # Double the permutation table to avoid modular arithmetic
        self.perm = np.concatenate([p, p])

    def __call__(self, x, y, z):
        """Evaluate noise at (x, y, z). Inputs may be numpy arrays."""
        p = self.perm

        # Integer grid coordinates
        xi = np.floor(x).astype(np.int32) & 255
        yi = np.floor(y).astype(np.int32) & 255
        zi = np.floor(z).astype(np.int32) & 255

        # Fractional part
        xf = x - np.floor(x)
        yf = y - np.floor(y)
        zf = z - np.floor(z)

        # Fade curves
        u = _fade(xf)
        v = _fade(yf)
        w = _fade(zf)

        # Hash the 8 unit cube corners
        A = p[xi] + yi
        AA = p[A] + zi
        AB = p[A + 1] + zi
        B = p[xi + 1] + yi
        BA = p[B] + zi
        BB = p[B + 1] + zi

        # Gradient dot products at each corner
        g000 = _grad(p[AA], xf, yf, zf)
        g100 = _grad(p[BA], xf - 1, yf, zf)
        g010 = _grad(p[AB], xf, yf - 1, zf)
        g110 = _grad(p[BB], xf - 1, yf - 1, zf)
        g001 = _grad(p[AA + 1], xf, yf, zf - 1)
        g101 = _grad(p[BA + 1], xf - 1, yf, zf - 1)
        g011 = _grad(p[AB + 1], xf, yf - 1, zf - 1)
        g111 = _grad(p[BB + 1], xf - 1, yf - 1, zf - 1)

        # Trilinear interpolation
        x1 = g000 + u * (g100 - g000)
        x2 = g010 + u * (g110 - g010)
        y1 = x1 + v * (x2 - x1)

        x3 = g001 + u * (g101 - g001)
        x4 = g011 + u * (g111 - g011)
        y2 = x3 + v * (x4 - x3)

        return y1 + w * (y2 - y1)


def fbm_3d(
    noise: PerlinNoise3D,
    x,
    y,
    z,
    octaves: int = 6,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
) -> np.ndarray:
    """Fractal Brownian motion using 3D Perlin noise."""
    result = np.zeros_like(x, dtype=np.float64)
    amplitude = 1.0
    frequency = 1.0
    max_val = 0.0
    for _ in range(octaves):
        result += amplitude * noise(x * frequency, y * frequency, z * frequency)
        max_val += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return result / max_val  # Normalize to roughly [-1, 1]


def spherical_fbm(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    sphere_x: np.ndarray,
    sphere_y: np.ndarray,
    sphere_z: np.ndarray,
    frequency: float = 4.0,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    """Generate fractal noise on a sphere.

    Maps lat/lon to 3D coordinates and evaluates fBm there.
    Seamless wrapping guaranteed by the spherical mapping.
    """
    noise = PerlinNoise3D(seed)
    return fbm_3d(
        noise,
        sphere_x * frequency,
        sphere_y * frequency,
        sphere_z * frequency,
        octaves=octaves,
        lacunarity=lacunarity,
        persistence=persistence,
    )


def spherical_noise_single(
    sphere_x: np.ndarray,
    sphere_y: np.ndarray,
    sphere_z: np.ndarray,
    frequency: float = 4.0,
    seed: int = 0,
) -> np.ndarray:
    """Single octave of noise on a sphere."""
    noise = PerlinNoise3D(seed)
    return noise(
        sphere_x * frequency,
        sphere_y * frequency,
        sphere_z * frequency,
    )
