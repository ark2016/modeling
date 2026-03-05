"""
Скачивание SRTM-тайла (1 arc-second, ~30 м) с AWS Terrain Tiles.

Использование:
    python download_srtm.py              # Москва (по умолчанию)
    python download_srtm.py 46 7         # Альпы (lat=46, lon=7)
    python download_srtm.py 37 -112      # Гранд-Каньон

Результат: dem.npy + dem_preview.png в текущей папке.
"""

import sys
import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# AWS Terrain Tiles — публичный доступ, без авторизации
TILE_URL = "https://elevation-tiles-prod.s3.amazonaws.com/skadi/{folder}/{name}.hgt.gz"

SRTM1_SIZE = 3601  # 1 arc-second: 3601×3601 пикселей на 1°×1° тайл


def tile_name(lat: int, lon: int) -> str:
    """N55E037, S02W079 и т.д."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"


def download_tile(lat: int, lon: int, cache_dir: Path) -> Path:
    name = tile_name(lat, lon)
    folder = name[:3]  # N55
    url = TILE_URL.format(folder=folder, name=name)
    gz_path = cache_dir / f"{name}.hgt.gz"

    if gz_path.exists():
        print(f"  Кэш: {gz_path}")
        return gz_path

    print(f"  Скачивание {url} ...")
    urllib.request.urlretrieve(url, gz_path)
    print(f"  Сохранено: {gz_path} ({gz_path.stat().st_size // 1024} КБ)")
    return gz_path


def parse_hgt(gz_path: Path) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        data = f.read()

    n = SRTM1_SIZE
    expected = n * n * 2  # int16, 2 байта
    if len(data) != expected:
        # SRTM3 (3 arc-second): 1201×1201
        n = 1201
        expected = n * n * 2
        if len(data) != expected:
            raise ValueError(f"Неожиданный размер: {len(data)} байт")

    # big-endian int16
    dem = np.array(struct.unpack(f">{n*n}h", data), dtype=np.float64).reshape(n, n)
    # -32768 = void
    dem[dem < -1000] = np.nan
    # .hgt хранит с севера на юг, переворачиваем
    dem = dem[::-1, :]
    return dem


def crop_center(dem: np.ndarray, size: int) -> np.ndarray:
    h, w = dem.shape
    r0 = (h - size) // 2
    c0 = (w - size) // 2
    return dem[r0:r0+size, c0:c0+size]


def main():
    lat = int(sys.argv[1]) if len(sys.argv) > 1 else 55  # Москва
    lon = int(sys.argv[2]) if len(sys.argv) > 2 else 37
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 500  # crop размер

    out_dir = Path(__file__).parent
    cache_dir = out_dir / ".srtm_cache"
    cache_dir.mkdir(exist_ok=True)

    print(f"Тайл: lat={lat}, lon={lon}, crop={size}×{size}")

    gz_path = download_tile(lat, lon, cache_dir)
    dem = parse_hgt(gz_path)
    print(f"  Полный тайл: {dem.shape}, z ∈ [{np.nanmin(dem):.0f}, {np.nanmax(dem):.0f}]")

    # Заполняем NaN средним (если есть)
    if np.any(np.isnan(dem)):
        dem[np.isnan(dem)] = np.nanmean(dem)

    if size < min(dem.shape):
        dem = crop_center(dem, size)
        print(f"  Crop: {dem.shape}")

    np.save(out_dir / "dem.npy", dem)
    print(f"  Сохранено: dem.npy ({dem.shape[0]}×{dem.shape[1]})")

    # Превью
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(dem.T, origin="lower", cmap="terrain", aspect="equal")
    ax.set_title(f"SRTM {tile_name(lat, lon)}  ({dem.shape[0]}×{dem.shape[1]})")
    plt.colorbar(im, ax=ax, label="Высота (м)")
    fig.tight_layout()
    fig.savefig(out_dir / "dem_preview.png", dpi=150)
    plt.close(fig)
    print(f"  Превью: dem_preview.png")


if __name__ == "__main__":
    main()
