
from __future__ import annotations

import io
import logging
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError  # type: ignore[import-not-found]

log = logging.getLogger("telegram.imaging")

# HEIC/HEIF support 
HEIF_OK = False
try:  
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
    HEIF_OK = True
except Exception:  
    log.info("pillow-heif not installed; HEIC images will be rejected politely")

Image.MAX_IMAGE_PIXELS = 80_000_000 


class ImageError(RuntimeError):
    """Raised with a message that is safe to send back to the user."""


@dataclass
class Normalised:
    data: bytes           
    filename: str
    width: int             
    height: int
    source_format: str     
    source_width: int
    source_height: int
    downscaled: bool
    low_resolution: bool   


def _looks_like_heic(raw: bytes, content_type: str) -> bool:
    if "heic" in content_type.lower() or "heif" in content_type.lower():
        return True
    # ISO-BMFF brand box.
    return len(raw) > 12 and raw[4:8] == b"ftyp" and raw[8:12] in (
        b"heic", b"heix", b"hevc", b"mif1", b"msf1",
    )


def normalise(raw: bytes, content_type: str = "", min_short_edge: int = 224,
              max_long_edge: int = 1600) -> Normalised:
    """Decode, fix orientation, flatten transparency, downscale, re-encode JPEG."""
    if not raw:
        raise ImageError("The image came through empty.")

    if content_type.startswith("video/") or content_type.startswith("audio/"):
        raise ImageError("That looks like a video or voice note, not a photo.")

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except UnidentifiedImageError as exc:
        if _looks_like_heic(raw, content_type) and not HEIF_OK:
            raise ImageError(
                "That photo is in Apple's HEIC format and this server can't read it. "
                "Install pillow-heif, or resend the photo as a JPEG."
            ) from exc
        raise ImageError("That file isn't an image I can read.") from exc
    except Exception as exc:
        raise ImageError(f"That image could not be opened ({type(exc).__name__}).") from exc

    source_format = (img.format or "unknown").upper()
    src_w, src_h = img.size
    if src_w == 0 or src_h == 0:
        raise ImageError("That image has no pixels.")

    # Phones store rotation in EXIF 
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        log.warning("exif_transpose failed; continuing with the original orientation")

    # Animated formats (GIF / animated WebP)
    if getattr(img, "n_frames", 1) > 1:
        try:
            img.seek(0)
        except Exception:
            pass


    if img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info
    ):
        rgba = img.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.split()[-1])
        img = canvas
    elif img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    low_resolution = min(w, h) < min_short_edge

    downscaled = False
    if max(w, h) > max_long_edge:
        scale = max_long_edge / float(max(w, h))
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.LANCZOS)
        downscaled = True
        w, h = img.size

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=92, optimize=True)

    return Normalised(
        data=out.getvalue(),
        filename="telegram_upload.jpg",
        width=w,
        height=h,
        source_format=source_format,
        source_width=src_w,
        source_height=src_h,
        downscaled=downscaled,
        low_resolution=low_resolution,
    )
