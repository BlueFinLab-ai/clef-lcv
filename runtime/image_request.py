"""Upstream processor options, with an explicit legacy fidelity adapter."""

IMAGE_FIDELITIES = {"standard": 256**2, "low": 128**2, "medium": 512**2, "high": 1024**2}


def media_options(media_kwargs, image_fidelity, legacy_max_pixels):
    """Absent fidelity preserves upstream processor defaults and caller options."""
    if image_fidelity is None:
        return media_kwargs
    if media_kwargs is not None:
        raise ValueError("Use media_kwargs or legacy image_fidelity, not both")
    pixels = IMAGE_FIDELITIES[image_fidelity]
    if pixels > legacy_max_pixels:
        raise ValueError("Selected image fidelity exceeds the server's legacy processing limit")
    return {"images_kwargs": {"min_pixels": min(65536, pixels), "max_pixels": pixels}}
