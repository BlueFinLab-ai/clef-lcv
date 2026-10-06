"""CPU-only queued lookahead. No GPU calls or cache-state restoration."""
from dataclasses import dataclass
from io import BytesIO
import base64

from PIL import Image
from host_memory import available_memory


@dataclass(frozen=True)
class PreparedDecision:
    request: object
    record: dict
    encoded: object
    boundary: int
    checkpoints: tuple
    usage: dict
    input_key: str

    def __getattr__(self, name):
        return getattr(self.request, name)


def preparation_fits(request, max_mib=1024, memory_probe=available_memory):
    """Bound one lookahead's estimated CPU workspace before decoding pixels.

    Header inspection does not load image pixels. 96 bytes/pixel allows for RGB,
    processor intermediates, temporal patches and the assembled media tensor.
    Large/unknown shapes stay inline; these bounds are not OS RAM reservations.
    """
    sample = memory_probe()
    if sample is None or max_mib <= 0:
        return False
    cap = min(int(max_mib)*2**20, sample['available_bytes']//8)
    estimate = len(str(request.state))*8 + len(str(request.context))*8 + 2**20
    try:
        for image in request.images:
            if not isinstance(image, str) or ',' not in image: return False
            data = base64.b64decode(image.split(',', 1)[1], validate=True)
            with Image.open(BytesIO(data)) as source:
                estimate += source.width*source.height*96
            if estimate > cap: return False
    except (OSError, ValueError):
        return False  # Full validation occurs in the normal request handler.
    return estimate <= cap
