"""Physical memory and visible cgroup v1/v2 budgets, without allocating RAM."""
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from host_memory import available_memory

with TemporaryDirectory() as temp:
    root = Path(temp)
    proc, mount = root / 'proc', root / 'cgroup'
    (proc / 'self').mkdir(parents=True)
    (mount / 'worker').mkdir(parents=True)
    (proc / 'meminfo').write_text('MemTotal: 1048576 kB\nMemAvailable: 262144 kB\nSwapFree: 999999999 kB\n')
    assert available_memory(proc)['available_bytes'] == 256 * 2**20
    # Parent headroom is tighter than the leaf; mount root need not be "/".
    (proc / 'self/cgroup').write_text('0::/tenant/worker\n')
    (proc / 'self/mountinfo').write_text(f'1 0 0:1 /tenant {mount} rw - cgroup2 cgroup rw\n')
    for path, limit, used in [(mount, 128, 80), (mount / 'worker', 64, 2)]:
        (path / 'memory.max').write_text(str(limit * 2**20))
        (path / 'memory.current').write_text(str(used * 2**20))
    snapshot = available_memory(proc)
    assert snapshot['available_bytes'] == snapshot['cgroup_available_bytes'] == 48 * 2**20
    (mount / 'worker/memory.high').write_text(str(16 * 2**20))
    assert available_memory(proc)['available_bytes'] == 14 * 2**20
    (mount / 'worker/memory.high').write_text('max')
    (mount / 'memory.current').write_text(str(130 * 2**20))
    assert available_memory(proc)['available_bytes'] == 0
    # Namespaced cgroup paths relative to a mount with a non-root source.
    (proc / 'self/cgroup').write_text('0::/worker\n')
    (mount / 'memory.max').write_text('max')
    assert available_memory(proc)['available_bytes'] == 62 * 2**20
    # Legacy memory controller, including its ancestor's limit.
    (proc / 'self/cgroup').write_text('5:memory:/worker\n')
    (proc / 'self/mountinfo').write_text(f'1 0 0:1 / {mount} rw - cgroup cgroup rw,memory\n')
    (mount / 'memory.limit_in_bytes').write_text(str(32 * 2**20))
    (mount / 'memory.usage_in_bytes').write_text(str(20 * 2**20))
    (mount / 'worker/memory.limit_in_bytes').write_text(str(2**63 - 4096))
    (mount / 'worker/memory.usage_in_bytes').write_text('0')
    assert available_memory(proc)['available_bytes'] == 12 * 2**20
    (proc / 'meminfo').write_text('MemAvailable: 1024 kB\n')
    assert available_memory(proc)['available_bytes'] == 2**20
    assert available_memory(root / 'missing') is None

print('PASS: MemAvailable, no swap credit, cgroup v1/v2 ancestors, namespaces, soft pressure and unavailable readings')
