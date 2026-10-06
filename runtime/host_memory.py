"""Linux allocatable RAM, bounded by visible cgroup ancestors. Never count swap."""
from pathlib import Path, PurePosixPath


def _read(path):
    try:
        return path.read_text().strip()
    except (OSError, ValueError):
        return None


def _number(path):
    value = _read(path)
    try:
        number = int(value)
        # cgroup v1 uses a huge integer for unlimited; v2 uses "max".
        return number if 0 <= number < 2**60 else None
    except (TypeError, ValueError):
        return None


def available_memory(proc_root=Path('/proc')):
    """Return a fresh physical/cgroup headroom sample or None if unavailable.

    MemAvailable already accounts for reclaimable OS caches. Cgroup headroom is
    conservative: usage includes file cache; no swap or inactive-file credit.
    Only ancestors visible in this process's cgroup mount can be inspected.
    """
    info = _read(proc_root / 'meminfo') or ''
    system = None
    for line in info.splitlines():
        if line.startswith('MemAvailable:'):
            try:
                system = max(0, int(line.split()[1])) * 1024
            except (ValueError, IndexError):
                pass
    constraints = []
    memberships = []
    for line in (_read(proc_root / 'self/cgroup') or '').splitlines():
        parts = line.split(':', 2)
        if len(parts) == 3:
            memberships.append((parts[1].split(','), parts[2]))
    for line in (_read(proc_root / 'self/mountinfo') or '').splitlines():
        before, separator, after = line.partition(' - ')
        left, right = before.split(), after.split()
        if not separator or len(left) < 5 or len(right) < 3:
            continue
        kind = right[0]
        if kind not in ('cgroup', 'cgroup2'):
            continue
        if kind == 'cgroup' and 'memory' not in right[2].split(','):
            continue
        # mountinfo encodes spaces/backslashes with octal escapes.
        unescape = lambda s: s.replace('\\040', ' ').replace('\\011', '\t').replace('\\134', '\\')
        mount_root = PurePosixPath(unescape(left[3]))
        mount = Path(unescape(left[4]))
        for controllers, member in memberships:
            if not ((kind == 'cgroup2' and controllers == ['']) or
                    (kind == 'cgroup' and 'memory' in controllers)):
                continue
            member = PurePosixPath(member)
            if '..' in member.parts:
                continue
            try:
                relative = member.relative_to(mount_root)
            except ValueError:
                # A cgroup namespace can report paths relative to its own root.
                relative = member.relative_to('/')
            current = mount / relative
            while True:
                if kind == 'cgroup2':
                    usage = _number(current / 'memory.current')
                    limits = [_number(current / 'memory.max'), _number(current / 'memory.high')]
                else:
                    usage = _number(current / 'memory.usage_in_bytes')
                    limits = [_number(current / 'memory.limit_in_bytes')]
                if usage is not None:
                    constraints.extend(max(0, limit - usage) for limit in limits if limit is not None)
                if current == mount:
                    break
                current = current.parent
    candidates = ([system] if system is not None else []) + constraints
    return {'available_bytes': min(candidates), 'system_available_bytes': system,
            'cgroup_available_bytes': min(constraints) if constraints else None} if candidates else None
