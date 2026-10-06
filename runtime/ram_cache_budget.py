"""A service-local retained-tensor budget shared by CPU cache tiers."""
from threading import RLock
from host_memory import available_memory


class RAMCacheBudget:
    def __init__(self, reserve_fraction=.25, memory_probe=None):
        if not 0 <= reserve_fraction < 1:
            raise ValueError('RAM reserve fraction must be in [0, 1)')
        self.reserve_fraction = reserve_fraction
        self.memory_probe = memory_probe or available_memory
        self.lock = RLock()
        self.owners = {}
        self.pending = {}

    def register(self, owner, size, evict, fraction=1.):
        if not 0 < fraction <= 1:
            raise ValueError('RAM tier fraction must be in (0, 1]')
        self.owners[owner] = (size, evict, fraction)

    def estimate(self, owner):
        sample = self.memory_probe()
        if sample is None:
            return 0, 0, sample
        capacity = sample['available_bytes'] + sum(size() for size, _, _ in self.owners.values()) + sum(size() for size in self.pending.values())
        reserve = int(capacity * self.reserve_fraction)
        return int((capacity-reserve) * self.owners[owner][2]), reserve, sample

    def admit(self, owner, addition=0, maximum=None):
        """Caller holds lock through publication, so retained copies cannot race."""
        size, evict, _ = self.owners[owner]
        while True:
            cap, _, sample = self.estimate(owner)
            total_cap = cap / self.owners[owner][2]
            if maximum is not None:
                cap = min(maximum, cap) if sample is not None else maximum
            if addition > cap:
                return False
            if size()+addition > cap:
                if not evict(): return False
                continue
            if sample is None:
                return maximum is not None
            if sum(s() for s, _, _ in self.owners.values()) + addition <= total_cap:
                return True
            # Preserve the arriving cache's working set when it is empty. A new
            # processed image can make room by evicting a cold prefix snapshot.
            if size() and evict(): continue
            if not any(drop() for name, (_, drop, _) in self.owners.items() if name != owner):
                return False
