"""Compressed exact-token metadata index. Never owns tensors or changes attention."""
from collections import OrderedDict
from threading import RLock


class Node:
    __slots__ = ('edge', 'children', 'terminals')
    def __init__(self, edge=()):
        self.edge, self.children, self.terminals = edge, {}, set()


class TokenRadixTree:
    def __init__(self):
        self.root = Node()
        self.count = 0

    def insert(self, ids, key):
        node, offset = self.root, 0
        while offset < len(ids):
            child = node.children.get(ids[offset])
            if child is None:
                child = Node(ids[offset:]); node.children[ids[offset]] = child
                node = child; offset = len(ids); break
            n = 0
            while n < len(child.edge) and offset+n < len(ids) and child.edge[n] == ids[offset+n]:
                n += 1
            if n < len(child.edge):
                split = Node(child.edge[:n]); node.children[ids[offset]] = split
                child.edge = child.edge[n:]; split.children[child.edge[0]] = child
                node = split; offset += n
            else:
                node = child; offset += n
        if key not in node.terminals:
            node.terminals.add(key); self.count += 1

    def remove(self, ids, key):
        node, offset, path = self.root, 0, []
        while offset < len(ids):
            child = node.children.get(ids[offset])
            if child is None or tuple(ids[offset:offset+len(child.edge)]) != child.edge:
                return
            path.append((node, ids[offset])); node = child; offset += len(child.edge)
        if key not in node.terminals:
            return
        node.terminals.remove(key); self.count -= 1
        for parent, token in reversed(path):
            child = parent.children[token]
            if not child.terminals and not child.children:
                del parent.children[token]
            elif not child.terminals and len(child.children) == 1:
                only = next(iter(child.children.values()))
                child.edge += only.edge; child.terminals = only.terminals; child.children = only.children

    def match(self, ids, limit, lower=0):
        node, offset, best = self.root, 0, None
        if node.terminals and lower == 0:
            best = next(iter(node.terminals))
        while offset < limit:
            child = node.children.get(ids[offset])
            if child is None:
                break
            n = 0
            while n < len(child.edge) and offset+n < limit and child.edge[n] == ids[offset+n]:
                n += 1
            offset += n
            if n != len(child.edge):
                break
            node = child
            if node.terminals and offset >= lower:
                best = next(iter(node.terminals))
        return best, offset


class PrefixRadixIndex:
    def __init__(self):
        self.lock = RLock()
        self.records = {}
        self.models, self.media, self.cached = {}, {}, {}
        self.queries = 0

    @staticmethod
    def _insert(bank, group, ids, key):
        bank.setdefault(group, TokenRadixTree()).insert(ids, key)

    @staticmethod
    def _remove(bank, group, ids, key):
        tree = bank[group]; tree.remove(ids, key)
        if tree.count == 0:
            del bank[group]

    def add(self, source, key, entry):
        if not isinstance(key, tuple) or len(key) != 4 or 'ids' not in entry:
            return  # Legacy unit-test banks can use arbitrary bookkeeping keys.
        if key not in self.records:
            ids, media = tuple(entry['ids']), entry.get('media_key', ('text', False))
            self.records[key] = {'ids': ids, 'media': media, 'sources': set()}
            self._insert(self.models, key[0], ids, key)
            self._insert(self.media, (key[0], media), ids, key)
        record = self.records[key]
        before = bool(record['sources'] & {'gpu', 'cpu'})
        record['sources'].add(source)
        if not before and source in {'gpu', 'cpu'}:
            self._insert(self.cached, key[:3], record['ids'], key)

    def remove(self, source, key):
        record = self.records.get(key)
        if record is None or source not in record['sources']:
            return
        before = bool(record['sources'] & {'gpu', 'cpu'})
        record['sources'].remove(source)
        if before and not record['sources'] & {'gpu', 'cpu'}:
            self._remove(self.cached, key[:3], record['ids'], key)
        if not record['sources']:
            self._remove(self.models, key[0], record['ids'], key)
            self._remove(self.media, (key[0], record['media']), record['ids'], key)
            del self.records[key]

    def match(self, model_id, tokens, boundary, input_key, pooling, media_start, media_end):
        with self.lock:
            return self._match(model_id, tokens, boundary, input_key, pooling, media_start, media_end)

    def _match(self, model_id, tokens, boundary, input_key, pooling, media_start, media_end):
        self.queries += 1
        def query(bank, group, end, lower=0):
            tree = bank.get(group)
            return tree.match(tokens, min(boundary, end), lower) if tree else (None, 0)
        if hasattr(input_key, 'intervals'):
            best, common, best_length = None, 0, -1
            for namespace, image, lower, stop in reversed(input_key.intervals(boundary)):
                if best is not None and best_length >= stop and common >= stop:
                    break
                pool = bool(pooling) if image else False
                candidate, _ = query(self.cached, (model_id, namespace, pool), stop, lower)
                if candidate is not None:
                    length = len(self.records[candidate]['ids'])
                    if length > best_length: best, best_length = candidate, length
                _, shared = query(self.media, (model_id, (namespace, pool)), stop)
                if shared >= lower: common = max(common, shared)
            # Text before media may be observed under an unrelated media key.
            before = 0
            if common < media_start:
                _, before = query(self.models, model_id, media_start)
            return best, max(common, before)
        if media_start is None:
            best, _ = query(self.cached, (model_id, 'text', False), boundary)
            _, common = query(self.models, model_id, boundary)
            return best, common
        best, _ = query(self.cached, (model_id, 'text', False), media_start)
        image_best, _ = query(self.cached, (model_id, input_key, bool(pooling)), boundary, media_end)
        if image_best is not None:
            best = image_best
        _, before = query(self.models, model_id, media_start)
        _, compatible = query(self.media, (model_id, (input_key, bool(pooling))), boundary)
        if media_start < compatible < media_end:
            compatible = media_start
        return best, max(before, compatible)

    def stats(self):
        return {'kind': 'compressed_token_radix', 'records': len(self.records),
                'cached_namespaces': len(self.cached), 'queries': self.queries}


class IndexedBank(OrderedDict):
    """Keep the index current on structural changes, including eviction/clear."""
    def __init__(self, index, source):
        super().__init__(); self.index, self.source = index, source

    def __setitem__(self, key, value):
        with self.index.lock:
            if key in self:
                self.index.remove(self.source, key)
            super().__setitem__(key, value); self.index.add(self.source, key, value)

    def __delitem__(self, key):
        with self.index.lock:
            super().__delitem__(key); self.index.remove(self.source, key)

    def pop(self, key, *default):
        if key not in self:
            if default: return default[0]
            raise KeyError(key)
        value = self[key]; del self[key]; return value

    def popitem(self, last=True):
        with self.index.lock:
            key, value = super().popitem(last=last)
            self.index.remove(self.source, key); return key, value

    def clear(self):
        with self.index.lock:
            keys = list(self); super().clear()
            for key in keys: self.index.remove(self.source, key)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items(): self[key] = value

    def setdefault(self, key, default=None):
        if key not in self: self[key] = default
        return self[key]
