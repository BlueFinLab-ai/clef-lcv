"""Differential radix/scan lookup for mixed image branches and exact identities."""
from pathlib import Path
import random,sys
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'scripts'),str(root/'runtime'),str(root/'vendor/cloudflare')]
from test_image_prefix_index import fixture
from optimized_inference import InferenceEngine
rng=random.Random(6801);engine=InferenceEngine(0);models=[object(),object()]
banks=[engine.entries,engine.host.entries,engine.observations]
for step in range(800):
    model=rng.choice(models);images=[rng.choice('ABCDE') for _ in range(rng.randint(1,6))]
    encoded,ancestry=fixture(images,count=rng.choice([4,8]),options=rng.choice([None,{'max_pixels':65536}]))
    pool=bool(rng.randrange(2));boundary=len(encoded.input_ids)
    if rng.random()<.7:
        end=rng.choice([3,*ancestry.checkpoints,boundary])
        key=engine._input_key(model,encoded.input_ids,end,ancestry,pool,3)
        bank=rng.choice(banks)
        bank[key]={'ids':encoded.input_ids[:end], 'media_key':key[1:3], **({'cache':object()} if bank is not engine.observations else {})}
    elif rng.random()<.85:
        bank=rng.choice(banks)
        if bank:bank.pop(rng.choice(list(bank)))
    else:rng.choice(banks).clear()
    actual=engine._match(model,encoded.input_ids,boundary,ancestry,pool,3,boundary-2)
    expected=engine._match_scan(model,encoded.input_ids,boundary,ancestry,pool,3,boundary-2)
    assert actual[1]==expected[1],(step,actual,expected)
    assert (actual[0] is None)==(expected[0] is None),(step,actual,expected)
    if actual[0] is not None:
        assert actual[0][:3]==expected[0][:3]
        assert engine._lookup(actual[0])['ids']==engine._lookup(expected[0])['ids']
print('PASS: 800 randomized image radix/scan comparisons, both tiers, observations, grids/settings/pooling/model isolation and eviction')
