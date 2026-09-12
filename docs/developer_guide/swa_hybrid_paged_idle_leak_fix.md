# Hybrid SWA + paged KV idle leak fix (Dots3-Note / EAGLE)

This note documents the minimal fix for `pool memory leak detected` on idle after
heavy serving workloads (for example gsm8k) on hybrid SWA models with paged KV
(`page_size=64`) and EAGLE/NEXTN speculative decoding.

## Background

### Symptom

After a long run, when the scheduler enters `on_idle()` and runs the pool
invariant check, serving can crash with:

```text
pool memory leak detected
```

Typical signature on Dots3-Note (`page_size=64`, sliding window, NEXTN/EAGLE):

| Pool | State |
|------|-------|
| Full | Balanced: `available + evictable + protected == total` |
| SWA  | **+64 over**: one SWA page counted twice in `available` |

So the bug is SWA-side only: the same physical SWA page is returned to the free
list more than once.

### Root cause

Hybrid SWA keeps two coupled pools (full-attention KV and sliding-window KV)
linked by a full→SWA mapping. Under paged allocation and batched cache updates,
three paths could double-return SWA pages:

1. **Deferred `free_swa` inside `free_group`**
   - Window eviction calls `free_swa` while a free group is open.
   - If mapping/pairing is not cleared immediately, a later insert or eviction
     can resolve and free the same SWA page again.

2. **`free_group_end` double flush**
   - `BaseTokenToKVPoolAllocator.free_group_end()` calls `free()` on deferred
     full indices, which runs `free_swa` again.
   - A separate flush of `swa_free_group` can then release the same SWA tokens
     a second time.

3. **Radix cache insert after window eviction**
   - Tombstone insert or finish paths may call `free()` / `free_swa` on ranges
     whose SWA was already returned by window eviction.
   - Evictable accounting that used `len(node.key)` instead of `len(node.value)`
     could also diverge from real KV usage.

## Goal

Restore the idle invariant:

```text
available + evictable + protected == size   (per pool)
```

without changing normal alloc/evict behavior for non-buggy paths.

## Approach (minimal fix)

The fix keeps **logical state (mapping) eager** and **physical SWA return
deferrable** inside a free group, and makes the radix cache respect already-freed
SWA prefixes.

### 1. `python/sglang/srt/mem_cache/allocator/swa.py`

| Change | Why |
|--------|-----|
| `free_swa`: resolve SWA indices and clear full→SWA mapping immediately; filter dummy page 0; `torch.unique`; defer **resolved SWA token ids** (cloned), not raw full indices | Prevents stale mapping from causing a second free of the same page |
| `free_group_end`: do **not** call `super().free_group_end()`; flush `free_group` as full-only, then flush `swa_free_group` once with `torch.unique` | Avoids `free()` → `free_swa` running twice on the same logical release |
| `_release_swa`: skip dummy page 0 at page granularity | Prevents page 0 from entering the SWA free list |

### 2. `python/sglang/srt/mem_cache/allocator/paged.py`

| Change | Why |
|--------|-----|
| `_release_page_ids`: skip page ids already present in `free_pages` / `release_pages` (non-debug) | Defensive guard against duplicate page re-insertion (+64 available) |

### 3. `python/sglang/srt/mem_cache/swa_radix_cache.py`

| Change | Why |
|--------|-----|
| `_swa_unmapped_prefix_len`: detect page-aligned prefix with no live SWA mapping | Distinguish “SWA already returned” from “still live” |
| `_add_new_node`: auto-tombstone fully unmapped ranges; call `free_swa` on tombstone insert only when mapping is still live | Avoid double-free after window eviction |
| `swa_evicted_seqlen >= total_length` (was `==`) | EAGLE page alignment can push eviction boundary past `len(key)` |
| Evictable accounting uses `len(node.value)` instead of `len(node.key)` | Match tree counters to actual KV tensor length |

## What was intentionally left out

The following were evaluated in ablation and are **not** part of this minimal
diff:

- Full-page→SWA-page pairing table and orphan recovery helpers in `swa.py`
- `alloc_decode` split-brain rollback on partial OOM
- Dedicated unit test file `test_swa_paged_eagle_idle_invariant.py`

They may still be useful as follow-ups but are not required for the core idle
+64 SWA leak described above.

## Verification (manual)

Recommended stress path (Dots3-Note, FP8, `page_size=64`, NEXTN):

1. Start server with hybrid SWA + paged KV + EAGLE.
2. Run gsm8k repeatedly (`--num-questions 3000`).
3. Restart server and repeat several cycles.
4. After each run, confirm the server log has **no** `pool memory leak` and
   `/health` stays 200.

Watch for:

- `pool memory leak detected` in scheduler logs after idle
- `Scheduler hit an exception` from `invariant_checker._report_leak`

## Related code

- Idle check: `python/sglang/srt/managers/scheduler.py` (`on_idle`)
- Invariant: `python/sglang/srt/managers/scheduler_components/invariant_checker.py`
- SWA window eviction batching: `python/sglang/srt/managers/schedule_batch.py`
  (`free_group_begin` / `free_group_end` around `_evict_swa`)
