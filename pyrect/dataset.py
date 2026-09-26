"""core/fit/dataset.cpp: the design loader, the split assignment and the training baseline."""
from __future__ import annotations

import os
import numpy as np
import re
from math import exp, floor, isfinite, log
import sys


from .stdlib import MT19937_64, clamp, fabs, to_int

SPLIT_BLOCK_SIZE = 24   # the local split's blocks: a median k-d partition down to at most this many pools


class Observation:
    __slots__ = ("x", "y", "z", "working_weight", "working_response", "events", "exposure", "pool_id", "offset", "attribute_offset")

    def __init__(self, x, y, working_weight, working_response, events, exposure, pool_id, offset, attribute_offset=0.0, z=0.0):
        self.x = x; self.y = y                                      # wala and wac on the unit square
        self.z = z                                                  # the third axis on the unit interval (DMESH_AXIS3)
        self.working_weight = working_weight; self.working_response = working_response   # the working data at the baseline
        self.events = events; self.exposure = exposure; self.pool_id = pool_id
        self.offset = offset                                        # the row's baseline logit: the pooled level plus any offset column
        self.attribute_offset = attribute_offset                    # the offset column, before the pooled level is added


class Dataset:
    def __init__(self):
        self.observations: list[Observation] = []
        self.pool_count = 0
        self.baseline_logit = 0.0
        self.has_offset = False
        self.baseline_source = "training"
        self.bounds = [0.0, 0.0, 0.0, 0.0]   # wala min/max, wac min/max, set at load
        # a three-dimensional mesh: DMESH_AXIS3 names the design column of the third axis, scaled
        # to the unit interval by its own range; empty for the usual two dimensions
        self.axis3 = ""
        self.axis3_bounds = [0.0, 0.0]
        self.training_mask: list[int] = []

    def dim(self):
        return 2 if not self.axis3 else 3


class DesignColumns:
    def __init__(self):
        self.wala = self.wac = self.exposure = self.events = -1
        self.offset = -1
        self.pool_id = -1
        self.axis3 = -1   # the column DMESH_AXIS3 names


def _strtod_ok(text: str) -> bool:
    """Whether std::strtod would consume at least one character (the header probe)."""
    return re.match(r"^[ \t\n\r\x0b\x0c]*[+-]?([0-9]|\.[0-9]|inf|nan|0[xX][0-9a-fA-F])", text, re.IGNORECASE | re.ASCII) is not None


def _strtod(text: str) -> float:
    """A CSV field as a plain decimal number, the whole field: digits, sign, point, exponent,
    or inf/nan. A numeric prefix with trailing text, whitespace, hexadecimal, a nan payload in
    parentheses, a digit separator or a non-ASCII digit is refused. The binary applies the
    same rule (core/fit/dataset.cpp), so both sides accept and refuse the same fields."""
    if not text or not text.isascii():
        raise ValueError(text)
    for ch in text:
        if ch in " \t\n\r\x0b\x0c" or ch in "xX(_":
            raise ValueError(text)
    return float(text)


_NOT_PLAIN = re.compile(r"[ \t\x0b\x0c(_xX]")


def plain_line(line):
    """True when no field of the line can be one _strtod refuses but float() accepts: ASCII
    only, and none of whitespace, the hexadecimal marker, a parenthesis or a digit separator."""
    return line.isascii() and _NOT_PLAIN.search(line) is None


def _field(fields, index, fallback, what, line_number, path):
    """One CSV cell as a number, with the loader's messages (a module function: the per-row
    closure it replaces cost as much as the parsing)."""
    if index < 0:
        return fallback
    if index >= len(fields) or fields[index] == "":
        from .errors import DesignError
        raise DesignError(f"row {line_number} has no {what} in {path}")
    try:
        return _strtod(fields[index])
    except ValueError:
        from .errors import DesignError
        raise DesignError(f"row {line_number} has a non-numeric {what} in {path}")


def parse_header(header: str) -> DesignColumns:
    c = DesignColumns()
    cells = header.split(",")
    if len(cells) > 1 and cells[-1] == "":
        cells.pop()                              # std::getline yields no cell after a trailing comma
    names = ["".join(ch for ch in cell if not ch.isspace() and ch != '"') for cell in cells]
    for i, n in enumerate(names):
        if n == "wala": c.wala = i
        elif n == "wac": c.wac = i
        elif n in ("n0", "exposure", "n"): c.exposure = i
        elif n in ("k_term", "events", "k"): c.events = i
        elif n in ("offset", "off5"): c.offset = i
        elif n in ("pool_id", "cusip"): c.pool_id = i
    axis3 = os.environ.get("DMESH_AXIS3")
    if axis3:
        for i, n in enumerate(names):
            if n == axis3: c.axis3 = i
        if c.axis3 < 0:
            raise RuntimeError(f"DMESH_AXIS3 names a column the design does not have: {axis3}")
    if c.wala < 0 and c.wac < 0 and len(names) >= 4:
        if all(_strtod_ok(n) for n in names):
            c.wala, c.wac, c.exposure, c.events = 0, 1, 2, 3
    if c.wala < 0 or c.wac < 0 or c.exposure < 0 or c.events < 0:
        raise RuntimeError("design header must name wala, wac, n0 and k_term (aliases: exposure/n, events/k)")
    return c


def load_real_data(path: str) -> Dataset:
    from .errors import DesignError as RuntimeError   # noqa: N812  (the loader's errors are DesignError, a RuntimeError)
    if os.path.isdir(path):
        raise RuntimeError(f"data file is empty: {path}")       # an ifstream on a directory opens and reads nothing
    try:
        with open(path, "r", newline="", encoding="utf-8", errors="surrogateescape") as f:
            raw = f.read()
    except OSError:
        raise RuntimeError(f"cannot open data file: {path}")
    lines = raw.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise RuntimeError(f"data file is empty: {path}")
    header = lines[0].rstrip("\r")
    if header.startswith("\ufeff"):
        header = header[1:]                     # a UTF-8 byte order mark before the header
    cols = parse_header(header)
    headerless = cols.wala == 0 and cols.wac == 1 and cols.exposure == 2 and cols.events == 3 and "wala" not in header
    body = lines if headerless else lines[1:]
    from array import array as _array
    rows = 0
    columns = [_array("d") for _ in range(6)]      # wala, wac, exposure, events, offset, third axis
    pools = _array("q")
    pool_ids: dict[str, int] = {}
    total_events = 0.0; total_exposure = 0.0
    min_wala = 1e9; max_wala = -1e9; min_wac = 1e9; max_wac = -1e9
    min_axis3 = 1e300; max_axis3 = -1e300
    line_number = 0 if headerless else 1
    for line in body:
        line_number += 1
        if line.endswith("\r"):
            line = line[:-1]                    # a CRLF line ending, before the blank test
        if line == "":
            continue
        fields = line.split(",")
        if len(fields) > 1 and fields[-1] == "":
            fields.pop()                        # std::getline yields no cell after a trailing comma
        try:
            if not plain_line(line):
                raise ValueError
            # a plain line: float() is _strtod on a field that has none of the characters
            # _strtod refuses, and an empty or missing field raises into the checked path
            wala = float(fields[cols.wala]) if cols.wala >= 0 else 0.0
            wac = float(fields[cols.wac]) if cols.wac >= 0 else 0.0
            exposure = float(fields[cols.exposure]) if cols.exposure >= 0 else 0.0
            events = float(fields[cols.events]) if cols.events >= 0 else 0.0
            offset = float(fields[cols.offset]) if cols.offset >= 0 else 0.0
            axis3 = float(fields[cols.axis3]) if cols.axis3 >= 0 else 0.0
        except (ValueError, IndexError):
            wala = _field(fields, cols.wala, 0.0, "wala", line_number, path); wac = _field(fields, cols.wac, 0.0, "wac", line_number, path)
            exposure = _field(fields, cols.exposure, 0.0, "exposure", line_number, path)
            events = _field(fields, cols.events, 0.0, "events", line_number, path)
            offset = _field(fields, cols.offset, 0.0, "offset", line_number, path)
            axis3 = _field(fields, cols.axis3, 0.0, "third-axis value", line_number, path)
        # rejected, never repaired (the binary's rule, same messages)
        if not (isfinite(wala) and isfinite(wac) and isfinite(exposure) and isfinite(events) and isfinite(offset) and isfinite(axis3)):
            raise RuntimeError(f"row {line_number} has a non-finite value in {path}")
        if (not (exposure >= 1.0) or exposure > 2147483647.0 or not (events >= 0.0) or events > exposure
                or exposure != floor(exposure) or events != floor(events)):
            raise RuntimeError(f"row {line_number} has invalid counts (need integer 1 <= n0 <= 2147483647 and 0 <= k_term <= n0) in {path}")
        if cols.pool_id >= 0:
            if cols.pool_id >= len(fields) or fields[cols.pool_id] == "":
                raise RuntimeError(f"row {line_number} has no pool_id in {path}")
            key = fields[cols.pool_id]
            if key not in pool_ids:
                pool_ids[key] = len(pool_ids)
            pool = pool_ids[key]
        else:
            pool = rows
        for column, value in zip(columns, (wala, wac, exposure, events, offset, axis3)):
            column.append(value)
        pools.append(pool)
        rows += 1
        total_events += events
        total_exposure += exposure
        if wala < min_wala: min_wala = wala
        if wala > max_wala: max_wala = wala
        if wac < min_wac: min_wac = wac
        if wac > max_wac: max_wac = wac
        if axis3 < min_axis3: min_axis3 = axis3
        if axis3 > max_axis3: max_axis3 = axis3
    if cols.axis3 >= 0 and not (max_axis3 > min_axis3):
        raise RuntimeError(f"the third axis (DMESH_AXIS3) is constant in {path}")
    if (rows == 0 or total_exposure <= 0.0 or total_events <= 0.0 or total_events >= total_exposure
            or max_wala <= min_wala or max_wac <= min_wac):
        raise RuntimeError(f"data file has invalid or degenerate contents: {path}")
    pooled_probability = total_events / total_exposure
    use_offset = cols.offset >= 0
    dataset = Dataset()
    dataset.has_offset = use_offset
    dataset.bounds = [min_wala, max_wala, min_wac, max_wac]
    if cols.axis3 >= 0:
        dataset.axis3 = os.environ.get("DMESH_AXIS3"); dataset.axis3_bounds = [min_axis3, max_axis3]
    # Repeated values share one object: WALA, WAC, counts and the constant columns repeat across
    # rows, and a float object costs more than the slot that points at it. Same values, same
    # arithmetic; only the identity of equal numbers is shared.
    # one table per column, so a table never mixes 1 and 1.0
    tx = {}; ty = {}; tn = {}; tk = {}; toff = {}
    for (wala, wac, exposure, events, roffset, raxis3, pool) in zip(*columns, pools):
        x = (wala - min_wala) / (max_wala - min_wala); x = tx.setdefault(x, x)
        y = (wac - min_wac) / (max_wac - min_wac); y = ty.setdefault(y, y)
        n = to_int(exposure); n = tn.setdefault(n, n)
        k = to_int(events); k = tk.setdefault(k, k)
        roffset = toff.setdefault(roffset, roffset)
        # the baseline and the working data are set by build_dataset once the split is known
        z = (raxis3 - min_axis3) / (max_axis3 - min_axis3) if cols.axis3 >= 0 else 0.0
        dataset.observations.append(Observation(x, y, 0.0, 0.0, k, n, pool, 0.0, roffset, z))
    dataset.pool_count = len(pool_ids) if cols.pool_id >= 0 else rows
    sys.stderr.write("[data] %d rows, %d pools, all-row rate %.4f (descriptive; the fit's baseline is the training rate, see [split]), WALA [%g,%g], WAC [%g,%g]%s\n"
                     % (rows, dataset.pool_count, pooled_probability, min_wala, max_wala, min_wac, max_wac,
                        ", offset column" if use_offset else ""))
    return dataset


def assign_split_mask(config, dataset: Dataset) -> None:
    n = len(dataset.observations)
    dataset.training_mask = [1] * n
    if not config.fit.heldout_split or n == 0:
        return
    mask_path = os.environ.get("DMESH_SPLIT_MASK")
    if mask_path is not None:
        try:
            with open(mask_path, "r", encoding="utf-8", errors="surrogateescape") as f:
                text = f.read()
        except OSError:
            raise RuntimeError("DMESH_SPLIT_MASK: cannot open file")
        count = 0
        # the binary extracts ints from a stream (`mask_file >> value`): C whitespace separates
        # tokens, each extraction takes the signed decimal prefix of the next token, and the
        # first token without one (or outside the int range) stops the stream
        for tok in re.split(r"[ \t\n\r\x0b\x0c]+", text):
            if tok == "":
                continue
            if count >= n:
                raise RuntimeError("DMESH_SPLIT_MASK: more entries than rows")
            m = re.match(r"^[+-]?[0-9]+", tok, re.ASCII)
            if not m or not (-2147483648 <= int(m.group(0)) <= 2147483647):
                break
            dataset.training_mask[count] = 1 if int(m.group(0)) != 0 else 0
            count += 1
            if m.end() < len(tok):
                break                               # the rest of the token fails the next extraction
        if count != n:
            raise RuntimeError("DMESH_SPLIT_MASK: entry count != row count")
        return
    pool_count = max(dataset.pool_count, 0)
    stats = {"blocks": 0, "imbalance": 0.0}
    # the per-pool sums with numpy: bincount accumulates in row order, so each pool's
    # weighted sums are the loop's sums; units are the pools with rows, in pool order
    observations = dataset.observations
    n_rows = len(observations)
    pid = np.fromiter((o.pool_id for o in observations), dtype=np.int64, count=n_rows)
    valid = (pid >= 0) & (pid < pool_count)
    w = np.maximum(np.fromiter((o.exposure for o in observations), dtype=float, count=n_rows), 1.0)[valid]
    ox = np.fromiter((o.x for o in observations), dtype=float, count=n_rows)[valid]
    oy = np.fromiter((o.y for o in observations), dtype=float, count=n_rows)[valid]
    pv = pid[valid]
    sx = np.bincount(pv, weights=w * ox, minlength=pool_count)
    sy = np.bincount(pv, weights=w * oy, minlength=pool_count)
    sw = np.bincount(pv, weights=w, minlength=pool_count)
    has = np.bincount(pv, minlength=pool_count) > 0
    unit_pool = np.flatnonzero(has).astype(np.int64)
    ux = sx[has] / sw[has]; uy = sy[has] / sw[has]; ue = sw[has]
    block_size = SPLIT_BLOCK_SIZE

    # The binary's partition with numpy: the orders are total (coordinates, then the other
    # coordinate, then the unique tie), so the set of units on each side of a median split
    # and the leaf order are unique and independent of the algorithm that finds them;
    # the leaves run the greedy assignment in the same left-to-right order, so the
    # generator is consulted identically.
    rng = MT19937_64(config.seed)
    ut = np.array([rng() for _ in range(len(unit_pool))], dtype=np.uint64)
    pool_train = np.zeros(pool_count, dtype=np.int32)
    stack = [np.arange(len(unit_pool))]
    leaves = []
    while stack:
        idx = stack.pop()
        count = len(idx)
        if count <= block_size:
            leaves.append(idx)
            continue
        x_axis = (ux[idx].max() - ux[idx].min()) >= (uy[idx].max() - uy[idx].min())
        primary = ux[idx] if x_axis else uy[idx]
        other = uy[idx] if x_axis else ux[idx]
        ranked = idx[np.lexsort((ut[idx], other, primary))]
        mid = count // 2
        stack.append(ranked[mid:])        # right after left: the left half is processed first
        stack.append(ranked[:mid])
    for idx in leaves:
        count = len(idx)
        stats["blocks"] += 1
        sorted_idx = idx[np.lexsort((ut[idx], -ue[idx]))]
        target_a = (count + 1) // 2
        target_b = count // 2
        side_a = [0] * count
        exp_a = 0.0; exp_b = 0.0; na = 0; nb = 0
        exposures = ue[sorted_idx].tolist()
        for j in range(count):
            if na >= target_a:
                to_a = False
            elif nb >= target_b:
                to_a = True
            elif fabs(exp_a - exp_b) <= 1e-12:
                to_a = (rng() & 1) != 0
            else:
                to_a = exp_a < exp_b
            side_a[j] = 1 if to_a else 0
            if to_a:
                exp_a += exposures[j]; na += 1
            else:
                exp_b += exposures[j]; nb += 1
        flip = (rng() & 1) != 0
        for j, u in enumerate(sorted_idx.tolist()):
            train = (side_a[j] != 0) ^ flip
            pool_train[unit_pool[u]] = 1 if train else 0
        train_exp = exp_b if flip else exp_a
        held_exp = exp_a if flip else exp_b
        total = train_exp + held_exp
        if total > 0.0:
            stats["imbalance"] = max(stats["imbalance"], fabs(train_exp - held_exp) / total)
    mask = np.ones(n_rows, dtype=np.int64)
    mask[valid] = np.asarray(pool_train, dtype=np.int64)[pv]
    dataset.training_mask = mask.tolist()
    sys.stderr.write("[split] local seed %u, block<=%d, %d blocks, max block exposure imbalance %.3f\n"
                     % (config.seed, block_size, stats["blocks"], stats["imbalance"]))
    train_rows = held_rows = 0
    train_n = held_n = train_k = held_k = 0.0
    for i, o in enumerate(dataset.observations):
        if dataset.training_mask[i]:
            train_rows += 1; train_n += o.exposure; train_k += o.events
        else:
            held_rows += 1; held_n += o.exposure; held_k += o.events
    sys.stderr.write("[split] mode local: train %d rows / %.0f exposure / rate %.6f; held %d rows / %.0f exposure / rate %.6f\n"
                     % (train_rows, train_n, train_k / max(train_n, 1.0),
                        held_rows, held_n, held_k / max(held_n, 1.0)))


def build_dataset(config) -> Dataset:
    if config.data_file is None:
        raise NotImplementedError("the synthetic generator (no DMESH_DATA) is not ported; pyrect fits real designs")
    dataset = load_real_data(config.data_file)
    assign_split_mask(config, dataset)
    k = 0.0; n = 0.0
    for i, o in enumerate(dataset.observations):
        if not dataset.training_mask[i]:
            continue
        k += o.events
        n += o.exposure
    if not (n > 0.0):
        raise RuntimeError("split has no training exposure")
    if config.fit.baseline == "external":
        if not dataset.has_offset:
            raise RuntimeError("DMESH_BASELINE=external needs an offset column carrying each row's baseline logit")
        dataset.baseline_logit = 0.0
        dataset.baseline_source = "external"
    else:
        p = clamp(k / n, 1e-9, 1.0 - 1e-9)
        dataset.baseline_logit = log(p / (1.0 - p))
        dataset.baseline_source = "training"
    for o in dataset.observations:
        o.offset = dataset.baseline_logit + o.attribute_offset
        p0 = 1.0 / (1.0 + exp(-o.offset))
        information = o.exposure * p0 * (1.0 - p0)
        o.working_weight = information
        o.working_response = (o.events - o.exposure * p0) / information
    return dataset
