"""Ансамбль сабмитов: Weighted Boxes Fusion по нескольким CSV (модели фолдов, TTA, разные imgsz).

Каждый входной файл — обычный сабмит (image_id,prediction). Рамки разных моделей, перекрывающиеся
по IoU >= --iou, сливаются в одну: координаты — среднее с весом score, score — среднее по всем моделям
(модель, не нашедшая объект, вносит 0 — так одиночные ложные срабатывания проседают),
форма и материал — голосование с весом score, объём и масса — взвешенное среднее в логарифме,
take — по итоговой массе (если --take-mass-g) или по объёму (--take-volume-ml), иначе голосованием.

    python tools/fuse_subs.py --subs fold1.csv fold2.csv fold3.csv --out fused.csv
    python tools/fuse_subs.py --subs a.csv b.csv --weights 2 1 --take-volume-ml 429 --out fused.csv
Проверка на train (если сабмиты сделаны по отложенным кадрам):
    python score.py check --labels dataset/labels/real_train.json --submission fused.csv
"""
import argparse
import csv
import math

MAX_PER_IMAGE = 30


def parse(cell):
    out = []
    cell = (cell or "").strip()
    if not cell or cell.lower() == "none":
        return out
    for part in cell.split(";"):
        t = part.split()
        if len(t) != 10:
            continue
        out.append(dict(box=[float(v) for v in t[:4]], score=float(t[4]), form=t[5], material=t[6],
                        volume=max(float(t[7]), 1e-3), mass=max(float(t[8]), 1e-3), take=int(float(t[9]))))
    return out


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def vote(items, key):
    w = {}
    for o, s in items:
        w[o[key]] = w.get(o[key], 0.0) + s
    return max(w, key=w.get)


def fuse(lists, weights, thr, skip):
    """lists: [[obj,...] от каждой модели]. Возвращает слитые объекты."""
    allobj = []
    for mi, (objs, wt) in enumerate(zip(lists, weights)):
        for o in objs:
            if o["score"] >= skip:
                allobj.append((o, o["score"] * wt, mi))
    allobj.sort(key=lambda x: -x[1])
    clusters = []   # [fused_box, [(obj, s, mi)]]
    for o, s, mi in allobj:
        best, bi = 0.0, -1
        for ci, (fb, mem) in enumerate(clusters):
            v = iou(fb, o["box"])
            if v > best:
                best, bi = v, ci
        if best >= thr:
            clusters[bi][1].append((o, s, mi))
            mem = clusters[bi][1]
            sw = sum(x[1] for x in mem)
            clusters[bi][0] = [sum(x[0]["box"][k] * x[1] for x in mem) / sw for k in range(4)]
        else:
            clusters.append([list(o["box"]), [(o, s, mi)]])
    W = sum(weights)
    out = []
    for fb, mem in clusters:
        # одна модель могла дать два бокса в кластер — для score берём лучший от каждой модели
        per_model = {}
        for o, s, mi in mem:
            per_model[mi] = max(per_model.get(mi, 0.0), s)
        score = sum(per_model.values()) / W
        items = [(o, s) for o, s, _ in mem]
        sw = sum(s for _, s in items)
        lv = sum(math.log(o["volume"]) * s for o, s in items) / sw
        lm = sum(math.log(o["mass"]) * s for o, s in items) / sw
        out.append(dict(box=fb, score=min(score, 1.0), form=vote(items, "form"), material=vote(items, "material"),
                        volume=math.exp(lv), mass=math.exp(lm),
                        take=int(sum(s for o, s in items if o["take"] == 1) >= sw / 2)))
    out.sort(key=lambda o: -o["score"])
    return out[:MAX_PER_IMAGE]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float)
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--skip", type=float, default=0.0, help="отбросить боксы со score ниже до слияния")
    ap.add_argument("--take-volume-ml", type=float)
    ap.add_argument("--take-mass-g", type=float)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    weights = a.weights or [1.0] * len(a.subs)
    assert len(weights) == len(a.subs)
    subs = []
    for p in a.subs:
        with open(p, newline="") as f:
            subs.append({r["image_id"]: parse(r["prediction"]) for r in csv.DictReader(f)})
    ids = list(subs[0].keys())
    for s in subs[1:]:
        missing = set(ids) - set(s)
        assert not missing, f"в одном из сабмитов нет кадров: {sorted(missing)[:5]}"
    n_in = n_out = 0
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "prediction"])
        for k in ids:
            objs = fuse([s[k] for s in subs], weights, a.iou, a.skip)
            n_in += sum(len(s[k]) for s in subs)
            n_out += len(objs)
            for o in objs:
                if a.take_mass_g is not None:
                    o["take"] = int(o["mass"] <= a.take_mass_g)
                elif a.take_volume_ml is not None:
                    o["take"] = int(o["volume"] <= a.take_volume_ml)
            cell = ";".join(f"{o['box'][0]:.1f} {o['box'][1]:.1f} {o['box'][2]:.1f} {o['box'][3]:.1f} {o['score']:.4f} "
                            f"{o['form']} {o['material']} {o['volume']:.1f} {o['mass']:.1f} {o['take']}" for o in objs)
            w.writerow([k, cell or "none"])
    print(f"кадров {len(ids)}, боксов на входе {n_in}, после слияния {n_out}; записано {a.out}")


if __name__ == "__main__":
    main()
