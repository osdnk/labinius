import re, sys
D, H, TEX = sys.argv[1], sys.argv[2], sys.argv[3]
SIZES = ["sizes", "sizem", "sizel", "sizexl"]

def ms(block, key):
    m = re.search(r"^\s*" + re.escape(key) + r"\s+([\d.]+) ms", block, re.M)
    return float(m.group(1))

def kb(block, key):
    m = re.search(r"^\s*" + re.escape(key) + r"\s+([\d.]+) KB", block, re.M)
    return float(m.group(1))

def section(text, start, end):
    i = text.index(start)
    j = text.index(end, i + 1) if end else len(text)
    return text[i:j]

def ours_clear(block):
    total = kb(block, "wire: commitment")
    m = re.search(r"wire: commitment ([\d.]+) KB, row evaluation ([\d.]+) KB, folded witness ([\d.]+) KB, TOTAL ([\d.]+) KB", block)
    c, row, fw, tot = map(float, m.groups())
    commit = ms(block, "commit") if "commit  " in block else ms(block, "commit (with the digits)")
    prover = ms(block, "total except encode") - commit
    verifier = ms(block, "total except decode")
    return dict(comm=commit, prover=prover, verifier=verifier, c=c, pi=row + fw, total=tot)

def ours_rec(block):
    commit = ms(block, "commit (with T_Y)")
    prover = ms(block, "total") - commit
    m = re.search(r"VERIFIER.*?^\s*total\s+([\d.]+) ms", block, re.S | re.M)
    verifier = float(m.group(1))
    ty = float(re.search(r"wire: T_Y ([\d.]+) KB", block).group(1))
    tot = float(re.search(r"proof: ([\d.]+) KB total", block).group(1))
    return dict(comm=commit, prover=prover, verifier=verifier, c=ty, pi=tot - ty, total=tot)

pcs = {}
for s in SIZES:
    t = open(f"{D}/bin-ntt-{s}.log").read()
    clear = ours_clear(section(t, "=== recursion off ===", "=== plain-bd ==="))
    bd = ours_clear(section(t, "=== plain-bd ===", "peak resident"))
    rec = ours_rec(open(f"{D}/bin-ntt-{s}-labrador.log").read())
    comp = {}
    for line in open(f"{D}/pcs-competitors-{s}.log"):
        m = re.match(r"\s*(binius64 BaseFold|binius64 WHIR|flock-core Ligerito Fast100|flock-core Ligerito Slim100|Brakedown tensor)\s+(1/[24]|0\.\d+)\s+\S+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)", line)
        if m:
            comp[(m.group(1), m.group(2))] = dict(comm=float(m.group(3)), prover=float(m.group(4)), verifier=float(m.group(5)), c=int(m.group(6)) / 1024, pi=int(m.group(7)) / 1024)
    pcs[s] = dict(clear=clear, bd=bd, rec=rec, comp=comp)

def cells(d):
    c = f"{d['c']:.2f}" if d["c"] < 0.1 else f"{d['c']:.1f}"
    return " & ".join([f"{d['comm']:.1f}", f"{d['prover']:.1f}", f"{d['verifier']:.1f}", c, f"{d['pi']:.1f}"])

PANELS = [SIZES[:2], SIZES[2:]]

def panels(pick):
    return [" & ".join(cells(pick(s)) for s in panel) for panel in PANELS]

rows = {}
rows[r"\basefold, $1/2$"] = panels(lambda s: pcs[s]["comp"][("binius64 BaseFold", "1/2")])
rows[r"\basefold, $1/4$"] = panels(lambda s: pcs[s]["comp"][("binius64 BaseFold", "1/4")])
rows[r"\whir, $1/2$"] = panels(lambda s: pcs[s]["comp"][("binius64 WHIR", "1/2")])
rows[r"\whir, $1/4$"] = panels(lambda s: pcs[s]["comp"][("binius64 WHIR", "1/4")])
rows[r"\ligerito, $1/2$"] = panels(lambda s: pcs[s]["comp"][("flock-core Ligerito Fast100", "1/2")])
rows[r"\ligerito, $1/4$"] = panels(lambda s: pcs[s]["comp"][("flock-core Ligerito Slim100", "1/4")])
rows[r"\brakedown, $0.704$"] = panels(lambda s: pcs[s]["comp"][("Brakedown tensor", "0.704")])
rows[r"\brakedown, $0.581$"] = panels(lambda s: pcs[s]["comp"][("Brakedown tensor", "0.581")])
best = []
for s in SIZES:
    pick = "bd" if pcs[s]["bd"]["total"] < pcs[s]["clear"]["total"] else "clear"
    best.append(pick)
    print(f"{s}: clear total {pcs[s]['clear']['total']:.1f} KB, bd total {pcs[s]['bd']['total']:.1f} KB -> {pick}", file=sys.stderr)
picked = dict(zip(SIZES, best))
rows[r"\ourwork"] = panels(lambda s: pcs[s][picked[s]])
rows[r"\ourwork +\labrador"] = panels(lambda s: pcs[s]["rec"])

def hash_tables(kind):
    out = {}
    for s in SIZES:
        try:
            t = open(f"{H}/hashes-{kind}-{s}.log").read()
        except FileNotFoundError:
            continue
        for block in re.split(r"(?=^bin-ntt over )", t, flags=re.M):
            if not block.startswith("bin-ntt over"):
                continue
            name = re.match(r"bin-ntt over \S+ (\S+),", block).group(1)
            cols = re.search(r"^\s+total\s+(.*?) ms$", section(block, "PROVER", "VERIFIER"), re.M)
            if kind == "binius":
                prov = [float(x) for x in re.search(r"^\s*total\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) ms", section(block, "PROVER", "VERIFIER"), re.M).groups()]
                ver = [float(x) for x in re.search(r"^\s*total\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) ms", section(block, "VERIFIER", "SIZES"), re.M).groups()]
                siz = [float(x) for x in re.search(r"^\s*total\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) KB", section(block, "SIZES", "peak"), re.M).groups()]
                count = re.search(r": (\d+) (permutations|compressions)", block).group(1)
            else:
                prov = [float(x) for x in re.search(r"^\s*total, witness included\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) ms", block, re.M).groups()][1:]
                ver = [float(x) for x in re.search(r"^\s*total\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) ms", section(block, "VERIFIER", "SIZES"), re.M).groups()][1:]
                siz = [float(x) for x in re.search(r"^\s*total\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) KB", section(block, "SIZES", "peak"), re.M).groups()][1:]
                count = re.search(r"^(\d+) compressions", block, re.M).group(1)
            out.setdefault(name, []).append((s, count, list(zip(prov, ver, siz))))
    return out

binius = hash_tables("binius")
flock = hash_tables("flock")
for name, runs in list(binius.items()) + list(flock.items()):
    print(name, [(s, c) for s, c, _ in runs], file=sys.stderr)

tex = open(TEX).read()
def patch_row(block, label, values):
    if isinstance(values, str):
        values = [values]
    loose = r"\s*".join(re.escape(c) for c in label if not c.isspace())
    pat = re.compile(r"^(\s*" + loose + r"\s*&\s*).*?(\s*\\\\)$", re.M)
    assert len(pat.findall(block)) == len(values), (label, len(pat.findall(block)), len(values))
    it = iter(values)
    return pat.sub(lambda m: m.group(1) + next(it) + m.group(2), block)

tables = re.split(r"(?=\\begin\{table\})", tex)
def fmt3(triples):
    return " & ".join(" & ".join(f"{v:.1f}" for v in t) for t in triples)
def hpanels(triples):
    return [fmt3(triples[:2]), fmt3(triples[2:])] if len(triples) == 4 else [fmt3(triples)]
for i, tb in enumerate(tables):
    if r"\label{tab:concrete-sizes}" in tb:
        for label, values in rows.items():
            tb = patch_row(tb, label, values)
    for kind, data, stock in (("binius", binius, r"\binius"), ("flock", flock, r"\flock")):
        for name, runs in data.items():
            key = {"keccak-256": "binius-keccak", "sha-256": "binius-sha256", "blake3": "binius-blake3", "BLAKE3": "flock-blake3", "SHA-256": "flock-sha256"}[name]
            if f"\\label{{tab:{key}}}" in tb:
                tb = patch_row(tb, stock, hpanels([r[2][0] for r in runs]))
                picks = [r[2][3] if r[2][3][2] < r[2][1][2] else r[2][1] for r in runs]
                for r, pk in zip(runs, picks):
                    print(f"{name} {r[0]}: clear {r[2][1][2]:.1f} KB, bd {r[2][3][2]:.1f} KB -> {'bd' if pk is r[2][3] else 'clear'}", file=sys.stderr)
                tb = patch_row(tb, r"\ourwork", hpanels(picks))
                tb = patch_row(tb, r"\ourwork +\labrador", hpanels([r[2][2] for r in runs]))
    tables[i] = tb
open(TEX, "w").write("".join(tables))
