"""Static checks on main.tex: reorder references by first citation, word counts, balance, unresolved keys."""
import re, sys
p = "../main.tex"; s = open(p, encoding="utf-8").read()
a, b = s.index("%REFSTART"), s.index("%REFEND")
body = s[:a]
items = re.split(r"\n(?=\\bibitem\{)", s[a + len("%REFSTART"):b].strip())
keys = [re.match(r"\\bibitem\{([^}]+)\}", it).group(1) for it in items]
order = []
for m in re.finditer(r"\\cite\{([^}]+)\}", body):
    for k in [x.strip() for x in m.group(1).split(",")]:
        if k not in order:
            order.append(k)
missing = [k for k in order if k not in keys]; unused = [k for k in keys if k not in order]
print("cited:", len(order), "bibitems:", len(keys), "| cited but missing:", missing, "| listed but uncited:", unused)
new = [items[keys.index(k)].strip() for k in order if k in keys] + [items[keys.index(k)].strip() for k in unused]
s = s[:a] + "%REFSTART\n" + "\n\n".join(new) + "\n" + s[b:]
open(p, "w", encoding="utf-8").write(s)

def strip(t):
    t = re.sub(r"(?s)\\begin\{(table|figure|equation)\}.*?\\end\{\1\}", " ", t)
    t = re.sub(r"(?m)^%.*$", "", t)
    t = re.sub(r"\\(cite|ref|label|input|url)\{[^}]*\}", " ", t)
    t = re.sub(r"\\authorinput\{[^{}]*(\{[^{}]*\}[^{}]*)*\}", " ", t)
    t = re.sub(r"\\verifyraw\{[^{}]*(\{[^{}]*\}[^{}]*)*\}", " ", t)
    t = re.sub(r"\$[^$]*\$", " X ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", t)
    t = re.sub(r"[{}~]", " ", t)
    return t

def sec(name, nxt):
    i = s.index("\\section*{%s}" % name); j = s.index("\\section*{%s}" % nxt)
    return s[i:j]

wc = lambda t: len(re.findall(r"[A-Za-z0-9][A-Za-z0-9.,'\-/%]*", strip(t)))
ab = sec("Abstract", "Introduction").split("\\textbf{Keywords")[0]
print("abstract words:", wc(ab) - 1)
tot = 0
for n1, n2 in (("Introduction", "Results"), ("Results", "Discussion"), ("Discussion", "Methods")):
    w = wc(sec(n1, n2)) - 1; tot += w; print(f"{n1}: {w}")
print("main text total (Intro+Results+Discussion):", tot, "| Methods:", wc(sec("Methods", "Data availability")))
title = re.search(r"\\LARGE\\bfseries ([^\\]+)\\par", s).group(1)
print("title words:", len(title.split()), "|", title)
print("braces: { %d  } %d" % (s.count("{"), s.count("}")), "| $ count even:", s.count("$") % 2 == 0)
for env in set(re.findall(r"\\begin\{([a-z*]+)\}", s)):
    if s.count("\\begin{%s}" % env) != s.count("\\end{%s}" % env):
        print("UNBALANCED ENV", env)
labels = set(re.findall(r"\\label\{([^}]+)\}", s)); refs = set(re.findall(r"\\ref\{([^}]+)\}", s))
print("refs without label:", refs - labels, "| labels never referenced:", labels - refs)
seen = []
for x in re.findall(r"(?:Fig\.|Table)~\\ref\{((?:fig|tab):[a-z]+)\}", body):
    if x not in seen:
        seen.append(x)
print("first-citation order of display items:", seen)
print("non-ASCII characters:", sorted(set(ch for ch in s if ord(ch) > 127)))
print("placeholders: authorinput", s.count("\\authorinput{") - 1, " verifyraw", s.count("\\verifyraw{") - 1)
for f in ("../supplementary.tex",):
    t = open(f, encoding="utf-8").read()
    print(f, "braces", t.count("{"), t.count("}"), "non-ASCII", sorted(set(ch for ch in t if ord(ch) > 127)))
