import torch
words = open("names.txt").read().splitlines()
chars = sorted(set(''.join(words)))
stoi = {s:i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
V = len(stoi)

g = torch.Generator().manual_seed(2147483647)
perm = torch.randperm(len(words), generator=g).tolist()
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
train = [words[i] for i in perm[:n1]]
dev = [words[i] for i in perm[n1: n2]]
test = [words[i] for i in perm[n2:]]

def bigram(ws):
    for w in ws:
        chs = ['.'] + list(w) + ['.']
        yield from ((stoi[a], stoi[b]) for a, b in zip(chs, chs[1:]))

def trigram(ws):
    for w in ws:
        chs = ['.', '.'] + list(w) + ['.']
        yield from ((stoi[a], stoi[b], stoi[c]) for a, b, c in zip(chs, chs[1:], chs[2:]))

bi = {s: torch.tensor(list(bigram(ws))).T for s, ws in [('tr', train), ('dev', dev), ('te', test)]}
tri = {s: torch.tensor(list(trigram(ws))).T for s, ws in [('tr', train), ('dev', dev), ('te', test)]}

Nb = torch.zeros((V, V))
Nb.index_put_(tuple(bi['tr']), torch.ones(bi['tr'].shape[1]), accumulate=True)
Nt = torch.zeros((V, V, V))
Nt.index_put_(tuple(tri['tr']), torch.ones(tri['tr'].shape[1]), accumulate=True)

def probs(N, alpha, dim):
    P = N + alpha
    return P/P.sum(dim, keepdim=True)

def nll(P, ix):
    return -P[tuple(ix)].log().mean().item()

Pb, Pt = probs(Nb, 1.0, 1), probs(Nt, 1.0, 2)

for s in ('tr', 'dev', 'te'):
    print(f"{s}: bigram: {nll(Pb, bi[s]): .4f} trigram: {nll(Pt, tri[s]): .4f}")

for a in [0.001, 0.01, 0.1, 0.3, 1, 3, 10, 100]:
    Pt = probs(Nt, a, 2)
    print(f"alpha={a:>6}: train {nll(Pt, tri['tr']):.4f}  dev {nll(Pt, tri['dev']):.4f}")

best = min([0.001,0.01,0.1,0.3,1,3,10,100], key=lambda a: nll(probs(Nt,a,2), tri['dev']))
print("best alpha:", best, "TEST:", nll(probs(Nt,best,2), tri['te']))  # touch test ONCE
