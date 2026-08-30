# Numerical check of PLAN-labrador.md section 3: the chunk/diagonal/carry encoding of a product in
# S = Z[Y]/(Y^162 - Y^81 + 1) using exact products of a 54-coefficient witness chunk with a
# 9-coefficient public sub-chunk (degree <= 61 < 64: no negacyclic wrap), and of the R_648 twist.
import random
random.seed(1)
D=162; C=54; P=9; NCH=D//C; NSUB=D//P   # 3 witness chunks, 18 public sub-chunks

def polymul(a,b):
    r=[0]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        if x:
            for j,y in enumerate(b): r[i+j]+=x*y
    return r
def mod_phi(p):           # reduce mod Y^162 - Y^81 + 1 : Y^162 = Y^81 - 1
    p=p[:]
    for t in range(len(p)-1,D-1,-1):
        c=p[t]; p[t]=0; p[t-81]+=c; p[t-162]-=c
    return p[:D]+[0]*(D-len(p))
def shift_reduce(a,s):    # a * Y^s mod Phi
    return mod_phi([0]*s+a)

for trial in range(20):
    A=[random.randint(-1944,1944) for _ in range(D)]          # public, centred mod 3889
    v=[random.randint(-300,300) for _ in range(D)]            # witness chunk source
    q=3889
    true=mod_phi(polymul(A,v))
    # public data: G_b = A * Y^{54 b} mod Phi, sub-chunked
    G=[shift_reduce(A,C*b) for b in range(NCH)]
    sub=lambda g,a: g[P*a:P*a+P]
    vch=[v[C*b:C*b+C] for b in range(NCH)]
    # diagonals D_a (exact, degree <= 8+53 = 61)
    Dg=[]
    for a in range(NSUB):
        acc=[0]*(P+C-1)
        for b in range(NCH):
            pr=polymul(sub(G[b],a),vch[b])
            for i,x in enumerate(pr): acc[i]+=x
        assert len(acc)==62
        Dg.append(acc)
    # unreduced P(Y) = sum_a Y^{9a} D_a ; its part >= 162 is the honest last carry
    Pun=[0]*(P*(NSUB-1)+62)
    for a in range(NSUB):
        for i,x in enumerate(Dg[a]): Pun[P*a+i]+=x
    e17=Pun[D:]+[0]*(53-len(Pun[D:]))
    assert len(e17)==53
    # k = (true product - (true mod q)) / q : here we take the "output" to be y = true mod q centred
    y=[((t+q//2)%q)-q//2 for t in true]
    k=[(t-yy)//q for t,yy in zip(true,y)]
    assert all((t-yy)%q==0 for t,yy in zip(true,y))
    kch=[k[C*b:C*b+C] for b in range(NCH)]
    # chain
    carry=[0]*53; out=[]
    carries=[]
    for a in range(NSUB):
        T=[0]*64
        for i,x in enumerate(Dg[a]): T[i]+=x
        for i,x in enumerate(carry): T[i]+=x
        if a==0:
            for i,x in enumerate(e17): T[i]-=x
        if a==9:
            for i,x in enumerate(e17): T[i]+=x
        if a%6==0:                       # k chunk enters whole with phi = -q
            for i,x in enumerate(kch[a//6]): T[i]-=q*x
        assert all(x==0 for x in T[62:]), "degree bound"
        win=T[:P]; carry=T[P:P+53]+[0]*0
        assert len(carry)==53
        out.append(win); carries.append(carry)
    # the chain's own last carry must equal the injected e17 (consistency of the cyclic wrap)
    assert carries[NSUB-1]==e17, (carries[NSUB-1][:5],e17[:5])
    # output windows must reproduce y = true mod q, i.e. every window equals y's chunk
    rec=[x for w in out for x in w]
    assert rec==y, "chain output != A v mod q"
print("chain encoding: 20 random trials OK (degree <= 61, carries 53 coefficients, wrap consistent, output = A v mod q)")

# the R_648 = S[X]/(X^4 - Y) twist: component m of A*v equals sum_{k+l=m} A_k v_l + Y sum_{k+l=m+4} A_k v_l
N=648
def mod_1944(p):
    p=p[:]
    for t in range(len(p)-1,N-1,-1):
        c=p[t]; p[t]=0; p[t-324]+=c; p[t-648]-=c
    return p[:N]+[0]*(N-len(p))
def comps(a): return [[a[4*m+l] for m in range(D)] for l in range(4)]
for trial in range(5):
    A=[random.randint(-1944,1944) for _ in range(N)]; v=[random.randint(-300,300) for _ in range(N)]
    true=comps(mod_1944(polymul(A,v)))
    Ak=comps(A); vl=comps(v)
    for m in range(4):
        acc=[0]*D
        for k in range(4):
            for l in range(4):
                if (k+l)%4!=m: continue
                pr=mod_phi(polymul(Ak[k],vl[l]))
                if k+l>=4: pr=shift_reduce(pr,1)
                for i,x in enumerate(pr): acc[i]+=x
        assert acc==true[m]
print("R_648 twist: component formula OK")
