#pragma once
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "transpose.hxx"
#include "dynamic.hxx"
#include "pagerank.hxx"
#include "pagerankSeq.hxx"

using std::vector;
using std::swap;





template <bool O, bool D, class T>
int pagerankMonolithicSeqLoopU(vector<T>& a, vector<T>& r, vector<T>& c, const vector<T>& f, const vector<int>& vfrom, const vector<int>& efrom, const vector<int>& vdata, int i, int n, int N, T p, T E, int L, int EF) {
int l = 0;
while (!O && l<L) {
T c0 = D? pagerankTeleport(r, vdata, N, p) : (1-p)/N;
pagerankCalculateW(a, c, vfrom, efrom, i, n, c0);  
multiplyValuesW(c, a, f, i, n);                    
T el = pagerankError(a, r, i, n, EF);              
swap(a, r); ++l;                                   
if (el<E) break;                                   
}
while (O && l<L) {
T c0 = D? pagerankTeleport(r, vdata, N, p) : (1-p)/N;
pagerankCalculateOrderedU(a, r, f, vfrom, efrom, i, n, c0);  
T el = pagerankError(a, i, n, EF); ++l;            
if (el<E) break;                                   
}
return l;
}





template <bool O, bool D, class G, class H, class T=float>
PagerankResult<T> pagerankMonolithicSeq(const G& x, const H& xt, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}) {
int  N  = xt.order();  if (N==0) return PagerankResult<T>::initial(xt, q);
auto ks = vertexKeys(xt);
return pagerankSeq(xt, ks, 0, N, pagerankMonolithicSeqLoopU<O, D, T>, q, o);
}

template <bool O, bool D, class G, class T=float>
PagerankResult<T> pagerankMonolithicSeq(const G& x, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}) {
auto xt = transposeWithDegree(x);
return pagerankMonolithicSeq<O, D>(x, xt, q, o);
}





template <bool O, bool D, class G, class H, class T=float>
PagerankResult<T> pagerankMonolithicSeqDynamic(const G& x, const H& xt, const G& y, const H& yt, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}) {
int  N = yt.order();                             if (N==0) return PagerankResult<T>::initial(yt, q);
auto [ks, n] = dynamicInVertices(x, xt, y, yt);  if (n==0) return PagerankResult<T>::initial(yt, q);
return pagerankSeq(yt, ks, 0, n, pagerankMonolithicSeqLoopU<O, D, T>, q, o);
}

template <bool O, bool D, class G, class T=float>
PagerankResult<T> pagerankMonolithicSeqDynamic(const G& x, const G& y, const vector<T> *q=nullptr, const PagerankOptions<T>& o={}) {
auto xt = transposeWithDegree(x);
auto yt = transposeWithDegree(y);
return pagerankMonolithicSeqDynamic<O, D>(x, xt, y, yt, q, o);
}