#include "estimator/quadrature.h"
#include "estimator/hermite.h"
#include "estimator/numeric.h"

#include <algorithm>
#include <cstdint>
#include "estimator/profile.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace estimator {

constexpr int kLanes=4;   // mode equations solved together (see the lane loop below)


namespace {
double rotation_norm(double x, double y) {
    // Jacobi entries are normally small. Avoid general hypot's scaling work
    // there, retaining it for extreme magnitudes that could overflow/underflow.
    const double magnitude=std::max(std::abs(x),std::abs(y));
    if (magnitude>=1e-150 && magnitude<=1e150) return std::sqrt(x*x+y*y);
    return std::hypot(x,y);
}
double logistic(double x) {
    if (x >= 0) return 1.0 / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1.0 + e);
}


// Implicit QL on a symmetric tridiagonal matrix. Only the first eigenvector
// row is needed: its squared entries are the Gaussian integration weights.
// Rows with all n+1 outcomes available never come here: the caller enumerates
// them (Gaussian quadrature is then exactly binomial enumeration).
bool nodes(int q, double n, double p, double* d, double* e, double* z) {
    for (int j=0; j<q; ++j) {
        d[j] = n*p + (1-2*p)*j;
        e[j] = j+1<q ? std::sqrt(p*(1-p)*(j+1)*(n-j)) : 0.;
        z[j] = j==0 ? 1. : 0.;
    }
    const double eps = std::numeric_limits<double>::epsilon();
    for (int l=0; l<q; ++l) {
        int iterations=0;
        for (;;) {
            int m=l;
            while (m<q-1 && std::abs(e[m]) > eps*(std::abs(d[m])+std::abs(d[m+1]))) ++m;
            if (m==l) break;
            if (++iterations>100) return false;
            double g=(d[l+1]-d[l])/(2*e[l]);
            double r=rotation_norm(g,1.);
            g=d[m]-d[l]+e[l]/(g+std::copysign(r,g));
            double s=1.,c=1.,shift=0.;
            for (int i=m-1; i>=l; --i) {
                const double f=s*e[i], b=c*e[i];
                // This form also handles a zero rotation without division 0/0.
                r=rotation_norm(f,g);
                e[i+1]=r;
                if (r==0.) { c=1.; s=0.; }
                else { c=g/r; s=f/r; }
                g=d[i+1]-shift;
                r=(d[i]-g)*s+2*c*b;
                shift=s*r;
                d[i+1]=g+shift;
                g=c*r-b;
                const double zi=z[i+1];
                z[i+1]=s*z[i]+c*zi;
                z[i]=c*z[i]-s*zi;
            }
            d[l]-=shift;
            e[l]=g;
            e[m]=0.;
        }
    }
    for (int i=0; i<q; ++i) {
        int k=i;
        for (int j=i+1; j<q; ++j) if (d[j]<d[k]) k=j;
        std::swap(d[i],d[k]); std::swap(z[i],z[k]);
    }
    return true;
}
}


int pool_quadrature_moments(
    int rows, const double* n, const double* a, const double* a1,
    double variance, int count_order, int pool_order,
    const double* gh_nodes, const double* gh_weights, int threads, double* output) {
    // Output columns: mean, second, fourth variance, weight, sensitivity,
    // mean derivative, information, mean effective curvature.
    const int qmax=std::max(count_order,kEnumerationLimit+1);
    std::vector<double> d(qmax),e(qmax),z(qmax);
    const int size=std::max(count_order*pool_order,kEnumerationLimit+1);
    std::vector<double> residual(size),weight(size),generation(size),squared(size);
    std::vector<double> probability(pool_order), ratio(qmax);
    int status=0;
    #pragma omp parallel for num_threads(threads) schedule(dynamic,256) reduction(max:status) firstprivate(d,e,z,residual,weight,generation,squared,probability,ratio)
    for (int i=0; i<rows; ++i) {
        const int q= (n[i]<=kEnumerationLimit) ? static_cast<int>(n[i])+1
                   : static_cast<int>(std::min(static_cast<double>(count_order),n[i]+1));
        double denominator=0.,numerator=0.;
        for (int g=0; g<pool_order; ++g) {
            const double p=logistic(a[i]+std::sqrt(2*variance)*gh_nodes[g]);
            probability[g]=p;
            const double pp=p*(1-p)*gh_weights[g];
            denominator+=pp; numerator+=pp*(1-2*p);
        }
        if (!(denominator>0)) { status=std::max(status,1); continue; }
        const double av=-.5*numerator/denominator;
        double mean=0.,effective_mean=0.,covariance=0.;
        int index=0;
        bool failed=false;
        const bool enumerate=n[i]==q-1 && n[i]<=kEnumerationLimit;
        if (enumerate) {
            std::fill_n(weight.data(),q,0.);
            std::fill_n(generation.data(),q,0.);
            std::fill_n(squared.data(),q,0.);
            for (int j=0; j<q; ++j) ratio[j]=(n[i]-j)/(j+1);
        }
        for (int g=0; g<pool_order; ++g) {
            const double p=probability[g];
            const double generating_effect=std::sqrt(2*variance)*gh_nodes[g];
            const double generating_information=n[i]*p*(1-p);
            if (enumerate) {
                // All nodes are integer counts here. The fitted mode depends
                // on that count and a, not on the generating pool-effect node.
                // Integrate its weights/scores first and solve each mode once.
                // Binomial enumeration, recurred from the more probable endpoint
                // to avoid tiny initial weights when p is near one, with the count
                // ratios (n-j)/(j+1) taken from the per-row table and the
                // accumulation indexed directly, ascending (p <= 1/2) or
                // descending (p > 1/2).
                const double small=std::min(p,1-p);
                const double odds=small/(1-small);
                double running=std::pow(1-small,n[i]);
                for (int j=0; j<q; ++j) { z[j]=running; running*=ratio[j]*odds; }   // the probability itself (no sqrt round trip)
                const double gw=gh_weights[g], np=n[i]*p;
                if (p>.5) {
                    const int ni=static_cast<int>(n[i]);
                    for (int j=0; j<q; ++j) {
                        const int k=ni-j;
                        const double w=z[j]*gw;
                        const double raw=(n[i]-j)-np;
                        weight[k]+=w;
                        generation[k]+=w*(av*raw+.5*(raw*raw-generating_information));
                        squared[k]+=w*raw;
                    }
                } else {
                    for (int j=0; j<q; ++j) {
                        const double w=z[j]*gw;
                        const double raw=j-np;
                        weight[j]+=w;
                        generation[j]+=w*(av*raw+.5*(raw*raw-generating_information));
                        squared[j]+=w*raw;
                    }
                }
                continue;
            }
            if (!nodes(q,n[i],p,d.data(),e.data(),z.data())) { status=std::max(status,2); failed=true; break; }
            // The q mode equations are independent; solving them in lanes lets the
            // processor overlap their exp latencies. Each lane runs exactly the
            // scalar iteration it would run alone, so the values are unchanged.
            for (int j0=0; j0<q; j0+=kLanes) {
                const int lanes=std::min(kLanes,q-j0);
                double u[kLanes]; bool done[kLanes];
                for (int l=0; l<lanes; ++l) {
                    // A local Gaussian start at the generating node. This changes
                    // only initialization; the identical nonlinear mode equation
                    // is still solved to the same tolerance below.
                    u[l]=variance*(generating_information*generating_effect+d[j0+l]-n[i]*p) /
                         (1+variance*generating_information);
                    done[l]=false;
                }
                for (int iteration=0; iteration<100; ++iteration) {
                    bool all=true;
                    for (int l=0; l<lanes; ++l) {
                        if (done[l]) continue;
                        const double fitted=logistic(a[i]+u[l]);
                        const double step=(u[l]-variance*(d[j0+l]-n[i]*fitted)) /
                                          (1+variance*n[i]*fitted*(1-fitted));
                        u[l]-=std::clamp(step,-2.,2.);
                        if (std::abs(step)<1e-11) done[l]=true; else all=false;
                    }
                    if (all) break;
                }
                for (int l=0; l<lanes; ++l) if (!done[l]) { status=std::max(status,3); failed=true; }
                if (failed) break;
                for (int l=0; l<lanes; ++l,++index) {
                const int j=j0+l;
                const double fitted=logistic(a[i]+u[l]);
                const double r=d[j]-n[i]*fitted;
                const double raw=d[j]-n[i]*p;
                const double w=z[j]*z[j]*gh_weights[g];
                const double information=n[i]*fitted*(1-fitted);
                residual[index]=r; weight[index]=w;
                generation[index]=av*raw+.5*(raw*raw-n[i]*p*(1-p));
                mean+=w*r;
                effective_mean+=w*information/(1+variance*information);
                covariance+=w*r*raw;
                }
            }
            if (failed) break;
        }
        if (failed) continue;
        if (enumerate) {
            index=q;
            const double p0=logistic(a[i]);
            // The pool's counts reduce to the contiguous range carrying more than
            // kEnumerationPrune of its weight: about half of all counts sit in the far binomial
            // tail, where their modes and terms change nothing at double precision. Outside it the
            // weight and both scores are zero, so every sum below runs over the solved counts only.
            double total=0.;
            for (int k=0; k<q; ++k) total+=weight[k];
            int lo=0, hi=q-1;
            while (lo<hi && !(weight[lo]>kEnumerationPrune*total)) ++lo;
            while (hi>lo && !(weight[hi]>kEnumerationPrune*total)) --hi;
            for (int k=0; k<q; ++k)
                if (k<lo || k>hi) { weight[k]=0.; generation[k]=0.; squared[k]=0.; residual[k]=0.; }
            // Each lane group starts from the previous group's last mode, stepped along the
            // mode's slope in the count there (du/dk = variance/(1+variance*information)): the
            // modes are smooth in k, so this start is far closer than the Gaussian one.
            double u_last=0., slope_last=0.; bool have_last=false;
            for (int k0=lo; k0<=hi; k0+=kLanes) {
                // lanes again: independent count modes, unchanged per-lane arithmetic
                const int lanes=std::min(kLanes,hi+1-k0);
                double u[kLanes]; bool done[kLanes];
                for (int l=0; l<lanes; ++l) {
                    u[l]=have_last ? u_last+(l+1)*slope_last
                                   : variance*((k0+l)-n[i]*p0)/(1+variance*n[i]*p0*(1-p0));
                    done[l]=false;
                }
                for (int iteration=0; iteration<100; ++iteration) {
                    bool all=true;
                    for (int l=0; l<lanes; ++l) {
                        if (done[l]) continue;
                        const double fitted=logistic(a[i]+u[l]);
                        const double step=(u[l]-variance*((k0+l)-n[i]*fitted))/(1+variance*n[i]*fitted*(1-fitted));
                        u[l]-=std::clamp(step,-2.,2.);
                        if (std::abs(step)<1e-11) done[l]=true; else all=false;
                    }
                    if (all) break;
                }
                for (int l=0; l<lanes; ++l) if (!done[l]) { status=std::max(status,3); failed=true; }
                if (failed) break;
                for (int l=0; l<lanes; ++l) {
                const int k=k0+l;
                const double fitted=logistic(a[i]+u[l]);
                const double information=n[i]*fitted*(1-fitted);
                residual[k]=k-n[i]*fitted;
                mean+=weight[k]*residual[k];
                effective_mean+=weight[k]*information/(1+variance*information);
                covariance+=residual[k]*squared[k];
                if (l==lanes-1) { u_last=u[l]; slope_last=variance/(1+variance*information); have_last=true; }
                }
            }
        }
        if (failed) continue;
        double second=0.,sensitivity=0.,fourth=0.;
        for (int j=0; j<index; ++j) {
            squared[j]=(residual[j]-mean)*(residual[j]-mean);
            second+=weight[j]*squared[j];
            sensitivity+=squared[j]*(enumerate ? generation[j] : weight[j]*generation[j]);
        }
        for (int j=0; j<index; ++j) {
            const double dev=squared[j]-second;
            fourth+=weight[j]*dev*dev;
        }
        if (n[i]<=1) sensitivity=0.;
        double* out=output+8*i;
        out[0]=mean; out[1]=second; out[2]=fourth;
        out[3]=fourth>1e-24 ? sensitivity/fourth : 0.;
        out[4]=sensitivity;
        out[5]=a1[i]*(-effective_mean+covariance);
        out[6]=a1[i]*a1[i]*covariance;
        out[7]=effective_mean;
        for (int j=0; j<8; ++j) if (!std::isfinite(out[j])) status=std::max(status,4);
    }
    return status;
}


int mean_neutral_rows(
    int rows, const double* eta, const double* sd, int order,
    const double* gh_nodes, const double* gh_weights, int threads, double* output) {
    // Same iteration as pymesh/estimator/neutral.py's mean_neutral, row by row: solve on the
    // rare-event side, Newton on the location with steps clipped to [-2, 2],
    // stop when |step| < 2e-12, at most 40 iterations. Output columns:
    // A, a1, a2. sd must be positive for every row (zero rows are
    // handled by the caller exactly as in the NumPy reference).
    int status=0;
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(max:status)
    for (int i=0; i<rows; ++i) {
        const double sign = eta[i]>0 ? -1. : 1.;
        const double work = eta[i]*sign;
        const double target = logistic(work);
        const double scale = std::sqrt(2.)*sd[i];
        double A = work-.5*sd[i]*sd[i];
        bool converged=false;
        for (int iteration=0; iteration<40; ++iteration) {
            double m1=0., mean=0.;
            for (int g=0; g<order; ++g) {
                const double p=logistic(A+scale*gh_nodes[g]);
                m1+=p*(1-p)*gh_weights[g];
                mean+=p*gh_weights[g];
            }
            const double step=(mean-target)/m1;
            A-=std::clamp(step,-2.,2.);
            if (std::abs(step)<2e-12) { converged=true; break; }
        }
        if (!converged) { status=std::max(status,5); continue; }
        double m1=0., m2=0.;
        for (int g=0; g<order; ++g) {
            const double p=logistic(A+scale*gh_nodes[g]);
            const double v=p*(1-p);
            m1+=v*gh_weights[g];
            m2+=v*(1-2*p)*gh_weights[g];
        }
        const double t1=target*(1-target), t2=t1*(1-2*target);
        const double a1=t1/m1;
        const double a2=(t2-m2*a1*a1)/m1;
        double* out=output+3*i;
        out[0]=sign*A; out[1]=a1; out[2]=sign*a2;
        for (int j=0; j<3; ++j) if (!std::isfinite(out[j])) status=std::max(status,4);
    }
    return status;
}



void Moments::resize(std::size_t n) {
    for (auto* v : {&mean, &second, &weight, &sensitivity, &mean_derivative, &information, &a1, &a2, &location}) v->assign(n, 0.);
}

void Moments::assign_rows(const std::vector<int64_t>& rows, const Moments& source) {
    std::vector<double>* mine[] = {&mean, &second, &weight, &sensitivity, &mean_derivative, &information, &a1, &a2, &location};
    const std::vector<double>* theirs[] = {&source.mean, &source.second, &source.weight, &source.sensitivity,
                                           &source.mean_derivative, &source.information, &source.a1, &source.a2, &source.location};
    for (int f = 0; f < 9; ++f)
        for (std::size_t r = 0; r < rows.size(); ++r) (*mine[f])[static_cast<std::size_t>(rows[r])] = (*theirs[f])[r];
}

int threads_setting() {
    // OMP_NUM_THREADS; unset, one per physical core (logical CPUs over two, for SMT): on an
    // 8-core/16-thread machine 8 threads fit the benchmark 30% faster than 4, and 16 are slower
    // again. The dumps do not depend on the count.
    if (const char* s = std::getenv("OMP_NUM_THREADS")) return std::max(1, std::atoi(s));
    const unsigned logical = std::thread::hardware_concurrency();
    return logical >= 2 ? static_cast<int>(logical / 2) : 1;
}

int pool_order_for(double variance, int base) {
    // the adaptive pool order: fewer Gauss-Hermite nodes where the pool effect is small
    if (variance <= .2) return std::min(base, 8);
    if (variance <= .5) return std::min(base, 12);
    if (variance <= 1.) return std::min(base, 16);
    return base;
}

int neutral_order_for(double variance) {
    // Gauss-Hermite nodes for the mean-neutral location: the fewest of the tabulated rules that
    // agree with an 80-node rule to 1e-14 relative on the benchmark's locations and a sweep of
    // logits from -12 to 4 (measured 2026-09-22: 24 nodes to variance 0.35, 32 to 0.5); above
    // that, 40 as before (1.3e-13 at variance 1).
    if (variance <= .35) return 24;
    if (variance <= .5) return 32;
    return 40;
}

int surface_count_order() { return 6; }   // count nodes for the surface solve; the variance side uses 8

namespace {
// The fit evaluates the moments at the same surface and variance with the surface's count order
// (6) and the variance side's (8). Only rows too large to enumerate (n > kEnumerationLimit)
// depend on the count order, and each row is computed on its own, so a recent result at the same
// inputs supplies every other row, the mean-neutral locations included, exactly.
// Two entries catch two of the three such pairs on the benchmark (four catch all three for no
// further speed); about 3% of a fit, 4 MB. The port does not do this: there the copies cost
// 15-20 MB for a 2-4% gain.
struct MomentsCacheEntry {
    std::vector<double> eta, n;
    double variance; int count_order, pool_order;
    Moments result;
};
std::mutex moments_cache_mutex;
std::deque<MomentsCacheEntry> moments_cache;
constexpr std::size_t kMomentsCacheEntries = 2, kMomentsCacheMinRows = 4096;

bool same_bytes(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

// Recompute the rows too large to enumerate at this count order, in place.
void refresh_large_rows(Moments& out, const std::vector<double>& n, double variance, int count_order,
                        int pool_order, int threads) {
    std::vector<std::size_t> large;
    for (std::size_t i = 0; i < n.size(); ++i) if (n[i] > kEnumerationLimit) large.push_back(i);
    if (large.empty()) return;
    const HermiteRule rule = hermite_rule(pool_order);
    std::vector<double> nn(large.size()), loc(large.size()), a1(large.size()), output(8 * large.size());
    for (std::size_t r = 0; r < large.size(); ++r) { nn[r] = n[large[r]]; loc[r] = out.location[large[r]]; a1[r] = out.a1[large[r]]; }
    int status;
    { ScopeTimer t("    pool_quadrature"); status = pool_quadrature_moments(static_cast<int>(large.size()), nn.data(), loc.data(), a1.data(), variance,
                                     count_order, pool_order, rule.nodes, rule.weights, threads, output.data()); }
    if (status) throw std::runtime_error("native pool quadrature failed (status " + std::to_string(status) + ")");
    for (std::size_t r = 0; r < large.size(); ++r) {
        const double* o = output.data() + 8 * r; const std::size_t i = large[r];
        out.mean[i] = o[0]; out.second[i] = o[1]; out.weight[i] = o[3];
        out.sensitivity[i] = o[4]; out.mean_derivative[i] = o[5]; out.information[i] = o[6];
    }
}
}  // namespace

Moments moments_uncached(const std::vector<double>& eta, const std::vector<double>& n, double variance,
                         int count_order, int pool_order, int threads);

Moments moments(const std::vector<double>& eta, const std::vector<double>& n, double variance, int count_order,
                int pool_order, int threads) {
    if (eta.size() < kMomentsCacheMinRows || variance == 0)
        return moments_uncached(eta, n, variance, count_order, pool_order, threads);
    Moments out;
    bool found = false, exact = false;
    {
        std::lock_guard<std::mutex> lock(moments_cache_mutex);
        for (const auto& entry : moments_cache)
            if (std::memcmp(&entry.variance, &variance, sizeof(double)) == 0 && entry.pool_order == pool_order
                && same_bytes(entry.eta, eta) && same_bytes(entry.n, n)) {
                out = entry.result; found = true; exact = entry.count_order == count_order;
                if (exact) break;
            }
    }
    if (found && exact) return out;
    if (found) refresh_large_rows(out, n, variance, count_order, pool_order, threads);
    else out = moments_uncached(eta, n, variance, count_order, pool_order, threads);
    std::lock_guard<std::mutex> lock(moments_cache_mutex);
    moments_cache.push_front({eta, n, variance, count_order, pool_order, out});
    if (moments_cache.size() > kMomentsCacheEntries) moments_cache.pop_back();
    return out;
}

Moments moments_uncached(const std::vector<double>& eta, const std::vector<double>& n, double variance,
                         int count_order, int pool_order, int threads) {
    const std::size_t rows = eta.size();
    Moments out;
    out.resize(rows);
    if (variance == 0) {
        // exact binomial moments, as pool.moments' zero-variance branch (numpy elementwise)
        for (std::size_t i = 0; i < rows; ++i) {
            const double p = expit(eta[i]);
            const double pq = p * (1 - p);
            const double second = n[i] * pq;
            // numpy's association: 2*second**2 is 2*(second*second); n*(n-1)*pq**2 is (n*(n-1))*(pq*pq)
            const double fourth = std::max(2 * (second * second) + second * (1 - 6 * pq), 0.);
            const double sensitivity = (n[i] * (n[i] - 1)) * (pq * pq);
            out.mean[i] = 0.; out.second[i] = second;
            out.weight[i] = fourth > 1e-24 ? sensitivity / fourth : 0.;
            out.sensitivity[i] = sensitivity; out.mean_derivative[i] = 0.; out.information[i] = second;
            out.a1[i] = 1.; out.a2[i] = 0.; out.location[i] = eta[i];
        }
        return out;
    }
    const int neutral_order = std::max(neutral_order_for(variance), pool_order);
    const HermiteRule neutral_rule = hermite_rule(neutral_order);
    const HermiteRule rule = hermite_rule(pool_order);
    if (neutral_rule.nodes == nullptr || rule.nodes == nullptr) throw std::invalid_argument("no Gauss-Hermite rule of that order");
    // The kernels write interleaved columns per row; they run over chunks of rows so the
    // interleaved blocks stay small (every row is independent, so the chunking changes nothing).
    constexpr std::size_t kChunk = 32768;
    std::vector<double> sd(std::min(rows, kChunk), std::sqrt(variance)), neutral(std::min(rows, kChunk) * 3), output(std::min(rows, kChunk) * 8);
    for (std::size_t start = 0; start < rows; start += kChunk) {
        const std::size_t count = std::min(kChunk, rows - start);
        int status;
        { ScopeTimer t("    mean_neutral"); status = mean_neutral_rows(static_cast<int>(count), eta.data() + start, sd.data(), neutral_order, neutral_rule.nodes,
                                       neutral_rule.weights, threads, neutral.data()); }
        if (status) throw std::runtime_error("native mean-neutral location failed (status " + std::to_string(status) + ")");
        for (std::size_t i = 0; i < count; ++i) { out.location[start + i] = neutral[3 * i]; out.a1[start + i] = neutral[3 * i + 1]; out.a2[start + i] = neutral[3 * i + 2]; }
        { ScopeTimer t("    pool_quadrature"); status = pool_quadrature_moments(static_cast<int>(count), n.data() + start, out.location.data() + start, out.a1.data() + start, variance, count_order,
                                         pool_order, rule.nodes, rule.weights, threads, output.data()); }
        if (status) throw std::runtime_error("native pool quadrature failed (status " + std::to_string(status) + ")");
        for (std::size_t i = 0; i < count; ++i) {
            const double* o = output.data() + 8 * i;
            out.mean[start + i] = o[0]; out.second[start + i] = o[1]; out.weight[start + i] = o[3];
            out.sensitivity[start + i] = o[4]; out.mean_derivative[start + i] = o[5]; out.information[start + i] = o[6];
        }
    }
    return out;
}

void scaled_mode(const std::vector<double>& a, const std::vector<double>& n, const std::vector<double>& k,
                 double variance, std::vector<double>& residual, std::vector<double>& effective) {
    // the vectorised iteration: every row takes every step until the largest step is small
    const std::size_t rows = a.size();
    std::vector<double> u(rows, 0.);
    bool converged = false;
    for (int iteration = 0; iteration < 100; ++iteration) {
        double largest = 0.;
        for (std::size_t i = 0; i < rows; ++i) {
            const double p = expit(a[i] + u[i]);
            const double step = (u[i] - variance * (k[i] - n[i] * p)) / (1 + variance * n[i] * p * (1 - p));
            u[i] -= std::clamp(step, -2., 2.);
            largest = std::max(largest, std::fabs(step));
        }
        if (largest < 1e-11) { converged = true; break; }
    }
    if (!converged) throw std::runtime_error("quadrature-node mode failed");
    residual.resize(rows); effective.resize(rows);
    for (std::size_t i = 0; i < rows; ++i) {
        const double p = expit(a[i] + u[i]);
        residual[i] = k[i] - n[i] * p;
        effective[i] = n[i] * p * (1 - p) / (1 + variance * n[i] * p * (1 - p));
    }
}

}  // namespace estimator
