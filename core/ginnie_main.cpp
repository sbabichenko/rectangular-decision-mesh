// Binomial rectangular-mesh fit of the frozen Ginnie designs at FULL
// data scale — C++ port of poc/ginnie_poc.py with the production
// triangular gate configuration (see docs/FDR_PORT.md):
//   adaptive unthresholded coarse stage (depth-capped, deviance-greedy),
//   penalized binomial IRLS with per-depth EM tau (floor 0.005),
//   Lindsey lfdr family decision (production clamps) with BH fallback,
//   two-sided candidate columns, GEV pool pricing + 0.25x parent
//   posterior variance, keff >= 0.5 and >= 10-point candidacy floors,
//   IRLS step cap 4, locally balanced WALA/WAC holdout.
//
// Usage: ginnie_rect <design.csv> <outdir> [max_admit]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "lfdr.h"
#include "rect_mesh.h"

using namespace rectmesh;

// ------------------------------ config (production triangular values)
static const double kGEV = 0.35;
static const double kTauInit = 0.5;
static const double kTauFloor = 0.005;
static const double kQ = 0.1;
static int kMaxAdmit = 200;
static const int kMaxRounds = 80;
static const int kMinPts = 10;
static const double kMinKeff = 0.5;
static const double kParentVarScale = 0.25;
static const double kIrlsStepCap = 4.0;
static const int kSeedSplits = 96;
static const int kSeedRefitEvery = 10;
static const int kSeedDepthCap = 8;  // coarse stage stays coarse (16x16 scale)

struct Data {
    std::vector<double> x, y, n, k;
    std::vector<char> hold;
};

// ------------------------------ small dense Cholesky
struct Chol {
    int p;
    std::vector<double> L;  // lower triangular, row-major
    bool factor(const std::vector<double>& A, int p_) {
        p = p_;
        L.assign((size_t)p * p, 0.0);
        for (int i = 0; i < p; ++i)
            for (int j = 0; j <= i; ++j) {
                double s = A[(size_t)i * p + j];
                for (int k2 = 0; k2 < j; ++k2)
                    s -= L[(size_t)i * p + k2] * L[(size_t)j * p + k2];
                if (i == j) {
                    if (s <= 0) return false;
                    L[(size_t)i * p + j] = std::sqrt(s);
                } else {
                    L[(size_t)i * p + j] = s / L[(size_t)j * p + j];
                }
            }
        return true;
    }
    void solve(const std::vector<double>& b, std::vector<double>* out) const {
        std::vector<double> yv(p);
        for (int i = 0; i < p; ++i) {
            double s = b[i];
            for (int j = 0; j < i; ++j) s -= L[(size_t)i * p + j] * yv[j];
            yv[i] = s / L[(size_t)i * p + i];
        }
        out->assign(p, 0.0);
        for (int i = p - 1; i >= 0; --i) {
            double s = yv[i];
            for (int j = i + 1; j < p; ++j)
                s -= L[(size_t)j * p + i] * (*out)[j];
            (*out)[i] = s / L[(size_t)i * p + i];
        }
    }
    // diagonal of A^{-1}
    void inverse_diag(std::vector<double>* out) const {
        out->assign(p, 0.0);
        std::vector<double> e(p), col;
        for (int j = 0; j < p; ++j) {
            std::fill(e.begin(), e.end(), 0.0);
            e[j] = 1.0;
            solve(e, &col);
            (*out)[j] = col[j];
        }
    }
};

static double expit(double f) {
    if (f > 12) f = 12;
    if (f < -12) f = -12;
    return 1.0 / (1.0 + std::exp(-f));
}

static double normal_sf(double z) { return 0.5 * std::erfc(z / std::sqrt(2.0)); }

// ------------------------------ fit: binomial IRLS + EM tau
struct FitState {
    std::vector<VKey> keys;
    std::unordered_map<VKey, int> kidx;
    std::map<int, double> taus;              // depth -> tau
    std::unordered_map<VKey, double> svv;    // posterior variance
    std::vector<double> delta;
};

static void fit_binom_em(RectMesh& mesh, const Data& d,
                         const std::vector<int>& tr, FitState* st,
                         int em_iters = 3, int irls_iters = 8) {
    st->keys = mesh.free_keys();
    std::sort(st->keys.begin(), st->keys.end());
    const int p = (int)st->keys.size();
    st->kidx.clear();
    for (int i = 0; i < p; ++i) st->kidx[st->keys[i]] = i;
    // sparse rows once per topology
    const size_t n = tr.size();
    std::vector<std::vector<std::pair<int, double>>> rows(n);
    {
        std::vector<std::pair<int, double>> row;
        for (size_t i = 0; i < n; ++i) {
            mesh.design_row(d.x[tr[i]], d.y[tr[i]], st->kidx, &row);
            rows[i] = row;
        }
    }
    std::vector<int> depth(p);
    for (int i = 0; i < p; ++i) depth[i] = mesh.vertex(st->keys[i])->depth;
    std::set<int> dset(depth.begin(), depth.end());
    st->taus.clear();
    for (int dd : dset)
        if (dd > 0) st->taus[dd] = kTauInit;
    if (st->delta.size() != (size_t)p) st->delta.assign(p, 0.0);
    std::vector<double> A((size_t)p * p), b(p), f(n), w(n), zw(n), diag;
    Chol ch;
    for (int em = 0; em < em_iters; ++em) {
        std::vector<double> lam(p, 0.0);
        for (int i = 0; i < p; ++i)
            if (depth[i] > 0) lam[i] = 1.0 / (st->taus[depth[i]] * st->taus[depth[i]]);
        for (int it = 0; it < irls_iters; ++it) {
            for (size_t i = 0; i < n; ++i) {
                double fi = 0;
                for (auto& e : rows[i]) fi += e.second * st->delta[e.first];
                f[i] = fi;
                const double pp = expit(fi);
                w[i] = std::max(d.n[tr[i]] * pp * (1 - pp), 1e-8);
                zw[i] = fi + (d.k[tr[i]] - d.n[tr[i]] * pp) / w[i];
            }
            std::fill(A.begin(), A.end(), 0.0);
            std::fill(b.begin(), b.end(), 0.0);
            for (size_t i = 0; i < n; ++i) {
                for (auto& e1 : rows[i]) {
                    b[e1.first] += w[i] * e1.second * zw[i];
                    for (auto& e2 : rows[i])
                        A[(size_t)e1.first * p + e2.first] +=
                            w[i] * e1.second * e2.second;
                }
            }
            for (int i = 0; i < p; ++i) A[(size_t)i * p + i] += lam[i];
            if (!ch.factor(A, p)) {
                for (int i = 0; i < p; ++i) A[(size_t)i * p + i] += 1e-6;
                ch.factor(A, p);
            }
            std::vector<double> nd;
            ch.solve(b, &nd);
            // IRLS step cap on the working scale
            double mx = 0;
            for (size_t i = 0; i < n; ++i) {
                double step = 0;
                for (auto& e : rows[i])
                    step += e.second * (nd[e.first] - st->delta[e.first]);
                mx = std::max(mx, std::fabs(step));
            }
            const double frac = mx <= kIrlsStepCap ? 1.0 : kIrlsStepCap / mx;
            for (int i = 0; i < p; ++i)
                st->delta[i] += frac * (nd[i] - st->delta[i]);
        }
        ch.inverse_diag(&diag);
        for (auto& kvt : st->taus) {
            double s = 0;
            int cnt = 0;
            for (int i = 0; i < p; ++i)
                if (depth[i] == kvt.first) {
                    s += st->delta[i] * st->delta[i] + diag[i];
                    ++cnt;
                }
            kvt.second = std::max(std::sqrt(s / std::max(cnt, 1)), kTauFloor);
        }
    }
    st->svv.clear();
    for (int i = 0; i < p; ++i) st->svv[st->keys[i]] = diag[i];
    std::unordered_map<VKey, double> dmap;
    for (int i = 0; i < p; ++i) dmap[st->keys[i]] = st->delta[i];
    mesh.resolve_heights_hier(dmap);
}

static double tau_at(const std::map<int, double>& taus, int d) {
    if (taus.empty()) return kTauInit;
    auto it = taus.find(d);
    if (it != taus.end()) return it->second;
    int bestd = taus.begin()->first;
    for (auto& kv : taus)
        if (std::abs(kv.first - d) < std::abs(bestd - d)) bestd = kv.first;
    return taus.at(bestd);
}

// ------------------------------ candidate statistics
struct CandStat {
    double beta = 0, z = 0;
    bool ok = false;
};

static CandStat cand_stats(const std::vector<double>& col,
                           const std::vector<double>& r,
                           const std::vector<double>& w, double parent_var) {
    CandStat s;
    double xtwx = 0, w2b2 = 0, xtwr = 0;
    for (size_t i = 0; i < col.size(); ++i) {
        const double b = col[i];
        if (b == 0) continue;
        xtwx += w[i] * b * b;
        w2b2 += w[i] * w[i] * b * b;
        xtwr += w[i] * b * r[i];
    }
    if (xtwx <= 0) return s;
    const double keff = xtwx * xtwx / w2b2;
    if (keff < kMinKeff) return s;
    s.beta = xtwr / xtwx;
    const double var = 1.0 / xtwx + kGEV * w2b2 / (xtwx * xtwx) +
                       kParentVarScale * parent_var;
    s.z = s.beta / std::sqrt(var);
    s.ok = true;
    return s;
}

// production family decision: Lindsey mean-lfdr prefix, BH fallback
static std::vector<int> lfdr_admit(const std::vector<double>& zs, double q) {
    if (zs.size() >= 50) {
        LfdrResult res = lindsey_lfdr(zs, true);
        if (res.used_lindsey) {
            std::vector<int> order(zs.size());
            for (size_t i = 0; i < zs.size(); ++i) order[i] = (int)i;
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return res.lfdr[a] < res.lfdr[b];
            });
            double cum = 0;
            size_t kk = 0;
            for (size_t i = 0; i < order.size(); ++i) {
                cum += res.lfdr[order[i]];
                if (cum / (i + 1) <= q) kk = i + 1;
            }
            return std::vector<int>(order.begin(), order.begin() + kk);
        }
    }
    std::vector<int> order(zs.size());
    for (size_t i = 0; i < zs.size(); ++i) order[i] = (int)i;
    std::vector<double> pv(zs.size());
    for (size_t i = 0; i < zs.size(); ++i) pv[i] = 2 * normal_sf(std::fabs(zs[i]));
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return pv[a] < pv[b]; });
    size_t kk = 0;
    for (size_t i = 0; i < order.size(); ++i)
        if (pv[order[i]] <= q * (i + 1) / order.size()) kk = i + 1;
    return std::vector<int>(order.begin(), order.begin() + kk);
}

// ------------------------------ main gate
struct Candidate {
    VKey pos;
    Cell* leaf = nullptr;  // null for existing-vertex candidates
    char axis = 0;
    CandStat st;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s design.csv outdir [max_admit]\n",
                     argv[0]);
        return 2;
    }
    if (argc > 3) kMaxAdmit = std::atoi(argv[3]);
    // ---- load
    Data d;
    {
        FILE* fp = std::fopen(argv[1], "r");
        if (!fp) { std::perror("open"); return 2; }
        char line[256];
        if (!std::fgets(line, sizeof line, fp)) return 2;  // header
        double wala, wac, n0, kt;
        while (std::fscanf(fp, "%lf,%lf,%lf,%lf", &wala, &wac, &n0, &kt) == 4) {
            d.x.push_back(wala / 360.0);
            d.y.push_back((wac - 1.8) / (9.0 - 1.8));
            d.n.push_back(n0);
            d.k.push_back(kt);
        }
        std::fclose(fp);
    }
    const size_t N = d.x.size();
    std::printf("loaded %zu pools\n", N);
    // ---- locally balanced holdout (25% within 12x12 blocks)
    d.hold.assign(N, 0);
    {
        std::mt19937 rng(7);
        std::map<std::pair<int, int>, std::vector<int>> blocks;
        for (size_t i = 0; i < N; ++i) {
            int bx = std::min(std::max(int(d.x[i] * 12), 0), 11);
            int by = std::min(std::max(int(d.y[i] * 12), 0), 11);
            blocks[{bx, by}].push_back((int)i);
        }
        for (auto& kv : blocks) {
            auto idx = kv.second;
            std::shuffle(idx.begin(), idx.end(), rng);
            const size_t nh = (size_t)std::lround(0.25 * idx.size());
            for (size_t i = 0; i < nh; ++i) d.hold[idx[i]] = 1;
        }
    }
    std::vector<int> tr, ho;
    for (size_t i = 0; i < N; ++i) (d.hold[i] ? ho : tr).push_back((int)i);

    RectMesh mesh;
    FitState st;
    // ---- adaptive unthresholded coarse stage (depth-capped)
    fit_binom_em(mesh, d, tr, &st);
    for (int s0 = 0; s0 < kSeedSplits; ++s0) {
        // working residuals at the current fit
        std::vector<double> pr(tr.size());
        for (size_t i = 0; i < tr.size(); ++i)
            pr[i] = mesh.eval(d.x[tr[i]], d.y[tr[i]]);
        double best_gain = 0;
        Cell* best_leaf = nullptr;
        char best_axis = 0;
        for (Cell* leaf : mesh.leaf_cells()) {
            if (leaf->lx + leaf->ly >= kSeedDepthCap) continue;
            double a00[2][2] = {{0, 0}, {0, 0}}, bb[2] = {0, 0};
            int cnt = 0;
            double a00y[2][2] = {{0, 0}, {0, 0}}, bby[2] = {0, 0};
            for (size_t i = 0; i < tr.size(); ++i) {
                const int ii = tr[i];
                const double xi = d.x[ii] * kS, yi = d.y[ii] * kS;
                const bool xin =
                    (leaf->x0 == 0 ? xi >= leaf->x0 : xi > leaf->x0) &&
                    xi <= leaf->x1;
                const bool yin =
                    (leaf->y0 == 0 ? yi >= leaf->y0 : yi > leaf->y0) &&
                    yi <= leaf->y1;
                if (!xin || !yin) continue;
                ++cnt;
                const double u = (xi - leaf->x0) / (leaf->x1 - leaf->x0);
                const double v = (yi - leaf->y0) / (leaf->y1 - leaf->y0);
                const double pp = expit(pr[i]);
                const double wi = std::max(d.n[ii] * pp * (1 - pp), 1e-8);
                const double ri = (d.k[ii] - d.n[ii] * pp) / wi;
                const double tx = 1 - std::fabs(2 * u - 1);
                const double ty = 1 - std::fabs(2 * v - 1);
                const double cx[2] = {tx * (1 - v), tx * v};
                const double cy[2] = {ty * (1 - u), ty * u};
                for (int a2 = 0; a2 < 2; ++a2) {
                    bb[a2] += wi * cx[a2] * ri;
                    bby[a2] += wi * cy[a2] * ri;
                    for (int b2 = 0; b2 < 2; ++b2) {
                        a00[a2][b2] += wi * cx[a2] * cx[b2];
                        a00y[a2][b2] += wi * cy[a2] * cy[b2];
                    }
                }
            }
            if (cnt < kMinPts) continue;
            auto gain2 = [](double A2[2][2], double b2[2]) {
                const double det =
                    A2[0][0] * A2[1][1] - A2[0][1] * A2[1][0] + 1e-12;
                const double i00 = A2[1][1] / det, i01 = -A2[0][1] / det,
                             i11 = A2[0][0] / det;
                return b2[0] * (i00 * b2[0] + i01 * b2[1]) +
                       b2[1] * (i01 * b2[0] + i11 * b2[1]);
            };
            const double gx = gain2(a00, bb), gy = gain2(a00y, bby);
            if (gx > best_gain) { best_gain = gx; best_leaf = leaf; best_axis = 'x'; }
            if (gy > best_gain) { best_gain = gy; best_leaf = leaf; best_axis = 'y'; }
        }
        if (!best_leaf) break;
        const int dnew = best_leaf->lx + best_leaf->ly + 1;
        mesh.split(best_leaf, best_axis, Origin::Seed);
        std::vector<VKey> to_promote;
        for (auto& kv : mesh.vertices())
            if (kv.second->depth == dnew && kv.second->origin == Origin::Seed &&
                kv.second->state != VState::Free)
                to_promote.push_back(kv.first);
        for (VKey k2 : to_promote) mesh.promote(k2);
        if ((s0 + 1) % kSeedRefitEvery == 0) fit_binom_em(mesh, d, tr, &st);
    }
    fit_binom_em(mesh, d, tr, &st);
    std::printf("seed done: %zu leaves, %zu free\n", mesh.leaf_cells().size(),
                mesh.free_keys().size());

    // ---- gate rounds
    std::vector<VKey> admitted;
    for (int rnd = 0; rnd < kMaxRounds; ++rnd) {
        if ((int)admitted.size() >= kMaxAdmit) break;
        const size_t n = tr.size();
        std::vector<double> f(n), w(n), r(n);
        for (size_t i = 0; i < n; ++i) {
            f[i] = mesh.eval(d.x[tr[i]], d.y[tr[i]]);
            const double pp = expit(f[i]);
            w[i] = std::max(d.n[tr[i]] * pp * (1 - pp), 1e-8);
            r[i] = (d.k[tr[i]] - d.n[tr[i]] * pp) / w[i];
        }
        // point -> leaf partition
        std::unordered_map<const Cell*, std::vector<int>> pts;
        for (size_t i = 0; i < n; ++i)
            pts[mesh.find_leaf(d.x[tr[i]], d.y[tr[i]])].push_back((int)i);
        auto parent_var = [&](const std::vector<VKey>& pk) {
            double s = 0;
            for (VKey q2 : pk) {
                auto it = st.svv.find(q2);
                if (it != st.svv.end()) s += it->second;
            }
            return s;
        };
        std::unordered_map<VKey, Candidate> cands;
        // new-cut candidates with two-sided columns
        std::map<std::tuple<int, int, int, int>, Cell*> leaf_by_box;
        for (Cell* l : mesh.leaf_cells())
            leaf_by_box[{l->x0, l->y0, l->x1, l->y1}] = l;
        for (Cell* leaf : mesh.leaf_cells()) {
            auto itp = pts.find(leaf);
            if (itp == pts.end() || (int)itp->second.size() < kMinPts) continue;
            for (char axis : {'x', 'y'}) {
                for (int end = 0; end < 2; ++end) {
                    VKey pos;
                    std::vector<VKey> pk;
                    if (axis == 'x') {
                        const int xm = (leaf->x0 + leaf->x1) / 2;
                        const int yy = end == 0 ? leaf->y0 : leaf->y1;
                        pos = key_of(xm, yy);
                        pk = {key_of(leaf->x0, yy), key_of(leaf->x1, yy)};
                    } else {
                        const int ym = (leaf->y0 + leaf->y1) / 2;
                        const int xx = end == 0 ? leaf->x0 : leaf->x1;
                        pos = key_of(xx, ym);
                        pk = {key_of(xx, leaf->y0), key_of(xx, leaf->y1)};
                    }
                    Vertex* vx = mesh.vertex(pos);
                    if (vx && vx->state == VState::Free) continue;
                    std::vector<double> col, rr, ww;
                    auto add_side = [&](Cell* c, bool mirrored) {
                        auto it2 = pts.find(c);
                        if (it2 == pts.end()) return;
                        for (int idx : it2->second) {
                            const int ii = tr[idx];
                            const double u =
                                (d.x[ii] * kS - c->x0) / (c->x1 - c->x0);
                            const double v =
                                (d.y[ii] * kS - c->y0) / (c->y1 - c->y0);
                            double tent, trans;
                            if (axis == 'x') {
                                tent = 1 - std::fabs(2 * u - 1);
                                trans = mirrored ? (end == 0 ? v : 1 - v)
                                                 : (end == 0 ? 1 - v : v);
                            } else {
                                tent = 1 - std::fabs(2 * v - 1);
                                trans = mirrored ? (end == 0 ? u : 1 - u)
                                                 : (end == 0 ? 1 - u : u);
                            }
                            col.push_back(tent * trans);
                            rr.push_back(r[idx]);
                            ww.push_back(w[idx]);
                        }
                    };
                    add_side(leaf, false);
                    std::tuple<int, int, int, int> nb;
                    if (axis == 'x')
                        nb = end == 0
                                 ? std::make_tuple(leaf->x0,
                                                   2 * leaf->y0 - leaf->y1,
                                                   leaf->x1, leaf->y0)
                                 : std::make_tuple(leaf->x0, leaf->y1, leaf->x1,
                                                   2 * leaf->y1 - leaf->y0);
                    else
                        nb = end == 0
                                 ? std::make_tuple(2 * leaf->x0 - leaf->x1,
                                                   leaf->y0, leaf->x0, leaf->y1)
                                 : std::make_tuple(leaf->x1, leaf->y0,
                                                   2 * leaf->x1 - leaf->x0,
                                                   leaf->y1);
                    auto itn = leaf_by_box.find(nb);
                    if (itn != leaf_by_box.end()) add_side(itn->second, true);
                    CandStat cs = cand_stats(col, rr, ww, parent_var(pk));
                    if (!cs.ok) continue;
                    auto prev = cands.find(pos);
                    if (prev == cands.end() ||
                        std::fabs(cs.z) > std::fabs(prev->second.st.z))
                        cands[pos] = {pos, leaf, axis, cs};
                }
            }
        }
        // existing non-free vertices on exact live columns
        {
            std::vector<double> xs2, ys2;
            for (size_t i = 0; i < n; ++i) {
                xs2.push_back(d.x[tr[i]]);
                ys2.push_back(d.y[tr[i]]);
            }
            for (auto& kv : mesh.vertices()) {
                if (kv.second->state == VState::Free) continue;
                std::vector<double> col;
                mesh.probe_column(kv.first, xs2, ys2, &col);
                int nz = 0;
                for (double c2 : col)
                    if (c2 != 0) ++nz;
                if (nz < kMinPts) continue;
                std::vector<VKey> pk;
                for (auto& mw : mesh.masters(kv.second))
                    pk.push_back(key_of(mw.first->ix, mw.first->iy));
                CandStat cs = cand_stats(col, r, w, parent_var(pk));
                if (!cs.ok) continue;
                cands[kv.first] = {kv.first, nullptr, 0, cs};
            }
        }
        if (cands.empty()) break;
        std::vector<VKey> keys;
        std::vector<double> zs;
        for (auto& kv : cands) {
            keys.push_back(kv.first);
            zs.push_back(kv.second.st.z);
        }
        std::vector<int> admit = lfdr_admit(zs, kQ);
        std::sort(admit.begin(), admit.end(), [&](int a, int b) {
            return std::fabs(zs[a]) > std::fabs(zs[b]);
        });
        int committed = 0;
        for (int ai : admit) {
            if ((int)admitted.size() >= kMaxAdmit) break;
            const Candidate& c = cands[keys[ai]];
            Vertex* vx = mesh.vertex(c.pos);
            if (vx && vx->state == VState::Free) continue;
            if (!vx) {
                if (!c.leaf || !c.leaf->leaf()) continue;  // stale
                mesh.split(c.leaf, c.axis, Origin::Selected);
            }
            mesh.promote(c.pos);
            // live re-profile on the exact column at the live fit
            std::vector<double> xs2, ys2, col;
            for (size_t i = 0; i < n; ++i) {
                xs2.push_back(d.x[tr[i]]);
                ys2.push_back(d.y[tr[i]]);
            }
            std::vector<double> fl(n), wl(n), rl(n);
            for (size_t i = 0; i < n; ++i) {
                fl[i] = mesh.eval(xs2[i], ys2[i]);
                const double pp = expit(fl[i]);
                wl[i] = std::max(d.n[tr[i]] * pp * (1 - pp), 1e-8);
                rl[i] = (d.k[tr[i]] - d.n[tr[i]] * pp) / wl[i];
            }
            mesh.probe_column(c.pos, xs2, ys2, &col);
            std::vector<VKey> pk;
            Vertex* vlive = mesh.vertex(c.pos);
            for (auto& mw : vlive->birth_parents)
                pk.push_back(key_of(mw.first->ix, mw.first->iy));
            CandStat live = cand_stats(col, rl, wl, parent_var(pk));
            const bool sign_ok =
                live.ok && ((live.beta > 0) == (c.st.beta > 0));
            // B1 analogue: live evidence must stand on its own (|z|>1
            // is the weakest positive-penalized-gain surrogate here)
            if (!sign_ok || std::fabs(live.z) <= 1.0) {
                vlive->state = VState::Weld;
                continue;
            }
            fit_binom_em(mesh, d, tr, &st);
            admitted.push_back(c.pos);
            ++committed;
        }
        std::printf("round %d: family %zu admit %zu committed %d (total %zu)\n",
                    rnd, cands.size(), admit.size(), committed,
                    admitted.size());
        if (committed == 0) break;
    }

    // ---- held-out deviance
    auto dev_per_pool = [&](const std::vector<int>& idx, bool constant) {
        double rate = 0, tot = 0;
        for (int i : tr) { rate += d.k[i]; tot += d.n[i]; }
        rate /= tot;
        double s = 0;
        for (int i : idx) {
            double p = constant ? rate : expit(mesh.eval(d.x[i], d.y[i]));
            p = std::min(std::max(p, 1e-12), 1 - 1e-12);
            double t = 0;
            if (d.k[i] > 0) t += d.k[i] * std::log(d.k[i] / (d.n[i] * p));
            if (d.n[i] - d.k[i] > 0)
                t += (d.n[i] - d.k[i]) *
                     std::log((d.n[i] - d.k[i]) / (d.n[i] - d.n[i] * p));
            s += 2 * t;
        }
        return s / idx.size();
    };
    const double dev = dev_per_pool(ho, false);
    const double dev0 = dev_per_pool(ho, true);
    std::printf("held-out deviance/pool %.4f (constant %.4f)\n", dev, dev0);

    // ---- outputs
    std::string out(argv[2]);
    {
        FILE* fp = std::fopen((out + "/results.json").c_str(), "w");
        std::fprintf(fp,
                     "{\n  \"n_pools\": %zu,\n  \"n_train\": %zu,\n"
                     "  \"n_heldout\": %zu,\n  \"n_admitted\": %zu,\n"
                     "  \"n_leaves\": %zu,\n  \"n_free\": %zu,\n"
                     "  \"heldout_deviance_per_pool\": %.6f,\n"
                     "  \"heldout_deviance_per_pool_constant_baseline\": %.6f,\n"
                     "  \"stack\": \"C++ binomial IRLS, full data\"\n}\n",
                     N, tr.size(), ho.size(), admitted.size(),
                     mesh.leaf_cells().size(), mesh.free_keys().size(), dev,
                     dev0);
        std::fclose(fp);
    }
    {
        FILE* fp = std::fopen((out + "/surface_grid.csv").c_str(), "w");
        std::fprintf(fp, "x,y,f\n");
        for (int iy = 0; iy < 96; ++iy)
            for (int ix = 0; ix < 96; ++ix) {
                const double x = (ix + 0.5) / 96, y = (iy + 0.5) / 96;
                std::fprintf(fp, "%.6f,%.6f,%.6f\n", x, y, mesh.eval(x, y));
            }
        std::fclose(fp);
    }
    {
        FILE* fp = std::fopen((out + "/mesh.csv").c_str(), "w");
        std::fprintf(fp, "x0,y0,x1,y1\n");
        for (Cell* l : mesh.leaf_cells())
            std::fprintf(fp, "%.6f,%.6f,%.6f,%.6f\n", (double)l->x0 / kS,
                         (double)l->y0 / kS, (double)l->x1 / kS,
                         (double)l->y1 / kS);
        std::fclose(fp);
        fp = std::fopen((out + "/verts.csv").c_str(), "w");
        std::fprintf(fp, "x,y,state\n");
        for (auto& kv : mesh.vertices())
            std::fprintf(fp, "%.6f,%.6f,%d\n", kv.second->x(), kv.second->y(),
                         (int)kv.second->state);
        std::fclose(fp);
    }
    return 0;
}
