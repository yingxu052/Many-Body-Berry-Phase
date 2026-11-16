#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <thread>
#include <vector>

using cpx = std::complex<double>;

namespace {

cpx pfaffian_ltl(const cpx* input, int n) {
    if (n == 0) return cpx(1.0, 0.0);
    if (n % 2 != 0) return cpx(0.0, 0.0);
    if (n == 2) return input[1];

    std::array<cpx, 4096> a{};
    for (int i = 0; i < n * n; ++i) a[i] = input[i];

    cpx result(1.0, 0.0);
    std::array<cpx, 64> x{}, y{};

    for (int k = 0; k < n - 1; k += 2) {
        int pivot = k + 1;
        double largest = std::abs(a[k * n + pivot]);
        for (int j = k + 2; j < n; ++j) {
            double candidate = std::abs(a[k * n + j]);
            if (candidate > largest) {
                largest = candidate;
                pivot = j;
            }
        }
        if (largest < 1e-30) return cpx(0.0, 0.0);

        if (pivot != k + 1) {
            for (int j = 0; j < n; ++j) {
                std::swap(a[(k + 1) * n + j], a[pivot * n + j]);
            }
            for (int i = 0; i < n; ++i) {
                std::swap(a[i * n + k + 1], a[i * n + pivot]);
            }
            result = -result;
        }

        const cpx pivot_value = a[k * n + k + 1];
        const cpx inverse_pivot = 1.0 / pivot_value;
        result *= pivot_value;

        for (int i = k + 2; i < n; ++i) {
            x[i] = a[k * n + i];
            y[i] = a[(k + 1) * n + i];
        }
        for (int i = k + 2; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                cpx updated = a[i * n + j]
                    + (y[i] * x[j] - x[i] * y[j]) * inverse_pivot;
                a[i * n + j] = updated;
                a[j * n + i] = -updated;
            }
        }
        for (int i = k + 2; i < n; ++i) {
            a[k * n + i] = cpx(0.0, 0.0);
            a[i * n + k] = cpx(0.0, 0.0);
            a[(k + 1) * n + i] = cpx(0.0, 0.0);
            a[i * n + k + 1] = cpx(0.0, 0.0);
        }
    }
    return result;
}

struct Accumulator {
    long double norm1 = 0.0L;
    long double norm2 = 0.0L;
    long double overlap_real = 0.0L;
    long double overlap_imag = 0.0L;
    std::uint64_t counted = 0;
};

int basis_change_sign(std::uint32_t original, std::uint32_t reference, int n_modes) {
    std::uint32_t state = 0;
    int sign = 1;

    auto apply_creator = [&](int mode) {
        const std::uint32_t bit = std::uint32_t(1) << mode;
        if (state & bit) { sign = 0; return; }
        if (__builtin_popcount(state & (bit - 1)) & 1) sign = -sign;
        state |= bit;
    };
    auto apply_annihilator = [&](int mode) {
        const std::uint32_t bit = std::uint32_t(1) << mode;
        if (!(state & bit)) { sign = 0; return; }
        if (__builtin_popcount(state & (bit - 1)) & 1) sign = -sign;
        state &= ~bit;
    };

    // Products are applied from right to left: ascending here gives the
    // descending written product for |0_c>.
    for (int mode = 0; mode < n_modes; ++mode) {
        if (reference & (std::uint32_t(1) << mode)) apply_creator(mode);
    }
    // The ascending written product for |S>_c is applied in descending order.
    for (int mode = n_modes - 1; mode >= 0; --mode) {
        if (original & (std::uint32_t(1) << mode)) {
            if (reference & (std::uint32_t(1) << mode)) apply_annihilator(mode);
            else apply_creator(mode);
        }
    }
    if (state != (original ^ reference)) return 0;
    return sign;
}

void enumerate_range(
    const double* z1_real, const double* z1_imag,
    const double* z2_real, const double* z2_imag,
    int n_modes, int n_cells, int parity,
    std::uint32_t reference1, std::uint32_t reference2,
    std::uint64_t begin, std::uint64_t end,
    Accumulator& result
) {
    std::array<int, 32> modes{};
    std::array<cpx, 1024> submatrix1{}, submatrix2{};

    auto amplitude = [&](
        std::uint32_t original, std::uint32_t reference,
        const double* z_real, const double* z_imag, cpx* submatrix
    ) -> cpx {
        const int sign = basis_change_sign(original, reference, n_modes);
        if (sign == 0) return cpx(0.0, 0.0);
        const std::uint32_t transformed = original ^ reference;
        int n_particles = 0;
        for (int mode = 0; mode < n_modes; ++mode) {
            if (transformed & (std::uint32_t(1) << mode)) modes[n_particles++] = mode;
        }
        if (n_particles == 0) return cpx(static_cast<double>(sign), 0.0);
        for (int i = 0; i < n_particles; ++i) {
            for (int j = 0; j < n_particles; ++j) {
                const int row = modes[i] * n_modes + modes[j];
                submatrix[i * n_particles + j] = cpx(z_real[row], z_imag[row]);
            }
        }
        return static_cast<double>(sign) * pfaffian_ltl(submatrix, n_particles);
    };

    for (std::uint64_t configuration = begin; configuration < end; ++configuration) {
        std::uint64_t encoded = configuration;
        std::uint32_t original = 0;
        int n_particles = 0;
        for (int cell = 0; cell < n_cells; ++cell) {
            const int digit = static_cast<int>(encoded % 3);
            encoded /= 3;
            if (digit != 0) {
                original |= std::uint32_t(1) << (2 * cell + digit - 1);
                ++n_particles;
            }
        }
        if ((n_particles & 1) != parity) continue;
        ++result.counted;

        const cpx amplitude1 = amplitude(original, reference1, z1_real, z1_imag, submatrix1.data());
        const cpx amplitude2 = amplitude(original, reference2, z2_real, z2_imag, submatrix2.data());
        result.norm1 += std::norm(amplitude1);
        result.norm2 += std::norm(amplitude2);
        const cpx product = std::conj(amplitude1) * amplitude2;
        result.overlap_real += product.real();
        result.overlap_imag += product.imag();
    }
}

void single_occupancy_range(
    const double* z1_real, const double* z1_imag,
    const double* z2_real, const double* z2_imag,
    int n_modes, int n_cells, int parity,
    std::uint32_t reference1, std::uint32_t reference2,
    std::uint64_t begin, std::uint64_t end,
    Accumulator& result
) {
    std::array<int, 32> modes{};
    std::array<cpx, 1024> submatrix1{}, submatrix2{};

    auto amplitude = [&](
        std::uint32_t original, std::uint32_t reference,
        const double* z_real, const double* z_imag, cpx* submatrix
    ) -> cpx {
        const int sign = basis_change_sign(original, reference, n_modes);
        if (sign == 0) return cpx(0.0, 0.0);
        const std::uint32_t transformed = original ^ reference;
        int n_particles = 0;
        for (int mode = 0; mode < n_modes; ++mode) {
            if (transformed & (std::uint32_t(1) << mode)) modes[n_particles++] = mode;
        }
        if (n_particles == 0) return cpx(static_cast<double>(sign), 0.0);
        for (int i = 0; i < n_particles; ++i) {
            for (int j = 0; j < n_particles; ++j) {
                const int row = modes[i] * n_modes + modes[j];
                submatrix[i * n_particles + j] = cpx(z_real[row], z_imag[row]);
            }
        }
        return static_cast<double>(sign) * pfaffian_ltl(submatrix, n_particles);
    };

    for (std::uint64_t configuration = begin; configuration < end; ++configuration) {
        std::uint32_t original = 0;
        int n_particles = 0;
        for (int cell = 0; cell < n_cells; ++cell) {
            const int orbital = static_cast<int>((configuration >> cell) & 1);
            original |= std::uint32_t(1) << (2 * cell + orbital);
            ++n_particles;
        }
        if ((n_particles & 1) != parity) continue;
        ++result.counted;

        const cpx amplitude1 = amplitude(original, reference1, z1_real, z1_imag, submatrix1.data());
        const cpx amplitude2 = amplitude(original, reference2, z2_real, z2_imag, submatrix2.data());
        result.norm1 += std::norm(amplitude1);
        result.norm2 += std::norm(amplitude2);
        const cpx product = std::conj(amplitude1) * amplitude2;
        result.overlap_real += product.real();
        result.overlap_imag += product.imag();
    }
}

struct FiniteGAccumulator {
    std::complex<long double> norm1{0.0L, 0.0L};
    std::complex<long double> norm2{0.0L, 0.0L};
    std::complex<long double> cross{0.0L, 0.0L};
};

void finite_g_range(
    const double* gamma1_real, const double* gamma1_imag,
    const double* gamma2_real, const double* gamma2_imag,
    const double* gamma12_real, const double* gamma12_imag,
    int n_cells, double lambda, std::uint64_t begin, std::uint64_t end,
    FiniteGAccumulator& result
) {
    const int n_operators = 4 * n_cells;
    std::array<cpx, 4096> matrix1{}, matrix2{}, matrix12{};
    std::array<int, 64> indices{};

    for (std::uint64_t subset = begin; subset < end; ++subset) {
        const int selected = __builtin_popcountll(subset);
        const int n = 4 * selected;
        int cursor = 0;
        for (int cell = 0; cell < n_cells; ++cell) {
            if (!(subset & (std::uint64_t(1) << cell))) continue;
            for (int branch = 0; branch < 4; ++branch) indices[cursor++] = 4 * cell + branch;
        }

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const int a = indices[i];
                const int b = indices[j];
                matrix1[i * n + j] = cpx(
                    gamma1_real[a * n_operators + b], gamma1_imag[a * n_operators + b]
                );
                matrix2[i * n + j] = cpx(
                    gamma2_real[a * n_operators + b], gamma2_imag[a * n_operators + b]
                );
                matrix12[i * n + j] = cpx(
                    gamma12_real[a * n_operators + b], gamma12_imag[a * n_operators + b]
                );
            }
        }
        const double weight = std::pow(-lambda, selected);
        result.norm1 += weight * pfaffian_ltl(matrix1.data(), n);
        result.norm2 += weight * pfaffian_ltl(matrix2.data(), n);
        result.cross += weight * pfaffian_ltl(matrix12.data(), n);
    }
}

void finite_g_batch_range(
    const double* gamma1_real, const double* gamma1_imag,
    const double* gamma2_real, const double* gamma2_imag,
    const double* gamma12_real, const double* gamma12_imag,
    int n_cells, const double* lambdas, int n_lambdas,
    std::uint64_t begin, std::uint64_t end, FiniteGAccumulator* results
) {
    const int n_operators = 4 * n_cells;
    std::array<cpx, 4096> matrix1{}, matrix2{}, matrix12{};
    std::array<int, 64> indices{};

    for (std::uint64_t subset = begin; subset < end; ++subset) {
        const int selected = __builtin_popcountll(subset);
        const int n = 4 * selected;
        int cursor = 0;
        for (int cell = 0; cell < n_cells; ++cell) {
            if (!(subset & (std::uint64_t(1) << cell))) continue;
            for (int branch = 0; branch < 4; ++branch) indices[cursor++] = 4 * cell + branch;
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const int a = indices[i];
                const int b = indices[j];
                matrix1[i * n + j] = cpx(
                    gamma1_real[a * n_operators + b], gamma1_imag[a * n_operators + b]
                );
                matrix2[i * n + j] = cpx(
                    gamma2_real[a * n_operators + b], gamma2_imag[a * n_operators + b]
                );
                matrix12[i * n + j] = cpx(
                    gamma12_real[a * n_operators + b], gamma12_imag[a * n_operators + b]
                );
            }
        }
        // The Pfaffians are independent of g; evaluate them once for the whole scan.
        const cpx value1 = pfaffian_ltl(matrix1.data(), n);
        const cpx value2 = pfaffian_ltl(matrix2.data(), n);
        const cpx value12 = pfaffian_ltl(matrix12.data(), n);
        for (int index = 0; index < n_lambdas; ++index) {
            const double weight = std::pow(-lambdas[index], selected);
            results[index].norm1 += weight * value1;
            results[index].norm2 += weight * value2;
            results[index].cross += weight * value12;
        }
    }
}

} // namespace

extern "C" {

double pfaffian_test(const double* real, const double* imag, int n, int imaginary_output) {
    std::array<cpx, 1024> matrix{};
    for (int i = 0; i < n * n; ++i) matrix[i] = cpx(real[i], imag[i]);
    const cpx result = pfaffian_ltl(matrix.data(), n);
    return imaginary_output ? result.imag() : result.real();
}

void projected_overlap(
    const double* z1_real, const double* z1_imag,
    const double* z2_real, const double* z2_imag,
    int n_modes, int n_cells, int parity,
    std::uint32_t reference1, std::uint32_t reference2,
    std::uint64_t limit, int workers,
    double* output
) {
    std::uint64_t total = 1;
    for (int i = 0; i < n_cells; ++i) total *= 3;
    if (limit != 0 && limit < total) total = limit;
    if (workers < 1) workers = 1;
    if (workers > 64) workers = 64;

    std::vector<std::thread> threads;
    std::vector<Accumulator> partial(static_cast<std::size_t>(workers));
    const std::uint64_t chunk = (total + workers - 1) / workers;

    for (int worker = 0; worker < workers; ++worker) {
        const std::uint64_t begin = std::min(total, static_cast<std::uint64_t>(worker) * chunk);
        const std::uint64_t end = std::min(total, begin + chunk);
        threads.emplace_back(
            enumerate_range,
            z1_real, z1_imag, z2_real, z2_imag,
            n_modes, n_cells, parity, reference1, reference2, begin, end,
            std::ref(partial[worker])
        );
    }
    for (auto& thread : threads) thread.join();

    Accumulator total_result;
    for (const auto& part : partial) {
        total_result.norm1 += part.norm1;
        total_result.norm2 += part.norm2;
        total_result.overlap_real += part.overlap_real;
        total_result.overlap_imag += part.overlap_imag;
        total_result.counted += part.counted;
    }
    output[0] = static_cast<double>(total_result.norm1);
    output[1] = static_cast<double>(total_result.norm2);
    output[2] = static_cast<double>(total_result.overlap_real);
    output[3] = static_cast<double>(total_result.overlap_imag);
    output[4] = static_cast<double>(total_result.counted);
    output[5] = static_cast<double>(total);
}

void single_occupancy_overlap(
    const double* z1_real, const double* z1_imag,
    const double* z2_real, const double* z2_imag,
    int n_modes, int n_cells, int parity,
    std::uint32_t reference1, std::uint32_t reference2,
    int workers, double* output
) {
    if (n_cells < 0 || n_cells > 16) return;
    const std::uint64_t total = std::uint64_t(1) << n_cells;
    if (workers < 1) workers = 1;
    if (workers > 64) workers = 64;

    std::vector<std::thread> threads;
    std::vector<Accumulator> partial(static_cast<std::size_t>(workers));
    const std::uint64_t chunk = (total + workers - 1) / workers;
    for (int worker = 0; worker < workers; ++worker) {
        const std::uint64_t begin = std::min(total, static_cast<std::uint64_t>(worker) * chunk);
        const std::uint64_t end = std::min(total, begin + chunk);
        threads.emplace_back(
            single_occupancy_range,
            z1_real, z1_imag, z2_real, z2_imag,
            n_modes, n_cells, parity, reference1, reference2, begin, end,
            std::ref(partial[worker])
        );
    }
    for (auto& thread : threads) thread.join();

    Accumulator total_result;
    for (const auto& part : partial) {
        total_result.norm1 += part.norm1;
        total_result.norm2 += part.norm2;
        total_result.overlap_real += part.overlap_real;
        total_result.overlap_imag += part.overlap_imag;
        total_result.counted += part.counted;
    }
    output[0] = static_cast<double>(total_result.norm1);
    output[1] = static_cast<double>(total_result.norm2);
    output[2] = static_cast<double>(total_result.overlap_real);
    output[3] = static_cast<double>(total_result.overlap_imag);
    output[4] = static_cast<double>(total_result.counted);
    output[5] = static_cast<double>(total);
}

void finite_g_factors(
    const double* gamma1_real, const double* gamma1_imag,
    const double* gamma2_real, const double* gamma2_imag,
    const double* gamma12_real, const double* gamma12_imag,
    int n_cells, double lambda, int workers, double* output
) {
    if (n_cells < 0 || n_cells > 16) return;
    const std::uint64_t total = std::uint64_t(1) << n_cells;
    if (workers < 1) workers = 1;
    if (workers > 64) workers = 64;

    std::vector<std::thread> threads;
    std::vector<FiniteGAccumulator> partial(static_cast<std::size_t>(workers));
    const std::uint64_t chunk = (total + workers - 1) / workers;
    for (int worker = 0; worker < workers; ++worker) {
        const std::uint64_t begin = std::min(total, static_cast<std::uint64_t>(worker) * chunk);
        const std::uint64_t end = std::min(total, begin + chunk);
        threads.emplace_back(
            finite_g_range,
            gamma1_real, gamma1_imag, gamma2_real, gamma2_imag, gamma12_real, gamma12_imag,
            n_cells, lambda, begin, end, std::ref(partial[worker])
        );
    }
    for (auto& thread : threads) thread.join();

    FiniteGAccumulator sum;
    for (const auto& part : partial) {
        sum.norm1 += part.norm1;
        sum.norm2 += part.norm2;
        sum.cross += part.cross;
    }
    output[0] = static_cast<double>(sum.norm1.real());
    output[1] = static_cast<double>(sum.norm1.imag());
    output[2] = static_cast<double>(sum.norm2.real());
    output[3] = static_cast<double>(sum.norm2.imag());
    output[4] = static_cast<double>(sum.cross.real());
    output[5] = static_cast<double>(sum.cross.imag());
}

void finite_g_factors_batch(
    const double* gamma1_real, const double* gamma1_imag,
    const double* gamma2_real, const double* gamma2_imag,
    const double* gamma12_real, const double* gamma12_imag,
    int n_cells, const double* lambdas, int n_lambdas, int workers, double* output
) {
    if (n_cells < 0 || n_cells > 16 || n_lambdas < 1) return;
    const std::uint64_t total = std::uint64_t(1) << n_cells;
    if (workers < 1) workers = 1;
    if (workers > 64) workers = 64;

    std::vector<std::thread> threads;
    std::vector<FiniteGAccumulator> partial(static_cast<std::size_t>(workers) * n_lambdas);
    const std::uint64_t chunk = (total + workers - 1) / workers;
    for (int worker = 0; worker < workers; ++worker) {
        const std::uint64_t begin = std::min(total, static_cast<std::uint64_t>(worker) * chunk);
        const std::uint64_t end = std::min(total, begin + chunk);
        threads.emplace_back(
            finite_g_batch_range,
            gamma1_real, gamma1_imag, gamma2_real, gamma2_imag, gamma12_real, gamma12_imag,
            n_cells, lambdas, n_lambdas, begin, end, partial.data() + worker * n_lambdas
        );
    }
    for (auto& thread : threads) thread.join();

    for (int index = 0; index < n_lambdas; ++index) {
        FiniteGAccumulator sum;
        for (int worker = 0; worker < workers; ++worker) {
            const FiniteGAccumulator& part = partial[worker * n_lambdas + index];
            sum.norm1 += part.norm1;
            sum.norm2 += part.norm2;
            sum.cross += part.cross;
        }
        output[6 * index] = static_cast<double>(sum.norm1.real());
        output[6 * index + 1] = static_cast<double>(sum.norm1.imag());
        output[6 * index + 2] = static_cast<double>(sum.norm2.real());
        output[6 * index + 3] = static_cast<double>(sum.norm2.imag());
        output[6 * index + 4] = static_cast<double>(sum.cross.real());
        output[6 * index + 5] = static_cast<double>(sum.cross.imag());
    }
}

} // extern "C"
