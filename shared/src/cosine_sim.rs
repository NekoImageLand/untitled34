use std::arch::x86_64::*;

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { cosine_sim_avx2(a, b) }
    } else {
        common_cosine_sim(a, b)
    }
}

#[inline]
fn common_cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn cosine_sim_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum_dot = _mm256_setzero_ps();
    let mut sum_a2 = _mm256_setzero_ps();
    let mut sum_b2 = _mm256_setzero_ps();
    let chunks = len / 8;
    for i in 0..chunks {
        let pa = a.as_ptr().add(i * 8);
        let pb = b.as_ptr().add(i * 8);
        let va = _mm256_loadu_ps(pa);
        let vb = _mm256_loadu_ps(pb);
        sum_dot = _mm256_fmadd_ps(va, vb, sum_dot);
        sum_a2 = _mm256_fmadd_ps(va, va, sum_a2);
        sum_b2 = _mm256_fmadd_ps(vb, vb, sum_b2);
    }
    #[inline(always)]
    unsafe fn hsum256(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps::<1>(v);
        let lo = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        let sums2 = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(sums2)
    }
    let mut dot = hsum256(sum_dot);
    let mut a2 = hsum256(sum_a2);
    let mut b2 = hsum256(sum_b2);
    for i in (chunks * 8)..len {
        let ai = *a.get_unchecked(i);
        let bi = *b.get_unchecked(i);
        dot += ai * bi;
        a2 += ai * ai;
        b2 += bi * bi;
    }
    dot / (a2.sqrt() * b2.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    const DIM: usize = 768;
    const EPS: f32 = 1e-5;

    #[test]
    fn test_cosine_sim_identical() {
        let v = vec![1.234_f32; DIM];
        let sim = cosine_sim(&v, &v);
        assert!(
            (sim - 1.0).abs() < EPS,
            "identical vectors should give 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_sim_opposite() {
        let v = vec![0.5_f32; DIM];
        let w = vec![-0.5_f32; DIM];
        let sim = cosine_sim(&v, &w);
        assert!(
            (sim + 1.0).abs() < EPS,
            "opposite vectors should give -1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_sim_orthogonal() {
        let mut a = vec![0.0_f32; DIM];
        let mut b = vec![0.0_f32; DIM];
        for i in 0..DIM {
            if i % 2 == 0 {
                a[i] = 1.0;
            } else {
                b[i] = 1.0;
            }
        }
        let sim = cosine_sim(&a, &b);
        assert!(
            sim.abs() < EPS,
            "orthogonal vectors should give 0.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_sim_random_against_cpu() {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut rng = StdRng::seed_from_u64(42);
            for _ in 0..10 {
                let a: Vec<f32> = (0..DIM).map(|_| rng.random_range(-1.0..1.0)).collect();
                let b: Vec<f32> = (0..DIM).map(|_| rng.random_range(-1.0..1.0)).collect();
                let sim = cosine_sim(&a, &b);
                let expected = common_cosine_sim(&a, &b);
                assert!(
                    (sim - expected).abs() < EPS,
                    "mismatch: got {} vs expected {}",
                    sim,
                    expected
                );
            }
        } else {
            panic!("AVX2 and FMA are not supported on this architecture, skipping test.");
        }
    }

    #[test]
    fn test_cosine_sim_random_fallback() {
        let mut rng = StdRng::seed_from_u64(123);
        for _ in 0..5 {
            let a: Vec<f32> = (0..DIM).map(|_| rng.random()).collect();
            let b: Vec<f32> = (0..DIM).map(|_| rng.random()).collect();
            let cpu = common_cosine_sim(&a, &b);
            let sim = cosine_sim(&a, &b);
            assert!((sim - cpu).abs() < EPS);
        }
    }
}
