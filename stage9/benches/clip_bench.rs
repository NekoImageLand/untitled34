use candle_core::DType::BF16;
use candle_transformers::models::clip::ClipConfig;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use stage9::clip_worker::ClipWorker;
use std::env;

fn bench_clip(c: &mut Criterion) {
    let worker = ClipWorker::new(
        &env::var("CLIP_MODEL_PATH").unwrap(),
        ClipConfig::baai_bge_vl_large(),
        BF16,
        true,
    )
    .unwrap();
    let mut group = c.benchmark_group("clip_inference");
    for &batch_size in &[1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512] {
        group.throughput(Throughput::Elements(batch_size as u64));
        let pics: Vec<_> = (0..batch_size)
            .map(|i| format!("../assets/test_images/bsn_{}.jpg", i % 2))
            .collect();
        group.bench_with_input(
            BenchmarkId::new("get_images_embedding_batched", batch_size),
            &pics,
            |b, pics| {
                b.iter(|| {
                    worker
                        .get_images_embedding_batched(pics.map(|s| &s))
                        .unwrap();
                });
            },
        );
        // group.bench_with_input(
        //     BenchmarkId::new("get_images_embedding", batch_size),
        //     &pics,
        //     |b, pics| {
        //         b.iter(|| {
        //             worker.get_images_embedding(pics).unwrap();
        //         });
        //     },
        // );
    }
    group.finish();
}

criterion_group!(benches, bench_clip);
criterion_main!(benches);
