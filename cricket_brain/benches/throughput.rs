// SPDX-License-Identifier: AGPL-3.0-only
use cricket_brain::brain::CricketBrain;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_step_5_neurons(c: &mut Criterion) {
    let mut brain = CricketBrain::new(Default::default()).unwrap();
    c.bench_function("step_5_neurons", |b| {
        b.iter(|| brain.step(black_box(4500.0)))
    });
}

fn bench_step_batch_100(c: &mut Criterion) {
    let mut brain = CricketBrain::new(Default::default()).unwrap();
    let inputs: Vec<f32> = (0..100)
        .map(|i| if i % 2 == 0 { 4500.0 } else { 0.0 })
        .collect();
    c.bench_function("step_batch_100", |b| {
        b.iter(|| brain.step_batch(black_box(&inputs)))
    });
}

fn bench_step_1000_neurons(c: &mut Criterion) {
    let mut brain = CricketBrain::new_scaled(1000, 3000).unwrap();
    c.bench_function("step_1000_neurons", |b| {
        b.iter(|| brain.step(black_box(4500.0)))
    });
}

criterion_group!(
    benches,
    bench_step_5_neurons,
    bench_step_batch_100,
    bench_step_1000_neurons,
);
criterion_main!(benches);
