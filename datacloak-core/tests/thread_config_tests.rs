use datacloak_core::{
    create_tokio_runtime, get_optimized_config, get_thread_pool_metrics, initialize_thread_pools,
    ThreadPoolConfig, WorkStealingScheduler,
};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[test]
fn test_default_thread_pool_config() {
    let config = ThreadPoolConfig::default();

    assert_eq!(config.rayon_threads, 0); // Auto-detect
    assert_eq!(config.rayon_thread_prefix, "datacloak-cpu");
    assert_eq!(config.rayon_stack_size, 8 * 1024 * 1024); // 8MB
    assert!(!config.enable_cpu_affinity);
    assert_eq!(config.tokio_worker_threads, 0); // Auto-detect
    assert_eq!(config.tokio_thread_prefix, "datacloak-io");
}

#[test]
fn test_optimized_config() {
    let config = get_optimized_config();
    let num_cpus = num_cpus::get();

    // Verify CPU-based optimization
    if num_cpus <= 4 {
        assert_eq!(config.rayon_threads, num_cpus);
        assert_eq!(config.tokio_worker_threads, num_cpus);
    } else if num_cpus <= 16 {
        assert_eq!(config.rayon_threads, num_cpus - 1);
        assert_eq!(config.tokio_worker_threads, num_cpus);
    } else {
        assert_eq!(config.rayon_threads, 16);
        assert_eq!(config.tokio_worker_threads, 16);
        assert_eq!(config.tokio_blocking_threads, 32);
    }
}

#[test]
fn test_thread_pool_initialization() {
    let config = ThreadPoolConfig {
        rayon_threads: 4,
        rayon_thread_prefix: "test-cpu".to_string(),
        rayon_stack_size: 4 * 1024 * 1024,
        enable_cpu_affinity: false,
        tokio_worker_threads: 2,
        tokio_blocking_threads: 4,
        tokio_thread_prefix: "test-io".to_string(),
    };

    // Initialize should succeed
    let result = initialize_thread_pools(&config);
    assert!(result.is_ok());

    // Verify Rayon configuration (may be already initialized by other tests)
    let current_threads = rayon::current_num_threads();
    assert!(
        current_threads > 0,
        "Rayon should have some threads configured"
    );
}

#[test]
fn test_custom_tokio_runtime() {
    let config = ThreadPoolConfig {
        rayon_threads: 2,
        rayon_thread_prefix: "custom-cpu".to_string(),
        rayon_stack_size: 4 * 1024 * 1024,
        enable_cpu_affinity: false,
        tokio_worker_threads: 3,
        tokio_blocking_threads: 6,
        tokio_thread_prefix: "custom-io".to_string(),
    };

    let runtime = create_tokio_runtime(&config).unwrap();

    // Test runtime functionality
    runtime.block_on(async {
        let handle = tokio::spawn(async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            42
        });

        let result = handle.await.unwrap();
        assert_eq!(result, 42);
    });

    // Check metrics
    let metrics = get_thread_pool_metrics(&runtime);
    assert_eq!(metrics.tokio_worker_threads, 3);
}

#[test]
fn test_work_stealing_scheduler() {
    let scheduler: WorkStealingScheduler<i32> = WorkStealingScheduler::new(4);

    // Push work to global queue
    for i in 0..20 {
        scheduler.push(i);
    }

    // Pop all work from the scheduler
    let mut total = 0;
    let mut count = 0;

    // Keep popping until no more work is available
    loop {
        let mut found_work = false;
        for worker_idx in 0..4 {
            if let Some(task) = scheduler.pop(worker_idx) {
                total += task;
                count += 1;
                found_work = true;
            }
        }
        if !found_work {
            break;
        }
    }

    assert_eq!(count, 20);
    assert_eq!(total, (0..20).sum::<i32>());
}

#[test]
fn test_work_stealing_concurrency() {
    let scheduler = Arc::new(WorkStealingScheduler::<i32>::new(4));
    let mut handles = vec![];

    // Producer thread
    let sched_clone = scheduler.clone();
    let producer = thread::spawn(move || {
        for i in 0..100 {
            sched_clone.push(i);
            thread::sleep(Duration::from_micros(10));
        }
    });

    // Consumer threads
    for worker_idx in 0..4 {
        let sched_clone = scheduler.clone();
        let handle = thread::spawn(move || {
            let mut sum = 0;
            let mut count = 0;

            for _ in 0..50 {
                if let Some(task) = sched_clone.pop(worker_idx) {
                    sum += task;
                    count += 1;
                }
                thread::sleep(Duration::from_micros(20));
            }

            (sum, count)
        });
        handles.push(handle);
    }

    // Wait for producer
    producer.join().unwrap();

    // Collect results
    let mut total_sum = 0;
    let mut total_count = 0;

    for handle in handles {
        let (sum, count) = handle.join().unwrap();
        total_sum += sum;
        total_count += count;
    }

    // Consume any remaining work
    loop {
        let mut found_work = false;
        for worker_idx in 0..4 {
            if let Some(task) = scheduler.pop(worker_idx) {
                total_sum += task;
                total_count += 1;
                found_work = true;
            }
        }
        if !found_work {
            break;
        }
    }

    // All work should be completed
    assert_eq!(total_sum, (0..100).sum::<i32>());
    assert_eq!(total_count, 100);
}

#[test]
fn test_thread_pool_metrics() {
    let config = ThreadPoolConfig {
        rayon_threads: 2,
        rayon_thread_prefix: "metrics-cpu".to_string(),
        rayon_stack_size: 4 * 1024 * 1024,
        enable_cpu_affinity: false,
        tokio_worker_threads: 3,
        tokio_blocking_threads: 6,
        tokio_thread_prefix: "metrics-io".to_string(),
    };

    initialize_thread_pools(&config).unwrap();
    let runtime = create_tokio_runtime(&config).unwrap();

    let metrics = get_thread_pool_metrics(&runtime);

    assert_eq!(metrics.rayon_threads, 2);
    assert_eq!(metrics.tokio_worker_threads, 3);
    assert_eq!(metrics.cpu_count, num_cpus::get());
}

#[test]
fn test_rayon_computation() {
    use rayon::prelude::*;

    let config = ThreadPoolConfig {
        rayon_threads: 4,
        ..Default::default()
    };

    initialize_thread_pools(&config).unwrap();

    // Perform parallel computation
    let sum: i32 = (0..1000).into_par_iter().map(|x| x * x).sum();

    let expected: i32 = (0..1000).map(|x| x * x).sum();
    assert_eq!(sum, expected);
}

#[test]
fn test_mixed_workload() {
    let config = ThreadPoolConfig {
        rayon_threads: 2,
        tokio_worker_threads: 2,
        ..Default::default()
    };

    initialize_thread_pools(&config).unwrap();
    let runtime = create_tokio_runtime(&config).unwrap();

    runtime.block_on(async {
        use rayon::prelude::*;

        // CPU-bound task on Rayon
        let cpu_handle = tokio::task::spawn_blocking(|| {
            let sum: u64 = (0..10000).into_par_iter().map(|x| x * x).sum();
            sum
        });

        // I/O-bound task on Tokio
        let io_handle = tokio::spawn(async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            "io_complete"
        });

        // Wait for both
        let (cpu_result, io_result) = tokio::join!(cpu_handle, io_handle);

        assert!(cpu_result.unwrap() > 0);
        assert_eq!(io_result.unwrap(), "io_complete");
    });
}

#[test]
fn test_panic_handling() {
    let config = ThreadPoolConfig {
        rayon_threads: 2,
        ..Default::default()
    };

    // Initialize with panic handler
    initialize_thread_pools(&config).unwrap();

    // This should not crash the thread pool
    let result = std::panic::catch_unwind(|| {
        use rayon::prelude::*;

        let _: Vec<i32> = (0..10)
            .into_par_iter()
            .map(|x| {
                if x == 5 {
                    panic!("Test panic");
                }
                x
            })
            .collect();
    });

    assert!(result.is_err());

    // Thread pool should still be functional
    use rayon::prelude::*;
    let sum: i32 = (0..10).into_par_iter().sum();
    assert_eq!(sum, 45);
}
