use serde::{Deserialize, Serialize};
use std::sync::Once;
use tracing::{error, info};

static THREAD_POOL_INIT: Once = Once::new();

/// Thread pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPoolConfig {
    /// Number of Rayon threads (0 = auto-detect)
    pub rayon_threads: usize,
    /// Rayon thread name prefix
    pub rayon_thread_prefix: String,
    /// Stack size for Rayon threads (bytes)
    pub rayon_stack_size: usize,
    /// Enable CPU affinity
    pub enable_cpu_affinity: bool,
    /// Tokio worker threads (0 = auto-detect)
    pub tokio_worker_threads: usize,
    /// Tokio blocking threads
    pub tokio_blocking_threads: usize,
    /// Tokio thread name prefix
    pub tokio_thread_prefix: String,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        let num_cpus = num_cpus::get();

        Self {
            rayon_threads: 0, // Auto-detect
            rayon_thread_prefix: "datacloak-cpu".to_string(),
            rayon_stack_size: 8 * 1024 * 1024, // 8MB
            enable_cpu_affinity: false,
            tokio_worker_threads: 0, // Auto-detect
            tokio_blocking_threads: num_cpus * 2,
            tokio_thread_prefix: "datacloak-io".to_string(),
        }
    }
}

/// Initialize thread pools with the given configuration
pub fn initialize_thread_pools(
    config: &ThreadPoolConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut result = Ok(());

    THREAD_POOL_INIT.call_once(|| {
        // Configure Rayon thread pool
        match configure_rayon(config) {
            Ok(_) => info!("Rayon thread pool initialized successfully"),
            Err(e) => {
                error!("Failed to initialize Rayon thread pool: {}", e);
                result = Err(e);
            }
        }
    });

    result
}

/// Configure Rayon thread pool
fn configure_rayon(config: &ThreadPoolConfig) -> Result<(), Box<dyn std::error::Error>> {
    let num_threads = if config.rayon_threads == 0 {
        num_cpus::get()
    } else {
        config.rayon_threads
    };

    let thread_prefix = config.rayon_thread_prefix.clone();
    let mut builder = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .thread_name(move |idx| format!("{}-{}", thread_prefix, idx))
        .stack_size(config.rayon_stack_size);

    // Set panic handler
    builder = builder.panic_handler(move |panic_info| {
        error!("Rayon thread panicked: {:?}", panic_info);
    });

    // Build and install as global thread pool
    builder.build_global()?;

    info!(
        "Rayon thread pool configured: {} threads, {}MB stack size",
        num_threads,
        config.rayon_stack_size / (1024 * 1024)
    );

    Ok(())
}

/// Create a custom Tokio runtime with the given configuration
pub fn create_tokio_runtime(
    config: &ThreadPoolConfig,
) -> Result<tokio::runtime::Runtime, Box<dyn std::error::Error>> {
    let worker_threads = if config.tokio_worker_threads == 0 {
        num_cpus::get()
    } else {
        config.tokio_worker_threads
    };

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .max_blocking_threads(config.tokio_blocking_threads)
        .thread_name(config.tokio_thread_prefix.clone())
        .enable_all()
        .build()?;

    info!(
        "Tokio runtime configured: {} worker threads, {} blocking threads",
        worker_threads, config.tokio_blocking_threads
    );

    Ok(runtime)
}

/// Get optimized thread pool configuration based on system resources
pub fn get_optimized_config() -> ThreadPoolConfig {
    let num_cpus = num_cpus::get();
    let mut config = ThreadPoolConfig::default();

    // Adjust based on CPU count
    if num_cpus <= 4 {
        // Small systems: use all cores for CPU work
        config.rayon_threads = num_cpus;
        config.tokio_worker_threads = num_cpus;
    } else if num_cpus <= 16 {
        // Medium systems: leave some headroom
        config.rayon_threads = num_cpus - 1;
        config.tokio_worker_threads = num_cpus;
    } else {
        // Large systems: cap thread counts
        config.rayon_threads = 16;
        config.tokio_worker_threads = 16;
        config.tokio_blocking_threads = 32;
    }

    // Check available memory
    if let Ok(mem_info) = sys_info::mem_info() {
        let available_gb = mem_info.avail / (1024 * 1024);
        if available_gb < 4 {
            // Low memory: reduce stack size
            config.rayon_stack_size = 4 * 1024 * 1024; // 4MB
        }
    }

    config
}

/// Thread pool metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPoolMetrics {
    pub rayon_threads: usize,
    pub rayon_active_threads: usize,
    pub tokio_worker_threads: usize,
    pub tokio_blocking_threads: usize,
    pub cpu_count: usize,
}

/// Get current thread pool metrics
pub fn get_thread_pool_metrics(runtime: &tokio::runtime::Runtime) -> ThreadPoolMetrics {
    ThreadPoolMetrics {
        rayon_threads: rayon::current_num_threads(),
        rayon_active_threads: rayon::current_thread_index().map(|_| 1).unwrap_or(0),
        tokio_worker_threads: runtime.metrics().num_workers(),
        tokio_blocking_threads: runtime.metrics().num_alive_tasks(),
        cpu_count: num_cpus::get(),
    }
}

/// Work-stealing task scheduler for better load balancing
pub struct WorkStealingScheduler<T: Send + Sync + 'static> {
    stealers: Vec<crossbeam::deque::Stealer<T>>,
    injector: crossbeam::deque::Injector<T>,
    #[allow(dead_code)]
    worker_count: usize,
}

impl<T: Send + Sync + 'static> WorkStealingScheduler<T> {
    pub fn new(num_workers: usize) -> Self {
        let mut stealers = Vec::with_capacity(num_workers);

        for _ in 0..num_workers {
            let worker = crossbeam::deque::Worker::new_fifo();
            stealers.push(worker.stealer());
        }

        Self {
            stealers,
            injector: crossbeam::deque::Injector::new(),
            worker_count: num_workers,
        }
    }

    /// Push work to the global queue
    pub fn push(&self, task: T) {
        self.injector.push(task);
    }

    /// Try to pop work for a specific worker
    pub fn pop(&self, _worker_idx: usize) -> Option<T> {
        // Try stealing from global queue
        loop {
            match self.injector.steal() {
                crossbeam::deque::Steal::Success(task) => return Some(task),
                crossbeam::deque::Steal::Empty => break,
                crossbeam::deque::Steal::Retry => continue,
            }
        }

        // Try stealing from other workers
        for stealer in &self.stealers {
            loop {
                match stealer.steal() {
                    crossbeam::deque::Steal::Success(task) => return Some(task),
                    crossbeam::deque::Steal::Empty => break,
                    crossbeam::deque::Steal::Retry => continue,
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ThreadPoolConfig::default();
        assert_eq!(config.rayon_threads, 0); // Auto-detect
        assert_eq!(config.rayon_stack_size, 8 * 1024 * 1024);
    }

    #[test]
    fn test_optimized_config() {
        let config = get_optimized_config();
        let num_cpus = num_cpus::get();

        if num_cpus <= 4 {
            assert_eq!(config.rayon_threads, num_cpus);
        } else if num_cpus <= 16 {
            assert_eq!(config.rayon_threads, num_cpus - 1);
        } else {
            assert_eq!(config.rayon_threads, 16);
        }
    }

    #[test]
    fn test_work_stealing_scheduler() {
        let scheduler = WorkStealingScheduler::new(4);

        // Push some work
        for i in 0..10 {
            scheduler.push(i);
        }

        // Pop from different workers
        let task1 = scheduler.pop(0);
        assert!(task1.is_some());

        let task2 = scheduler.pop(1);
        assert!(task2.is_some());
    }
}
