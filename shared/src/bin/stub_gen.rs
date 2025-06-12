use pyo3_stub_gen::Result;
use std::env;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

fn main() -> Result<()> {
    let stdout = tracing_subscriber::fmt::layer().with_filter(EnvFilter::new(
        env::var("STDOUT_LOG_LEVEL").unwrap_or_else(|_| "debug".to_string()),
    ));
    tracing_subscriber::registry().with(stdout).init();
    let stub = shared::point_explorer::pyo3::stub_info()?;
    stub.generate()?;
    Ok(())
}
