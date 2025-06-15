use pyo3_stub_gen::Result;
use std::path::PathBuf;
use std::{env, fs};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

const INIT_PY_CONTENT: &str = r#"from .shared import *
__doc__ = shared.__doc__
__all__ = shared.__all__ if hasattr(shared, "__all__") else __all__
"#;

fn main() -> Result<()> {
    let stdout = tracing_subscriber::fmt::layer().with_filter(EnvFilter::new(
        env::var("STDOUT_LOG_LEVEL").unwrap_or_else(|_| "debug".to_string()),
    ));
    tracing_subscriber::registry().with(stdout).init();
    let stub = shared::point_explorer::pyo3::stub_info()?;
    stub.generate()?;
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let pkg_dir = manifest_dir.join("shared");
    let init_py = pkg_dir.join("__init__.py");
    fs::write(init_py, INIT_PY_CONTENT)?;
    Ok(())
}
