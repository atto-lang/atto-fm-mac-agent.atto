//! Runs the REAL LLVM Oreslang compiler for the owned native bigram fixture.
//! This tool only coordinates compilation/execution; it does not implement ML.
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn check(llvm_root: &Path) -> Result<(), String> {
    if !llvm_root.is_absolute() || llvm_root.is_symlink() || !llvm_root.is_dir() {
        return Err("provide an existing absolute LLVM Oreslang root directory".into());
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let source = root.join("native/src/bigram_bootstrap.ores");
    let driver = llvm_root.join("scripts/compile-aot.sh");
    if !source.is_file() || !driver.is_file() || driver.is_symlink() {
        return Err("missing native .ores source or LLVM compile-aot driver".into());
    }
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    let work = env::temp_dir().join(format!(
        "ores-llm-native-{}-{}", std::process::id(), nanos
    ));
    fs::create_dir(&work).map_err(|error| error.to_string())?;
    let _cleanup = Scratch(work.clone());
    let binary = work.join("bigram");
    let compiled = Command::new("bash")
        .arg(&driver)
        .arg(&source)
        .arg(&binary)
        .output()
        .map_err(|error| format!("failed to launch native compiler: {error}"))?;
    if !compiled.status.success() {
        return Err(format!(
            "native compiler failed (status {:?}):\n{}\n{}",
            compiled.status.code(),
            String::from_utf8_lossy(&compiled.stdout),
            String::from_utf8_lossy(&compiled.stderr)
        ));
    }
    if !binary.is_file() {
        return Err("compiler claimed success without a native executable".into());
    }
    let executed = Command::new(&binary)
        .output()
        .map_err(|error| format!("failed to execute Oreslang AOT binary: {error}"))?;
    if executed.status.code() != Some(42) {
        return Err(format!(
            "FAIL: native train-and-predict expected exit 42, got {:?}; stderr: {}",
            executed.status.code(),
            String::from_utf8_lossy(&executed.stderr)
        ));
    }
    println!("PASS: genuine Oreslang source compiled to AOT; trained bigram exit=42");
    Ok(())
}

fn main() {
    let mut args = env::args().skip(1);
    let root = match (args.next().as_deref(), args.next(), args.next()) {
        (Some("--llvm-root"), Some(path), None) => PathBuf::from(path),
        _ => {
            eprintln!(
                "usage: cargo run -p ores-llm-native-gate -- --llvm-root /ABS/PATH/TO/oreslang-source.llvm"
            );
            std::process::exit(64);
        }
    };
    if let Err(error) = check(&root) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_nonexistent_compiler_root() {
        assert!(check(Path::new("/this/oreslang/compiler/does/not/exist")).is_err());
    }
    #[test]
    fn native_source_is_part_of_workspace() {
        assert!(Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../native/src/bigram_bootstrap.ores")
            .exists());
    }
}
