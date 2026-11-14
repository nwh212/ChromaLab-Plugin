//! Prints the manifest the host reads: `cargo run -p chromadust --example chromadust_manifest`.
fn main() {
    println!("{}", chromadust::manifest());
}
