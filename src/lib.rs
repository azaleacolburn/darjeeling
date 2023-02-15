#![allow(
    clippy::collapsible_if,
    clippy::module_inception,
    clippy::needless_range_loop,
    clippy::useless_attribute,
    dead_code
)]
pub static DEBUG: bool = false;

pub mod input;
pub mod network;
pub mod node;
pub mod tests;