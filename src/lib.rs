#![allow(
    clippy::clone_double_ref,
    clippy::clone_on_copy,
    clippy::collapsible_if,
    clippy::extra_unused_lifetimes,
    clippy::module_inception,
    clippy::needless_borrow,
    clippy::needless_range_loop,
    clippy::needless_return,
    clippy::print_with_newline,
    clippy::ptr_arg,
    clippy::useless_attribute,
    clippy::useless_conversion,
    dead_code
)]
pub static DEBUG: bool = false;

pub mod activation;
pub mod categorize;
pub mod error;
pub mod generation;
pub mod input;
pub mod node;
#[cfg(test)]
pub mod tests;
pub mod types;
mod utils;
