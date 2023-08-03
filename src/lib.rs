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
#![feature(iter_collect_into, try_blocks)]
pub static DEBUG: bool = false;

pub mod input;
pub mod categorize;
pub mod node;
pub mod dataframe;
pub mod series;
pub mod error;
pub mod activation;
pub mod types;
pub mod generation;
#[cfg(test)]
pub mod tests;