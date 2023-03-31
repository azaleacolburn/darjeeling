#![allow(
    clippy::clone_double_ref,
    clippy::collapsible_if,
    clippy::module_inception,
    clippy::needless_borrow,
    clippy::needless_range_loop,
    clippy::print_with_newline,
    clippy::ptr_arg,
    clippy::useless_attribute,
    dead_code
)]
pub static DEBUG: bool = false;

pub mod input;
pub mod categorize;
pub mod node;
pub mod dataframe;
pub mod series;
pub mod error;
pub mod activation;
pub mod types;
#[cfg(test)]
pub mod tests;