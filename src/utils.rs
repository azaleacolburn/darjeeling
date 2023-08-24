pub use std::time::Instant;
#[macro_export]
macro_rules! dbg_println {
    // `()` indicates that the macro takes no argument.
    ($($arg:tt)*) => {
        if DEBUG { println!($($arg)*) }
    };
}

#[macro_export]
macro_rules! bench {
    ($($arg:tt)*) => {
        let now = $crate::utils::Instant::now();
        $($arg)*;
        println!("Elapsed: {:?}", $crate::utils::Instant::now() - now);
    };
}