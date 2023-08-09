#[macro_export]
macro_rules! dbg_println {
    // `()` indicates that the macro takes no argument.
    ($($arg:tt)*) => {
        if DEBUG { println!($($arg)*) }
    };
}