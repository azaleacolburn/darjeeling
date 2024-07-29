use std::collections::HashSet;
pub use std::time::Instant;

use rand::Rng;
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
        if DEBUG {
            println!("Elapsed: {:?}", $crate::utils::Instant::now() - now);
        }
    };
}

pub struct RandomIter<'a, T> {
    data: &'a [T],
    indices: HashSet<usize>,
    count: usize,
}

impl<'a, T> RandomIter<'a, T> {
    pub fn new(data: &'a [T]) -> Self {
        let count = data.len();
        RandomIter {
            data,
            indices: HashSet::with_capacity(count),
            count,
        }
    }
}

impl<'a, T> Iterator for RandomIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.indices.len() == self.count {
            return None;
        }

        let mut rng = rand::thread_rng();
        loop {
            let index = rng.gen_range(0..self.count);
            if self.indices.insert(index) {
                return Some(&self.data[index]);
            }
        }
    }
}
