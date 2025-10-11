use rand::Rng;
pub use std::time::Instant;
use std::{
    collections::HashSet,
    fs,
    io::{BufRead, BufReader},
};

use crate::series::Series;

#[macro_export]
macro_rules! dbg_println {
    // `()` indicates that the macro takes no argument.
    ($($arg:tt)*) => {
        if DEBUG { println!($($arg)*) }
    };
}

#[macro_export]
macro_rules! cond_println{
    // `()` indicates that the macro takes no argument.
    ($($arg:tt)*) => {
        if print { println!($($arg)*) }
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

/// Read the file you want to and format it as Inputs
// In utils so it can be accessed by doctests
pub fn xor_file<'a>() -> Box<[Series]> {
    let file = match fs::File::open("training_data/xor.txt") {
        Ok(file) => file,
        Err(error) => panic!("Panic opening the file: {:?}", error),
    };

    let reader = BufReader::new(file);
    let mut inputs: Vec<Series> = vec![];

    for l in reader.lines() {
        let line: String = match l {
            Ok(line) => line,
            Err(error) => panic!("{:?}", error),
        };

        let init_inputs: Vec<&str> = line.split(";").collect();
        // Confused about how this is supposed to work
        let float_inputs: Vec<f32> = init_inputs[0]
            .split(" ")
            .map(|n| n.parse().unwrap())
            .collect::<Vec<f32>>();

        let answer_inputs = String::from(init_inputs[1]);
        let input = Series::new(float_inputs, answer_inputs);
        inputs.push(input);
    }
    inputs.into_boxed_slice()
}
