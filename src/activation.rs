use core::fmt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    Linear,
    Step,
}

impl ActivationFunction {
    pub fn to_function(&self) -> fn(f32) -> f32 {
        match self {
            ActivationFunction::Sigmoid => sigmoid,
            ActivationFunction::Tanh => tanh,
            ActivationFunction::Step => step,
            ActivationFunction::Linear => linear,
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + ((-x).exp()))
}

fn linear(x: f32) -> f32 {
    2.0 * x
}

fn tanh(x: f32) -> f32 {
    let e = std::f64::consts::E as f32;

    2.00 / (1.00 + e.powf(-2.00 * x)) - 1.00
}

fn step(x: f32) -> f32 {
    if x < 0.00 {
        -1.00
    } else {
        1.00
    }
}

impl fmt::Display for ActivationFunction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ActivationFunction::Sigmoid => write!(f, "sigmoid"),
            ActivationFunction::Linear => write!(f, "linear"),
            ActivationFunction::Tanh => write!(f, "tanh"),
            ActivationFunction::Step => write!(f, "step"),
        }
    }
}
