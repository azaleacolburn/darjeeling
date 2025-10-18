use core::fmt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    Linear,
}

impl ActivationFunction {
    pub fn to_function(&self) -> fn(f32) -> f32 {
        match self {
            ActivationFunction::Sigmoid => sigmoid,
            ActivationFunction::Tanh => tanh,
            ActivationFunction::Linear => linear,
        }
    }

    pub fn to_derivative(&self) -> fn(f32) -> f32 {
        match self {
            ActivationFunction::Sigmoid => bell_curve,
            ActivationFunction::Linear => constant,
            ActivationFunction::Tanh => sech_squared,
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

// Derivative functions
fn bell_curve(y: f32) -> f32 {
    y * (1.0 - y)
}

fn constant(_y: f32) -> f32 {
    1.
}

fn sech_squared(y: f32) -> f32 {
    1. - y.powi(2)
}

impl fmt::Display for ActivationFunction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ActivationFunction::Sigmoid => write!(f, "sigmoid"),
            ActivationFunction::Linear => write!(f, "linear"),
            ActivationFunction::Tanh => write!(f, "tanh"),
        }
    }
}
