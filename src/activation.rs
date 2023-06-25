use core::fmt;
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    Linear,
    Step,
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