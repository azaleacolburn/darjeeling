use core::fmt;
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum ActivationFunction {
    
    Sigmoid,
    Tanh,
    Linear,
    Step,

    Catcher // Catcher 'catches' if the function isn't loaded from the model correctly. Will panic when unwrapped.
}

impl fmt::Display for ActivationFunction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ActivationFunction::Sigmoid => write!(f, "sigmoid"),

            ActivationFunction::Linear => write!(f, "linear"),

            ActivationFunction::Tanh => write!(f, "tanh"),

            ActivationFunction::Step => write!(f, "step"),

            ActivationFunction::Catcher => write!(f, "")
        }
    }
}