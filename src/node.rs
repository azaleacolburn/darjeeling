use crate::{activation::ActivationFunction, dbg_println, types::Types, DEBUG};
use serde::{Deserialize, Serialize};

/// Represents a node in the network
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Node {
    pub link_weights: Box<[f32]>,
    pub link_vals: Box<[f32]>,
    pub err_sig: Option<f32>,
    pub correct_answer: Option<f32>,
    pub category: Option<Types>,
    pub b_weight: f32,
    pub cached_output: Option<f32>,
}

impl Node {
    pub fn new(link_weights: Box<[f32]>, b_weight: f32) -> Node {
        let link_vals: Box<[f32]> = (0..link_weights.len())
            .map(|_| 0.00)
            .collect::<Box<[f32]>>();

        Node {
            link_weights,
            link_vals,
            b_weight,
            err_sig: None,
            cached_output: None,
            correct_answer: None,
            category: None,
        }
    }

    fn input(&mut self) -> f32 {
        (0..self.link_weights.len())
            .into_iter()
            .map(|i| {
                dbg_println!("Link Val: {:?}", self.link_vals[i]);
                self.link_vals[i] * self.link_weights[i]
            })
            .sum::<f32>()
            * self.b_weight
    }

    pub fn output(&mut self, activation: &ActivationFunction) -> f32 {
        match *activation {
            ActivationFunction::Sigmoid => Node::sigmoid(self.input()),
            ActivationFunction::Linear => Node::linear(self.input()),
            ActivationFunction::Tanh => Node::tanh(self.input()),
            // ActivationFunction::Step => Node::step(self.input()),
        }
    }

    pub fn compute_answer_err_sig(
        &mut self,
        cached_output: f32,
        activation: &ActivationFunction,
    ) -> f32 {
        let derivative: f32 = match activation {
            ActivationFunction::Sigmoid => cached_output * (1.0 - cached_output),
            ActivationFunction::Linear => 2.0,
            ActivationFunction::Tanh => 1.0, //- unsafe { std::intrinsics::powf32(y, 2.0) };
        };
        self.err_sig = Some((self.correct_answer.unwrap() - cached_output) * derivative);
        dbg_println!("Err Signal Post: {:?}", self.err_sig);
        self.err_sig.unwrap()
    }

    pub fn compute_answer_err_sig_gen(
        &mut self,
        mse: f32,
        cached_output: &str,
        activation: &ActivationFunction,
    ) -> f32 {
        // This is where the derivative of the activation function goes I think
        let derivative: f32 = match activation {
            ActivationFunction::Sigmoid => cached_output * (1.0 - cached_output),
            ActivationFunction::Linear => 2.0,
            ActivationFunction::Tanh => 1.0, //- unsafe { std::intrinsics::powf32(y, 2.0) };
        };
        self.err_sig = Some(mse * derivative);
        dbg_println!("Err Signal Post: {:?}", self.err_sig.unwrap());
        self.err_sig.unwrap()
    }

    pub fn adjust_weights(&mut self, learning_rate: f32) {
        self.b_weight = self.b_weight + self.err_sig.unwrap() * learning_rate;
        self.link_weights = (0..self.link_weights.len())
            .into_iter()
            .map(|link| {
                dbg_println!(
                    "\nInitial weights: {:?}\nLink Value: {:?}\nErr: {:?}",
                    self.link_weights[link],
                    self.link_vals[link],
                    self.err_sig
                );
                let new_weight: f32 = self.err_sig.unwrap() * self.link_vals[link] * learning_rate;
                dbg_println!("Adjusted Weight: {:?}\n", self.link_weights[link]);
                new_weight
            })
            .collect::<Box<[f32]>>()
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
}
