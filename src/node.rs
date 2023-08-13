use serde::{Deserialize, Serialize};
use crate::{DEBUG, types::Types, activation::ActivationFunction, dbg_println};

/// Represents a node in the network
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Node {
    pub link_weights: Vec<f32>,
    pub link_vals: Vec<Option<f32>>,
    pub links: usize,
    pub err_sig: Option<f32>,
    pub correct_answer: Option<f32>,
    pub cached_output: Option<f32>,
    pub category: Option<Types>,
    pub b_weight: Option<f32>,
}

impl Node {
    
    pub fn new(link_weights: &Vec<f32>, b_weight: Option<f32>) -> Node {

        let mut link_vals: Vec<Option<f32>> = vec![];
        for _i in 0..link_weights.len() {
            link_vals.push(None);
        }

        Node { link_weights: link_weights.to_vec(), link_vals, links: link_weights.len(), err_sig: None, correct_answer: None, cached_output: None, category: None, b_weight }
    }

    fn input(&mut self) -> f32 {
        let mut sum: f32 = 0.00;
        for i in 0..self.links {
            if DEBUG { println!("Link Val: {:?}", self.link_vals[i]); }
            let val: f32 = match self.link_vals[i] {

                Some(val) => val,
                None => 0.00
            };
            sum += val * self.link_weights[i]
        }

        sum + self.b_weight.unwrap()
    }

    pub fn output(&mut self, activation: &ActivationFunction) -> f32 {
        self.cached_output = Some(match *activation 
        {
            ActivationFunction::Sigmoid => Node::sigmoid(self.input()),

            ActivationFunction::Linear => Node::linear(self.input()),

            ActivationFunction::Tanh => Node::tanh(self.input()),

            // ActivationFunction::Step => Node::step(self.input()),
        });
        
        self.cached_output.unwrap()
    }
    
    pub fn compute_answer_err_sig(&mut self, activation: &ActivationFunction) {
        if DEBUG { println!("Err Signal Pre: {:?}", self.err_sig); }
        let y = self.cached_output.unwrap();
        let derivative: f32;
        match activation {
            ActivationFunction::Sigmoid => {
                derivative = y * (1.0 - y);
            },
            ActivationFunction::Linear => {
                derivative = 1.0;
            }, 
            ActivationFunction::Tanh => {
                derivative = 1.0 - unsafe { std::intrinsics::powf32(y, 2.0) };
            }
        }
        self.err_sig = Some((self.correct_answer.unwrap() - y) * derivative);
        if DEBUG { println!("Err Signal Post: {:?}", self.err_sig.unwrap()) }
    }

    pub fn compute_answer_err_sig_gen(&mut self, mse: f32, activation: &ActivationFunction) {
        if DEBUG { println!("Err Signal Pre: {:?}", self.err_sig); }
        // This is where the derivative of the activation function goes I think
        let y = self.cached_output.unwrap();
        let derivative: f32;
        match activation {
            ActivationFunction::Sigmoid => {
                derivative = y * (1.0 - y);
            },
            ActivationFunction::Linear => {
                derivative = 2.0;
            }, 
            ActivationFunction::Tanh => {
                derivative = 1.0 - unsafe { std::intrinsics::powf32(y, 2.0) };
            }
        }
        self.err_sig = Some(mse * derivative);
        if DEBUG { println!("Err Signal Post: {:?}", self.err_sig.unwrap()) }
    }

    pub fn adjust_weights(&mut self, learning_rate: f32) {
        self.b_weight = Some(self.b_weight.unwrap() + self.err_sig.unwrap() * learning_rate);
        for link in 0..self.links {
            dbg_println!("\nInitial weights: {:?}\nLink Value: {:?}\nErr: {:?}", self.link_weights[link], self.link_vals[link].unwrap(), self.err_sig);
            self.link_weights[link] += self.err_sig.unwrap() * self.link_vals[link].unwrap() * learning_rate;
            dbg_println!("Adjusted Weight: {:?}\n", self.link_weights[link]);
        }
    }

    fn sigmoid(x: f32) -> f32 {

        1.0/(1.0+((-x).exp()))
    }

    fn linear(x: f32) -> f32 {
        
        2.0 * x
    }

    fn tanh(x: f32) -> f32 {
        let e = std::f64::consts::E as f32;

        2.00 / (1.00 + e.powf(-2.00 * x)) - 1.00
    }

    fn step(x: f32) -> f32 {

        if x < 0.00 { -1.00 }
        else { 1.00 }
    }
}
