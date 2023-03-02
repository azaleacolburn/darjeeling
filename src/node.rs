#[allow(dead_code)]
use serde::{Deserialize, Serialize};
use crate::DEBUG;

/// Represents a node in the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub link_weights: Vec<f32>,
    pub link_vals: Vec<Option<f32>>,
    pub links: i32,
    pub err_sig: Option<f32>,
    pub correct_answer: Option<f32>,
    pub cached_output: Option<f32>,
    pub category: Option<String>,
    pub b_weight: Option<f32>,
}

impl Node {
    fn input(&mut self) -> Option<f32> {
        let mut sum:f32 = 0.00;
        for i in 0..self.links {
            if DEBUG { println!("Link Val: {:?}", self.link_vals[i as usize]); }
            sum += self.link_vals[i as usize].unwrap() * self.link_weights[i as usize];
        }

        Some(sum + self.b_weight.unwrap())
    }

    pub fn output(&mut self) -> Option<f32> {
        self.cached_output = Some(Node::sigmoid(self.input().unwrap()));
        
        self.cached_output
    }
    
    pub fn compute_answer_err_sig(&mut self) {
        if DEBUG { println!("Err Signal Pre: {:?}", self.err_sig); }
        self.err_sig = Some((self.correct_answer.unwrap() - self.cached_output.unwrap()) * self.cached_output.unwrap() * (1.00 - self.cached_output.unwrap()));
        if DEBUG { println!("Err Signal Post: {:?}", self.err_sig.unwrap()) }
    }

    pub fn adjust_weights(&mut self, learning_rate: f32){
        self.b_weight = Some(self.b_weight.unwrap() + self.err_sig.unwrap() * learning_rate);
        for link in 0..self.links {
            if DEBUG {
                println!("\nInitial weights: {:?}", self.link_weights[link as usize]);
                println!("Link Value: {:?}", self.link_vals[link as usize]);
                println!("Err: {:?}", self.err_sig);
            }
            self.link_weights[link as usize] += self.err_sig.unwrap() * self.link_vals[link as usize].unwrap() * learning_rate;
            if DEBUG { println!("Adjusted Weight: {:?}\n", self.link_weights[link as usize]); }
        }
    }

    fn sigmoid(x: f32) -> f32{

        1.0/(1.0+((-x).exp()))
    }
}
