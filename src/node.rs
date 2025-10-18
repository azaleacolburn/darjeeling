use crate::{activation::ActivationFunction, link::Link};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Represents a node in the network
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Node {
    pub links: Box<[Link]>,
    pub b_weight: f32,
    #[serde(skip_serializing)]
    pub err_sig: Option<f32>,
    #[serde(skip_serializing)]
    pub correct_answer: Option<f32>,
    #[serde(skip_serializing)]
    pub category: Option<String>,
    #[serde(skip_serializing)]
    pub cached_output: Option<f32>,
}

impl Node {
    pub fn new(link_count: usize) -> Node {
        let mut rng = rand::thread_rng();
        let b_weight = rng.gen_range(-0.05..0.05);
        let links = Link::generate_links(link_count, &mut rng);

        Node {
            links,
            b_weight,
            err_sig: None,
            cached_output: None,
            correct_answer: None,
            category: None,
        }
    }

    // Outputs the sum of all the link values times the link weights plus the bias for the neuron
    fn input(&mut self) -> f32 {
        self.links.iter().map(Link::evaluate).sum::<f32>() + self.b_weight
    }

    pub fn output(&mut self, activation: ActivationFunction) -> f32 {
        match activation {
            ActivationFunction::Sigmoid => Node::sigmoid(self.input()),
            ActivationFunction::Linear => Node::linear(self.input()),
            ActivationFunction::Tanh => Node::tanh(self.input()),
            // ActivationFunction::Step => Node::step(self.input()),
        }
    }

    // pub fn compute_answer_err_sig_gen(&mut self, mse: f32, activation: ActivationFunction) -> f32 {
    //     // This is where the derivative of the activation function goes I think
    //     let output = self
    //         .cached_output
    //         .expect("Answer Node Missing Cached Output");
    //     let slope: f32 = match activation {
    //         ActivationFunction::Sigmoid => output * (1.0 - output),
    //         ActivationFunction::Linear => 2.0,
    //         ActivationFunction::Tanh => 1.0, //- unsafe { std::intrinsics::powf32(y, 2.0) };
    //     };
    //     self.err_sig = Some(mse * slope);
    //     dbg_println!("Err Signal Post: {:?}", self.err_sig.unwrap());
    //     self.err_sig.unwrap()
    // }

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
