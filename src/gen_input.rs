use crate::{
    input::Input,
    types::Types
};

#[allow(dead_code)]

#[derive(Debug, Clone)]
/// Represents the input for the neural network
pub struct GenInput {
    pub inputs: Vec<f32>,
}

impl GenInput  {

    /// Creates new input
    /// # Params
    /// - Inputs: A list of 32-bit floating point numbers.
    /// - Answer: A string, representing the correct answer(or cateogry) of this input.
    /// 
    /// # Examples
    /// This example is for one input into an XOR gate
    /// ```
    /// use darjeeling::input::Input;
    /// use darjeeling::types::Types;
    /// let inputs: Vec<f32> = vec![0.0,1.0];
    /// let formated_input: Input = Input::new(inputs);
    /// ```
    pub fn new(inputs: Vec<f32>) -> GenInput {

        GenInput { inputs }
    }

    pub fn to_input(&self) -> Input {

        Input::new(self.inputs, Types::Integer(0))
    }
}
