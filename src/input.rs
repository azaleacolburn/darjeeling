#[allow(dead_code)]

#[derive(Debug, Clone)]
/// Represents the input for the neural network, in a way it can understand
pub struct Input {
    pub inputs: Vec<f32>,
    pub answer: String
}

impl Input {

    /// Creates new input
    /// # Params
    /// - Inputs: A list of 32-bit floating point numbers.
    /// - Answer: A string, representing the correct answer(or cateogry) of this input.
    /// 
    /// # Examples
    /// This example is for one input into an XOR gate
    /// ```
    /// use darjeeling::input::Input;
    /// let inputs: Vec<f32> = vec![0.0,1.0];
    /// let answer: String = String::from("1");
    /// let formated_input: Input = Input::new(inputs, answer);
    /// ```
    pub fn new(inputs: Vec<f32>, answer: String) -> Input {

        Input { inputs, answer }
    }

    // TODO: Write format_as_input function
}
