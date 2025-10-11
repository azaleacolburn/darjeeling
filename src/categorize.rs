use crate::{
    activation::ActivationFunction, bench, cond_println, dbg_println, error::DarjeelingError,
    neural_network::NeuralNetwork, node::Node, series::Series, utils::RandomIter, DEBUG,
};
use rand::{rngs::ThreadRng, Rng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fmt::{self, Debug},
    fs,
};

/// The categorization Neural Network struct
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CatNetwork {
    node_array: Box<[Box<[Node]>]>,
    activation_function: Option<ActivationFunction>,
}

fn generate_layer(layer_size: usize, prev_layer_size: usize) -> Box<[Node]> {
    (0..layer_size)
        .into_iter()
        .map(|_| Node::new(prev_layer_size))
        .collect::<Box<[Node]>>()
}

impl NeuralNetwork for CatNetwork {
    /// Constructor function for a categorization neural network
    /// Fills a Neural Network's node_array with empty nodes.
    /// Initializes random starting link and bias weights between -.5 and .5
    ///
    /// ## Params
    /// - Inputs: The number of sensors in the input layer
    /// - Hidden: The number of hidden nodes per hidden layer
    /// - Answer: The number of answer nodes, or possible categories
    /// - Hidden Layers: The number of different hidden layers
    /// - Activation Function: Which activation function is used by the network. This can be changed layer with the [`set_activation_func`](fn@set_activation_func) method.
    ///
    /// ## Examples
    /// ``` rust
    /// use darjeeling::{
    ///     activation::ActivationFunction,
    ///     categorize::CatNetwork
    /// };
    ///
    /// let inputs: usize = 10;
    /// let hidden: usize = 40;
    /// let answer: usize = 2;
    /// let hidden_layers: usize = 1;
    /// let mut net = CatNetwork::new(inputs, hidden, answer, hidden_layers, ActivationFunction::Sigmoid);
    /// ```
    fn new(
        input_nodes: usize,
        hidden_nodes: usize,
        answer_nodes: usize,
        hidden_layers: usize,
        activation_function: Option<ActivationFunction>,
    ) -> CatNetwork {
        // links point backwards to previous layer
        let mut node_array: Vec<Box<[Node]>> = Vec::with_capacity(2 + hidden_layers);

        let input_layer = generate_layer(input_nodes, 0); // might need to be 1
        node_array.push(input_layer);

        // links point backwards, so the first hidden layer has a different number of them
        let first_hidden_layer = generate_layer(hidden_nodes, input_nodes);
        node_array.push(first_hidden_layer);

        for _ in 1..hidden_layers {
            let hidden_vec: Box<[Node]> = generate_layer(hidden_nodes, hidden_nodes);
            node_array.push(hidden_vec);
        }

        let answer_layer = generate_layer(answer_nodes, hidden_nodes);
        node_array.push(answer_layer);

        CatNetwork {
            node_array: node_array.into_boxed_slice(),
            activation_function,
        }
    }

    /// Trains the neural network model to be able to categorize items in a dataset into given categories
    ///
    /// ## Params
    /// - Data: List of inputs
    /// - Categories: List of Strings, each denoting an answer category.
    /// The number of answer nodes should be the same of the number of categories
    /// - Learning Rate: The modifier that is applied to link weights as they're adjusted.
    /// Try fiddling with this one, but -1.5 - 1.5 is recommended to start.
    /// - Name: The name of the network
    /// - Target Error Percent: The error percent at which the network will be stop training, checked at the begining of each new epoch.
    /// - Write: True of you want to write the model to a file, false otherwise
    ///
    /// ## Returns
    /// The fallible:
    /// - name of the model that this neural network trained(the name parameter with a random u32 appended)
    /// some if write is true, none is write is false
    /// - the error percentage of the last epoch
    /// - the mse of the training
    ///
    /// ## Err
    /// - ### WriteModelFailed
    /// There was a problem when saving the model to a file
    /// - ### ModelNameAlreadyExists
    /// The random model name chosen already exists
    ///
    /// Change the name or retrain
    /// - ### UnknownError
    /// Not sure what happened, but something failed
    ///
    /// Make an issue on the [darjeeling](https://github.com/Ewie21/darjeeling) github page
    ///
    /// Or contact me at azaleacolburn@gmail.com
    ///
    /// ## Examples
    /// ```ignore
    /// use darjeeling::{
    /// categorize::CatNetwork,
    /// activation::ActivationFunction,
    /// input::Input,
    /// // This file may not be avaliable
    /// // Everything found here will be hyper-specific to your project.
    /// tests::{categories_str_format, xor_file}
    /// };
    ///
    /// // A file containing all possible inputs and correct outputs still needs to be make by you
    /// // 0 0;0
    /// // 0 1;1
    /// // 1 0;1
    /// // 1 1;0
    /// // You also need to write the file input function
    /// // Automatic file reading and formatting function coming soon
    /// let categories: Vec<String> = categories_str_format(vec!["0", "1"]);
    /// let mut data: Vec<Input> = xor_file();
    /// let mut net = CatNetwork::new(2, 2, 2, 1, ActivationFunction::Sigmoid);
    /// let learning_rate = 1.0;
    /// let (model_name, error_percentage, mse) = net.train(&data, categories, learning_rate, "xor", 99.0, true).unwrap();
    /// ```
    fn train(
        &mut self,
        data: &Box<[Series]>,
        categories: Box<[String]>,
        learning_rate: f32,
        name: &str,
        target_err_percent: f32,
        write: bool,
        print: bool,
    ) -> Result<(Option<String>, f32, f32), DarjeelingError> {
        let activation_function = match self.activation_function {
            Some(s) => s,
            None => return Err(DarjeelingError::ModelMissingActivationFunction),
        };

        let mut epochs = 0.0;
        let mut sum = 0.0;
        let mut count = 0.0;
        let mut err_percent = 0.0;
        let mut mse = 0.0;

        dbg_println!("Categorize");
        bench!(self.categorize(categories));

        while err_percent < target_err_percent {
            count = 0.0;
            sum = 0.0;

            let data_iter = RandomIter::new(data);
            for series in data_iter {
                dbg_println!("Training Checkpoint One Passed");

                dbg_println!("Assign");
                self.assign_answers(series);

                dbg_println!("Push");
                self.feedforward_pass(series, activation_function);

                dbg_println!("Sum: {:?} Count: {:?}", sum, count);
                dbg_println!("Analysis");
                if print {
                    self.self_analysis(&mut Some(epochs), &mut sum, &mut count, series, &mut mse);
                }

                dbg_println!("Sum: {:?} Count: {:?}", sum, count);

                dbg_println!("Backpropogate");
                self.backpropogate(learning_rate, activation_function);
            }

            err_percent = (sum / count) * 100.0;
            epochs += 1.0;
            if print {
                println!("Epoch: {:?}", epochs);
                println!("Training Accuracy: {:?}", err_percent);
            }
            //if err_percent - old_err_percent < 0.00000001 { break; }
        }
        let mut model_name: Option<String> = None;
        if write {
            model_name = Some(self.write_model(&name)?);
        }

        if print {
            println!("Training: Finished with accuracy of {:?}/{:?} or {:?} percent after {:?} epochs\nmse: {}", sum, count, err_percent, epochs, mse);
        }

        Ok((model_name, err_percent, mse))
    }

    /// Tests a pretrained model
    fn test(
        &mut self,
        data: &Box<[Series]>,
        categories: Box<[String]>,
        print: bool,
    ) -> Result<Vec<String>, DarjeelingError> {
        let mut sum = 0.0;
        let mut count = 0.0;
        // let mut category: Option<Types> = None;
        let mut answers: Vec<String> = vec![];
        let mut mse = 0.0;

        let activation_function = match self.activation_function {
            Some(s) => s,
            None => return Err(DarjeelingError::ModelMissingActivationFunction),
        };

        self.categorize(categories);

        data.iter().for_each(|series| {
            dbg_println!("Testing Checkpoint One Passed");
            self.feedforward_pass(series, activation_function);
            dbg_println!("Sum: {:?} Count: {:?}", sum, count);
            answers.push(
                self.self_analysis(&mut None, &mut sum, &mut count, series, &mut mse)
                    .0
                    .clone(),
            );

            dbg_println!("Sum: {:?} Count: {:?}", sum, count);

            if print {
                println!("Correct answer: {:?}", series.answer)
            }
        });

        let err_percent: f32 = (sum / count) * 100.0;
        mse /= count;
        if print {
            println!(
                "Testing: Finished with accuracy of {:?}/{:?} or {:?} percent\nMSE: {}",
                sum, count, err_percent, mse
            );
        }

        Ok(answers)
    }
}

impl CatNetwork {
    fn answer_layer(&mut self) -> &mut Box<[Node]> {
        self.node_array
            .last_mut()
            .expect("Network has no answer layer")
    }
    /// Assigns categories to answer nodes based on a list of given categories
    fn categorize(&mut self, categories: Box<[String]>) {
        self.answer_layer()
            .iter_mut()
            .enumerate()
            .for_each(|(i, node)| node.category = Some(categories[i].clone()));
    }

    fn assign_answers(&mut self, input: &Series) {
        self.answer_layer().iter_mut().for_each(|node| {
            node.correct_answer = if node.category.clone().unwrap() == input.answer {
                Some(1.0)
            } else {
                Some(0.0)
            }
        });
    }

    /// Passes in data to the sensors, pushs data 'downstream' through the network
    fn feedforward_pass(&mut self, data: &Series, activation_function: ActivationFunction) {
        // Pass data to the input layer
        self.node_array
            .first_mut()
            .expect("Neural Network has no layers")
            .iter_mut()
            .enumerate()
            .for_each(|(i, node)| {
                node.cached_output = Some(data.data[i]);
            });

        // Push forward hidden and output layers
        for layer_i in 1..self.node_array.len() {
            // Clone the cached outputs from the previous layer
            let prev_cached_outputs: Box<[f32]> = self.node_array[layer_i - 1]
                .iter()
                .map(|prev_node| {
                    prev_node
                        .cached_output
                        .expect("Previous nodes do not have cached outputs")
                })
                .collect();

            let layer = self.node_array[layer_i].iter_mut();

            layer.for_each(|node| {
                prev_cached_outputs
                    .iter()
                    .enumerate()
                    .for_each(|(prev_node_i, prev_output)| {
                        node.links[prev_node_i].value = *prev_output;
                        node.cached_output = Some(node.output(activation_function));
                    });
            });
        }
    }

    /// Analyses the chosen answer node's result.
    /// Also increments sum and count
    fn self_analysis(
        &self,
        epochs: &mut Option<f32>,
        sum: &mut f32,
        count: &mut f32,
        series: &Series,
        mse: &mut f32,
    ) -> (String, Option<f32>) {
        dbg_println!("answer {}", self.node_array.len() - 1);
        dbg_println!("largest index {}", self.largest_node());
        dbg_println!("{:?}", self);

        let answer_layer = self.node_array.last().expect("Network has no layer");
        let brightest_node: &Node = &answer_layer[self.largest_node()];
        let brightness: f32 = brightest_node.cached_output.unwrap();

        dbg_println!(
            "Category: {:?} \nBrightness: {:?}",
            brightest_node.category.as_ref().unwrap(),
            brightness
        );

        if brightest_node.category.clone().unwrap() == series.answer {
            dbg_println!("Sum++");
            *sum += 1.0;
        }

        *count += 1.0;

        match epochs {
            Some(epochs) => {
                // This won't happen during testing
                if *epochs % 10.0 != 0.0 || *epochs == 0.0 {
                    return (brightest_node.category.clone().unwrap(), None);
                }
                println!("\n-------------------------\n");
                println!("Epoch: {:?}", epochs);
                println!(
                    "Category: {:?} \nBrightness: {:?}",
                    brightest_node.category.as_ref().unwrap(),
                    brightness
                );
                if DEBUG {
                    let dimest_node: &Node =
                        &answer_layer[answer_layer.len() - self.largest_node() - 1];
                    println!(
                        "Chosen category: {:?} \nDimest Brightness: {:?}",
                        dimest_node.category.as_ref().unwrap(),
                        dimest_node.cached_output.unwrap()
                    );
                }
                (brightest_node.category.clone().unwrap(), None)
            }
            None => (
                brightest_node.category.clone().unwrap(),
                Some(CatNetwork::calculate_err_for_generation_model(
                    mse,
                    brightest_node,
                )),
            ),
        }
    }

    pub fn calculate_err_for_generation_model(mse: &mut f32, node: &Node) -> f32 {
        *mse += f32::powi(
            node.correct_answer.unwrap() - node.cached_output.unwrap(),
            2,
        );
        *mse
    }

    /// Finds the index and the brightest node in an array and returns it
    fn largest_node(&self) -> usize {
        let mut largest_index = 0;
        let answer_layer = self.node_array.last().expect("Network has no layers");
        for (i, node) in answer_layer.iter().enumerate() {
            if node.cached_output > answer_layer[largest_index].cached_output {
                largest_index = i;
            }
        }

        largest_index
    }
    /// Goes back through the network adjusting the weights of the all the neurons based on their error signal
    fn backpropogate(&mut self, learning_rate: f32, activation_function: ActivationFunction) {
        let hidden_layers = self.node_array.len() - 2;

        self.answer_layer().iter_mut().for_each(|node| {
            node.compute_answer_err_sig(activation_function);
        });

        self.adjust_hidden_weights(learning_rate, hidden_layers);

        self.answer_layer().iter_mut().for_each(|node| {
            node.adjust_weights(learning_rate);
        });
    }

    // Modifies the error signal
    fn compute_hidden_node_err_signal(
        hidden_layer: &mut Box<[Node]>,
        node: usize,
        next_layer: &Box<[Node]>,
    ) {
        hidden_layer[node].err_sig = Some(0.0);
        // link weights, err sigs
        let next_layer_eval: Box<[(f32, f32)]> = next_layer
            .iter()
            .map(|next_node| {
                (
                    next_node.links[node].weight,
                    next_node.err_sig.unwrap_or(0.00),
                )
            })
            .collect();

        let err_sum: f32 = next_layer_eval
            .iter()
            .map(|(next_weight, err_sig)| {
                // This changes based on the activation function
                let product = err_sig * next_weight;

                dbg_println!("next err sig {:?}", err_sig);
                dbg_println!("next weight {:?}", next_weight);

                product
            })
            .sum();

        let hidden_result = hidden_layer[node].cached_output.unwrap();
        // TODO: This is contains the derivative and changes based on the activation function
        hidden_layer[node].err_sig = Some(err_sum * hidden_result * (1.0 - hidden_result));
    }

    fn adjust_hidden_layer_weight(&mut self, hidden_layer_number: usize, learning_rate: f32) {
        let next_layer = self.node_array[hidden_layer_number + 1].clone();
        let hidden_layer = &mut self.node_array[hidden_layer_number];

        for node in 0..hidden_layer.len() {
            CatNetwork::compute_hidden_node_err_signal(hidden_layer, node, &next_layer);

            dbg_println!(
                "New hidden errsig multiply: {:?}",
                hidden_layer[node].err_sig
            );
            dbg_println!("\nLayer: {:?}", hidden_layer);
            dbg_println!("Node: {:?}", node);

            hidden_layer[node].adjust_weights(learning_rate);
        }
    }

    /// Adjusts the weights of all the hidden neurons in a network
    fn adjust_hidden_weights(&mut self, learning_rate: f32, hidden_layers: usize) {
        for hidden_layer in 1..=hidden_layers {
            self.adjust_hidden_layer_weight(hidden_layer, learning_rate);
        }
    }

    /// Serializes a trained model so it can be used later
    ///
    /// ## Returns
    /// The name of the model
    ///
    /// ## Err
    /// ### WriteModelFailed:
    /// Wraps the models name
    /// ### UnknownError:
    /// Wraps error
    ///  
    pub fn write_model(&mut self, name: &str) -> Result<String, DarjeelingError> {
        let bin = match bincode::serialize(self) {
            Ok(v) => v,
            Err(err) => return Err(DarjeelingError::SaveModelFailed(err.to_string())),
        };

        let mut rng = rand::thread_rng();
        let num = rng.gen_range(0..i32::MAX);

        let model_name = format!("{}_{}.darj", name, num);
        match fs::write(name, bin) {
            Ok(_) => Ok(model_name),
            Err(err) => Err(DarjeelingError::SaveModelFailed(err.to_string())),
        }
    }

    /// Reads a file containing a serizalized Categorization Network
    ///
    /// ## Params
    /// - Model Name: The name(or more helpfully the path) of the model to be read
    ///
    /// ## Returns
    /// A neural network read from a serialized .darj file
    ///
    /// ## Err
    /// If the file cannnot be read, or if the file does not contain a valid serialized Neural Network
    pub fn read_model(model_name: String) -> Result<CatNetwork, DarjeelingError> {
        let read_bin = match fs::read(&model_name) {
            Ok(bin) => bin,
            Err(err) => return Err(DarjeelingError::LoadModelFailed(err.to_string())),
        };

        match bincode::deserialize(&read_bin) {
            Ok(net) => Ok(net),
            Err(err) => Err(DarjeelingError::LoadModelFailed(err.to_string())),
        }
    }

    pub fn set_activation_func(&mut self, new_activation_function: ActivationFunction) {
        self.activation_function = Some(new_activation_function);
    }
}

impl fmt::Display for CatNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buff = String::from("");
        self.node_array.iter().for_each(|layer| {
            layer.iter().for_each(|node| {
                buff.push_str(format!("{:?}", node).as_str());
            })
        });
        write!(f, "{}", buff)
    }
}
