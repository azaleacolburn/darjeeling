// Deprecated

use std::fmt::Debug;
use crate::types::Types;

#[derive(Debug, Clone)]
pub struct Series<I: Clone + Debug + PartialEq> {
    data: Vec<Types>,
    indexes: Vec<I>
}

impl<'a, I: Clone + Debug + PartialEq> Series<I> {

    pub fn new(data: Vec<Types>, indexes: Vec<I>) -> Series<I> {

        Series { data, indexes }
    }

    pub fn mut_add(&mut self, data: Types, index: I) {

        self.data.push(data);
        self.indexes.push(index);

    }

    pub fn no_mut_add(&self, data: Types, index: I) -> Series<I> {

        let mut new_self = self.clone();

        new_self.data.push(data);
        new_self.indexes.push(index);

        new_self
    }

    pub fn mut_sub(&mut self, index: I) -> Result<(), &str> {

        let mut index_to_remove: Option<usize> = None;
        for i in 0..self.indexes.len() {
            if self.indexes[i] == index {
                index_to_remove = Some(i);
                self.indexes.remove(i);
            }
        }
        match index_to_remove {

            Some(i) => self.data.remove(i),
            None => return Err("Index not found")
        };
        
        Ok(())    
    }

    pub fn no_mut_sub(&self, index: I) -> Result<Series<I>, &str> {

        let mut new_self: Series<I> = self.clone();
        let mut index_to_remove:Option<usize> = None;
        for i in 0..self.indexes.len() {
            if new_self.indexes[i] == index {
                index_to_remove = Some(i);
                new_self.indexes.remove(i);
            }
        }
        match index_to_remove {

            Some(i) => new_self.data.remove(i),
            None => return Err("Index not found")
        };
        
        Ok(new_self)   
    }

    pub fn get(&self, index: I) -> Option<Types> {
        for i in 0..self.indexes.len() {
            if self.indexes[i] == index {
                return Some(self.data[i].clone());
            }
        }
        return None;
    } 

    pub fn display(&self) {

        assert_eq!(self.data.len(), self.indexes.len());

        for i in 0..self.data.len() {
            print!("{:?} ", self.indexes[i]);
            self.data[i].display();
            print!("\n");
        }
        println!("\n");
    }
}
