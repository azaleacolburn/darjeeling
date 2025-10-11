use rand::{rngs::ThreadRng, Rng};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Link {
    pub weight: f32,
    pub value: f32,
}

impl Link {
    pub fn evaluate(&self) -> f32 {
        self.weight * self.value
    }

    pub fn generate_links(link_count: usize, rng: &mut ThreadRng) -> Box<[Link]> {
        (0..link_count)
            .map(|_| Link {
                weight: rng.gen_range(-0.05..0.05),
                value: 0.00,
            })
            .collect::<Box<[Link]>>()
    }
}
