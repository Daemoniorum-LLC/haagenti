//! Genetic algorithm for architecture and hyperparameter search

use crate::{OptError, Result};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Genetic search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneticConfig {
    /// Population size
    pub population_size: usize,
    /// Number of generations
    pub generations: usize,
    /// Mutation rate
    pub mutation_rate: f32,
    /// Crossover rate
    pub crossover_rate: f32,
    /// Elite count (preserved without mutation)
    pub elite_count: usize,
    /// Tournament size for selection
    pub tournament_size: usize,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for GeneticConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            generations: 100,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            elite_count: 2,
            tournament_size: 3,
            seed: None,
        }
    }
}

/// Search space definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSpace {
    /// Dimensions and their ranges
    pub dimensions: Vec<Dimension>,
}

/// Single dimension in search space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dimension {
    /// Dimension name
    pub name: String,
    /// Dimension type
    pub dim_type: DimensionType,
}

/// Dimension type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DimensionType {
    /// Discrete choices
    Discrete { choices: Vec<i32> },
    /// Continuous range
    Continuous { min: f32, max: f32 },
    /// Boolean
    Boolean,
}

impl SearchSpace {
    /// Create new search space
    pub fn new() -> Self {
        Self { dimensions: Vec::new() }
    }

    /// Add discrete dimension
    pub fn add_discrete(&mut self, name: impl Into<String>, choices: Vec<i32>) -> &mut Self {
        self.dimensions.push(Dimension {
            name: name.into(),
            dim_type: DimensionType::Discrete { choices },
        });
        self
    }

    /// Add continuous dimension
    pub fn add_continuous(&mut self, name: impl Into<String>, min: f32, max: f32) -> &mut Self {
        self.dimensions.push(Dimension {
            name: name.into(),
            dim_type: DimensionType::Continuous { min, max },
        });
        self
    }

    /// Add boolean dimension
    pub fn add_boolean(&mut self, name: impl Into<String>) -> &mut Self {
        self.dimensions.push(Dimension {
            name: name.into(),
            dim_type: DimensionType::Boolean,
        });
        self
    }
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self::new()
    }
}

/// Individual in population
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Individual {
    /// Genes (parameter values)
    pub genes: HashMap<String, Gene>,
    /// Fitness score
    pub fitness: f32,
    /// Generation born
    pub generation: usize,
}

/// Gene value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Gene {
    Discrete(i32),
    Continuous(f32),
    Boolean(bool),
}

impl Gene {
    /// Get as i32
    pub fn as_int(&self) -> Option<i32> {
        match self {
            Gene::Discrete(i) => Some(*i),
            Gene::Boolean(b) => Some(if *b { 1 } else { 0 }),
            Gene::Continuous(f) => Some(*f as i32),
        }
    }

    /// Get as f32
    pub fn as_float(&self) -> Option<f32> {
        match self {
            Gene::Discrete(i) => Some(*i as f32),
            Gene::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
            Gene::Continuous(f) => Some(*f),
        }
    }

    /// Get as bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Gene::Boolean(b) => Some(*b),
            Gene::Discrete(i) => Some(*i != 0),
            Gene::Continuous(f) => Some(*f > 0.5),
        }
    }
}

impl Individual {
    /// Create random individual
    pub fn random(space: &SearchSpace, rng: &mut StdRng) -> Self {
        let mut genes = HashMap::new();

        for dim in &space.dimensions {
            let gene = match &dim.dim_type {
                DimensionType::Discrete { choices } => {
                    let idx = rng.gen_range(0..choices.len());
                    Gene::Discrete(choices[idx])
                }
                DimensionType::Continuous { min, max } => {
                    Gene::Continuous(rng.gen::<f32>() * (max - min) + min)
                }
                DimensionType::Boolean => Gene::Boolean(rng.gen()),
            };
            genes.insert(dim.name.clone(), gene);
        }

        Self {
            genes,
            fitness: f32::NEG_INFINITY,
            generation: 0,
        }
    }

    /// Crossover with another individual
    pub fn crossover(&self, other: &Self, space: &SearchSpace, rng: &mut StdRng) -> Self {
        let mut genes = HashMap::new();

        for dim in &space.dimensions {
            // Uniform crossover
            let gene = if rng.gen() {
                self.genes.get(&dim.name).cloned()
            } else {
                other.genes.get(&dim.name).cloned()
            };

            if let Some(g) = gene {
                genes.insert(dim.name.clone(), g);
            }
        }

        Self {
            genes,
            fitness: f32::NEG_INFINITY,
            generation: self.generation.max(other.generation) + 1,
        }
    }

    /// Mutate individual
    pub fn mutate(&mut self, space: &SearchSpace, rate: f32, rng: &mut StdRng) {
        for dim in &space.dimensions {
            if rng.gen::<f32>() < rate {
                let gene = match &dim.dim_type {
                    DimensionType::Discrete { choices } => {
                        let idx = rng.gen_range(0..choices.len());
                        Gene::Discrete(choices[idx])
                    }
                    DimensionType::Continuous { min, max } => {
                        // Gaussian mutation
                        if let Some(Gene::Continuous(old)) = self.genes.get(&dim.name) {
                            let std = (max - min) * 0.1;
                            let new_val = old + rng.gen::<f32>() * std * 2.0 - std;
                            Gene::Continuous(new_val.clamp(*min, *max))
                        } else {
                            Gene::Continuous(rng.gen::<f32>() * (max - min) + min)
                        }
                    }
                    DimensionType::Boolean => Gene::Boolean(rng.gen()),
                };
                self.genes.insert(dim.name.clone(), gene);
            }
        }
    }
}

/// Genetic search algorithm
#[derive(Debug)]
pub struct GeneticSearch {
    /// Configuration
    config: GeneticConfig,
    /// Search space
    space: SearchSpace,
    /// Current population
    population: Vec<Individual>,
    /// Best individual found
    best: Option<Individual>,
    /// Current generation
    generation: usize,
    /// RNG
    rng: StdRng,
}

impl GeneticSearch {
    /// Create new genetic search
    pub fn new(config: GeneticConfig, space: SearchSpace) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        Self {
            config,
            space,
            population: Vec::new(),
            best: None,
            generation: 0,
            rng,
        }
    }

    /// Initialize population
    pub fn initialize(&mut self) {
        self.population.clear();
        for _ in 0..self.config.population_size {
            self.population.push(Individual::random(&self.space, &mut self.rng));
        }
    }

    /// Evolve one generation
    pub fn evolve<F>(&mut self, fitness_fn: &F) -> Result<f32>
    where
        F: Fn(&Individual) -> f32,
    {
        // Evaluate fitness
        for ind in &mut self.population {
            if ind.fitness == f32::NEG_INFINITY {
                ind.fitness = fitness_fn(ind);
            }
        }

        // Sort by fitness (descending)
        self.population.sort_by(|a, b| {
            b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update best
        if let Some(ref mut best) = self.best {
            if self.population[0].fitness > best.fitness {
                *best = self.population[0].clone();
            }
        } else {
            self.best = Some(self.population[0].clone());
        }

        let best_fitness = self.population[0].fitness;

        // Create new population
        let mut new_pop = Vec::with_capacity(self.config.population_size);

        // Elitism
        for i in 0..self.config.elite_count.min(self.population.len()) {
            new_pop.push(self.population[i].clone());
        }

        // Fill with offspring
        while new_pop.len() < self.config.population_size {
            // Select parents by index to avoid borrow conflicts
            let parent1_idx = self.tournament_select_idx();
            let parent2_idx = self.tournament_select_idx();

            let mut child = if self.rng.gen::<f32>() < self.config.crossover_rate {
                let parent1 = &self.population[parent1_idx];
                let parent2 = &self.population[parent2_idx];
                parent1.crossover(parent2, &self.space, &mut self.rng)
            } else {
                self.population[parent1_idx].clone()
            };

            child.mutate(&self.space, self.config.mutation_rate, &mut self.rng);
            child.generation = self.generation + 1;

            new_pop.push(child);
        }

        self.population = new_pop;
        self.generation += 1;

        Ok(best_fitness)
    }

    /// Tournament selection - returns index to avoid borrow conflicts
    fn tournament_select_idx(&mut self) -> usize {
        let mut best_idx = self.rng.gen_range(0..self.population.len());
        let mut best_fitness = self.population[best_idx].fitness;

        for _ in 1..self.config.tournament_size {
            let idx = self.rng.gen_range(0..self.population.len());
            if self.population[idx].fitness > best_fitness {
                best_idx = idx;
                best_fitness = self.population[idx].fitness;
            }
        }

        best_idx
    }

    /// Run full search
    pub fn run<F>(&mut self, fitness_fn: F) -> Result<Individual>
    where
        F: Fn(&Individual) -> f32,
    {
        self.initialize();

        for _ in 0..self.config.generations {
            self.evolve(&fitness_fn)?;
        }

        self.best.clone().ok_or_else(|| {
            OptError::OptimizationFailed("No best individual found".into())
        })
    }

    /// Get best individual
    pub fn best(&self) -> Option<&Individual> {
        self.best.as_ref()
    }

    /// Current generation
    pub fn current_generation(&self) -> usize {
        self.generation
    }

    /// Get population
    pub fn population(&self) -> &[Individual] {
        &self.population
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_space() {
        let mut space = SearchSpace::new();
        space
            .add_discrete("num_layers", vec![2, 4, 6, 8])
            .add_continuous("learning_rate", 0.0001, 0.1)
            .add_boolean("use_dropout");

        assert_eq!(space.dimensions.len(), 3);
    }

    #[test]
    fn test_individual_creation() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut space = SearchSpace::new();
        space.add_continuous("x", 0.0, 1.0);

        let ind = Individual::random(&space, &mut rng);
        assert!(ind.genes.contains_key("x"));
    }

    #[test]
    fn test_genetic_search() {
        let config = GeneticConfig {
            population_size: 10,
            generations: 5,
            seed: Some(42),
            ..Default::default()
        };

        let mut space = SearchSpace::new();
        space.add_continuous("x", -5.0, 5.0);

        let mut search = GeneticSearch::new(config, space);

        // Optimize x^2
        let fitness_fn = |ind: &Individual| {
            let x = ind.genes.get("x").and_then(|g| g.as_float()).unwrap_or(0.0);
            -(x * x) // Maximize negative x^2 = minimize x^2
        };

        let best = search.run(fitness_fn).unwrap();
        let x = best.genes.get("x").and_then(|g| g.as_float()).unwrap();

        // Should be close to 0
        assert!(x.abs() < 2.0);
    }

    #[test]
    fn test_crossover() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut space = SearchSpace::new();
        space.add_continuous("x", 0.0, 1.0);
        space.add_continuous("y", 0.0, 1.0);

        let ind1 = Individual::random(&space, &mut rng);
        let ind2 = Individual::random(&space, &mut rng);

        let child = ind1.crossover(&ind2, &space, &mut rng);
        assert!(child.genes.contains_key("x"));
        assert!(child.genes.contains_key("y"));
    }
}
