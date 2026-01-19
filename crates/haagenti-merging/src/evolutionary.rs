//! Evolutionary model merging using genetic algorithms

use crate::{MergeError, Result, WeightTensor, ModelWeights};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Evolutionary merge configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionaryConfig {
    /// Population size
    pub population_size: usize,
    /// Number of generations
    pub generations: usize,
    /// Mutation rate (0.0 - 1.0)
    pub mutation_rate: f32,
    /// Mutation strength (std dev for gaussian mutation)
    pub mutation_strength: f32,
    /// Crossover rate
    pub crossover_rate: f32,
    /// Elite count (preserved without mutation)
    pub elite_count: usize,
    /// Tournament size for selection
    pub tournament_size: usize,
    /// Random seed
    pub seed: Option<u64>,
    /// Early stopping patience
    pub early_stopping_patience: usize,
}

impl Default for EvolutionaryConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            generations: 100,
            mutation_rate: 0.1,
            mutation_strength: 0.1,
            crossover_rate: 0.8,
            elite_count: 2,
            tournament_size: 3,
            seed: None,
            early_stopping_patience: 10,
        }
    }
}

/// Genome representing merge weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    /// Model weights (one per source model)
    pub model_weights: Vec<f32>,
    /// Per-layer weight overrides (layer_name -> weights)
    pub layer_weights: HashMap<String, Vec<f32>>,
    /// Fitness score (higher is better)
    pub fitness: f32,
    /// Generation this genome was created
    pub generation: usize,
}

impl Genome {
    /// Create random genome
    pub fn random(num_models: usize, rng: &mut StdRng) -> Self {
        let model_weights: Vec<f32> = (0..num_models)
            .map(|_| rng.gen::<f32>())
            .collect();

        // Normalize
        let sum: f32 = model_weights.iter().sum();
        let model_weights: Vec<f32> = model_weights.iter().map(|w| w / sum).collect();

        Self {
            model_weights,
            layer_weights: HashMap::new(),
            fitness: f32::NEG_INFINITY,
            generation: 0,
        }
    }

    /// Create from specific weights
    pub fn from_weights(model_weights: Vec<f32>) -> Self {
        Self {
            model_weights,
            layer_weights: HashMap::new(),
            fitness: f32::NEG_INFINITY,
            generation: 0,
        }
    }

    /// Crossover with another genome
    pub fn crossover(&self, other: &Self, rng: &mut StdRng) -> Self {
        let crossover_point = rng.gen_range(0..self.model_weights.len());

        let mut new_weights = Vec::with_capacity(self.model_weights.len());
        for i in 0..self.model_weights.len() {
            if i < crossover_point {
                new_weights.push(self.model_weights[i]);
            } else {
                new_weights.push(other.model_weights[i]);
            }
        }

        // Normalize
        let sum: f32 = new_weights.iter().sum();
        if sum > 0.0 {
            for w in &mut new_weights {
                *w /= sum;
            }
        }

        Self {
            model_weights: new_weights,
            layer_weights: HashMap::new(),
            fitness: f32::NEG_INFINITY,
            generation: self.generation.max(other.generation) + 1,
        }
    }

    /// Mutate genome
    pub fn mutate(&mut self, rate: f32, strength: f32, rng: &mut StdRng) {
        for weight in &mut self.model_weights {
            if rng.gen::<f32>() < rate {
                *weight += rng.gen::<f32>() * strength * 2.0 - strength;
                *weight = weight.max(0.0);
            }
        }

        // Normalize
        let sum: f32 = self.model_weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.model_weights {
                *w /= sum;
            }
        }
    }

    /// Get weights for a layer
    pub fn weights_for_layer(&self, layer: &str) -> &[f32] {
        self.layer_weights
            .get(layer)
            .map(|w| w.as_slice())
            .unwrap_or(&self.model_weights)
    }
}

/// Evolutionary merger
pub struct EvolutionaryMerger {
    /// Configuration
    config: EvolutionaryConfig,
    /// Random number generator
    rng: StdRng,
    /// Current population
    population: Vec<Genome>,
    /// Best genome found
    best_genome: Option<Genome>,
    /// Fitness history
    fitness_history: Vec<f32>,
}

impl EvolutionaryMerger {
    /// Create new evolutionary merger
    pub fn new(config: EvolutionaryConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        Self {
            config,
            rng,
            population: Vec::new(),
            best_genome: None,
            fitness_history: Vec::new(),
        }
    }

    /// Initialize population
    pub fn initialize(&mut self, num_models: usize) {
        self.population.clear();

        for _ in 0..self.config.population_size {
            self.population.push(Genome::random(num_models, &mut self.rng));
        }
    }

    /// Tournament selection
    fn tournament_select(&mut self) -> &Genome {
        let mut best_idx = self.rng.gen_range(0..self.population.len());
        let mut best_fitness = self.population[best_idx].fitness;

        for _ in 1..self.config.tournament_size {
            let idx = self.rng.gen_range(0..self.population.len());
            if self.population[idx].fitness > best_fitness {
                best_idx = idx;
                best_fitness = self.population[idx].fitness;
            }
        }

        &self.population[best_idx]
    }

    /// Evolve one generation
    pub fn evolve<F>(&mut self, fitness_fn: &F) -> Result<f32>
    where
        F: Fn(&Genome) -> f32,
    {
        // Evaluate fitness
        for genome in &mut self.population {
            if genome.fitness == f32::NEG_INFINITY {
                genome.fitness = fitness_fn(genome);
            }
        }

        // Sort by fitness (descending)
        self.population.sort_by(|a, b| {
            b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update best
        if let Some(ref mut best) = self.best_genome {
            if self.population[0].fitness > best.fitness {
                *best = self.population[0].clone();
            }
        } else {
            self.best_genome = Some(self.population[0].clone());
        }

        let best_fitness = self.population[0].fitness;
        self.fitness_history.push(best_fitness);

        // Create new population
        let mut new_population = Vec::with_capacity(self.config.population_size);

        // Elitism
        for i in 0..self.config.elite_count.min(self.population.len()) {
            new_population.push(self.population[i].clone());
        }

        // Fill rest with crossover and mutation
        while new_population.len() < self.config.population_size {
            let parent1 = self.tournament_select().clone();
            let parent2 = self.tournament_select().clone();

            let mut child = if self.rng.gen::<f32>() < self.config.crossover_rate {
                parent1.crossover(&parent2, &mut self.rng)
            } else {
                parent1
            };

            child.mutate(
                self.config.mutation_rate,
                self.config.mutation_strength,
                &mut self.rng,
            );
            child.fitness = f32::NEG_INFINITY; // Will be evaluated next generation

            new_population.push(child);
        }

        self.population = new_population;

        Ok(best_fitness)
    }

    /// Run full evolution
    pub fn run<F>(&mut self, num_models: usize, fitness_fn: F) -> Result<Genome>
    where
        F: Fn(&Genome) -> f32,
    {
        self.initialize(num_models);

        let mut no_improvement = 0;
        let mut best_so_far = f32::NEG_INFINITY;

        for gen in 0..self.config.generations {
            let fitness = self.evolve(&fitness_fn)?;

            if fitness > best_so_far {
                best_so_far = fitness;
                no_improvement = 0;
            } else {
                no_improvement += 1;
            }

            // Early stopping
            if no_improvement >= self.config.early_stopping_patience {
                break;
            }
        }

        self.best_genome.clone().ok_or_else(|| {
            MergeError::EvolutionFailed("No best genome found".into())
        })
    }

    /// Apply best genome to merge models
    pub fn merge_with_genome(
        &self,
        models: &[&ModelWeights],
        genome: &Genome,
    ) -> Result<ModelWeights> {
        if models.is_empty() {
            return Err(MergeError::InvalidWeights("No models provided".into()));
        }

        if genome.model_weights.len() != models.len() {
            return Err(MergeError::ConfigError(
                "Genome weights don't match model count".into(),
            ));
        }

        let base = models[0];
        let mut result = ModelWeights::new("evolved_merge");

        for layer_name in base.layer_names() {
            let weights = genome.weights_for_layer(layer_name);

            let tensors: Vec<&WeightTensor> = models
                .iter()
                .filter_map(|m| m.get_layer(layer_name))
                .collect();

            if tensors.len() != models.len() {
                return Err(MergeError::MissingLayer(layer_name.to_string()));
            }

            // Weighted sum
            let n = tensors[0].data.len();
            let mut merged = vec![0.0f32; n];

            for (tensor, &weight) in tensors.iter().zip(weights) {
                for (i, &val) in tensor.data.iter().enumerate() {
                    merged[i] += val * weight;
                }
            }

            result.add_layer(WeightTensor {
                name: layer_name.to_string(),
                shape: tensors[0].shape.clone(),
                data: merged,
                dtype: tensors[0].dtype,
            });
        }

        Ok(result)
    }

    /// Get fitness history
    pub fn fitness_history(&self) -> &[f32] {
        &self.fitness_history
    }

    /// Get best genome
    pub fn best(&self) -> Option<&Genome> {
        self.best_genome.as_ref()
    }
}

impl std::fmt::Debug for EvolutionaryMerger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvolutionaryMerger")
            .field("config", &self.config)
            .field("population_size", &self.population.len())
            .field("best_fitness", &self.best_genome.as_ref().map(|g| g.fitness))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genome_random() {
        let mut rng = StdRng::seed_from_u64(42);
        let genome = Genome::random(4, &mut rng);

        assert_eq!(genome.model_weights.len(), 4);

        // Should sum to 1
        let sum: f32 = genome.model_weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_genome_crossover() {
        let mut rng = StdRng::seed_from_u64(42);

        let g1 = Genome::from_weights(vec![1.0, 0.0, 0.0]);
        let g2 = Genome::from_weights(vec![0.0, 0.0, 1.0]);

        let child = g1.crossover(&g2, &mut rng);

        assert_eq!(child.model_weights.len(), 3);

        // Should sum to 1
        let sum: f32 = child.model_weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_genome_mutate() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut genome = Genome::from_weights(vec![0.5, 0.5]);

        genome.mutate(1.0, 0.1, &mut rng);

        // Should still sum to 1
        let sum: f32 = genome.model_weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_evolution() {
        let config = EvolutionaryConfig {
            population_size: 10,
            generations: 5,
            seed: Some(42),
            ..Default::default()
        };

        let mut merger = EvolutionaryMerger::new(config);

        // Simple fitness function that prefers equal weights
        let fitness_fn = |g: &Genome| {
            let target = 1.0 / g.model_weights.len() as f32;
            -g.model_weights
                .iter()
                .map(|w| (w - target).powi(2))
                .sum::<f32>()
        };

        let best = merger.run(3, fitness_fn).unwrap();

        // Should converge toward equal weights
        assert!(best.fitness > -0.5);
    }

    #[test]
    fn test_tournament_selection() {
        let config = EvolutionaryConfig {
            population_size: 5,
            tournament_size: 2,
            seed: Some(42),
            ..Default::default()
        };

        let mut merger = EvolutionaryMerger::new(config);
        merger.initialize(3);

        // Set known fitnesses
        merger.population[0].fitness = 1.0;
        merger.population[1].fitness = 2.0;
        merger.population[2].fitness = 3.0;
        merger.population[3].fitness = 4.0;
        merger.population[4].fitness = 5.0;

        // Tournament should usually pick higher fitness
        let selected = merger.tournament_select();
        assert!(selected.fitness >= 1.0);
    }
}
