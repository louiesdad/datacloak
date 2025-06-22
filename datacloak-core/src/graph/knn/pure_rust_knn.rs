use anyhow::{Result, anyhow};
use ordered_float::OrderedFloat;
use std::collections::BinaryHeap;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{Write, Read};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PureRustKnn {
    dimension: usize,
    vectors: Vec<Vec<f32>>,
}

impl PureRustKnn {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            vectors: Vec::new(),
        }
    }
    
    pub fn add(&mut self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(anyhow!(
                "Vector dimension {} doesn't match index dimension {}",
                vector.len(),
                self.dimension
            ));
        }
        
        self.vectors.push(vector.to_vec());
        Ok(())
    }
    
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        if query.len() != self.dimension {
            return Err(anyhow!(
                "Query dimension {} doesn't match index dimension {}",
                query.len(),
                self.dimension
            ));
        }
        
        if self.vectors.is_empty() {
            return Ok(vec![]);
        }
        
        // Use a max heap to keep track of k nearest neighbors
        let mut heap: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();
        
        for (idx, vector) in self.vectors.iter().enumerate() {
            let distance = euclidean_distance(query, vector);
            let ordered_dist = OrderedFloat(distance);
            
            if heap.len() < k {
                heap.push((ordered_dist, idx));
            } else if let Some(&(max_dist, _)) = heap.peek() {
                if ordered_dist < max_dist {
                    heap.pop();
                    heap.push((ordered_dist, idx));
                }
            }
        }
        
        // Extract results and sort by distance
        let mut results: Vec<(usize, f32)> = heap
            .into_iter()
            .map(|(dist, idx)| (idx, dist.0))
            .collect();
        
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        Ok(results)
    }
    
    pub fn batch_search(&self, queries: &[Vec<f32>], k: usize) -> Result<Vec<Vec<(usize, f32)>>> {
        queries.iter()
            .map(|query| self.search(query, k))
            .collect()
    }
    
    pub fn size(&self) -> usize {
        self.vectors.len()
    }
    
    pub fn save(&self, path: &str) -> Result<()> {
        let serialized = bincode::serialize(self)?;
        let mut file = File::create(path)?;
        file.write_all(&serialized)?;
        Ok(())
    }
    
    pub fn load(path: &str) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let index = bincode::deserialize(&buffer)?;
        Ok(index)
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pure_rust_knn_basic() {
        let mut index = PureRustKnn::new(2);
        
        index.add(&vec![0.0, 0.0]).unwrap();
        index.add(&vec![1.0, 0.0]).unwrap();
        index.add(&vec![0.0, 1.0]).unwrap();
        
        let results = index.search(&vec![0.1, 0.1], 2).unwrap();
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Closest to origin
        assert!(results[0].1 < results[1].1);
    }
    
    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert_eq!(euclidean_distance(&a, &b), 5.0);
        
        let c = vec![1.0, 1.0, 1.0];
        let d = vec![1.0, 1.0, 1.0];
        assert_eq!(euclidean_distance(&c, &d), 0.0);
    }
    
    #[test]
    fn test_save_load() {
        let mut index = PureRustKnn::new(3);
        index.add(&vec![1.0, 0.0, 0.0]).unwrap();
        index.add(&vec![0.0, 1.0, 0.0]).unwrap();
        
        let temp_path = "/tmp/test_pure_rust_knn.bin";
        index.save(temp_path).unwrap();
        
        let loaded = PureRustKnn::load(temp_path).unwrap();
        assert_eq!(loaded.size(), 2);
        assert_eq!(loaded.dimension, 3);
        
        std::fs::remove_file(temp_path).ok();
    }
}