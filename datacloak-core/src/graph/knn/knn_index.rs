use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnnSearchResult {
    pub index: usize,
    pub distance: f32,
    pub id: Option<String>,
}

pub struct KnnIndex {
    dimension: usize,
    pure_rust_index: super::PureRustKnn,
    id_map: HashMap<usize, String>,
    next_id: usize,
}

impl KnnIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            pure_rust_index: super::PureRustKnn::new(dimension),
            id_map: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn new_pure_rust(dimension: usize) -> Self {
        Self::new(dimension)
    }

    pub fn add(&mut self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(anyhow!(
                "Vector dimension {} doesn't match index dimension {}",
                vector.len(),
                self.dimension
            ));
        }

        self.pure_rust_index.add(vector)?;
        self.next_id += 1;
        Ok(())
    }

    pub fn add_with_id(&mut self, vector: &[f32], id: &str) -> Result<()> {
        let index = self.next_id;
        self.add(vector)?;
        self.id_map.insert(index, id.to_string());
        Ok(())
    }

    pub fn add_batch(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        for vector in vectors {
            self.add(vector)?;
        }
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<KnnSearchResult>> {
        if query.len() != self.dimension {
            return Err(anyhow!(
                "Query dimension {} doesn't match index dimension {}",
                query.len(),
                self.dimension
            ));
        }

        if self.size() == 0 {
            return Ok(vec![]);
        }

        let results = self.pure_rust_index.search(query, k)?;

        Ok(results
            .into_iter()
            .map(|(idx, dist)| KnnSearchResult {
                index: idx,
                distance: dist,
                id: self.id_map.get(&idx).cloned(),
            })
            .collect())
    }

    pub fn batch_search(
        &self,
        queries: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<Vec<KnnSearchResult>>> {
        queries.iter().map(|query| self.search(query, k)).collect()
    }

    pub fn size(&self) -> usize {
        self.pure_rust_index.size()
    }

    pub fn save(&self, path: &str) -> Result<()> {
        // Save pure Rust index
        self.pure_rust_index.save(path)?;

        // Save ID map
        let id_map_path = format!("{}.ids", path);
        let mut file = File::create(id_map_path)?;
        let serialized = serde_json::to_string(&self.id_map)?;
        file.write_all(serialized.as_bytes())?;

        Ok(())
    }

    pub fn load(path: &str, dimension: usize) -> Result<Self> {
        let id_map_path = format!("{}.ids", path);
        let mut id_map = HashMap::new();

        if Path::new(&id_map_path).exists() {
            let mut file = File::open(id_map_path)?;
            let mut contents = String::new();
            file.read_to_string(&mut contents)?;
            id_map = serde_json::from_str(&contents)?;
        }

        // Load pure Rust index
        let pure_rust_index = super::PureRustKnn::load(path)?;

        let next_id = id_map.len();
        Ok(Self {
            dimension,
            pure_rust_index,
            id_map,
            next_id,
        })
    }
}
