use anyhow::{anyhow, Result};
use std::alloc::{alloc, dealloc, Layout};
use std::marker::PhantomData;
use std::ptr::{self, NonNull};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_bytes: usize,
    pub allocated_bytes: usize,
    pub free_bytes: usize,
    pub allocation_count: usize,
    pub fragmentation_ratio: f32,
}

struct Block {
    ptr: NonNull<u8>,
    size: usize,
    is_free: bool,
}

pub struct MemoryPool {
    blocks: Arc<Mutex<Vec<Block>>>,
    total_size: usize,
    base_ptr: NonNull<u8>,
    alignment: usize,
}

pub struct PoolAllocation<T> {
    ptr: NonNull<T>,
    size: usize,
    pool: Arc<Mutex<Vec<Block>>>,
    _phantom: PhantomData<T>,
}

impl MemoryPool {
    pub fn new(size: usize) -> Self {
        let alignment = 64; // Cache line alignment
        let layout = Layout::from_size_align(size, alignment).unwrap();

        unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                panic!("Failed to allocate memory pool");
            }

            let base_ptr = NonNull::new(ptr).unwrap();
            let blocks = vec![Block {
                ptr: base_ptr,
                size,
                is_free: true,
            }];

            Self {
                blocks: Arc::new(Mutex::new(blocks)),
                total_size: size,
                base_ptr,
                alignment,
            }
        }
    }

    pub fn allocate<T>(&self, count: usize) -> Result<PoolAllocation<T>> {
        let size = count * std::mem::size_of::<T>();
        let _align = std::mem::align_of::<T>().max(self.alignment);

        let mut blocks = self.blocks.lock().unwrap();

        // Find a free block that fits
        for i in 0..blocks.len() {
            if blocks[i].is_free && blocks[i].size >= size {
                let block = &mut blocks[i];

                // Simple allocation without alignment for now
                if block.size >= size {
                    let block_ptr = block.ptr;
                    let block_size = block.size;
                    block.is_free = false;

                    // Split the block if there's significant space left
                    if block_size > size + 128 {
                        let remaining_size = block_size - size;
                        let new_block = Block {
                            ptr: unsafe { NonNull::new_unchecked(block_ptr.as_ptr().add(size)) },
                            size: remaining_size,
                            is_free: true,
                        };
                        block.size = size;
                        blocks.insert(i + 1, new_block);
                    }

                    let allocation_ptr = NonNull::new(block_ptr.as_ptr() as *mut T).unwrap();

                    // Initialize memory to zero
                    unsafe {
                        ptr::write_bytes(allocation_ptr.as_ptr(), 0, count);
                    }

                    return Ok(PoolAllocation {
                        ptr: allocation_ptr,
                        size,
                        pool: self.blocks.clone(),
                        _phantom: PhantomData,
                    });
                }
            }
        }

        Err(anyhow!("No suitable block found in memory pool"))
    }

    pub fn stats(&self) -> PoolStats {
        let blocks = self.blocks.lock().unwrap();

        let allocated_bytes: usize = blocks.iter().filter(|b| !b.is_free).map(|b| b.size).sum();

        let free_bytes: usize = blocks.iter().filter(|b| b.is_free).map(|b| b.size).sum();

        let allocation_count = blocks.iter().filter(|b| !b.is_free).count();

        let fragmentation_ratio = if blocks.len() > 1 {
            (blocks.len() as f32 - 1.0) / blocks.len() as f32
        } else {
            0.0
        };

        PoolStats {
            total_bytes: self.total_size,
            allocated_bytes,
            free_bytes,
            allocation_count,
            fragmentation_ratio,
        }
    }

    pub fn defragment(&self) {
        let mut blocks = self.blocks.lock().unwrap();

        // Merge adjacent free blocks
        let mut i = 0;
        while i < blocks.len() - 1 {
            if blocks[i].is_free && blocks[i + 1].is_free {
                // Check if blocks are adjacent
                let end_of_current = unsafe { blocks[i].ptr.as_ptr().add(blocks[i].size) };
                if end_of_current == blocks[i + 1].ptr.as_ptr() {
                    blocks[i].size += blocks[i + 1].size;
                    blocks.remove(i + 1);
                    continue;
                }
            }
            i += 1;
        }
    }
}

impl<T> PoolAllocation<T> {
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.ptr.as_ptr(), self.size / std::mem::size_of::<T>())
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size / std::mem::size_of::<T>())
        }
    }
}

impl<T> Drop for PoolAllocation<T> {
    fn drop(&mut self) {
        let mut blocks = self.pool.lock().unwrap();

        // Find the block and mark it as free
        let ptr_addr = self.ptr.as_ptr() as usize;
        for block in blocks.iter_mut() {
            let block_addr = block.ptr.as_ptr() as usize;
            if ptr_addr >= block_addr && ptr_addr < block_addr + block.size {
                block.is_free = true;
                break;
            }
        }
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::from_size_align(self.total_size, self.alignment).unwrap();
            dealloc(self.base_ptr.as_ptr(), layout);
        }
    }
}

unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}
unsafe impl<T: Send> Send for PoolAllocation<T> {}
unsafe impl<T: Sync> Sync for PoolAllocation<T> {}
