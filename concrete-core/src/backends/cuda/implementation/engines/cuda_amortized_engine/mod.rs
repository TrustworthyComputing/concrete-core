use crate::backends::cuda::engines::CudaError;
use crate::backends::cuda::private::device::{CudaStream, GpuIndex, NumberOfGpus};
use crate::prelude::sealed::AbstractEngineSeal;
use crate::prelude::{AbstractEngine, SharedMemoryAmount};
use concrete_cuda::cuda_bind::cuda_get_number_of_gpus;
use std::sync::{Arc, RwLock};

/// A variant of CudaEngine exposed by the cuda backend.
///
/// This engine implements an amortized version of bootstrap on the GPU.
/// It is dedicated to the execution of bootstraps over larger amounts of
/// input ciphertexts than the CudaEngine's bootstrap implementation.
#[derive(Debug, Clone)]
pub struct AmortizedCudaEngine {
    streams: Vec<Arc<RwLock<CudaStream>>>,
    max_shared_memory: usize,
}

impl AbstractEngineSeal for AmortizedCudaEngine {}

impl AbstractEngine for AmortizedCudaEngine {
    type EngineError = CudaError;

    type Parameters = ();

    fn new(_parameters: Self::Parameters) -> Result<Self, Self::EngineError> {
        let number_of_gpus = unsafe { cuda_get_number_of_gpus() as usize };
        if number_of_gpus == 0 {
            Err(CudaError::DeviceNotFound)
        } else {
            let mut streams: Vec<Arc<RwLock<CudaStream>>> = Vec::new();
            for gpu_index in 0..number_of_gpus {
                streams.push(Arc::new(RwLock::new(CudaStream::new(GpuIndex(gpu_index))?)));
            }
            let max_shared_memory = streams[0].read().unwrap().get_max_shared_memory()?;

            Ok(AmortizedCudaEngine {
                streams,
                max_shared_memory: max_shared_memory as usize,
            })
        }
    }
}

impl AmortizedCudaEngine {
    /// Get the number of available GPUs from the engine
    pub fn get_number_of_gpus(&self) -> NumberOfGpus {
        NumberOfGpus(self.streams.len())
    }
    /// Get the Cuda streams from the engine
    pub fn get_cuda_streams(&self) -> &Vec<Arc<RwLock<CudaStream>>> {
        &self.streams
    }
    /// Get the size of the shared memory (on device 0)
    pub fn get_cuda_shared_memory(&self) -> SharedMemoryAmount {
        SharedMemoryAmount(self.max_shared_memory)
    }
}

mod lwe_ciphertext_vector_discarding_bootstrap;
