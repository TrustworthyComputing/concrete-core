use crate::backends::cuda::private::device::{CudaStream, GpuIndex, NumberOfGpus};
use crate::prelude::sealed::AbstractEngineSeal;
use crate::prelude::{AbstractEngine, CudaError, SharedMemoryAmount};
use concrete_cuda::cuda_bind::{cuda_get_number_of_gpus, cuda_get_number_of_sms};
use std::sync::{Arc, RwLock};
/// The main engine exposed by the cuda backend.
///
/// This engine handles single-GPU and multi-GPU computations for the user. It always associates
/// one Cuda stream to each available Nvidia GPU, and splits the input ciphertexts evenly over
/// the GPUs (the last GPU may be a bit more loaded if the number of GPUs does not divide the
/// number of input ciphertexts). This engine does not give control over the streams, nor the GPU
/// load balancing. In this way, we can overlap computations done on different GPUs, but not
/// computations done on a given GPU, which are executed in a sequence.
// A finer access to streams could allow for more overlapping of computations
// on a given device. We'll probably want to support it in the future, in an AdvancedCudaEngine
// for example.

#[derive(Debug, Clone)]
pub struct CudaEngine {
    streams: Vec<Arc<RwLock<CudaStream>>>,
    max_shared_memory: usize,
    number_of_gpus: usize,
}

impl AbstractEngineSeal for CudaEngine {}

impl AbstractEngine for CudaEngine {
    type EngineError = CudaError;

    type Parameters = ();

    fn new(_parameters: Self::Parameters) -> Result<Self, Self::EngineError> {
        let number_of_gpus = unsafe { cuda_get_number_of_gpus() as usize };
        if number_of_gpus == 0 {
            Err(CudaError::DeviceNotFound)
        } else {
            let number_of_sms = unsafe { cuda_get_number_of_sms(0) as usize };
            let mut streams: Vec<Arc<RwLock<CudaStream>>> = Vec::new();
            for gpu_index in 0..(number_of_gpus * number_of_sms) {
                let curr_gpu: usize = gpu_index / number_of_sms;
                streams.push(Arc::new(RwLock::new(CudaStream::new(GpuIndex(curr_gpu))?)));
            }
            let max_shared_memory = streams[0].read().unwrap().get_max_shared_memory()?;

            Ok(CudaEngine {
                streams,
                max_shared_memory: max_shared_memory as usize,
                number_of_gpus: number_of_gpus as usize,
            })
        }
    }
}

impl CudaEngine {
    /// Get the number of available GPUs from the engine
    pub fn get_number_of_gpus(&self) -> NumberOfGpus {
        NumberOfGpus(self.number_of_gpus)
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

macro_rules! check_poly_size {
    ($poly_size: ident) => {
        if $poly_size.0 != 256
            && $poly_size.0 != 512
            && $poly_size.0 != 1024
            && $poly_size.0 != 2048
            && $poly_size.0 != 4096
            && $poly_size.0 != 8192
        {
            return Err(CudaError::PolynomialSizeNotSupported.into());
        }
    };
}

mod cleartext_vector_conversion;
mod ggsw_ciphertext_conversion;
mod glwe_ciphertext_conversion;
mod glwe_ciphertext_discarding_conversion;
mod glwe_ciphertext_vector_conversion;
mod glwe_ciphertext_vector_discarding_conversion;
mod lwe_bootstrap_key_conversion;
mod lwe_ciphertext_conversion;
mod lwe_ciphertext_discarding_bit_extraction;
mod lwe_ciphertext_discarding_bootstrap;
mod lwe_ciphertext_discarding_conversion;
mod lwe_ciphertext_discarding_keyswitch;
mod lwe_ciphertext_vector_cleartext_vector_discarding_multiplication;
mod lwe_ciphertext_vector_conversion;
mod lwe_ciphertext_vector_discarding_addition;
mod lwe_ciphertext_vector_discarding_and;
mod lwe_ciphertext_vector_discarding_bootstrap;
mod lwe_ciphertext_vector_discarding_circuit_bootstrap_boolean_vertical_packing;
mod lwe_ciphertext_vector_discarding_conversion;
mod lwe_ciphertext_vector_discarding_keyswitch;
mod lwe_ciphertext_vector_discarding_nand;
mod lwe_ciphertext_vector_discarding_nor;
mod lwe_ciphertext_vector_discarding_not;
mod lwe_ciphertext_vector_discarding_opposite;
mod lwe_ciphertext_vector_discarding_or;
mod lwe_ciphertext_vector_discarding_xnor;
mod lwe_ciphertext_vector_discarding_xor;
mod lwe_ciphertext_vector_glwe_ciphertext_discarding_private_functional_packing_keyswitch;
mod lwe_ciphertext_vector_plaintext_vector_discarding_addition;
mod lwe_circuit_bootstrap_private_functional_packing_keyswitch_keys_conversion;
mod lwe_keyswitch_key_conversion;
mod lwe_private_functional_packing_keyswitch_key_conversion;
mod plaintext_vector_conversion;
