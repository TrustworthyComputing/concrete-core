//! Bootstrap key with Cuda.
use crate::backends::cuda::engines::SharedMemoryAmount;
use crate::backends::cuda::private::crypto::glwe::list::CudaGlweList;
use crate::backends::cuda::private::crypto::lwe::list::CudaLweList;
use crate::backends::cuda::private::device::{CudaStream, GpuIndex, NumberOfGpus};
use crate::backends::cuda::private::vec::CudaVec;
use crate::backends::cuda::private::{compute_number_of_samples_on_gpu, number_of_active_gpus};
use crate::commons;
use crate::commons::crypto::bootstrap::StandardBootstrapKey;
use crate::commons::math::tensor::{AsRefSlice, AsRefTensor};
use crate::commons::numeric::{CastInto, UnsignedInteger};
use crate::prelude::{
    CiphertextCount, DecompositionBaseLog, DecompositionLevelCount, GlweDimension,
    LweCiphertextIndex, LweDimension, PolynomialSize,
};
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

#[cfg(test)]
mod test;

#[derive(Debug)]
pub(crate) struct CudaBootstrapKey<T: UnsignedInteger> {
    // Pointers to GPU data: one cuda vec per GPU
    pub(crate) d_vecs: Vec<CudaVec<f64>>,
    // Input LWE dimension
    pub(crate) input_lwe_dimension: LweDimension,
    // Size of polynomials in the key
    pub(crate) polynomial_size: PolynomialSize,
    // GLWE dimension
    pub(crate) glwe_dimension: GlweDimension,
    // Number of decomposition levels
    pub(crate) decomp_level: DecompositionLevelCount,
    // Value of the base log for the decomposition
    pub(crate) decomp_base_log: DecompositionBaseLog,
    // Field to hold type T
    pub(crate) _phantom: PhantomData<T>,
}

unsafe impl<T> Send for CudaBootstrapKey<T> where T: Send + UnsignedInteger {}
unsafe impl<T> Sync for CudaBootstrapKey<T> where T: Sync + UnsignedInteger {}

pub(crate) unsafe fn convert_lwe_bootstrap_key_from_cpu_to_gpu<T: UnsignedInteger, Cont>(
    streams: &Vec<Arc<RwLock<CudaStream>>>,
    input: &StandardBootstrapKey<Cont>,
    number_of_gpus: NumberOfGpus,
) -> Vec<CudaVec<f64>>
where
    Cont: AsRefSlice<Element = T>,
{
    // Copy the entire input vector over all GPUs
    let mut vecs = Vec::with_capacity(number_of_gpus.0);
    // TODO
    //   Check if it would be better to have GPU 0 compute the BSK and copy it back to the
    //   CPU, then copy the BSK to the other GPUs. The order of instructions varies depending on
    //   the Cuda warp scheduling, which we cannot assume is deterministic, so we'll end up with
    //   slightly different BSKs on the GPUs. It is unclear how significantly this affects the
    //   noise after the bootstrap.
    let total_polynomials =
        input.key_size().0 * input.glwe_size().0 * input.glwe_size().0 * input.level_count().0;
    let alloc_size = total_polynomials * input.polynomial_size().0;
    for i in 0..number_of_gpus.0 {
        let stream = &*streams[i].write().unwrap();
        let input_slice = input.as_tensor().as_slice();
        let mut d_vec = stream.malloc::<f64>(alloc_size as u32);
        stream.convert_lwe_bootstrap_key::<T>(
            &mut d_vec,
            input_slice,
            input.key_size(),
            input.glwe_size().to_glwe_dimension(),
            input.level_count(),
            input.polynomial_size(),
        );
        vecs.push(d_vec);
    }
    vecs
}

pub(crate) unsafe fn execute_lwe_ciphertext_vector_low_latency_bootstrap_on_gpu<
    T: UnsignedInteger + commons::numeric::CastFrom<usize>,
>(
    streams: &Vec<Arc<RwLock<CudaStream>>>,
    output: &mut CudaLweList<T>,
    input: &CudaLweList<T>,
    acc: &CudaGlweList<T>,
    bsk: &CudaBootstrapKey<T>,
    number_of_available_gpus: NumberOfGpus,
    cuda_shared_memory: SharedMemoryAmount,
    stream_idx: usize,
) {
    let number_of_gpus = number_of_active_gpus(
        number_of_available_gpus,
        CiphertextCount(input.lwe_ciphertext_count.0),
    );
    let samples_on_gpu_0 = compute_number_of_samples_on_gpu(
        number_of_gpus,
        CiphertextCount(input.lwe_ciphertext_count.0),
        GpuIndex(0),
    );

    for (gpu_index, _) in streams.iter().enumerate().take(number_of_gpus.0) {
        let samples = compute_number_of_samples_on_gpu(
            number_of_gpus,
            CiphertextCount(input.lwe_ciphertext_count.0),
            GpuIndex(gpu_index),
        );
        // FIXME this is hard set at the moment because concrete-core does not support a more
        //   general API for the bootstrap
        let mut test_vector_indexes: Vec<T> = Vec::with_capacity(samples.0);
        for (i, ind) in test_vector_indexes.iter_mut().enumerate() {
            *ind = <usize as CastInto<T>>::cast_into(i);
        }
        let stream_unlocked = &*streams[stream_idx + gpu_index].write().unwrap();
        let mut d_test_vector_indexes = stream_unlocked.malloc_async::<T>(samples.0 as u32);
        stream_unlocked.copy_to_gpu_async::<T>(&mut d_test_vector_indexes, &test_vector_indexes);

        stream_unlocked.discard_bootstrap_low_latency_lwe_ciphertext_vector::<T>(
            output.d_vecs.get_mut(gpu_index).unwrap(),
            acc.d_vecs.get(gpu_index).unwrap(),
            &d_test_vector_indexes,
            input.d_vecs.get(gpu_index).unwrap(),
            bsk.d_vecs.get(gpu_index).unwrap(),
            input.lwe_dimension,
            bsk.glwe_dimension,
            bsk.polynomial_size,
            bsk.decomp_base_log,
            bsk.decomp_level,
            samples,
            LweCiphertextIndex(samples_on_gpu_0.0 * gpu_index),
            cuda_shared_memory,
        );
    }
}

pub(crate) unsafe fn execute_lwe_ciphertext_vector_amortized_bootstrap_on_gpu<
    T: UnsignedInteger + commons::numeric::CastFrom<usize>,
>(
    streams: &Vec<Arc<RwLock<CudaStream>>>,
    output: &mut CudaLweList<T>,
    input: &CudaLweList<T>,
    acc: &CudaGlweList<T>,
    bsk: &CudaBootstrapKey<T>,
    number_of_available_gpus: NumberOfGpus,
    cuda_shared_memory: SharedMemoryAmount,
    stream_idx: usize,
) {
    let number_of_gpus = number_of_active_gpus(
        number_of_available_gpus,
        CiphertextCount(input.lwe_ciphertext_count.0),
    );
    let samples_on_gpu_0 = compute_number_of_samples_on_gpu(
        number_of_gpus,
        CiphertextCount(input.lwe_ciphertext_count.0),
        GpuIndex(0),
    );

    let mut d_test_vector_indexes_vec = Vec::with_capacity(number_of_gpus.0);
    for i in 0..number_of_gpus.0 {
        // FIXME this is hard set at the moment because concrete-core does not support a more
        //   general API for the bootstrap
        let stream_unlocked = &*streams[stream_idx + i].write().unwrap();
        let mut test_vector_indexes: Vec<T> = Vec::with_capacity(input.lwe_ciphertext_count.0);
        for (i, ind) in test_vector_indexes.iter_mut().enumerate() {
            *ind = <usize as CastInto<T>>::cast_into(i);
        }
        let mut d_test_vector_indexes =
            stream_unlocked.malloc_async::<T>(input.lwe_ciphertext_count.0 as u32);
        stream_unlocked.copy_to_gpu_async::<T>(&mut d_test_vector_indexes, &test_vector_indexes);
        stream_unlocked.synchronize_stream();
        d_test_vector_indexes_vec.push(d_test_vector_indexes);
    }

    for (gpu_index, _) in streams.iter().enumerate().take(number_of_gpus.0) {
        let samples = compute_number_of_samples_on_gpu(
            number_of_gpus,
            CiphertextCount(input.lwe_ciphertext_count.0),
            GpuIndex(gpu_index),
        );
        let stream_unlocked = &*streams[stream_idx + gpu_index].write().unwrap();
        stream_unlocked.discard_bootstrap_amortized_lwe_ciphertext_vector::<T>(
            output.d_vecs.get_mut(gpu_index).unwrap(),
            acc.d_vecs.get(gpu_index).unwrap(),
            d_test_vector_indexes_vec.get(gpu_index).unwrap(),
            input.d_vecs.get(gpu_index).unwrap(),
            bsk.d_vecs.get(gpu_index).unwrap(),
            input.lwe_dimension,
            bsk.glwe_dimension,
            bsk.polynomial_size,
            bsk.decomp_base_log,
            bsk.decomp_level,
            samples,
            LweCiphertextIndex(samples_on_gpu_0.0 * gpu_index),
            cuda_shared_memory,
        );
    }
}
