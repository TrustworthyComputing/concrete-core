use crate::backends::cuda::implementation::engines::{CudaEngine, CudaError};
use crate::backends::cuda::implementation::entities::{
    CudaCleartextVector32, CudaCleartextVector64,
};
use crate::backends::cuda::private::crypto::cleartext::list::{
    copy_cleartext_vector_from_cpu_to_gpu, copy_cleartext_vector_from_gpu_to_cpu, CudaCleartextList,
};
use crate::backends::cuda::private::device::GpuIndex;
use crate::backends::cuda::private::{compute_number_of_samples_on_gpu, number_of_active_gpus};
use crate::commons::crypto::encoding::CleartextList;
use crate::prelude::{CiphertextCount, CleartextVector32, CleartextVector64};
use crate::specification::engines::{
    CleartextVectorConversionGpuEngine, CleartextVectorConversionGpuError,
};
use crate::specification::entities::CleartextVectorEntity;

impl From<CudaError> for CleartextVectorConversionGpuError<CudaError> {
    fn from(err: CudaError) -> Self {
        Self::Engine(err)
    }
}

impl CleartextVectorConversionGpuEngine<CleartextVector32, CudaCleartextVector32> for CudaEngine {
    fn convert_cleartext_vector(
        &self,
        input: &CleartextVector32,
    ) -> Result<CudaCleartextVector32, CleartextVectorConversionGpuError<CudaError>> {
        let number_of_gpus = number_of_active_gpus(
            self.get_number_of_gpus(),
            CiphertextCount(input.cleartext_count().0),
        );
        for gpu_index in 0..number_of_gpus.0 {
            let stream = &*self.streams[gpu_index].read().unwrap();
            let samples = compute_number_of_samples_on_gpu(
                self.get_number_of_gpus(),
                CiphertextCount(input.cleartext_count().0),
                GpuIndex(gpu_index),
            )
            .0;
            let size = samples as u64 * std::mem::size_of::<u32>() as u64;
            stream.check_device_memory(size)?;
        }
        Ok(unsafe { self.convert_cleartext_vector_unchecked(input) })
    }

    unsafe fn convert_cleartext_vector_unchecked(
        &self,
        input: &CleartextVector32,
    ) -> CudaCleartextVector32 {
        let vecs = copy_cleartext_vector_from_cpu_to_gpu::<u32, _>(
            self.get_cuda_streams(),
            &input.0,
            self.get_number_of_gpus(),
        );
        CudaCleartextVector32(CudaCleartextList::<u32> {
            d_vecs: vecs,
            cleartext_count: input.cleartext_count(),
        })
    }
}

impl CleartextVectorConversionGpuEngine<CudaCleartextVector32, CleartextVector32> for CudaEngine {
    fn convert_cleartext_vector(
        &self,
        input: &CudaCleartextVector32,
    ) -> Result<CleartextVector32, CleartextVectorConversionGpuError<CudaError>> {
        Ok(unsafe { self.convert_cleartext_vector_unchecked(input) })
    }

    unsafe fn convert_cleartext_vector_unchecked(
        &self,
        input: &CudaCleartextVector32,
    ) -> CleartextVector32 {
        let output = copy_cleartext_vector_from_gpu_to_cpu::<u32>(
            self.get_cuda_streams(),
            &input.0,
            self.get_number_of_gpus(),
        );
        CleartextVector32(CleartextList::from_container(output))
    }
}

impl CleartextVectorConversionGpuEngine<CleartextVector64, CudaCleartextVector64> for CudaEngine {
    fn convert_cleartext_vector(
        &self,
        input: &CleartextVector64,
    ) -> Result<CudaCleartextVector64, CleartextVectorConversionGpuError<CudaError>> {
        let number_of_gpus = number_of_active_gpus(
            self.get_number_of_gpus(),
            CiphertextCount(input.cleartext_count().0),
        );
        for gpu_index in 0..number_of_gpus.0 {
            let stream = &*self.streams[gpu_index].write().unwrap();
            let samples = compute_number_of_samples_on_gpu(
                self.get_number_of_gpus(),
                CiphertextCount(input.cleartext_count().0),
                GpuIndex(gpu_index),
            )
            .0;
            let size = samples as u64 * std::mem::size_of::<u64>() as u64;
            stream.check_device_memory(size)?;
        }
        Ok(unsafe { self.convert_cleartext_vector_unchecked(input) })
    }

    unsafe fn convert_cleartext_vector_unchecked(
        &self,
        input: &CleartextVector64,
    ) -> CudaCleartextVector64 {
        let vecs = copy_cleartext_vector_from_cpu_to_gpu::<u64, _>(
            self.get_cuda_streams(),
            &input.0,
            self.get_number_of_gpus(),
        );
        CudaCleartextVector64(CudaCleartextList::<u64> {
            d_vecs: vecs,
            cleartext_count: input.cleartext_count(),
        })
    }
}

impl CleartextVectorConversionGpuEngine<CudaCleartextVector64, CleartextVector64> for CudaEngine {
    fn convert_cleartext_vector(
        &self,
        input: &CudaCleartextVector64,
    ) -> Result<CleartextVector64, CleartextVectorConversionGpuError<CudaError>> {
        Ok(unsafe { self.convert_cleartext_vector_unchecked(input) })
    }

    unsafe fn convert_cleartext_vector_unchecked(
        &self,
        input: &CudaCleartextVector64,
    ) -> CleartextVector64 {
        let output = copy_cleartext_vector_from_gpu_to_cpu::<u64>(
            self.get_cuda_streams(),
            &input.0,
            self.get_number_of_gpus(),
        );
        CleartextVector64(CleartextList::from_container(output))
    }
}
