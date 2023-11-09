use crate::backends::cuda::private::crypto::lwe::list::execute_lwe_ciphertext_vector_addition_on_gpu;
use crate::prelude::{
    CudaEngine, CudaLweCiphertextVector32, CudaLweCiphertextVector64,
    LweCiphertextVectorDiscardingAdditionGpuEngine, LweCiphertextVectorDiscardingAdditionGpuError,
};

impl
    LweCiphertextVectorDiscardingAdditionGpuEngine<
        CudaLweCiphertextVector32,
        CudaLweCiphertextVector32,
    > for CudaEngine
{
    fn discard_add_lwe_ciphertext_vector(
        &self,
        output: &mut CudaLweCiphertextVector32,
        input_1: &CudaLweCiphertextVector32,
        input_2: &CudaLweCiphertextVector32,
        stream_idx: usize,
    ) -> Result<(), LweCiphertextVectorDiscardingAdditionGpuError<Self::EngineError>> {
        LweCiphertextVectorDiscardingAdditionGpuError::perform_generic_checks(
            output, input_1, input_2,
        )?;
        unsafe {
            self.discard_add_lwe_ciphertext_vector_unchecked(output, input_1, input_2, stream_idx)
        };
        Ok(())
    }

    unsafe fn discard_add_lwe_ciphertext_vector_unchecked(
        &self,
        output: &mut CudaLweCiphertextVector32,
        input_1: &CudaLweCiphertextVector32,
        input_2: &CudaLweCiphertextVector32,
        stream_idx: usize,
    ) {
        execute_lwe_ciphertext_vector_addition_on_gpu::<u32>(
            self.get_cuda_streams(),
            &mut output.0,
            &input_1.0,
            &input_2.0,
            self.get_number_of_gpus(),
            stream_idx,
        );
    }
}

impl
    LweCiphertextVectorDiscardingAdditionGpuEngine<
        CudaLweCiphertextVector64,
        CudaLweCiphertextVector64,
    > for CudaEngine
{
    fn discard_add_lwe_ciphertext_vector(
        &self,
        output: &mut CudaLweCiphertextVector64,
        input_1: &CudaLweCiphertextVector64,
        input_2: &CudaLweCiphertextVector64,
        stream_idx: usize,
    ) -> Result<(), LweCiphertextVectorDiscardingAdditionGpuError<Self::EngineError>> {
        LweCiphertextVectorDiscardingAdditionGpuError::perform_generic_checks(
            output, input_1, input_2,
        )?;
        unsafe {
            self.discard_add_lwe_ciphertext_vector_unchecked(output, input_1, input_2, stream_idx)
        };
        Ok(())
    }

    unsafe fn discard_add_lwe_ciphertext_vector_unchecked(
        &self,
        output: &mut CudaLweCiphertextVector64,
        input_1: &CudaLweCiphertextVector64,
        input_2: &CudaLweCiphertextVector64,
        stream_idx: usize,
    ) {
        execute_lwe_ciphertext_vector_addition_on_gpu::<u64>(
            self.get_cuda_streams(),
            &mut output.0,
            &input_1.0,
            &input_2.0,
            self.get_number_of_gpus(),
            stream_idx,
        );
    }
}
