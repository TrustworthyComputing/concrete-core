use crate::backends::cuda::private::crypto::lwe::list::execute_lwe_ciphertext_vector_plaintext_vector_addition_on_gpu;
use crate::prelude::{
    CudaEngine, CudaLweCiphertextVector32, CudaLweCiphertextVector64, CudaPlaintextVector32,
    CudaPlaintextVector64, LweCiphertextVectorPlaintextVectorDiscardingAdditionGpuEngine,
    LweCiphertextVectorPlaintextVectorDiscardingAdditionGpuError,
};

impl
    LweCiphertextVectorPlaintextVectorDiscardingAdditionGpuEngine<
        CudaLweCiphertextVector32,
        CudaPlaintextVector32,
        CudaLweCiphertextVector32,
    > for CudaEngine
{
    fn discard_add_lwe_ciphertext_vector_plaintext_vector(
        &self,
        output: &mut CudaLweCiphertextVector32,
        input_1: &CudaLweCiphertextVector32,
        input_2: &CudaPlaintextVector32,
        stream_idx: usize,
    ) -> Result<(), LweCiphertextVectorPlaintextVectorDiscardingAdditionGpuError<Self::EngineError>>
    {
        LweCiphertextVectorPlaintextVectorDiscardingAdditionGpuError::perform_generic_checks(
            output, input_1, input_2,
        )?;
        unsafe {
            self.discard_add_lwe_ciphertext_vector_plaintext_vector_unchecked(
                output, input_1, input_2, stream_idx,
            )
        };
        Ok(())
    }

    unsafe fn discard_add_lwe_ciphertext_vector_plaintext_vector_unchecked(
        &self,
        output: &mut CudaLweCiphertextVector32,
        input_1: &CudaLweCiphertextVector32,
        input_2: &CudaPlaintextVector32,
        stream_idx: usize,
    ) {
        let stream = &*self.get_cuda_streams()[stream_idx].read().unwrap();
        execute_lwe_ciphertext_vector_plaintext_vector_addition_on_gpu::<u32>(
            stream,
            &mut output.0,
            &input_1.0,
            &input_2.0,
            self.get_number_of_gpus(),
        );
    }
}

impl
    LweCiphertextVectorPlaintextVectorDiscardingAdditionGpuEngine<
        CudaLweCiphertextVector64,
        CudaPlaintextVector64,
        CudaLweCiphertextVector64,
    > for CudaEngine
{
    fn discard_add_lwe_ciphertext_vector_plaintext_vector(
        &self,
        output: &mut CudaLweCiphertextVector64,
        input_1: &CudaLweCiphertextVector64,
        input_2: &CudaPlaintextVector64,
        stream_idx: usize,
    ) -> Result<(), LweCiphertextVectorPlaintextVectorDiscardingAdditionGpuError<Self::EngineError>>
    {
        LweCiphertextVectorPlaintextVectorDiscardingAdditionGpuError::perform_generic_checks(
            output, input_1, input_2,
        )?;
        unsafe {
            self.discard_add_lwe_ciphertext_vector_plaintext_vector_unchecked(
                output, input_1, input_2, stream_idx,
            )
        };
        Ok(())
    }

    unsafe fn discard_add_lwe_ciphertext_vector_plaintext_vector_unchecked(
        &self,
        output: &mut CudaLweCiphertextVector64,
        input_1: &CudaLweCiphertextVector64,
        input_2: &CudaPlaintextVector64,
        stream_idx: usize,
    ) {
        let stream = &*self.get_cuda_streams()[stream_idx].read().unwrap();
        execute_lwe_ciphertext_vector_plaintext_vector_addition_on_gpu::<u64>(
            stream,
            &mut output.0,
            &input_1.0,
            &input_2.0,
            self.get_number_of_gpus(),
        );
    }
}
