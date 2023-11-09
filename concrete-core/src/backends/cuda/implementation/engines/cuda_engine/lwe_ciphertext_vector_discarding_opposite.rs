use crate::backends::cuda::private::crypto::lwe::list::execute_lwe_ciphertext_vector_opposite_on_gpu;
use crate::prelude::{
    CudaEngine, CudaLweCiphertextVector32, CudaLweCiphertextVector64,
    LweCiphertextVectorDiscardingOppositeEngine, LweCiphertextVectorDiscardingOppositeError,
};

impl
    LweCiphertextVectorDiscardingOppositeEngine<
        CudaLweCiphertextVector32,
        CudaLweCiphertextVector32,
    > for CudaEngine
{
    fn discard_opp_lwe_ciphertext_vector(
        &self,
        output: &mut CudaLweCiphertextVector32,
        input: &CudaLweCiphertextVector32,
        stream_idx: usize,
    ) -> Result<(), LweCiphertextVectorDiscardingOppositeError<Self::EngineError>> {
        LweCiphertextVectorDiscardingOppositeError::perform_generic_checks(output, input)?;
        unsafe { self.discard_opp_lwe_ciphertext_vector_unchecked(output, input, stream_idx) };
        Ok(())
    }

    unsafe fn discard_opp_lwe_ciphertext_vector_unchecked(
        &self,
        output: &mut CudaLweCiphertextVector32,
        input: &CudaLweCiphertextVector32,
        stream_idx: usize,
    ) {
        execute_lwe_ciphertext_vector_opposite_on_gpu::<u32>(
            self.get_cuda_streams(),
            &mut output.0,
            &input.0,
            self.get_number_of_gpus(),
            stream_idx,
        );
    }
}

impl
    LweCiphertextVectorDiscardingOppositeEngine<
        CudaLweCiphertextVector64,
        CudaLweCiphertextVector64,
    > for CudaEngine
{
    fn discard_opp_lwe_ciphertext_vector(
        &self,
        output: &mut CudaLweCiphertextVector64,
        input: &CudaLweCiphertextVector64,
        stream_idx: usize,
    ) -> Result<(), LweCiphertextVectorDiscardingOppositeError<Self::EngineError>> {
        LweCiphertextVectorDiscardingOppositeError::perform_generic_checks(output, input)?;
        unsafe { self.discard_opp_lwe_ciphertext_vector_unchecked(output, input, stream_idx) };
        Ok(())
    }

    unsafe fn discard_opp_lwe_ciphertext_vector_unchecked(
        &self,
        output: &mut CudaLweCiphertextVector64,
        input: &CudaLweCiphertextVector64,
        stream_idx: usize,
    ) {
        execute_lwe_ciphertext_vector_opposite_on_gpu::<u64>(
            self.get_cuda_streams(),
            &mut output.0,
            &input.0,
            self.get_number_of_gpus(),
            stream_idx,
        );
    }
}
