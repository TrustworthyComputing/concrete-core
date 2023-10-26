use crate::backends::cuda::engines::CudaError;
use crate::backends::cuda::implementation::engines::CudaEngine;
use crate::backends::cuda::private::device::NumberOfSamples;
use crate::backends::cuda::implementation::entities::CudaLweCiphertextVector32;
use crate::specification::engines::{
    LweCiphertextVectorDiscardingNotEngine, LweCiphertextVectorDiscardingNotError,
};

impl
    LweCiphertextVectorDiscardingNotEngine<
        CudaLweCiphertextVector32,
        CudaLweCiphertextVector32,
    > for CudaEngine
{
    fn discard_not_lwe_ciphertext_vector(
        &mut self,
        output: &mut CudaLweCiphertextVector32,
        input: &CudaLweCiphertextVector32,
    ) -> Result<(), LweCiphertextVectorDiscardingNotError<CudaError>> {
        LweCiphertextVectorDiscardingNotError::perform_generic_checks(
            output, input
        )?;
        unsafe { self.discard_not_lwe_ciphertext_vector_unchecked(output, input) };
        Ok(())
    }

    unsafe fn discard_not_lwe_ciphertext_vector_unchecked(
        &mut self,
        output: &mut CudaLweCiphertextVector32,
        input: &CudaLweCiphertextVector32,
    ) {
        self.streams[self.stream_idx].discard_not_amortized_lwe_ciphertext_vector::<u32>(
            &mut output.0.d_vecs.get_mut(0).unwrap(),
            &input.0.d_vecs.get(0).unwrap(),
            input.0.lwe_dimension,
            NumberOfSamples(input.0.lwe_ciphertext_count.0),
        );
        self.inc_stream_idx();
    }
}