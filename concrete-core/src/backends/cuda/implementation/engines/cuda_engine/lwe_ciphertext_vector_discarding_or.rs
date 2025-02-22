use crate::backends::cuda::engines::CudaError;
use crate::backends::cuda::implementation::engines::CudaEngine;
use crate::backends::cuda::implementation::entities::{
    CudaFourierLweBootstrapKey32, CudaLweCiphertextVector32, CudaLweKeyswitchKey32,
};
use crate::backends::cuda::private::device::NumberOfSamples;
use crate::specification::engines::{
    LweCiphertextVectorDiscardingOrEngine, LweCiphertextVectorDiscardingOrError,
};

impl
    LweCiphertextVectorDiscardingOrEngine<
        CudaLweCiphertextVector32,
        CudaLweCiphertextVector32,
        CudaFourierLweBootstrapKey32,
        CudaLweKeyswitchKey32,
    > for CudaEngine
{
    fn discard_or_lwe_ciphertext_vector(
        &self,
        output: &mut CudaLweCiphertextVector32,
        input_1: &CudaLweCiphertextVector32,
        input_2: &CudaLweCiphertextVector32,
        bsk: &CudaFourierLweBootstrapKey32,
        ksk: &CudaLweKeyswitchKey32,
        stream_idx: usize,
    ) -> Result<(), LweCiphertextVectorDiscardingOrError<CudaError>> {
        LweCiphertextVectorDiscardingOrError::perform_generic_checks(output, input_1, input_2)?;
        unsafe {
            self.discard_or_lwe_ciphertext_vector_unchecked(
                output, input_1, input_2, bsk, ksk, stream_idx,
            )
        };
        Ok(())
    }

    unsafe fn discard_or_lwe_ciphertext_vector_unchecked(
        &self,
        output: &mut CudaLweCiphertextVector32,
        input_1: &CudaLweCiphertextVector32,
        input_2: &CudaLweCiphertextVector32,
        bsk: &CudaFourierLweBootstrapKey32,
        ksk: &CudaLweKeyswitchKey32,
        stream_idx: usize,
    ) {
        self.streams[stream_idx]
            .write()
            .unwrap()
            .discard_or_amortized_lwe_ciphertext_vector::<u32>(
                &mut output.0.d_vecs.get_mut(0).unwrap(),
                &input_1.0.d_vecs.get(0).unwrap(),
                &input_2.0.d_vecs.get(0).unwrap(),
                &bsk.0.d_vecs.get(0).unwrap(),
                &ksk.0.d_vecs.get(0).unwrap(),
                input_1.0.lwe_dimension,
                bsk.0.glwe_dimension,
                bsk.0.polynomial_size,
                bsk.0.decomp_base_log,
                bsk.0.decomp_level,
                ksk.0.decomp_base_log,
                ksk.0.decomp_level,
                NumberOfSamples(input_1.0.lwe_ciphertext_count.0),
                self.get_cuda_shared_memory(),
            );
    }
}
