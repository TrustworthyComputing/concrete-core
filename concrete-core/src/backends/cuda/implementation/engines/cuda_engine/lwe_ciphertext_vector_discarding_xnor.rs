use crate::backends::cuda::engines::CudaError;
use crate::backends::cuda::implementation::engines::CudaEngine;
use crate::backends::cuda::private::device::NumberOfSamples;
use crate::backends::cuda::implementation::entities::{
    CudaLweCiphertextVector32,
    CudaLweKeyswitchKey32,
    CudaFourierLweBootstrapKey32,
};
use crate::specification::engines::{
    LweCiphertextVectorDiscardingXnorEngine, LweCiphertextVectorDiscardingXnorError,
};

impl
    LweCiphertextVectorDiscardingXnorEngine<
        CudaLweCiphertextVector32,
        CudaLweCiphertextVector32,
        CudaFourierLweBootstrapKey32,
        CudaLweKeyswitchKey32,
    > for CudaEngine
{

    fn discard_xnor_lwe_ciphertext_vector(
        &mut self,
        output: &mut CudaLweCiphertextVector32,
        input_1: &CudaLweCiphertextVector32,
        input_2: &CudaLweCiphertextVector32,
        bsk: &CudaFourierLweBootstrapKey32,
        ksk: &CudaLweKeyswitchKey32,
    ) -> Result<(), LweCiphertextVectorDiscardingXnorError<CudaError>> {
        LweCiphertextVectorDiscardingXnorError::perform_generic_checks(
            output, input_1, input_2
        )?;
        unsafe { self.discard_xnor_lwe_ciphertext_vector_unchecked(
                        output, 
                        input_1, 
                        input_2,
                        bsk, 
                        ksk, 
                    ) };
        Ok(())
    }

    unsafe fn discard_xnor_lwe_ciphertext_vector_unchecked(
        &mut self,
        output: &mut CudaLweCiphertextVector32,
        input_1: &CudaLweCiphertextVector32,
        input_2: &CudaLweCiphertextVector32,
        bsk: &CudaFourierLweBootstrapKey32,
        ksk: &CudaLweKeyswitchKey32,
    ) {
        self.streams[0].discard_xnor_amortized_lwe_ciphertext_vector::<u32>(
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