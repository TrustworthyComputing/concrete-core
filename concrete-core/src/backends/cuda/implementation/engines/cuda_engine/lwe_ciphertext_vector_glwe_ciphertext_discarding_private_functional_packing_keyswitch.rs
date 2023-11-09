use crate::backends::cuda::private::crypto::keyswitch::execute_lwe_ciphertext_vector_fp_keyswitch_on_gpu;
use crate::prelude::{
    CudaEngine, CudaGlweCiphertext32, CudaGlweCiphertext64, CudaLweCiphertextVector32,
    CudaLweCiphertextVector64, CudaLwePrivateFunctionalPackingKeyswitchKey32,
    CudaLwePrivateFunctionalPackingKeyswitchKey64,
};
use crate::specification::engines::{
    LweCiphertextVectorGlweCiphertextDiscardingPrivateFunctionalPackingKeyswitchGpuEngine,
    LweCiphertextVectorGlweCiphertextDiscardingPrivateFunctionalPackingKeyswitchGpuError,
};

/// # Description:
/// Implementation of
/// [`LweCiphertextVectorGlweCiphertextDiscardingPrivateFunctionalPackingKeyswitchGpuEngine`] for
/// [`CudaEngine`] that operates on 32 bits integers.
impl
    LweCiphertextVectorGlweCiphertextDiscardingPrivateFunctionalPackingKeyswitchGpuEngine<
        CudaLwePrivateFunctionalPackingKeyswitchKey32,
        CudaLweCiphertextVector32,
        CudaGlweCiphertext32,
    > for CudaEngine
{
    fn discard_private_functional_packing_keyswitch_lwe_ciphertext_vector(
        &self,
        output: &mut CudaGlweCiphertext32,
        input: &CudaLweCiphertextVector32,
        pfpksk: &CudaLwePrivateFunctionalPackingKeyswitchKey32,
        stream_idx: usize,
    ) -> Result<
        (),
        LweCiphertextVectorGlweCiphertextDiscardingPrivateFunctionalPackingKeyswitchGpuError<
            Self::EngineError,
        >,
    > {
        LweCiphertextVectorGlweCiphertextDiscardingPrivateFunctionalPackingKeyswitchGpuError
        ::perform_generic_checks(
            output, input, pfpksk,
        )?;
        unsafe {
            self.discard_private_functional_packing_keyswitch_lwe_ciphertext_vector_unchecked(
                output, input, pfpksk, stream_idx,
            )
        };
        Ok(())
    }

    unsafe fn discard_private_functional_packing_keyswitch_lwe_ciphertext_vector_unchecked(
        &self,
        output: &mut CudaGlweCiphertext32,
        input: &CudaLweCiphertextVector32,
        pfpksk: &CudaLwePrivateFunctionalPackingKeyswitchKey32,
        stream_idx: usize,
    ) {
        execute_lwe_ciphertext_vector_fp_keyswitch_on_gpu::<u32>(
            self.get_cuda_streams(),
            &mut output.0,
            &input.0,
            &pfpksk.0,
            stream_idx,
            self.get_number_of_gpus(),
        );
    }
}
/// # Description:
/// Implementation of
/// [`LweCiphertextVectorGlweCiphertextDiscardingPrivateFunctionalPackingKeyswitchGpuEngine`] for
/// [`CudaEngine`] that operates on 64 bits integers.
impl
    LweCiphertextVectorGlweCiphertextDiscardingPrivateFunctionalPackingKeyswitchGpuEngine<
        CudaLwePrivateFunctionalPackingKeyswitchKey64,
        CudaLweCiphertextVector64,
        CudaGlweCiphertext64,
    > for CudaEngine
{
    fn discard_private_functional_packing_keyswitch_lwe_ciphertext_vector(
        &self,
        output: &mut CudaGlweCiphertext64,
        input: &CudaLweCiphertextVector64,
        pfpksk: &CudaLwePrivateFunctionalPackingKeyswitchKey64,
        stream_idx: usize,
    ) -> Result<
        (),
        LweCiphertextVectorGlweCiphertextDiscardingPrivateFunctionalPackingKeyswitchGpuError<
            Self::EngineError,
        >,
    > {
        LweCiphertextVectorGlweCiphertextDiscardingPrivateFunctionalPackingKeyswitchGpuError
        ::perform_generic_checks(
            output, input, pfpksk,
        )?;
        unsafe {
            self.discard_private_functional_packing_keyswitch_lwe_ciphertext_vector_unchecked(
                output, input, pfpksk, stream_idx,
            )
        };
        Ok(())
    }

    unsafe fn discard_private_functional_packing_keyswitch_lwe_ciphertext_vector_unchecked(
        &self,
        output: &mut CudaGlweCiphertext64,
        input: &CudaLweCiphertextVector64,
        pfpksk: &CudaLwePrivateFunctionalPackingKeyswitchKey64,
        stream_idx: usize,
    ) {
        execute_lwe_ciphertext_vector_fp_keyswitch_on_gpu::<u64>(
            self.get_cuda_streams(),
            &mut output.0,
            &input.0,
            &pfpksk.0,
            stream_idx,
            self.get_number_of_gpus(),
        );
    }
}
