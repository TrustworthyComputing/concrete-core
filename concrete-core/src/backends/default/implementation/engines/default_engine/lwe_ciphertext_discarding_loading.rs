use crate::backends::default::implementation::engines::DefaultEngine;
use crate::backends::default::implementation::entities::{LweCiphertext32, LweCiphertext64};
// use crate::commons::crypto::lwe::LweList as ImplLweList;
use crate::commons::math::tensor::AsMutSlice;
use crate::commons::math::tensor::AsMutTensor;
use crate::commons::math::tensor::AsRefSlice;
use crate::commons::math::tensor::AsRefTensor;
use crate::prelude::{LweCiphertextIndex, LweCiphertextVector32, LweCiphertextVector64};
use crate::specification::engines::{
    LweCiphertextDiscardingLoadingEngine, LweCiphertextDiscardingLoadingError,
};

/// Implementation of [`LweCiphertextDiscardingLoadingEngine`] for
/// [`DefaultEngine`] which extracts an [`LweCiphertext64`] from an [`LweCiphertextVector64`].
impl LweCiphertextDiscardingLoadingEngine<LweCiphertextVector64, LweCiphertext64>
    for DefaultEngine
{
    fn discard_load_lwe_ciphertext(
        &mut self,
        ciphertext: &mut LweCiphertext64,
        vector: &LweCiphertextVector64,
        i: LweCiphertextIndex,
    ) -> Result<(), LweCiphertextDiscardingLoadingError<Self::EngineError>> {
        LweCiphertextDiscardingLoadingError::<Self::EngineError>::perform_generic_checks(
            ciphertext, vector, i,
        )?;
        unsafe { self.discard_load_lwe_ciphertext_unchecked(ciphertext, vector, i) };
        Ok(())
    }
    unsafe fn discard_load_lwe_ciphertext_unchecked(
        &mut self,
        ciphertext: &mut LweCiphertext64,
        vector: &LweCiphertextVector64,
        i: LweCiphertextIndex,
    ) {
        let poly_size: usize = ciphertext.0.lwe_size().0;
        let ciphertext_slice = ciphertext.0.as_mut_tensor().as_mut_slice();
        let vector_elements = &mut vector.0.as_tensor().as_slice();
        for j in 0..poly_size {
            ciphertext_slice[j] = vector_elements[i.0 * poly_size + j];
        }
    }
}

/// Implementation of [`LweCiphertextDiscardingLoadingEngine`] for
/// [`DefaultEngine`] which extracts an [`LweCiphertext32`] from an [`LweCiphertextVector32`].
impl LweCiphertextDiscardingLoadingEngine<LweCiphertextVector32, LweCiphertext32>
    for DefaultEngine
{
    fn discard_load_lwe_ciphertext(
        &mut self,
        ciphertext: &mut LweCiphertext32,
        vector: &LweCiphertextVector32,
        i: LweCiphertextIndex,
    ) -> Result<(), LweCiphertextDiscardingLoadingError<Self::EngineError>> {
        LweCiphertextDiscardingLoadingError::<Self::EngineError>::perform_generic_checks(
            ciphertext, vector, i,
        )?;
        unsafe { self.discard_load_lwe_ciphertext_unchecked(ciphertext, vector, i) };
        Ok(())
    }
    unsafe fn discard_load_lwe_ciphertext_unchecked(
        &mut self,
        ciphertext: &mut LweCiphertext32,
        vector: &LweCiphertextVector32,
        i: LweCiphertextIndex,
    ) {
        let poly_size: usize = ciphertext.0.lwe_size().0;
        let ciphertext_slice = ciphertext.0.as_mut_tensor().as_mut_slice();
        let vector_elements = &mut vector.0.as_tensor().as_slice();
        for j in 0..poly_size {
            ciphertext_slice[j] = vector_elements[i.0 * poly_size + j];
        }
    }
}
