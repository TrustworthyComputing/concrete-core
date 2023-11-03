use crate::backends::default::implementation::engines::DefaultEngine;
use crate::backends::default::implementation::entities::{LweCiphertext32, LweCiphertext64};
// use crate::commons::crypto::lwe::LweList as ImplLweList;
use crate::commons::math::tensor::AsRefTensor;
use crate::prelude::{LweCiphertextIndex, LweCiphertextVector32, LweCiphertextVector64};
use crate::specification::engines::{
    LweCiphertextDiscardingStoringEngine, LweCiphertextDiscardingStoringError,
};

/// Implementation of [`LweCiphertextDiscardingStoringEngine`] for [`DefaultEngine`] which modifies
/// an [`LweCiphertextVector64`].
impl LweCiphertextDiscardingStoringEngine<LweCiphertext64, LweCiphertextVector64>
    for DefaultEngine
{
    fn discard_store_lwe_ciphertext(
        &mut self,
        vector: &mut LweCiphertextVector64,
        ciphertext: &LweCiphertext64,
        i: LweCiphertextIndex,
    ) -> Result<(), LweCiphertextDiscardingStoringError<Self::EngineError>> {
        LweCiphertextDiscardingStoringError::<Self::EngineError>::perform_generic_checks(
            vector, ciphertext, i,
        )?;
        unsafe { self.discard_store_lwe_ciphertext_unchecked(vector, ciphertext, i) };
        Ok(())
    }
    unsafe fn discard_store_lwe_ciphertext_unchecked(
        &mut self,
        vector: &mut LweCiphertextVector64,
        ciphertext: &LweCiphertext64,
        i: LweCiphertextIndex,
    ) {
        let poly_size: usize = ciphertext.0.lwe_size().0;
        let ct_tensor = ciphertext.0.as_tensor();
        for j in 0..poly_size {
            vector.0.tensor.set_element(
                i.0 * ciphertext.0.lwe_size().0 + j,
                *ct_tensor.get_element(j),
            );
        }
    }
}

/// Implementation of [`LweCiphertextDiscardingStoringEngine`] for [`DefaultEngine`] which modifies
/// an [`LweCiphertextVector32`].
impl LweCiphertextDiscardingStoringEngine<LweCiphertext32, LweCiphertextVector32>
    for DefaultEngine
{
    fn discard_store_lwe_ciphertext(
        &mut self,
        vector: &mut LweCiphertextVector32,
        ciphertext: &LweCiphertext32,
        i: LweCiphertextIndex,
    ) -> Result<(), LweCiphertextDiscardingStoringError<Self::EngineError>> {
        LweCiphertextDiscardingStoringError::<Self::EngineError>::perform_generic_checks(
            vector, ciphertext, i,
        )?;
        unsafe { self.discard_store_lwe_ciphertext_unchecked(vector, ciphertext, i) };
        Ok(())
    }
    unsafe fn discard_store_lwe_ciphertext_unchecked(
        &mut self,
        vector: &mut LweCiphertextVector32,
        ciphertext: &LweCiphertext32,
        i: LweCiphertextIndex,
    ) {
        let poly_size: usize = ciphertext.0.lwe_size().0;
        let ct_tensor = ciphertext.0.as_tensor();
        for j in 0..poly_size {
            vector.0.tensor.set_element(
                i.0 * ciphertext.0.lwe_size().0 + j,
                *ct_tensor.get_element(j),
            );
        }
    }
}
