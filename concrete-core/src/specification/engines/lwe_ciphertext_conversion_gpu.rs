use super::engine_error;
use crate::specification::engines::AbstractEngine;
use crate::specification::entities::LweCiphertextEntity;

engine_error! {
    LweCiphertextConversionGpuError for LweCiphertextConversionGpuEngine @
}

/// A trait for engines converting LWE ciphertexts.
///
/// # Semantics
///
/// This [pure](super#operation-semantics) operation generates a LWE ciphertext containing the
/// conversion of the `input` LWE ciphertext to a type with a different representation (for instance
/// from cpu to gpu memory).
///
/// # Formal Definition
pub trait LweCiphertextConversionGpuEngine<Input, Output>: AbstractEngine
where
    Input: LweCiphertextEntity,
    Output: LweCiphertextEntity,
{
    /// Converts a LWE ciphertext.
    fn convert_lwe_ciphertext(
        &self,
        input: &Input,
    ) -> Result<Output, LweCiphertextConversionGpuError<Self::EngineError>>;

    /// Unsafely converts a LWE ciphertext.
    ///
    /// # Safety
    /// For the _general_ safety concerns regarding this operation, refer to the different variants
    /// of [`LweCiphertextConversionGpuError`]. For safety concerns _specific_ to an engine, refer
    /// to the implementer safety section.
    unsafe fn convert_lwe_ciphertext_unchecked(&self, input: &Input) -> Output;
}
